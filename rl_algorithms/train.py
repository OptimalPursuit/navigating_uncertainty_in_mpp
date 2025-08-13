from datetime import datetime
import os
import copy
import wandb
import tqdm
import yaml
from dotmap import DotMap
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from typing import Dict, Tuple, Optional, List, Union

# Torch
import torch
import torch.nn as nn

# TorchRL
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor
from torchrl.objectives.ddpg import DDPGLoss
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.sac import SACLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Custom code
from rl_algorithms.utils import make_env
from rl_algorithms.loss import FeasibilityClipPPOLoss, FeasibilitySACLoss

# Classes
class EarlyStopping:
    def __init__(self, divergence_threshold=1e6, divergence_patience=10, validation_patience=3):
        """Early stopping based on a divergence threshold and a patience parameter."""
        # Divergence threshold and patience
        self.divergence_threshold = divergence_threshold
        self.divergence_patience = divergence_patience
        self.div_counter = 0

        # Validation patience
        self.validation_patience = validation_patience
        self.val_counter = 0
        self.val_rewards_history = []

    def update_rewards(self, reward:float) -> None:
        """
        Add a new validation reward to the history.
        Args:
            reward (float): The latest validation reward.
        """
        self.val_rewards_history.append(reward)

    def validation_check(self) -> bool:
        """
        Check for early stopping based on consecutive decreases in validation rewards.
        Returns:
            bool: True if stopping criteria are met, False otherwise.
        """
        # Only start checking if we have enough history
        if len(self.val_rewards_history) < 2:
            return False

        # Check the last two rewards
        if self.val_rewards_history[-1] < self.val_rewards_history[-2]:
            self.val_counter += 1
        else:
            self.val_counter = 0

        # Stop if consecutive decreases exceed patience
        return self.val_counter >= self.validation_patience

    def divergence_check(self, loss:torch.Tensor) -> bool:
        """Check for early stopping based on a threshold for the loss value."""
        if torch.isnan(loss):
            print(f"Early stopping due to nan in loss.")
            return True
        elif torch.isinf(loss):
            print(f"Early stopping due to inf in loss.")
            return True
        elif torch.abs(loss) > self.divergence_threshold:
            self.div_counter += 1
            if self.div_counter >= self.divergence_patience:
                print(f"Early stopping at epoch due to loss divergence.")
                return True  # Stop training
        else:
            self.div_counter = 0  # Reset if loss is stable
        return False

# Functions
def convert_to_dict(obj:object) -> Union[Dict, List]:
    """Recursively convert DotMap or other custom objects to standard Python dictionaries."""
    if isinstance(obj, DotMap):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, dict):  # Handle nested dictionaries
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # Handle lists containing DotMaps or dicts
        return [convert_to_dict(item) for item in obj]
    return obj  # Return primitive data types as-is

def run_training(policy: nn.Module, critic: nn.Module, device:str="cuda", **kwargs) -> None:
    """Train the policy using the specified algorithm."""
    # Algorithm hyperparameters
    lr = kwargs["training"]["lr"]
    pd_lr = kwargs["training"]["pd_lr"] if "pd_lr" in kwargs["training"] else lr
    batch_size = kwargs["model"]["batch_size"]
    mini_batch_size = int(kwargs["algorithm"]["mini_batch_size"] * batch_size)
    num_epochs = kwargs["algorithm"]["ppo_epochs"]
    gamma = kwargs["algorithm"]["gamma"]
    gae_lambda = kwargs["algorithm"]["gae_lambda"]
    primal_dual = kwargs["algorithm"]["primal_dual"]
    # Loss hyperparameters
    vf_lambda = kwargs["algorithm"]["vf_lambda"]
    feasibility_lambda = kwargs["algorithm"]["feasibility_lambda"]
    entropy_lambda = kwargs["algorithm"]["entropy_lambda"]
    clip_epsilon = kwargs["algorithm"]["clip_range"]
    max_grad_norm = kwargs["algorithm"]["max_grad_norm"]
    tau = kwargs["algorithm"]["tau"]
    # Training hyperparameters
    train_data_size = kwargs["training"]["train_data_size"]
    validation_freq = kwargs["training"]["validation_freq"]
    validation_patience = kwargs["training"]["validation_patience"]

    # Environment
    train_env = make_env(env_kwargs=kwargs["env"], batch_size=[batch_size], device=device)
    n_step = train_env.T * train_env.K
    n_constraints = train_env.n_constraints
    if f"lagrangian_multiplier_0" in kwargs["algorithm"]:
        lagrangian_multiplier = torch.tensor([
            kwargs["algorithm"][f"lagrangian_multiplier_{i}"] for i in range(n_constraints)], device=device)
    else:
        lagrangian_multiplier = torch.ones(n_constraints, device=device)
    if primal_dual:
        lagrangian_multiplier = None

    # Optimizer, loss module, data collector, and scheduler
    advantage_module = GAE(gamma=gamma, lmbda=gae_lambda, value_network=critic, average_gae=True)
    if kwargs["algorithm"]["type"] == "sac":
        loss_module = FeasibilitySACLoss(
            actor_network=policy,
            qvalue_network=critic,
            separate_losses=True,
            fixed_alpha=False,
            alpha_init=1.0,
            min_alpha=1e-2, #[1e-2, 1e-3]
            max_alpha=1.0, #[1.0, 10]
            lagrangian_multiplier=lagrangian_multiplier,
        )
    elif kwargs["algorithm"]["type"] == "ppo":
        loss_module = FeasibilityClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_lambda),
            entropy_coef=entropy_lambda,
            critic_coef=vf_lambda,
            loss_critic_type="smooth_l1",
            normalize_advantage=True,
            lagrangian_multiplier=lagrangian_multiplier,
        )
    elif kwargs["algorithm"]["type"] == "ddpg":
        # Create the DDPG loss module
        loss_module = DDPGLoss(
            actor_network=policy,
            value_network=critic,
            delay_actor=True,
            delay_value=True,
        )
    else:
        raise ValueError(f"Algorithm {kwargs['algorithm']['type']} not recognized.")

    # Data collector and replay buffer
    collector = SyncDataCollector(
        train_env,
        policy,
        frames_per_batch=batch_size*n_step, # batch_size * steps_per_episode
        total_frames=train_data_size,
        split_trajs=False,
        device=device,
    )
    collector.set_seed(train_env.seed)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=batch_size),
        sampler=SamplerWithoutReplacement(),
    )

    # Optimizer and scheduler
    if kwargs["algorithm"]["type"] == "sac":
        actor_optim = torch.optim.Adam(policy.parameters(), lr=lr)
        critic_params = [p for name, p in loss_module.qvalue_network_params.named_parameters() if not name.startswith("dual_head")]
        critic_optim = torch.optim.Adam(critic_params, lr=lr)
        # critic_optim = torch.optim.Adam(loss_module.qvalue_network_params.parameters(), lr=lr)
        if not loss_module.fixed_alpha:
            alpha_optim = torch.optim.Adam([loss_module.log_alpha], lr=lr)
        actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, train_data_size)
        critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_optim, train_data_size)
    elif kwargs["algorithm"]["type"] == "ppo":
        actor_critic_params = list(policy.parameters()) + list(critic.parameters())
        actor_critic_optim = torch.optim.Adam(actor_critic_params, lr=lr)
        actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_critic_optim, train_data_size)
        # critic_params = [p for name, p in critic.named_parameters() if not name.startswith("dual_head")]
        # critic_optim = torch.optim.Adam(critic_params, lr=lr)
    else:
        raise ValueError(f"Algorithm {kwargs['algorithm']['type']} not recognized.")
    if primal_dual:
        dual_params = critic.module.dual_head.parameters()
        dual_optim = torch.optim.Adam(dual_params, lr=pd_lr)

    train_updates = train_data_size // (batch_size * n_step)
    pbar = tqdm.tqdm(range(train_updates))

    # Early stopping criteria
    early_stopping = EarlyStopping()

    policy.train()
    # torch.autograd.set_detect_anomaly(True)
    # Training loop
    for step, td in enumerate(collector):
        if kwargs["algorithm"]["type"] == "ppo":
            advantage_module(td)
        replay_buffer.extend(td)
        for _ in range(batch_size // mini_batch_size):
            # Sample mini-batch (including actions, n-step returns, old log likelihoods, target_values)
            subdata = replay_buffer.sample(mini_batch_size).to(device)
            # Loss computation and backpropagation
            if kwargs["algorithm"]["type"] == "sac":
                # Loss computation
                loss_out = loss_module(subdata.to(device))

                # Critic Update
                critic_optim.zero_grad()
                loss_out["loss_qvalue"].backward()
                qvalue_params = loss_module.qvalue_network_params.flatten_keys().values()
                loss_out["gn_critic"] = torch.nn.utils.clip_grad_norm_(qvalue_params, max_grad_norm)
                critic_optim.step()
                # Soft update target critics
                with torch.no_grad():
                    soft_update(loss_module.target_qvalue_network_params, loss_module.qvalue_network_params, tau)

                # Actor, Dual and Alpha Update
                # Zero out gradients
                actor_optim.zero_grad()
                if kwargs["algorithm"]["primal_dual"]:
                    dual_optim.zero_grad()
                if not loss_module.fixed_alpha:
                    alpha_optim.zero_grad()

                # Compute actor loss
                actor_loss = loss_out["loss_actor"] + feasibility_lambda * loss_out["loss_feasibility"]
                actor_loss.backward(retain_graph=primal_dual)

                # Compute primal-dual loss if applicable
                if kwargs["algorithm"]["primal_dual"]:
                    loss_out["loss_feasibility"].backward()
                # Compute alpha loss if applicable
                if not loss_module.fixed_alpha:
                    loss_out["loss_alpha"].backward()

                # Step optimizers
                loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                actor_optim.step()
                if kwargs["algorithm"]["primal_dual"]:
                    loss_out["gn_dual"] = torch.nn.utils.clip_grad_norm_(dual_params, max_grad_norm)
                    dual_optim.step()
                if not loss_module.fixed_alpha:
                    alpha_optim.step()

            elif kwargs["algorithm"]["type"] == "ppo":
                for _ in range(num_epochs):
                    # Compute loss
                    loss_out = loss_module(subdata.to(device))
                    loss_out["total_loss"] = loss_out["loss_objective"] + loss_out["loss_critic"] + \
                                             loss_out["loss_entropy"] + feasibility_lambda * (loss_out["loss_feasibility"])

                    # Actor update
                    actor_critic_optim.zero_grad()
                    loss_out["total_loss"].backward(retain_graph=primal_dual)
                    loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(actor_critic_params, max_grad_norm)

                    # Dual update if applicable
                    if primal_dual:
                        dual_optim.zero_grad()
                        loss_out["loss_feasibility"].backward()
                        loss_out["gn_dual"] = torch.nn.utils.clip_grad_norm_(dual_params, max_grad_norm)

                    # Step optimizers
                    actor_critic_optim.step()
                    if primal_dual:
                        dual_optim.step()
            else:
                raise ValueError(f"Algorithm {kwargs['algorithm']['type']} not recognized.")

        # Log metrics
        train_performance = get_performance_metrics(subdata, td, train_env)
        log = {
            # Losses
            "total_loss": loss_out.get("total_loss", 0),
            "loss_actor": loss_out.get("loss_actor", 0) or loss_out.get("loss_objective", 0),
            "loss_critic": loss_out.get("loss_qvalue", 0) or loss_out.get("loss_critic", 0),
            "loss_feasibility":loss_out.get("loss_feasibility", 0),
            "violation": loss_out.get("violation", 0),
            "loss_entropy": loss_out.get("loss_alpha", 0) or loss_out.get("loss_entropy", 0),
            # Supporting metrics
            "step": step,
            "gn_actor": loss_out.get("gn_actor", 0),
            "gn_critic": loss_out.get("gn_critic", 0),
            "clip_fraction": loss_out.get("clip_fraction", 0),
            **train_performance,
        }
        log["mean_total_violation"] = log["violation"].sum(dim=(-2, -1)).mean().item() if log["violation"].dim() > 1 else 0
        log["lagrangian_multiplier"] = loss_out["lagrangian_multiplier"].mean().item() if loss_out.get("lagrangian_multiplier") is not None else 0.0
        pbar.update(1)
        # Log metrics
        pbar.set_description(
            # Loss, gn and rewards
            f"return: {log['return']: 4.4f}, "
            f"traj_return: {log['traj_return']: 4.4f}, "
            f"total_loss: {log['total_loss']: 4.4f}, "
            f"loss_actor:  {log['loss_actor']: 4.4f}, "
            f"loss_critic:  {log['loss_critic']: 4.4f}, "
            f"feasibility_loss: {log['loss_feasibility']: 4.4f}, "
            f"mean_violation: {log['mean_total_violation']: 4.4f}, "  
            # Prediction
            f"x: {log['x']: 4.4f}, "
            f"loc(x): {log['loc(x)']: 4.4f}, "
            f"scale(x): {log['scale(x)']: 4.4f}, "
            f"lagrangian_multiplier: {log['lagrangian_multiplier']: 4.4f}, "
            # Performance
            f"total_profit: {log['total_profit']: 4.4f}, "
            f"violation: {log['total_violation']: 4.4f}, "
        )
        if "excess_POD_violation" in log:
            pbar.set_description(f"excess_POD_violation: {log['excess_POD_violation']: 4.4f}, ")

        # Validation step
        if (step + 1) % int(train_updates * validation_freq) == 0:
            policy.eval()
            validation_performance = validate_policy(train_env, policy, n_step=n_step, )
            log.update(validation_performance)
            early_stopping.update_rewards(validation_performance["validation"]["traj_return"])
            if early_stopping.validation_check():
                print(f"Early stopping at epoch {step} due to {validation_patience} consecutive decreases in validation reward.")
                break
            # Save models (create a new directory for each validation); note that final model is also saved at the end
            save_models(policy, loss_module, critic, kwargs["algorithm"]["type"], kwargs, save_dir="saved_models/validation")
            policy.train()

        # Early stopping due to divergence
        check_loss = log["loss_actor"] if kwargs["algorithm"]["type"] == "sac" else log["total_loss"]
        if early_stopping.divergence_check(check_loss):
            break

        # Update wandb and scheduler
        wandb.log(log)
        actor_scheduler.step()
        if kwargs["algorithm"]["type"] == "sac":
            critic_scheduler.step()

    # Save models and close environment
    save_models(policy, loss_module, critic, kwargs["algorithm"]["type"], kwargs)
    train_env.close()

def get_performance_metrics(subdata:Dict, td:TensorDict, env:nn.Module) -> Dict:
    """Compute performance metrics for the policy."""
    out = {# Return
            "return": subdata["next", "reward"].mean().item(),
            "traj_return": subdata["next", "reward"].sum(dim=(-2, -1)).mean().item(),

            # Prediction
            "x": subdata["action"].mean().item(),
            "loc(x)": subdata["loc"].mean().item(),
            "scale(x)": subdata["scale"].mean().item(),

            # Constraints
            "total_violation": subdata["violation"].sum(dim=(-2,-1)).mean().item(),
            "demand_violation": subdata["violation"][...,0].sum(dim=(1)).mean().item(),
            "capacity_violation": subdata["violation"][...,1:-4].sum(dim=(1)).mean().item(),
            "LCG_violation": subdata["violation"][..., env.next_port_mask, -4:-2].sum(dim=(1,2)).mean().item(),
            "VCG_violation": subdata["violation"][..., env.next_port_mask, -2:].sum(dim=(1,2)).mean().item(),

            # Environment
            "total_revenue": subdata["revenue"].sum(dim=(-2,-1)).mean().item(),
            "total_cost": subdata["cost"].sum(dim=(-2,-1)).mean().item(),
            "total_profit": subdata["revenue"].sum(dim=(-2,-1)).mean().item() -
                            subdata["cost"].sum(dim=(-2,-1)).mean().item(),
            "total_loaded": subdata["action"].sum(dim=(-2,-1)).mean().item(),
            "total_demand":subdata["observation", "realized_demand"][:,0,:].sum(dim=-1).mean(),
            "total_e[x]_demand": td["observation", "init_expected_demand"][:, 0, :].sum(dim=-1).mean(),
            "mean_std[x]_demand": subdata["observation", "std_demand"][:, 0, :].std(dim=-1).mean(),
        }
    if "excess_POD_violation" in subdata["observation"]:
        out["excess_POD_violation"] = subdata["observation", "excess_POD_violation"].sum(dim=(1)).mean().item()
    return out

def validate_policy(env: nn.Module, policy_module: ProbabilisticActor, num_episodes: int = 10, n_step: int = 100,) -> Dict:
    """Validate the policy using the environment."""
    # Perform a rollout to evaluate the policy
    with torch.no_grad():
        trajectory = env.rollout(policy=policy_module, max_steps=n_step, auto_reset=True)
    val_metrics = get_performance_metrics(trajectory, trajectory, env)
    return {"validation": val_metrics}

def soft_update(target_params, source_params, tau):
    """Soft update the target parameters using the source parameters."""
    for target, source in zip(target_params.flatten_keys().values(), source_params.flatten_keys().values()):
        target.copy_(tau * source + (1.0 - tau) * target)

def save_models(policy, loss_module, critic, algorithm_type, kwargs_train, save_dir="saved_models"):
    """
    Save the policy and critic models with a timestamped directory structure.

    Args:
        policy (torch.nn.Module): The policy model to save.
        loss_module: Loss module containing Q-value networks and target Q-value networks (for SAC).
        critic (torch.nn.Module): The critic model to save (for non-SAC algorithms).
        algorithm_type (str): The type of algorithm (e.g., "sac").
        save_dir (str): Base directory for saving models.
    """
    # Generate a timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    # Save the policy model
    policy_save_path = os.path.join(save_path, "policy.pth")
    torch.save(policy.state_dict(), policy_save_path)
    wandb.save(policy_save_path)

    # Save the critic model(s)
    if algorithm_type == "sac":
        critic_paths = {
            "critic1": os.path.join(save_path, "critic1.pth"),
            "critic2": os.path.join(save_path, "critic2.pth"),
            "target_critic1": os.path.join(save_path, "target_critic1.pth"),
            "target_critic2": os.path.join(save_path, "target_critic2.pth"),
        }

        torch.save(loss_module.qvalue_network_params[0].state_dict(), critic_paths["critic1"])
        torch.save(loss_module.qvalue_network_params[1].state_dict(), critic_paths["critic2"])
        torch.save(loss_module.target_qvalue_network_params[0].state_dict(), critic_paths["target_critic1"])
        torch.save(loss_module.target_qvalue_network_params[1].state_dict(), critic_paths["target_critic2"])

        # Log critic models to wandb
        for path in critic_paths.values():
            wandb.save(path)
    else:
        critic_save_path = os.path.join(save_path, "critic.pth")
        torch.save(critic.state_dict(), critic_save_path)
        wandb.save(critic_save_path)

    # Save the configuration to a YAML file
    config_save_path = os.path.join(save_path, "config.yaml")
    cleaned_config = convert_to_dict(kwargs_train)  # Convert DotMap to dictionary
    with open(config_save_path, "w") as yaml_file:
        yaml.dump(cleaned_config, yaml_file, default_flow_style=False)
    wandb.save(config_save_path)