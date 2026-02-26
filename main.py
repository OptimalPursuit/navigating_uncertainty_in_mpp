## Imports
import os
import argparse
import json

# Datatypes
import yaml
from dotmap import DotMap
from typing import Union
from tensordict.nn import TensorDictModule

# Machine learning
import wandb

# TorchRL
from torchrl.envs.utils import check_env_specs
from torchrl.modules import TruncatedNormal

# Custom:
# Training
from rl_algorithms.utils import make_env, adapt_env_kwargs
from rl_algorithms.train import run_training
# Models
from models.embeddings import *
from models.autoencoder import Autoencoder
from models.encoder import MLPEncoder, AttentionEncoder
from models.decoder import AttentionDecoderWithCache, MLPDecoderWithCache
from models.critic import CriticNetwork
from models.projection import ProjectionFactory
from rl_algorithms.projection_prob_actor import ProjectionProbabilisticActor
from rl_algorithms.test import evaluate_model

# Functions
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)
    return config

import random
import numpy as np
import torch

def setup_torch(seed: int = 42) -> None:
    """Initialize Torch settings for deterministic behavior and efficiency."""
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)

def load_trained_hyperparameters(path: str) -> DotMap:
    """Load hyperparameters from a previously trained model."""
    config_path = f"{path}/config.yaml"
    config = load_config(config_path)

    # Add hyperparameters if they exist
    for i in range(25):
        key = f"lagrangian_multiplier_{i}"
        if key in config.algorithm:
            config.algorithm[key] = config.algorithm[key]

    return config


def initialize_encoder(encoder_type:str, encoder_args:Dict, device:str) -> nn.Module:
    """Initialize the encoder based on the type."""
    if encoder_type == "attention":
        return AttentionEncoder(**encoder_args).to(device)
    elif encoder_type == "mlp":
        return MLPEncoder(**encoder_args).to(device)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

def initialize_decoder(decoder_type:str, decoder_args:Dict, device:str) -> nn.Module:
    """Initialize the decoder based on the type."""
    if decoder_type == "attention":
        return AttentionDecoderWithCache(**decoder_args).to(device)
    elif decoder_type == "mlp":
        return MLPDecoderWithCache(**decoder_args).to(device)
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")

def initialize_critic(algorithm_type:str, encoder:nn.Module, critic_args:Dict, device:str) -> nn.Module:
    """Initialize the critic based on the algorithm type."""
    if algorithm_type == "sac":
        out_keys = ["state_action_value", "lagrangian_multiplier"] if critic_args.get("primal_dual", True) else ["state_action_value"]
        return TensorDictModule(
            CriticNetwork(encoder, customized=True, use_q_value=True, **critic_args).to(device),
            in_keys=["observation", "action"],
            out_keys=out_keys
        )
    else:
        out_keys = ["state_value", "lagrangian_multiplier"] if critic_args.get("primal_dual", True) else ["state_value"]
        return TensorDictModule(
            CriticNetwork(encoder, customized=True, **critic_args).to(device),
            in_keys=["observation"],
            out_keys=out_keys
        )

def initialize_projection_layer(projection_type:str, projection_kwargs:DotMap,
                                action_dim:int, n_constraints:int) -> nn.Module:
    """Initialize the projection layer based on the projection type."""
    projection_type = (projection_type or "").lower()  # Normalize to lowercase and handle None
    projection_kwargs["n_action"] = action_dim
    projection_kwargs["n_constraints"] = n_constraints
    return ProjectionFactory.create_class(projection_type, projection_kwargs)

def initialize_policy_and_critic(config: DotMap, env:nn.Module, device:Union[str,torch.device]) -> Tuple[nn.Module, nn.Module]:
    """
    Initializes the policy and critic models based on the given configuration and environment.

    Args:
        config: Configuration object containing model, training, and algorithm settings.
        env: Environment object containing action specifications and other parameters.
        device: The device (CPU/GPU) to initialize the models on.

    Returns:
        policy: The initialized policy model.
        critic: The initialized critic model.
    """
    # Validate input
    assert hasattr(config, 'model'), "Config object must have a 'model' attribute."
    assert hasattr(env, 'action_spec'), "Environment must have an 'action_spec' attribute."

    # Embedding dimensions
    embed_dim = config.model.embed_dim
    action_dim = env.action_spec.shape[0]
    sequence_dim = env.K * env.T if env.action_spec.shape[0] > env.P-1 else env.P - 1

    # Embedding initialization
    critic_embed = CriticEmbedding(action_dim, embed_dim, sequence_dim, env,)
    init_embed = CargoEmbedding(action_dim, embed_dim, sequence_dim, env)
    context_embed = ContextEmbedding(action_dim, embed_dim, sequence_dim, env,)
    if config.model.dyn_embed == "self_attention":
        dynamic_embed = DynamicSelfAttentionEmbedding(embed_dim, sequence_dim, env)
    elif config.model.dyn_embed == "ffn":
        dynamic_embed = DynamicEmbedding(embed_dim, sequence_dim, env)
    else:
        raise ValueError(f"Unsupported dynamic embedding type: {config.model.dyn_embed}")

    # Model arguments
    decoder_args = {
        "embed_dim": embed_dim,
        "action_dim": action_dim,
        "seq_dim": sequence_dim,
        "init_embedding": init_embed,
        "context_embedding": context_embed,
        "dynamic_embedding": dynamic_embed,
        "critic_embedding": critic_embed,
        "obs_embedding": critic_embed,
        **config.model,
    }
    encoder_args = {
        "embed_dim": embed_dim,
        "init_embedding": init_embed,
        "env_name": env.name,
        **config.model,
    }
    critic_args = {
        "embed_dim": embed_dim,
        "action_dim": action_dim,
        "critic_embedding": critic_embed,
        "primal_dual": config.algorithm.primal_dual,
        "n_constraints": config.training.projection_kwargs.n_constraints,
        **config.model,
    }

    # Init models: encoder, decoder, and critic
    encoder = initialize_encoder(config.model.encoder_type, encoder_args, device)
    decoder = initialize_decoder(config.model.decoder_type, decoder_args, device)
    critic = initialize_critic(config.algorithm.type, encoder, critic_args, device)

    # Init projection layer
    projection_layer = initialize_projection_layer(
        config.training.projection_type,
        config.training.projection_kwargs,
        action_dim,
        env.n_constraints
    )

    # Init actor (policy)
    actor = TensorDictModule(
        Autoencoder(encoder, decoder, env).to(device),
        in_keys=["observation"],  # Input tensor key in TensorDict
        out_keys=["loc", "scale", "mask"]  # Output tensor key in TensorDict
    )
    policy = ProjectionProbabilisticActor(
        module=actor,
        in_keys=["loc", "scale",],
        distribution_class=TruncatedNormal,
        distribution_kwargs={"low": env.action_spec.low[0], "high": env.action_spec.high[0]},
        return_log_prob=True,
        projection_layer=projection_layer,
        projection_type=config.training.projection_type,
        jacobian_correction=config.training.projection_kwargs.get("jacobian_correction", True),
        spec=env.action_spec,
        revenues=env.revenues,
    )

    return policy, critic


# Main function
def main(config: Optional[DotMap] = None, **kwargs) -> None:
    """
    Main function to train or test the model based on the configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_torch(config.env.seed)

    ## Environment initialization
    env = make_env(config.env)
    env.set_seed(config.env.seed)
    check_env_specs(env)

    ## Main loop
    path = f"{config.testing.path}/{config.testing.folder}"
    if config.model.phase in {"train", "tuned_training"}:
        # Initialize models and run training
        wandb.init(config=config,) #mode="offline")
        policy, critic = initialize_policy_and_critic(config, env, device)
        run_training(policy, critic, **config)

    elif config.model.phase == "test":
        alpha = config.training.projection_kwargs.alpha
        delta = config.training.projection_kwargs.delta
        use_spectral_eta = config.training.projection_kwargs.use_spectral_eta
        max_iter = config.training.projection_kwargs.max_iter
        power_iters = config.training.projection_kwargs.power_iters
        vp_str = f"{max_iter}_{power_iters}_spectr{use_spectral_eta}_alpha{config.training.projection_kwargs.enable_alpha_map}"
        policy, critic = initialize_policy_and_critic(config, env, device)

        # Evaluate policy
        policy_load_path = f"{path}/policy.pth"
        missing, unexpected = policy.load_state_dict(torch.load(policy_load_path, map_location=device), strict=False)

        metrics, summary_stats = evaluate_model(policy, config, device=device, critic=critic, **config.testing)
        print(summary_stats)

        # Save summary statistics in path
        if "feasibility_recovery" in config.testing:
            file_name = f"P{config.env.ports}_frec{config.testing.feasibility_recovery}_" \
                   f"cv{config.env.cv_demand}_gen{config.env.generalization}_{config.training.projection_type}" \
                        f"_{config.training.projection_kwargs.slack_penalty}_PBS{config.env.block_stowage_mask}" \
                        f"_UR{config.env.utilization_rate_initial_demand}_VP{vp_str}.yaml"
        else:
            file_name = f"P{config.env.ports}_cv{config.env.cv_demand}" \
                        f"_gen{config.env.generalization}_{config.training.projection_type}" \
                        f"_{config.training.projection_kwargs.slack_penalty}_PBS{config.env.block_stowage_mask}" \
                        f"_UR{config.env.utilization_rate_initial_demand}_VP{vp_str}.yaml"
        with open(f"{path}/{file_name}", "w") as file:
                        yaml.dump(summary_stats, file)

def parse_args(sweep: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script with WandB integration.")
    # Sweep parameters
    if sweep:
        parser.add_argument("--sweep", nargs="?", default=None, const=None,
                            help="Provide a sweep name to resume an existing sweep, or leave empty to create a new sweep.")
        parser.add_argument('--runs_per_agent', type=int, default=100, help="Number of runs per agent.")

    # Environment parameters
    parser.add_argument('--env_name', type=str, default='mpp', help="Name of the environment.")
    parser.add_argument('--ports', type=int, default=4, help="Number of ports in env.")
    parser.add_argument('--teu', type=int, default=1000, help="TEU capacity of the ship.")
    parser.add_argument('--gen', type=bool, default=False, help="Whether to test generalization to different demand distributions.")
    parser.add_argument('--ur', type=float, default=1.1, help="Utilization rate of initial demand.")
    parser.add_argument('--cv', type=float, default=0.5, help="Coefficient of variation for demand generation.")
    # Generator parameters
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--demand_sparsity', type=int, default=0.3, help="Sparsity level of demand.")
    parser.add_argument('--demand_perturbation', type=float, default=0.2, help="Perturbation level of demand.")
    parser.add_argument('--duration_variable_revenue', type=bool, default=False, help="Variable revenue parameter over duration.")
    parser.add_argument('--loading_discharge_region', type=bool, default=False, help="Use loading/discharge regions in generator.")
    parser.add_argument('--use_dirichlet_partition', type=bool, default=True, help="Use Dirichlet partition for demand generation.")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, help="Alpha parameter for Dirichlet distribution.")
    parser.add_argument('--spot_percentage', type=float, default=0.3, help="Percentage of spot demand.")

    # Algorithm parameters
    parser.add_argument('--algorithm_type', type=str, default='sac', help="Type of algorithm to use.")
    parser.add_argument('--feasibility_lambda', type=float, default=0., help="Lambda for feasibility.")
    parser.add_argument('--primal_dual', type=bool, default=False, help="Enable primal-dual method.")

    # Model parameters
    parser.add_argument('--encoder_type', type=str, default='attention', help="Type of encoder to use.")
    parser.add_argument('--decoder_type', type=str, default='attention', help="Type of decoder to use.")
    parser.add_argument('--dyn_embed', type=str, default='self_attention', help="Dynamic embedding type.")
    parser.add_argument('--embed_dim', type=int, default=128, help="Dimension of embeddings.")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Dimension of hidden layers.")
    parser.add_argument('--temperature', type=int, default=1.0, help="Temperature of policy.")
    parser.add_argument('--scale_max', type=float, default=2.0, help="Maximum value of policy scale.")
    parser.add_argument('--block_stowage_mask', type=bool, default=False, help="Block stowage mask.")
    parser.add_argument('--use_mask_head', type=bool, default=False, help="Learn mask to optimize paired block stowage.")
    parser.add_argument('--use_preload_mask', type=bool, default=False, help="Use preloaded mask for paired block stowage.")
    parser.add_argument('--normalize_constraints', type=bool, default=False, help="Normalize constraints.")
    parser.add_argument('--projection_type', type=str, default="none", help="Projection type.")
    parser.add_argument('--projection_kwargs', type=json.loads, default={
        'alpha': 0.01, 'delta': 0.01, 'max_iter': 100, 'slack_penalty': 10000, 'n_action': 20, 'n_constraints': 25,
        'spectral_norm': 'svd',  # power_iters, power_iters, 'frobenius'
        'power_iters': 3, 'enable_alpha_map':False, 'enforce_nonneg':True,  'jacobian_correction':True,
        }, help="Projection parameters as JSON string.")

    # Run parameters
    # lr: 0.00014690714579803494
    # pd_lr: 0.000034690714579803494
    parser.add_argument('--optimizer', type=str, default="Adam", help="Optimizer type.")
    parser.add_argument('--learning_rate', type=float, default=0.000012, help="Learning rate for the optimizer.")
    parser.add_argument('--pd_learning_rate', type=float, default=0.0003, help="Learning rate for primal-dual optimizer.")
    parser.add_argument('--testing_path', type=str, default='results/trained_models/navigating_uncertainty_ECML', help="Path for testing results.")
    parser.add_argument('--folder', type=str, default='sac-vp', help="Folder name for the run.")
    parser.add_argument('--phase', type=str, default='train', help="WandB project name.")
    parser.add_argument('--feasibility_recovery', type=bool, default=False, help="Enable feasibility recovery.")
    parser.add_argument('--num_episodes', type=int, default=30, help="Number of test episodes.")
    return parser.parse_args()

def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


if __name__ == "__main__":
    # Load static configuration from the YAML file
    file_path = os.getcwd()

    # Load config and possibly re-load config is one in results folder
    config = load_config(f'{file_path}/config.yaml')
    folder_path = os.path.join(file_path, config.testing.path, config.testing.folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Check if a config file exists in the folder and load it
    config_path = os.path.join(folder_path, "config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)

    # Parse command-line arguments for dynamic configuration
    args = parse_args()
    # Env
    config.env.env_name = args.env_name
    config.env.ports = args.ports
    config.env.TEU = args.teu
    config.env.generalization = args.gen
    config.env.utilization_rate_initial_demand = args.ur
    config.env.cv_demand = args.cv
    config.env.block_stowage_mask = args.block_stowage_mask
    config.env.normalize_constraints = args.normalize_constraints
    # Generator
    config.env.seed = args.seed
    config.env.demand_sparsity = args.demand_sparsity
    config.env.demand_perturbation = args.demand_perturbation
    config.env.duration_variable_revenue = args.duration_variable_revenue
    config.env.loading_discharge_region = args.loading_discharge_region
    config.env.use_dirichlet_partition = args.use_dirichlet_partition
    config.env.dirichlet_alpha = args.dirichlet_alpha
    config.env.spot_percentage = args.spot_percentage

    # Algorithm
    config.algorithm.type = args.algorithm_type
    config.algorithm.feasibility_lambda = args.feasibility_lambda
    config.algorithm.primal_dual = args.primal_dual
    # Model
    config.model.encoder_type = args.encoder_type
    config.model.decoder_type = args.decoder_type
    config.model.dyn_embed = args.dyn_embed
    config.model.embed_dim = args.embed_dim
    config.model.hidden_dim = args.hidden_dim
    config.model.temperature = args.temperature
    config.model.scale_max = args.scale_max
    config.model.use_mask_head = args.use_mask_head
    config.model.use_preload_mask = args.use_preload_mask
    # Run
    config.training.optimizer = args.optimizer
    config.training.learning_rate = args.learning_rate
    config.training.pd_learning_rate = args.pd_learning_rate
    config.training.projection_type = args.projection_type
    config.training.projection_kwargs = DotMap(args.projection_kwargs)
    config.testing.path = args.testing_path
    config.testing.folder = args.folder
    config.model.phase = args.phase
    config.testing.feasibility_recovery = args.feasibility_recovery
    config.testing.num_episodes = args.num_episodes

    if args.feasibility_recovery or (config.training.projection_type == "frank_wolfe" and config.model.phase == "test"):
        config.training.projection_type = "convex_program"
    if config.training.projection_type == "inner_convex_violation_alpha":
        config.training.projection_kwargs.enable_alpha_map = True

    # todo: check the logic of load_config and adapt_env_kwargs, as they are currently called multiple times.
    #  Maybe they can be merged into one function that also handles the command-line arguments?
    config = adapt_env_kwargs(config)

    print(f"Running with folder: {config.testing.folder}, "
          f"algorithm type: {config.algorithm.type},"
          f"generalization: {config.env.generalization},"
          f"projection type: {config.training.projection_type},"
          f"spectral_norm: {config.training.projection_kwargs.spectral_norm},",
          f"proj_iterations: {config.training.projection_kwargs.max_iter},"
          f"power_iters: {config.training.projection_kwargs.power_iters}")

    # Call your main() function
    ## todo: Likely a bunch of warnings will be thrown, but they are not critical. Should be fixed soon.
    try:
        model = main(config)
    except Exception as e:
        # Log the error to WandB
        wandb.log({"error": str(e)})

        # Optionally, use WandB alert for critical errors
        wandb.alert(
            title="Training Error",
            text=f"An error occurred during training: {e}",
            level="error"  # 'info' or 'warning' levels can be used as needed
        )

        # Print the error for local console logging as well
        print(f"An error occurred during training: {e}")
    finally:
        wandb.finish()