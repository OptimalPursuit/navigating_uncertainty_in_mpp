import time
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from tensordict import TensorDict
from dotmap import DotMap
from typing import Dict, Optional, Tuple, Union
from rl_algorithms.utils import make_env
from rl_algorithms.train import get_performance_metrics
from rl_algorithms.utils import set_unique_seed

# Functions
def get_z_score_torch(confidence_level:float) -> float:
    """
    Compute the z-score for a given confidence level using PyTorch.
    Args:
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence).
    Returns:
        float: The z-score corresponding to the confidence level.
    """
    alpha = 1 - confidence_level
    return torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)).item()

def compute_summary_stats(metrics:Dict, confidence_level:float=0.95) -> Dict:
    """
    Compute mean, median, std, min, and max for each metric in the dictionary.
    Args:
        metrics (dict): Dictionary with metrics as tensors.
    Returns:
        dict: Summary statistics for each metric.
    """
    summary_stats = {}
    z = get_z_score_torch(confidence_level)
    for key, values in metrics.items():
        # transform values to list:
        val_list = values.tolist()

        n = values.numel()  # Number of elements in the tensor
        mean = values.mean().item()
        std = values.std().item()
        margin_of_error = z * (std / math.sqrt(n)) if n > 1 else 0.0  # CI margin of error

        summary_stats[key] = {
            "mean": mean,
            "median": values.median().item(),
            "std": std,
            "min": values.min().item(),
            "max": values.max().item(),
            "lb_ci": mean - margin_of_error,  # Lower bound of the CI
            "ub_ci": mean + margin_of_error,  # Upper bound of the CI
            "value": val_list,  # Store the tensor for further analysis
        }
    return summary_stats

# Main function
def evaluate_model(policy:nn.Module, config:DotMap, device:Union[str,torch.device]="cuda", **kwargs) -> Tuple[Dict, Dict]:
    """
    Evaluate the policy and critic on the test environment.
    Args:
        policy (nn.Module): The policy to evaluate.
        config: Configuration object with environment parameters.
        device (torch.device): Device to run the evaluation on.
        **kwargs: Additional arguments for evaluation.
    """
    # Extract evaluation hyperparameters
    env_kwargs = config.env
    num_episodes = kwargs.get("num_episodes", 10)

    # Create the test environment
    max_paths = 2 # Run small batch, as we care about instances
    test_env = make_env(env_kwargs, batch_size=[max_paths], device=device)
    n_step = test_env.T * test_env.K  # Maximum steps per episode (T x K)
    feas_threshold = 1.0
    delta = 1.0
    num_rollouts = 1  # Number of rollouts per episode

    # Set policy to evaluation mode
    policy.eval()  # Set policy to evaluation mode

    # Initialize metrics storage
    metrics = {
        "total_profit": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "total_violations": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "inference_times": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "feasible_instance":torch.zeros(num_episodes, device=device),  # [num_episodes]
        "demand_violations": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "capacity_violations": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "stability_violations": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "pbs_violations": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "max_revenues": torch.zeros(num_episodes, device=device),  # [num_episodes]
    }

    with torch.no_grad():
        # Warm-up phase
        for _ in tqdm(range(3), desc="Warm-up"):
            _ = test_env.rollout(
                policy=policy,
                max_steps=n_step,
                auto_reset=True,
            )

        for episode in tqdm(range(num_episodes), desc="Episodes"):
            seed = config.env.seed + episode + 1
            set_unique_seed(seed)
            config.env.seed = seed

            # Setup new environment
            gen_env = make_env(config.env, batch_size=[max_paths], device='cpu')
            td = gen_env.reset().to(device)

            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.perf_counter()

            feasible_rollouts = []
            fallback_rollouts = []

            for r in range(num_rollouts):
                td_r = td.clone()

                traj = test_env.rollout(
                    policy=policy,
                    max_steps=n_step,
                    auto_reset=False,
                    tensordict=td_r,
                )

                # Post-process violations
                zero_idx = torch.where((traj["action"][0] == 0.0) & (traj["clip_max"][0] == 0.0))
                traj["violation"][0][zero_idx] = 0.0

                # Calculate profit and max revenue
                profit = traj["profit"][0].sum().item()
                max_revenue = (test_env.revenues * traj["observation", "realized_demand"][0,0]).sum().item()

                # Feasibility check (same as your logic)
                violation_adjusted = traj["violation"][0].clone()
                v_mask = violation_adjusted[:, :-4] < delta
                violation_adjusted[:, :-4][v_mask] = 0.0
                violation_adjusted[:, -4:][~test_env.next_port_mask] = 0.0
                total_violation = violation_adjusted.sum().item()
                is_feasible = total_violation <= feas_threshold

                rollout_info = {
                    "trajectory": traj,
                    "profit": profit,
                    "total_violation": total_violation,
                    "feasible": is_feasible,
                    "max_revenue": max_revenue,
                }

                if is_feasible:
                    feasible_rollouts.append(rollout_info)
                fallback_rollouts.append(rollout_info)

            # Choose best feasible rollout (or fallback to best overall)
            if feasible_rollouts:
                best = max(feasible_rollouts, key=lambda x: x["profit"])
            else:
                best = max(fallback_rollouts, key=lambda x: x["profit"])

            trajectory = best["trajectory"]

            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.perf_counter()

            # Metrics
            metrics["total_profit"][episode] = best["profit"]
            metrics["total_violations"][episode] = best["total_violation"]
            metrics["inference_times"][episode] = end_time - start_time
            metrics["feasible_instance"][episode] = 1.0 if best["feasible"] else 0.0
            metrics["max_revenues"][episode] = best["max_revenue"]

            # Detailed violation metrics
            violation_adjusted = trajectory["violation"][0].clone()
            v_mask = violation_adjusted[:, :-4] < delta
            violation_adjusted[:, :-4][v_mask] = 0.0
            violation_adjusted[:, -4:][~test_env.next_port_mask] = 0.0

            metrics["demand_violations"][episode] = violation_adjusted[:, 0].sum()
            metrics["capacity_violations"][episode] = violation_adjusted[:, 1:-4].sum()
            metrics["stability_violations"][episode] = violation_adjusted[:, -4:].sum()
            if "excess_pod_locations" in trajectory["observation"]:
                # Assuming excess_pod_locations is a tensor in the observation
                metrics["pbs_violations"][episode] = trajectory["observation", "excess_pod_locations"][0].sum()

            # Cleanup
            gen_env.close()
            del gen_env

    # Summarize episode-level metrics (mean and std)
    summary_stats = compute_summary_stats(metrics)
    test_env.close()
    return metrics, summary_stats