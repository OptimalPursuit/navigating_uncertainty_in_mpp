import json
import numpy as np
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Scenario Tree Evaluation Parameters")

parser.add_argument('--s', type=int, default=28, help='scenarios')
parser.add_argument('--p', type=int, default=3, help='Planning horizon or number of periods (p)')
parser.add_argument('--episodes', type=int, default=30, help='Number of episodes to evaluate')
parser.add_argument('--perfect_information', type=bool, default=True)
parser.add_argument('--gen', type=bool, default=False)
parser.add_argument('--cv', type=float, default=0.5, help='Coefficient of variation (cv)')
parser.add_argument('--teu', type=int, default=1000, help='TEU value')
parser.add_argument('--block_mpp', type=bool, default=False,)
parser.add_argument('--stochastic_algorithm', type=str, default="multi_stage", help='Stochastic algorithm type')
args = parser.parse_args()

# Access arguments like variables
s = args.s
p = args.p
episodes = args.episodes
perfect_information = args.perfect_information
gen = args.gen
teu = args.teu
cv = args.cv
block_mpp = args.block_mpp
stochastic_algorithm = args.stochastic_algorithm

# path = f'{folder}/cv={cv}'
base_dir = os.path.dirname(os.path.abspath(__file__))
folder_test_gen = 'testing' if not gen else 'generalization'
folder_pi_na = "multi_stage_pi" if perfect_information else "multi_stage_na"

if teu == 1000 and not block_mpp:
    input_path = os.path.join(base_dir, "navigating_uncertainty", "teu1k", folder_pi_na, "instances")
    output_path = os.path.join(base_dir, "navigating_uncertainty", "teu1k", folder_pi_na)
elif teu == 20000 and block_mpp:
    input_path = os.path.join(base_dir, "scenario_tree", "block_mpp",)
    output_path = os.path.join(base_dir, "AI2STOW", "teu20k", f"p{p}", folder_pi_na)
else:
    raise ValueError("Invalid TEU or block_mpp configuration")

# Data containers
obj = []
time = []
total_load = []
total_demand = []
total_revenue = []
total_cost = []

# Load data
for e in range(episodes):
    if teu == 1000 and not block_mpp:
        json_path = f'{input_path}/results_scenario_tree_teu{teu}_p{p}_e{e}_s{s}_alg{stochastic_algorithm}_pi{perfect_information}_gen{gen}.json'
    elif teu == 20000 and block_mpp:
        json_path = f'{input_path}/results_scenario_tree_teu{teu}_p{p}_e{e}_s{s}_pi{perfect_information}_gen{gen}.json'
    else:
        raise ValueError("Invalid TEU or block_mpp configuration")

    with open(json_path, 'r') as file:
        x = json.load(file)
        obj.append(x["obj"])
        time.append(abs(x["time"])) # ensure non-negative time
        total_load.append(np.sum(x["mean_load_per_port"], axis=-1))
        total_demand.append(np.sum(x["mean_demand"], axis=-1))
        total_revenue.append(np.sum(x["mean_revenue"], axis=-1))
        total_cost.append(np.sum(x["mean_cost"], axis=-1))

# Statistics container
stats_summary = {}

# Helper function to collect and print statistics
def collect_stats(data, name=""):
    """Collects and prints summary statistics of the data"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    z = 1.96
    LB = mean - z * std / np.sqrt(n)
    UB = mean + z * std / np.sqrt(n)

    # Print
    print(f"\n{name}")
    print(f"n: {n}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"95% CI: ({LB}, {UB})")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")

    # Save in dictionary
    stats_summary[name] = {
        "n": n,
        "mean": mean,
        "std": std,
        "95% CI": [LB, UB],
        "min": min_val,
        "max": max_val
    }

# Collect and print stats
collect_stats(obj, "Objective")
collect_stats(time, "Time")
collect_stats(total_load, "Load")
collect_stats(total_demand, "Demand")
collect_stats(total_revenue, "Revenue")
collect_stats(total_cost, "Cost")

# Save to JSON file
summary_json_path = os.path.join(output_path, f"summary_stats_teu{teu}_p{p}_s{s}_pi{perfect_information}_gen{gen}.json")
with open(summary_json_path, 'w') as f:
    json.dump(stats_summary, f, indent=4)

print(f"\nSummary statistics saved to {output_path}")
