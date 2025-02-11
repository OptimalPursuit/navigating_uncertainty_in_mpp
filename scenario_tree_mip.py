# Imports
import numpy as np
import torch as th
import random
import yaml
from dotmap import DotMap
from docplex.mp.model import Model
import sys
import os
import json

path = 'add path to cplex here'
sys.path.append(path)

# Module imports
path_to_main = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(path_to_main)
from main import adapt_env_kwargs, make_env
from environment.utils import get_pol_pod_pair
from rl_algorithms.utils import set_unique_seed


# Precompute functions
def precompute_node_list(stages, scenarios_per_stage):
    """Precompute the list of nodes and their coordinates in the scenario tree"""
    node_list = []  # List to store the coordinates of all nodes
    # Loop over each stage, starting from stage 1 (root is stage 1)
    for stage in range(stages):
        nodes_in_current_stage = scenarios_per_stage ** (stage)  # Number of nodes at this stage

        # For each node in the current stage
        for node_id in range(nodes_in_current_stage):
            node_list.append((stage, node_id))

    return node_list

def precompute_demand(node_list, max_paths, stages, env):
    """Precompute the demand scenarios for each node in the scenario tree"""
    td = env.reset()
    pregen_demand = td["observation", "realized_demand"].detach().cpu().numpy().reshape(-1, env.T, env.K)

    # Preallocate demand array for transport demands
    demand_ = np.zeros((max_paths, env.K, env.P, env.P))
    # Precompute transport demands for all paths
    for transport in range(env.T):
        pol, pod = get_pol_pod_pair(th.tensor(transport), env.P)
        demand_[:, :, pol, pod] = pregen_demand[:, transport, :]

    demand_ = demand_.transpose(2, 0, 1, 3)

    # Populate demand scenarios
    demand_scenarios = {}
    for (stage, node_id) in node_list:
        demand_scenarios[stage, node_id] = demand_[stage, node_id, :, :]

    # todo: Allow for deterministic
    # Real demand
    real_demand = {}
    for stage in range(stages):
        real_demand[stage, max_paths, 0] = demand_[stage, 0, :, :]

    if deterministic:
        for (stage, node_id) in node_list:
            demand_scenarios[stage, node_id] = real_demand[stage, max_paths, 0]

    return demand_scenarios, real_demand

def get_scenario_tree_indices(scenario_tree, num_scenarios):
    """
    Extracts data from a scenario tree structure, keeping all stages but limiting nodes
    according to the number of scenarios.

    Args:
        scenario_tree (dict): Dictionary representing the tree with keys [stage, nodes].
        num_scenarios (int): Number of scenarios to extract at each stage.

    Returns:
        dict: Filtered scenario tree with limited nodes at each stage.
    """
    filtered_tree = {}

    for (stage, node), value in scenario_tree.items():
        # Calculate the maximum number of nodes for this stage
        max_nodes = num_scenarios ** stage
        # Include only nodes within the allowed range
        if node < max_nodes:
            filtered_tree[(stage, node)] = value
    return filtered_tree

# Support functions
def get_demand_history(stage, demand, num_nodes_per_stage):
    """Get the demand history up to the given stage for the given scenario"""
    if stage > 0:
        demand_history = []
        for s in range(stage):
            for node_id in range(num_nodes_per_stage[s]):
                # Concatenate predicted demand history for the current scenario up to the given stage
                demand_history.append(demand[s, node_id,].flatten())
        return np.array(demand_history)
    else:
        # If there's no history (stage 0), return an empty array or some other initialization
        return np.array([])  # Or use np.zeros((shape,))

def onboard_groups(ports:int, pol:int, transport_indices:list) -> np.array:
    load_index = np.array([transport_indices.index((pol, i)) for i in range(ports) if i > pol])  # List of cargo groups to load
    load = np.array([transport_indices[idx] for idx in load_index]).reshape((-1,2))
    discharge_index = np.array([transport_indices.index((i, pol)) for i in range(ports) if i < pol])  # List of cargo groups to discharge
    discharge = np.array([transport_indices[idx] for idx in discharge_index]).reshape((-1,2))
    port_moves = np.vstack([load, discharge]).astype(int)
    on_board = [(i, j) for i in range(ports) for j in range(ports) if i <= pol and j > pol]  # List of cargo groups to load
    return np.array(on_board), port_moves, load

# Main function
def main(env, demand, scenarios_per_stage=28, stages=3, max_paths=784,
         seed=42, perfect_information=False, deterministic=False, warm_start=None):
    # Scenario tree parameters
    M = 10 ** 3 # Big M
    num_nodes_per_stage = [1*scenarios_per_stage**stage for stage in range(stages)]

    # Problem parameters
    P = env.P
    B = env.B
    D = env.D
    K = env.K
    T = env.T
    stab_delta = env.stab_delta
    LCG_target = env.LCG_target
    VCG_target = env.VCG_target
    CI_target_parameter = env.CI_target
    teus = env.teus.detach().cpu().numpy()
    weights = env.weights.detach().cpu().numpy()
    revenues = env.revenues.detach().cpu().numpy()
    capacity = env.capacity.detach().cpu().numpy()
    longitudinal_position = env.longitudinal_position.detach().cpu().numpy()
    vertical_position = env.vertical_position.detach().cpu().numpy()

    # Create a CPLEX model
    mdl = Model(name="multistage_mpp")

    # Decision variable dictionaries
    x = {} # Cargo allocation
    HO = {} # Hatch overstowage
    HM = {} # Hatch move
    CI = {} # Crane intensity
    CI_target = {} # Crane intensity target
    CM = {} # Crane move
    spread_moves_bay = {} # Spread moves bay
    LM = {} # Longitudinal moment
    VM = {} # Vertical moment
    TW = {} # Total weight

    # Sets of different stages
    on_boards = []
    all_port_moves = []
    all_load_moves = []
    transport_indices = [(i, j) for i in range(P) for j in range(P) if i < j]

    def build_tree(stages, demand, mip_start=None):
        """Function to build the scenario tree; with decisions and constraints for each node"""
        for stage in range(stages):
            for node_id in range(num_nodes_per_stage[stage]):
                # Crane intensity:
                CI[stage, node_id] = mdl.continuous_var(name=f'CI_{stage}_{node_id}')
                CI_target[stage, node_id] = mdl.continuous_var(name=f'CI_target_{stage}_{node_id}')
                CM[stage, node_id] = mdl.continuous_var(name=f'CM_{stage}_{node_id}')
                spread_moves_bay[stage, node_id] = mdl.continuous_var(name=f'spread_moves_bay_{stage}_{node_id}')

                # Stability:
                LM[stage, node_id] = mdl.continuous_var(name=f'LM_{stage}_{node_id}')
                VM[stage, node_id] = mdl.continuous_var(name=f'VM_{stage}_{node_id}')
                TW[stage, node_id] = mdl.continuous_var(name=f'TW_{stage}_{node_id}')

                for bay in range(B):
                    # Hatch overstowage:
                    HO[stage, node_id, bay] = mdl.continuous_var(name=f'HO_{stage}_{node_id}_{bay}')
                    HM[stage, node_id, bay] = mdl.binary_var(name=f'HM_{stage}_{node_id}_{bay}')

                    for deck in range(D):
                        for cargo_class in range(K):
                            for pol in range(stage + 1):
                                for pod in range(pol + 1, P):
                                    # Cargo allocation:
                                    x[stage, node_id, bay, deck, cargo_class, pol, pod] = \
                                            mdl.continuous_var(name=f'x_{stage}_{node_id}_{bay}_{deck}_{cargo_class}_{pol}_{pod}')

            # Define sets
            # Current port
            on_board, port_moves, load_moves = onboard_groups(P, stage, transport_indices)
            on_boards.append(on_board)
            all_port_moves.append(port_moves)
            all_load_moves.append(load_moves)

            # constraints for the current stage and node
            for node_id in range(num_nodes_per_stage[stage]):
                print(f"Stage {stage}, Node {node_id}")
                for (i, j) in on_board:
                    for k in range(K):
                        # Demand satisfaction
                        mdl.add_constraint(
                            mdl.sum(x[stage, node_id, b, d, k, i, j] for b in range(B) for d in range(D))
                            <= demand[stage, node_id][k, j]
                        )

                for b in range(B):
                    for d in range(D):
                        # TEU capacity
                        mdl.add_constraint(
                            mdl.sum(teus[k] * mdl.sum(x[stage, node_id, b, d, k, i, j]
                                                      for (i, j) in on_board) for k in range(K))
                            <= capacity[b, d]
                        )

                        if not perfect_information:
                            demand_history1 = get_demand_history(stage, demand, num_nodes_per_stage)
                            for node_id2 in range(node_id + 1, num_nodes_per_stage[stage]):
                                demand_history2 = get_demand_history(stage, demand, num_nodes_per_stage)
                                if np.allclose(demand_history1, demand_history2, atol=1e-5):  # Use a tolerance if floats
                                    for k in range(K):
                                        for (i, j) in load_moves:
                                            # Non-anticipation at stage, provided demand history is similar
                                            mdl.add_constraint(
                                                x[stage, node_id, b, d, k, i, j] == x[stage, node_id2, b, d, k, i, j]
                                            )

                    # Open hatch (d=1 is below deck)
                    mdl.add_constraint(
                        mdl.sum(x[stage, node_id, b, 1, k, i, j] for (i, j) in all_port_moves[stage] for k in range(K))
                        <= M * HM[stage, node_id, b]
                    )

                    # Hatch overstows (d=0 is on deck)
                    # Overstowage is arrival condition of previous port: on_boards[stage - 1]
                    # Vessel is empty before stage 0, hence no overstows
                    if stage > 0:
                        mdl.add_constraint(
                            mdl.sum(x[stage, node_id, b, 0, k, i, j]for (i, j) in on_boards[stage - 1]
                                    for k in range(K) if j > stage) - M * (1 - HM[stage, node_id, b] )
                            <= HO[stage, node_id, b]
                        )

                # Stability
                mdl.add_constraint(
                    TW[stage, node_id] == mdl.sum(weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j]
                                                                               for (i, j) in on_board for d in range(D))
                                                          for k in range(K) for b in range(B))
                )

                # LCG
                mdl.add_constraint(
                    LM[stage, node_id] == mdl.sum(longitudinal_position[b] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D))
                    for k in range(K)) for b in range(B)))
                mdl.add_constraint(
                    stab_delta * mdl.sum(weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D))
                    for k in range(K) for b in range(B)) >= mdl.sum(longitudinal_position[b] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K)) for b in range(B)) - LCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K) for b in range(B)))
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K) for b in range(B)) >= - mdl.sum(longitudinal_position[b] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K)) for b in range(B)) + LCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for d in range(D)) for k in
                    range(K) for b in range(B)))

                # VCG
                mdl.add_constraint(VM[stage, node_id] == mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K)) for d in range(D)))
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)) >= mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K)) for d in range(D)) - VCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)))
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)) >= - mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K)) for d in range(D))  + VCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in on_board for b in range(B)) for k in
                    range(K) for d in range(D)))

                # Compute lower bound on long crane
                for adj_bay in range(B - 1):
                    mdl.add_constraint(
                        mdl.sum(x[stage, node_id, b, d, k, i, j] for (i, j) in all_port_moves[stage] for k in range(K)
                                for d in range(D) for b in [adj_bay, adj_bay + 1])
                        <= CI[stage, node_id]
                    )
                # Ensure CI[stage] is bounded by CI_target
                mdl.add_constraint(
                    CI_target[stage, node_id] ==
                    CI_target_parameter * 2 / B * mdl.sum(demand[stage, node_id][k, j]
                                                          for (i, j) in all_port_moves[stage] for k in range(K))
                )
                mdl.add_constraint(
                    spread_moves_bay[stage, node_id] ==
                    2 / B * mdl.sum(demand[stage, node_id][k, j]
                                    for (i, j) in all_port_moves[stage] for k in range(K))
                )
                # mdl.add_constraint(CI[stage, node_id] <= CI_target[stage, node_id])
                mdl.add_constraint(CI[stage, node_id] - CI_target[stage, node_id] <= CM[stage, node_id])

        # Add mip start
        if mip_start:
            mdl.add_mip_start({x[key]: value for key, value in mip_start.items()}, write_level=1)

    # Build the scenario tree
    build_tree(stages, demand, warm_start)

    # Define the objective function
    # Reshape revenues to match the shape of x
    revenues_ = np.zeros((stages, K, P, ))
    for stage in range(stages):
        for pod in range(stage + 1, P):
            for cargo_class in range(K):
                t = cargo_class + transport_indices.index((stage, pod)) * K
                revenues_[stage, cargo_class, pod,] = revenues[t]

    # Objective: Maximize the total expected revenue over all scenarios
    probabilities = {}
    for (stage, node_id) in node_list:
        probabilities[stage, node_id] = 1 / num_nodes_per_stage[stage]

    objective = mdl.sum(
        probabilities[stage, node_id] * (
            mdl.sum(
                revenues_[stage, k, j] * x[stage, node_id, b, d, k, stage, j]
                for j in range(stage + 1, P) # Loop over discharge ports
                for b in range(B)  # Loop over bays
                for d in range(D)  # Loop over decks
                for k in range(K)  # Loop over cargo classes
            )
            - mdl.sum(env.ho_costs * HO[stage, node_id, b] for b in range(B))
            - env.lc_costs * CM[stage, node_id]
        )
        for stage in range(stages)  # Iterate over all stages
        for node_id in range(num_nodes_per_stage[stage])  # Iterate over nodes at each stage
    )
    mdl.maximize(objective)
    mdl.context.cplex_parameters.read.datacheck = 2
    mdl.parameters.mip.strategy.file = 3
    mdl.parameters.emphasis.memory = 1  # Prioritize memory savings over speed
    mdl.parameters.threads = 1  # Use only 1 thread to reduce memory usage
    mdl.parameters.mip.tolerances.mipgap = 0.001  # 0.1%

    # Solve the model
    solution = mdl.solve(log_output=True)

    # Print the solution
    if solution:
        print(f"Objective value: {solution.objective_value}")

        # Analyze the solution
        x_ = np.zeros((stages, max_paths, B, D, K, P, P))
        HO_ = np.zeros((stages, max_paths, B,))
        HM_ = np.zeros((stages, max_paths, B,))
        CI_ = np.zeros((stages, max_paths,))
        CI_target_ = np.zeros((stages, max_paths,))
        spread_moves_bay_ = np.zeros((stages, max_paths,))
        LM_ = np.zeros((stages, max_paths,))
        VM_ = np.zeros((stages, max_paths,))
        TW_ = np.zeros((stages, max_paths,))
        demand_ = np.zeros((stages, max_paths, K, P))
        revenue_ = np.zeros((stages,max_paths))
        cost_ = np.zeros((stages,max_paths))
        for stage in range(stages):
            for node_id in range(num_nodes_per_stage[stage]):
                for bay in range(B):
                    for deck in range(D):
                        for cargo_class in range(K):
                            for pol in range(stage + 1):
                                for pod in range(pol + 1, P):
                                    x_[stage, node_id, bay, deck, cargo_class, pol, pod] = x[stage, node_id, bay, deck, cargo_class, pol, pod].solution_value
                                    revenue_[stage, node_id,] += revenues_[stage, cargo_class, pod] * x[stage, node_id, bay, deck, cargo_class, pol, pod].solution_value
                                    demand_[stage, node_id, cargo_class, pod] = demand[stage, node_id][cargo_class, pod]

                    HO_[stage, node_id, bay] = HO[stage, node_id, bay].solution_value
                    HM_[stage, node_id, bay] = HM[stage, node_id, bay].solution_value
                    cost_[stage,node_id,] += env.ho_costs * HO[stage, node_id, bay].solution_value

                CI_[stage, node_id] = CI[stage, node_id].solution_value
                CI_target_[stage, node_id] = CI_target[stage, node_id].solution_value
                spread_moves_bay_[stage, node_id] = spread_moves_bay[stage, node_id].solution_value

                LM_[stage, node_id] = LM[stage, node_id].solution_value
                VM_[stage, node_id] = VM[stage, node_id].solution_value
                TW_[stage, node_id] = TW[stage, node_id].solution_value

        # Get metrics from the solution
        num_nodes_per_stage = np.array(num_nodes_per_stage)
        mean_load_per_port = np.sum(x_, axis=(1, 2, 3, 4, 5, 6)) / num_nodes_per_stage # Shape (stages,)
        mean_load_per_location = np.sum(x_, axis=(1, 4, 5, 6)) / num_nodes_per_stage.reshape(-1, 1, 1) # Shape (stages, B, D)
        mean_hatch_overstowage = np.sum(HO_, axis=(1, 2)) / num_nodes_per_stage # Shape (stages,)
        mean_ci = np.sum(CI_, axis=1) / num_nodes_per_stage # Shape (stages,)
        # Auxiliary metrics
        mean_demand = np.sum(demand_, axis=(1, 2, 3)) / num_nodes_per_stage # Shape (stages,)
        mean_revenue = np.sum(revenue_, axis=1) / num_nodes_per_stage # Shape (stages,)
        mean_cost = np.sum(cost_, axis=1) / num_nodes_per_stage # Shape (stages,)

        results = {
            # Input parameters
            "seed":seed,
            "ports":P,
            "scenarios":scenarios_per_stage,
            # Solver results
            "obj":solution.objective_value,
            "time":mdl.solve_details.time,
            "gap":mdl.solve_details.mip_relative_gap,
            # Solution metrics
            "mean_load_per_port":mean_load_per_port.tolist(),
            "mean_load_per_location":mean_load_per_location.tolist(),
            "mean_hatch_overstowage":mean_hatch_overstowage.tolist(),
            "mean_ci":mean_ci.tolist(),
            "mean_demand":mean_demand.tolist(),
            "mean_revenue":mean_revenue.tolist(),
            "mean_cost":mean_cost.tolist(),
        }
        vars = {
            "seed": seed,
            "ports": P,
            "scenarios": scenarios_per_stage,
            "x": x_.tolist(),
            "HO_": HO_.tolist(),
            "HM_": HM_.tolist(),
            "CI_": CI_.tolist(),
            "CI_target_": CI_target_.tolist(),
            "spread_moves_bay_": spread_moves_bay_.tolist(),
            "LM_": LM_.tolist(),
            "VM_": VM_.tolist(),
            "TW_": TW_.tolist(),
        }

        return results, vars
    else:
        # Print the error
        print("No solution found")

if __name__ == "__main__":
    # Load the configuration file
    with open(f'{path_to_main}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    # Run main for different seeds and number of scenarios
    perfect_information = False
    deterministic = False
    debug = False
    generalization = config.env.generalization
    num_episodes = config.testing.num_episodes

    if not deterministic:
        num_scenarios = [4,8,12,16,20,24,28] if not generalization else [28]
    else:
        num_scenarios = [1]

    # Precompute largest scenario tree
    stages = config.env.ports - 1  # Number of load ports (P-1)
    max_scenarios_per_stage = max(num_scenarios) if max(num_scenarios) >= 28 else 28
    # Number of scenarios per stage
    max_paths = max_scenarios_per_stage ** (stages-1) + 1
    node_list = precompute_node_list(stages, max_scenarios_per_stage)

    for x in range(num_episodes):  # Iterate over episodes
        # Create the environment on cpu
        seed = config.env.seed + x + 1
        config.env.seed = seed
        set_unique_seed(seed)
        env = make_env(config.env, batch_size=[max_paths], device='cpu')
        # Precompute for each episode
        demand_tree, real_demand = precompute_demand(node_list, max_paths, stages, env)

        for scen in num_scenarios:  # Iterate over scenarios
            # Filter sub-tree for the number of scenarios
            demand_sub_tree = get_scenario_tree_indices(demand_tree, scen)

            # Run the main logic and get results and variables
            result, var = main(env, demand_sub_tree, scen, stages, max_paths, seed, perfect_information, deterministic)

            # Save results for this episode and scenario
            if debug:
                # Save debug results
                with open(f"./test_results/scenario_tree/results_scenario_tree_e{x}_s{scen}_debug.json","w") as json_file:
                    json.dump(result, json_file, indent=4)
            else:
                # Save regular results
                with open(f"./test_results/scenario_tree/results_scenario_tree_e{x}_s{scen}_pi{perfect_information}"
                        f"_gen{generalization}.json", "w") as json_file:
                    json.dump(result, json_file, indent=4)