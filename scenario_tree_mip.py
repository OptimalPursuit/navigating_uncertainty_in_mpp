# Imports
import numpy as np
import torch as th
import torch.nn as nn
import random
import yaml
from dotmap import DotMap
from docplex.mp.model import Model
import sys
import os
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any

path = 'add path to cplex here'
sys.path.append(path)

# Module imports
path_to_main = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(path_to_main)
from main import *
from environment.utils import get_pol_pod_pair
from rl_algorithms.utils import set_unique_seed


# Precompute functions
def precompute_node_list(stages:int, scenarios_per_stage:int, deterministic=False) -> List:
    """Precompute the list of nodes and their coordinates in the scenario tree"""
    node_list = []  # List to store the coordinates of all nodes
    # Loop over each stage, starting from stage 1 (root is stage 1)
    for stage in range(stages):
        # Number of nodes at this stage
        nodes_in_current_stage = scenarios_per_stage ** (stage) if not deterministic else 1

        # For each node in the current stage
        for node_id in range(nodes_in_current_stage):
            node_list.append((stage, node_id))

    return node_list

def precompute_demand(node_list:List, max_paths:int, stages:int, env:nn.Module) -> Tuple[Dict, Dict]:
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

def get_scenario_tree_indices(scenario_tree:Dict, num_scenarios:int) -> Dict:
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
        # For stage > 0, increase max_nodes by 1 for realization in rolling horizon
        if stage > 0 and stochastic_algorithm == 'rolling_horizon':
            max_nodes += 1
        # Include only nodes within the allowed range
        if node < max_nodes:
            filtered_tree[(stage, node)] = value
    return filtered_tree

# Support functions
def get_demand_history(stage:int, demand:np.array, num_nodes_per_stage:List) -> np.array:
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
def main(env:nn.Module, demand:np.array, scenarios_per_stage:int=28, stages:int=3, max_paths:int=784, seed:int=42,
         perfect_information:bool=False, deterministic:bool=False, algorithm:str='multi_stage', warm_solution:bool=False,) -> Tuple[Dict, Dict]:
    # Scenario tree parameters
    M = 10 ** 3 # Big M
    num_nodes_per_stage = [1*scenarios_per_stage**stage for stage in range(stages)]

    # Problem parameters
    P = env.P
    B = env.B
    D = env.D
    K = env.K
    T = env.T
    BL = env.BL if hasattr(env, 'BL') else 1
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
    PD = {} # Binary POD
    mixing = {} # Mixing
    HO = {} # Hatch overstowage
    HM = {} # Hatch move
    CI = {} # Crane intensity
    CI_target = {} # Crane intensity target
    CM = {} # Crane move
    LM = {} # Longitudinal moment
    VM = {} # Vertical moment
    TW = {} # Total weight

    # Sets of different stages
    on_boards = []
    all_port_moves = []
    all_load_moves = []
    transport_indices = [(i, j) for i in range(P) for j in range(P) if i < j]

    def _get_warm_start_dict(x:Optional[Dict]) -> Dict:
        """Function to get the warm start dictionary"""
        warm_start_dict = {}

        # f"x_{stage}_{node_id}_{bay}_{deck}_{cargo_class}_{pol}_{pod}"

        for stage in range(stages):
            for node_id in range(num_nodes_per_stage[stage]):
                for bay in range(B):
                    for deck in range(D):
                        for cargo_class in range(K):
                            for pol in range(stage + 1):
                                for pod in range(pol + 1, P):
                                    warm_start_dict[f'x_{stage}_{node_id}_{bay}_{deck}_{cargo_class}_{pol}_{pod}'] = x[
                                        stage, node_id, bay, deck, cargo_class, pol, pod].solution_value
        print(f'Warm start dict: {warm_start_dict}')
        breakpoint() # todo: remove later
        return warm_start_dict

    def build_tree_mpp(stages:int, demand:np.array, warm_solution:Optional[Dict]=None) -> None:
        """Function to build the scenario tree; with decisions and constraints for each node"""
        for stage in range(stages):
            for node_id in range(num_nodes_per_stage[stage]):
                # Crane intensity:
                CI[stage, node_id] = mdl.continuous_var(name=f'CI_{stage}_{node_id}', lb=0)
                CI_target[stage, node_id] = mdl.continuous_var(name=f'CI_target_{stage}_{node_id}', lb=0)
                CM[stage, node_id] = mdl.continuous_var(name=f'CM_{stage}_{node_id}', lb=0)

                # Stability:
                LM[stage, node_id] = mdl.continuous_var(name=f'LM_{stage}_{node_id}')
                VM[stage, node_id] = mdl.continuous_var(name=f'VM_{stage}_{node_id}')
                TW[stage, node_id] = mdl.continuous_var(name=f'TW_{stage}_{node_id}')

                for bay in range(B):
                    # Hatch overstowage:
                    HO[stage, node_id, bay] = mdl.continuous_var(name=f'HO_{stage}_{node_id}_{bay}', lb=0)
                    HM[stage, node_id, bay] = mdl.binary_var(name=f'HM_{stage}_{node_id}_{bay}')

                    for deck in range(D):
                        for cargo_class in range(K):
                            for pol in range(stage + 1):
                                for pod in range(pol + 1, P):
                                    # Cargo allocation:
                                    x[stage, node_id, bay, deck, cargo_class, pol, pod] = \
                                        mdl.continuous_var(name=f'x_{stage}_{node_id}_{bay}_{deck}_{cargo_class}_{pol}_{pod}', lb=0)

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
                            mdl.sum(x[stage, node_id, b, 0, k, i, j] for (i, j) in on_boards[stage - 1]
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
                # mdl.add_constraint(CI[stage, node_id] <= CI_target[stage, node_id])
                mdl.add_constraint(CI[stage, node_id] - CI_target[stage, node_id] <= CM[stage, node_id])

        # Add mip start
        if warm_solution:
            # todo: add warm solution
            start_dict = _get_warm_start_dict(warm_solution)
            mdl.add_mip_start({x[key]: value for key, value in start_dict.items()}, write_level=1)

    def initialize_vars_tree_block_mpp(stages:int, start_stage:int=0) -> None:
        # Build variables for full scenario tree
        print("Initializing variables for full scenario tree")
        for stage in range(start_stage + stages):
            for node_id in range(num_nodes_per_stage[stage] + 1):
                # Crane intensity:
                CI[stage, node_id] = mdl.continuous_var(name=f'CI_{stage}_{node_id}', lb=0)
                CI_target[stage, node_id] = mdl.continuous_var(name=f'CI_target_{stage}_{node_id}', lb=0)
                CM[stage, node_id] = mdl.continuous_var(name=f'CM_{stage}_{node_id}', lb=0)

                # Stability:
                LM[stage, node_id] = mdl.continuous_var(name=f'LM_{stage}_{node_id}')
                VM[stage, node_id] = mdl.continuous_var(name=f'VM_{stage}_{node_id}')
                TW[stage, node_id] = mdl.continuous_var(name=f'TW_{stage}_{node_id}')

                for bay in range(B):
                    for block in range(BL):
                        # Hatch overstowage:
                        HO[stage, node_id, bay, block] = mdl.continuous_var(name=f'HO_{stage}_{node_id}_{bay}_{block}',
                                                                            lb=0)
                        HM[stage, node_id, bay, block] = mdl.binary_var(name=f'HM_{stage}_{node_id}_{bay}_{block}')
                        for deck in range(D):
                            for cargo_class in range(K):
                                for pol in range(stage + 1):
                                    for pod in range(pol + 1, P):
                                        # Cargo allocation:
                                        x[stage, node_id, bay, deck, block, cargo_class, pol, pod] = \
                                            mdl.continuous_var(name=f'x_{stage}_{node_id}_{bay}_{deck}_{block}'
                                                                    f'_{cargo_class}_{pol}_{pod}', lb=0)

                        for pod in range(P):
                            # PODs in locations
                            PD[stage, node_id, bay, block, pod] = \
                                mdl.binary_var(name=f'PD_{stage}_{node_id}_{bay}_{block}_{pod}')
                        # Excess PODs
                        mixing[stage, node_id, bay, block] = \
                            mdl.binary_var(name=f'mixing_{stage}_{node_id}_{bay}_{block}')

    def build_tree_block_mpp(stages:int, demand:np.array, warm_solution:Optional[Dict]=None, start_stage:int=0) -> None:
        """Function to build the scenario tree; with decisions and constraints for each node"""
        print("Building constraints for scenario tree")
        for stage in range(start_stage, start_stage + stages):
            # Define sets at current stage/port
            on_board, port_moves, load_moves = onboard_groups(P, stage, transport_indices)
            on_boards.append(on_board)
            all_port_moves.append(port_moves)
            all_load_moves.append(load_moves)

            if stochastic_algorithm == 'rolling_horizon' and stage == start_stage and stage < stages - 1:
                realized_demand = demand[stage + 1, 0] # Store realized demand from t+1
                demand[stage + 1, 0] = demand[stage + 1, scen] # Overwrite root node with simulated scenario in t+1

                # If t > 0, update demand at t with realized demand
                if stage > 0:
                    demand[stage, 0] = realized_demand

            # Scenario range
            scenarios = num_nodes_per_stage[:2][stage-start_stage] if stochastic_algorithm in ['rolling_horizon', 'myopic'] \
                else num_nodes_per_stage[stage]
            for node_id in range(scenarios):
                print(f"Stage {stage}, Node {node_id}")
                for (i, j) in load_moves:
                    for k in range(K):
                        # Demand satisfaction
                        mdl.add_constraint(
                            mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for b in range(B) for d in range(D) for bl in range(BL))
                            <= demand[stage, node_id][k, j], ctname=f'demand_{stage}_{node_id}_{k}_{j}'
                        )

                for b in range(B):
                    for bl in range(BL):
                        for d in range(D):
                            # TEU capacity
                            mdl.add_constraint(
                                mdl.sum(teus[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j]
                                                          for (i, j) in on_board) for k in range(K))
                                <= capacity[b, d, bl], ctname=f'capacity_{stage}_{node_id}_{b}_{d}_{bl}'
                            )

                            if not perfect_information and stochastic_algorithm not in ['rolling_horizon', 'myopic']:
                                demand_history1 = get_demand_history(stage, demand, num_nodes_per_stage)
                                for node_id2 in range(node_id + 1, num_nodes_per_stage[stage]):
                                    demand_history2 = get_demand_history(stage, demand, num_nodes_per_stage)
                                    if np.allclose(demand_history1, demand_history2, atol=1e-5):  # Use a tolerance if floats
                                        for k in range(K):
                                            for (i, j) in load_moves:
                                                # Non-anticipation at stage, provided demand history is similar
                                                mdl.add_constraint(
                                                    x[stage, node_id, b, d, bl, k, i, j] == x[stage, node_id2, b, d, bl, k, i, j],
                                                    ctname=f'non_anticipation_{stage}_{node_id}_{node_id2}_{b}_{d}_{bl}_{k}_{i}_{j}'
                                                )

                        # Open hatch (d=1 is below deck)
                        mdl.add_constraint(
                            mdl.sum(x[stage, node_id, b, 1, bl, k, i, j] for (i, j) in all_port_moves[stage] for k in range(K))
                            <= M * HM[stage, node_id, b, bl], ctname=f'hatch_move_{stage}_{node_id}_{b}_{bl}'
                        )

                        # Hatch overstows (d=0 is on deck)
                        # Overstowage is arrival condition of previous port: on_boards[stage - 1]
                        # Vessel is empty before stage 0, hence no overstows
                        if stage > 0:
                            mdl.add_constraint(
                                mdl.sum(x[stage, node_id, b, 0, bl, k, i, j] for (i, j) in on_boards[stage - 1]
                                        for k in range(K) if j > stage) - M * (1 - HM[stage, node_id, b, bl] )
                                <= HO[stage, node_id, b, bl], ctname=f'hatch_overstow_{stage}_{node_id}_{b}_{bl}'
                            )

                # Stability
                mdl.add_constraint(
                    TW[stage, node_id] == mdl.sum(weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j]
                                                                       for (i, j) in on_board for d in range(D))
                                                  for k in range(K) for b in range(B) for bl in range(BL)),
                    ctname=f'total_weight_{stage}_{node_id}'
                )

                # LCG
                mdl.add_constraint(
                    LM[stage, node_id] == mdl.sum(longitudinal_position[b] * mdl.sum(
                        weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for d in range(D) for bl in range(BL))
                        for k in range(K)) for b in range(B)), ctname=f'longitudinal_moment_{stage}_{node_id}')
                mdl.add_constraint(
                    stab_delta * mdl.sum(weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for d in range(D) for bl in range(BL))
                                         for k in range(K) for b in range(B)) >= mdl.sum(longitudinal_position[b] * mdl.sum(
                        weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for d in range(D) for bl in range(BL)) for k in
                        range(K)) for b in range(B)) - LCG_target * mdl.sum(
                        weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for d in range(D) for bl in range(BL)) for k in
                        range(K) for b in range(B)), ctname=f'lcg_ub_{stage}_{node_id}')
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for d in range(D) for bl in range(BL)) for k in
                    range(K) for b in range(B)) >= - mdl.sum(longitudinal_position[b] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for d in range(D) for bl in range(BL)) for k in
                    range(K)) for b in range(B)) + LCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for d in range(D) for bl in range(BL)) for k in
                    range(K) for b in range(B)), ctname=f'lcg_lb_{stage}_{node_id}')

                # VCG
                mdl.add_constraint(VM[stage, node_id] == mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for b in range(B) for bl in range(BL))
                    for k in range(K)) for d in range(D)), ctname=f'vertical_moment_{stage}_{node_id}')
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for b in range(B) for bl in range(BL)) for k in
                    range(K) for d in range(D)) >= mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for b in range(B) for bl in range(BL)) for k in
                    range(K)) for d in range(D)) - VCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for b in range(B) for bl in range(BL)) for k in
                    range(K) for d in range(D)), ctname=f'vcg_ub_{stage}_{node_id}')
                mdl.add_constraint(stab_delta * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for b in range(B) for bl in range(BL)) for k in
                    range(K) for d in range(D)) >= - mdl.sum(vertical_position[d] * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for b in range(B) for bl in range(BL)) for k in
                    range(K)) for d in range(D))  + VCG_target * mdl.sum(
                    weights[k] * mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in on_board for b in range(B) for bl in range(BL)) for k in
                    range(K) for d in range(D)), ctname=f'vcg_lb_{stage}_{node_id}')

                # Compute lower bound on long crane
                for adj_bay in range(B - 1):
                    mdl.add_constraint(
                        mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in all_port_moves[stage]
                                for k in range(K) for bl in range(BL) for d in range(D) for b in [adj_bay, adj_bay + 1])
                        <= CI[stage, node_id], ctname=f'crane_intensity_{stage}_{node_id}_{adj_bay}'
                    )
                # Ensure CI[stage] is bounded by CI_target
                mdl.add_constraint(
                    CI_target[stage, node_id] ==
                    CI_target_parameter * 2 / B * mdl.sum(demand[stage, node_id][k, j]
                                                          for (i, j) in all_port_moves[stage] for k in range(K)),
                    ctname=f'crane_intensity_target_{stage}_{node_id}'
                )
                # mdl.add_constraint(CI[stage, node_id] <= CI_target[stage, node_id])
                mdl.add_constraint(CI[stage, node_id] - CI_target[stage, node_id] <= CM[stage, node_id],
                                   ctname=f'crane_move_{stage}_{node_id}')

                # BLock stowage patterns: Binary j in each b,d
                for b in range(B):
                    for bl in range(BL):
                        for d in range(D):
                            for j in range(stage+1, P):
                                # Get POD indicator
                                mdl.add_constraint(
                                    mdl.sum(x[stage, node_id, b, d, bl, k, stage, j] for k in range(K)) <= M * PD[stage, node_id, b, bl, j],
                                    ctname=f'pod_indicator_{stage}_{node_id}_{b}_{d}_{bl}_{j}'
                                )
                        # Max PODs per bay/block
                        mdl.add_constraint(
                            mdl.sum(PD[stage, node_id, b, bl, j] for j in range(P)) <= 1, ctname=f'max_pod_{stage}_{node_id}_{b}_{bl}'
                        )
                #         # Get mixing bay
                #         mdl.add_constraint(
                #             mdl.sum(PD[stage, node_id, b, bl, j] for j in range(P)) <= 1 + M*mixing[stage, node_id, b, bl]
                #         )
                #
                # # Maximum number of mixing bays
                # mdl.add_constraint(
                #     mdl.sum(mixing[stage, node_id, b, bl,] for b in range(B) for bl in range(BL)) <= int(BL * B * bay_mix)
                # )

        # Add mip start
        if warm_solution:
            # todo: add warm solution
            start_dict = _get_warm_start_dict(warm_solution)
            mdl.add_mip_start({x[key]: value for key, value in start_dict.items()}, write_level=1)

    def build_rolling_block_mpp(stages: int,
                                demand: np.array,
                                warm_solution: Optional[Dict] = None) -> Dict:
        """
        Rolling horizon wrapper around build_tree_block_mpp.
        At each stage t:
            - Build a 2-stage problem (current + next port scenarios)
            - Solve and apply stage-0 decisions
            - Move forward
        """
        rolling_horizon_x = {}
        rolling_horizon_HO = {}
        rolling_horizon_CM = {}

        for t in range(stages):
            print(f"\n--- Rolling horizon solve at stage {t} ---")
            # Re-initialize variables for full tree
            mdl.clear()
            initialize_vars_tree_block_mpp(stages=stages)
            # Myopic + only stages=1 (see else)
            # initialize_vars_tree_block_mpp(stages=1, start_stage=t)

            # Freeze previous solution x_{t-1} to current solution x_t
            if t > 0:
                prev_stage = t - 1
                for b in range(B):
                    for bl in range(BL):
                        for d in range(D):
                            for k in range(K):
                                # only fix x_{t-1} to x_t
                                for i in range(prev_stage, prev_stage + 1):
                                    for j in range(i + 1, P):
                                        x[t, 0, b, d, bl, k, i, j].set_lb(rolling_horizon_x[prev_stage, 0, b, d, bl, k, i, j])
                                        x[t, 0, b, d, bl, k, i, j].set_ub(rolling_horizon_x[prev_stage, 0, b, d, bl, k, i, j])

            if t < stages - 1:
                # Rolling horizon → 2-stage problem (current + next port scenarios)
                two_stage_demand = {(t, 0): demand[(t, 0)]}
                # Get S+1 scenarios (stochastic nodes + realization node)
                for s in range(scenarios_per_stage + 1):
                    two_stage_demand[(t + 1, s)] = demand[(t + 1, s)]

                build_tree_block_mpp(stages=2,
                                     demand=two_stage_demand,
                                     warm_solution=warm_solution,
                                     start_stage=t)
            else:
                # Last port → deterministic one-stage
                last_stage_demand = {(t, 0): demand[(t, 0)]}
                build_tree_block_mpp(stages=1,
                                     demand=last_stage_demand,
                                     warm_solution=warm_solution,
                                     start_stage=t)
            objective = objective_function(x, HO, CM, revenues_, start_stage=t)
            sol = solve_model(mdl, objective)

            if sol is None:
                raise RuntimeError(f"No solution found at stage {t}")

            # Extract current stage decisions
            rolling_horizon_CM[t, 0] = CM[(t, 0)].solution_value
            for b in range(B):
                for bl in range(BL):
                    rolling_horizon_HO[t, 0, b, bl] = HO[(t, 0, b, bl)].solution_value
                    for d in range(D):
                        for k in range(K):
                            for j in range(t + 1, P):
                                rolling_horizon_x[t, 0, b, d, bl, k, t, j] = x[(t, 0, b, d, bl, k, t, j)].solution_value

        # Compute objective over all current stage decisions
        objective = objective_function(rolling_horizon_x, rolling_horizon_HO, rolling_horizon_CM, revenues_,)

        total_allocated = np.zeros((K, stages, P))
        for (stage, node_id, b, d, bl, k, i, j), value in rolling_horizon_x.items():
            total_allocated[k, stage, j] += value
        # print(f"Objective value: {objective}")
        # print("Total allocated containers per stage:")
        # print(total_allocated.sum(axis=(0,2)))
        # print(total_allocated.sum(axis=(0,)))
        # breakpoint() # todo: remove later


        return {"x": rolling_horizon_x, "HO": rolling_horizon_HO, "CM": rolling_horizon_CM, "objective_value": objective}

    def build_myopic_block_mpp(stages: int,
                                demand: np.array,
                                warm_solution: Optional[Dict] = None) -> Dict:
        """
        Rolling horizon wrapper around build_tree_block_mpp.
        At each stage t:
            - Build a 2-stage problem (current + next port scenarios)
            - Solve and apply stage-0 decisions
            - Move forward
        """
        rolling_horizon_x = {}
        rolling_horizon_HO = {}
        rolling_horizon_CM = {}

        for t in range(stages):
            print(f"\n--- Rolling horizon solve at stage {t} ---")
            # Re-initialize variables for full tree
            mdl.clear()
            initialize_vars_tree_block_mpp(stages=1, start_stage=t)

            # Freeze previous solution x_{t-1} to current solution x_t
            if t > 0:
                prev_stage = t - 1
                for b in range(B):
                    for bl in range(BL):
                        for d in range(D):
                            for k in range(K):
                                # only fix x_{t-1} to x_t
                                for i in range(prev_stage, prev_stage + 1):
                                    for j in range(i + 1, P):
                                        x[t, 0, b, d, bl, k, i, j].set_lb(rolling_horizon_x[prev_stage, 0, b, d, bl, k, i, j])
                                        x[t, 0, b, d, bl, k, i, j].set_ub(rolling_horizon_x[prev_stage, 0, b, d, bl, k, i, j])


            # Last port → deterministic one-stage
            last_stage_demand = {(t, 0): demand[(t, 0)]}
            build_tree_block_mpp(stages=1,
                                 demand=last_stage_demand,
                                 warm_solution=warm_solution,
                                 start_stage=t)
            objective = objective_function(x, HO, CM, revenues_, start_stage=t)
            sol = solve_model(mdl, objective)

            if sol is None:
                raise RuntimeError(f"No solution found at stage {t}")

            # Extract current stage decisions
            rolling_horizon_CM[t, 0] = CM[(t, 0)].solution_value
            for b in range(B):
                for bl in range(BL):
                    rolling_horizon_HO[t, 0, b, bl] = HO[(t, 0, b, bl)].solution_value
                    for d in range(D):
                        for k in range(K):
                            for j in range(t + 1, P):
                                rolling_horizon_x[t, 0, b, d, bl, k, t, j] = x[(t, 0, b, d, bl, k, t, j)].solution_value

        # Compute objective over all current stage decisions
        objective = objective_function(rolling_horizon_x, rolling_horizon_HO, rolling_horizon_CM, revenues_,)

        total_allocated = np.zeros((K, stages, P))
        for (stage, node_id, b, d, bl, k, i, j), value in rolling_horizon_x.items():
            total_allocated[k, stage, j] += value
        # print("Total allocated containers per stage:")
        # print(total_allocated.sum(axis=(0,2)))
        # print(total_allocated.sum(axis=(0,)))
        # print(objective)
        # breakpoint() # todo: remove later


        return {"x": rolling_horizon_x, "HO": rolling_horizon_HO, "CM": rolling_horizon_CM, "objective_value": objective}

    def objective_function(x:Any, HO:Any, CM:Any, revenues:Any, start_stage:int=None) -> Any:
        """Define the objective function"""
        # todo: simplify this function

        # Objective: Maximize the total expected revenue over all scenarios
        probabilities = {}
        if stochastic_algorithm == "multi_stage":
            for (stage, node_id) in node_list:
                probabilities[stage, node_id] = 1 / num_nodes_per_stage[stage]

            if config.env.env_name == "mpp":
                objective = mdl.sum(
                    probabilities[stage, node_id] * (
                            mdl.sum(
                                revenues[stage, k, j] * x[stage, node_id, b, d, k, stage, j]
                                for j in range(stage + 1, P)  # Loop over discharge ports
                                for b in range(B)  # Loop over bays
                                for d in range(D)  # Loop over decks
                                for k in range(K)  # Loop over cargo classes
                            )
                            - mdl.sum(env.ho_costs * HO[stage, node_id, b] for b in range(B))
                            - env.cm_costs * CM[stage, node_id]
                    )
                    for stage in range(stages)  # Iterate over all stages
                    for node_id in range(num_nodes_per_stage[stage])  # Iterate over nodes at each stage
                )
            elif config.env.env_name == "block_mpp":
                objective = mdl.sum(
                    probabilities[stage, node_id] * (
                            mdl.sum(
                                revenues[stage, k, j] * x[stage, node_id, b, d, bl, k, stage, j]
                                for j in range(stage + 1, P)  # Loop over discharge ports
                                for b in range(B)  # Loop over bays
                                for d in range(D)  # Loop over decks
                                for k in range(K)  # Loop over cargo classes
                                for bl in range(BL)  # Loop over blocks
                            )
                            - mdl.sum(env.ho_costs * HO[stage, node_id, b, bl] for b in range(B) for bl in range(BL))
                            - env.cm_costs * CM[stage, node_id]
                    )
                    for stage in range(stages)  # Iterate over all stages
                    for node_id in range(num_nodes_per_stage[stage])  # Iterate over nodes at each stage
                )
            else:
                raise ValueError("Invalid environment name")

        elif stochastic_algorithm == "rolling_horizon":
            if start_stage is None:
                objective = mdl.sum(
                    (
                            mdl.sum(
                                revenues[stage, k, j] * x[stage, 0, b, d, bl, k, stage, j]
                                for j in range(stage + 1, P)  # Loop over discharge ports
                                for b in range(B)  # Loop over bays
                                for d in range(D)  # Loop over decks
                                for k in range(K)  # Loop over cargo classes
                                for bl in range(BL)  # Loop over blocks
                            )
                            - mdl.sum(env.ho_costs * HO[stage, 0, b, bl] for b in range(B) for bl in range(BL))
                            - env.cm_costs * CM[stage, 0]
                    )
                    for stage in range(stages)  # Iterate over all stages
                    )

            elif start_stage is not None:
                # Probabilities for two stages
                scenarios_per_stage = num_nodes_per_stage[:2]
                for stage in range(start_stage, start_stage+2):
                    for node_id in range(scenarios_per_stage[stage-start_stage]):
                        probabilities[stage, node_id] = 1 / scenarios_per_stage[stage-start_stage]

                # Objective
                if start_stage < stages - 1:
                    objective = mdl.sum(
                        probabilities[stage, node_id] * (
                                mdl.sum(
                                    revenues_[stage, k, j] * x[stage, node_id, b, d, bl, k, stage, j]
                                    for j in range(stage + 1, P)  # Loop over discharge ports
                                    for b in range(B)  # Loop over bays
                                    for d in range(D)  # Loop over decks
                                    for k in range(K)  # Loop over cargo classes
                                    for bl in range(BL)  # Loop over blocks
                                )
                                - mdl.sum(env.ho_costs * HO[stage, node_id, b, bl] for b in range(B) for bl in range(BL))
                                - env.cm_costs * CM[stage, node_id]
                        )
                        for stage in range(start_stage, start_stage+2)  # Iterate over all stages
                        for node_id in range(scenarios_per_stage[stage-start_stage])  # Iterate over nodes at each stage
                    )
                else:
                    objective = mdl.sum(
                        probabilities[start_stage, node_id] * (
                                mdl.sum(
                                    revenues_[start_stage, k, j] * x[start_stage, node_id, b, d, bl, k, start_stage, j]
                                    for j in range(start_stage + 1, P)  # Loop over discharge ports
                                    for b in range(B)  # Loop over bays
                                    for d in range(D)  # Loop over decks
                                    for k in range(K)  # Loop over cargo classes
                                    for bl in range(BL)  # Loop over blocks
                                )
                                - mdl.sum(env.ho_costs * HO[start_stage, node_id, b, bl] for b in range(B) for bl in range(BL))
                                - env.cm_costs * CM[start_stage, node_id]
                        )
                        for node_id in range(scenarios_per_stage[0])  # Iterate over nodes at the last stage
                    )
            else:
                raise ValueError("Invalid start_stage value")
        elif stochastic_algorithm == "myopic":
            if start_stage is None:
                objective = mdl.sum(
                    (
                            mdl.sum(
                                revenues[stage, k, j] * x[stage, 0, b, d, bl, k, stage, j]
                                for j in range(stage + 1, P)  # Loop over discharge ports
                                for b in range(B)  # Loop over bays
                                for d in range(D)  # Loop over decks
                                for k in range(K)  # Loop over cargo classes
                                for bl in range(BL)  # Loop over blocks
                            )
                            - mdl.sum(env.ho_costs * HO[stage, 0, b, bl] for b in range(B) for bl in range(BL))
                            - env.cm_costs * CM[stage, 0]
                    )
                    for stage in range(stages)  # Iterate over all stages
                )
            elif start_stage is not None:
                objective = mdl.sum(
                    (
                            mdl.sum(
                                revenues[stage, k, j] * x[stage, 0, b, d, bl, k, stage, j]
                                for j in range(stage + 1, P)  # Loop over discharge ports
                                for b in range(B)  # Loop over bays
                                for d in range(D)  # Loop over decks
                                for k in range(K)  # Loop over cargo classes
                                for bl in range(BL)  # Loop over blocks
                            )
                            - mdl.sum(env.ho_costs * HO[stage, 0, b, bl] for b in range(B) for bl in range(BL))
                            - env.cm_costs * CM[stage, 0]
                    )
                    for stage in range(start_stage,start_stage+1)  # Iterate over all stages
                )
            else:
                raise ValueError("Invalid start_stage value")
        else:
            raise ValueError("Invalid stochastic algorithm")

        return objective

    def solve_model(mdl:Model, objective:Any) -> Any:
        mdl.maximize(objective)
        mdl.context.cplex_parameters.read.datacheck = 2
        mdl.parameters.mip.strategy.file = 3
        mdl.parameters.emphasis.memory = 1  # Prioritize memory savings over speed
        mdl.parameters.threads = 1  # Use only 1 thread to reduce memory usage
        mdl.parameters.mip.tolerances.mipgap = 0.001  # 0.1%
        mdl.set_time_limit(3600)  # 1 hour
        solution = mdl.solve(log_output=True)
        return solution

    # Reshape revenues to match the shape of x
    revenues_ = np.zeros((stages, K, P,))
    for stage in range(stages):
        for pod in range(stage + 1, P):
            for cargo_class in range(K):
                t = cargo_class + transport_indices.index((stage, pod)) * K
                revenues_[stage, cargo_class, pod,] = revenues[t]

    # Build the scenario tree
    if config.env.env_name == "mpp":
        build_tree_mpp(stages, demand, warm_solution)
        objective = objective_function(x, HO, CM, revenues_)
        solution = solve_model(mdl, objective)
    elif config.env.env_name == "block_mpp":
        if stochastic_algorithm == "multi_stage":
            initialize_vars_tree_block_mpp(stages)
            build_tree_block_mpp(stages, demand, warm_solution)
            objective = objective_function(x, HO, CM, revenues_)
            solution = solve_model(mdl, objective)
        elif stochastic_algorithm == "rolling_horizon":
            solution = build_rolling_block_mpp(stages, demand, warm_solution)
        elif stochastic_algorithm == "myopic":
            solution = build_myopic_block_mpp(stages, demand, warm_solution)
        else:
            raise ValueError("Invalid stochastic algorithm")
    else:
        raise ValueError("Invalid environment name")

    # Print the solution # todo: fix this properly for mpp and block_mpp; now set to block_mpp
    if solution:
        if "objective_value" in solution:
            obj = solution["objective_value"]
            print(f"Objective value (rolling horizon): {solution['objective_value']}")
        else:
            obj = solution.objective_value
            print(f"Objective value: {solution.objective_value}")

        # Analyze the solution
        x_ = np.zeros((stages, max_paths, B, D, BL, K, P, P))
        ob_demand = np.zeros((stages, max_paths, K, P))
        PD_ = np.zeros((stages, max_paths, B, BL, P,))
        mixing_ = np.zeros((stages, max_paths, B, BL,))
        HO_ = np.zeros((stages, max_paths, B, BL,))
        HM_ = np.zeros((stages, max_paths, B, BL,))
        CI_ = np.zeros((stages, max_paths,))
        CI_target_ = np.zeros((stages, max_paths,))
        CM_ = np.zeros((stages, max_paths,))
        LM_ = np.zeros((stages, max_paths,))
        VM_ = np.zeros((stages, max_paths,))
        TW_ = np.zeros((stages, max_paths,))
        demand_ = np.zeros((stages, max_paths, K, P))
        revenue_ = np.zeros((stages,max_paths))
        cost_ = np.zeros((stages,max_paths))

        # scenarios
        scenarios = num_nodes_per_stage if stochastic_algorithm == 'multi_stage' else [1] * stages
        for stage in range(stages):
            for node_id in range(scenarios[stage]):
                for bay in range(B):
                    for block in range(BL):
                        for deck in range(D):
                            for cargo_class in range(K):
                                for pol in range(stage + 1):
                                    for pod in range(pol + 1, P):
                                        if stochastic_algorithm == "rolling_horizon":
                                            x_[stage, node_id, bay, deck, block, cargo_class, pol, pod] = solution["x"].get((stage, node_id, bay, deck, block, cargo_class, pol, pod), 0)
                                        else:
                                            x_[stage, node_id, bay, deck, block, cargo_class, pol, pod] = x[stage, node_id, bay, deck, block, cargo_class, pol, pod].solution_value
                                        revenue_[stage, node_id,] += revenues_[stage, cargo_class, pod] * x_[stage, node_id, bay, deck, block, cargo_class, pol, pod]
                                        demand_[stage, node_id, cargo_class, pod] = demand[stage, node_id][cargo_class, pod]

                        for pod in range(stage + 1, P):
                            PD_[stage, node_id, bay, block, pod] = PD[stage, node_id, bay, block, pod].solution_value
                            for k in range(K):
                               ob_demand[stage, node_id, k, pod] = demand[stage, node_id][k, pod]
                        mixing_[stage, node_id, bay, block,] = mixing[stage, node_id, bay, block,].solution_value

                        if stochastic_algorithm == "rolling_horizon":
                            HO_[stage, node_id, bay, block,] = solution["HO"].get((stage, node_id, bay, block), 0)
                            HM_[stage, node_id, bay, block,] = 1 if HO_[stage, node_id, bay, block,] > 0 else 0
                        else:
                            HO_[stage, node_id, bay, block,] = HO[stage, node_id, bay, block,].solution_value
                            HM_[stage, node_id, bay, block,] = HM[stage, node_id, bay, block,].solution_value
                        cost_[stage, node_id,] += env.ho_costs * HO_[stage, node_id, bay, block]


                if stochastic_algorithm == "rolling_horizon":
                    CM_[stage, node_id] = solution["CM"].get((stage, node_id), 0)
                    # todo: implement CI, stability etc. in rolling horizon
                else:
                    CM_[stage, node_id] = CM[stage, node_id].solution_value
                    CI_[stage, node_id] = CI[stage, node_id].solution_value
                    CI_target_[stage, node_id] = CI_target[stage, node_id].solution_value
                    LM_[stage, node_id] = LM[stage, node_id].solution_value
                    VM_[stage, node_id] = VM[stage, node_id].solution_value
                    TW_[stage, node_id] = TW[stage, node_id].solution_value
                cost_[stage, node_id,] += env.cm_costs * CM_[stage, node_id]

        # print("Total allocated containers per stage:")
        # num_nodes_per_stage = np.array(num_nodes_per_stage)
        # mean_load_per_port = np.sum(x_, axis=(1, 2, 3, 4, 5, 6, 7)) / num_nodes_per_stage # Shape (stages,)
        # print(mean_load_per_port)
        # mean_load_per_port2 = np.sum(x_, axis=(1, 2, 3, 4, 5, 7)) / num_nodes_per_stage.reshape(-1, 1,) # Shape (stages,)
        # print(mean_load_per_port2)
        # mean_load_per_port3 = np.sum(x_, axis=(1, 2, 3, 4, 5, 6)) / num_nodes_per_stage.reshape(-1, 1,) # Shape (stages,)
        # print(mean_load_per_port3)
        # breakpoint() # todo: remove later

        # Get metrics from the solution
        # todo: some computations here are wrong!
        num_nodes_per_stage = np.array(num_nodes_per_stage)
        mean_load_per_port = np.sum(x_, axis=(1, 2, 3, 4, 5, 6, 7)) / num_nodes_per_stage # Shape (stages,)
        mean_load_per_demand = np.sum(x_, axis=(1, 2, 3, 4, 5)) / num_nodes_per_stage.reshape(-1, 1, 1) # Shape (stages, P, P)
        mean_teu_load_per_port = (x_ * teus.reshape(1, 1, 1, 1, 1, K, 1, 1)).sum(axis=(1,2, 3, 4, 5, 6, 7)) # Shape (stages,)
        mean_load_per_location = np.sum(x_, axis=(1, 4, 5, 6)) / num_nodes_per_stage.reshape(-1, 1, 1, 1) # Shape (stages, B, D, BL)
        mean_hatch_overstowage = np.sum(HO_, axis=(1, 2, 3)) / num_nodes_per_stage # Shape (stages,)
        # mean_ci = np.sum(CI_, axis=1) / num_nodes_per_stage # Shape (stages,)
        mean_pd = np.sum(PD_, axis=(1, 2, 3, 4)) / num_nodes_per_stage # Shape (stages,)
        mean_mixing = np.sum(mixing_, axis=(1, 2, 3)) / num_nodes_per_stage # Shape (stages,)

        # todo: on board demand works well for P=4,5,6; B=10 -> check B=20
        # for pol in range(P):
        #     ob_dem = 0
        #     on_boards = onboard_groups(P, pol, transport_indices)[0]
        #     print(f"on_boards p{pol}:{on_boards}")
        #     for (i,j) in on_boards:
        #         for k in range(K):
        #             ob_dem += demand[pol, 0][k, j]
        #     print(f"ob_demand p{pol}:{ob_dem}")

        # Auxiliary metrics
        mean_demand = np.sum(demand_, axis=(1, 2, 3)) / num_nodes_per_stage # Shape (stages,)
        mean_revenue = np.sum(revenue_, axis=1) / num_nodes_per_stage # Shape (stages,)
        mean_cost = np.sum(cost_, axis=1) / num_nodes_per_stage # Shape (stages,)

        # print(f"Mean load per port: {mean_load_per_port}")
        # print(f"Mean teu load per port: {mean_teu_load_per_port}")
        # print(f"Mean load per demand: {mean_load_per_demand}")
        # print(f"Realized demand: {mean_demand}")
        # breakpoint() # todo: remove later

        results = {
            # Input parameters
            "seed":seed,
            "ports":P,
            "scenarios":scenarios_per_stage,
            # Solver results
            "obj":obj,
            "time":mdl.solve_details.time,
            "gap":mdl.solve_details.mip_relative_gap,
            # Solution metrics
            "mean_load_per_port":mean_load_per_port.tolist(),
            "mean_teu_load_per_port":mean_teu_load_per_port.tolist(),
            "mean_load_per_location":mean_load_per_location.tolist(),
            "mean_hatch_overstowage":mean_hatch_overstowage.tolist(),
            # "mean_ci":mean_ci.tolist(),
            "mean_demand":mean_demand.tolist(),
            "mean_revenue":mean_revenue.tolist(),
            "mean_cost":mean_cost.tolist(),
            "mean_pd":mean_pd.tolist(),
            "mean_mixing":mean_mixing.tolist(),
        }
        vars = {
            "seed": seed,
            "ports": P,
            "scenarios": scenarios_per_stage,
            "x": x_.tolist(),
            "PD": PD_.tolist(),
            "HO_": HO_.tolist(),
            "HM_": HM_.tolist(),
            # "CI_": CI_.tolist(),
            # "CI_target_": CI_target_.tolist(),
            # "LM_": LM_.tolist(),
            # "VM_": VM_.tolist(),
            # "TW_": TW_.tolist(),
        }
    else:
        # Print the error
        print("No solution found")
        results = {}
        vars = {}

    mdl.end()
    del mdl
    return results, vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", type=int, default=5)
    parser.add_argument("--teu", type=int, default=20000)
    parser.add_argument("--deterministic", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--perfect_information", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--generalization", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--scenarios", type=int, default=16) # 20
    parser.add_argument("--scenario_range", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--stochastic_algorithm", type=str, default="myopic")  # multi_stage, rolling_horizon, myopic
    # todo: add warm solution
    parser = parser.parse_args()

    # Load the configuration file
    path = f'{path_to_main}/config.yaml'
    config = load_config(path)

    # Add the arguments to the configuration
    config.env.ports = parser.ports
    config.env.teu = parser.teu
    config.env.perfect_information = parser.perfect_information
    config.env.deterministic = parser.deterministic
    config.env.generalization = parser.generalization
    config.testing.num_episodes = parser.num_episodes

    # Set parameters
    perfect_information = parser.perfect_information
    deterministic = parser.deterministic
    generalization = config.env.generalization
    num_episodes = config.testing.num_episodes
    stochastic_algorithm = parser.stochastic_algorithm
    scenario_range = parser.scenario_range if not generalization else False
    if deterministic:
        num_scenarios = [1]
    elif scenario_range:
        num_scenarios = list(range(4, parser.scenarios + 1, 4))
    else:
        num_scenarios = [parser.scenarios]

    # Precompute largest scenario tree
    stages = config.env.ports - 1  # Number of load ports (P-1)
    teu = config.env.teu
    max_scenarios_per_stage = max(num_scenarios) if max(num_scenarios) >= 28 else 28
    # Number of scenarios per stage
    max_paths = max_scenarios_per_stage ** (stages-1) + 1 if not deterministic else 1
    node_list = precompute_node_list(stages, max_scenarios_per_stage, deterministic)

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
            result, var = main(env, demand_sub_tree, scen, stages, max_paths, seed, perfect_information, deterministic, stochastic_algorithm)
            # log result as error (to show in .err file):
            print(f"Result: TEU{teu}_P{stages+1}_E{x}_S{scen}_PI{perfect_information}_Gen{generalization}: Obj {result.get('obj', None)}, Time {result.get('time', None)}, Gap {result.get('gap', None)}")
            break
        break

            # breakpoint() # todo: remove later
            #
            # # setup folder
            # if not os.path.exists("results/SMIP/scenario_tree/block_mpp/"):
            #     os.makedirs("results/SMIP/scenario_tree/block_mpp/")
            #
            # # Save results in json
            # with open(f"results/SMIP/scenario_tree/block_mpp/results_scenario_tree_teu{teu}_p{stages}_"
            #           f"e{x}_s{scen}_pi{perfect_information}"
            #           f"_gen{generalization}.json", "w") as json_file:
            #     json.dump(result, json_file, indent=4)