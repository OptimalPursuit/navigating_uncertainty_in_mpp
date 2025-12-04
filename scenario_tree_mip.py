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
from tqdm.auto import tqdm
import copy

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

def get_scenario_tree_indices(scenario_tree:Dict, num_scenarios:int, real_out_tree:bool=False) -> Dict:
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
        # Add extra node for realized outcome if required
        if real_out_tree:
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
         perfect_information:bool=False, deterministic:bool=False, algorithm:str='multi_stage', warm_solution:bool=False,
         look_ahead:int=2, print_results:bool=False) -> Tuple[Dict, Dict]:
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
    # only if mpp, then add singleton dimension
    if capacity.ndim == 2:
        capacity = capacity[:, :, np.newaxis]
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
        breakpoint() # todo: fix warm start
        return warm_start_dict

    def initialize_vars_tree(stages:int, start_stage:int=0, block_mpp:bool=False, real_out_tree:bool=False) -> None:
        # Build variables for full scenario tree
        for stage in range(start_stage, start_stage + stages):
            n = num_nodes_per_stage[stage - start_stage]
            node_range = range(n + 1) if real_out_tree else range(n)
            for node_id in node_range:
                # Crane intensity:
                CI[stage, node_id] = mdl.continuous_var(name=f'CI_{start_stage}_{stage}_{node_id}', lb=0)
                CI_target[stage, node_id] = mdl.continuous_var(name=f'CI_target_{start_stage}_{stage}_{node_id}', lb=0)
                CM[stage, node_id] = mdl.continuous_var(name=f'CM_{start_stage}_{stage}_{node_id}', lb=0)

                # Stability:
                LM[stage, node_id] = mdl.continuous_var(name=f'LM_{start_stage}_{stage}_{node_id}')
                VM[stage, node_id] = mdl.continuous_var(name=f'VM_{start_stage}_{stage}_{node_id}')
                TW[stage, node_id] = mdl.continuous_var(name=f'TW_{start_stage}_{stage}_{node_id}')

                for bay in range(B):
                    for block in range(BL):
                        # Hatch overstowage:
                        HO[stage, node_id, bay, block] = mdl.continuous_var(name=f'HO_{start_stage}_{stage}_{node_id}_{bay}_{block}', lb=0)
                        HM[stage, node_id, bay, block] = mdl.binary_var(name=f'HM_{start_stage}_{stage}_{node_id}_{bay}_{block}')
                        for deck in range(D):
                            for cargo_class in range(K):
                                for pol in range(stage + 1):
                                    for pod in range(pol + 1, P):
                                        # Cargo allocation:
                                        x[stage, node_id, bay, deck, block, cargo_class, pol, pod] = \
                                            mdl.continuous_var(name=f'x_{start_stage}_{stage}_{node_id}_{bay}_{deck}_{block}'
                                                                    f'_{cargo_class}_{pol}_{pod}', lb=0)
                        if block_mpp:
                            for pod in range(P):
                                # PODs in locations
                                PD[stage, node_id, bay, block, pod] = \
                                    mdl.binary_var(name=f'PD_{start_stage}_{stage}_{node_id}_{bay}_{block}_{pod}')
                            # Excess PODs
                            mixing[stage, node_id, bay, block] = \
                                mdl.binary_var(name=f'mixing_{start_stage}_{stage}_{node_id}_{bay}_{block}')

    def build_tree(stages:int, input_demand:np.array, warm_solution:Optional[Dict]=None, start_stage:int=0,
                   look_ahead:int=2, block_mpp:bool=False, strict_no_overstow:bool=False, real_out_tree:bool=False) -> None:
        """Function to build the scenario tree; with decisions and constraints for each node"""
        demand = copy.deepcopy(input_demand)

        # todo: ensure demand realization from node 0 is properly handled
        #  Now non-anticipativity constraints are only added if demand histories are similar

        for stage in range(start_stage, start_stage + stages):

            # Define sets at current stage/port
            on_board, port_moves, load_moves = onboard_groups(P, stage, transport_indices)
            prev_on_board, _, _ = onboard_groups(P, stage-1, transport_indices) if stage > 0 else ([], [], [])

            # Scenario range
            n = num_nodes_per_stage[:look_ahead][stage - start_stage]
            if stochastic_algorithm in ['rolling_horizon', 'myopic', 'mpc']:
                if stage == start_stage:
                    node_range = range(n)
                else:
                    node_range = range(1, n + 1)

            elif stochastic_algorithm == 'multi_stage':
                node_range = range(n)
            else:
                raise ValueError(f'Unknown stochastic algorithm: {stochastic_algorithm}')
            for node_id in node_range:
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
                                demand_history1 = get_demand_history(stage-start_stage, demand, num_nodes_per_stage)
                                for node_id2 in range(node_id + 1, num_nodes_per_stage[stage-start_stage]):
                                    demand_history2 = get_demand_history(stage-start_stage, demand, num_nodes_per_stage)
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
                            mdl.sum(x[stage, node_id, b, 1, bl, k, i, j] for (i, j) in port_moves for k in range(K))
                            <= M * HM[stage, node_id, b, bl], ctname=f'hatch_move_{stage}_{node_id}_{b}_{bl}'
                        )

                        # Hatch overstows (d=0 is on deck)
                        # Overstowage is arrival condition of previous port: prev_on_board
                        # Vessel is empty before stage 0, hence no overstows
                        if stage > 0:
                            mdl.add_constraint(
                                mdl.sum(x[stage, node_id, b, 0, bl, k, i, j] for (i, j) in prev_on_board
                                        for k in range(K) if j > stage) - M * (1 - HM[stage, node_id, b, bl] )
                                <= HO[stage, node_id, b, bl], ctname=f'hatch_overstow_{stage}_{node_id}_{b}_{bl}'
                            )
                            if strict_no_overstow:
                                # Add HO == 0
                                mdl.add_constraint(
                                    HO[stage, node_id, b, bl] == 0, ctname=f'hatch_overstow_zero_{stage}_{node_id}_{b}_{bl}'
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
                        mdl.sum(x[stage, node_id, b, d, bl, k, i, j] for (i, j) in port_moves
                                for k in range(K) for bl in range(BL) for d in range(D) for b in [adj_bay, adj_bay + 1])
                        <= CI[stage, node_id], ctname=f'crane_intensity_{stage}_{node_id}_{adj_bay}'
                    )
                # Ensure CI[stage] is bounded by CI_target
                mdl.add_constraint(
                    CI_target[stage, node_id] ==
                    CI_target_parameter * 2 / B * mdl.sum(demand[stage, node_id][k, j]
                                                          for (i, j) in port_moves for k in range(K)),
                    ctname=f'crane_intensity_target_{stage}_{node_id}'
                )
                # mdl.add_constraint(CI[stage, node_id] <= CI_target[stage, node_id])
                mdl.add_constraint(CI[stage, node_id] - CI_target[stage, node_id] <= CM[stage, node_id],
                                   ctname=f'crane_move_{stage}_{node_id}')

                # BLock stowage patterns: Binary j in each b,d
                if block_mpp:
                    for b in range(B):
                        for bl in range(BL):
                            for j in range(stage+1, P):
                                # Get POD indicator
                                mdl.add_constraint(
                                    mdl.sum(x[stage, node_id, b, d, bl, k, stage, j] for k in range(K) for d in range(D))
                                    <= M * PD[stage, node_id, b, bl, j], ctname=f'pod_indicator_{stage}_{node_id}_{b}_{d}_{bl}_{j}'
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
            start_dict = _get_warm_start_dict(warm_solution)
            mdl.add_mip_start({x[key]: value for key, value in start_dict.items()}, write_level=1)

    def build_scenario_tree_mpc(stages: int,
                                demand: np.array,
                                warm_solution: Optional[Dict] = None,
                                block_mpp:bool=False,
                                look_ahead:int=100,
                                real_out_tree:bool=False) -> Dict:
        """
        Recedding horizon model predictive control with solving multi-stage SMIP each step.
        """
        rolling_horizon_x = {}
        rolling_horizon_HO = {}
        rolling_horizon_CM = {}

        for t in range(stages):
            # Reuse the same model object, just clear it
            mdl.clear()

            # clear ALL variable containers before creating new variables
            x.clear(); HO.clear(); HM.clear(); CM.clear()
            PD.clear(); mixing.clear()
            CI.clear(); CI_target.clear()
            LM.clear(); VM.clear(); TW.clear()

            on_boards.clear()
            all_port_moves.clear()
            all_load_moves.clear()

            # Determine the remaining horizon and adjust the lookahead window
            remaining_horizon = stages - t  # min(look_ahead, stages - t)
            rh_stages = min(look_ahead, remaining_horizon)  # adaptively shrink horizon

            # Initialize variables for the current horizon
            initialize_vars_tree(rh_stages, start_stage=t, block_mpp=block_mpp, real_out_tree=real_out_tree)

            # Freeze previous solution x_{t-1} to current solution x_t
            if t > 0:
                prev_stage = t - 1
                for b in range(B):
                    for bl in range(BL):
                        for d in range(D):
                            for k in range(K):
                                # All origins that existed up to previous stage
                                for i in range(prev_stage + 1):
                                    for j in range(i + 1, P):
                                        if j == t:
                                            # Everything with pod = current port must be discharged
                                            lb = ub = 0.0
                                        else:
                                            val = rolling_horizon_x[prev_stage, 0, b, d, bl, k, i, j]
                                            lb = ub = val
                                        x[t, 0, b, d, bl, k, i, j].set_lb(lb)
                                        x[t, 0, b, d, bl, k, i, j].set_ub(ub)

            # Build the scenario tree for the current horizon
            build_tree(rh_stages, demand, warm_solution, start_stage=t, look_ahead=rh_stages, block_mpp=block_mpp)

            # Define the objective function for the current horizon
            objective = objective_function(x, HO, CM, revenues_, start_stage=t)

            # Solve the optimization problem
            solution = solve_model(mdl, objective)

            # Check if a solution was found
            if solution is None:
                raise RuntimeError(f"No solution found at stage {t}")

            # Apply the first-stage decision
            # Extract current stage decisions
            rolling_horizon_CM[t, 0] = CM[(t, 0)].solution_value
            for b in range(B):
                for bl in range(BL):
                    rolling_horizon_HO[t, 0, b, bl] = HO[(t, 0, b, bl)].solution_value
                    for d in range(D):
                        for k in range(K):
                            for i in range(t + 1):  # all origins so far
                                for j in range(i + 1, P):  # all destinations
                                    rolling_horizon_x[t, 0, b, d, bl, k, i, j] = x[(t, 0, b, d, bl, k, i, j)].solution_value

        # Compute objective over all current stage decisions
        objective_expr = final_objective(rolling_horizon_x, rolling_horizon_HO, rolling_horizon_CM, revenues_, stages)
        objective_value = mdl.solution.get_value(objective_expr)
        return {"x": rolling_horizon_x, "HO": rolling_horizon_HO, "CM": rolling_horizon_CM, "objective_value": objective_value}


    def build_rolling(stages: int,
                                demand: np.array,
                                look_ahead: int = 2,
                                warm_solution: Optional[Dict] = None,
                                block_tree:bool = False) -> Dict:
        """
        Rolling horizon wrapper around build_tree.
        At each stage t:
            - Build a 2-stage problem (current + next port scenarios)
            - Solve and apply stage-0 decisions
            - Move forward
        """
        rolling_horizon_x = {}
        rolling_horizon_HO = {}
        rolling_horizon_CM = {}
        demand = copy.deepcopy(demand)

        for t in range(stages):
            mdl.clear()
            on_boards.clear()
            all_port_moves.clear()
            all_load_moves.clear()

            # Determine remaining horizon and RH window size
            remaining = stages - t
            rh_stages = min(look_ahead, remaining)  # adaptively shrink horizon

            # Initialize variable tree for this RH window
            initialize_vars_tree(stages=rh_stages, start_stage=t, block_mpp=block_tree)

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
                                        if j == t:
                                            x[t, 0, b, d, bl, k, i, j].set_lb(0)
                                            x[t, 0, b, d, bl, k, i, j].set_ub(0)

            def _nodes_at(dct, s):
                return sorted({n for (ss, n) in dct.keys() if ss == s})

            # in build_rolling_block_mpp:
            local_demand = copy.deepcopy(demand)
            if t < stages - 1:
                avail = _nodes_at(local_demand, t + 1)
                realized_scen = 0 if deterministic else random.choice(avail)
                local_demand[(t + 1, 0)] = demand[(t + 1, realized_scen)]  # remap realized → node 0

            demand_window = {}
            for offset in range(rh_stages):
                s = t + offset
                if offset == 0:
                    demand_window[(s, 0)] = local_demand[(s, 0)]
                else:
                    for n in _nodes_at(local_demand, s):
                        demand_window[(s, n)] = local_demand[(s, n)]

            # Build constraints for this RH window
            build_tree(
                stages=rh_stages,
                input_demand=demand_window,
                warm_solution=warm_solution,
                start_stage=t,
                look_ahead=look_ahead,
                block_mpp=True,
            )

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
        objective_expr = final_objective(rolling_horizon_x, rolling_horizon_HO, rolling_horizon_CM, revenues_, stages)
        objective_value = mdl.solution.get_value(objective_expr)
        return {"x": rolling_horizon_x, "HO": rolling_horizon_HO, "CM": rolling_horizon_CM, "objective_value": objective_value}

    def final_objective(x: Dict, HO: Dict, CM: Dict, revenues: Any, stages: int) -> Any:
        """Compute total objective over all stored stage-0 decisions."""
        return mdl.sum((mdl.sum(revenues[stage, k, j] * x[stage, 0, b, d, bl, k, stage, j]
                                for j in range(stage + 1, P) # Loop over discharge ports
                                for b in range(B) # Loop over bays
                                for d in range(D) # Loop over decks
                                for k in range(K) # Loop over cargo classes
                                for bl in range(BL) # Loop over blocks
        )
                        - mdl.sum(env.ho_costs * HO[stage, 0, b, bl] for b in range(B) for bl in range(BL))
                        - env.cm_costs * CM[stage, 0]
        )
                       for stage in range(stages) # Iterate over all stages
        )

    def objective_function(x: Any, HO: Any, CM: Any, revenues: Any, start_stage: int = None) -> Any:
        """Define the optimization objective function."""

        def revenue_term(stage, node_id, with_blocks: bool):
            """Total revenue from x decisions."""
            return mdl.sum(
                revenues[stage, k, j] * x[stage, node_id, b, d, bl, k, stage, j]
                for j in range(stage + 1, P)
                for b in range(B)
                for d in range(D)
                for k in range(K)
                for bl in range(BL)
            )

        def ho_cost_term(stage, node_id, with_blocks: bool):
            """Hatch overstowage cost."""
            return mdl.sum(env.ho_costs * HO[stage, node_id, b, 0] for b in range(B))

        def cm_cost_term(stage, node_id):
            """Excess crane move cost."""
            return env.cm_costs * CM[stage, node_id]

        # ------------------------------------------------------------
        # MAIN LOGIC
        # ------------------------------------------------------------
        probabilities = {}

        if stochastic_algorithm == "multi_stage":
            for (stage, node_id) in node_list:
                probabilities[stage, node_id] = 1 / num_nodes_per_stage[stage]

            with_blocks = config.env.env_name == "block_mpp"

            objective = mdl.sum(
                probabilities[stage, node_id] * (
                        revenue_term(stage, node_id, with_blocks)
                        - ho_cost_term(stage, node_id, with_blocks)
                        - cm_cost_term(stage, node_id)
                )
                for stage in range(stages)
                for node_id in range(num_nodes_per_stage[stage])
            )
        elif stochastic_algorithm in ["mpc", "rolling_horizon", "myopic"]:
            h_start = start_stage
            h_end = min(stages, h_start + look_ahead)

            objective = mdl.sum(
                (1.0 / num_nodes_per_stage[s - h_start]) * (
                        revenue_term(s, n, True)
                        - ho_cost_term(s, n, True)
                        - cm_cost_term(s, n)
                )
                for s in range(h_start, h_end)
                for n in (
                    range(num_nodes_per_stage[s - h_start])  # stage == start
                    if s == start_stage
                    else range(1, num_nodes_per_stage[s - h_start] + 1)  # stage > start
                )
            )

        else:
            raise ValueError(f"Invalid stochastic algorithm: {stochastic_algorithm}")

        return objective

    def solve_model(mdl:Model, objective:Any) -> Any:
        mdl.maximize(objective)
        mdl.context.cplex_parameters.read.datacheck = 2
        mdl.parameters.mip.strategy.file = 3
        mdl.parameters.emphasis.memory = 1  # Prioritize memory savings over speed
        mdl.parameters.threads = 1  # Use only 1 thread to reduce memory usage
        mdl.parameters.mip.tolerances.mipgap = 0.01  # 1% or 0.1%
        mdl.set_time_limit(3600)  # 1 hour
        solution = mdl.solve(log_output=print_results)
        return solution

    # Reshape revenues to match the shape of x
    revenues_ = np.zeros((stages, K, P,))
    for stage in range(stages):
        for pod in range(stage + 1, P):
            for cargo_class in range(K):
                t = cargo_class + transport_indices.index((stage, pod)) * K
                revenues_[stage, cargo_class, pod,] = revenues[t]

    # Build the scenario tree
    block_mpp = (config.env.env_name == "block_mpp")
    if stochastic_algorithm == "multi_stage":
        initialize_vars_tree(stages, block_mpp=block_mpp)
        build_tree(stages, demand, warm_solution, block_mpp=block_mpp)
        objective = objective_function(x, HO, CM, revenues_)
        solution = solve_model(mdl, objective)
    elif stochastic_algorithm == "mpc":
        solution = build_scenario_tree_mpc(stages, demand, warm_solution, block_mpp=block_mpp, real_out_tree=real_out_tree)
    elif stochastic_algorithm == "rolling_horizon":
        solution = build_scenario_tree_mpc(stages, demand, look_ahead=look_ahead, block_mpp=block_mpp, real_out_tree=real_out_tree)
    elif stochastic_algorithm == "myopic":
        # todo: should this lookahead be 1 or 0?
        solution = build_scenario_tree_mpc(stages, demand, look_ahead=1, block_mpp=block_mpp, real_out_tree=real_out_tree)
    else:
        raise ValueError("Invalid stochastic algorithm")

    # Print the solution # todo: fix this properly for mpp and block_mpp; now set to block_mpp
    if solution:
        if "objective_value" in solution:
            obj = solution["objective_value"]
        else:
            obj = solution.objective_value

        # Analyze the solution
        x_ = np.zeros((stages, max_paths, B, D, BL, K, P, P), dtype=float)
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
                                        if stochastic_algorithm in ["rolling_horizon", "myopic", "mpc"]:
                                            x_[stage, node_id, bay, deck, block, cargo_class, pol, pod] = solution["x"].get((stage, node_id, bay, deck, block, cargo_class, pol, pod), 0)
                                        else:
                                            x_[stage, node_id, bay, deck, block, cargo_class, pol, pod] = x[stage, node_id, bay, deck, block, cargo_class, pol, pod].solution_value
                                        revenue_[stage, node_id,] += revenues_[stage, cargo_class, pod] * x_[stage, node_id, bay, deck, block, cargo_class, pol, pod]
                                        demand_[stage, node_id, cargo_class, pod] = demand[stage, node_id][cargo_class, pod]

                        if env.name == "block_mpp":
                            for pod in range(stage + 1, P):
                                PD_[stage, node_id, bay, block, pod] = PD[stage, node_id, bay, block, pod].solution_value
                            mixing_[stage, node_id, bay, block,] = mixing[stage, node_id, bay, block,].solution_value

                        if stochastic_algorithm in ["rolling_horizon", "myopic", "mpc"]:
                            HO_[stage, node_id, bay, block,] = solution["HO"][stage, node_id, bay, block]
                            HM_[stage, node_id, bay, block,] = 1 if HO_[stage, node_id, bay, block,] > 0 else 0
                        else:
                            HO_[stage, node_id, bay, block,] = HO[stage, node_id, bay, block,].solution_value
                            HM_[stage, node_id, bay, block,] = HM[stage, node_id, bay, block,].solution_value
                        cost_[stage, node_id,] += env.ho_costs * HO_[stage, node_id, bay, block]

                if stochastic_algorithm in ["rolling_horizon", "myopic", "mpc"]:
                    # todo: implement CI, stability etc. in rolling horizon
                    CM_[stage, node_id] = solution["CM"][stage, node_id]
                else:
                    CM_[stage, node_id] = CM[stage, node_id].solution_value
                    CI_[stage, node_id] = CI[stage, node_id].solution_value
                    CI_target_[stage, node_id] = CI_target[stage, node_id].solution_value
                    LM_[stage, node_id] = LM[stage, node_id].solution_value
                    VM_[stage, node_id] = VM[stage, node_id].solution_value
                    TW_[stage, node_id] = TW[stage, node_id].solution_value
                cost_[stage, node_id,] += env.cm_costs * CM_[stage, node_id]

        # Get metrics from the solution
        # todo: redo computations here!
        num_nodes_per_stage = np.array(num_nodes_per_stage)
        mean_load_per_port = np.sum(x_, axis=(1, 2, 3, 4, 5, 6, 7)) / num_nodes_per_stage # Shape (stages,)
        mean_load_per_demand = np.sum(x_, axis=(1, 2, 3, 4, 5)) / num_nodes_per_stage.reshape(-1, 1, 1) # Shape (stages, P, P)
        mean_teu_load_per_port = (x_ * teus.reshape(1, 1, 1, 1, 1, K, 1, 1)).sum(axis=(1,2, 3, 4, 5, 6, 7)) # Shape (stages,)
        mean_load_per_location = np.sum(x_, axis=(1, 4, 5, 6)) / num_nodes_per_stage.reshape(-1, 1, 1, 1) # Shape (stages, B, D, BL)
        mean_hatch_overstowage = np.sum(HO_, axis=(1, 2, 3)) / num_nodes_per_stage # Shape (stages,)
        # mean_ci = np.sum(CI_, axis=1) / num_nodes_per_stage # Shape (stages,)
        mean_pd = np.sum(PD_, axis=(1, 2, 3, 4)) / num_nodes_per_stage # Shape (stages,)
        mean_mixing = np.sum(mixing_, axis=(1, 2, 3)) / num_nodes_per_stage # Shape (stages,)

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
            "obj":obj,
            "time":mdl.solve_details.time,
            "gap":mdl.solve_details.mip_relative_gap,
            # Solution metrics
            "mean_load_per_port":mean_load_per_port.tolist(),
            "mean_teu_load_per_port":mean_teu_load_per_port.tolist(),
            "mean_load_per_location":mean_load_per_location.tolist(),
            "mean_hatch_overstowage":mean_hatch_overstowage.tolist(),
            "demand": demand_.tolist(),
            # "mean_ci":mean_ci.tolist(),
            "mean_demand":mean_demand.tolist(),
            "mean_revenue":mean_revenue.tolist(),
            "mean_cost":mean_cost.tolist(),
            "mean_pd":mean_pd.tolist(),
            "mean_mixing":mean_mixing.tolist(),
            # Max revenue (only for myopic)
            "max_revenue": solution.get("max_revenue", None) if isinstance(solution, dict) else None,
        }
        vars = {
            "seed": seed,
            "ports": P,
            "scenarios": scenarios_per_stage,
            "x": x_.tolist(),
            "PD": PD_.tolist(),
            "HO": HO_.tolist(),
            "HM": HM_.tolist(),
            "CM": CM_.tolist(),
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
    parser.add_argument("--ports", type=int, default=4)
    parser.add_argument("--teu", type=int, default=1000) #20000)
    parser.add_argument("--deterministic", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--perfect_information", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--generalization", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--scenarios", type=int, default=28) # 20
    parser.add_argument("--scenario_range", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--utilization_rate_initial_demand", type=float, default=1.1)
    parser.add_argument("--cv_demand", type=float, default=0.5)
    parser.add_argument("--look_ahead", type=int, default=4)  # only for rolling horizon
    parser.add_argument("--stochastic_algorithm", type=str, default="multi_stage") # rolling_horizon, myopic, multi_stage, mpc
    parser = parser.parse_args()

    # Load the configuration file
    path = f'{path_to_main}/config.yaml'
    config = load_config(path)
    env_name = config.env.env_name
    if env_name == "mpp":
        output_path = f"{path_to_main}/results/SMIP/navigating_uncertainty/teu1k"
    elif env_name == "block_mpp":
        output_path = f"{path_to_main}/results/SMIP/AI2STOW/teu20k"
    else:
        raise ValueError("Invalid environment name in config.yaml")

    # Add the arguments to the configuration
    config.env.ports = parser.ports
    config.env.teu = parser.teu
    config.env.perfect_information = parser.perfect_information
    config.env.deterministic = parser.deterministic
    config.env.generalization = parser.generalization
    config.env.utilization_rate_initial_demand = parser.utilization_rate_initial_demand
    config.env.cv_demand = parser.cv_demand
    config.testing.num_episodes = parser.num_episodes

    # Set parameters
    perfect_information = parser.perfect_information
    deterministic = parser.deterministic
    generalization = config.env.generalization
    num_episodes = config.testing.num_episodes
    stochastic_algorithm = parser.stochastic_algorithm
    look_ahead = 1 if stochastic_algorithm == "myopic" else parser.look_ahead
    scenario_range = parser.scenario_range if not generalization else False
    if deterministic:
        num_scenarios = [1]
    elif scenario_range:
        if stochastic_algorithm == "multi_stage":
            num_scenarios = list(range(4, parser.scenarios + 1, 4))
        elif stochastic_algorithm in ["mpc"]:
            num_scenarios = [1, 3, 5, 10]
        else:
            raise ValueError("Scenario range only supported for multi_stage and mpc algorithms")
    else:
        num_scenarios = [parser.scenarios]

    # Precompute largest scenario tree
    stages = config.env.ports - 1  # Number of load ports (P-1)
    teu = config.env.teu
    max_scenarios_per_stage = max(num_scenarios) if max(num_scenarios) > 28 else 28
    # Number of scenarios per stage
    max_paths = max_scenarios_per_stage ** (stages-1) if not deterministic else 1
    node_list = precompute_node_list(stages, max_scenarios_per_stage, deterministic)
    obj_list = []
    tot_demand_list = []
    max_ob_demand_list = []
    total_x_list = []
    total_ho_list = []
    total_cm_list = []
    max_revenue_list = []
    running_sum_obj = 0.0
    running_count = 0

    # setup folder
    if not os.path.exists(f"{output_path}/{stochastic_algorithm}/instances/"):
        os.makedirs(f"{output_path}/{stochastic_algorithm}/instances/")

    t = tqdm(range(num_episodes), desc="Episodes", unit="ep")
    for x in t:
        # Create the environment on cpu
        seed = config.env.seed + x + 1
        config.env.seed = seed
        set_unique_seed(seed)
        env = make_env(config.env, batch_size=[max_paths], device='cpu')
        # Precompute for each episode
        demand_tree, real_demand = precompute_demand(node_list, max_paths, stages, env)

        t2 = tqdm(num_scenarios, desc="Scenarios", unit="scen", leave=False)
        for scen in t2:
            # Filter sub-tree for the number of scenarios
            real_out_tree = stochastic_algorithm in {"mpc", "rolling_horizon", "myopic"}
            demand_sub_tree = get_scenario_tree_indices(demand_tree, scen, real_out_tree=real_out_tree,)

            # Run the main logic and get results and variables
            result, var = main(env=env, demand=demand_sub_tree, scenarios_per_stage=scen, stages=stages,
                               max_paths=max_paths, seed=seed, perfect_information=perfect_information,
                               deterministic=deterministic, algorithm=stochastic_algorithm, look_ahead=look_ahead,
                               print_results=False)
            # log result as error (to show in .err file):

            # print total teu on board per port
            # print("--------------------------------------------------")
            # print(f"Result: TEU{teu}_P{stages+1}_E{x}_S{scen}_PI{perfect_information}_Gen{generalization}: Obj {result.get('obj', None)}, Time {result.get('time', None)}, Gap {result.get('gap', None)}")
            total_x_list.append(np.sum(np.array(var.get('x', []), dtype=float)))
            total_ho_list.append(np.sum(np.array(var.get('HO', []), dtype=float)))
            total_cm_list.append(np.sum(np.array(var.get('CM', []), dtype=float)))
            # print("Total sum of X:", np.sum(np.array(var.get('x', []), dtype=float)))
            # print(f"Total TEU load per port: {np.array(result.get('mean_teu_load_per_port', []), dtype=float)}")
            demand = np.array(result.get('demand', []), dtype=int)
            ob_demand = []
            ob_teus = []
            transport_indices = [(i, j) for i in range(env.P) for j in range(env.P) if i < j]
            for p in range(env.P - 1):
                ob = 0
                ob_teu = 0
                for (i,j) in onboard_groups(env.P, p, transport_indices)[0]:
                    for k in range(env.K):
                        ob += demand[:,0][i, k, j]
                        ob_teu += demand[:,0][i, k, j] * env.teus[k]
                ob_demand.append(ob)
                ob_teus.append(ob_teu.item())

            # print(f"Load demand: {demand[:,0].sum(axis=(-2,-1))}")
            # print(f"Onboard demand per port: {ob_demand}")
            # print(f"Onboard TEU per port: {ob_teus}")
            tot_demand_list.append(demand[:,0].sum())
            max_ob_demand_list.append(max(x for x in ob_teus))
            max_revenue_list.append(result["max_revenue"])

            # get onboard set and compute onboard demand
            obj = result.get("obj", None)
            obj_list.append(obj)
            if obj is not None:
                running_sum_obj += obj
                running_count += 1
                avg_obj = running_sum_obj / running_count
                t2.set_description(f"Episodes (avg obj={avg_obj:.2f})")

            # Save results in json
            with open(f"{output_path}/{stochastic_algorithm}/instances/"
                      f"results_scenario_tree_teu{teu}_p{stages}_e{x}_s{scen}_alg{stochastic_algorithm}_"
                      f"pi{perfect_information}_gen{generalization}.json", "w") as json_file:
                json.dump(result, json_file, indent=4)

    print("==================================================")
    print(f"Type of algorithm: {stochastic_algorithm} with {num_episodes} episodes")
    if stochastic_algorithm in ["rolling_horizon", "myopic"]:
        print(f"Look-ahead window size: {look_ahead}")

    print(f"Avg/Std Sum(x) {np.mean(total_x_list)}/{np.std(total_x_list)}")
    print(f"Avg/Std Sum(HO) {np.mean(total_ho_list)}/{np.std(total_ho_list)}")
    print(f"Avg/Std Sum(CM) {np.mean(total_cm_list)}/{np.std(total_cm_list)}")
    print(f"Avg/Std Obj {np.mean(obj_list)}/{np.std(obj_list)}")
    print(f"Avg/Std Total demand {np.mean(tot_demand_list)}/{np.std(tot_demand_list)}")
    print(f"Avg/Std Max onboard demand {np.mean(max_ob_demand_list)}/{np.std(max_ob_demand_list)}")
    print(f"Avg/Std Max revenue {np.mean(max_revenue_list)}/{np.std(max_revenue_list)}")