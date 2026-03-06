# Imports
import numpy as np
import pandas as pd
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
import time
from collections import defaultdict

path = 'add path to cplex here'
sys.path.append(path)

# Module imports
path_to_main = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(path_to_main)
from main import *
from environment.utils import get_pol_pod_pair
from rl_algorithms.utils import set_unique_seed


# Precompute functions
def precompute_node_list(stages:int, scenarios_per_stage:int, deterministic=False, stochastic_algorithm:str="mpc") -> List:
    """Precompute the list of nodes and their coordinates in the scenario tree"""
    node_list = []  # List to store the coordinates of all nodes
    # Loop over each stage, starting from stage 1 (root is stage 1)
    for stage in range(stages):
        # Number of nodes at this stage
        if deterministic:
            nodes_in_current_stage = 1
        else:
            if stochastic_algorithm in ["mpc", "rolling_horizon", "myopic"]:
                nodes_in_current_stage = scenarios_per_stage ** (stage) + 1
            elif stochastic_algorithm == "multi_stage":
                nodes_in_current_stage = scenarios_per_stage ** (stage)
            else:
                raise ValueError(f"Unknown stochastic_algorithm='{stochastic_algorithm}'")

        # For each node in the current stage
        for node_id in range(nodes_in_current_stage):
            node_list.append((stage, node_id))

    return node_list

def precompute_demand(node_list: List,
                      max_paths: int,
                      stages: int,
                      env: nn.Module,
                      stochastic_algorithm: str,
                      deterministic: bool = False
                      ) -> Tuple[Dict, Dict]:
    """
    Precompute:
      - demand_scenarios[(stage, node_id)] for the scenario tree
      - real_demand[stage] for the realized trajectory

    Conventions:
      - path 0 is the 'real' trajectory
      - For multi_stage:
          scenario node (stage, node_id) uses path = node_id
        (so scenario 0 == real path)
      - For MPC-style algorithms (mpc, rolling_horizon, myopic):
          scenario node (stage, node_id) uses path = node_id + 1
        (so real path 0 is outside the tree)
    """
    td = env.reset()
    pregen_demand = td["observation", "realized_demand"] \
        .detach().cpu().numpy().reshape(-1, env.T, env.K)

    num_paths_generated = pregen_demand.shape[0]
    if num_paths_generated < max_paths:
        raise RuntimeError(
            f"Environment generated only {num_paths_generated} paths, "
            f"but max_paths={max_paths}. Increase batch_size or reduce max_paths."
        )

    # demand_raw[path, k, pol, pod]
    demand_raw = np.zeros((max_paths, env.K, env.P, env.P))
    for transport in range(env.T):
        pol, pod = get_pol_pod_pair(th.tensor(transport), env.P)
        demand_raw[:, :, pol, pod] = pregen_demand[:, transport, :]

    # demand_[stage, path, k, j]
    demand_ = demand_raw.transpose(2, 0, 1, 3)

    demand_scenarios: Dict[Tuple[int, int], np.ndarray] = {}
    real_demand: Dict[int, np.ndarray] = {}

    # 1) Realized demand: path 0 (common to all algorithms)
    for stage in range(stages):
        real_demand[stage] = demand_[stage, 0, :, :]

    # 2) Fill scenario-tree demands
    if deterministic:
        # All nodes see the same realized demand
        for (stage, node_id) in node_list:
            demand_scenarios[stage, node_id] = real_demand[stage]

    elif stochastic_algorithm == "multi_stage":
        # Realization is part of the tree: scenario 0 == real path
        for (stage, node_id) in node_list:
            demand_scenarios[stage, node_id] = demand_[stage, node_id, :, :]

    elif stochastic_algorithm in {"mpc", "rolling_horizon", "myopic"}:
        # Build tree without realization, but shift node_id by 1 to get regular indexing
        for (stage, node_id) in node_list:
            if stage == 0:
                demand_scenarios[stage, node_id] = demand_[stage, node_id, :, :]
            elif stage > 0 and node_id > 0:
                demand_scenarios[stage, node_id-1] = demand_[stage, node_id, :, :]
    else:
        raise ValueError(f"Unknown stochastic_algorithm='{stochastic_algorithm}'")
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
        # Include only nodes within the allowed range
        if node < max_nodes:
            filtered_tree[(stage, node)] = value
    return filtered_tree

# Support functions
def get_demand_history(stage: int,
                       node_id: int,
                       demand: np.ndarray,
                       parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]]
                       ) -> np.ndarray:
    """History of realized demand along the path to (stage, node_id) up to 'stage' (excluded)."""
    history = []
    s, n = stage, node_id
    # walk backwards through parents
    while s > 0:
        s_prev, n_prev = parent[(s, n)]
        history.append(demand[s_prev, n_prev].flatten())
        s, n = s_prev, n_prev
    history.reverse()
    return np.concatenate(history) if history else np.array([])


def onboard_groups(ports:int, pol:int, transport_indices:list) -> np.array:
    load_index = np.array([transport_indices.index((pol, i)) for i in range(ports) if i > pol])  # List of cargo groups to load
    load = np.array([transport_indices[idx] for idx in load_index]).reshape((-1,2))
    discharge_index = np.array([transport_indices.index((i, pol)) for i in range(ports) if i < pol])  # List of cargo groups to discharge
    discharge = np.array([transport_indices[idx] for idx in discharge_index]).reshape((-1,2))
    port_moves = np.vstack([load, discharge]).astype(int)
    on_board = [(i, j) for i in range(ports) for j in range(ports) if i <= pol and j > pol]  # List of cargo groups to load
    return np.array(on_board), port_moves, load

def parent_node_id(parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]],
                   stage: int,
                   node_id: int) -> int:
    p = parent[(stage, node_id)]
    if p is None:
        raise ValueError(f"Requested parent of root node {(stage,node_id)}")
    ps, pn = p
    if ps != stage - 1:
        raise ValueError(f"Bad parent stage for {(stage,node_id)}: {p}")
    return pn


def debug_info_sets(stage_nodes: Dict[int, List[int]],
                    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]],
                    stages: int) -> None:
    """
    Prints how many distinct information sets exist per stage and the max group size.
    If every node is its own info set at stage>0, NA is effectively trivial.
    """
    def info_key(stage: int, node_id: int) -> Tuple[Tuple[int,int], ...]:
        chain = []
        s, n = stage, node_id
        while s > 0:
            p = parent.get((s,n), None)
            if p is None:
                break
            chain.append(p)
            s, n = p
        chain.reverse()
        return tuple(chain)

    for s in range(stages):
        nodes = stage_nodes.get(s, [])
        if s == 0 or len(nodes) == 0:
            print(f"[NA DEBUG] stage {s}: nodes={len(nodes)} (root/no nodes)")
            continue

        groups = defaultdict(list)
        for n in nodes:
            groups[info_key(s, n)].append(n)

        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        print(f"[NA DEBUG] stage {s}: nodes={len(nodes)}, info_sets={len(groups)}, "
              f"max_group={sizes[0] if sizes else 0}, group_sizes_top5={sizes[:5]}")


def check_state_linking_solution(x_val: Dict[Tuple, float],
                                 parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]],
                                 stages: int,
                                 stage_nodes: Dict[int, List[int]],
                                 B: int,
                                 D: int,
                                 BL: int,
                                 K: int,
                                 P: int,
                                 atol: float = 1e-6,
                                 max_violations_to_print: int = 10) -> bool:
    """
    Verifies the intended carry + auto-discharge relations on a solved solution.

    Inputs:
      x_val: dictionary mapping full x-index keys to numeric values, e.g.
             x_val[(stage,node,b,d,bl,k,pol,pod)] = value
             You can build this from docplex via:
               x_val[key] = x[key].solution_value

    Checks for each stage p>0 and each node:
      - if pod == p and pol < p: x[p,*,pol,p] == 0
      - if pod > p and pol < p: x[p,node,...,pol,pod] == x[p-1,parent(node),...,pol,pod]
    """
    violations = 0

    for p in range(1, stages):
        for n in stage_nodes.get(p, []):
            pn = parent_node_id(parent, p, n)

            for b in range(B):
                for d in range(D):
                    for bl in range(BL):
                        for k in range(K):
                            for pol in range(p):
                                for pod in range(pol + 1, P):
                                    key_now = (p, n, b, d, bl, k, pol, pod)
                                    v_now = float(x_val.get(key_now, 0.0))

                                    if pod == p:
                                        if abs(v_now) > atol:
                                            violations += 1
                                            if violations <= max_violations_to_print:
                                                print(f"[LINK VIOL] discharge not zero: "
                                                      f"x{key_now}={v_now}")
                                    elif pod > p:
                                        key_prev = (p-1, pn, b, d, bl, k, pol, pod)
                                        v_prev = float(x_val.get(key_prev, 0.0))
                                        if abs(v_now - v_prev) > atol:
                                            violations += 1
                                            if violations <= max_violations_to_print:
                                                print(f"[LINK VIOL] carry mismatch: "
                                                      f"x{key_now}={v_now} vs x{key_prev}={v_prev}")

    ok = (violations == 0)
    print(f"[LINK DEBUG] state-linking check: {'OK' if ok else 'FAILED'} "
          f"(violations={violations})")
    return ok


def count_na_constraints(mdl: Model, prefix: str = "na_") -> int:
    """
    Counts constraints whose names start with `prefix`.
    Useful to sanity-check that NA constraints are being added.
    """
    cnt = 0
    for ct in mdl.iter_constraints():
        name = ct.name
        if name and name.startswith(prefix):
            cnt += 1
    print(f"[NA DEBUG] constraints with prefix '{prefix}': {cnt}")
    return cnt

# Main function
def main(env:nn.Module, demand:np.array, real_demand:Dict, scenarios_per_stage:int=28, stages:int=3, max_paths:int=784, seed:int=42,
         perfect_information:bool=False, warm_solution:bool=False, look_ahead:int=2, print_results:bool=False) -> Tuple[Dict, Dict]:

    # Wallclock time:
    t_start = time.perf_counter()

    # Scenario tree parameters
    M = 10 ** 3 # Big M
    num_nodes_per_stage = [1*scenarios_per_stage**stage for stage in range(stages)]

    # Build mapping: stage -> list of node_ids
    stage_nodes = {s: [] for s in range(stages)}

    for s, n in node_list:
        # root always kept
        if s == 0:
            stage_nodes[0].append(n)
            continue

        # node_list has nodes 0..(scenarios_per_stage**s - 1) per stage
        max_nodes = scenarios_per_stage ** s
        if 0 <= n < max_nodes:
            stage_nodes[s].append(n)

    # Build parent mapping: (stage, node) -> (stage-1, parent_node)
    parent = {}
    for s in range(stages):
        nodes = stage_nodes[s]
        if s == 0:
            for n in nodes:
                parent[(s, n)] = None
        else:
            prev_nodes = stage_nodes[s - 1]
            n_curr = len(nodes)
            n_prev = len(prev_nodes)
            if n_prev == 0:
                raise ValueError(f"No nodes at stage {s - 1}, cannot assign parents for stage {s}.")

            # Even if branching is uneven, assign parents sequentially in order
            for idx, n in enumerate(nodes):
                p = prev_nodes[min(idx // max(1, n_curr // n_prev), n_prev - 1)]
                parent[(s, n)] = (s - 1, p)

    # After you build `parent` in main(), once:
    children = defaultdict(list)
    for (s, n), p in parent.items():
        if p is None:
            continue
        children[p].append((s, n))

    # Choose a realized node for each stage (here: first node at that stage)
    realized_node = {}
    for s in range(stages):
        if not stage_nodes[s]:
            raise ValueError(f"No nodes at stage {s}")
        # Pick the first node at that stage as realized scenario (can be refined later)
        realized_node[s] = 0

    debug_info_sets(stage_nodes=stage_nodes, parent=parent, stages=stages)

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

    def get_ancestor_chain(stage: int,
                           node_id: int,
                           parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]]
                           ) -> List[Tuple[int, int]]:
        """
        Return the full ancestor chain from stage 0 up to stage-1 for (stage, node_id).
        Example output for stage=3: [(0,0), (1,2), (2,7)].
        """
        chain = []
        s, n = stage, node_id
        while s > 0:
            s_prev, n_prev = parent[(s, n)]
            chain.append((s_prev, n_prev))
            s, n = s_prev, n_prev
        chain.reverse()
        return chain

    def debug_non_anticipativity(stage: int,
                                 stage_nodes: Dict[int, List[int]],
                                 parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]]
                                 ) -> None:
        """
        For a given stage, print:
          - two nodes with identical ancestor chains (same information set),
          - two nodes with different ancestor chains (different information sets).
        """

        nodes = stage_nodes[stage]
        if len(nodes) < 2:
            print(f"[DEBUG] Stage {stage}: fewer than 2 nodes, nothing to compare.")
            return

        # Compute ancestor chains for all nodes at this stage
        chains = {n: get_ancestor_chain(stage, n, parent) for n in nodes}

        # Group nodes by their chain
        groups = defaultdict(list)
        for n, ch in chains.items():
            groups[tuple(ch)].append(n)

        print(f"[DEBUG] Stage {stage}: {len(nodes)} nodes, {len(groups)} distinct ancestor chains")

        # 1) Find two nodes with the same ancestor chain (if any)
        similar_pair = None
        for ch, g in groups.items():
            if len(g) >= 2:
                similar_pair = (g[0], g[1], ch)
                break

        if similar_pair is not None:
            n1, n2, ch = similar_pair
            print(f"[DEBUG] Similar nodes at stage {stage}: {n1} and {n2}")
            print(f"        Shared ancestor chain: {ch}")
        else:
            print(f"[DEBUG] No two nodes at stage {stage} share the same ancestor chain.")

        # 2) Find two nodes with different ancestor chains
        if len(groups) >= 2:
            # Take first two distinct groups
            (ch1, g1), (ch2, g2) = list(groups.items())[:2]
            m1 = g1[0]
            m2 = g2[0]
            print(f"[DEBUG] Dissimilar nodes at stage {stage}: {m1} and {m2}")
            print(f"        Ancestor chain of {m1}: {ch1}")
            print(f"        Ancestor chain of {m2}: {ch2}")
        else:
            print(f"[DEBUG] All nodes at stage {stage} share the same ancestor chain.")

    def build_reachable_subtree(start_stage: int,
                                rh_stages: int,
                                realized_node: Dict[int, int],
                                parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
                                children: Dict[Tuple[int, int], List[Tuple[int, int]]]
                                ) -> Dict[int, List[int]]:
        """
        Return stage_nodes_local[s] for s in [start_stage, start_stage+rh_stages-1],
        where each list contains only nodes reachable from (start_stage, realized_node[start_stage])
        by following children.
        """
        max_stage = start_stage + rh_stages - 1
        stage_nodes_local: Dict[int, List[int]] = {s: [] for s in range(start_stage, max_stage + 1)}

        root = (start_stage, realized_node[start_stage])

        # Optional but strongly recommended sanity check
        if root not in parent and start_stage != 0:
            raise RuntimeError(
                f"Root node {root} not in parent mapping. "
                f"Check stage_nodes/realized_node construction."
            )

        frontier = [root]
        visited = {root}

        stage_nodes_local[start_stage].append(root[1])

        while frontier:
            s, n = frontier.pop()
            if s == max_stage:
                continue
            for (sc, nc) in children[(s, n)]:
                if sc > max_stage:
                    continue
                if (sc, nc) in visited:
                    continue
                visited.add((sc, nc))
                stage_nodes_local[sc].append(nc)
                frontier.append((sc, nc))

        # sanity: no empty stage inside horizon
        for s in range(start_stage, max_stage + 1):
            if not stage_nodes_local[s]:
                raise RuntimeError(f"Local subtree has no nodes at stage {s}.")
        return stage_nodes_local

    def initialize_vars_tree(stages: int,
                             start_stage: int = 0,
                             block_mpp: bool = False,
                             real_out_tree: bool = False,
                             stage_nodes_current: Optional[Dict[int, List[int]]] = None) -> None:
        sn = stage_nodes_current if stage_nodes_current is not None else stage_nodes

        for stage in range(start_stage, start_stage + stages):
            node_ids = sn[stage]
            for node_id in node_ids:
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

    def info_key(stage: int, node_id: int, parent):
        chain = []
        s, n = stage, node_id
        while s > 0:
            p = parent.get((s, n), None)
            if p is None:
                break
            if not (isinstance(p, tuple) and len(p) == 2):
                raise TypeError(f"Bad parent entry at {(s, n)}: {p} (type={type(p)})")
            ps, pn = p
            chain.append((ps, pn))
            s, n = ps, pn
        chain.reverse()
        return tuple(chain)

    def add_non_anticipativity(stage, node_ids, load_moves, parent, block_mpp: bool):
        if perfect_information or stochastic_algorithm != "multi_stage" or stage == 0:
            return

        groups = defaultdict(list)
        for n in node_ids:
            groups[info_key(stage, n, parent)].append(n)

        for _, g in groups.items():
            if len(g) <= 1:
                continue
            base = g[0]
            for other in g[1:]:
                # 1) NA for x (your existing constraints)
                for b in range(B):
                    for bl in range(BL):
                        for d in range(D):
                            for k in range(K):
                                for (i, j) in load_moves:
                                    mdl.add_constraint(
                                        x[stage, base, b, d, bl, k, i, j] ==
                                        x[stage, other, b, d, bl, k, i, j],
                                        ctname=f"na_x_{stage}_{base}_{other}_{b}_{d}_{bl}_{k}_{i}_{j}"
                                    )

                # 2) NA for hatch decisions/cost drivers
                for b in range(B):
                    for bl in range(BL):
                        mdl.add_constraint(
                            HM[stage, base, b, bl] == HM[stage, other, b, bl],
                            ctname=f"na_HM_{stage}_{base}_{other}_{b}_{bl}"
                        )
                        mdl.add_constraint(
                            HO[stage, base, b, bl] == HO[stage, other, b, bl],
                            ctname=f"na_HO_{stage}_{base}_{other}_{b}_{bl}"
                        )

                # 3) NA for crane-related scalars (per node)
                mdl.add_constraint(
                    CI[stage, base] == CI[stage, other],
                    ctname=f"na_CI_{stage}_{base}_{other}"
                )
                mdl.add_constraint(
                    CM[stage, base] == CM[stage, other],
                    ctname=f"na_CM_{stage}_{base}_{other}"
                )

                # 4) NA for block_mpp-only binaries
                if block_mpp:
                    for b in range(B):
                        for bl in range(BL):
                            mdl.add_constraint(
                                mixing[stage, base, b, bl] == mixing[stage, other, b, bl],
                                ctname=f"na_mix_{stage}_{base}_{other}_{b}_{bl}"
                            )
                            for pod in range(P):
                                mdl.add_constraint(
                                    PD[stage, base, b, bl, pod] == PD[stage, other, b, bl, pod],
                                    ctname=f"na_PD_{stage}_{base}_{other}_{b}_{bl}_{pod}"
                                )

    def build_tree(stages: int,
                   input_demand: np.array,
                   warm_solution: Optional[Dict] = None,
                   start_stage: int = 0,
                   look_ahead: int = 2,
                   block_mpp: bool = False,
                   strict_no_overstow: bool = False,
                   real_out_tree: bool = False,
                   stage_nodes_current: Optional[Dict[int, List[int]]] = None,
                   real_demand: Optional[Dict] = None
                   ) -> None:
        demand = copy.deepcopy(input_demand)
        sn = stage_nodes_current if stage_nodes_current is not None else stage_nodes

        for stage in range(start_stage, start_stage + stages):
            node_ids = sn[stage]

            # Decide whether this is the "root" stage of the MPC horizon
            use_real = (real_demand is not None and stage == start_stage)

            # Build sets, on_board, etc. as before
            on_board, port_moves, load_moves = onboard_groups(P, stage, transport_indices)
            prev_on_board, _, _ = onboard_groups(P, stage - 1, transport_indices) if stage > 0 else ([], [], [])

            for node_id in node_ids:

                # Choose the demand driver:
                if use_real:
                    d_stage_node = real_demand[stage]
                else:
                    d_stage_node = demand[stage, node_id]  # original scenario demand

                if stage > 0:
                    ps, pn = parent[(stage, node_id)]  # pn is parent node id at stage-1
                    for b in range(B):
                        for bl in range(BL):
                            for d in range(D):
                                for k in range(K):
                                    for pol in range(stage):  # loaded earlier ports
                                        for pod in range(pol + 1, P):
                                            if pod == stage:
                                                mdl.add_constraint(
                                                    x[stage, node_id, b, d, bl, k, pol, pod] == 0,
                                                    ctname=f"auto_discharge_{stage}_{node_id}_{b}_{d}_{bl}_{k}_{pol}_{pod}"
                                                )
                                            elif pod > stage:
                                                mdl.add_constraint(
                                                    x[stage, node_id, b, d, bl, k, pol, pod] ==
                                                    x[stage - 1, pn, b, d, bl, k, pol, pod],
                                                    ctname=f"carry_{stage}_{node_id}_{b}_{d}_{bl}_{k}_{pol}_{pod}"
                                                )

                # Demand satisfaction now uses d_stage_node instead of demand[stage, node_id]
                for (i, j) in load_moves:
                    for k in range(K):
                        mdl.add_constraint(
                            mdl.sum(x[stage, node_id, b, d, bl, k, i, j]
                                    for b in range(B) for d in range(D) for bl in range(BL))
                            <= d_stage_node[k, j],
                            ctname=f'demand_{stage}_{node_id}_{k}_{j}'
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
                    CI_target_parameter * 2 / B * mdl.sum(d_stage_node[k, j]
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

                # todo: Mixing bays constraints (optional)
                #         # Get mixing bay
                #         mdl.add_constraint(
                #             mdl.sum(PD[stage, node_id, b, bl, j] for j in range(P)) <= 1 + M*mixing[stage, node_id, b, bl]
                #         )
                #
                # # Maximum number of mixing bays
                # mdl.add_constraint(
                #     mdl.sum(mixing[stage, node_id, b, bl,] for b in range(B) for bl in range(BL)) <= int(BL * B * bay_mix)
                # )

            add_non_anticipativity(stage, sn[stage], load_moves, parent, block_mpp=block_mpp)

        # Add mip start
        if warm_solution:
            start_dict = _get_warm_start_dict(warm_solution)
            mdl.add_mip_start({x[key]: value for key, value in start_dict.items()}, write_level=1)

    def build_scenario_tree_mpc(stages: int,
                                demand: np.array,
                                real_demand: Dict,
                                warm_solution: Optional[Dict] = None,
                                block_mpp: bool = False,
                                look_ahead: int = 100,
                                real_out_tree: bool = False) -> Dict:
        rolling_horizon_x = {}
        rolling_horizon_HO = {}
        rolling_horizon_CM = {}
        time_spent = 0.0
        gaps = []


        for t in range(stages):
            mdl.clear()
            x.clear(); HO.clear(); HM.clear(); CM.clear()
            PD.clear(); mixing.clear()
            CI.clear(); CI_target.clear()
            LM.clear(); VM.clear();
            TW.clear()
            on_boards.clear(); all_port_moves.clear(); all_load_moves.clear()

            remaining_horizon = stages - t
            rh_stages = min(look_ahead, remaining_horizon)

            # todo: maybe rebuild the tree with new nodes
            stage_nodes_local = build_reachable_subtree(
                start_stage=t,
                rh_stages=rh_stages,
                realized_node=realized_node,
                parent=parent,
                children=children,
            )

            # Initialize variables on local subtree
            initialize_vars_tree(rh_stages,
                                 start_stage=t,
                                 block_mpp=block_mpp,
                                 real_out_tree=real_out_tree,
                                 stage_nodes_current=stage_nodes_local)

            # Freeze previous solution at t-1 → t along realized path
            if t > 0:
                prev_stage = t - 1
                prev_node = realized_node[prev_stage]
                curr_node = realized_node[t]
                for b in range(B):
                    for bl in range(BL):
                        for d in range(D):
                            for k in range(K):
                                for i in range(prev_stage + 1):
                                    for j in range(i + 1, P):
                                        if j == t:
                                            lb = ub = 0.0
                                        else:
                                            val = rolling_horizon_x[prev_stage, prev_node, b, d, bl, k, i, j]
                                            lb = ub = val
                                        key = (t, curr_node, b, d, bl, k, i, j)
                                        if key in x:
                                            x[key].set_lb(lb)
                                            x[key].set_ub(ub)

            # Build constraints on local subtree
            build_tree(rh_stages,
                       demand,
                       warm_solution,
                       start_stage=t,
                       look_ahead=rh_stages,
                       block_mpp=block_mpp,
                       strict_no_overstow=False,
                       real_out_tree=real_out_tree,
                       stage_nodes_current=stage_nodes_local,
                       real_demand=real_demand)

            # Objective for this horizon, using local subtree
            objective = objective_function(x, HO, CM, revenues_,
                                           start_stage=t,
                                           stage_nodes_current=stage_nodes_local)

            solution = solve_model(mdl, objective)
            time_spent += mdl.solve_details.time
            gaps.append(mdl.solve_details.mip_relative_gap)

            if solution is None:
                raise RuntimeError(f"No solution found at stage {t}")

            # Extract decisions on the local subtree
            curr_node = realized_node[t]
            rolling_horizon_CM[t, curr_node] = CM[(t, curr_node)].solution_value

            for b in range(B):
                for bl in range(BL):
                    rolling_horizon_HO[t, curr_node, b, bl] = HO[(t, curr_node, b, bl)].solution_value
                    for d in range(D):
                        for k in range(K):
                            for i in range(t + 1):
                                for j in range(i + 1, P):
                                    key = (t, curr_node, b, d, bl, k, i, j)
                                    if key in x:
                                        rolling_horizon_x[t, curr_node, b, d, bl, k, i, j] = x[key].solution_value
                                    else:
                                        # unreachable or zero
                                        rolling_horizon_x[t, curr_node, b, d, bl, k, i, j] = 0.0

        # Compute total objective along realized path
        objective_expr = final_objective(rolling_horizon_x, rolling_horizon_HO, rolling_horizon_CM, revenues_, stages)
        objective_value = mdl.solution.get_value(objective_expr)

        return {
            "x": rolling_horizon_x,
            "HO": rolling_horizon_HO,
            "CM": rolling_horizon_CM,
            "objective_value": objective_value,
            "time": time_spent,
            "gap": gaps,
        }

    def final_objective(x: Dict, HO: Dict, CM: Dict, revenues: Any, stages: int) -> Any:
        """Compute total objective over the realized path (for MPC / rolling horizon)."""
        return mdl.sum(
            (
                    mdl.sum(
                        revenues[stage, k, j] *
                        x[stage, realized_node[stage], b, d, bl, k, stage, j]
                        for j in range(stage + 1, P)  # discharge ports
                        for b in range(B)  # bays
                        for d in range(D)  # decks
                        for k in range(K)  # cargo classes
                        for bl in range(BL)  # blocks
                    )
                    - mdl.sum(
                env.ho_costs * HO[stage, realized_node[stage], b, bl]
                for b in range(B)
                for bl in range(BL)
            )
                    - env.cm_costs * CM[stage, realized_node[stage]]
            )
            for stage in range(stages)
        )

    def objective_function(x: Any, HO: Any, CM: Any, revenues: Any,
                           start_stage: int = None,
                           stage_nodes_current: Optional[Dict[int, List[int]]] = None) -> Any:
        sn = stage_nodes_current if stage_nodes_current is not None else stage_nodes

        def revenue_term(stage, node_id):
            return mdl.sum(
                revenues[stage, k, j] * x[stage, node_id, b, d, bl, k, stage, j]
                for j in range(stage + 1, P)
                for b in range(B)
                for d in range(D)
                for k in range(K)
                for bl in range(BL)
            )

        def ho_cost_term(stage, node_id):
            return mdl.sum(env.ho_costs * HO[stage, node_id, b, bl] for b in range(B) for bl in range(BL))

        def cm_cost_term(stage, node_id):
            return env.cm_costs * CM[stage, node_id]

        if stochastic_algorithm == "multi_stage":
            probabilities = {}
            for stage in range(stages):
                nodes = sn[stage]
                if not nodes:
                    continue
                p = 1.0 / len(nodes)
                for node_id in nodes:
                    probabilities[stage, node_id] = p

            objective = mdl.sum(
                probabilities[stage, node_id] *
                (revenue_term(stage, node_id)
                 - ho_cost_term(stage, node_id)
                 - cm_cost_term(stage, node_id))
                for stage in range(stages)
                for node_id in sn[stage]
            )

        elif stochastic_algorithm in ["mpc", "rolling_horizon", "myopic"]:
            # Receding-horizon objective: expected value over local subtree
            h_start = start_stage
            h_end = min(stages, h_start + look_ahead)
            terms = []

            for s in range(h_start, h_end):
                nodes = sn[s]
                if not nodes:
                    continue

                # Equal weights for nodes at this stage in the local subtree
                p = 1.0 / len(nodes)

                for n in nodes:
                    terms.append(
                        p * (
                                revenue_term(s, n)
                                - ho_cost_term(s, n)
                                - cm_cost_term(s, n)
                        )
                    )

            objective = mdl.sum(terms)

        else:
            raise ValueError(f"Invalid stochastic algorithm: {stochastic_algorithm}")

        return objective

    def solve_model(mdl:Model, objective:Any) -> Any:
        mdl.maximize(objective)
        mdl.context.cplex_parameters.read.datacheck = 2
        mdl.parameters.mip.strategy.file = 3
        mdl.parameters.emphasis.memory = 1  # Prioritize memory savings over speed
        mdl.parameters.threads = 1  # Use only 1 thread to reduce memory usage
        mdl.parameters.mip.tolerances.mipgap = 0.001  # 1% or 0.1%
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
        count_na_constraints(mdl, prefix="na_")

        objective = objective_function(x, HO, CM, revenues_)
        solution = solve_model(mdl, objective)
    elif stochastic_algorithm == "mpc":
        # compare first nodes in demand, and real_demand
        solution = build_scenario_tree_mpc(stages, demand, real_demand, warm_solution, block_mpp=block_mpp, real_out_tree=real_out_tree)
    elif stochastic_algorithm == "rolling_horizon":
        solution = build_scenario_tree_mpc(stages, demand, real_demand, look_ahead=look_ahead, block_mpp=block_mpp, real_out_tree=real_out_tree)
    elif stochastic_algorithm == "myopic":
        # todo: should this lookahead be 1 or 0?
        solution = build_scenario_tree_mpc(stages, demand, real_demand, look_ahead=1, block_mpp=block_mpp, real_out_tree=real_out_tree)
    else:
        raise ValueError("Invalid stochastic algorithm")

    elapsed = time.perf_counter() - t_start

    if solution is not None:
        # build x_val dictionary from docplex vars
        x_val = {key: var.solution_value for key, var in x.items()}
        check_state_linking_solution(
            x_val=x_val,
            parent=parent,
            stages=stages,
            stage_nodes=stage_nodes,
            B=B, D=D, BL=BL, K=K, P=P,
            atol=1e-6
        )

    # Print the solution # todo: fix this properly for mpp and block_mpp; now set to block_mpp
    if solution:
        if "objective_value" in solution:
            obj = solution.get("objective_value", None)
            solver_time = solution.get("time", None)
            gap = solution.get("gap", None)
        else:
            obj = solution.objective_value
            solver_time = mdl.solve_details.time
            gap = mdl.solve_details.mip_relative_gap
        wallclock_time = elapsed

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
            # For MPC / rolling horizon / myopic, we only have the realized node in solution["x"], etc.
            if stochastic_algorithm in ["rolling_horizon", "myopic", "mpc"]:
                src_node = realized_node[stage]  # this is the node index used in the solver
            else:
                src_node = None

            for node_id in range(scenarios[stage]):
                for bay in range(B):
                    for block in range(BL):
                        for deck in range(D):
                            for cargo_class in range(K):
                                for pol in range(stage + 1):
                                    for pod in range(pol + 1, P):
                                        if stochastic_algorithm in ["rolling_horizon", "myopic", "mpc"]:
                                            # Map from (stage, src_node, ...) -> dense index [stage, node_id, ...]
                                            x_[stage, node_id, bay, deck, block, cargo_class, pol, pod] = \
                                                solution["x"].get(
                                                    (stage, src_node, bay, deck, block, cargo_class, pol, pod),
                                                    0.0
                                                )
                                        else:
                                            x_[stage, node_id, bay, deck, block, cargo_class, pol, pod] = \
                                                x[
                                                    stage, node_id, bay, deck, block, cargo_class, pol, pod].solution_value

                                        revenue_[stage, node_id] += \
                                            revenues_[stage, cargo_class, pod] * \
                                            x_[stage, node_id, bay, deck, block, cargo_class, pol, pod]

                                        demand_[stage, node_id, cargo_class, pod] = \
                                            demand[
                                                stage, src_node if stochastic_algorithm in ["rolling_horizon", "myopic",
                                                                                            "mpc"] else node_id][
                                                cargo_class, pod]

                        # block_mpp section will NOT work for MPC as written, because PD/mixing are not stored in solution.
                        # If you need those for MPC, you must store them in build_scenario_tree_mpc similarly to HO/CM/x.
                        if env.name == "block_mpp" and stochastic_algorithm == "multi_stage":
                            for pod in range(stage + 1, P):
                                PD_[stage, node_id, bay, block, pod] = PD[
                                    stage, node_id, bay, block, pod].solution_value
                            mixing_[stage, node_id, bay, block] = mixing[stage, node_id, bay, block].solution_value

                        if stochastic_algorithm in ["rolling_horizon", "myopic", "mpc"]:
                            HO_[stage, node_id, bay, block] = solution["HO"][(stage, src_node, bay, block)]
                            HM_[stage, node_id, bay, block] = 1 if HO_[stage, node_id, bay, block] > 0 else 0
                        else:
                            HO_[stage, node_id, bay, block] = HO[stage, node_id, bay, block].solution_value
                            HM_[stage, node_id, bay, block] = HM[stage, node_id, bay, block].solution_value

                        cost_[stage, node_id] += env.ho_costs * HO_[stage, node_id, bay, block]

                # After bay/block loops, handle CM, CI, etc.
                if stochastic_algorithm in ["rolling_horizon", "myopic", "mpc"]:
                    CM_[stage, node_id] = solution["CM"][(stage, src_node)]
                else:
                    CM_[stage, node_id] = CM[stage, node_id].solution_value
                    CI_[stage, node_id] = CI[stage, node_id].solution_value
                    CI_target_[stage, node_id] = CI_target[stage, node_id].solution_value
                    LM_[stage, node_id] = LM[stage, node_id].solution_value
                    VM_[stage, node_id] = VM[stage, node_id].solution_value
                    TW_[stage, node_id] = TW[stage, node_id].solution_value

                cost_[stage, node_id] += env.cm_costs * CM_[stage, node_id]

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
            "solver_time":solver_time,
            "time":wallclock_time,
            "gap":gap,
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
    parser.add_argument("--env_name", type=str, default="mpp")
    parser.add_argument("--ports", type=int, default=4)
    parser.add_argument("--teu", type=int, default=1000) #20000)
    parser.add_argument("--deterministic", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--perfect_information", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--generalization", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--scenarios", type=int, default=80)
    parser.add_argument("--scenario_range", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--utilization_rate_initial_demand", type=float, default=1.1)
    parser.add_argument("--cv_demand", type=float, default=0.5)
    parser.add_argument("--look_ahead", type=int, default=4)
    parser.add_argument("--stochastic_algorithm", type=str, default="multi_stage") # rolling_horizon, myopic, multi_stage, mpc
    parser = parser.parse_args()

    # Load the configuration file
    path = f'{path_to_main}/config.yaml'
    config = load_config(path)
    env_name = parser.env_name if parser.env_name else config.env.env_name
    if env_name == "mpp":
        output_path = f"{path_to_main}/results/SMIP/navigating_uncertainty/teu1k"
    elif env_name == "block_mpp":
        output_path = f"{path_to_main}/results/SMIP/AI2STOW/teu20k"
    else:
        raise ValueError("Invalid environment name in config.yaml")

    # Add the arguments to the configuration
    config.env.ports = parser.ports
    config.env.TEU = parser.teu
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
        num_scenarios = [5, 10, 20, 40, 80]
        #list(range(4, parser.scenarios + 1, 4))
    else:
        num_scenarios = [parser.scenarios]

    # Precompute largest scenario tree
    stages = config.env.ports - 1
    teu = config.env.TEU
    max_scenarios_per_stage = max(num_scenarios)

    if deterministic:
        max_paths = 1
    else:
        max_paths = max_scenarios_per_stage ** (stages - 1) + 1

    node_list = precompute_node_list(stages, max_scenarios_per_stage, deterministic, stochastic_algorithm)

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
    if stochastic_algorithm == "multi_stage":
        stochastic_algorithm_path = stochastic_algorithm + ("_pi" if perfect_information else "_na")
    else:
        stochastic_algorithm_path = stochastic_algorithm
    if not os.path.exists(f"{output_path}/{stochastic_algorithm_path}/instances/"):
        os.makedirs(f"{output_path}/{stochastic_algorithm_path}/instances/")

    # Load demand from csv file (pre-generated to ensure comparability across algorithms and episodes)
    df = pd.read_csv(
        f"{output_path}/demand_P{config.env.ports}_gen{generalization}_UR{parser.utilization_rate_initial_demand}_cv{parser.cv_demand}.csv",
        header=None, index_col=False)
    real_demand = df.to_numpy()

    # Main loop over episodes and scenarios
    start_ep = parser.start_episode
    t = tqdm(range(start_ep, start_ep+num_episodes), desc="Episodes", unit="ep")
    for x in t:
        # Create the environment on cpu
        seed = config.env.seed + x + 1
        config.env.seed = seed
        set_unique_seed(seed)
        env = make_env(config.env, batch_size=[max_paths], device='cpu')
        # Precompute for each episode
        real_demand_episode = real_demand[x]
        demand_tree, _ = precompute_demand(
            node_list=node_list,
            max_paths=max_paths,
            stages=stages,
            env=env,
            stochastic_algorithm=stochastic_algorithm,
            deterministic=deterministic,
        )

        t2 = tqdm(num_scenarios, desc="Scenarios", unit="scen", leave=False)
        for scen in t2:
            # Filter sub-tree for the number of scenarios
            scen_sub_tree = scen + int(stochastic_algorithm in {"mpc", "rolling_horizon", "myopic"})
            real_out_tree = stochastic_algorithm in {"mpc", "rolling_horizon", "myopic"}
            demand_sub_tree = get_scenario_tree_indices(demand_tree, scen_sub_tree, real_out_tree=real_out_tree,)

            # Run the main logic and get results and variables
            result, var = main(env=env, demand=demand_sub_tree, real_demand=real_demand_episode,
                               scenarios_per_stage=scen, stages=stages, max_paths=max_paths,
                               seed=seed, perfect_information=perfect_information, look_ahead=look_ahead, print_results=False)
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
            with open(f"{output_path}/{stochastic_algorithm_path}/instances/"
                      f"results_scenario_tree_teu{teu}_p{stages}_e{x}_s{scen}_alg{stochastic_algorithm_path}_"
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