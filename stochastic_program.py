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


# ============================================================
# Existing / reusable helpers
# ============================================================

def precompute_node_list(stages: int,
                         scenarios_per_stage: int,
                         deterministic: bool = False,
                         stochastic_algorithm: str = "multi_stage") -> List[Tuple[int, int]]:
    """Precompute the list of nodes and their coordinates in the scenario tree."""
    node_list = []
    for stage in range(stages):
        if deterministic:
            nodes_in_current_stage = 1
        else:
            if stochastic_algorithm == "multi_stage":
                nodes_in_current_stage = scenarios_per_stage ** stage
            elif stochastic_algorithm in ["mpc", "rolling_horizon", "myopic"]:
                nodes_in_current_stage = scenarios_per_stage ** stage + 1
            else:
                raise ValueError(f"Unknown stochastic_algorithm='{stochastic_algorithm}'")

        for node_id in range(nodes_in_current_stage):
            node_list.append((stage, node_id))
    return node_list


def precompute_demand(node_list: List[Tuple[int, int]],
                      max_paths: int,
                      stages: int,
                      env: nn.Module,
                      stochastic_algorithm: str,
                      deterministic: bool = False) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[int, np.ndarray]]:
    """
    Precompute:
      - demand_scenarios[(stage, node_id)] for the scenario tree
      - real_demand[stage] for the realized trajectory

    Conventions:
      - path 0 is the 'real' trajectory
      - For multi_stage:
          scenario node (stage, node_id) uses path = node_id
      - For MPC-style algorithms:
          scenario node (stage, node_id) uses path = node_id + 1
    """
    td = env.reset()
    pregen_demand = td["observation", "realized_demand"].detach().cpu().numpy().reshape(-1, env.T, env.K)

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

    # demand_[stage=pol, path, k, pod]
    demand_ = demand_raw.transpose(2, 0, 1, 3)

    demand_scenarios: Dict[Tuple[int, int], np.ndarray] = {}
    real_demand: Dict[int, np.ndarray] = {}

    for stage in range(stages):
        real_demand[stage] = demand_[stage, 0, :, :]

    if deterministic:
        for (stage, node_id) in node_list:
            demand_scenarios[(stage, node_id)] = real_demand[stage]

    elif stochastic_algorithm == "multi_stage":
        for (stage, node_id) in node_list:
            demand_scenarios[(stage, node_id)] = demand_[stage, node_id, :, :]

    elif stochastic_algorithm in {"mpc", "rolling_horizon", "myopic"}:
        for (stage, node_id) in node_list:
            if stage == 0:
                demand_scenarios[(stage, node_id)] = demand_[stage, node_id, :, :]
            elif stage > 0 and node_id > 0:
                demand_scenarios[(stage, node_id - 1)] = demand_[stage, node_id, :, :]
    else:
        raise ValueError(f"Unknown stochastic_algorithm='{stochastic_algorithm}'")

    return demand_scenarios, real_demand


def get_scenario_tree_indices(scenario_tree: Dict[Tuple[int, int], np.ndarray],
                              num_scenarios: int,
                              real_out_tree: bool = False) -> Dict[Tuple[int, int], np.ndarray]:
    """Filter a full scenario tree to the first `num_scenarios**stage` nodes at each stage."""
    filtered_tree = {}
    for (stage, node), value in scenario_tree.items():
        max_nodes = num_scenarios ** stage
        if node < max_nodes:
            filtered_tree[(stage, node)] = value
    return filtered_tree


def onboard_groups(ports: int, pol: int, transport_indices: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    load_index = np.array([transport_indices.index((pol, i)) for i in range(ports) if i > pol])
    load = np.array([transport_indices[idx] for idx in load_index]).reshape((-1, 2))
    discharge_index = np.array([transport_indices.index((i, pol)) for i in range(ports) if i < pol])
    discharge = np.array([transport_indices[idx] for idx in discharge_index]).reshape((-1, 2))
    port_moves = np.vstack([load, discharge]).astype(int)
    on_board = [(i, j) for i in range(ports) for j in range(ports) if i <= pol and j > pol]
    return np.array(on_board), port_moves, load


def count_na_constraints(mdl: Model, prefix: str = "na_") -> int:
    cnt = 0
    for ct in mdl.iter_constraints():
        name = ct.name
        if name and name.startswith(prefix):
            cnt += 1
    print(f"[NA DEBUG] constraints with prefix '{prefix}': {cnt}")
    return cnt


# ============================================================
# New structural helpers
# ============================================================

def current_load_pods(stage: int, P: int):
    """PODs that can be loaded at current stage."""
    return range(stage + 1, P)


def carry_pairs(stage: int, P: int):
    """Pairs (pol,pod) already loaded before stage and still onboard after stage."""
    return [(pol, pod) for pol in range(stage) for pod in range(stage + 1, P)]


def onboard_pairs(stage: int, P: int):
    """All (pol,pod) onboard after stage operations."""
    return [(pol, pod) for pol in range(stage + 1) for pod in range(stage + 1, P)]


def build_regular_tree(stages: int, scenarios_per_stage: int):
    """
    Regular S-ary tree:
      stage s has scenarios_per_stage ** s nodes
      parent of node n at stage s>0 is n // scenarios_per_stage
    """
    stage_nodes = {s: list(range(scenarios_per_stage ** s)) for s in range(stages)}
    parent = {(0, 0): None}
    children = defaultdict(list)

    for s in range(1, stages):
        for n in stage_nodes[s]:
            p = n // scenarios_per_stage
            parent[(s, n)] = (s - 1, p)
            children[(s - 1, p)].append((s, n))

    return stage_nodes, parent, children


def info_key(stage: int,
             node_id: int,
             parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]]):
    chain = []
    s, n = stage, node_id
    while s > 0:
        p = parent[(s, n)]
        chain.append(p)
        s, n = p
    chain.reverse()
    return tuple(chain)


def debug_info_sets(stage_nodes: Dict[int, List[int]],
                    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
                    stages: int) -> None:
    for s in range(stages):
        nodes = stage_nodes.get(s, [])
        if s == 0 or len(nodes) == 0:
            print(f"[NA DEBUG] stage {s}: nodes={len(nodes)} (root/no nodes)")
            continue

        groups = defaultdict(list)
        for n in nodes:
            groups[info_key(s, n, parent)].append(n)

        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        print(
            f"[NA DEBUG] stage {s}: nodes={len(nodes)}, info_sets={len(groups)}, "
            f"max_group={sizes[0] if sizes else 0}, group_sizes_top5={sizes[:5]}"
        )


def build_big_m_bounds(env, stages: int):
    """
    Tighter per-location bounds than a single global M.
    """
    B = env.B
    D = env.D
    BL = env.BL if hasattr(env, "BL") else 1

    teus = env.teus.detach().cpu().numpy()
    capacity = env.capacity.detach().cpu().numpy()
    if capacity.ndim == 2:
        capacity = capacity[:, :, np.newaxis]

    min_teu = float(np.min(teus))
    M_loc = np.ceil(capacity / min_teu)  # [B, D, BL]

    deck_on = 0
    deck_below = min(1, D - 1)

    M_hatch = np.zeros((stages, B, BL))
    M_over = np.zeros((stages, B, BL))
    for s in range(stages):
        for b in range(B):
            for bl in range(BL):
                # crude but valid upper bounds
                M_hatch[s, b, bl] = 2.0 * M_loc[b, deck_below, bl]
                M_over[s, b, bl] = M_loc[b, deck_on, bl]

    return M_loc, M_hatch, M_over


def build_revenues_array(env, stages: int) -> np.ndarray:
    P = env.P
    K = env.K
    revenues = env.revenues.detach().cpu().numpy()
    transport_indices = [(i, j) for i in range(P) for j in range(P) if i < j]

    revenues_ = np.zeros((stages, K, P))
    for stage in range(stages):
        for pod in range(stage + 1, P):
            for cargo_class in range(K):
                t = cargo_class + transport_indices.index((stage, pod)) * K
                revenues_[stage, cargo_class, pod] = revenues[t]
    return revenues_


def _transport_to_stage_pod_demand(transport_demand: np.ndarray, env, stages: int) -> np.ndarray:
    """
    transport_demand shape:
      - [T, K]
    returns:
      - [stages, K, P]
    """
    if transport_demand.shape != (env.T, env.K):
        raise ValueError(f"Expected transport_demand shape {(env.T, env.K)}, got {transport_demand.shape}")

    out = np.zeros((stages, env.K, env.P))
    for transport in range(env.T):
        pol, pod = get_pol_pod_pair(th.tensor(transport), env.P)
        if pol < stages:
            out[pol, :, pod] = transport_demand[transport, :]
    return out


def coerce_real_demand_path(real_demand_episode: Any, env, stages: int) -> np.ndarray:
    """
    Tries to coerce the episode-level realized demand into shape [stages, K, P].

    Accepted inputs:
      - dict {stage: [K,P]}
      - ndarray [stages, K, P]
      - ndarray [T, K]
      - flat ndarray length T*K
      - flat ndarray length stages*K*P
    """
    if isinstance(real_demand_episode, dict):
        arr = np.zeros((stages, env.K, env.P))
        for s in range(stages):
            arr[s] = np.asarray(real_demand_episode[s], dtype=float)
        return arr

    arr = np.asarray(real_demand_episode, dtype=float)

    if arr.ndim == 3:
        if arr.shape == (stages, env.K, env.P):
            return arr
        if arr.shape == (stages, env.P, env.K):
            return arr.transpose(0, 2, 1)
        raise ValueError(f"Unsupported 3D real demand shape {arr.shape}")

    if arr.ndim == 2:
        if arr.shape == (env.T, env.K):
            return _transport_to_stage_pod_demand(arr, env, stages)
        if arr.shape == (stages, env.K * env.P):
            return arr.reshape(stages, env.K, env.P)
        if arr.shape == (stages, env.K,):  # unlikely
            out = np.zeros((stages, env.K, env.P))
            out[:, :, -1] = arr
            return out
        raise ValueError(f"Unsupported 2D real demand shape {arr.shape}")

    if arr.ndim == 1:
        if arr.size == env.T * env.K:
            return _transport_to_stage_pod_demand(arr.reshape(env.T, env.K), env, stages)
        if arr.size == stages * env.K * env.P:
            return arr.reshape(stages, env.K, env.P)
        raise ValueError(f"Unsupported flat real demand size {arr.size}")

    raise ValueError(f"Unsupported real demand format with ndim={arr.ndim}")


def check_state_linking_solution_y(y_val: Dict[Tuple, float],
                                   parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
                                   stage_nodes: Dict[int, List[int]],
                                   stages: int,
                                   B: int,
                                   D: int,
                                   BL: int,
                                   K: int,
                                   P: int,
                                   atol: float = 1e-6,
                                   max_violations_to_print: int = 10) -> bool:
    """
    Check y-state carry constraints:
      y[s,node,...,pol,pod] = y[s-1,parent,...,pol,pod] for pol < s < pod
    """
    violations = 0
    for s in range(1, stages):
        for n in stage_nodes[s]:
            pn = parent[(s, n)][1]
            for b in range(B):
                for d in range(D):
                    for bl in range(BL):
                        for k in range(K):
                            for pol, pod in carry_pairs(s, P):
                                key_now = (s, n, b, d, bl, k, pol, pod)
                                key_prev = (s - 1, pn, b, d, bl, k, pol, pod)
                                v_now = float(y_val.get(key_now, 0.0))
                                v_prev = float(y_val.get(key_prev, 0.0))
                                if abs(v_now - v_prev) > atol:
                                    violations += 1
                                    if violations <= max_violations_to_print:
                                        print(f"[LINK VIOL] y{key_now}={v_now} vs y{key_prev}={v_prev}")

    ok = (violations == 0)
    print(f"[LINK DEBUG] y-state linking check: {'OK' if ok else 'FAILED'} (violations={violations})")
    return ok


# ============================================================
# New NA solver: split action/state variables
# ============================================================

def solve_multistage_na(env,
                        demand_tree: Dict[Tuple[int, int], np.ndarray],
                        scenarios_per_stage: int,
                        stages: int,
                        revenues_: np.ndarray,
                        block_mpp: bool = False,
                        perfect_information: bool = False,
                        print_results: bool = False) -> Dict[str, Any]:
    """
    Multistage stochastic model with split variables:
      l[s,n,...,pod] = load-now action at port s
      y[s,n,...,pol,pod] = onboard state after stage s operations
    """
    P = env.P
    B = env.B
    D = env.D
    K = env.K
    BL = env.BL if hasattr(env, "BL") else 1
    deck_on = 0
    deck_below = min(1, D - 1)

    teus = env.teus.detach().cpu().numpy()
    weights = env.weights.detach().cpu().numpy()
    capacity = env.capacity.detach().cpu().numpy()
    if capacity.ndim == 2:
        capacity = capacity[:, :, np.newaxis]

    longitudinal_position = env.longitudinal_position.detach().cpu().numpy()
    vertical_position = env.vertical_position.detach().cpu().numpy()

    stab_delta = env.stab_delta
    LCG_target = env.LCG_target
    VCG_target = env.VCG_target
    CI_target_parameter = env.CI_target

    stage_nodes, parent, children = build_regular_tree(stages, scenarios_per_stage)
    M_loc, M_hatch, M_over = build_big_m_bounds(env, stages)

    if print_results and not perfect_information:
        debug_info_sets(stage_nodes=stage_nodes, parent=parent, stages=stages)

    mdl = Model(name="multistage_mpp_na_split")

    # Variables
    l = {}
    y = {}
    HM = {}
    HO = {}
    CI = {}
    CM = {}
    PD = {}
    mixing = {}

    for s in range(stages):
        for n in stage_nodes[s]:
            CI[(s, n)] = mdl.continuous_var(lb=0, name=f"CI_{s}_{n}")
            CM[(s, n)] = mdl.continuous_var(lb=0, name=f"CM_{s}_{n}")

            for b in range(B):
                for bl in range(BL):
                    HM[(s, n, b, bl)] = mdl.binary_var(name=f"HM_{s}_{n}_{b}_{bl}")
                    HO[(s, n, b, bl)] = mdl.continuous_var(lb=0, name=f"HO_{s}_{n}_{b}_{bl}")

                    if block_mpp:
                        for pod in current_load_pods(s, P):
                            PD[(s, n, b, bl, pod)] = mdl.binary_var(name=f"PD_{s}_{n}_{b}_{bl}_{pod}")
                        mixing[(s, n, b, bl)] = mdl.binary_var(name=f"mixing_{s}_{n}_{b}_{bl}")

                    for d in range(D):
                        for k in range(K):
                            for pod in current_load_pods(s, P):
                                l[(s, n, b, d, bl, k, pod)] = mdl.continuous_var(
                                    lb=0, name=f"l_{s}_{n}_{b}_{d}_{bl}_{k}_{pod}"
                                )

                            for pol, pod in onboard_pairs(s, P):
                                y[(s, n, b, d, bl, k, pol, pod)] = mdl.continuous_var(
                                    lb=0, name=f"y_{s}_{n}_{b}_{d}_{bl}_{k}_{pol}_{pod}"
                                )

    # Useful expressions
    def total_weight_expr(s, n):
        return mdl.sum(
            weights[k] * y[(s, n, b, d, bl, k, pol, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pol, pod in onboard_pairs(s, P)
        )

    def long_moment_expr(s, n):
        return mdl.sum(
            longitudinal_position[b] * weights[k] * y[(s, n, b, d, bl, k, pol, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pol, pod in onboard_pairs(s, P)
        )

    def vert_moment_expr(s, n):
        return mdl.sum(
            vertical_position[d] * weights[k] * y[(s, n, b, d, bl, k, pol, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pol, pod in onboard_pairs(s, P)
        )

    def discharge_expr(s, n, b, d, bl):
        if s == 0:
            return 0
        pn = parent[(s, n)][1]
        return mdl.sum(
            y[(s - 1, pn, b, d, bl, k, pol, s)]
            for k in range(K)
            for pol in range(s)
        )

    def ondeck_carry_before_ops_expr(s, n, b, bl):
        if s == 0:
            return 0
        pn = parent[(s, n)][1]
        return mdl.sum(
            y[(s - 1, pn, b, deck_on, bl, k, pol, pod)]
            for k in range(K)
            for pol, pod in carry_pairs(s, P)
        )

    def revenue_term(s, n):
        return mdl.sum(
            revenues_[s, k, pod] * l[(s, n, b, d, bl, k, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pod in current_load_pods(s, P)
        )

    def ho_cost_term(s, n):
        return mdl.sum(env.ho_costs * HO[(s, n, b, bl)] for b in range(B) for bl in range(BL))

    def cm_cost_term(s, n):
        return env.cm_costs * CM[(s, n)]

    # Constraints
    for s in range(stages):
        for n in stage_nodes[s]:
            d_stage_node = demand_tree[(s, n)]  # [K, P]

            # Demand: only new loads at current port
            for k in range(K):
                for pod in current_load_pods(s, P):
                    mdl.add_constraint(
                        mdl.sum(l[(s, n, b, d, bl, k, pod)] for b in range(B) for d in range(D) for bl in range(BL))
                        <= d_stage_node[k, pod],
                        ctname=f"demand_{s}_{n}_{k}_{pod}"
                    )

            # Transition from previous state + new loads
            if s == 0:
                for b in range(B):
                    for d in range(D):
                        for bl in range(BL):
                            for k in range(K):
                                for pod in current_load_pods(0, P):
                                    mdl.add_constraint(
                                        y[(0, n, b, d, bl, k, 0, pod)] == l[(0, n, b, d, bl, k, pod)],
                                        ctname=f"init_load_{n}_{b}_{d}_{bl}_{k}_{pod}"
                                    )
            else:
                pn = parent[(s, n)][1]

                # carry surviving cargo
                for b in range(B):
                    for d in range(D):
                        for bl in range(BL):
                            for k in range(K):
                                for pol, pod in carry_pairs(s, P):
                                    mdl.add_constraint(
                                        y[(s, n, b, d, bl, k, pol, pod)] ==
                                        y[(s - 1, pn, b, d, bl, k, pol, pod)],
                                        ctname=f"carry_{s}_{n}_{b}_{d}_{bl}_{k}_{pol}_{pod}"
                                    )

                                # newly loaded cargo becomes onboard state
                                for pod in current_load_pods(s, P):
                                    mdl.add_constraint(
                                        y[(s, n, b, d, bl, k, s, pod)] ==
                                        l[(s, n, b, d, bl, k, pod)],
                                        ctname=f"load_to_state_{s}_{n}_{b}_{d}_{bl}_{k}_{pod}"
                                    )

            # Capacity after stage operations
            for b in range(B):
                for d in range(D):
                    for bl in range(BL):
                        mdl.add_constraint(
                            mdl.sum(
                                teus[k] * y[(s, n, b, d, bl, k, pol, pod)]
                                for k in range(K)
                                for pol, pod in onboard_pairs(s, P)
                            ) <= capacity[b, d, bl],
                            ctname=f"capacity_{s}_{n}_{b}_{d}_{bl}"
                        )

            # Hatch move / overstowage
            for b in range(B):
                for bl in range(BL):
                    below_deck_load = mdl.sum(
                        l[(s, n, b, deck_below, bl, k, pod)]
                        for k in range(K)
                        for pod in current_load_pods(s, P)
                    )

                    below_deck_discharge = discharge_expr(s, n, b, deck_below, bl)

                    mdl.add_constraint(
                        below_deck_load + below_deck_discharge
                        <= M_hatch[s, b, bl] * HM[(s, n, b, bl)],
                        ctname=f"hatch_move_{s}_{n}_{b}_{bl}"
                    )

                    if s > 0:
                        mdl.add_constraint(
                            ondeck_carry_before_ops_expr(s, n, b, bl)
                            - M_over[s, b, bl] * (1 - HM[(s, n, b, bl)])
                            <= HO[(s, n, b, bl)],
                            ctname=f"hatch_over_{s}_{n}_{b}_{bl}"
                        )

            # Crane intensity
            ci_target_const = CI_target_parameter * 2.0 / B * float(np.sum(d_stage_node[:, s + 1:P]))

            for adj_bay in range(B - 1):
                port_activity = mdl.sum(
                    l[(s, n, b, d, bl, k, pod)]
                    for b in [adj_bay, adj_bay + 1]
                    for d in range(D)
                    for bl in range(BL)
                    for k in range(K)
                    for pod in current_load_pods(s, P)
                )

                if s > 0:
                    pn = parent[(s, n)][1]
                    port_activity += mdl.sum(
                        y[(s - 1, pn, b, d, bl, k, pol, s)]
                        for b in [adj_bay, adj_bay + 1]
                        for d in range(D)
                        for bl in range(BL)
                        for k in range(K)
                        for pol in range(s)
                    )

                mdl.add_constraint(
                    port_activity <= CI[(s, n)],
                    ctname=f"crane_intensity_{s}_{n}_{adj_bay}"
                )

            mdl.add_constraint(
                CI[(s, n)] - ci_target_const <= CM[(s, n)],
                ctname=f"crane_move_{s}_{n}"
            )

            # Stability after stage operations
            tw = total_weight_expr(s, n)
            lm = long_moment_expr(s, n)
            vm = vert_moment_expr(s, n)

            mdl.add_constraint(stab_delta * tw >= lm - LCG_target * tw, ctname=f"lcg_ub_{s}_{n}")
            mdl.add_constraint(stab_delta * tw >= -lm + LCG_target * tw, ctname=f"lcg_lb_{s}_{n}")
            mdl.add_constraint(stab_delta * tw >= vm - VCG_target * tw, ctname=f"vcg_ub_{s}_{n}")
            mdl.add_constraint(stab_delta * tw >= -vm + VCG_target * tw, ctname=f"vcg_lb_{s}_{n}")

            # block_mpp POD indicators: based only on current-stage loads
            if block_mpp:
                for b in range(B):
                    for bl in range(BL):
                        for pod in current_load_pods(s, P):
                            mdl.add_constraint(
                                mdl.sum(l[(s, n, b, d, bl, k, pod)] for d in range(D) for k in range(K))
                                <= (M_loc[b, deck_on, bl] + M_loc[b, deck_below, bl]) * PD[(s, n, b, bl, pod)],
                                ctname=f"pod_indicator_{s}_{n}_{b}_{bl}_{pod}"
                            )
                        mdl.add_constraint(
                            mdl.sum(PD[(s, n, b, bl, pod)] for pod in current_load_pods(s, P)) <= 1,
                            ctname=f"max_pod_{s}_{n}_{b}_{bl}"
                        )

    # NA only on load-now action variables
    if not perfect_information:
        for s in range(1, stages):
            groups = defaultdict(list)
            for n in stage_nodes[s]:
                groups[info_key(s, n, parent)].append(n)

            for _, g in groups.items():
                if len(g) <= 1:
                    continue

                base = g[0]
                for other in g[1:]:
                    for b in range(B):
                        for d in range(D):
                            for bl in range(BL):
                                for k in range(K):
                                    for pod in current_load_pods(s, P):
                                        mdl.add_constraint(
                                            l[(s, base, b, d, bl, k, pod)] ==
                                            l[(s, other, b, d, bl, k, pod)],
                                            ctname=f"na_l_{s}_{base}_{other}_{b}_{d}_{bl}_{k}_{pod}"
                                        )

    # Objective
    probabilities = {}
    for s in range(stages):
        p = 1.0 / len(stage_nodes[s])
        for n in stage_nodes[s]:
            probabilities[(s, n)] = p

    objective = mdl.sum(
        probabilities[(s, n)] * (revenue_term(s, n) - ho_cost_term(s, n) - cm_cost_term(s, n))
        for s in range(stages)
        for n in stage_nodes[s]
    )

    mdl.maximize(objective)
    mdl.context.cplex_parameters.read.datacheck = 2
    mdl.parameters.mip.strategy.file = 3
    mdl.parameters.emphasis.memory = 1
    mdl.parameters.threads = 1
    mdl.parameters.mip.tolerances.mipgap = 0.001
    mdl.set_time_limit(3600)

    if not perfect_information and print_results:
        count_na_constraints(mdl, prefix="na_")

    solution = mdl.solve(log_output=print_results)
    if solution is None:
        mdl.end()
        raise RuntimeError("No solution found for multistage NA model")

    out = {
        "objective_value": solution.objective_value,
        "solver_time": mdl.solve_details.time,
        "gap": mdl.solve_details.mip_relative_gap,
        "l": {k: v.solution_value for k, v in l.items()},
        "y": {k: v.solution_value for k, v in y.items()},
        "HM": {k: v.solution_value for k, v in HM.items()},
        "HO": {k: v.solution_value for k, v in HO.items()},
        "CM": {k: v.solution_value for k, v in CM.items()},
        "stage_nodes": stage_nodes,
        "parent": parent,
    }

    if block_mpp:
        out["PD"] = {k: v.solution_value for k, v in PD.items()}
        out["mixing"] = {k: v.solution_value for k, v in mixing.items()}

    y_val = out["y"]
    check_state_linking_solution_y(
        y_val=y_val,
        parent=parent,
        stage_nodes=stage_nodes,
        stages=stages,
        B=B, D=D, BL=BL, K=K, P=P,
        atol=1e-6
    )

    mdl.end()
    return out


# ============================================================
# New PI solver: deterministic path, no tree / no NA
# ============================================================

def solve_pi_path(env,
                  demand_path: np.ndarray,   # [stages, K, P]
                  stages: int,
                  revenues_: np.ndarray,
                  block_mpp: bool = False,
                  print_results: bool = False) -> Dict[str, Any]:
    """
    Perfect-information deterministic path model.
    No scenario tree. No node index. No NA.
    """
    P = env.P
    B = env.B
    D = env.D
    K = env.K
    BL = env.BL if hasattr(env, "BL") else 1
    deck_on = 0
    deck_below = min(1, D - 1)

    teus = env.teus.detach().cpu().numpy()
    weights = env.weights.detach().cpu().numpy()
    capacity = env.capacity.detach().cpu().numpy()
    if capacity.ndim == 2:
        capacity = capacity[:, :, np.newaxis]

    longitudinal_position = env.longitudinal_position.detach().cpu().numpy()
    vertical_position = env.vertical_position.detach().cpu().numpy()

    stab_delta = env.stab_delta
    LCG_target = env.LCG_target
    VCG_target = env.VCG_target
    CI_target_parameter = env.CI_target

    M_loc, M_hatch, M_over = build_big_m_bounds(env, stages)

    mdl = Model(name="multistage_mpp_pi_path")

    l = {}
    y = {}
    HM = {}
    HO = {}
    CI = {}
    CM = {}
    PD = {}
    mixing = {}

    # Variables
    for s in range(stages):
        CI[s] = mdl.continuous_var(lb=0, name=f"CI_{s}")
        CM[s] = mdl.continuous_var(lb=0, name=f"CM_{s}")

        for b in range(B):
            for bl in range(BL):
                HM[(s, b, bl)] = mdl.binary_var(name=f"HM_{s}_{b}_{bl}")
                HO[(s, b, bl)] = mdl.continuous_var(lb=0, name=f"HO_{s}_{b}_{bl}")

                if block_mpp:
                    for pod in current_load_pods(s, P):
                        PD[(s, b, bl, pod)] = mdl.binary_var(name=f"PD_{s}_{b}_{bl}_{pod}")
                    mixing[(s, b, bl)] = mdl.binary_var(name=f"mixing_{s}_{b}_{bl}")

                for d in range(D):
                    for k in range(K):
                        for pod in current_load_pods(s, P):
                            l[(s, b, d, bl, k, pod)] = mdl.continuous_var(
                                lb=0, name=f"l_{s}_{b}_{d}_{bl}_{k}_{pod}"
                            )

                        for pol, pod in onboard_pairs(s, P):
                            y[(s, b, d, bl, k, pol, pod)] = mdl.continuous_var(
                                lb=0, name=f"y_{s}_{b}_{d}_{bl}_{k}_{pol}_{pod}"
                            )

    # Expressions
    def total_weight_expr(s):
        return mdl.sum(
            weights[k] * y[(s, b, d, bl, k, pol, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pol, pod in onboard_pairs(s, P)
        )

    def long_moment_expr(s):
        return mdl.sum(
            longitudinal_position[b] * weights[k] * y[(s, b, d, bl, k, pol, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pol, pod in onboard_pairs(s, P)
        )

    def vert_moment_expr(s):
        return mdl.sum(
            vertical_position[d] * weights[k] * y[(s, b, d, bl, k, pol, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pol, pod in onboard_pairs(s, P)
        )

    def discharge_expr(s, b, d, bl):
        if s == 0:
            return 0
        return mdl.sum(
            y[(s - 1, b, d, bl, k, pol, s)]
            for k in range(K)
            for pol in range(s)
        )

    def ondeck_carry_before_ops_expr(s, b, bl):
        if s == 0:
            return 0
        return mdl.sum(
            y[(s - 1, b, deck_on, bl, k, pol, pod)]
            for k in range(K)
            for pol, pod in carry_pairs(s, P)
        )

    # Constraints
    for s in range(stages):
        d_stage = demand_path[s]  # [K, P]

        # Demand: only current-stage loads
        for k in range(K):
            for pod in current_load_pods(s, P):
                mdl.add_constraint(
                    mdl.sum(l[(s, b, d, bl, k, pod)] for b in range(B) for d in range(D) for bl in range(BL))
                    <= d_stage[k, pod],
                    ctname=f"demand_{s}_{k}_{pod}"
                )

        # Transition
        if s == 0:
            for b in range(B):
                for d in range(D):
                    for bl in range(BL):
                        for k in range(K):
                            for pod in current_load_pods(0, P):
                                mdl.add_constraint(
                                    y[(0, b, d, bl, k, 0, pod)] == l[(0, b, d, bl, k, pod)],
                                    ctname=f"init_load_{b}_{d}_{bl}_{k}_{pod}"
                                )
        else:
            for b in range(B):
                for d in range(D):
                    for bl in range(BL):
                        for k in range(K):
                            for pol, pod in carry_pairs(s, P):
                                mdl.add_constraint(
                                    y[(s, b, d, bl, k, pol, pod)] ==
                                    y[(s - 1, b, d, bl, k, pol, pod)],
                                    ctname=f"carry_{s}_{b}_{d}_{bl}_{k}_{pol}_{pod}"
                                )

                            for pod in current_load_pods(s, P):
                                mdl.add_constraint(
                                    y[(s, b, d, bl, k, s, pod)] ==
                                    l[(s, b, d, bl, k, pod)],
                                    ctname=f"load_to_state_{s}_{b}_{d}_{bl}_{k}_{pod}"
                                )

        # Capacity
        for b in range(B):
            for d in range(D):
                for bl in range(BL):
                    mdl.add_constraint(
                        mdl.sum(
                            teus[k] * y[(s, b, d, bl, k, pol, pod)]
                            for k in range(K)
                            for pol, pod in onboard_pairs(s, P)
                        ) <= capacity[b, d, bl],
                        ctname=f"capacity_{s}_{b}_{d}_{bl}"
                    )

        # Hatch move / overstowage
        for b in range(B):
            for bl in range(BL):
                below_deck_load = mdl.sum(
                    l[(s, b, deck_below, bl, k, pod)]
                    for k in range(K)
                    for pod in current_load_pods(s, P)
                )

                below_deck_discharge = discharge_expr(s, b, deck_below, bl)

                mdl.add_constraint(
                    below_deck_load + below_deck_discharge
                    <= M_hatch[s, b, bl] * HM[(s, b, bl)],
                    ctname=f"hatch_move_{s}_{b}_{bl}"
                )

                if s > 0:
                    mdl.add_constraint(
                        ondeck_carry_before_ops_expr(s, b, bl)
                        - M_over[s, b, bl] * (1 - HM[(s, b, bl)])
                        <= HO[(s, b, bl)],
                        ctname=f"hatch_over_{s}_{b}_{bl}"
                    )

        # Crane intensity
        ci_target_const = CI_target_parameter * 2.0 / B * float(np.sum(d_stage[:, s + 1:P]))

        for adj_bay in range(B - 1):
            port_activity = mdl.sum(
                l[(s, b, d, bl, k, pod)]
                for b in [adj_bay, adj_bay + 1]
                for d in range(D)
                for bl in range(BL)
                for k in range(K)
                for pod in current_load_pods(s, P)
            )

            if s > 0:
                port_activity += mdl.sum(
                    y[(s - 1, b, d, bl, k, pol, s)]
                    for b in [adj_bay, adj_bay + 1]
                    for d in range(D)
                    for bl in range(BL)
                    for k in range(K)
                    for pol in range(s)
                )

            mdl.add_constraint(
                port_activity <= CI[s],
                ctname=f"crane_intensity_{s}_{adj_bay}"
            )

        mdl.add_constraint(
            CI[s] - ci_target_const <= CM[s],
            ctname=f"crane_move_{s}"
        )

        # Stability
        tw = total_weight_expr(s)
        lm = long_moment_expr(s)
        vm = vert_moment_expr(s)

        mdl.add_constraint(stab_delta * tw >= lm - LCG_target * tw, ctname=f"lcg_ub_{s}")
        mdl.add_constraint(stab_delta * tw >= -lm + LCG_target * tw, ctname=f"lcg_lb_{s}")
        mdl.add_constraint(stab_delta * tw >= vm - VCG_target * tw, ctname=f"vcg_ub_{s}")
        mdl.add_constraint(stab_delta * tw >= -vm + VCG_target * tw, ctname=f"vcg_lb_{s}")

        if block_mpp:
            for b in range(B):
                for bl in range(BL):
                    for pod in current_load_pods(s, P):
                        mdl.add_constraint(
                            mdl.sum(l[(s, b, d, bl, k, pod)] for d in range(D) for k in range(K))
                            <= (M_loc[b, deck_on, bl] + M_loc[b, deck_below, bl]) * PD[(s, b, bl, pod)],
                            ctname=f"pod_indicator_{s}_{b}_{bl}_{pod}"
                        )
                    mdl.add_constraint(
                        mdl.sum(PD[(s, b, bl, pod)] for pod in current_load_pods(s, P)) <= 1,
                        ctname=f"max_pod_{s}_{b}_{bl}"
                    )

    # Objective
    objective = mdl.sum(
        mdl.sum(
            revenues_[s, k, pod] * l[(s, b, d, bl, k, pod)]
            for b in range(B)
            for d in range(D)
            for bl in range(BL)
            for k in range(K)
            for pod in current_load_pods(s, P)
        )
        - mdl.sum(env.ho_costs * HO[(s, b, bl)] for b in range(B) for bl in range(BL))
        - env.cm_costs * CM[s]
        for s in range(stages)
    )

    mdl.maximize(objective)
    mdl.context.cplex_parameters.read.datacheck = 2
    mdl.parameters.mip.strategy.file = 3
    mdl.parameters.emphasis.memory = 1
    mdl.parameters.threads = 1
    mdl.parameters.mip.tolerances.mipgap = 0.001
    mdl.set_time_limit(3600)

    solution = mdl.solve(log_output=print_results)
    if solution is None:
        mdl.end()
        raise RuntimeError("No solution found for PI path model")

    stage_nodes = {s: [0] for s in range(stages)}
    parent = {(0, 0): None}
    for s in range(1, stages):
        parent[(s, 0)] = (s - 1, 0)

    # Reindex to match NA style
    l_out = {}
    y_out = {}
    HM_out = {}
    HO_out = {}
    CM_out = {}

    for (s, b, d, bl, k, pod), val in {k: v.solution_value for k, v in l.items()}.items():
        l_out[(s, 0, b, d, bl, k, pod)] = val

    for (s, b, d, bl, k, pol, pod), val in {k: v.solution_value for k, v in y.items()}.items():
        y_out[(s, 0, b, d, bl, k, pol, pod)] = val

    for (s, b, bl), val in {k: v.solution_value for k, v in HM.items()}.items():
        HM_out[(s, 0, b, bl)] = val

    for (s, b, bl), val in {k: v.solution_value for k, v in HO.items()}.items():
        HO_out[(s, 0, b, bl)] = val

    for s, val in {k: v.solution_value for k, v in CM.items()}.items():
        CM_out[(s, 0)] = val

    out = {
        "objective_value": solution.objective_value,
        "solver_time": mdl.solve_details.time,
        "gap": mdl.solve_details.mip_relative_gap,
        "l": l_out,
        "y": y_out,
        "HM": HM_out,
        "HO": HO_out,
        "CM": CM_out,
        "stage_nodes": stage_nodes,
        "parent": parent,
    }

    if block_mpp:
        PD_out = {}
        mixing_out = {}
        for (s, b, bl, pod), val in {k: v.solution_value for k, v in PD.items()}.items():
            PD_out[(s, 0, b, bl, pod)] = val
        for (s, b, bl), val in {k: v.solution_value for k, v in mixing.items()}.items():
            mixing_out[(s, 0, b, bl)] = val
        out["PD"] = PD_out
        out["mixing"] = mixing_out

    check_state_linking_solution_y(
        y_val=out["y"],
        parent=parent,
        stage_nodes=stage_nodes,
        stages=stages,
        B=B, D=D, BL=BL, K=K, P=P,
        atol=1e-6
    )

    mdl.end()
    return out


# ============================================================
# Optional expected PI benchmark (average over sampled paths)
# ============================================================

def solve_pi_expected(env,
                      demand_paths: np.ndarray,   # [path, stages, K, P]
                      stages: int,
                      revenues_: np.ndarray,
                      num_paths: int,
                      block_mpp: bool = False,
                      print_results: bool = False) -> Dict[str, Any]:
    objs = []
    times = []
    gaps = []

    for path_id in range(num_paths):
        sol = solve_pi_path(
            env=env,
            demand_path=demand_paths[path_id],
            stages=stages,
            revenues_=revenues_,
            block_mpp=block_mpp,
            print_results=print_results
        )
        objs.append(sol["objective_value"])
        times.append(sol["solver_time"])
        gaps.append(sol["gap"])

    return {
        "objective_value": float(np.mean(objs)),
        "solver_time": float(np.sum(times)),
        "gap": gaps,
        "path_objectives": objs,
    }


# ============================================================
# Packing outputs back into your old-style results/vars format
# ============================================================

def package_results_and_vars(sol: Dict[str, Any],
                             env,
                             stages: int,
                             max_paths: int,
                             scenarios_per_stage: int,
                             perfect_information: bool,
                             revenues_: np.ndarray,
                             demand_source: Any,
                             block_mpp: bool) -> Tuple[Dict, Dict]:
    """
    Build outputs close to your old result/vars structure.

    Conventions:
      - vars["x"] stores onboard state y for compatibility
      - vars["l"] stores new load decisions
    """
    P = env.P
    B = env.B
    D = env.D
    K = env.K
    BL = env.BL if hasattr(env, 'BL') else 1
    teus = env.teus.detach().cpu().numpy()

    scenarios = [1] * stages if perfect_information else [scenarios_per_stage ** s for s in range(stages)]

    x_ = np.zeros((stages, max_paths, B, D, BL, K, P, P), dtype=float)   # onboard state y
    l_ = np.zeros((stages, max_paths, B, D, BL, K, P), dtype=float)      # current-stage loads only by POD
    PD_ = np.zeros((stages, max_paths, B, BL, P), dtype=float)
    mixing_ = np.zeros((stages, max_paths, B, BL), dtype=float)
    HO_ = np.zeros((stages, max_paths, B, BL), dtype=float)
    HM_ = np.zeros((stages, max_paths, B, BL), dtype=float)
    CM_ = np.zeros((stages, max_paths), dtype=float)
    demand_ = np.zeros((stages, max_paths, K, P), dtype=float)
    revenue_ = np.zeros((stages, max_paths), dtype=float)
    cost_ = np.zeros((stages, max_paths), dtype=float)

    # Fill demand
    if perfect_information:
        real_path = np.asarray(demand_source, dtype=float)  # [stages, K, P]
        for s in range(stages):
            demand_[s, 0, :, :] = real_path[s]
    else:
        demand_tree = demand_source
        for s in range(stages):
            for n in range(scenarios[s]):
                demand_[s, n, :, :] = demand_tree[(s, n)]

    # Fill decision/state arrays
    for s in range(stages):
        for n in range(scenarios[s]):
            src_n = 0 if perfect_information else n

            for b in range(B):
                for bl in range(BL):
                    HO_[s, n, b, bl] = sol["HO"].get((s, src_n, b, bl), 0.0)
                    HM_[s, n, b, bl] = sol["HM"].get((s, src_n, b, bl), 0.0)
                    cost_[s, n] += env.ho_costs * HO_[s, n, b, bl]

                    if block_mpp:
                        mixing_[s, n, b, bl] = sol.get("mixing", {}).get((s, src_n, b, bl), 0.0)
                        for pod in range(P):
                            PD_[s, n, b, bl, pod] = sol.get("PD", {}).get((s, src_n, b, bl, pod), 0.0)

                    for d in range(D):
                        for k in range(K):
                            for pod in current_load_pods(s, P):
                                val_l = sol["l"].get((s, src_n, b, d, bl, k, pod), 0.0)
                                l_[s, n, b, d, bl, k, pod] = val_l
                                revenue_[s, n] += revenues_[s, k, pod] * val_l

                            for pol, pod in onboard_pairs(s, P):
                                val_y = sol["y"].get((s, src_n, b, d, bl, k, pol, pod), 0.0)
                                x_[s, n, b, d, bl, k, pol, pod] = val_y

            CM_[s, n] = sol["CM"].get((s, src_n), 0.0)
            cost_[s, n] += env.cm_costs * CM_[s, n]

    # Metrics
    num_nodes_per_stage = np.array(scenarios, dtype=float)
    mean_load_per_port = np.sum(l_, axis=(1, 2, 3, 4, 5, 6)) / num_nodes_per_stage
    mean_teu_load_per_port = np.sum(
        l_ * teus.reshape(1, 1, 1, 1, 1, K, 1),
        axis=(1, 2, 3, 4, 5, 6)
    ) / num_nodes_per_stage
    mean_load_per_location = np.sum(x_, axis=(1, 5, 6, 7)) / num_nodes_per_stage.reshape(-1, 1, 1, 1)
    mean_hatch_overstowage = np.sum(HO_, axis=(1, 2, 3)) / num_nodes_per_stage
    mean_pd = np.sum(PD_, axis=(1, 2, 3, 4)) / num_nodes_per_stage
    mean_mixing = np.sum(mixing_, axis=(1, 2, 3)) / num_nodes_per_stage
    mean_demand = np.sum(demand_, axis=(1, 2, 3)) / num_nodes_per_stage
    mean_revenue = np.sum(revenue_, axis=1) / num_nodes_per_stage
    mean_cost = np.sum(cost_, axis=1) / num_nodes_per_stage
    max_revenue = float(np.sum(mean_revenue))

    results = {
        "obj": sol["objective_value"],
        "solver_time": sol["solver_time"],
        "time": sol["solver_time"],
        "gap": sol["gap"],
        "mean_load_per_port": mean_load_per_port.tolist(),
        "mean_teu_load_per_port": mean_teu_load_per_port.tolist(),
        "mean_load_per_location": mean_load_per_location.tolist(),
        "mean_hatch_overstowage": mean_hatch_overstowage.tolist(),
        "demand": demand_.tolist(),
        "mean_demand": mean_demand.tolist(),
        "mean_revenue": mean_revenue.tolist(),
        "mean_cost": mean_cost.tolist(),
        "mean_pd": mean_pd.tolist(),
        "mean_mixing": mean_mixing.tolist(),
        "max_revenue": max_revenue,
    }

    vars_out = {
        "x": x_.tolist(),   # onboard state y
        "l": l_.tolist(),   # new load action
        "PD": PD_.tolist(),
        "HO": HO_.tolist(),
        "HM": HM_.tolist(),
        "CM": CM_.tolist(),
    }

    return results, vars_out


# ============================================================
# Main entry point (same broad signature as your old main)
# ============================================================

def main(env: nn.Module,
         demand: Dict[Tuple[int, int], np.ndarray],
         real_demand: Any,
         scenarios_per_stage: int = 28,
         stages: int = 3,
         max_paths: int = 784,
         seed: int = 42,
         perfect_information: bool = False,
         warm_solution: bool = False,
         look_ahead: int = 2,
         print_results: bool = False) -> Tuple[Dict, Dict]:
    """
    Overhauled main:
      - multi_stage only
      - NA solved on scenario tree with split action/state variables
      - PI solved on deterministic realized path, no scenario tree
    """
    t_start = time.perf_counter()

    if stochastic_algorithm != "multi_stage":
        raise NotImplementedError(
            "This overhaul only implements stochastic_algorithm='multi_stage'. "
            "MPC / rolling_horizon / myopic should be rebuilt separately on top of l/y."
        )

    block_mpp = (config.env.env_name == "block_mpp")
    revenues_ = build_revenues_array(env, stages)

    if perfect_information:
        realized_path = coerce_real_demand_path(real_demand, env, stages)
        sol = solve_pi_path(
            env=env,
            demand_path=realized_path,
            stages=stages,
            revenues_=revenues_,
            block_mpp=block_mpp,
            print_results=print_results,
        )
        results, vars_out = package_results_and_vars(
            sol=sol,
            env=env,
            stages=stages,
            max_paths=max_paths,
            scenarios_per_stage=scenarios_per_stage,
            perfect_information=True,
            revenues_=revenues_,
            demand_source=realized_path,
            block_mpp=block_mpp,
        )
    else:
        sol = solve_multistage_na(
            env=env,
            demand_tree=demand,
            scenarios_per_stage=scenarios_per_stage,
            stages=stages,
            revenues_=revenues_,
            block_mpp=block_mpp,
            perfect_information=False,
            print_results=print_results,
        )
        results, vars_out = package_results_and_vars(
            sol=sol,
            env=env,
            stages=stages,
            max_paths=max_paths,
            scenarios_per_stage=scenarios_per_stage,
            perfect_information=False,
            revenues_=revenues_,
            demand_source=demand,
            block_mpp=block_mpp,
        )

    elapsed = time.perf_counter() - t_start
    results["time"] = elapsed
    results["seed"] = seed
    results["ports"] = env.P
    results["scenarios"] = scenarios_per_stage

    vars_out["seed"] = seed
    vars_out["ports"] = env.P
    vars_out["scenarios"] = scenarios_per_stage

    return results, vars_out


# ============================================================
# Script
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="mpp")
    parser.add_argument("--ports", type=int, default=4)
    parser.add_argument("--teu", type=int, default=1000)
    parser.add_argument("--deterministic", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--perfect_information", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--generalization", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--scenarios", type=int, default=40)
    parser.add_argument("--scenario_range", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--utilization_rate_initial_demand", type=float, default=1.1)
    parser.add_argument("--cv_demand", type=float, default=0.5)
    parser.add_argument("--look_ahead", type=int, default=4)
    parser.add_argument("--stochastic_algorithm", type=str, default="multi_stage")
    parser = parser.parse_args()

    # Load config
    path = f'{path_to_main}/config.yaml'
    config = load_config(path)
    env_name = parser.env_name if parser.env_name else config.env.env_name

    if env_name == "mpp":
        output_path = f"{path_to_main}/results/SMIP/navigating_uncertainty/teu1k"
    elif env_name == "block_mpp":
        output_path = f"{path_to_main}/results/SMIP/AI2STOW/teu20k"
    else:
        raise ValueError("Invalid environment name in config.yaml")

    # Update config
    config.env.ports = parser.ports
    config.env.TEU = parser.teu
    config.env.perfect_information = parser.perfect_information
    config.env.deterministic = parser.deterministic
    config.env.generalization = parser.generalization
    config.env.utilization_rate_initial_demand = parser.utilization_rate_initial_demand
    config.env.cv_demand = parser.cv_demand
    config.testing.num_episodes = parser.num_episodes

    # Params
    perfect_information = parser.perfect_information
    deterministic = parser.deterministic
    generalization = config.env.generalization
    num_episodes = config.testing.num_episodes
    stochastic_algorithm = parser.stochastic_algorithm
    look_ahead = parser.look_ahead
    scenario_range = parser.scenario_range if not generalization else False

    if stochastic_algorithm != "multi_stage":
        raise NotImplementedError(
            "This overhauled file supports only stochastic_algorithm='multi_stage'."
        )

    if deterministic:
        num_scenarios = [1]
    elif scenario_range:
        num_scenarios = [5, 10, 20, 40, 80]
    else:
        num_scenarios = [parser.scenarios]

    stages = config.env.ports - 1
    teu = config.env.TEU
    max_scenarios_per_stage = max(num_scenarios)

    if deterministic:
        max_paths = 1
    else:
        max_paths = max_scenarios_per_stage ** (stages - 1) + 1

    node_list = precompute_node_list(
        stages=stages,
        scenarios_per_stage=max_scenarios_per_stage,
        deterministic=deterministic,
        stochastic_algorithm=stochastic_algorithm
    )

    obj_list = []
    tot_demand_list = []
    max_ob_demand_list = []
    total_x_list = []
    total_ho_list = []
    total_cm_list = []
    max_revenue_list = []
    running_sum_obj = 0.0
    running_count = 0

    stochastic_algorithm_path = stochastic_algorithm + ("_pi" if perfect_information else "_na")
    if not os.path.exists(f"{output_path}/{stochastic_algorithm_path}/instances/"):
        os.makedirs(f"{output_path}/{stochastic_algorithm_path}/instances/")

    # Load realized demand benchmark file
    df = pd.read_csv(
        f"{output_path}/demand_P{config.env.ports}_gen{generalization}_UR{parser.utilization_rate_initial_demand}_cv{parser.cv_demand}.csv",
        header=None,
        index_col=False
    )
    real_demand = df.to_numpy()

    start_ep = parser.start_episode
    t = tqdm(range(start_ep, start_ep + num_episodes), desc="Episodes", unit="ep")

    for ep in t:
        seed = config.env.seed + ep + 1
        config.env.seed = seed
        set_unique_seed(seed)

        env = make_env(config.env, batch_size=[max_paths], device='cpu')

        real_demand_episode = real_demand[ep]

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
            demand_sub_tree = get_scenario_tree_indices(demand_tree, scen)

            result, var = main(
                env=env,
                demand=demand_sub_tree,
                real_demand=real_demand_episode,
                scenarios_per_stage=scen,
                stages=stages,
                max_paths=max_paths,
                seed=seed,
                perfect_information=perfect_information,
                look_ahead=look_ahead,
                print_results=False
            )

            total_x_list.append(np.sum(np.array(var.get('x', []), dtype=float)))
            total_ho_list.append(np.sum(np.array(var.get('HO', []), dtype=float)))
            total_cm_list.append(np.sum(np.array(var.get('CM', []), dtype=float)))

            demand_arr = np.array(result.get('demand', []), dtype=float)
            ob_demand = []
            ob_teus = []
            transport_indices = [(i, j) for i in range(env.P) for j in range(env.P) if i < j]

            for p in range(env.P - 1):
                ob = 0.0
                ob_teu = 0.0
                for (i, j) in onboard_groups(env.P, p, transport_indices)[0]:
                    for k in range(env.K):
                        ob += demand_arr[:, 0][i, k, j]
                        ob_teu += demand_arr[:, 0][i, k, j] * env.teus[k]
                ob_demand.append(ob)
                ob_teus.append(float(ob_teu))

            tot_demand_list.append(float(demand_arr[:, 0].sum()) if demand_arr.size > 0 else 0.0)
            max_ob_demand_list.append(max(ob_teus) if ob_teus else 0.0)
            max_revenue_list.append(float(result.get("max_revenue", 0.0)))

            obj = result.get("obj", None)
            obj_list.append(obj if obj is not None else 0.0)

            if obj is not None:
                running_sum_obj += obj
                running_count += 1
                avg_obj = running_sum_obj / running_count
                t2.set_description(f"Scenarios (avg obj={avg_obj:.2f})")

            with open(
                f"{output_path}/{stochastic_algorithm_path}/instances/"
                f"results_scenario_tree_teu{teu}_p{stages}_e{ep}_s{scen}_alg{stochastic_algorithm_path}_"
                f"pi{perfect_information}_gen{generalization}.json",
                "w"
            ) as json_file:
                json.dump(result, json_file, indent=4)

    print("==================================================")
    print(f"Type of algorithm: {stochastic_algorithm} with {num_episodes} episodes")
    print(f"Avg/Std Sum(x) {np.mean(total_x_list)}/{np.std(total_x_list)}")
    print(f"Avg/Std Sum(HO) {np.mean(total_ho_list)}/{np.std(total_ho_list)}")
    print(f"Avg/Std Sum(CM) {np.mean(total_cm_list)}/{np.std(total_cm_list)}")
    print(f"Avg/Std Obj {np.mean(obj_list)}/{np.std(obj_list)}")
    print(f"Avg/Std Total demand {np.mean(tot_demand_list)}/{np.std(tot_demand_list)}")
    print(f"Avg/Std Max onboard demand {np.mean(max_ob_demand_list)}/{np.std(max_ob_demand_list)}")
    print(f"Avg/Std Max revenue {np.mean(max_revenue_list)}/{np.std(max_revenue_list)}")