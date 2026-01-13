from dotmap import DotMap
from typing import Any, Dict, Generator, Optional, Type, TypeVar, Union, Tuple
import torch as th
from tensordict import TensorDict

# Transport sets
def get_transport_idx(P: int, device:Union[th.device,str]) -> Union[th.Tensor,]:
    # Get above-diagonal indices of the transport matrix
    origins, destinations = th.triu_indices(P, P, offset=1, device=device)
    return th.stack((origins, destinations), dim=-1)

def get_load_pods(POD: Union[th.Tensor]) -> Union[th.Tensor]:
    # Get non-zero column indices
    return (POD > 0)

def get_load_transport(transport_idx:th.Tensor, POL:th.Tensor) -> Union[th.Tensor]:
    # Boolean mask for valid transports
    mask = (transport_idx[:, 0] == POL) & (transport_idx[:, 1] > POL)
    return mask

def get_discharge_transport(transport_idx:th.Tensor, POL:th.Tensor) -> Union[th.Tensor]:
    # Boolean mask for valid transports
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] == POL)
    return mask

def get_on_board_transport(transport_idx:th.Tensor, POL:th.Tensor) -> Union[th.Tensor]:
    # Boolean mask for valid cargo groups:
    mask = (transport_idx[:, 0] <= POL) & (transport_idx[:, 1] > POL)
    return mask

def get_not_on_board_transport(transport_idx:th.Tensor, POL:th.Tensor) -> Union[th.Tensor]:
    # Boolean mask for valid cargo groups:
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] >= POL)
    return mask

def get_remain_on_board_transport(transport_idx:th.Tensor, POL:th.Tensor) -> Union[th.Tensor]:
    # Boolean mask for valid transport:
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] > POL)
    return mask

def get_pols_from_transport(transport_idx:th.Tensor, P:int, dtype:th.dtype) -> Union[th.Tensor]:
    # Get transform array from transport to POL:
    T = transport_idx.size(0)
    one_hot = th.zeros(T, P, device=transport_idx.device, dtype=dtype)
    one_hot[th.arange(T), transport_idx[:, 0].long()] = 1
    return one_hot

def get_pods_from_transport(transport_idx:th.Tensor, P:int, dtype:th.dtype) -> Union[th.Tensor]:
    # Get transform array from transport to POD
    T = transport_idx.size(0)
    one_hot = th.zeros(T, P, device=transport_idx.device, dtype=dtype)
    one_hot[th.arange(T), transport_idx[:, 1].long()] = 1
    return one_hot

# Get step variables
def get_k_tau_pair(step:th.Tensor, K:int) -> Tuple[th.Tensor, th.Tensor]:
    """Get the cargo class from the step number in the episode
    - step: step number in the episode
    - T: number of transports per episode
    """
    k = step % K
    tau = step // K
    return k, tau

def get_pol_pod_pair(tau:th.Tensor, P:th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """Get the origin-destination (pol,y) pair of the transport with index i
    - i: index of the transport
    - P: number of ports
    - pol: origin
    - pod: destination
    """
    # Calculate pol using the inverse of triangular number formula
    ## todo: check if this is formulation is correct for P!=4. (empirically it seems to work)
    pol = P - 2 - th.floor(th.sqrt(2*(P*(P-1)//2 - 1 - tau) + 0.25) - 0.5).to(th.int64)
    # Calculate y based on pol
    pod = tau - (P*(P-1)//2 - (P-pol)*(P-pol-1)//2) + pol + 1
    return pol, pod

def get_transport_from_pol_pod(pol:th.Tensor, pod:th.Tensor, transport_idx:th.Tensor) -> th.Tensor:
    """Get the transport index from the origin-destination pair
    - pol: origin
    - pod: destination
    - transport_idx: transport tensor to look up row that matches the origin-destination pair
    """
    # Find rows where both the first column is `pol` and the second column is `pod`
    mask = (transport_idx[:, 0].unsqueeze(1) == pol) & (transport_idx[:, 1].unsqueeze(1) == pod)
    # Use th.where to get the indices where the mask is True
    output = th.where(mask)[0] # [0] extracts the first dimension (row indices)

    # Check if the output is empty
    if output.numel() == 0:
        return th.tensor([0], device=transport_idx.device)

    return output

# States
def update_state_discharge(utilization:th.Tensor, disc_idx:th.Tensor,) -> th.Tensor:
    """Update state as result of discharge"""
    utilization[..., disc_idx, :] = 0.0
    return utilization

def update_state_loading(action: th.Tensor, utilization: th.Tensor, tau:th.Tensor, k:th.Tensor,) -> th.Tensor:
    """Transition to the next state based on the action."""
    new_utilization = utilization.clone()
    new_utilization[..., tau, k] = action
    return new_utilization

def compute_stability(utilization: th.Tensor, weights: th.Tensor, longitudinal_position: th.Tensor,
                      vertical_position: th.Tensor, block=False) -> Tuple[th.Tensor, th.Tensor]:
    """Compute the LCG and VCG based on utilization, weights, longitudinal and vertical position"""
    # Dynamically determine sum_dim and shape based on number of dimensions
    sum_dims = tuple(range(-3, 0)) if block else tuple(range(-2, 0))
    u_dims = utilization.dim()
    w_shape = (1,) * (u_dims - 1) + (-1,)
    location_weight = (utilization * weights.view(w_shape)).sum(dim=(-2, -1))
    # Get shapes
    lw_dims = location_weight.dim()
    if lw_dims < 2:
        raise ValueError("lw_dims must be at least 2.")
    lp_shape = [1] * lw_dims
    vp_shape = [1] * lw_dims
    axis = 0 if (block and lw_dims == 3) or (not block and lw_dims == 2) else 1
    lp_shape[axis] = -1
    vp_shape[axis + 1] = -1
    lp_shape, vp_shape = tuple(lp_shape), tuple(vp_shape)

    # Compute LCG and VCG; if total_weight is zero make lcg and vcg equal to 1
    total_weight = location_weight.sum(dim=(sum_dims))
    lcg = th.where(
        total_weight > 0,
        (location_weight * longitudinal_position.view(lp_shape)).sum(dim=(sum_dims)) / total_weight,
        th.ones_like(total_weight)
    )
    vcg = th.where(
        total_weight > 0,
        (location_weight * vertical_position.view(vp_shape)).sum(dim=(sum_dims)) / total_weight,
        th.ones_like(total_weight)
    )
    return lcg, vcg

def compute_target_long_crane(realized_demand: th.Tensor, moves: th.Tensor,
                              capacity:th.Tensor, B:int, CI_target:float) -> th.Tensor:
    """Compute target crane moves per port:
    - Get total crane moves per port: load_moves + discharge_moves
    - Get optimal crane moves per adjacent bay by: 2 * total crane moves / B
    - Get adjacent capacity by: sum of capacity of adjacent bays
    - Get max capacity of adjacent bays by: max of adjacent capacity

    Return element-wise minimum of optimal crane moves and max capacity"""
    # Calculate optimal crane moves based per adjacent bay based on loading and discharging
    total_crane_moves = realized_demand[..., moves, :].sum(dim=(-1,-2))
    # Compute adjacent capacity and max capacity
    max_capacity = ((capacity[:-1] + capacity[1:]).sum(dim=-1)).max()
    # Compute element-wise minimum of crane moves and target long crane
    optimal_crane_moves_per_adj_bay = 2 * total_crane_moves / B
    return CI_target * th.minimum(optimal_crane_moves_per_adj_bay, max_capacity)

def compute_long_crane(utilization: th.Tensor, moves: th.Tensor, T: int, block:bool=False) -> th.Tensor:
    """Compute long crane moves based on utilization, automatically handling both standard and block environments."""
    # Dynamically determine sum_dim and shape based on number of dimensions
    dims = utilization.dim()
    moves_shape = (1,) * (dims - 2) + (T, 1)
    sum_dims = tuple(range(-4, 0)) if block else tuple(range(-3, 0))
    # Compute moves per bay and long crane moves
    moves_idx = moves.to(utilization.dtype).view(moves_shape)
    moves_per_bay = (utilization * moves_idx).sum(dim=sum_dims)
    return moves_per_bay[..., :-1] + moves_per_bay[..., 1:]

def compute_long_crane_excess_cost(lc_moves:th.Tensor, target_long_crane:th.Tensor, cm_costs:th.Tensor) -> th.Tensor:
    """Computes the crane excess cost  """
    lc_excess = th.clamp(lc_moves - target_long_crane.view(-1, 1), min=0)
    return lc_excess.sum(dim=-1, keepdim=True) * cm_costs

def compute_pol_pod_locations(utilization: th.Tensor, transform_tau_to_pol, transform_tau_to_pod, eps:float=1e-2,
                              differentiable:bool=False) -> Tuple[th.Tensor, th.Tensor]:
    """Compute POL and POD locations based on utilization"""
    util = utilization.transpose(-1, -2)
    if util.dim() < 4 or util.dim() > 7:
        raise ValueError("Utilization tensor has wrong dimensions.")

    trans_pol_util = util @ transform_tau_to_pol
    trans_pod_util = util @ transform_tau_to_pod

    if differentiable:
        pol_locations = (trans_pol_util - eps).sum(dim=-2).clamp(min=0.0, max=1.0)
        pod_locations = (trans_pod_util - eps).sum(dim=-2).clamp(min=0.0, max=1.0)
    else:
        pol_locations = (trans_pol_util).sum(dim=-2) > eps
        pod_locations = (trans_pod_util).sum(dim=-2) > eps
    return pol_locations, pod_locations

def generate_POD_mask(
    tr_demand_teu: th.Tensor,
    residual_capacity: th.Tensor,     # [B,D,BL] or [*batch,B,D,BL]
    capacity: th.Tensor,              # [B,D,BL] or [*batch,B,D,BL]
    pod_locations: th.Tensor,          # [B,D,BL,P] or [*batch,B,D,BL,P] (indicator or amount)
    pod: int,
    batch_size: tuple = (),
    *,
    POD_mix_in_block: float = 0.0,     # allowed noncurrent share in a block
    oversubscribe: float = 1.2,        # open ~20% extra blocks beyond minimum needed to cover demand
    eps: float = 1e-9,
) -> th.Tensor:
    """
    Action mask for placing current POD into [Bay,Deck,Block]. Output is flattened to [*batch, B*D*BL].
    It creates a mask that prevents POD violations, but also controls opening of new blocks based on demand.

    Rules:
      (1) Slot must have space: residual_capacity > 0.
      (2) Block POD rule (deck-collapsed):
            - empty block is compatible (but may be blocked by opening rule),
            - non-empty block is compatible only if it already contains current POD and
              noncurrent share in that block <= POD_mix_in_block.
      (3) Opening rule for empty blocks: among empty+openable blocks, select enough blocks to cover near-term demand.
    """
    # --- shapes / batch handling ---
    B, D, BL, P = pod_locations.shape[-4:]
    device = pod_locations.device
    if not (0 <= pod < P):
        raise ValueError(f"pod={pod} out of range for P={P}")

    has_batch = (pod_locations.dim() > 4)
    batch_dims = pod_locations.shape[:-4] if has_batch else batch_size

    def _expand_to_batch(x: th.Tensor, target_shape: tuple) -> th.Tensor:
        # If unbatched, add a leading batch dim and expand; otherwise rely on expand/broadcast.
        if x.shape == target_shape:
            return x
        if x.shape == target_shape[-len(x.shape):]:
            return x.unsqueeze(0).expand(*target_shape)
        return x.expand(*target_shape)

    bc_shape = (*batch_dims, B, D, BL)
    pl_shape = (*batch_dims, B, D, BL, P)
    residual_capacity = _expand_to_batch(residual_capacity, bc_shape)
    capacity = _expand_to_batch(capacity, bc_shape)
    pod_locations = _expand_to_batch(pod_locations, pl_shape)

    # --- (1) per-location feasibility ---
    loc_has_space = residual_capacity > 0  # [*batch,B,D,BL]

    # --- block-level POD composition (sum over decks) ---
    # If pod_locations is binary, these are "location counts"; if amount-valued, they are "amount totals".
    pod_count_block = pod_locations.sum(dim=-3)                 # [*batch,B,BL,P]
    total_count_block = pod_count_block.sum(dim=-1)             # [*batch,B,BL]
    cur_count_block = pod_count_block[..., pod]                 # [*batch,B,BL]
    noncur_count_block = total_count_block - cur_count_block    # [*batch,B,BL]

    # Block occupancy flags
    block_empty = total_count_block <= 0                        # no POD present in block
    block_has_cur = cur_count_block > 0                         # current POD present in block

    # Noncurrent share among what is already in the block (safe due to eps)
    noncur_share_block = noncur_count_block / (total_count_block + eps)
    noncur_share_ok = noncur_share_block <= (POD_mix_in_block + 1e-7)

    # --- (2) block compatibility rule for current POD ---
    block_allowed_for_pod = block_empty | (block_has_cur & noncur_share_ok)  # [*batch,B,BL]

    # Expand block rules to every bay-deck inside the block
    block_empty_bd = block_empty.unsqueeze(-2).expand(*batch_dims, B, D, BL)
    block_has_cur_bd = block_has_cur.unsqueeze(-2).expand(*batch_dims, B, D, BL)
    block_allowed_bd = block_allowed_for_pod.unsqueeze(-2).expand(*batch_dims, B, D, BL)

    # --- (3) opening rule: allow only a subset of empty blocks to be used this step ---
    # Demand capped by total remaining residual space
    total_residual_teu = residual_capacity.sum(dim=(-1, -2, -3))          # [*batch]
    demand_to_cover_teu = th.minimum(total_residual_teu, tr_demand_teu)   # [*batch]

    # Per-block residual/capacity across decks (block is the unit that gets "opened")
    block_residual_teu = residual_capacity.sum(dim=-2)  # [*batch,B,BL]
    block_capacity_teu = capacity.sum(dim=-2)           # [*batch,B,BL]

    # A block can be opened only if it's empty and has some residual capacity
    block_openable = block_empty & (block_residual_teu > 0)  # [*batch,B,BL]

    # Random mirrored scores across bays (symmetry); non-openable blocks get score 0.
    half_B = B // 2 + (B % 2)
    rand_half = th.rand((*batch_dims, half_B, BL), device=device)
    rand_full = th.cat([rand_half, rand_half.flip(dims=[-2])], dim=-2)
    rand_full = rand_full * block_openable.to(rand_full.dtype)

    # Select top-scoring blocks until cumulative capacity covers demand (then oversubscribe)
    scores_flat = rand_full.reshape(*batch_dims, -1)                    # [*batch,B*BL]
    sorted_idx = scores_flat.argsort(dim=-1, descending=True)
    # Gather capacities in sorted order and compute cumulative capacity
    cap_flat = block_capacity_teu.reshape(*batch_dims, -1)
    sorted_caps = th.gather(cap_flat, dim=-1, index=sorted_idx)
    csum = sorted_caps.cumsum(dim=-1)
    # Find minimal k with cumulative capacity >= demand_to_cover_teu
    enough = csum >= demand_to_cover_teu.unsqueeze(-1)
    any_enough = enough.any(dim=-1)
    first_enough = enough.int().argmax(dim=-1)
    k_min = th.where(any_enough, first_enough + 1, th.full_like(first_enough, scores_flat.shape[-1]))

    # If demand is zero, open nothing; else open ceil(k_min * oversubscribe) blocks.
    k = th.where(
        demand_to_cover_teu > 0,
        th.ceil(k_min.to(th.float32) * oversubscribe).to(th.long),
        th.zeros_like(k_min),
    )
    # Build selection mask over flattened blocks: select top-k in sorted order
    ar = th.arange(scores_flat.shape[-1], device=device).expand(*batch_dims, -1)
    topk_mask = ar < k.unsqueeze(-1)
    # Scatter to get final selection mask over flattened blocks
    open_sel_flat = th.zeros_like(scores_flat, dtype=th.bool)
    open_sel_flat.scatter_(-1, sorted_idx, topk_mask)
    # Reshape selection back to [*batch,B,BL] and expand over decks
    open_selected_block = open_sel_flat.view(*batch_dims, B, BL)                 # [*batch,B,BL]
    open_selected_bd = open_selected_block.unsqueeze(-2).expand(*batch_dims, B, D, BL)

    # --- final availability: must have space AND satisfy block rule AND (existing or opened-empty) ---
    # Final output mask
    allow_in_new_block = block_empty_bd & open_selected_bd
    out = loc_has_space & block_allowed_bd & (block_has_cur_bd | allow_in_new_block)
    return out.reshape(*batch_dims, -1)

def compute_targets_to_prevent_POD_violations(
    residual_capacity: th.Tensor,   # [batch_size,B,D,BL]
    pod_locations: th.Tensor,        # [batch_size,B,D,BL,P] (binary or amounts)
    pod: th.Tensor,                  # [batch_size] int64 POD id per sample
    B: int,
    *,
    D: int = 2,
    POD_mix_in_block: float = 0.0,
    delta: float = 1e-3,
    eps: float = 1e-7,
) -> th.Tensor:
    """
    Target y for all valid actions for paired block stowage (PBS) with control over POD mix in blocks.
    It essentially gives a mask that prevents POD violations.

    PBS is enforced by:
      - Available TEU capacity:     residual_capacity >= delta
      - Block available for POD:    block empty OR
                                    (block with current POD AND noncurrent POD share <= allowed POD mix in blocks)
    Returns:
      y: [batch_size,B,D,BL] bool
    """
    # Dimensions
    batch_size, action_shape = residual_capacity.shape[0], residual_capacity.shape[-1]
    steps = residual_capacity.shape[1] if residual_capacity.dim() >= 3 else 1
    BL = action_shape // (B * D)
    P = pod_locations.shape[-1] // action_shape

    # Resize inputs
    residual_capacity = residual_capacity.view(batch_size, steps, B, D, BL)
    pod_locations = pod_locations.view(batch_size, steps, B, D, BL, P)

    # Collapse decks -> how many PODs are in each (bay, block)
    pod_count_block = pod_locations.sum(dim=-3)           # [batch_size,B,BL,P]
    total_count = pod_count_block.sum(dim=-1)              # [batch_size,B,BL] total PODs in blocks

    # Current-POD count in each block
    pod_idx = pod.view(batch_size, steps, 1, 1, 1).expand(batch_size, steps, B, BL, 1)
    cur_count = th.gather(pod_count_block, dim=-1, index=pod_idx).squeeze(-1)  # [batch_size,B,BL]

    # Split block cargo into "current POD" vs "noncurrent PODs"
    noncur_count = total_count - cur_count
    block_empty = total_count <= 0                        # no cargo in block
    block_has_cur = cur_count > 0                         # block already contains current POD

    # Expand block-level quantities back to [B,D,BL]
    total_bd = total_count.unsqueeze(-2).expand(batch_size, steps, B, D, BL)
    noncur_bd = noncur_count.unsqueeze(-2).expand(batch_size, steps, B, D, BL)
    empty_bd = block_empty.unsqueeze(-2).expand(batch_size, steps, B, D, BL)
    has_cur_bd = block_has_cur.unsqueeze(-2).expand(batch_size, steps, B, D, BL)

    # Compute noncurrent POD share after placing `delta` of current POD, then check against allowed mix
    noncur_share = noncur_bd / (total_bd + delta + eps)
    mix_allowed_post = noncur_share <= (POD_mix_in_block + eps)

    # There must be at least `delta` free capacity in that location
    has_space = residual_capacity >= delta

    # Final label per action/location (bay, deck, block)
    y = has_space & (empty_bd | (has_cur_bd & mix_allowed_post))
    return y


def aggregate_indices(binary_matrix:th.Tensor, get_highest:bool=True) -> th.Tensor:
    # Shape: [bays, ports]
    bays, ports = binary_matrix.shape[-2:]

    # Create a tensor of indices [0, 1, ..., columns - 1]
    indices = th.arange(ports, device=binary_matrix.device).expand(bays, -1)
    if get_highest:
        # Find the highest True index
        # Reverse the indices and binary matrix along the last dimension
        reversed_indices = th.flip(indices, dims=[-1])
        reversed_binary = th.flip(binary_matrix, dims=[-1])

        # Get the highest index where the value is True (1)
        highest_indices = th.where(reversed_binary.bool(), reversed_indices, 0)
        result = highest_indices.max(dim=-1).values
    else:
        # Find the lowest True index
        lowest_indices = th.where(binary_matrix.bool(), indices, th.inf)
        result = lowest_indices.min(dim=-1).values
        result[result==th.inf] = 0

    return result

def aggregate_pol_pod_location(pol_locations: th.Tensor, pod_locations: th.Tensor, float_type:th.dtype,
                               block:bool=True) -> Tuple:
    """Aggregate pol_locations and pod_locations into:
        - pod: [max(pod_d0), min(pod_d1)]
        - pol: [min(pol_d0), max(pol_d1)]"""

    ## Get load indicators - we load below deck that is blocked
    # For above deck (d=0):
    if block:
        min_pol_d0_idx = (..., 0, slice(None), slice(None))
        max_pol_d1_idx = (..., 1, slice(None), slice(None))
        max_pod_d0_idx = (..., 0, slice(None), slice(None))
        min_pod_d1_idx = (..., 1, slice(None), slice(None))
        agg_dim = -2
    else:
        min_pol_d0_idx = (..., 0, slice(None))
        max_pol_d1_idx = (..., 1, slice(None))
        max_pod_d0_idx = (..., 0, slice(None))
        min_pod_d1_idx = (..., 1, slice(None))
        agg_dim = -1

    min_pol_d0 = aggregate_indices(pol_locations[min_pol_d0_idx], get_highest=False)
    #th.where(pol_locations[..., 0, :] > 0, ports + 1, 0).min(dim=-1).values
    # For below deck (d=1):
    max_pol_d1 = aggregate_indices(pol_locations[max_pol_d1_idx], get_highest=True)
    # th.where(pol_locations[..., 1, :] > 0, ports + 1, 0).max(dim=-1).values
    agg_pol_locations = th.stack((min_pol_d0, max_pol_d1), dim=agg_dim)

    ## Get discharge indicators - we discharge below deck that is blocked
    # For above deck (d=0):
    max_pod_d0 = aggregate_indices(pod_locations[max_pod_d0_idx], get_highest=True)
    # th.where(pod_locations[..., 0, :] > 0, ports+1, 0).max(dim=-1).values
    # For below deck (d=1):
    min_pod_d1 = aggregate_indices(pod_locations[min_pod_d1_idx], get_highest=False)
    # th.where(pod_locations[..., 1, :] > 0, ports+1, 0).min(dim=-1).values
    agg_pod_locations = th.stack((max_pod_d0, min_pod_d1), dim=agg_dim)
    # Return indicators
    return agg_pol_locations.to(float_type), agg_pod_locations.to(float_type)

def compute_hatch_overstowage(utilization: th.Tensor, moves: th.Tensor, ac_transport:th.Tensor, block=False) -> th.Tensor:
    """Get hatch overstowage based on ac_transport and moves"""
    # Dynamic dependence of dims, sum_dims and indices
    if block:
        sum_dims = tuple(range(-4, 0))
        index_hatch_open = (..., slice(1, None), slice(None), moves, slice(None))
        index_hatch_overstowage = (..., slice(None, 1), slice(None), ac_transport, slice(None))
    else:
        sum_dims = tuple(range(-3, 0))
        index_hatch_open = (..., slice(1, None), moves, slice(None))
        index_hatch_overstowage = (..., slice(None, 1), ac_transport, slice(None))

    # Compute hatch overstowage
    hatch_open = utilization[index_hatch_open].sum(dim=sum_dims) > 0
    return utilization[index_hatch_overstowage].sum(dim=sum_dims) * hatch_open

def compute_min_pod(pod_locations: th.Tensor, P:int, dtype:th.dtype) -> th.Tensor:
    """Compute min_pod based on utilization"""
    min_pod = th.argmax(pod_locations.to(dtype), dim=-1)
    min_pod[min_pod == 0] = P
    return min_pod

def compute_HO_mask(mask:th.Tensor, pod: th.Tensor,pod_locations:th.Tensor, min_pod:th.Tensor) -> th.Tensor:
    """
    Mask action to prevent hatch overstowage. Deck indices: 0 is above-deck, 1 is below-deck.

    Variables:
        - Utilization: Current state of onboard cargo (bay,deck,cargo_class,transport)
        - POD_locations: Indicator to show PODs loaded in locations (bay,deck,P)
        - Min_pod: Minimum POD location based on POD_locations (bay,deck)

    Utilization is filled/emptied incrementally. Hence, we have certain circumstances to observe utilization:
        - Step after reset: Utilization is empty
        - Step of new POL:  Discharge utilization destined for new POL
        - Any other step:   Load utilization of current cargo_class and transport

    Two ways to prevent hatch overstowage:
    - If above-deck is empty, we can freely place below-deck. Otherwise, we need to restow above-deck directly.
        E.g.:
                | 3 | 3 | o |
                +---+---+---+
                | x | x | o |   , where int is min_pod of location, x is blocked location, o is open location

    - Above-deck actions are allowed if current POD <= min_pod below-deck. Otherwise, we need to restow
        above-deck when below-deck will be discharged.
        E.g.:   POD = 2
                | x | o | o |
                +---+---+---+
                | 1 | 2 | 3 |   , where int is min_pod of location, x is blocked location, o is open location
    """
    # Create mask:
    mask = mask.view(min_pod.shape)
    # Action below-deck (d=1) allowed if above-deck (d=0) is empty
    mask[..., 1, :] = pod_locations[..., 0, :, :].sum(dim=-1) == 0
    # Action above-deck (d=0) allowed if POD <= min_pod below deck (d=1)
    mask[..., 0, :] = pod.unsqueeze(-1) <= min_pod[..., 1, :]
    return mask

def compute_strict_BS_mask(pod:th.Tensor, pod_locations:th.Tensor,) -> th.Tensor:
    """
    Mask actions to enforce strict block stowage: only a single POD per block.

    Conditional:
    - If pod is X and pod_location is empty, then True
    - If pod and pod_location are the exclusive, then True
    - If pod and pod_location are different, then False
    """
    # Get number of pods per block
    pod_block = pod_locations.any(dim=-3).sum(dim=(-1))
    # Set to true if pod is empty or exclusive
    is_empty = pod_block == 0
    is_exclusive = (pod_block == 1) & pod_locations.any(dim=-3)[..., pod]
    return is_empty | is_exclusive

def compute_violation(action:th.Tensor, lhs_A:th.Tensor, rhs:th.Tensor, ) -> th.Tensor:
    """Compute violations and loss of compact form"""
    # If dimension lhs_A is one more than action, unsqueeze action
    if (lhs_A.dim() - action.dim()) == 1:
        action = action.unsqueeze(-2)
    lhs = (lhs_A * action).sum(dim=(-1))
    output = th.clamp(lhs-rhs, min=0)
    return output

def compute_POD_violation(vessel_state:Dict, transform_tau_to_pol:th.Tensor, transform_tau_to_pod:th.Tensor, float_type:th.float) -> Dict:
    vessel_state["pol_locations"], vessel_state["pod_locations"] = \
        compute_pol_pod_locations(vessel_state["utilization"], transform_tau_to_pol, transform_tau_to_pod)
    vessel_state["agg_pol_location"], vessel_state["agg_pod_location"] = \
        aggregate_pol_pod_location(vessel_state["pol_locations"], vessel_state["pod_locations"], float_type, block=True)

    # Compute unique number of pods at each bay,block
    vessel_state["excess_pod_locations"] = th.clamp((vessel_state["pod_locations"].sum(dim=-3) > 0).sum(dim=-1) - 1, min=0.0)
    return vessel_state

def flatten_values_td(td: TensorDict, batch_size:Tuple[int, ...]) -> TensorDict:
    return td.apply(lambda x: x.view(*batch_size, -1))

def inspect_tensordict(td, prefix=""):
    for key, value in td.items():
        if isinstance(value, TensorDict):
            print(f"{prefix}TensorDict: {key}")
            inspect_tensordict(value, prefix + "  ")
        else:
            try:
                shape = value.shape
                dtype = value.dtype
                mean = value.float().mean().item()
                print(f"{prefix}Key: {key}, Shape: {shape}, Dtype: {dtype}, Mean: {mean}")
            except Exception as e:
                print(f"{prefix}Key: {key}, <uninspectable>: {e}")

if __name__ == "__main__":
    # Test the transport sets
    print(get_pol_pod_pair(tau=th.tensor(7), P=th.tensor(5)))
