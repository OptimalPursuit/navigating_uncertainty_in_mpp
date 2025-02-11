from dotmap import DotMap
from typing import Any, Dict, Generator, Optional, Type, TypeVar, Union, Tuple
import torch as th

# Transport sets
def get_transport_idx(P: int, device) -> Union[th.Tensor,]:
    # Get above-diagonal indices of the transport matrix
    origins, destinations = th.triu_indices(P, P, offset=1, device=device)
    return th.stack((origins, destinations), dim=-1)

def get_load_pods(POD: Union[th.Tensor]):
    # Get non-zero column indices
    return (POD > 0)

def get_load_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid transports
    mask = (transport_idx[:, 0] == POL) & (transport_idx[:, 1] > POL)
    return mask

def get_discharge_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid transports
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] == POL)
    return mask

def get_on_board_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid cargo groups:
    mask = (transport_idx[:, 0] <= POL) & (transport_idx[:, 1] > POL)
    return mask

def get_not_on_board_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid cargo groups:
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] >= POL)
    return mask

def get_remain_on_board_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid transport:
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] > POL)
    return mask

def get_pols_from_transport(transport_idx, P, dtype) -> Union[th.Tensor]:
    # Get transform array from transport to POL:
    T = transport_idx.size(0)
    one_hot = th.zeros(T, P, device=transport_idx.device, dtype=dtype)
    one_hot[th.arange(T), transport_idx[:, 0].long()] = 1
    return one_hot

def get_pods_from_transport(transport_idx, P, dtype) -> Union[th.Tensor]:
    # Get transform array from transport to POD
    T = transport_idx.size(0)
    one_hot = th.zeros(T, P, device=transport_idx.device, dtype=dtype)
    one_hot[th.arange(T), transport_idx[:, 1].long()] = 1
    return one_hot

# Get step variables
def get_k_tau_pair(step, K):
    """Get the cargo class from the step number in the episode
    - step: step number in the episode
    - T: number of transports per episode
    """
    k = step % K
    tau = step // K
    return k, tau

def get_pol_pod_pair(tau, P):
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

def get_transport_from_pol_pod(pol, pod, transport_idx):
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

def compute_long_crane(utilization: th.Tensor, moves:th.Tensor, T:int) -> th.Tensor:
    """Compute long crane moves based on utilization"""
    moves_idx = moves.to(utilization.dtype).view(1, 1, 1, T, 1)
    moves_per_bay = (utilization * moves_idx).sum(dim=(-3, -2, -1))
    return moves_per_bay[..., :-1] + moves_per_bay[..., 1:]


def compute_pol_pod_locations(utilization: th.Tensor, transform_tau_to_pol, transform_tau_to_pod) -> Tuple[th.Tensor, th.Tensor]:
    """Compute POL and POD locations based on utilization"""
    if utilization.dim() == 4:
        util = utilization.permute(0, 1, 3, 2)
    elif utilization.dim() == 5:
        util = utilization.permute(0, 1, 2, 4, 3)
    else:
        raise ValueError("Utilization tensor has wrong dimensions.")
    pol_locations = (util @ transform_tau_to_pol).sum(dim=-2) != 0
    pod_locations = (util @ transform_tau_to_pod).sum(dim=-2) != 0
    return pol_locations, pod_locations

def aggregate_indices(binary_matrix, get_highest=True):
    # Shape: [bays, ports]
    bays, ports = binary_matrix.shape[-2:]

    # Create a tensor of indices [0, 1, ..., columns - 1]
    indices = th.arange(ports, device=binary_matrix.device).expand(bays, -1) + 1
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

def aggregate_pol_pod_location(pol_locations: th.Tensor, pod_locations: th.Tensor, float_type:th.dtype) -> Tuple:
    """Aggregate pol_locations and pod_locations into:
        - pod: [max(pod_d0), min(pod_d1)]
        - pol: [min(pol_d0), max(pol_d1)]"""

    ## Get load indicators - we load below deck that is blocked
    # For above deck (d=0):
    min_pol_d0 = aggregate_indices(pol_locations[..., 0, :], get_highest=False)
    #th.where(pol_locations[..., 0, :] > 0, ports + 1, 0).min(dim=-1).values
    # For below deck (d=1):
    max_pol_d1 = aggregate_indices(pol_locations[..., 1, :], get_highest=True)
    # th.where(pol_locations[..., 1, :] > 0, ports + 1, 0).max(dim=-1).values
    agg_pol_locations = th.stack((min_pol_d0, max_pol_d1), dim=-1)

    ## Get discharge indicators - we discharge below deck that is blocked
    # For above deck (d=0):
    max_pod_d0 = aggregate_indices(pod_locations[..., 0, :], get_highest=True)
    # th.where(pod_locations[..., 0, :] > 0, ports+1, 0).max(dim=-1).values
    # For below deck (d=1):
    min_pod_d1 = aggregate_indices(pod_locations[..., 0, :], get_highest=False)
    # th.where(pod_locations[..., 1, :] > 0, ports+1, 0).min(dim=-1).values
    agg_pod_locations = th.stack((max_pod_d0, min_pod_d1), dim=-1)
    # Return indicators
    return agg_pol_locations.to(float_type), agg_pod_locations.to(float_type)


def compute_hatch_overstowage(utilization: th.Tensor, moves: th.Tensor, ac_transport:th.Tensor) -> th.Tensor:
    """Get hatch overstowage based on ac_transport and moves"""
    # Compute hatch overstowage
    hatch_open = utilization[..., 1:, moves, :].sum(dim=(-3, -2, -1)) > 0
    hatch_overstowage = utilization[..., :1, ac_transport, :].sum(dim=(-3, -2, -1)) * hatch_open
    return hatch_overstowage

def compute_min_pod(pod_locations: th.Tensor, P:int, dtype:th.dtype) -> th.Tensor:
    """Compute min_pod based on utilization"""
    min_pod = th.argmax(pod_locations.to(dtype), dim=-1)
    min_pod[min_pod == 0] = P
    return min_pod

def compute_HO_mask(mask:th.Tensor, pod: th.Tensor,pod_locations:th.Tensor, min_pod:th.Tensor, B:int, D:int) -> th.Tensor:
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
    mask = mask.view(-1, B, D)
    # Action below-deck (d=1) allowed if above-deck (d=0) is empty
    mask[..., 1] = pod_locations[..., 0, :].sum(dim=-1) == 0
    # Action above-deck (d=0) allowed if POD <= min_pod below deck (d=1)
    mask[..., 0] = pod.unsqueeze(-1) <= min_pod[..., 1]
    return mask.view(-1, B*D)

def compute_violation(action, lhs_A, rhs, ) -> th.Tensor:
    """Compute violations and loss of compact form"""
    # If dimension lhs_A is one more than action, unsqueeze action
    if (lhs_A.dim() - action.dim()) == 1:
        action = action.unsqueeze(-2)
    lhs = (lhs_A * action).sum(dim=(-1))
    output = th.clamp(lhs-rhs, min=0)
    return output

if __name__ == "__main__":
    # Test the transport sets
    print(get_pol_pod_pair(tau=th.tensor(7), P=th.tensor(5)))
