# Imports
import torch as th
from typing import Tuple, Optional, Union
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator

# todo: improvements:
"""
- Now voyages are mixed, but make also voyage with only load / and only discharge ports
- Add reefers
- No customer contracts
"""


# Classes
class MPP_Generator(Generator):
    """
    Demand generator for container vessel stowage planning:
    - Generates demand with variable size: ports, customer classes, cargo classes, weight classes
    - iid demand samples from any distribution
    - moments based on available capacity, expected utilization rate, and demand variability
    - supports spot and longterm contracts with different demand variability
    """
    # todo: initialize with config dict for kwargs
    def __init__(self, device="cuda", **kwargs):
        """
        Initialize the MPP_Generator.

        Args:
            device (str): Device to run computations on ('cuda' or 'cpu').
            **kwargs: Configuration parameters for the generator:
                - ports (int): Number of ports (P).
                - bays (int): Number of bays (B).
                - decks (int): Number of decks (D).
                - customer_classes (int): Number of customer classes (CC).
                - cargo_classes (int): Number of cargo classes per customer class.
                - weight_classes (int): Number of weight classes (W).
                - capacity (list or int): Capacity per bay and deck.
                - utilization_rate_initial_demand (float): Initial demand utilization rate.
                - spot_percentage (float): Percentage of spot contracts.
                - iid_demand (bool): If True, demand is IID; otherwise, correlated.
                - cv_demand (float): Coefficient of variation for demand.
                - perturbation (float): Perturbation factor for demand variability.

        """
        super().__init__(**kwargs)
        # Input simulation
        self.device = th.device(device)
        self.seed = kwargs.get("seed")
        self.rng = th.Generator(device=self.device).manual_seed(self.seed)

        # Configuration
        self.P = kwargs.get("ports")
        self.B = kwargs.get("bays")
        self.D = kwargs.get("decks")
        self.CC = kwargs.get("customer_classes")
        self.K = kwargs.get("cargo_classes") * self.CC
        self.W = kwargs.get("weight_classes")
        c = kwargs.get("capacity")
        self.c = th.full((self.B, self.D,), c[0]) if len(c) == 1 else c
        self.total_capacity = th.sum(self.c)

        # Derived values
        self.T = int((self.P ** 2 - self.P) / 2)
        self.teus = th.arange(1, self.K // (self.CC * self.W) + 1, dtype=th.float16, device=self.device)\
            .repeat_interleave(self.W).repeat(self.CC)

        # Demand control
        self.utilization_rate_initial_demand = kwargs.get("utilization_rate_initial_demand", 1.2)
        self.spot_percentage = kwargs.get("spot_percentage", 0.3)
        self.spot_lc_percentage = th.cat([
            th.full((self.K // 2,), 2 * (1 - self.spot_percentage), device=self.device),
            th.full((self.K // 2,), 2 * self.spot_percentage, device=self.device)
        ])
        self.iid_demand = kwargs.get("iid_demand", True)
        self.cv_demand = kwargs.get("cv_demand", 1.5)
        self.perturbation = kwargs.get("perturbation", 0.1)

        # Precomputations:
        # Demand variability
        self.cv = th.empty((self.K,), device=self.device, dtype=th.float16)
        self.cv[self.K // 2:] = self.cv_demand * (2/3)      # Less variability for longterm contracts
        self.cv[:self.K // 2] = self.cv_demand              # Higher variability for spot contracts
        # Wave for initial demand
        self.wave = self._create_wave(self.P - 1)
        # Transport indices and related logic # todo: check if this can be optimized/simplified
        self.transport_idx = self.get_transport_idx(self.P, device=self.device)
        self.num_loads = self._get_num_loads_in_voyage(self.transport_idx, self.P)
        self.num_discharges = self.num_loads.flip(0)
        self.tr_wave = th.repeat_interleave(self.wave, self.num_loads)
        self.tr_loads = th.repeat_interleave(self.num_loads, self.num_loads)
        self.tr_discharges = th.repeat_interleave(self.num_discharges, self.num_loads)
        POL = th.arange(self.P, device=self.device).unsqueeze(1).unsqueeze(1)
        self.num_ac = self._get_num_AC_in_voyage(self.transport_idx, POL)
        self.num_ob = self._get_num_OB_in_voyage(self.transport_idx, POL)
        self.tr_ac = th.repeat_interleave(self.num_ac, self.num_loads)
        self.tr_ob = th.repeat_interleave(self.num_ob, self.num_loads)

    def __call__(self, batch_size:Tuple[int, ...], td: Optional[TensorDict] = None, rng:Optional=None) -> TensorDict:
        """Generate demand for the MPP."""
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size, td)

    ## Generate demand
    def _iid_normal_distribution(self, e_x: th.Tensor, std_x: th.Tensor,) -> Tuple[th.Tensor, th.Tensor, th.distributions.Normal]:
        """Get normal distribution for demand"""
        return e_x, std_x, th.distributions.Normal(loc=e_x, scale=std_x)

    def _generate(self, batch_size:Tuple[int, ...], td:Optional=None,) -> TensorDict:
        """Generate demand matrix for voyage"""
        # Get moments and distribution
        e_x_init_demand, std_x = self._generate_moments(batch_size, self._initialize_demand_ub(batch_size), self.cv)
        e_x, std_x, dist = self._iid_normal_distribution(e_x_init_demand, std_x,)

        # Sample demand with lower bound of 0
        demand = th.clamp(dist.sample(), min=0)

        # Return demand matrix
        return TensorDict({"observation":
                               {"realized_demand": demand.view(*batch_size, self.T*self.K),
                                "expected_demand": e_x.view(*batch_size, self.T*self.K),
                                "std_demand":std_x.view(*batch_size, self.T*self.K),
                                }}, batch_size=batch_size, device=self.device,)

    ## Initial demand
    def _initialize_demand_ub(self, batch_size:Tuple[int, ...],) -> th.Tensor:
        """Get initial demand upper bound based on capacity

        Bound without wave:
        f(x) = (2*utilization_rate * th.sum(self.c)) / (self.K *  num_load (1-num_ac/num_ob))

        This computes the bound for each (K,T) pair assuming full capacity utilization. In reality, however,
        the utilization rate is only approaching 80-90% of the total capacity at the middle of the voyage. Hence,
        we introduce a wave parameter to increase the bound until the middle of the voyage and decrease it afterward.
        Note: 2*utilization_rate is used to account for the fact this is an upper bound of uniform distribution.

        Bound with wave:
        g(x) = wave * (2*utilization_rate * th.sum(self.c)) / (self.K *  num_load (1-num_ac/num_ob))

        Note that if wave is added, then capacity used in previous steps is underestimated with wave < 1. Similarly,
        if wave >1 then capacity used in previous steps is overestimated. Nonetheless, this bound used for an initial
        E[X] and V[X], hence this is acceptable if we tune wave parameters to obtain sufficient levels of utilization.
        """
        # Get transport bound with wave
        utilization_bound = self.tr_wave * 2 * self.utilization_rate_initial_demand * self.total_capacity
        num_cargo = (self.K * self.tr_loads / (1 - self.tr_ac / self.tr_ob))
        # todo: this one is a bit complictated: need to check if this is correct

        utilization_bound /= num_cargo
        # Get bound with spot, lc using self.spot_lc_percentage
        return th.ger(utilization_bound, self.spot_lc_percentage) / self.teus.view(1, -1,)

    def _random_perturbation(self, input:th.Tensor, perturb_factor=0.1) -> th.Tensor:
        """Apply a random perturbation to bound while keeping it close to the original value."""
        perturbation = 1 + (th.rand_like(input) * 2 - 1) * perturb_factor  # U(1-α, 1+α)
        return th.clamp(input * perturbation, min=1.0)  # Ensure positive output

    def _generate_moments(self, batch_size:Tuple[int, ...], bound:th.Tensor, cv:th.Tensor, eps:float=1e-2) -> Tuple[th.Tensor, th.Tensor]:
        """Generate initial E[X] and Std[X] for spot and longterm contracts.
        - E[X] = Uniform sample * bound
        - Std[X] = (E[X] * cv)"""
        # shape of bound: [batch_size, T, K]
        if len(bound.shape) == 2:
            bound = bound.unsqueeze(0)

        # Sample uniformly from 0 to bound (inclusive) using torch.rand
        expected = th.rand(*batch_size, self.T, self.K, dtype=bound.dtype, device=self.device, generator=self.rng) * bound
        st_dev = (expected * cv.view(1, 1, self.K,))
        return th.where(expected < eps, eps, expected), st_dev

    # Support functions
    def get_transport_idx(self, P: int, device: Union[th.device, str]) -> Union[th.Tensor,]:
        # Get above-diagonal indices of the transport matrix
        origins, destinations = th.triu_indices(P, P, offset=1, device=device)
        return th.stack((origins, destinations), dim=-1)

    def _create_wave(self, length:int, param:float=0.3, ) -> th.Tensor:
        """Create a wave function for the bound"""
        mid_index = length // 2
        increasing_values = 1 + param * th.cos(th.linspace(th.pi, th.pi / 2, steps=mid_index + 1,
                                                           device=self.device, dtype=th.float32, ))
        decreasing_values = 1 + param * th.cos(th.linspace(th.pi / 2, th.pi, steps=length - mid_index,
                                                           device=self.device, dtype=th.float32, ))
        return th.cat([increasing_values[:-1], decreasing_values, th.zeros((1,), device=self.device, dtype=th.float32)])

    def _get_num_loads_in_voyage(self, transport_idx:th.Tensor, P:Union[th.Tensor,int], ) -> th.Tensor:
        """Get number of transports loaded per POL"""
        # Create a boolean mask for load pairs using broadcasting and advanced indexing
        load_mask = th.zeros((P, P), dtype=th.bool, device=self.device, )
        load_mask[transport_idx[:, 0], transport_idx[:, 1]] = True
        # Count loads for each port
        return load_mask.sum(dim=1)

    def _get_num_AC_in_voyage(self, transport_idx:th.Tensor, POL:Union[th.Tensor,int], ) -> th.Tensor:
        """Get number of transport in arrival condition per POL"""
        mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] > POL)
        return mask.sum(dim=-1).squeeze()  # Shape: [num_POL]

    def _get_num_OB_in_voyage(self, transport_idx:th.Tensor, POL:Union[th.Tensor,int], ) -> th.Tensor:
        """Get number of transports in onboard per POL"""
        mask = (transport_idx[:, 0] <= POL) & (transport_idx[:, 1] > POL)
        return mask.sum(dim=-1).squeeze()  # Shape: [num_POL]

    def _get_ub_demand_normalization(self, bound:th.Tensor, eps:float=1e-2) -> th.Tensor:
        """Get upper bound for demand normalization"""
        return (bound + 4 * (bound / 2 * 0.5)).max()

class UniformMPP_Generator(MPP_Generator):
    """Subclass for generating demand for stowage plans using uniform distribution."""

    def __call__(self, batch_size:Tuple[int, ...], td: Optional[TensorDict] = None, rng:Optional=None) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size, td)

    def _generate(self, batch_size:Tuple[int, ...], td: Optional[TensorDict] = None, ) -> TensorDict:
        """Generate demand matrix for voyage with uniform distribution"""
        # Get initial demand bound based on capacity
        bound = self._initialize_demand_ub(batch_size)
        if batch_size != []: bound = bound.unsqueeze(0).expand(*batch_size, -1, -1) # Expand to batch size

        # Get initial demand based on random perturbed bound
        bound = self._random_perturbation(bound, self.perturbation)
        demand, _ = self._generate_moments(batch_size, bound, self.cv)
        # Get moments from uniform distribution
        e_x = (bound * 0.5).expand_as(demand)
        std_x = bound / th.sqrt(th.tensor(12, device=self.device)).expand_as(demand)
        batch_updates = th.zeros(batch_size, device=self.device).view(*batch_size, 1)

        # Return demand matrix
        return TensorDict({"observation":
                               {"realized_demand": demand.view(*batch_size, self.T * self.K),
                                "expected_demand": e_x.view(*batch_size, self.T * self.K),
                                "std_demand": std_x.view(*batch_size, self.T * self.K),
                                }}, batch_size=batch_size, device=self.device, )


if __name__ == "__main__":
    # Example usage
    config = {
        "ports": 4,
        "bays": 20,
        "decks": 2,
        "customer_classes": 3,
        "cargo_classes": 2,
        "weight_classes": 3,
        "capacity": [1000],
        "utilization_rate_initial_demand": 1.2,
        "spot_percentage": 0.3,
        "iid_demand": True,
        "cv_demand": 1.5,
        "perturbation": 0.1,
        "seed": 42
    }
    generator = MPP_Generator(device="cpu", **config)
    print(generator.tr_ob)
    print(generator.tr_ac)
    breakpoint()
    batch_size = (4,)
    td = generator(batch_size)