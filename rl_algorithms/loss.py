from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import distributions as d
import contextlib
import math

# Typing
from torch import Tensor
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union, Callable
from tensordict import (
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.utils import NestedKey
from tensordict.nn import (
    dispatch,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)


# TorchRL
from torchrl.data.tensor_specs import Composite, TensorSpec
from torchrl.data.utils import _find_action_space
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    ValueEstimators,
    distance_loss,
)

from torchrl.objectives.ppo import PPOLoss
from torchrl.objectives.sac import SACLoss, _delezify, compute_log_prob

# Custom
from environment.utils import *
from rl_algorithms.utils import *
from models.decoder import hard_sigmoid_ste

def weight_violations(violations, lagrange_multiplier):
    """ Weight violations by the lagrange multiplier of lm.dim()\in{3,4}."""
    if lagrange_multiplier is None:
        return violations

    lm = lagrange_multiplier
    k = lm.ndim - violations.ndim
    assert k in (-2, -1, 0, 1,), f"incompatible shapes: λ {lm.shape} vs vio {violations.shape}"

    if k == 1:
        lm = lm.mean(dim=0)
    # if not primal-dual, add dims to lm for broadcasting
    elif k == -1:
        lm = lm.view(1, *lm.shape)
    elif k == -2:
        lm = lm.view(1, 1, *lm.shape)

    # now shapes are broadcastable; env/batch/time untouched
    return violations * lm


def loss_feasibility(td:TensorDictBase, action:Tensor, lagrange_multiplier:Optional[Tensor]=None,
                     aggregate_feasibility:str="sum", **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
    """ Compute feasibility loss based on the action and the lagrange multiplier."""
    # Environment parameters
    lhs_A = td.get("lhs_A")
    rhs = td.get("rhs")
    # real_excess_pod_locations = td["observation"].get("excess_pod_locations", None)
    utilization = td["observation"].get("utilization", None)
    env_init = kwargs.get("env_init", None)
    transform_tau_to_pol = env_init.get("transform_tau_to_pol", None)
    transform_tau_to_pod = env_init.get("transform_tau_to_pod", None)
    bays = env_init.get("B", None)
    blocks = env_init.get("BL", None)
    decks = env_init.get("D", None)
    transports = env_init.get("T", None)
    cargo_types = env_init.get("K", None)
    time = transports * cargo_types

    # Violations
    violations = compute_violation(action, lhs_A, rhs)
    weighted_violations = weight_violations(violations, lagrange_multiplier)

    # Excess pod locations computation
    # Broadcast action over identity matrix to add to utilization
    utilization = utilization.reshape(*action.shape, -1)
    I = torch.eye(time, device=action.device, dtype=action.dtype)
    utilization = utilization + action[..., None] * I[None, :, None, :]

    # Get excess pod locations from utilization
    # todo: test if smooth operations work well
    utilization = utilization.reshape(*action.shape[:-1], bays, blocks, decks, transports, cargo_types)
    _, pod_locations = compute_pol_pod_locations(utilization, transform_tau_to_pol, transform_tau_to_pod, differentiable=True)
    mass = pod_locations.sum(dim=-3)
    soft_any = th.clamp(mass, 0.0, 1.0)
    excess_pod_locations = th.relu(soft_any.sum(dim=-1) - 1.0)

    # Compute loss from weighted violations
    agg_fn = make_feasibility_aggregator(aggregate_feasibility, ndims=violations.ndim)
    loss = agg_fn(weighted_violations)
    convex_violations = agg_fn(violations)
    pod_agg_fn = make_feasibility_aggregator(aggregate_feasibility, ndims=excess_pod_locations.ndim)
    pod_violations = pod_agg_fn(excess_pod_locations) if excess_pod_locations is not None else torch.tensor(0.0, device=loss.device)
    loss += pod_violations
    return loss, {"total_convex_violations":convex_violations, "total_pod_violations":pod_violations, "violations":violations, "pod_violations":excess_pod_locations}


def make_feasibility_aggregator(
    aggregate_feasibility: str,
    ndims: int,
) -> Callable[[th.Tensor], th.Tensor]:
    """
    Returns an aggregation function that maps a tensor -> scalar.

    Aggregation behavior:
      - "sum": sum over all dims except dim 0, then mean over dim 0
      - "mean": global mean over all elements

    ndims must match the tensor rank you will pass to the returned function.
    """
    if ndims < 1:
        raise ValueError(f"ndims must be >= 1, got {ndims}")

    sum_dims = tuple(range(1, ndims))  # sum all but batch dim 0

    if aggregate_feasibility == "sum":
        def agg_fn(x: th.Tensor) -> th.Tensor:
            if x.dim() != ndims:
                raise ValueError(f"Expected tensor with dim={ndims}, got {x.dim()}")
            return x.sum(dim=sum_dims).mean()
        return agg_fn

    if aggregate_feasibility == "mean":
        def agg_fn(x: th.Tensor) -> th.Tensor:
            if x.dim() != ndims:
                raise ValueError(f"Expected tensor with dim={ndims}, got {x.dim()}")
            return x.mean()
        return agg_fn

    raise ValueError(f"Unknown aggregation method: {aggregate_feasibility}")

def extract_mask_supervision_tensors(
    td: TensorDictBase,
    *,
    mask_logits_key: str = "mask",
    feasible_key: Tuple[str, str] = ("observation", "preload_mask"),
    residual_key: Tuple[str, str] = ("observation", "residual_capacity"),
    podloc_key: Tuple[str, str] = ("observation", "pod_locations"),
    pod_key_candidates: Tuple[Tuple[str, ...], ...] = (("observation", "pod"), ("pod",)),
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      mask_logits: [B,T,K] or [B,K]
      feasible:    same shape as mask_logits (bool) or None
      residual_capacity_grid: [B,T,Bay,Deck,Block]
      pod_locations_grid:     [B,T,Bay,Deck,Block,P]
      pod: [B,T] or [B]
    """
    mask_logits = td.get(mask_logits_key, None)
    if mask_logits is None:
        raise KeyError(f"Missing {mask_logits_key} in tensordict. "
                       f"Ensure actor writes td['{mask_logits_key}']=mask_logits during forward().")

    feasible = td.get(feasible_key, None)
    residual_capacity = td.get(residual_key, None)
    pod_locations = td.get(podloc_key, None)

    if residual_capacity is None:
        raise KeyError(f"Missing {residual_key} in tensordict.")
    if pod_locations is None:
        raise KeyError(f"Missing {podloc_key} in tensordict.")

    pod = None
    for k in pod_key_candidates:
        if td.get(k, None) is not None:
            pod = td.get(k)
            break
    if pod is None:
        raise KeyError(f"Missing pod key. Tried {pod_key_candidates}.")
    return mask_logits, feasible, residual_capacity, pod_locations, pod.long()

def add_mask_bce_loss(
    td: TensorDictBase,
    *,
    w_pos: float = 1.0,
    w_neg: float = 10.0,
    mask_logits_key: str = "mask",
    feasible_key: Tuple[str, str] = ("observation", "preload_mask"),
    residual_key: Tuple[str, str] = ("observation", "residual_capacity"),
    podloc_key: Tuple[str, str] = ("observation", "pod_locations"),
    pod_key_candidates: Tuple[Tuple[str, ...], ...] = (("observation", "pod"), ("pod",)),
    symmterical: bool = False,
    # one-sided / subset selection knobs (used when symmterical=False)
    lam_illegal: float = 1.0,  # penalize opening illegal actions (y==0)
    lam_sparse: float = 0.001,  # penalize opening too many legal actions (y==1)
    lam_cover: float = 0.1,  # prevent all-zero collapse (expects mask_logits is [B,T,K])
    K_min: float = 1.0,  # expected minimum opens among legal per (B,T)
    tau: float = 1.0,  # sigmoid temperature for sparse/cover terms
) -> torch.Tensor:
    """
    Computes BCE supervision for a learned action mask: targets are post-action validity labels from the checker.
    If `feasible` is not true environment feasibility, set it to None or pass a correct feasibility key to avoid bias.
    - td: TensorDict with necessary keys
    - mask_logits_key: key for the predicted mask logits
    - y : target labels from checker

    """
    mask_logits, feasible, residual_capacity, pod_locations, pod = extract_mask_supervision_tensors(
        td,
        mask_logits_key=mask_logits_key,
        feasible_key=feasible_key,
        residual_key=residual_key,
        podloc_key=podloc_key,
        pod_key_candidates=pod_key_candidates,
    )

    # Normalize shapes to [B,T,K] (and pod to [B,T])
    if mask_logits.dim() == 2:
        mask_logits = mask_logits.unsqueeze(1)
    if feasible is not None and feasible.dim() == 2:
        feasible = feasible.unsqueeze(1)
    if pod.dim() == 1:
        pod = pod[:, None].expand(mask_logits.shape[0], mask_logits.shape[1])

    # Post-action validity labels on grid -> flatten to [B,T,K]
    # y target are labels that prevents POD-violations
    B = td["observation"]["long_crane_moves_discharge"].shape[-1] + 1
    y_grid = compute_targets_to_prevent_POD_violations(
        residual_capacity,
        pod_locations,
        pod,
        B,
    ).view_as(residual_capacity)
    y = y_grid.reshape_as(mask_logits).float()

    # Optionally exclude infeasible entries from the supervised loss
    if feasible is not None:
        y = y.masked_fill(~feasible, 0.0)
        mask_logits = mask_logits.masked_fill(~feasible, 0.0)


    if symmterical:
        # Weighted BCE loss with symmetrical weights on positives and negatives
        weight = torch.where(
            y > 0.5,
            torch.tensor(w_pos, device=y.device),
            torch.tensor(w_neg, device=y.device),
        )
        return F.binary_cross_entropy_with_logits(mask_logits, y, weight=weight)

    # --- one-sided + subset selection (y is an upper bound) ---
    # valid entries to supervise
    if feasible is None:
        valid = torch.ones_like(y, dtype=torch.bool)
    else:
        valid = feasible.bool()

    y_bool = (y > 0.5)
    illegal = (~y_bool) & valid
    legal = y_bool & valid

    # (A) Safety: penalize opening illegal actions (false positives)
    # For y==0, BCE-with-logits term is softplus(logit). Minimizing pushes logits -> -inf (closed).
    if illegal.any():
        L_illegal = F.softplus(mask_logits[illegal]).float().mean()
    else:
        L_illegal = mask_logits.new_tensor(0.0)

    # Convert to probabilities (for sparse/cover)
    p_open = torch.sigmoid(mask_logits / tau)

    # (B) Sparsity: prefer subset within legal set (discourage opening everything legal)
    if lam_sparse != 0.0:
        if legal.any():
            L_sparse = p_open[legal].mean()
        else:
            L_sparse = mask_logits.new_tensor(0.0)
    else:
        L_sparse = mask_logits.new_tensor(0.0)

    # (C) Coverage floor: avoid collapse to all-zeros (optional)
    # Enforce at least K_min opens in expectation per (B,T).
    if lam_cover != 0.0:
        BT = mask_logits.shape[0] * mask_logits.shape[1]
        p_legal = (p_open * legal.to(p_open.dtype)).reshape(BT, -1)
        mass = p_legal.sum(dim=-1)  # [B*T]
        L_cover = F.relu(K_min - mass).mean()
    else:
        L_cover = mask_logits.new_tensor(0.0)

    return lam_illegal * L_illegal + lam_sparse * L_sparse + lam_cover * L_cover

class FeasibilitySACLoss(SACLoss):
    """TorchRL implementation of the SAC loss with feasibility constraints.

    Based on "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
    https://arxiv.org/abs/1801.01290 and "Soft Actor-Critic Algorithms and Applications"
    https://arxiv.org/abs/1812.05905.

    Adds feasibility constraints to the standard SAC loss.

    Args:
        actor_network (ProbabilisticActor): Stochastic actor network.
        qvalue_network (TensorDictModule): Q(s, a) parametric model. Typically outputs "state_action_value".
        value_network (Optional[TensorDictModule]): V(s) parametric model, outputs "state_value".
        num_qvalue_nets (int): Number of Q-value networks. Defaults to 2.
        loss_function (str): Loss function for the value network. Defaults to "smooth_l1".
        alpha_init (float): Initial entropy multiplier. Defaults to 1.0.
        min_alpha (float): Minimum value of alpha. Defaults to None.
        max_alpha (float): Maximum value of alpha. Defaults to None.
        action_spec: Action tensor spec for auto entropy computation. Defaults to None.
        fixed_alpha (bool): Whether to fix alpha to its initial value. Defaults to False.
        target_entropy (Union[str, float]): Target entropy for the policy. Defaults to "auto".
        delay_actor (bool): Whether to delay updates to the actor network. Defaults to False.
        delay_qvalue (bool): Whether to delay updates to Q-value networks. Defaults to True.
        delay_value (bool): Whether to delay updates to value networks. Defaults to True.
        gamma (float): Discount factor for future rewards.
        priority_key (str): Key to write priorities for prioritized replay. Defaults to "td_error".
        separate_losses (bool): Whether to compute separate gradients for shared parameters. Defaults to False.
        reduction (str): Specifies reduction for the loss: "none", "mean", "sum". Defaults to "mean".
    """

    @dataclass
    class _AcceptedKeys:
        """Default tensordict keys for the loss."""

        action: str = "action"
        value: str = "state_value"
        state_action_value: str = "state_action_value"
        log_prob: str = "sample_log_prob"
        priority: str = "td_error"
        reward: str = "reward"
        done: str = "done"
        terminated: str = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: Union[TensorDictModule, List[TensorDictModule]],
        value_network: Optional[TensorDictModule] = None,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        action_spec=None,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        delay_actor: bool = False,
        delay_qvalue: bool = True,
        delay_value: bool = True,
        gamma: float = None,
        priority_key: str = None,
        separate_losses: bool = True,
        reduction: str = None,
        lagrangian_multiplier: torch.Tensor = None,
        **kwargs,
    ) -> None:
        super().__init__(
            actor_network=actor_network,
            qvalue_network=qvalue_network,
            value_network=value_network,
            num_qvalue_nets=num_qvalue_nets,
            loss_function=loss_function,
            alpha_init=alpha_init,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            action_spec=action_spec,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            delay_value=delay_value,
            gamma=gamma,
            priority_key=priority_key,
            separate_losses=separate_losses,
            reduction=reduction,
        )
        self.register_buffer("lagrangian_multiplier", lagrangian_multiplier)
        self._env_init = kwargs.get("env_init", None)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Computes SAC loss with feasibility constraints."""
        # Actor loss
        loss_actor, metadata_actor = self._actor_loss(tensordict)

        # Q-value loss
        loss_qvalue, metadata_qvalue = self._qvalue_v2_loss(tensordict)

        # Feasibility loss
        action = metadata_actor["action"]
        if "lhs_A" not in tensordict or "rhs" not in tensordict:
            raise ValueError("Feasibility loss requires 'lhs_A' and 'rhs' in tensordict.")
        lagrangian_multiplier = metadata_actor.get("lagrangian_multiplier", self.lagrangian_multiplier)
        feasibility_loss, violation_dict = loss_feasibility(tensordict, action, lagrangian_multiplier, env_init=self._env_init,)

        # Alpha loss
        loss_alpha = self._alpha_loss(metadata_actor["log_prob"])

        # Combine losses
        entropy = -metadata_actor["log_prob"]
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_alpha": loss_alpha,
            "alpha": self._alpha,
            "entropy": entropy.detach().mean(),
            "loss_feasibility": feasibility_loss,
            "violation": violation_dict["violations"],
            "total_violation": violation_dict["total_convex_violations"],
            "pod_violation": violation_dict["pod_violations"],
            "total_pod_violation": violation_dict["total_pod_violations"],
            "lagrangian_multiplier": lagrangian_multiplier,
        }
        if "pod_locations" in tensordict["observation"]:
            loss_mask = add_mask_bce_loss(tensordict,)
            out["loss_mask"] = loss_mask

        # Reduce outputs based on reduction mode
        td_out = TensorDict(out, [])
        td_out = td_out.named_apply(
            lambda name, value: value.mean() if name.startswith("loss_") else value,
            batch_size=[],
        )
        return td_out

    def _actor_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)
            tensordict["action"] = dist.rsample()
        tensordict = self.actor_network(tensordict) # Perform projection
        log_prob = tensordict["sample_log_prob"] # Use sample log prob
        # (non projection on SAC)
        # log_prob = compute_log_prob(dist, tensordict["action"], self.tensor_keys.log_prob)

        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        td_q.set(self.tensor_keys.action, tensordict["action"])
        td_q = self._vmap_qnetworkN0(
            td_q,
            self._cached_detached_qvalue_params,  # should we clone?
        )
        min_q_logprob = (
            td_q.get(self.tensor_keys.state_action_value).min(0)[0].squeeze(-1)
        )
        lagrangian_multiplier = td_q.get("lagrangian_multiplier", self.lagrangian_multiplier)
        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )
        return self._alpha * log_prob - min_q_logprob, {"log_prob": log_prob.detach(), "action": tensordict["action"],
                                                        "lagrangian_multiplier": lagrangian_multiplier}

class FeasibilityClipPPOLoss(PPOLoss):
    """Clipped PPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``. Set ``critic_coef`` to ``None`` to exclude the value
            loss from the forward outputs.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (bool or float, optional): If a ``float`` is provided, it will be used to compute a clipped
            version of the value prediction with respect to the input tensordict value estimate and use it to
            calculate the value loss. The purpose of clipping is to limit the impact of extreme value predictions,
            helping stabilize training and preventing large updates. However, it will have no impact if the value
            estimate was done by the current version of the value estimator. If instead ``True`` is provided, the
            ``clip_epsilon`` parameter will be used as the clipping threshold. If not provided or ``False``, no
            clipping will be performed. Defaults to ``False``.

    """

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
        clip_value: bool | float | None = None,
        lagrangian_multiplier: torch.Tensor = None,
        **kwargs,
    ):
        # Define clipping of the value loss
        if isinstance(clip_value, bool):
            clip_value = clip_epsilon if clip_value else None

        super(FeasibilityClipPPOLoss, self).__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            reduction=reduction,
            clip_value=clip_value,
            **kwargs,
        )
        for p in self.parameters():
            device = p.device
            break
        else:
            device = None
        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon, device=device))
        self.register_buffer("lagrangian_multiplier", lagrangian_multiplier)
        self._env_init = kwargs.get("env_init", None)

    @property
    def _clip_bounds(self) -> Tuple[Tensor, Tensor]:
        return (
            (-self.clip_epsilon).log1p(),
            self.clip_epsilon.log1p(),
        )

    @property
    def out_keys(self) -> List[str]:
        if self._out_keys is None:
            keys = ["loss_objective", "clip_fraction"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            keys.append("ESS")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values) -> None:
        self._out_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        log_weight, dist, kl_approx = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        ratio = log_weight_clip.exp()
        gain2 = ratio * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain}, batch_size=[])
        td_out.set("clip_fraction", clip_fraction)

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)

        # Feasibility loss based on policy mean
        loc = dist.loc if hasattr(dist, 'loc') else dist.base_dist.loc
        if "lhs_A" not in tensordict or "rhs" not in tensordict:
            raise ValueError("Feasibility loss requires 'lhs_A' and 'rhs' in tensordict.")
        lagrangian_multiplier = tensordict.get("lagrangian_multiplier", self.lagrangian_multiplier)
        feasibility_loss, violation_dict = loss_feasibility(tensordict, loc, lagrangian_multiplier, env_init=self._env_init,)
        if "pod_locations" in tensordict["observation"]:
            loss_mask = add_mask_bce_loss(tensordict,)
            td_out.set("loss_mask", loss_mask)
        td_out.set("loss_feasibility", feasibility_loss)
        td_out.set("violation", violation_dict["violations"])
        td_out.set("total_violation", violation_dict["total_convex_violations"])
        td_out.set("pod_violation", violation_dict["pod_violations"])
        td_out.set("total_pod_violation", violation_dict["total_pod_violations"])
        td_out.set("lagrangian_multiplier", lagrangian_multiplier)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

    def _log_weight(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, d.Distribution, torch.Tensor]:
        action = tensordict.get("raw_action", self.tensor_keys.action) # use raw action if available
        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            dist = self.actor_network.get_dist(tensordict)

        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.sample_log_prob} requires grad."
            )

        if action.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.action} requires grad."
            )
        if isinstance(action, torch.Tensor):
            td = self.actor_network(tensordict, action=action) # Re-use unprojected action
            log_prob = td.get(self.tensor_keys.sample_log_prob)
            # This does not include Jacobian adjustment:
            # log_prob = dist.log_prob(action)
        else:
            maybe_log_prob = dist.log_prob(tensordict)
            if not isinstance(maybe_log_prob, torch.Tensor):
                # In some cases (Composite distribution with aggregate_probabilities toggled off) the returned type may not
                # be a tensor
                log_prob = maybe_log_prob.get(self.tensor_keys.sample_log_prob)
            else:
                log_prob = maybe_log_prob

        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        kl_approx = (prev_log_prob - log_prob).unsqueeze(-1)

        return log_weight, dist, kl_approx