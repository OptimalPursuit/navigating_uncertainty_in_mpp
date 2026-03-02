# Datatypes
from typing import Optional, Tuple, Dict, Union, Sequence, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Distribution, TransformedDistribution, Normal, Independent

# TorchRL
from torchrl.modules import ProbabilisticActor
from torchrl.data.tensor_specs import Composite, TensorSpec

# Custom
from environment.utils import compute_violation
from rl_algorithms.clipped_gaussian import ClippedGaussian
from rl_algorithms.utils import inspect_tensordict

class ProjectionProbabilisticActor(ProbabilisticActor):
    """Probabilistic actor with projection layer for enforcing constraints."""
    def __init__(self,
                 module: TensorDictModule,
                 in_keys: Union[NestedKey, Sequence[NestedKey]],
                 out_keys: Optional[Sequence[NestedKey]] = None,
                 *,
                 spec: Optional[TensorSpec] = None,
                 projection_layer: Optional[nn.Module] = None,
                 projection_type: Optional[str] = None,
                 revenues: Optional[Tensor] = None,
                 jacobian_correction: bool = True,
                 **kwargs):
        super().__init__(module, in_keys, out_keys, spec=spec, **kwargs)

        # Initialize projection layer
        self.projection_layer = projection_layer
        self.projection_type = projection_type.lower()
        self.jacobian_correction = jacobian_correction

        # Initialize clipped Gaussian
        initial_loc = torch.zeros(spec.shape, device=spec.device, dtype=spec.dtype)
        initial_scale = torch.ones(spec.shape, device=spec.device, dtype=spec.dtype)
        clip_min = spec.space.low
        clip_max = spec.space.high
        self.clipped_gaussian = ClippedGaussian(initial_loc, initial_scale, clip_min, clip_max)
        self.revenues_per_timestep = revenues

        self.projection_methods = {
            "weighted_scaling": self.weighted_scaling_projection,
            "linear_violation": self.violation_projection,
            "linear_violation_policy_clipping": self.violation_projection_policy_clipping,
            "inner_convex_violation": self.violation_projection,
            'inner_convex_violation_alpha': self.alpha_cheby_projection,
            "policy_clipping": self.policy_clipping_projection,
            "weighted_scaling_policy_clipping": self.weighted_scaling_policy_clipping_projection,
            "convex_program":self.convex_program,
            "convex_program_policy_clipping":self.convex_program_policy_clipping,
            "alpha_chebyshev": self.alpha_cheby_projection,
            "log_barrier": self.violation_projection,
            "frank_wolfe": self.franke_wolfe_improvement,
            "none": self.identity_fn,
        }

        self.jacobian_methods = {
            "weighted_scaling": self.jacobian_weighted_scaling,
            "weighted_scaling_policy_clipping": self.jacobian_weighted_scaling,
            "linear_violation": self.jacobian_uvp_K_steps,
            "inner_convex_violation": self.jacobian_uvp_K_steps,
        }

    def get_logprobs(self, action:Tensor, dist:Distribution) -> Tensor:
        """Compute the log probabilities of the actions given the distribution."""
        return dist.base_dist.log_prob(action) # Shape: [Batch, Features]

    # Projections
    @staticmethod
    def weighted_scaling(sample:Tensor, ub:Tensor, epsilon:float=1e-8) -> Tensor:
        sum_sample = sample.sum(dim=-1, keepdim=True)
        upper_bound = ub.unsqueeze(-1)
        scaling_factor = upper_bound / (sum_sample + epsilon)  # Avoid division by zero
        out = torch.where(
            sum_sample > upper_bound,
            sample * scaling_factor,
            sample
        )
        return out

    def get_max_revenue_action(self, out:TensorDict) -> Tensor:
        """Compute the action that maximizes revenue given the demand and revenues per timestep."""
        t = out["observation", "timestep"].long()  # [B]
        var_mask = out["observation", "action_mask"].float() if "action_mask" in out["observation"] else torch.ones_like(out["action"])
        relative_action = out["action"] * var_mask / out["action"].sum(dim=-1, keepdim=True).clamp(min=1e-8)
        demand = out["observation", "realized_demand"].gather(-1, t.unsqueeze(-1)).squeeze(-1)

        revenue_action = demand.unsqueeze(-1) * relative_action * self.revenues_per_timestep[t].unsqueeze(-1)  # [B,n]
        # revenue_action = out["action"] * self.revenues_per_timestep[t].unsqueeze(-1)  # [B,n]
        return revenue_action


    def weighted_scaling_projection(self, out:TensorDict) -> TensorDict:
        out["action"] = self.weighted_scaling(out["action"], ub=out["ub"])
        return out

    def policy_clipping_projection(self, out:TensorDict) -> TensorDict:
        out["action"] = out["action"].clamp(min=out["clip_min"], max=out["clip_max"])
        return out

    def weighted_scaling_policy_clipping_projection(self, out:TensorDict) -> TensorDict:
        out["action"] = self.weighted_scaling_projection(out)
        out["action"] = self.policy_clipping_projection(out)
        return out

    def quadratic_program(self, out:TensorDict) -> TensorDict:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        return out

    def convex_program(self, out:TensorDict) -> TensorDict:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        return out

    def convex_program_policy_clipping(self, out:TensorDict) -> TensorDict:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        out["action"] = self.policy_clipping_projection(out)
        return out

    def violation_projection(self, out:TensorDict) -> TensorDict:
        var_mask = out["observation", "action_mask"].float() if "action_mask" in out["observation"] else None
        shared = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], var_mask=var_mask, return_uvp_masks=self.jacobian_correction)
        if self.jacobian_correction:
            out["action"], out["violation_mask"] = shared
        else:
            out["action"] = shared
        return out

    def franke_wolfe_improvement(self, out:TensorDict) -> TensorDict:
        if "critic_fn" not in out:
            out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], s=out, critic_fn=None, mode="proj")
            return out
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], s=out, critic_fn=out["critic_fn"])
        return out

    def alpha_cheby_projection(self, out:TensorDict) -> TensorDict:
        var_mask = out["observation", "action_mask"].float() if "action_mask" in out["observation"] else None
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], var_mask=var_mask)
        return out

    def violation_projection_policy_clipping(self, out:TensorDict) -> TensorDict:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        out["action"] = self.policy_clipping_projection(out)
        return out

    def identity_fn(self, out:TensorDict) -> TensorDict:
        return out

    def handle_action_projection(self, out:TensorDict) -> TensorDict:
        """Handle with policy projection"""
        projection_fn = self.projection_methods.get(self.projection_type, self.identity_fn)
        return projection_fn(out)

    def jacobian_weighted_scaling(self, out:TensorDict, epsilon:float=1e-8) -> Tuple[Tensor, Tensor]:
        """Compute the Jacobian of the direct scaling projection:
            J_g(x) = y / sum(x)^2 * (diag(sum(x)) - x * 1^T)"""
        # Input
        x = out["action"]
        y = out["ub"]

        # Shapes
        x_dim = x.dim()
        if x_dim == 2:
            batch, n = x.shape
            seq = 1
        elif x_dim == 3:
            batch, seq, n = x.shape
            # treat as batch of sequences
            x = x.view(batch * seq, n)
            y = y.view(batch * seq,)
        else:
            raise ValueError(f"Invalid dimension of x: {x.dim()}")

        # Compute
        sum_x = x.sum(dim=-1, keepdim=True)
        scaling_factor = y.unsqueeze(-1) / (sum_x)**2 # Shape: (batch_size, n)
        kronecker_delta = torch.eye(n).unsqueeze(0).expand(batch*seq, n, n).to(x.device)  # Shape: (batch_size, n, n)
        x_product = x.unsqueeze(-1) * torch.ones(x.size(-1), device=x.device)  # Shape: (batch_size, n, n)
        jacobian = scaling_factor.unsqueeze(-1) * (kronecker_delta * sum_x.unsqueeze(-1)) - x_product # Shape: (batch_size, n, n)
        if x_dim == 3:
            jacobian = jacobian.view(batch, seq, n, n)

        _, logabsdet = torch.linalg.slogdet(jacobian)
        return jacobian, logabsdet

    def jacobian_uvp_K_steps(self, out: TensorDict) -> Tuple[Tensor, Tensor]:
        """
        Exact Jacobian of the K-step UVP loop, by re-simulating x_k.
        Returns:
          - [B,n,n] or [B,S,n,n]
        """
        # Use pre-projection action if available
        x0 = out.get("raw_action", out["action"])
        A = out["lhs_A"]
        b = out["rhs"]
        masks = out.get("uvp_masks", out.get("violation_mask", None))
        proj = self.projection_layer
        eps = 1e-6

        # Shapes
        b_ = b.unsqueeze(1) if b.dim() == 2 else b
        A_ = A.unsqueeze(1) if A.dim() == 3 else A
        x = x0.unsqueeze(1) if x0.dim() == 2 else x0
        B, S, m, n = A_.shape

        A_work, b_work = proj._normalize_constraints(A_, b_)
        At = A_work.transpose(-2, -1)  # [B,S,n,m]
        eta = proj.get_eta(A_work, b_work).unsqueeze(-1)  # [B,S,1,1]

        I = torch.eye(n, device=x.device, dtype=x.dtype).view(1, 1, n, n).expand(B, S, n, n)
        J = I.clone()

        for k in range(proj.K):
            d = masks[..., k, :].to(x.dtype).view(B, S, m, 1)  # [B,S,m] (0/1)
            DA = A_work * d  # [B,S,m,n] == D @ A_work (without [m,m])
            M = At @ DA  # [B,S,n,n] == A^T D A
            J_step = (I - eta * M) + eps * I
            J = J_step @ J

        if A.dim() == 3:
            J = J.squeeze(1)
        return J

    def handle_jacobian_adjustment(self, out:TensorDict) -> Optional[Tuple[Tensor,Tensor]]:
        """Handle with Jacobian adjustment of projection methods;
        We only have Jacobian for weighted scaling and linear violation."""
        jacobian_fn = self.jacobian_methods.get(self.projection_type, None)
        return jacobian_fn(out) if jacobian_fn else None

    def jacobian_adaptation(self, sample_logp: Tensor, jacobian: Tensor) -> Tensor:
        """
        sample_logp: [B] or [B,T] (already summed over action dims)
        jacobian:    [B,N,N] or [B,T,N,N]
        """
        if jacobian is None:
            return sample_logp

        # CoV correction
        lad = torch.linalg.slogdet(jacobian)[1]  # [B,S] or [B]
        lad_clamped = torch.nan_to_num(lad, neginf=-20.0, posinf=20.0).clamp(-20.0, 20.0)
        output = sample_logp - lad_clamped

        # “no update from invalid”:
        valid = torch.isfinite(lad)
        output = torch.where(valid, output, torch.zeros_like(output))


        # # and track how many got dropped
        # drop_frac = (~valid).float().mean()
        # print(f"Jacobian adaptation: dropping {drop_frac:.2%} of samples due to invalid Jacobian (log|det|={lad_clamped})")

        # todo: there is an error here!
        #in mpp: batch | batch, action , action | batch
        #in block_mpp: batch, var | batch, var , var | batch --> Fix this!
        # print("shapes", sample_logp.shape, jacobian.shape, log_abs_det.shape)
        # print("Means:",
        #         "jacobian mean:", jacobian.mean().item(),
        #         "sample_logp mean:", sample_logp.mean().item(),
        #         "log_abs_det mean:", lad.mean().item(),
        #         "output mean:", output.mean().item())

        return output

    @staticmethod
    def _expand_mask(mask: Tensor, batch_shape: torch.Size, N: int) -> Tensor:
        # mask: [N] or [B,N] or [B,T,N] -> [*batch_shape, N]
        if mask.dim() == 1:
            mask = mask.view((1,) * len(batch_shape) + (N,))
        else:
            missing = (len(batch_shape) + 1) - mask.dim()
            if missing > 0:
                mask = mask.view((1,) * missing + tuple(mask.shape))
        return mask.expand(*batch_shape, N)

    @staticmethod
    def _expand_Ab(A: Tensor, b: Tensor, batch_shape: torch.Size, N: int) -> Tuple[Tensor, Tensor]:
        # A: [B,m,N] or [B,T,m,N] -> [*batch_shape, m, N]
        # b: [B,m]   or [B,T,m]   -> [*batch_shape, m]
        m = A.shape[-2]

        missing_A = (len(batch_shape) + 2) - A.dim()
        if missing_A > 0:
            A = A.view((1,) * missing_A + tuple(A.shape))
        A = A.expand(*batch_shape, m, N)

        missing_b = (len(batch_shape) + 1) - b.dim()
        if missing_b > 0:
            b = b.view((1,) * missing_b + tuple(b.shape))
        b = b.expand(*batch_shape, b.shape[-1])

        return A, b

    def masked_subspace_logprobs(
            self,
            out: TensorDict,
            action: Tensor,               # [B,N] or [B,T,N]
            handle_jacobian_adjustment: Optional[Callable] = None,
            jacobian_adaptation: Optional[Callable] = None,
            clamp_min: float = -50.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            log_prob_full: same shape as action, zeros on masked dims
            sample_logp:   action.shape[:-1], sum over active dims
        """

        # ---- no jacobian: fully vectorized ----
        if not self.jacobian_correction:
            return out["log_prob"], out["sample_log_prob"]

        # Shapes
        batch_shape = action.shape[:-1]
        N = action.shape[-1]
        M = int(torch.tensor(batch_shape, device=action.device).prod().item()) if batch_shape else 1

        # Dense per-dimension logp in the original shape, via your helper
        logp_full = out["log_prob"]

        # Reshape
        mask = self._expand_mask(out["observation", "action_mask"].bool(), batch_shape, N)
        mask = mask.reshape(M, N)
        action = action.reshape(M, N)
        logp_full = logp_full.reshape(M, N)

        # ---- jacobian: ragged active dims -> group by mask pattern ----
        A, b = self._expand_Ab(out["lhs_A"], out["rhs"], batch_shape, N)
        A = A.reshape(M, A.shape[-2], N)      # [M,m,N]
        b = b.reshape(M, b.shape[-1])         # [M,m]

        # We must output zeros on masked dims
        logp_out = action.new_zeros(M, N)
        sample = action.new_zeros(M)

        # Group by unique mask patterns
        packed = torch_packbits_bool(mask, dim=-1)             # [M, ceil(N/8)]
        _, inv = torch.unique(packed, dim=0, return_inverse=True)         # [M]
        n_groups = int(inv.max().item()) + 1
        print(f"Unique mask patterns: {n_groups} (out of {2**N} possible)")

        for g in range(n_groups):
            rows = (inv == g).nonzero(as_tuple=True)[0]
            print(f"Group {g}: {rows.numel()} rows")
            active = mask[rows[0]].nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                continue

            z = action[rows][:, active]                         # [G,k]
            logp_active = logp_full[rows][:, active]            # [G,k]

            td = TensorDict(
                {
                    "action": z,
                    "lhs_A": A[rows].index_select(-1, active),  # [G,m,k]
                    "rhs": b[rows],                             # [G,m]
                },
                batch_size=torch.Size([rows.numel()]),
            )
            J = handle_jacobian_adjustment(td)
            logp_active = jacobian_adaptation(logp_active, jacobian=J).clamp(min=clamp_min)

            logp_out[rows[:, None], active[None, :]] = logp_active
            sample[rows] = logp_active.sum(-1)

        print("Final logp_out shape:", logp_out.shape)
        print("Final sample_logp shape:", sample.shape)
        breakpoint()

        return logp_out.reshape(*batch_shape, N), sample.reshape(*batch_shape)

    def forward(self, *args, **kwargs) -> TensorDict:
        out = args[0] if "action" in kwargs else super().forward(*args, **kwargs)
        dist = self.get_dist(out)

        # print(out.keys())
        # breakpoint()
            # tensordict.set("critic_fn", self.qvalue_network)

        # ---- move UB computation UP (needed for weighted_scaling projection) ----
        timestep_idx = out["observation", "timestep"].squeeze(0)
        out["ub"] = out["observation", "realized_demand"].gather(-1, timestep_idx.unsqueeze(-1)).squeeze(-1)

        # ---- store raw action BEFORE any projection ----
        out["raw_action"] = out["action"]  # no need to clone unless you mutate it elsewhere

        # ---- project NOW (so it can store masks into out) ----
        out = self.handle_action_projection(out)
        out["observation", "env_action"] = out["action"]

        # ---- compute log-prob using raw action (policy sample space) ----
        raw = out["raw_action"]

        if self.projection_type in ["policy_clipping", "weighted_scaling_policy_clipping"]:
            # For clipping-style distributions, you probably want log_prob at the *clipped* action.
            # Keep your existing behavior or switch to out["action"] depending on your ClippedGaussian definition.
            self.clipped_gaussian.mean = out["loc"]
            self.clipped_gaussian.var = out["scale"]
            self.clipped_gaussian.low = out["clip_min"]
            self.clipped_gaussian.high = out["clip_max"]
            out["log_prob"] = self.clipped_gaussian.log_prob(out["action"])
        else:
            out["log_prob"] = self.get_logprobs(raw, dist)

        # ---- sample_log_prob should also be based on raw ----
        logp_full = out["log_prob"]
        out["sample_log_prob"] = logp_full.sum(dim=-1)

        # ---- jacobian correction uses raw_action implicitly (via patch #1) ----
        if self.jacobian_correction:
            J = self.handle_jacobian_adjustment(out)
            if J is not None:
                out["adj_sample_log_prob"] = self.jacobian_adaptation(out["sample_log_prob"], jacobian=J).clamp(min=-20.0, max=20.0)

        return out

    # def forward(self, *args, **kwargs,) -> TensorDict:
    #     out = args[0] if "action" in kwargs else super().forward(*args, **kwargs)
    #
    #     # Raise error for projection layers without log prob adaptation implementations
    #     if self.projection_type not in self.projection_methods.keys():
    #         raise ValueError(f"Log prob adaptation for projection type \'{self.projection_type}\' not supported.")
    #
    #     # Get distribution
    #     dist = self.get_dist(out)
    #
    #     # Get log probabilities
    #     if self.projection_type in ["policy_clipping", "weighted_scaling_policy_clipping"]:
    #         # Apply log_prob adjustment of clipping based on https://arxiv.org/pdf/1802.07564v2.pdf
    #         self.clipped_gaussian.mean = out["loc"]
    #         self.clipped_gaussian.var = out["scale"]
    #         self.clipped_gaussian.low = out["clip_min"]
    #         self.clipped_gaussian.high = out["clip_max"]
    #         out["log_prob"] = self.clipped_gaussian.log_prob(out["action"])
    #     else:
    #         out["log_prob"] = self.get_logprobs(out["action"], dist)
    #
    #     # Pre-compute upper bound for weighted_scaling
    #     timestep_idx = out["observation", "timestep"].squeeze(0)
    #     out["ub"] = out["observation", "realized_demand"].gather(-1, timestep_idx.unsqueeze(-1)).squeeze(-1)
    #
    #     # Store raw action before projection
    #     raw = out["action"].clone()
    #     out["raw_action"] = raw
    #
    #     # compute jacobian using active raw action
    #     if ("action_mask" in out["observation"]):
    #         # todo: check if this is generally effective
    #         out["log_prob"], out["sample_log_prob"] = self.masked_subspace_logprobs(
    #             out, raw, clamp_min=-50.0, handle_jacobian_adjustment=self.handle_jacobian_adjustment, jacobian_adaptation=self.jacobian_adaptation
    #         )
    #     else:
    #         # todo: check if jacobian adjustment is added effectively!
    #         # Ensure sample_log_prob exists
    #         logp_full = out["log_prob"]  # shape [..., N]
    #         out["sample_log_prob"] = logp_full.sum(dim=-1)  # shape [...]
    #
    #         # Apply jacobian correction in full space if implemented for projection_type
    #         if self.jacobian_correction:
    #             J, log_abs_det = self.handle_jacobian_adjustment(out)  # returns [..., N, N] or None
    #             if J is not None:
    #                 out["adj_sample_log_prob"]  = self.jacobian_adaptation(out["sample_log_prob"], jacobian=J, log_abs_det=log_abs_det).clamp(min=-20.0, max=20)
    #
    #     # Use critic for FW improvement direction if implemented
    #     if self.projection_type == "frank_wolfe":
    #         out["critic_fn"] = out.get("critic_fn", None)
    #
    #
    #     # project once before env execution
    #     out = self.handle_action_projection(out)
    #     out["observation", "env_action"] = out["action"]
    #     return out

def torch_packbits_bool(x: torch.Tensor) -> torch.Tensor:
    """
    x: bool tensor [..., m]
    returns: uint8 tensor [..., ceil(m/8)] packed along last dim
    """
    assert x.dtype == torch.bool
    m = x.size(-1)
    nbytes = (m + 7) // 8
    pad = nbytes * 8 - m
    if pad:
        x = F.pad(x, (0, pad), value=False)

    x = x.reshape(*x.shape[:-1], nbytes, 8).to(torch.uint8)  # 0/1
    weights = (1 << torch.arange(8, device=x.device, dtype=torch.uint8))  # [8] uint8
    packed = (x * weights).sum(dim=-1)  # becomes int64 by reduction
    return packed.to(torch.uint8)       # safe: values in [0,255]

def torch_unpackbits_uint8(packed: torch.Tensor, m: int) -> torch.Tensor:
    """
    packed: uint8 tensor [P, nbytes]
    returns: bool tensor [P, m]
    """
    assert packed.dtype == torch.uint8
    device = packed.device
    bits = ((packed.unsqueeze(-1) >> torch.arange(8, device=device, dtype=torch.uint8)) & 1).to(torch.bool)
    bits = bits.reshape(packed.shape[0], -1)
    return bits[:, :m]
