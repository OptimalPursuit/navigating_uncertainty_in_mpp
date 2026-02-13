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
        print("jacobian_correction:", self.jacobian_correction)
        breakpoint()

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
            "linear_violation": self.jacobian_uvp_K_steps,
            "inner_convex_violation": self.jacobian_uvp_K_steps,
            "weighted_scaling_policy_clipping": self.jacobian_weighted_scaling,
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


    def weighted_scaling_projection(self, out:TensorDict) -> Tensor:
        return self.weighted_scaling(out["action"], ub=out["ub"])

    def policy_clipping_projection(self, out:TensorDict) -> Tensor:
        return out["action"].clamp(min=out["clip_min"], max=out["clip_max"])

    def weighted_scaling_policy_clipping_projection(self, out:TensorDict) -> Tensor:
        out["action"] = self.weighted_scaling_projection(out)
        return self.policy_clipping_projection(out)

    def quadratic_program(self, out:TensorDict) -> Tensor:
        return self.projection_layer(out["action"], out["lhs_A"], out["rhs"])

    def convex_program(self, out:TensorDict) -> Tensor:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        return out["action"]

    def convex_program_policy_clipping(self, out:TensorDict) -> Tensor:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        out["action"] = self.policy_clipping_projection(out)
        return out["action"]

    def violation_projection(self, out:TensorDict) -> Tensor:
        var_mask = out["observation", "action_mask"].float() if "action_mask" in out["observation"] else None
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], var_mask=var_mask)
        return out["action"]

    def franke_wolfe_improvement(self, out:TensorDict) -> Tensor:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], s=out, critic_fn=out["critic_fn"])
        return out["action"]

    def alpha_cheby_projection(self, out:TensorDict) -> Tensor:
        var_mask = out["observation", "action_mask"].float() if "action_mask" in out["observation"] else None
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], var_mask=var_mask)
        return out["action"]

    def violation_projection_policy_clipping(self, out:TensorDict) -> Tensor:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        out["action"] = self.policy_clipping_projection(out)
        return out["action"]

    def identity_fn(self, out:TensorDict) -> Tensor:
        return out["action"]

    def handle_action_projection(self, out:TensorDict) -> Tensor:
        """Handle with policy projection"""
        projection_fn = self.projection_methods.get(self.projection_type, self.identity_fn)
        return projection_fn(out)

    def jacobian_weighted_scaling(self, out:TensorDict, epsilon:float=1e-8) -> Tensor:
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
        return jacobian

    def jacobian_uvp_K_steps(self, out: TensorDict) -> torch.Tensor:
        """
        Exact Jacobian of the K-step UVP loop, by re-simulating x_k.
        Returns:
          - [B,n,n] or [B,S,n,n]
        """
        x0 = out["action"]
        A = out["lhs_A"]
        b = out["rhs"]

        proj = self.projection_layer

        b_ = b.unsqueeze(1) if b.dim() == 2 else b
        A_ = A.unsqueeze(1) if A.dim() == 3 else A
        x = x0.unsqueeze(1) if x0.dim() == 2 else x0

        B, S, m, n = A_.shape

        A_work, b_work = proj._normalize_constraints(A_, b_)
        b_tight = b_work - proj.mu_inside
        eta = proj.get_eta(A, b).unsqueeze(-1)  # [B,S,1,1]

        I = torch.eye(n, device=x.device, dtype=x.dtype).view(1, 1, n, n).expand(B, S, n, n)
        J = I.clone()

        for _ in range(proj.K):
            r = (A_work @ x.unsqueeze(-1)).squeeze(-1) - b_tight  # [B,S,m]
            D = torch.diag_embed((r > 0).to(x.dtype))  # [B,S,m,m]

            J_step = I - eta * (A_work.transpose(-2, -1) @ (D @ A_work))  # [B,S,n,n]
            J = J_step @ J  # composition

            # advance x (match forward)
            v = torch.relu(r)
            g = (A_work.transpose(-2, -1) @ v.unsqueeze(-1)).squeeze(-1)
            x = x - eta.squeeze(-1) * g  # eta back to [B,S,1]

        if A.dim() == 3:
            J = J.squeeze(1)
        return J

    def jacobian_violation_bound(self, out:TensorDict) -> Tensor:
        """
           Compute the Jacobian of a two-sided violation projection:
               J_g(x) = I + lr * A^T D A
           where D = diag(g'(r)), with g'(r_i) = 1 if r_i>0, -1 if r_i<-epsilon, else 0.

           Supports batch and optional sequence dimensions:
             A: [batch, m, n] or [batch, seq, m, n]
             x: [batch, n] or [batch, seq, n]
             b: [batch, m] or [batch, seq, m]
           """
        # Input
        x = out["action"]
        A = out["lhs_A"]
        b = out["rhs"]
        epsilon = 1e-6
        lr = self.projection_layer.lr

        # Compute residual
        r = torch.matmul(x.unsqueeze(-2), A.transpose(-2, -1)).squeeze(-2) - b  # [batch, n_step, m]

        # Compute signed derivative
        D = torch.zeros_like(r)
        D[r > 0] = 1.0
        D[r < -epsilon] = -1.0  # negative side
        # D[r in [-epsilon, 0]] = 0 implicitly

        # Embed into diagonal matrices
        D_diag = torch.diag_embed(D)  # [batch, m, m] or [batch, seq, m, m]

        # Identity matrix
        n = x.shape[-1]
        if x.dim() == 2:
            I = torch.eye(n, device=x.device).unsqueeze(0)  # [1, n, n]
            jacobian = I + lr * torch.bmm(A.transpose(1, 2), torch.bmm(D_diag, A))  # [batch, n, n]
        elif x.dim() == 3:
            batch, seq, n = x.shape
            I = torch.eye(n, device=x.device).view(1, 1, n, n)  # [1,1,n,n]
            jacobian = I + lr * torch.matmul(A.transpose(-2, -1), torch.matmul(D_diag, A))  # [batch, seq, n, n]
        else:
            raise ValueError(f"Unsupported x dimension: {x.shape}")

        return jacobian

    def handle_jacobian_adjustment(self, out:TensorDict) -> Optional[Tensor]:
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

        sign, log_abs_det = torch.linalg.slogdet(jacobian)
        valid = sign != 0
        # sign > 0 can be considered

        # if invalid: no correction (or clamp)
        log_abs_det = torch.where(valid, log_abs_det, torch.zeros_like(log_abs_det))

        # CoV correction
        return sample_logp - log_abs_det


    def forward(self, *args, **kwargs,) -> TensorDict:
        out = args[0] if "action" in kwargs else super().forward(*args, **kwargs)

        # Raise error for projection layers without log prob adaptation implementations
        if self.projection_type not in self.projection_methods.keys():
            raise ValueError(f"Log prob adaptation for projection type \'{self.projection_type}\' not supported.")

        # Get distribution
        dist = self.get_dist(out)

        # Get log probabilities
        if self.projection_type in ["policy_clipping", "weighted_scaling_policy_clipping"]:
            # Apply log_prob adjustment of clipping based on https://arxiv.org/pdf/1802.07564v2.pdf
            self.clipped_gaussian.mean = out["loc"]
            self.clipped_gaussian.var = out["scale"]
            self.clipped_gaussian.low = out["clip_min"]
            self.clipped_gaussian.high = out["clip_max"]
            out["log_prob"] = self.clipped_gaussian.log_prob(out["action"])
        else:
            out["log_prob"] = self.get_logprobs(out["action"], dist)

        # Pre-compute upper bound for weighted_scaling
        timestep_idx = out["observation", "timestep"].squeeze(0)
        out["ub"] = out["observation", "realized_demand"].gather(-1, timestep_idx.unsqueeze(-1)).squeeze(-1)

        # Store raw action before projection
        raw = out["action"].clone()
        out["raw_action"] = raw

        # compute jacobian using active raw action
        if ("action_mask" in out["observation"]):
            # todo: check if this is generally effective
            out["log_prob"], out["sample_log_prob"] = self.masked_subspace_logprobs(
                out, raw, clamp_min=-50.0, handle_jacobian_adjustment=self.handle_jacobian_adjustment, jacobian_adaptation=self.jacobian_adaptation
            )
        else:
            # todo: check if jacobian adjustment is added effectively!
            # Ensure sample_log_prob exists
            logp_full = out["log_prob"]  # shape [..., N]
            sample_logp = logp_full.sum(dim=-1)  # shape [...]

            # Apply jacobian correction in full space if implemented for projection_type
            if self.jacobian_correction:
                J = self.handle_jacobian_adjustment(out)  # returns [..., N, N] or None
                if J is not None:
                    sample_logp = logp_full.sum(dim=-1)
                    sample_logp = self.jacobian_adaptation(sample_logp, jacobian=J).clamp(min=-50.0)

            # out["log_prob"] = logp_full
            out["sample_log_prob"] = sample_logp

        # Use critic for FW improvement direction if implemented
        if self.projection_type == "frank_wolfe":
            out["critic_fn"] = out.get("critic_fn", None)


        # project once before env execution
        proj_action = self.handle_action_projection(out)
        out["action"] = proj_action
        out["observation", "env_action"] = proj_action
        return out

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
        packed = torch.packbits(mask.to(torch.uint8), dim=-1)             # [M, ceil(N/8)]
        _, inv = torch.unique(packed, dim=0, return_inverse=True)         # [M]
        n_groups = int(inv.max().item()) + 1

        for g in range(n_groups):
            rows = (inv == g).nonzero(as_tuple=True)[0]
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

        return logp_out.reshape(*batch_shape, N), sample.reshape(*batch_shape)