# Datatypes
from typing import Optional, Tuple, Dict, Union, Sequence
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Distribution, TransformedDistribution

# TorchRL
from torchrl.modules import ProbabilisticActor
from torchrl.data.tensor_specs import Composite, TensorSpec

# Custom
from environment.utils import compute_violation
from rl_algorithms.clipped_gaussian import ClippedGaussian

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
                 **kwargs):
        super().__init__(module, in_keys, out_keys, spec=spec, **kwargs)

        # Initialize projection layer
        self.projection_layer = projection_layer
        self.projection_type = projection_type.lower()

        # Initialize clipped Gaussian
        initial_loc = torch.zeros(spec.shape, device=spec.device, dtype=spec.dtype)
        initial_scale = torch.ones(spec.shape, device=spec.device, dtype=spec.dtype)
        clip_min = spec.space.low
        clip_max = spec.space.high
        self.clipped_gaussian = ClippedGaussian(initial_loc, initial_scale, clip_min, clip_max)

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
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        return out["action"]

    def bound_violation_projection(self, out:TensorDict) -> Tensor:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"], var_mask=out["observation", "action_mask"].float())
        return out["action"]

    def violation_projection_policy_clipping(self, out:TensorDict) -> Tensor:
        out["action"] = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
        out["action"] = self.policy_clipping_projection(out)
        return out["action"]

    def identity_fn(self, out:TensorDict) -> Tensor:
        return out["action"]

    def handle_action_projection(self, out:TensorDict) -> Tensor:
        """Handle with policy projection"""
        projection_methods = {
            "weighted_scaling": self.weighted_scaling_projection,
            "linear_violation": self.violation_projection,
            "linear_violation_policy_clipping": self.violation_projection_policy_clipping,
            "inner_convex_violation": self.violation_projection,
            "bound_convex_violation": self.bound_violation_projection,
            "policy_clipping": self.policy_clipping_projection,
            "weighted_scaling_policy_clipping": self.weighted_scaling_policy_clipping_projection,
            "convex_program":self.convex_program,
            "convex_program_policy_clipping":self.convex_program_policy_clipping,

        }
        projection_fn = projection_methods.get(self.projection_type, self.identity_fn)
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

    def jacobian_violation(self, out:TensorDict) -> Tensor:
        """Compute the Jacobian of the violation projection:
            J_g(x) = I + lr * A^T D A, where D = diag((Ax-b) > 0)"""
        # Input
        x = out["action"]
        A = out["lhs_A"]
        b = out["rhs"]
        lr = self.projection_layer.lr

        # Shapes
        if A.dim() == 2:
            batch, n = A.shape
        elif A.dim() == 3:
            batch, _, n = A.shape
            shape = (batch, n, n)
        elif A.dim() == 4:
            batch, seq, _, n = A.shape
            shape = (batch, seq, n, n)
        else:
            raise ValueError(f"Invalid dimension of A: {A.dim()}")

        # Compute
        I = torch.eye(n, device=x.device).expand(shape)  # Identity matrix for each batch
        violation = compute_violation(x, A, b)
        D = torch.diag_embed(violation > 0).float()

        if A.dim() == 3:
            jacobian = I + lr * torch.bmm(A.transpose(-2, -1), D).bmm(A)
        elif A.dim() > 3:
            jacobian = I + lr * torch.matmul(A.transpose(-2, -1), torch.matmul(D, A))
        else:
            raise ValueError(f"Invalid dimension of A: {A.dim()}")
        return jacobian

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
        jacobian_methods = {
            "weighted_scaling": self.jacobian_weighted_scaling,
            "linear_violation": self.jacobian_violation,
            "inner_convex_violation": self.jacobian_violation,
            "bound_convex_violation": self.jacobian_violation_bound,
            "weighted_scaling_policy_clipping": self.jacobian_weighted_scaling,
        }
        jacobian_fn = jacobian_methods.get(self.projection_type, None)
        return jacobian_fn(out) if jacobian_fn else None

    def jacobian_adaptation(self, logprob:Tensor, jacobian:Tensor, epsilon:float=1e-8) -> Tensor:
        """Perform logprob adaptation for invertible and differentiable (bijective) functions with non-singular Jacobians.
        - log pi'(x|s) = log pi(x|s) - log|det(J_g(x))|
        """
        # If no Jacobian is provided, return the original log probabilities
        if jacobian is None:
            return logprob

        # log pi'(x|s) = log pi(x|s) - log|det(J_g(x))|
        _, log_abs_det = torch.linalg.slogdet(jacobian) # Compute the sign and log absolute determinant of the Jacobian
        logprob_ = logprob - log_abs_det.unsqueeze(-1) # Note log_abs_det is a scalar applied to all batch elements
        # This is appropriate for our projection functions, as the actions are transformed globally rather than per element
        return logprob_ # Apply reduction for loss computations. Shape: [Batch]

    def forward(self, *args, **kwargs,) -> TensorDict:
        if "action" in kwargs:
            out = args[0]
            out["action"] = out["unprojected_action"].clone()
        else:
            out = super().forward(*args, **kwargs)

        # Raise error for projection layers without log prob adaptation implementations
        if self.projection_type not in ["linear_violation", "linear_violation_policy_clipping", "inner_convex_violation",
                                        "bound_convex_violation", "weighted_scaling_policy_clipping", "none",
                                        "quadratic_program", "convex_program", "convex_program_policy_clipping"]:
            raise ValueError(f"Log prob adaptation for projection type \'{self.projection_type}\' not supported.")

        # Get distribution
        dist = self.get_dist(out)
        out["unprojected_action"] = out["action"].clone()

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

        # Mask out the action
        if "action_mask" in out["observation"]:
            out["action"] = torch.where(out["observation", "action_mask"], out["action"], 0)

        # Pre-compute upper bound for weighted_scaling
        timestep_idx = out["observation", "timestep"].squeeze(0)
        out["ub"] = out["observation", "realized_demand"].gather(-1, timestep_idx.unsqueeze(-1)).squeeze(-1)

        # Projection and log probs adjustment
        out["action"] = self.handle_action_projection(out)
        jacobian = self.handle_jacobian_adjustment(out)
        out["log_prob"] = self.jacobian_adaptation(out["log_prob"], jacobian=jacobian).clamp(min=-50)

        # Mask out the action again; after projection
        if "action_mask" in out["observation"]:
            out["action"] = torch.where(out["observation", "action_mask"], out["action"],  1e-6)
            out["log_prob"] = torch.where(out["observation", "action_mask"], out["log_prob"], torch.tensor(-100, device=out["log_prob"].device))

        # Get sample log probabilities for loss computations
        out["sample_log_prob"] = out["log_prob"].sum(dim=-1)
        return out