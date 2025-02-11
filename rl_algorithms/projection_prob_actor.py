# Datatypes
from typing import Optional, Tuple, Dict, Union, Sequence
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F

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

    def get_logprobs(self, action, dist):
        """Compute the log probabilities of the actions given the distribution."""
        return dist.base_dist.log_prob(action) # Shape: [Batch, Features]

    # Projections
    @staticmethod
    def weighted_scaling(sample, ub, epsilon=1e-8):
        sum_sample = sample.sum(dim=-1, keepdim=True)
        upper_bound = ub.unsqueeze(-1)
        scaling_factor = upper_bound / (sum_sample + epsilon)  # Avoid division by zero
        scaled_sample = torch.where(
            sum_sample > upper_bound,
            sample * scaling_factor,
            sample
        )
        return scaled_sample

    def weighted_scaling_projection(self, out):
        return self.weighted_scaling(out["action"], ub=out["ub"])

    def policy_clipping_projection(self, out):
        if "clip_min" not in out or "clip_max" not in out:
            raise ValueError("Policy clipping not supported due to absence of clip_min or clip_max in out")
        return out["action"].clamp(min=out["clip_min"], max=out["clip_max"])

    def weighted_scaling_policy_clipping_projection(self, out):
        out["action"] = self.weighted_scaling_projection(out)
        return self.policy_clipping_projection(out)

    def violation_projection(self, out):
        return self.projection_layer(out["action"], out["lhs_A"], out["rhs"])

    def identity_fn(self, out):
        return out["action"]

    def handle_action_projection(self, out):
        """Handle with policy projection"""
        projection_methods = {
            "weighted_scaling": self.weighted_scaling_projection,
            "linear_violation": self.violation_projection,
            "policy_clipping": self.policy_clipping_projection,
            "weighted_scaling_policy_clipping": self.weighted_scaling_policy_clipping_projection,
        }
        projection_fn = projection_methods.get(self.projection_type, self.identity_fn)
        return projection_fn(out)

    def jacobian_weighted_scaling(self, out, epsilon=1e-8):
        """Compute the Jacobian of the direct scaling projection:
            J_g(x) = y / sum(x)^2 * (diag(sum(x)) - x * 1^T)"""
        # Input
        x = out["action"]
        y = out["ub"]

        # Shapes
        batch, n = x.shape

        # Compute
        sum_x = x.sum(dim=-1, keepdim=True)
        scaling_factor = y.unsqueeze(-1) / (sum_x)**2 # Shape: (batch_size, n)
        kronecker_delta = torch.eye(n).unsqueeze(0).expand(batch, n, n).to(x.device)  # Shape: (batch_size, n, n)
        x_product = x.unsqueeze(-1) * torch.ones(x.size(-1), device=x.device)  # Shape: (batch_size, n, n)
        jacobian = scaling_factor.unsqueeze(-1) * (kronecker_delta * sum_x.unsqueeze(-1)) - x_product # Shape: (batch_size, n, n)
        return jacobian

    def jacobian_violation(self, out):
        """Compute the Jacobian of the violation projection:
            J_g(x) = I + alpha * A^T D A, where D = diag((Ax-b) > 0)"""
        # Input
        x = out["action"]
        A = out["lhs_A"]
        b = out["rhs"]
        alpha = self.projection_layer.alpha

        # Shapes
        if A.dim() == 2:
            batch, n = A.shape
        elif A.dim() == 3:
            batch, _, n = A.shape
        elif A.dim() == 4:
            batch, _, _, n = A.shape
        else:
            raise ValueError(f"Invalid dimension of A: {A.dim()}")

        # Compute
        I = torch.eye(n, device=x.device).expand(batch, n, n)  # Identity matrix for each batch
        violation = compute_violation(x, A, b)
        D = torch.diag_embed(violation > 0).float()
        jacobian = I + alpha * torch.bmm(A.transpose(1, 2), D).bmm(A)
        return jacobian

    def handle_jacobian_adjustment(self, out):
        """Handle with Jacobian adjustment of projection methods;
        We only have Jacobian for weighted scaling and linear violation."""
        jacobian_methods = {
            "weighted_scaling": self.jacobian_weighted_scaling,
            "linear_violation": self.jacobian_violation,
            "weighted_scaling_policy_clipping": self.jacobian_weighted_scaling,
        }
        jacobian_fn = jacobian_methods.get(self.projection_type, None)
        return jacobian_fn(out) if jacobian_fn else None

    def jacobian_adaptation(self, logprob, jacobian, epsilon=1e-8) -> Tensor:
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

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        # Get distribution and full log probabilities
        dist = self.get_dist(out)
        out["log_prob"] = self.get_logprobs(out["action"], dist)

        # Raise error for projection layers without log prob adaptation implementations
        if self.projection_type not in ["linear_violation", "weighted_scaling_policy_clipping", "none"]:
            raise ValueError(f"Log prob adaptation for projection type \'{self.projection_type}\' not supported.")

        # Pre-compute upper bound for weighted_scaling
        timestep_idx = out["observation", "timestep"].squeeze(0)
        out["ub"] = out["observation", "realized_demand"].gather(-1, timestep_idx.unsqueeze(-1)).squeeze(-1)

        # Projection and log probs adjustment
        out["action"] = self.handle_action_projection(out)
        jacobian = self.handle_jacobian_adjustment(out)
        out["log_prob"] = self.jacobian_adaptation(out["log_prob"], jacobian=jacobian)

        # Apply log_prob adjustment of clipping based on https://arxiv.org/pdf/1802.07564v2.pdf
        if self.projection_type in ["policy_clipping", "weighted_scaling_policy_clipping"]:
            self.clipped_gaussian.mean = out["loc"]
            self.clipped_gaussian.var = out["scale"]
            self.clipped_gaussian.low = out["clip_min"]
            self.clipped_gaussian.high = out["clip_max"]
            out["log_prob"] = self.clipped_gaussian.log_prob(out["action"])

        # Apply log_prob clamp for numerical stability
        out["log_prob"] = torch.clamp(out["log_prob"], min=-10.0, max=0.0)
        # todo: infeasible action masking (where masked out values get log prob -inf)

        # Get sample log probabilities for loss computations
        out["sample_log_prob"] = out["log_prob"].sum(dim=-1)
        return out