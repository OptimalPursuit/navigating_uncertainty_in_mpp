import torch
import torch.nn as nn
from torch import Tensor
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional

class EmptyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x:Tensor, **kwargs) -> Tensor:
        return x

class LinearViolationAdaption(nn.Module):
    """Convex violation layer to enforce soft feasibility by projecting solutions towards feasible region."""

    def __init__(self, **kwargs):
        super(LinearViolationAdaption, self).__init__()
        self.alpha = kwargs.get('alpha', 0.005)
        self.scale = kwargs.get('scale', 0.001)
        self.delta = kwargs.get('delta', 0.1)
        self.max_iter = kwargs.get('max_iter', 100)
        self.use_early_stopping = kwargs.get('use_early_stopping', True)

    def forward(self, x:Tensor, A:Tensor, b:Tensor, **kwargs) -> Tensor:
        # Raise error is dimensions are invalid
        if b.dim() not in [2, 3] or A.dim() not in [3, 4]:
            raise ValueError("Invalid dimensions: 'b' must have dim 2 or 3 and 'A' must have dim 3 or 4.")

        # Shapes
        batch_size = b.shape[0]
        m = b.shape[-1]
        n_step = 1 if b.dim() == 2 else b.shape[-2] if b.dim() == 3 else None

        # Tensors shapes
        x_ = x.clone()
        b = b.unsqueeze(1) if b.dim() == 2 else b
        A = A.unsqueeze(1) if A.dim() == 3 else A
        x_ = x_.unsqueeze(1) if x_.dim() == 2 else x_
        # Initialize tensors
        active_mask = torch.ones(batch_size, n_step, dtype=torch.bool, device=x.device)  # Start with all batches active

        # Start loop with early exit in case of nans
        if torch.isnan(x_).any():
            return x_.squeeze(1)
        count = 0
        while torch.any(active_mask):
            # Compute current violation for each batch and step
            violation_new = torch.clamp(torch.matmul(x_.unsqueeze(2), A.transpose(-2, -1)).squeeze(2) - b, min=0)
            # Shape: [batch_size, n_step, m]
            total_violation = torch.sum(violation_new, dim=-1)  # Sum violations in [batch_size, n_step]

            # Define batch-wise stopping conditions
            no_violation = total_violation < self.delta

            # Update active mask: only keep batches and steps that are not within tolerance
            active_mask = ~(no_violation)

            # Break if no batches/steps are left active
            if self.use_early_stopping and not torch.any(active_mask):
                break

            # Calculate penalty gradient for adjustment
            penalty_gradient = torch.matmul(violation_new.unsqueeze(2), A).squeeze(2)  # Shape: [32, 1, 20]

            # Apply penalty gradient update only for active batches/steps
            # scale = 1 / (torch.std(penalty_gradient, dim=0, keepdim=True) + 1e-6)
            lr = self.alpha #/ (1 + self.scale * penalty_gradient)
            x_ = torch.where(active_mask.unsqueeze(2), x_ - lr * penalty_gradient, x_)
            x_ = torch.clamp(x_, min=0) # Ensure non-negativity

            count += 1
            if count > self.max_iter:
                break
        # Return the adjusted x_, reshaped to remove n_step dimension if it was initially 2D
        return x_.squeeze(1) if n_step == 1 else x_

class CvxpyProjectionLayer(nn.Module):
    def __init__(self, n_action=80, n_constraints=85, slack_penalty=1, **kwargs):
        """
        n: number of decision variables
        m: number of linear inequality constraints
        slack_penalty: how much to penalize constraint violation (higher = stricter)
        """
        super().__init__()
        self.n = n_action
        self.m = n_constraints
        self.slack_penalty = slack_penalty
        stab_idx = -4

        # Define CVXPY variables and parameters
        x = cp.Variable(n_action)
        s = cp.Variable(4)

        x_raw_param = cp.Parameter(n_action)
        A_param = cp.Parameter((n_constraints, n_action))
        b_param = cp.Parameter(n_constraints)
        lower_param = cp.Parameter(n_action)
        upper_param = cp.Parameter(n_action)

        # Objective: projection + slack penalty
        objective = cp.Minimize(
            0.5 * cp.sum_squares(x - x_raw_param) +
            slack_penalty * cp.sum_squares(s)
        )
        constraints = [
            A_param[:stab_idx] @ x <= b_param[:stab_idx],
            A_param[stab_idx:] @ x <= b_param[stab_idx:] + s, # stability slack
            s >= 0,
            x >= lower_param,
            x <= upper_param
        ]

        problem = cp.Problem(objective, constraints)

        # Wrap in differentiable layer
        self.cvxpy_layer = CvxpyLayer(
            problem,
            parameters=[x_raw_param, A_param, b_param, lower_param, upper_param],
            variables=[x]
        )

    def forward(self, x_raw:Tensor, A:Tensor, b:Tensor, lower:Optional[Tensor]=None, upper:Optional[Tensor]=None) -> Tensor:
        """
        x_raw: [batch, n]
        A: [batch, m, n]
        b: [batch, m]
        lower, upper: [n] or [batch, n] (optional)
        Returns: projected x: [batch, n]
        """
        batch_size = x_raw.shape[0]
        device = x_raw.device

        # Default bounds
        if lower is None:
            lower = torch.zeros_like(x_raw)
        if upper is None:
            upper = torch.ones_like(x_raw) * 100

        # Handle broadcasting if bounds are 1D
        if lower.dim() == 1:
            lower = lower.unsqueeze(0).expand(batch_size, -1)
        if upper.dim() == 1:
            upper = upper.unsqueeze(0).expand(batch_size, -1)

        # Handle batch and step dimensions
        needs_flattening = x_raw.dim() == 3 # if [batch, n_step, n]
        if needs_flattening:
            # Flatten to [batch*n_step, ...] for processing
            x_raw = x_raw.view(-1, x_raw.shape[-1])  # [batch*n_step, n]
            A = A.view(-1, *A.shape[-2:])  # [batch*n_step, m, n]
            b = b.view(-1, b.shape[-1])  # [batch*n_step, m]
            lower = lower.view(-1, lower.shape[-1])  # [batch*n_step, n]
            upper = upper.view(-1, upper.shape[-1])  # [batch*n_step, n]

        # Call the CVXPY layer
        x_proj, = self.cvxpy_layer(x_raw, A, b, lower, upper)

        # Reshape back if necessary
        if needs_flattening:
            x_proj = x_proj.view(batch_size, -1, x_raw.shape[-1])
        return x_proj

class ProjectionFactory:
    _class_map = {
        'linear_violation':LinearViolationAdaption,
        'linear_violation_policy_clipping':LinearViolationAdaption,
        'convex_program':CvxpyProjectionLayer,
        'convex_program_policy_clipping':CvxpyProjectionLayer,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict) -> nn.Module:
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
