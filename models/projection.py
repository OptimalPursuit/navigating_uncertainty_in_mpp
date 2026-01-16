import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional

class EmptyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x:Tensor, **kwargs) -> Tensor:
        return x

class InnerConvexViolationProjection(nn.Module):
    """Convex violation layer to enforce soft feasibility by projecting solutions inside the feasible region."""

    def __init__(self, **kwargs):
        super(InnerConvexViolationProjection, self).__init__()
        self.lr = kwargs.get('alpha', 0.005)
        self.scale = kwargs.get('scale', 0.001)
        self.delta = kwargs.get('delta', 0.1)
        self.max_iter = kwargs.get('max_iter', 100)
        self.use_early_stopping = kwargs.get('use_early_stopping', True)
        self.use_gradient_scaling = kwargs.get('use_gradient_scaling', True)


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
            scale = torch.norm(penalty_gradient, dim=-1, keepdim=True) + 1e-6 if self.use_gradient_scaling else 1.0

            # Apply penalty gradient update only for active batches/steps
            # scale = 1 / (torch.std(penalty_gradient, dim=0, keepdim=True) + 1e-6)
            update = self.lr * penalty_gradient / scale
            x_ = torch.where(active_mask.unsqueeze(2), x_ - update, x_)
            x_ = torch.clamp(x_, min=0) # Ensure non-negativity

            count += 1
            if count > self.max_iter:
                break
        # Return the adjusted x_, reshaped to remove n_step dimension if it was initially 2D
        return x_.squeeze(1) if n_step == 1 else x_

class BoundConvexViolationProjection(nn.Module):
    """
    Convex violation projection layer with optional variable and constraint masking.
    Projects x iteratively onto a feasible region defined by Ax <= b (or similar),
    respecting frozen variables and optionally ignoring masked constraints.
    """

    def __init__(self, **kwargs):
        super(BoundConvexViolationProjection, self).__init__()
        self.lr = kwargs.get('alpha', 0.005)
        self.scale = kwargs.get('scale', 0.001)
        self.delta = kwargs.get('delta', 0.1)
        self.max_iter = kwargs.get('max_iter', 100)
        self.use_early_stopping = kwargs.get('use_early_stopping', True)
        self.use_gradient_scaling = kwargs.get('use_gradient_scaling', True)

    def forward(
            self,
            x: Tensor,
            A: Tensor,
            b: Tensor,
            var_mask: Tensor = None,  # shape [batch, n] or [n], 1=active,0=frozen
    ) -> Tensor:

        # Check dimensions
        if b.dim() not in [2, 3] or A.dim() not in [3, 4]:
            raise ValueError("Invalid dimensions: 'b' must have dim 2 or 3 and 'A' must have dim 3 or 4.")

        # Shapes
        batch_size = b.shape[0]
        m = b.shape[-1]
        n_step = 1 if b.dim() == 2 else b.shape[-2]

        # Prepare tensors
        x_ = x.clone()
        b = b.unsqueeze(1) if b.dim() == 2 else b  # [batch, 1, m] or [batch, n_step, m]
        A = A.unsqueeze(1) if A.dim() == 3 else A  # [batch, 1, m, n] or [batch, n_step, m, n]
        x_ = x_.unsqueeze(1) if x_.dim() == 2 else x_  # [batch, 1, n] or [batch, n_step, n]

        # Variable mask
        if var_mask is None:
            var_mask = torch.ones(batch_size, x_.shape[-1], device=x.device)
        if var_mask.dim() == 2:
            var_mask = var_mask.unsqueeze(-2) # [batch, n_step, n]

        # Constraint mask: only compute constraints affected by active variables
        constr_mask = (torch.matmul(var_mask.unsqueeze(-2), A.abs().transpose(-2,-1)).squeeze(-2) > 0)  # [batch, n_step, m]

        # Active mask: per batch/step
        active_mask = torch.ones(batch_size, n_step, dtype=torch.bool, device=x.device)

        # Early exit if NaNs
        if torch.isnan(x_).any():
            return x_.squeeze(1) if n_step == 1 else x_

        count = 0
        while torch.any(active_mask):
            # Apply variable mask before computing residual
            x_masked = x_ * var_mask

            # Compute residual: r = Ax - b
            residual = torch.matmul(x_masked.unsqueeze(-2), A.transpose(-2, -1)).squeeze(-2) - b  # [batch, n_step, m]
            residual = residual * constr_mask  # mask inactive constraints

            # Two-sided violation
            violation_term = F.relu(residual) - F.relu(-residual - self.delta)  # [batch, n_step, m]

            # Gradient: A^T g(r)
            penalty_gradient = torch.matmul(violation_term.unsqueeze(-2), A).squeeze(-2)  # [batch, n_step, n]

            # Total violation per batch/step for early stopping
            total_violation = torch.sum(F.relu(residual), dim=-1)  # [batch, n_step]
            active_mask = total_violation >= self.delta  # keep only violating steps active

            if self.use_early_stopping and not torch.any(active_mask):
                break

            # Update only active variables
            scale = torch.norm(penalty_gradient, dim=-1, keepdim=True) + 1e-6 if self.use_gradient_scaling else 1.0
            # print("scale:", scale.mean().item())
            update = self.lr * penalty_gradient / scale
            x_ = torch.where(active_mask.unsqueeze(-1), x_ - update, x_)
            x_ = torch.clamp(x_, min=0)  # enforce non-negativity

            count += 1
            if count >= self.max_iter:
                break

        # Apply variable mask before returning
        x_ = x_ * var_mask
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
        'linear_violation':InnerConvexViolationProjection,
        'linear_violation_policy_clipping':InnerConvexViolationProjection,
        'inner_convex_violation':InnerConvexViolationProjection,
        'bound_convex_violation':BoundConvexViolationProjection,
        'convex_program':CvxpyProjectionLayer,
        'convex_program_policy_clipping':CvxpyProjectionLayer,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict) -> nn.Module:
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
