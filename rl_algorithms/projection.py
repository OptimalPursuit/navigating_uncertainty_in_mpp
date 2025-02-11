import torch as th
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class EmptyLayer(th.nn.Module):
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x, **kwargs):
        return x

class LinearViolationAdaption(th.nn.Module):
    """Convex violation layer to enforce soft feasibility by projecting solutions towards feasible region."""

    def __init__(self, **kwargs):
        super(LinearViolationAdaption, self).__init__()
        self.alpha = kwargs.get('alpha', 0.005)
        self.delta = kwargs.get('delta', 0.1)
        self.max_iter = kwargs.get('max_iter', 100)
        self.use_early_stopping = kwargs.get('use_early_stopping', False)

    def forward(self, x, A, b, **kwargs):
        """
        - alpha => 0.04 diverges, 0.025 might also cause nans
        - delta, tolerance = 0.05 causes overshooting demand

        Good settings (project per t): alpha=0.005, delta=0.01, tolerance=0.01
        """
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
        active_mask = th.ones(batch_size, n_step, dtype=th.bool, device=x.device)  # Start with all batches active

        # Start loop with early exit in case of nans
        if th.isnan(x_).any():
            return x_
        count = 0
        while th.any(active_mask):
            # Compute current violation for each batch and step
            violation_new = th.clamp(th.matmul(x_.unsqueeze(2), A.transpose(-2, -1)).squeeze(2) - b, min=0)
            # Shape: [batch_size, n_step, m]
            total_violation = th.sum(violation_new, dim=-1)  # Sum violations in [batch_size, n_step]

            # Define batch-wise stopping conditions
            no_violation = total_violation < self.delta

            # Update active mask: only keep batches and steps that are not within tolerance
            active_mask = ~(no_violation)

            # Break if no batches/steps are left active
            if self.use_early_stopping and not th.any(active_mask):
                break

            # Calculate penalty gradient for adjustment
            penalty_gradient = th.matmul(violation_new.unsqueeze(2), A).squeeze(2)  # Shape: [32, 1, 20]

            # Apply penalty gradient update only for active batches/steps
            x_ = th.where(active_mask.unsqueeze(2), x_ - self.alpha * penalty_gradient, x_)
            x_ = th.clamp(x_, min=0) # Ensure non-negativity

            count += 1
            if count > self.max_iter:
                break
        # Return the adjusted x_, reshaped to remove n_step dimension if it was initially 2D
        return x_.squeeze(1) if n_step == 1 else x_

class ProjectionFactory:
    """
    Projection layers to project the input tensor to the feasible set.
    - Convex programming:   Use cvxpylayers to solve a convex optimization problem.
    - Linear programming:   Call CPU solver to solve a linear programming problem.
    - Differentiable lp:    Use differentiable linear programming to solve a linear programming problem.
    - Maximum violation:    Scale action with respect to maximum constraint violation.
    - None:                 Empty layer.
    """
    _class_map = {
        'linear_violation':LinearViolationAdaption,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict):
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
