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
    """
    UVP 2.0 (Jacobian-friendly):
      x_{k+1} = x_k - eta(s) * Atilde^T * relu(Atilde x_k - btilde)

    - Integrates nonnegativity (and optional upper bounds) as linear constraints.
    - Uses a deterministic Lipschitz step eta(s)=1/(||Atilde||^2 + rho) (no branching).
    - Fixed K iterations (no early stopping) to keep the map explicit and Jacobian-correctable.
    """

    def __init__(self,**kwargs):
        super().__init__()
        self.K = kwargs.get('max_iter', 30)
        self.eta = kwargs.get('alpha', 0.01)
        self.delta = kwargs.get('delta', 0.01)
        self.scale = kwargs.get('scale', 0.001)
        self.use_early_stopping = kwargs.get('use_early_stopping', False)  # Not used; fixed iters
        self.rho = kwargs.get('rho', 1e-12)
        # True: power iteration approx ||A||_2^2, False: Frobenius ||A||_F^2
        self.use_spectral_eta = kwargs.get('use_spectral_eta', True)
        self.power_iters = kwargs.get('power_iters', 5)
        print(f"InnerConvexViolationProjection: K={self.K}, eta method={'spectral' if self.use_spectral_eta else 'frobenius'}, "
              f"add_nonnegativity={self.add_nonnegativity}, early_stopping={self.use_early_stopping}")

    def _eta_frobenius(self, Atilde: Tensor) -> Tensor:
        # eta = 1 / (||A||_F^2 + rho) per batch-step
        # ||A||_F^2: sum over (m',n)
        denom = torch.sum(Atilde * Atilde, dim=(-2, -1)) + self.rho  # [B,S]
        return 1.0 / denom

    def _eta_spectral(self, Atilde: Tensor) -> Tensor:
        # eta = 1 / (||A||_2^2 + rho) via power iteration on (A^T A)
        B, S, m, n = Atilde.shape
        v = torch.randn((B, S, n, 1), device=Atilde.device, dtype=Atilde.dtype)
        v = v / (torch.norm(v, dim=-2, keepdim=True) + self.rho)

        for _ in range(self.power_iters):
            Av = torch.matmul(Atilde, v)                       # [B,S,m,1]
            AtAv = torch.matmul(Atilde.transpose(-2, -1), Av)  # [B,S,n,1]
            v = AtAv / (torch.norm(AtAv, dim=-2, keepdim=True) + self.rho)

        Av = torch.matmul(Atilde, v)
        num = torch.sum(Av * Av, dim=(-2, -1))  # ||A v||^2  [B,S]
        # For unit v, ||A v||^2 estimates ||A||_2^2
        return 1.0 / (num + self.rho)

    def forward(self, x, A, b, var_mask=None, **kwargs):
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: 'b' dim 2/3, 'A' dim 3/4.")

        b_ = b.unsqueeze(1) if b.dim() == 2 else b
        A_ = A.unsqueeze(1) if A.dim() == 3 else A
        x_ = x.unsqueeze(1) if x.dim() == 2 else x

        if torch.isnan(x_).any():
            out = x_.squeeze(1) if b.dim() == 2 else x_
            return out * var_mask if var_mask is not None else out

        eta = (self._eta_spectral(A_) if self.use_spectral_eta else self._eta_frobenius(A_)).unsqueeze(-1)

        for _ in range(self.K):
            r = torch.matmul(A_, x_.unsqueeze(-1)).squeeze(-1) - b_  # [B,S,m]
            v = torch.relu(r)
            g = torch.matmul(A_.transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1)  # [B,S,n]
            x_ = x_ - eta * g

            if var_mask is not None:
                x_ = x_ * var_mask

        if b.dim() == 2:
            x_ = x_.squeeze(1)
        return x_


class BoundConvexViolationProjection(nn.Module):
    """
    Convex violation projection layer with optional variable and constraint masking.
    Projects x iteratively onto a feasible region defined by Ax <= b (or similar),
    respecting frozen variables and optionally ignoring masked constraints.
    """

    # todo: this one diverges; needs investigation

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

        # If n_step == 1, x_ is expected to have a singleton step dim at 1.
        if n_step == 1:
            x_ = x_.squeeze(1)

        # Apply variable mask before returning
        if var_mask is not None:
            return x_ * var_mask  # relies on same broadcasting

        return x_

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
