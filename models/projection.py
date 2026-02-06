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
    UVP 2.0 + UVP-tightening + (optional) UVP->policy alpha-map.

    Stage 1 (UVP on tightened constraints):
        x <- x - eta(A) * A^T * relu(Ax - (b - mu_inside))    for K steps

    Stage 2 (alpha-map from UVP anchor toward original x_in), applied only if UVP anchor is feasible:
        d = x_in - x_uvp
        alpha* = min_{i: (Ad)_i > 0} (b_i - (Ax_uvp)_i) / (Ad)_i
        alpha = clip(alpha* - eps_inside, 0, 1)
        x <- x_uvp + alpha d

    Extras:
      - Optional row-normalization of constraints (recommended).
      - Deterministic spectral eta (no RNG).
      - Alpha-map gating based on max constraint violation after UVP.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.K = int(kwargs.get("max_iter", 30))
        self.rho = float(kwargs.get("rho", 1e-12))

        # step size controls
        self.use_spectral_eta = bool(kwargs.get("use_spectral_eta", True))
        self.power_iters = int(kwargs.get("power_iters", 5))

        # UVP improvements
        self.row_normalize = bool(kwargs.get("row_normalize", True))
        self.mu_inside = float(kwargs.get("mu_inside", 1e-3))         # meaningful when row_normalize=True

        # Optional alpha-map controls
        self.enable_alpha_map = bool(kwargs.get("enable_alpha_map", True))
        self.alpha_feas_tol = float(kwargs.get("alpha_feas_tol", 1e-7))
        self.eps_inside = float(kwargs.get("eps_inside", 1e-6))
        self.normalize_d = bool(kwargs.get("normalize_d", False))     # usually False for alpha-map

    def _eta_frobenius(self, A: Tensor) -> Tensor:
        denom = torch.sum(A * A, dim=(-2, -1)) + self.rho  # [B,S]
        return 1.0 / denom

    def _eta_spectral(self, A: Tensor) -> Tensor:
        # Deterministic power iteration for ||A||_2^2 (avoid RNG)
        B, S, m, n = A.shape
        v = torch.ones((B, S, n, 1), device=A.device, dtype=A.dtype)
        v = v / (torch.norm(v, dim=-2, keepdim=True) + 1e-12)

        for _ in range(self.power_iters):
            Av = torch.matmul(A, v)                        # [B,S,m,1]
            AtAv = torch.matmul(A.transpose(-2, -1), Av)   # [B,S,n,1]
            v = AtAv / (torch.norm(AtAv, dim=-2, keepdim=True) + 1e-12)

        Av = torch.matmul(A, v)
        num = torch.sum(Av * Av, dim=(-2, -1))             # [B,S] ~ ||A||_2^2
        return 1.0 / (num + self.rho)

    def _normalize_constraints(self, A: Tensor, b: Tensor):
        # expects A: [B,S,m,n], b: [B,S,m]
        if not self.row_normalize:
            return A, b
        row_norm = torch.norm(A, dim=-1, keepdim=True).clamp(min=1e-12)  # [B,S,m,1]
        A_work = A / row_norm
        b_work = b / row_norm.squeeze(-1)
        return A_work, b_work

    def get_eta(self, A: Tensor, b: Tensor) -> Tensor:
        if A.dim() not in (3, 4) or b.dim() not in (2, 3):
            raise ValueError("get_eta expects A dim 3/4 and b dim 2/3.")

        b_ = b.unsqueeze(1) if b.dim() == 2 else b
        A_ = A.unsqueeze(1) if A.dim() == 3 else A
        A_work, _ = self._normalize_constraints(A_, b_)
        eta = (self._eta_spectral(A_work) if self.use_spectral_eta else self._eta_frobenius(A_work)).unsqueeze(-1)
        return eta

    def _alpha_map_from_anchor(self, x_anchor: Tensor, x_target: Tensor, A: Tensor, b: Tensor) -> Tensor:
        """
        x_anchor: [B,S,n]  (UVP output)
        x_target: [B,S,n]  (original proposal)
        A:        [B,S,m,n]
        b:        [B,S,m]
        """
        d = x_target - x_anchor  # [B,S,n]
        if self.normalize_d:
            d = d / torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-12)

        Ax = torch.matmul(A, x_anchor.unsqueeze(-1)).squeeze(-1)      # [B,S,m]
        Ad = torch.matmul(A, d.unsqueeze(-1)).squeeze(-1)             # [B,S,m]
        slack = b - Ax                                                # TRUE slack, [B,S,m]

        inf = torch.tensor(float("inf"), device=x_anchor.device, dtype=x_anchor.dtype)
        alpha_i = torch.where(Ad > 0, slack / (Ad + 1e-12), inf)       # [B,S,m]
        alpha = torch.amin(alpha_i, dim=-1, keepdim=True)              # [B,S,1]

        # If no Ad>0, alpha=inf => unblocked => alpha=1
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))

        alpha = torch.clamp(alpha - self.eps_inside, min=0.0, max=1.0)
        return x_anchor + alpha * d

    def forward(self, x: Tensor, A: Tensor, b: Tensor, var_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: 'b' dim 2/3, 'A' dim 3/4.")

        # Shape to [B,S,...]
        b_ = b.unsqueeze(1) if b.dim() == 2 else b                   # [B,S,m]
        A_ = A.unsqueeze(1) if A.dim() == 3 else A                   # [B,S,m,n]
        x_in = x.unsqueeze(1) if x.dim() == 2 else x                 # [B,S,n]

        if torch.isnan(x_in).any():
            out = x_in.squeeze(1) if b.dim() == 2 else x_in
            return out * var_mask if var_mask is not None else out

        # Optional constraint row normalization
        A_work, b_work = self._normalize_constraints(A_, b_)

        # Tighten constraints to encourage interior solutions
        b_tight = b_work - self.mu_inside

        # Step size eta depends only on A_work
        eta = (self._eta_spectral(A_work) if self.use_spectral_eta else self._eta_frobenius(A_work)).unsqueeze(-1)  # [B,S,1]

        # Stage 1: UVP on tightened constraints
        x_ = x_in
        for _ in range(self.K):
            r = torch.matmul(A_work, x_.unsqueeze(-1)).squeeze(-1) - b_tight         # [B,S,m]
            v = torch.relu(r)                                                        # [B,S,m]
            g = torch.matmul(A_work.transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1)  # [B,S,n]
            x_ = x_ - eta * g
            if var_mask is not None:
                x_ = x_ * var_mask

        # Stage 2: alpha-map back toward original proposal, only if feasible wrt TRUE constraints
        if self.enable_alpha_map:
            Ax_true = torch.matmul(A_work, x_.unsqueeze(-1)).squeeze(-1)              # [B,S,m]
            max_viol = (Ax_true - b_work).clamp(min=0.0).amax(dim=-1, keepdim=True)   # [B,S,1]
            do_alpha = (max_viol <= self.alpha_feas_tol)                               # [B,S,1] bool

            x_target = x_in
            if var_mask is not None:
                x_target = x_target * var_mask

            x_alpha = self._alpha_map_from_anchor(x_, x_target, A_work, b_work)
            x_ = torch.where(do_alpha, x_alpha, x_)

            if var_mask is not None:
                x_ = x_ * var_mask

        # Back to [B,n] if input had no step dimension
        if b.dim() == 2:
            x_ = x_.squeeze(1)

        return x_


class BoundConvexViolationProjection(nn.Module):
    """
    Convex violation projection layer with optional variable and constraint masking.
    Projects x iteratively onto a feasible region defined by Ax <= b (or similar),
    respecting frozen variables and optionally ignoring masked constraints.
    """

    # todo: this one diverges; needs investigation / remove!

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


class AlphaChebyshevProjection(nn.Module):
    """
    Classical alpha-mapping with an interior anchor y0(s) chosen as the Chebyshev center.

    Given polytope:  P(s) = {x: A(s) x <= b(s)} (and optionally x >= 0)
    Compute Chebyshev center (x0, r):
        maximize r
        s.t. A_i x0 + r * ||A_i||_2 <= b_i   for all i
             (optional) x0 >= 0

    Then alpha-map an unconstrained proposal x_hat to P(s) using a ray from x0 toward x_hat:
        x(alpha) = x0 + alpha (x_hat - x0),  alpha in [0,1]
        alpha* = min_i  (b_i - A_i x0) / (A_i (x_hat - x0))   over i with denom > 0, clipped to [0,1]
        x_proj = x0 + alpha* (x_hat - x0)

    Notes:
      - Computing Chebyshev center is an LP/SOCP-type solve per sample (state-dependent); expensive.
      - This implementation uses cvxpy if available; otherwise raises an ImportError.

    Shapes:
      - x_hat: [B,n] or [B,S,n]
      - A:     [B,m,n] or [B,S,m,n]
      - b:     [B,m] or [B,S,m]
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.add_nonnegativity = bool(kwargs.get("add_nonnegativity", True))
        self.eps_inside = float(kwargs.get("eps_inside", 1e-9))  # numeric margin
        self.clip_alpha = True

        # solver controls (cvxpy)
        self.cvx_solver = kwargs.get("cvx_solver", None)  # e.g., "ECOS", "SCS"
        self.cvx_verbose = bool(kwargs.get("cvx_verbose", False))

    @staticmethod
    def _alpha_map_from_center(x_hat: Tensor, A: Tensor, b: Tensor, x0: Tensor, eps_inside: float) -> Tensor:
        """
        x_hat: [n], A: [m,n], b: [m], x0: [n]
        returns x_proj: [n]
        """
        d = x_hat - x0                                  # [n]
        Ax0 = A @ x0                                    # [m]
        Ad  = A @ d                                     # [m]
        slack = (b - Ax0).clamp(min=0.0)                # [m]

        # constraints with Ad <= 0 do not limit alpha along the ray
        # alpha_i = slack_i / Ad_i for Ad_i > 0
        inf = torch.tensor(float("inf"), device=x_hat.device, dtype=x_hat.dtype)
        alpha_i = torch.where(Ad > 0, slack / (Ad + 1e-12), inf)

        alpha_star = torch.min(alpha_i)
        alpha_star = torch.clamp(alpha_star - eps_inside, min=0.0, max=1.0)  # keep inside & clip to [0,1]
        return x0 + alpha_star * d

    def _chebyshev_center_one(self, A: Tensor, b: Tensor) -> Tensor:
        """
        Solve Chebyshev center for a single polytope A x <= b.

        Returns x0: [n] on CPU tensor; caller moves to device if needed.
        """
        try:
            import cvxpy as cp
        except Exception as e:
            raise ImportError(
                "AlphaChebyshevProjection requires cvxpy. Install cvxpy (and a solver like ECOS/SCS)."
            ) from e

        # Move to CPU numpy for cvxpy
        A_np = A.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()

        m, n = A_np.shape
        x = cp.Variable(n)
        r = cp.Variable()

        row_norms = (A_np**2).sum(axis=1) ** 0.5  # ||A_i||_2

        constraints = [A_np @ x + r * row_norms <= b_np]
        if self.add_nonnegativity:
            constraints += [x >= 0]

        prob = cp.Problem(cp.Maximize(r), constraints)

        # Pick solver if provided
        if self.cvx_solver is None:
            prob.solve(verbose=self.cvx_verbose)
        else:
            prob.solve(solver=self.cvx_solver, verbose=self.cvx_verbose)

        if x.value is None:
            # Infeasible or solver failed; fall back to a feasible-ish point: zeros (may be infeasible)
            x0 = torch.zeros((n,), dtype=A.dtype)
        else:
            x0 = torch.tensor(x.value, dtype=A.dtype)

        return x0

    def forward(self, x_hat: Tensor, A: Tensor, b: Tensor, var_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: 'b' dim 2/3, 'A' dim 3/4.")

        b_ = b.unsqueeze(1) if b.dim() == 2 else b     # [B,S,m]
        A_ = A.unsqueeze(1) if A.dim() == 3 else A     # [B,S,m,n]
        x_ = x_hat.unsqueeze(1) if x_hat.dim() == 2 else x_hat  # [B,S,n]

        B, S, m, n = A_.shape
        out = torch.empty_like(x_)

        # NOTE: Chebyshev solve per (B,S) element (slow but correct).
        for bi in range(B):
            for si in range(S):
                A_bs = A_[bi, si]   # [m,n]
                b_bs = b_[bi, si]   # [m]
                x0 = self._chebyshev_center_one(A_bs, b_bs).to(device=x_.device, dtype=x_.dtype)  # [n]
                out[bi, si] = self._alpha_map_from_center(x_[bi, si], A_bs, b_bs, x0, self.eps_inside)

        if var_mask is not None:
            out = out * var_mask

        if b.dim() == 2:
            out = out.squeeze(1)  # [B,n]

        return out



class ProjectionFactory:
    _class_map = {
        'linear_violation':InnerConvexViolationProjection,
        'linear_violation_policy_clipping':InnerConvexViolationProjection,
        'inner_convex_violation':InnerConvexViolationProjection,
        'inner_convex_violation_alpha':InnerConvexViolationProjection,
        'bound_convex_violation':BoundConvexViolationProjection,
        'alpha_chebyshev':AlphaChebyshevProjection,
        'convex_program':CvxpyProjectionLayer,
        'convex_program_policy_clipping':CvxpyProjectionLayer,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict) -> nn.Module:
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
