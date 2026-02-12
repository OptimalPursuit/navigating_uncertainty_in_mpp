import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional, Tuple

class EmptyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x:Tensor, **kwargs) -> Tensor:
        return x

class InnerConvexViolationProjection(nn.Module):
    """
    UVP 2.0 + tightening + (optional) alpha-map.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.K = int(kwargs.get("max_iter", 30))
        self.rho = float(kwargs.get("rho", 1e-12))

        # step size controls
        self.use_spectral_eta = bool(kwargs.get("use_spectral_eta", True))
        self.power_iters = int(kwargs.get("power_iters", 5))
        self.eta_margin = float(kwargs.get("eta_margin", 0.99))

        # UVP improvements
        self.row_normalize = bool(kwargs.get("row_normalize", True))
        self.mu_inside = float(kwargs.get("mu_inside", 1e-3))  # meaningful when row_normalize=True

        # alpha-map controls
        self.enable_alpha_map = bool(kwargs.get("enable_alpha_map", True))
        self.alpha_feas_tol = float(kwargs.get("alpha_feas_tol", 1e-7))
        self.eps_inside = float(kwargs.get("eps_inside", 1e-6))
        self.normalize_d = bool(kwargs.get("normalize_d", False))

        # fairness: enforce nonnegativity
        self.enforce_nonneg = bool(kwargs.get("enforce_nonneg", True))

    def _eta_frobenius(self, A: Tensor) -> Tensor:
        denom = torch.sum(A * A, dim=(-2, -1)) + self.rho  # [B,S]
        return 1.0 / denom

    def _eta_spectral(self, A: Tensor) -> Tensor:
        B, S, m, n = A.shape
        v = torch.ones((B, S, n, 1), device=A.device, dtype=A.dtype)
        v = v / (torch.norm(v, dim=-2, keepdim=True) + 1e-12)
        for _ in range(self.power_iters):
            Av = torch.matmul(A, v)
            AtAv = torch.matmul(A.transpose(-2, -1), Av)
            v = AtAv / (torch.norm(AtAv, dim=-2, keepdim=True) + 1e-12)
        Av = torch.matmul(A, v)
        num = torch.sum(Av * Av, dim=(-2, -1))  # [B,S] ~ ||A||_2^2
        return self.eta_margin / (num + self.rho)

    def _normalize_constraints(self, A: Tensor, b: Tensor):
        if not self.row_normalize:
            return A, b
        row_norm = torch.norm(A, dim=-1, keepdim=True).clamp(min=1e-12)  # [B,S,m,1]
        return A / row_norm, b / row_norm.squeeze(-1)

    def _alpha_map_from_anchor(self, x_anchor: Tensor, x_target: Tensor, A: Tensor, b: Tensor) -> Tensor:
        d = x_target - x_anchor
        if self.normalize_d:
            d = d / torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-12)

        Ax = torch.matmul(A, x_anchor.unsqueeze(-1)).squeeze(-1)  # [B,S,m]
        Ad = torch.matmul(A, d.unsqueeze(-1)).squeeze(-1)         # [B,S,m]
        slack = b - Ax                                            # [B,S,m]

        inf = torch.tensor(float("inf"), device=x_anchor.device, dtype=x_anchor.dtype)
        alpha_i = torch.where(Ad > 0, slack / (Ad + 1e-12), inf)   # [B,S,m]
        alpha = torch.amin(alpha_i, dim=-1, keepdim=True)          # [B,S,1]
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))
        alpha = torch.clamp(alpha - self.eps_inside, min=0.0, max=1.0)

        x_new = x_anchor + alpha * d
        if self.enforce_nonneg:
            x_new = x_new.clamp_min(0.0)
        return x_new

    def forward(self, x: Tensor, A: Tensor, b: Tensor, var_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: 'b' dim 2/3, 'A' dim 3/4.")

        b_ = b.unsqueeze(1) if b.dim() == 2 else b
        A_ = A.unsqueeze(1) if A.dim() == 3 else A
        x_in = x.unsqueeze(1) if x.dim() == 2 else x

        if torch.isnan(x_in).any():
            out = x_in.squeeze(1) if b.dim() == 2 else x_in
            if self.enforce_nonneg:
                out = out.clamp_min(0.0)
            return out * var_mask if var_mask is not None else out

        # fairness: enforce x>=0 at input
        if self.enforce_nonneg:
            x_in = x_in.clamp_min(0.0)

        A_work, b_work = self._normalize_constraints(A_, b_)
        b_tight = b_work - self.mu_inside
        eta = (self._eta_spectral(A_work) if self.use_spectral_eta else self._eta_frobenius(A_work)).unsqueeze(-1)

        x_ = x_in
        for _ in range(self.K):
            r = torch.matmul(A_work, x_.unsqueeze(-1)).squeeze(-1) - b_tight
            v = torch.relu(r)
            g = torch.matmul(A_work.transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1)
            x_ = x_ - eta * g

            if self.enforce_nonneg:
                x_ = x_.clamp_min(0.0)
            if var_mask is not None:
                x_ = x_ * var_mask

        if self.enable_alpha_map:
            Ax_true = torch.matmul(A_work, x_.unsqueeze(-1)).squeeze(-1)
            max_viol = (Ax_true - b_work).clamp(min=0.0).amax(dim=-1, keepdim=True)
            do_alpha = (max_viol <= self.alpha_feas_tol)

            x_target = x_in
            if var_mask is not None:
                x_target = x_target * var_mask

            x_alpha = self._alpha_map_from_anchor(x_, x_target, A_work, b_work)
            x_ = torch.where(do_alpha, x_alpha, x_)

            if self.enforce_nonneg:
                x_ = x_.clamp_min(0.0)
            if var_mask is not None:
                x_ = x_ * var_mask

        if b.dim() == 2:
            x_ = x_.squeeze(1)
        if self.enforce_nonneg:
            x_ = x_.clamp_min(0.0)
        return x_


class CvxpyProjectionLayer(nn.Module):
    """
    QP projection + (optional) slack on last 4 constraints.
    """

    def __init__(self, n_action=80, n_constraints=85, slack_penalty=1.0, stab_idx: int = -4, **kwargs):
        super().__init__()
        self.n = int(n_action)
        self.m = int(n_constraints)
        self.slack_penalty = float(slack_penalty)
        self.stab_idx = int(stab_idx)

        x = cp.Variable(self.n)
        s = cp.Variable(abs(self.stab_idx) if self.stab_idx < 0 else (self.m - self.stab_idx))

        x_raw_param = cp.Parameter(self.n)
        A_param = cp.Parameter((self.m, self.n))
        b_param = cp.Parameter(self.m)
        lower_param = cp.Parameter(self.n)

        objective = cp.Minimize(0.5 * cp.sum_squares(x - x_raw_param) + self.slack_penalty * cp.sum_squares(s))

        # slicing semantics
        if self.stab_idx < 0:
            hard = slice(0, self.stab_idx)
            soft = slice(self.stab_idx, None)
        else:
            hard = slice(0, self.stab_idx)
            soft = slice(self.stab_idx, self.m)

        constraints = [
            A_param[hard, :] @ x <= b_param[hard],
            A_param[soft, :] @ x <= b_param[soft] + s,
            s >= 0,
            x >= lower_param,  # nonnegativity if lower=0
        ]

        problem = cp.Problem(objective, constraints)
        self.cvxpy_layer = CvxpyLayer(problem, parameters=[x_raw_param, A_param, b_param, lower_param], variables=[x])

    def forward(
        self,
        x_raw: Tensor,
        A: Tensor,
        b: Tensor,
        lower: Optional[Tensor] = None,
        upper: Optional[Tensor] = None,  # ignored
    ) -> Tensor:
        batch_size = x_raw.shape[0]

        if lower is None:
            lower = torch.zeros_like(x_raw)
        if lower.dim() == 1:
            lower = lower.unsqueeze(0).expand(batch_size, -1)

        needs_flattening = (x_raw.dim() == 3)
        if needs_flattening:
            x_raw = x_raw.view(-1, x_raw.shape[-1])
            A = A.view(-1, *A.shape[-2:])
            b = b.view(-1, b.shape[-1])
            lower = lower.view(-1, lower.shape[-1])

        x_proj, = self.cvxpy_layer(x_raw, A, b, lower)

        if needs_flattening:
            x_proj = x_proj.view(batch_size, -1, x_raw.shape[-1])

        return x_proj


class AlphaChebyshevProjection(nn.Module):
    """
    Alpha-map with Chebyshev center anchor.

    Shapes:
      x_hat: [B,n] or [B,S,n]
      A:     [B,m,n] or [B,S,m,n]
      b:     [B,m]   or [B,S,m]
    """

    def __init__(self, n_action: int = 80, n_constraints: int = 85, **kwargs):
        super().__init__()
        self.n = int(n_action)
        self.m = int(n_constraints)

        self.add_nonnegativity = bool(kwargs.get("add_nonnegativity", True))
        self.eps_inside = float(kwargs.get("eps_inside", 1e-9))
        self.cvx_solver = kwargs.get("cvx_solver", None)
        self.cvx_verbose = bool(kwargs.get("cvx_verbose", False))
        self.enforce_nonneg = bool(kwargs.get("enforce_nonneg", True))

        # Chebyshev center layer:
        # max r
        # s.t. A x + r * row_norms <= b
        #      x >= 0 (optional)
        #      r >= 0
        x = cp.Variable(self.n)
        r = cp.Variable()

        A_param = cp.Parameter((self.m, self.n))
        b_param = cp.Parameter(self.m)
        d_param = cp.Parameter(self.m, nonneg=True)  # row norms >= 0

        cons = [A_param @ x + cp.multiply(d_param, r) <= b_param, r >= 0]
        if self.add_nonnegativity:
            cons += [x >= 0]

        prob = cp.Problem(cp.Maximize(r), cons)

        self.cheby_layer = CvxpyLayer(
            prob,
            parameters=[A_param, b_param, d_param],
            variables=[x, r],
        )

    @staticmethod
    def _alpha_map_batched(x_hat: Tensor, A: Tensor, b: Tensor, x0: Tensor, eps_inside: float) -> Tensor:
        # x_hat,x0: [B,S,n], A: [B,S,m,n], b: [B,S,m]
        d = x_hat - x0                                      # [B,S,n]
        Ax0 = torch.matmul(A, x0.unsqueeze(-1)).squeeze(-1) # [B,S,m]
        Ad  = torch.matmul(A, d.unsqueeze(-1)).squeeze(-1)  # [B,S,m]
        slack = (b - Ax0).clamp(min=0.0)                    # [B,S,m]

        inf = torch.tensor(float("inf"), device=x_hat.device, dtype=x_hat.dtype)
        alpha_i = torch.where(Ad > 0, slack / (Ad + 1e-12), inf)   # [B,S,m]
        alpha = torch.amin(alpha_i, dim=-1, keepdim=True)          # [B,S,1]
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))
        alpha = torch.clamp(alpha - eps_inside, 0.0, 1.0)

        return x0 + alpha * d

    def forward(self, x_hat: Tensor, A: Tensor, b: Tensor, var_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: b dim 2/3, A dim 3/4 expected.")

        # Normalize to [B,S,*]
        b_BS = b.unsqueeze(1) if b.dim() == 2 else b
        A_BS = A.unsqueeze(1) if A.dim() == 3 else A
        x_BS = x_hat.unsqueeze(1) if x_hat.dim() == 2 else x_hat

        B, S, m, n = A_BS.shape
        if m != self.m or n != self.n:
            raise ValueError(f"Layer configured for (m={self.m}, n={self.n}) but got (m={m}, n={n}).")

        # Compute row norms d = ||A_i||_2 (batched)
        d_BS = torch.norm(A_BS, dim=-1).clamp(min=1e-12)  # [B,S,m]

        # Flatten for batched cvxpylayer call
        A_f = A_BS.reshape(-1, m, n)
        b_f = b_BS.reshape(-1, m)
        d_f = d_BS.reshape(-1, m)

        # Solve Chebyshev centers in batch
        x0_f, r_f = self.cheby_layer(A_f, b_f, d_f, solver_args={"verbose": self.cvx_verbose} if self.cvx_solver is None
                                     else {"solver": self.cvx_solver, "verbose": self.cvx_verbose})

        x0 = x0_f.view(B, S, n)

        # Alpha-map (fully vectorized)
        out = self._alpha_map_batched(x_BS, A_BS, b_BS, x0, self.eps_inside)

        if self.enforce_nonneg:
            out = out.clamp_min(0.0)
        if var_mask is not None:
            out = out * (var_mask.unsqueeze(1) if var_mask.dim() == 2 else var_mask)

        return out.squeeze(1) if b.dim() == 2 else out


class LogBarrierProjection(nn.Module):
    """
    Log-barrier solve for:
        min_x 0.5||x-x_raw||^2 - mu * sum_i log(b_i - (Ax)_i) - mu * sum_j log(x_j - lower_j)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.K = int(kwargs.get("max_iter", 30))
        self.mu = float(kwargs.get("mu", 1e-2))
        self.lr = float(kwargs.get("lr", 1.0))
        self.backtrack = float(kwargs.get("backtrack", 0.5))
        self.max_ls = int(kwargs.get("max_ls", 25))
        self.eps = float(kwargs.get("eps_inside", 1e-6))

        self.row_normalize = bool(kwargs.get("row_normalize", True))
        self.cvx_solver = kwargs.get("cvx_solver", None)
        self.cvx_verbose = bool(kwargs.get("cvx_verbose", False))

    def _normalize_constraints(self, A: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.row_normalize:
            return A, b
        row_norm = torch.norm(A, dim=-1, keepdim=True).clamp(min=1e-12)
        return A / row_norm, b / row_norm.squeeze(-1)

    def _chebyshev_center_one(self, A: Tensor, b: Tensor, lower: Tensor) -> Tensor:
        """
        Chebyshev center with only lower bounds:
            max r
            s.t. A_i x + r||A_i|| <= b_i
                 x >= lower + r
                 r >= 0
        """
        A_np = A.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        lo_np = lower.detach().cpu().numpy()

        m, n = A_np.shape
        x = cp.Variable(n)
        r = cp.Variable()

        row_norms = (A_np**2).sum(axis=1) ** 0.5
        constraints = [
            A_np @ x + r * row_norms <= b_np,
            x >= lo_np + r,
            r >= 0,
        ]
        prob = cp.Problem(cp.Maximize(r), constraints)
        if self.cvx_solver is None:
            prob.solve(verbose=self.cvx_verbose)
        else:
            prob.solve(solver=self.cvx_solver, verbose=self.cvx_verbose)

        if x.value is None:
            # safe-ish fallback: clamp raw lower (still might violate Ax<=b; line search will handle)
            return torch.tensor(lo_np, dtype=torch.float32)
        return torch.tensor(x.value, dtype=torch.float32)

    def _is_strictly_feasible(self, x: Tensor, A: Tensor, b: Tensor, lower: Tensor) -> Tensor:
        Ax = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
        c1 = (Ax <= (b - self.eps)).all(dim=-1, keepdim=True)
        c2 = (x >= (lower + self.eps)).all(dim=-1, keepdim=True)
        return (c1 & c2)

    def forward(
        self,
        x_raw: Tensor,
        A: Tensor,
        b: Tensor,
        lower: Optional[Tensor] = None,
        upper: Optional[Tensor] = None,  # ignored
        var_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: 'b' dim 2/3 and 'A' dim 3/4 expected.")

        b_ = b.unsqueeze(1) if b.dim() == 2 else b
        A_ = A.unsqueeze(1) if A.dim() == 3 else A
        x0 = x_raw.unsqueeze(1) if x_raw.dim() == 2 else x_raw
        B, S, m, n = A_.shape

        if lower is None:
            lower_ = torch.zeros((B, S, n), device=x0.device, dtype=x0.dtype)
        else:
            lower_ = lower
            if lower_.dim() == 1:
                lower_ = lower_.view(1, 1, n).expand(B, S, n)
            elif lower_.dim() == 2:
                lower_ = lower_.unsqueeze(1).expand(B, S, n)
            lower_ = lower_.to(device=x0.device, dtype=x0.dtype)

        if var_mask is not None:
            vm = var_mask
            if vm.dim() == 2:
                vm = vm.unsqueeze(1)
        else:
            vm = None

        A_work, b_work = self._normalize_constraints(A_, b_)

        # Chebyshev init per (B,S)
        x = torch.empty_like(x0)
        for bi in range(B):
            for si in range(S):
                x_cs = self._chebyshev_center_one(A_work[bi, si], b_work[bi, si], lower_[bi, si])
                x[bi, si] = x_cs.to(device=x0.device, dtype=x0.dtype)

        feas = self._is_strictly_feasible(x, A_work, b_work, lower_)
        if not feas.all():
            x = torch.where(feas, x, (x.clamp_min(0.0) + lower_) * 0.5)

        # Barrier descent
        for _ in range(self.K):
            grad = (x - x0)

            Ax = torch.matmul(A_work, x.unsqueeze(-1)).squeeze(-1)
            slack = (b_work - Ax).clamp(min=1e-12)
            g_bar = torch.matmul(A_work.transpose(-2, -1), (self.mu / slack).unsqueeze(-1)).squeeze(-1)
            grad = grad + g_bar

            s_lo = (x - lower_).clamp(min=1e-12)
            grad = grad + self.mu * (1.0 / s_lo)

            if vm is not None:
                grad = grad * vm

            step = self.lr
            x_new = x
            for _ls in range(self.max_ls):
                cand = x - step * grad
                if vm is not None:
                    cand = torch.where(vm.bool(), cand, x)
                ok = self._is_strictly_feasible(cand, A_work, b_work, lower_)
                if ok.all():
                    x_new = cand
                    break
                step *= self.backtrack
            x = x_new

        if vm is not None:
            x = torch.where(vm.bool(), x, x0)

        x = x.clamp_min(0.0)
        if b.dim() == 2:
            x = x.squeeze(1)
        return x


class FrankWolfePolicyImprovement(nn.Module):
    """
    FAIRNESS UPDATE:
      - Removes artificial upper bounds (no x<=upper constraints).
      - Uses lower as nonnegativity (default 0).
      - Keeps stability slack as you had it.
      - Signature keeps 'upper' for compatibility but ignores it.
    """

    def __init__(
        self,
        n_action: int,
        n_constraints: int,
        alpha: float = 0.1,
        stab_idx: int = -4,
        slack_penalty: float = 1.0,
        default_lower: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.n = int(n_action)
        self.m = int(n_constraints)
        self.alpha = float(alpha)
        self.stab_idx = int(stab_idx)
        self.slack_penalty = float(slack_penalty)
        self.default_lower = float(default_lower)

        self.cvx_solver = kwargs.get("cvx_solver", None)
        self.cvx_verbose = bool(kwargs.get("cvx_verbose", False))

        # (1) Feasible anchor QP repair (no upper)
        x = cp.Variable(self.n)
        s = cp.Variable(abs(self.stab_idx) if self.stab_idx < 0 else (self.m - self.stab_idx))

        x_raw_param = cp.Parameter(self.n)
        A_param = cp.Parameter((self.m, self.n))
        b_param = cp.Parameter(self.m)
        lower_param = cp.Parameter(self.n)

        obj = cp.Minimize(0.5 * cp.sum_squares(x - x_raw_param) + self.slack_penalty * cp.sum_squares(s))

        if self.stab_idx < 0:
            hard = slice(0, self.stab_idx)
            soft = slice(self.stab_idx, None)
        else:
            hard = slice(0, self.stab_idx)
            soft = slice(self.stab_idx, self.m)

        cons = [
            A_param[hard, :] @ x <= b_param[hard],
            A_param[soft, :] @ x <= b_param[soft] + s,
            s >= 0,
            x >= lower_param,
        ]
        anchor_problem = cp.Problem(obj, cons)
        self.anchor_layer = CvxpyLayer(
            anchor_problem,
            parameters=[x_raw_param, A_param, b_param, lower_param],
            variables=[x],
        )

        # (2) LMO: argmax <c,g> s.t. A c <= b, c >= lower (no upper)
        c = cp.Variable(self.n)
        g_param = cp.Parameter(self.n)
        A2_param = cp.Parameter((self.m, self.n))
        b2_param = cp.Parameter(self.m)
        lower2_param = cp.Parameter(self.n)

        lmo_obj = cp.Minimize(-g_param @ c)
        lmo_cons = [
            A2_param @ c <= b2_param,
            c >= lower2_param,
        ]
        lmo_problem = cp.Problem(lmo_obj, lmo_cons)
        self.lmo_layer = CvxpyLayer(
            lmo_problem,
            parameters=[g_param, A2_param, b2_param, lower2_param],
            variables=[c],
        )

    def _broadcast_lower(self, lower: Optional[Tensor], ref: Tensor) -> Tensor:
        # ref: [B,S,n]
        B, S, n = ref.shape
        device, dtype = ref.device, ref.dtype
        if lower is None:
            lower = torch.full((B, S, n), self.default_lower, device=device, dtype=dtype)
        else:
            if lower.dim() == 1:
                lower = lower.view(1, 1, n).expand(B, S, n)
            elif lower.dim() == 2:
                lower = lower.unsqueeze(1).expand(B, S, n)
            lower = lower.to(device=device, dtype=dtype)
        return lower

    def feasible_anchor(
        self,
        x: Tensor,
        A: Tensor,
        b: Tensor,
        lower: Optional[Tensor] = None,
        upper: Optional[Tensor] = None,  # ignored
        detach: bool = True,
    ) -> Tensor:
        b_BS = b.unsqueeze(1) if b.dim() == 2 else b
        A_BS = A.unsqueeze(1) if A.dim() == 3 else A
        x_raw_BS = x.unsqueeze(1) if x.dim() == 2 else x

        lower_BS = self._broadcast_lower(lower, x_raw_BS)

        B, S, m, n = A_BS.shape
        x_f = x_raw_BS.reshape(-1, n)
        A_f = A_BS.reshape(-1, m, n)
        b_f = b_BS.reshape(-1, m)
        lo_f = lower_BS.reshape(-1, n)

        x_feas_f, = self.anchor_layer(x_f, A_f, b_f, lo_f)
        x_feas = x_feas_f.view(B, S, n)

        if detach:
            x_feas = x_feas.detach()
        return x_feas.squeeze(1)

    def lmo(
        self,
        g: Tensor,
        A: Tensor,
        b: Tensor,
        lower: Optional[Tensor] = None,
        upper: Optional[Tensor] = None,  # ignored
        detach: bool = True,
    ) -> Tensor:
        g_BS = g.unsqueeze(1) if g.dim() == 2 else g
        A_BS = A.unsqueeze(1) if A.dim() == 3 else A
        b_BS = b.unsqueeze(1) if b.dim() == 2 else b

        lower_BS = self._broadcast_lower(lower, g_BS)

        B, S, n = g_BS.shape
        m = b_BS.shape[-1]

        g_f = g_BS.reshape(-1, n)
        A_f = A_BS.reshape(-1, m, n)
        b_f = b_BS.reshape(-1, m)
        lo_f = lower_BS.reshape(-1, n)

        c_f, = self.lmo_layer(g_f, A_f, b_f, lo_f)
        c = c_f.view(B, S, n)

        if detach:
            c = c.detach()
        return c.squeeze(1) if g.dim() == 2 else c

    def forward(
        self,
        x_raw: Tensor,
        A: Tensor,
        b: Tensor,
        critic_fn,
        s,  # TensorDict
        lower: Optional[Tensor] = None,
        upper: Optional[Tensor] = None,  # ignored
        detach_solvers: bool = True,
        mode: str = "auto",
    ) -> Tensor:
        x_feas = self.feasible_anchor(x_raw, A, b, lower=lower, upper=None, detach=detach_solvers)

        if mode not in ("auto", "proj", "fw"):
            raise ValueError(f"Unknown mode={mode}")
        if mode == "proj":
            return x_feas

        want_fw = (mode == "fw") or (mode == "auto" and self.training and torch.is_grad_enabled() and critic_fn is not None)
        if not want_fw:
            return x_feas

        if not torch.is_grad_enabled():
            raise RuntimeError("mode='fw' requires grad enabled.")
        if critic_fn is None:
            raise RuntimeError("mode='fw' requires critic_fn.")

        s2 = s.clone()
        x_for_grad = x_feas.detach().requires_grad_(True)
        s2["action"] = x_for_grad

        q_out = critic_fn(s2)
        q = q_out["state_action_value"]
        g = torch.autograd.grad(q.sum(), x_for_grad, create_graph=False, allow_unused=True)[0].detach()

        c = self.lmo(g, A, b, lower=lower, upper=None, detach=detach_solvers)
        x_fw = (1.0 - self.alpha) * x_feas + self.alpha * c
        return x_fw

class ProjectionFactory:
    _class_map = {
        'linear_violation':InnerConvexViolationProjection,
        'linear_violation_policy_clipping':InnerConvexViolationProjection,
        'inner_convex_violation':InnerConvexViolationProjection,
        'inner_convex_violation_alpha':InnerConvexViolationProjection,
        'alpha_chebyshev':AlphaChebyshevProjection,
        'convex_program':CvxpyProjectionLayer,
        'convex_program_policy_clipping':CvxpyProjectionLayer,
        'frank_wolfe':FrankWolfePolicyImprovement,
        'log_barrier':LogBarrierProjection,
    }

    @staticmethod
    def create_class(class_type: str, kwargs:dict) -> nn.Module:
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        else:
            return EmptyLayer()
