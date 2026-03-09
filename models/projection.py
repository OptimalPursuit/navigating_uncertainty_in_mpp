import torch
import torch.nn as nn
from torch import Tensor
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional, Tuple
from diffcp.cone_program import SolverError as DiffcpSolverError
import warnings

class NormalizedProjection(nn.Module):
    def __init__(self, row_normalize: bool = True, eps_row_norm: float = 1e-12, **kwargs):
        super().__init__()
        self.row_normalize = bool(row_normalize)
        self.eps_row_norm = float(eps_row_norm)

    def _normalize_constraints(self, A: Tensor, b: Tensor):
        if not self.row_normalize:
            return A, b
        row_norm = torch.norm(A, dim=-1, keepdim=True).clamp(min=self.eps_row_norm)  # [B,S,m,1]
        return A / row_norm, b / row_norm.squeeze(-1)

class EmptyLayer(NormalizedProjection):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x

class InnerConvexViolationProjection(NormalizedProjection):
    """UVP + alpha-map projection for linear constraints Ax <= b.
    The alpha-map is a post-processing step that can improve performance when the UVP iterate is close to the boundary
    but not quite feasible. It uses the same anchor point as the UVP (no separate solve) and moves along the ray
    from the anchor to the UVP output until just inside the constraints."""

    def __init__(self, **kwargs):
        super().__init__()
        self.K = int(kwargs.get("max_iter", 30))
        self.rho = float(kwargs.get("rho", 1e-12))

        # step size controls
        self.eta = float(kwargs.get("eta", 0.01))
        self.spectral_norm = kwargs.get("spectral_norm", "power_iters")  # "svd", "power_iters", "frobenius"
        self.power_iters = int(kwargs.get("power_iters", 5))
        self.eta_margin = float(kwargs.get("eta_margin", 0.99))

        # UVP improvements
        self.mu_inside = float(kwargs.get("mu_inside", 1e-3))  # meaningful when row_normalize=True

        # alpha-map controls
        self.enable_alpha_map = bool(kwargs.get("enable_alpha_map", True))
        self.alpha_feas_tol = float(kwargs.get("alpha_feas_tol", 1e-7))
        self.eps_inside = float(kwargs.get("eps_inside", 1e-6))
        self.normalize_d = bool(kwargs.get("normalize_d", False))
        self.enforce_nonneg = bool(kwargs.get("enforce_nonneg", False))

    def get_eta(self, A: Tensor, b: Tensor) -> Tensor:
        if A.dim() not in (3, 4) or b.dim() not in (2, 3):
            raise ValueError("get_eta expects A dim 3/4 and b dim 2/3.")

        b_ = b.unsqueeze(1) if b.dim() == 2 else b
        A_ = A.unsqueeze(1) if A.dim() == 3 else A
        A_work, _ = self._normalize_constraints(A_, b_)
        if self.spectral_norm == "svd":
            eta = self._eta_svd(A_work).unsqueeze(-1)
        elif self.spectral_norm == "power_iters":
            eta = self._eta_power_iters(A_work).unsqueeze(-1)
        elif self.spectral_norm == "frobenius":
            eta = self._eta_frobenius(A_work).unsqueeze(-1)
        elif self.spectral_norm == "none":
            eta = self.eta
        else:
            raise ValueError(f"Unknown spectral_norm={self.spectral_norm}")
        return eta

    def _eta_frobenius(self, A: Tensor) -> Tensor:
        denom = torch.sum(A * A, dim=(-2, -1)) + self.rho  # [B,S]
        return 1.0 / denom

    def _eta_svd(self, A: Tensor) -> Tensor:
        # A: [B, S, m, n]
        # torch.linalg.norm with ord=2 over the last two dims gives the matrix spectral norm (largest singular value)
        sigma = torch.linalg.norm(A, ord=2, dim=(-2, -1))  # [B, S] = ||A||_2
        num = sigma * sigma  # [B, S] = ||A||_2^2
        return self.eta_margin / (num + self.rho)

    def _eta_power_iters(self, A: Tensor) -> Tensor:
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

    def _alpha_map_from_anchor(self, x_anchor: Tensor, x_target: Tensor, A: Tensor, b: Tensor) -> Tensor:
        d = x_target - x_anchor
        if self.normalize_d:
            d = d / torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-12)

        B, S, m, n = A.shape
        BS = B * S
        A_f = A.reshape(BS, m, n)
        xa_f = x_anchor.reshape(BS, n, 1)
        d_f = d.reshape(BS, n, 1)

        Ax = torch.bmm(A_f, xa_f).squeeze(-1).reshape(B, S, m)
        Ad = torch.bmm(A_f, d_f).squeeze(-1).reshape(B, S, m)
        slack = b - Ax

        inf = torch.tensor(float("inf"), device=x_anchor.device, dtype=x_anchor.dtype)
        alpha_i = torch.where(Ad > 0, slack / (Ad + 1e-12), inf)
        alpha = torch.amin(alpha_i, dim=-1, keepdim=True)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))
        alpha = torch.clamp(alpha - self.eps_inside, min=0.0, max=1.0)

        x_new = x_anchor + alpha * d
        if self.enforce_nonneg:
            x_new = x_new.clamp_min(0.0)
        return x_new

    def forward(
            self,
            x: Tensor,
            A: Tensor,
            b: Tensor,
            var_mask: Optional[Tensor] = None,
            return_uvp_masks: bool = True,
            **kwargs
    ):
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: b dim 2/3, A dim 3/4 expected.")

        # normalize shapes to [B,S,*]
        b_BS = b.unsqueeze(1) if b.dim() == 2 else b
        A_BS = A.unsqueeze(1) if A.dim() == 3 else A
        x_BS = x.unsqueeze(1) if x.dim() == 2 else x

        if torch.isnan(x_BS).any():
            out = x_BS.squeeze(1) if b.dim() == 2 else x_BS
            if self.enforce_nonneg:
                out = out.clamp_min(0.0)
            out = out * var_mask if var_mask is not None else out
            if return_uvp_masks:
                # Cannot infer meaningful masks on NaN early-exit; return empty.
                masks = torch.empty(0, device=out.device, dtype=torch.bool)
                return out, masks
            return out

        if self.enforce_nonneg:
            x_BS = x_BS.clamp_min(0.0)

        A_work, b_work = self._normalize_constraints(A_BS, b_BS)
        b_tight = b_work - self.mu_inside

        # step-size
        if self.spectral_norm == "svd":
            eta = self._eta_svd(A_work).unsqueeze(-1)
        elif self.spectral_norm == "power_iters":
            eta = self._eta_power_iters(A_work).unsqueeze(-1)
        elif self.spectral_norm == "frobenius":
            eta = self._eta_frobenius(A_work).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown spectral_norm={self.spectral_norm}")

        B, S, m, n = A_work.shape
        BS = B * S

        # flatten for bmm
        A_f = A_work.reshape(BS, m, n)  # [BS,m,n]
        b_f = b_tight.reshape(BS, m, 1)  # [BS,m,1]
        x_f = x_BS.reshape(BS, n, 1)  # [BS,n,1]
        eta_f = eta.reshape(BS, 1, 1)  # [BS,1,1]

        if var_mask is not None:
            vm = var_mask.unsqueeze(1) if var_mask.dim() == 2 else var_mask
            vm_f = vm.reshape(BS, n, 1)
        else:
            vm_f = None

        masks_f = None
        if return_uvp_masks:
            masks_f = torch.empty((BS, self.K, m), device=x.device, dtype=torch.bool)

        # ---- UVP loop ----
        for t in range(self.K):
            Ax = torch.bmm(A_f, x_f)  # [BS,m,1]
            r = Ax - b_f  # [BS,m,1]

            if return_uvp_masks:
                masks_f[:, t, :] = (r.squeeze(-1) > 0)

            v = torch.relu(r)  # [BS,m,1]
            g = torch.bmm(A_f.transpose(1, 2), v)  # [BS,n,1]
            x_f = x_f - eta_f * g

            if self.enforce_nonneg:
                x_f = x_f.clamp_min(0.0)
            if vm_f is not None:
                x_f = x_f * vm_f

        x_ = x_f.squeeze(-1).reshape(B, S, n)  # [B,S,n]

        # ---- Alpha-map (unchanged logic; only compute if any sample is feasible) ----
        if self.enable_alpha_map:
            # A_work @ x_ with bmm to avoid another broadcasted matmul
            Ax_true = torch.bmm(A_f, x_f).squeeze(-1).reshape(B, S, m)  # [B,S,m]
            max_viol = (Ax_true - b_work).clamp(min=0.0).amax(dim=-1, keepdim=True)  # [B,S,1]
            do_alpha = (max_viol <= self.alpha_feas_tol)

            if do_alpha.any():
                x_target = x_BS
                if var_mask is not None:
                    x_target = x_target * (var_mask.unsqueeze(1) if var_mask.dim() == 2 else var_mask)

                x_alpha = self._alpha_map_from_anchor(x_, x_target, A_work, b_work)
                x_ = torch.where(do_alpha, x_alpha, x_)

        if b.dim() == 2:
            x_ = x_.squeeze(1)
        if self.enforce_nonneg:
            x_ = x_.clamp_min(0.0)

        if not return_uvp_masks:
            return x_

        masks_out = masks_f.reshape(B, S, self.K, m)
        if b.dim() == 2:
            masks_out = masks_out.squeeze(1)  # -> [B,K,m]

        return x_, masks_out


class CvxpyProjectionLayer(NormalizedProjection):
    """
    QP projection with nonnegativity and slack on a soft subset of constraints.

    Solves (per item):
      min_x,s  0.5||x-x_raw||^2 + slack_penalty||s||^2
      s.t.     A_hard x <= b_hard
               A_soft x <= b_soft + s
               s >= 0
               x >= lower
    """

    def __init__(self, n_action=80, n_constraints=85, slack_penalty=1000, stab_idx: int = -4, **kwargs):
        super().__init__()
        self.n = int(n_action)
        self.m = int(n_constraints)
        self.slack_penalty = float(slack_penalty)
        self.stab_idx = int(stab_idx)

        x = cp.Variable(self.n)

        if self.stab_idx < 0:
            hard = slice(0, self.stab_idx)
            soft = slice(self.stab_idx, None)
            soft_dim = abs(self.stab_idx)
            n_hard = self.m - soft_dim
        else:
            hard = slice(0, self.stab_idx)
            soft = slice(self.stab_idx, self.m)
            soft_dim = self.m - self.stab_idx
            n_hard = self.stab_idx

        s = cp.Variable(soft_dim)

        x_raw_param = cp.Parameter(self.n)
        A_param = cp.Parameter((self.m, self.n))
        b_param = cp.Parameter(self.m)
        lower_param = cp.Parameter(self.n)

        objective = cp.Minimize(
            0.5 * cp.sum_squares(x - x_raw_param) +
            self.slack_penalty * cp.sum_squares(s)
        )

        constraints = [x >= lower_param]

        if n_hard > 0:
            constraints.append(A_param[hard, :] @ x <= b_param[hard])

        if soft_dim > 0:
            constraints += [
                A_param[soft, :] @ x <= b_param[soft] + s,
                s >= 0,
            ]

        problem = cp.Problem(objective, constraints)
        self.cvxpy_layer = CvxpyLayer(problem, parameters=[x_raw_param, A_param, b_param, lower_param], variables=[x])

    def forward(self, x_raw: Tensor, A: Tensor, b: Tensor, lower: Optional[Tensor] = None, upper: Optional[Tensor] = None) -> Tensor:
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

        A, b = self._normalize_constraints(A, b)
        x_proj, = self.cvxpy_layer(x_raw, A, b, lower)

        if needs_flattening:
            x_proj = x_proj.view(batch_size, -1, x_raw.shape[-1])
        return x_proj


class AlphaChebyshevProjection(NormalizedProjection):
    """
    Chebyshev-center anchor + alpha-map, with the SAME soft slack semantics as the Chebyshev solve.

    Chebyshev (soft):
      max_{x,r,s>=0}  radius_weight*r - slack_penalty*||s||^2
      s.t.            A_hard x + r d_hard <= b_hard
                      A_soft x + r d_soft <= b_soft + s
                      x >= 0 (optional), r>=0
    Alpha-map:
      uses b_eff where b_eff_soft = b_soft + s and b_eff_hard = b_hard.
    """

    def __init__(
        self,
        n_action: int = 80,
        n_constraints: int = 85,
        stab_idx: int = -4,
        slack_penalty: float = 1.0,
        radius_weight: float = 1.0,
        add_nonnegativity: bool = True,
        eps_inside: float = 1e-9,
        cvx_solver: Optional[str] = None,
        cvx_verbose: bool = False,
        enforce_nonneg: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.n = int(n_action)
        self.m = int(n_constraints)
        self.stab_idx = int(stab_idx)

        self.slack_penalty = float(slack_penalty)
        self.radius_weight = float(radius_weight)

        self.add_nonnegativity = bool(add_nonnegativity)
        self.eps_inside = float(eps_inside)
        self.cvx_solver = cvx_solver
        self.cvx_verbose = bool(cvx_verbose)
        self.enforce_nonneg = bool(enforce_nonneg)

        # Compute counts explicitly; less brittle than relying on empty slices.
        if self.stab_idx < 0:
            self.soft_dim = abs(self.stab_idx)
            self.hard_dim = self.m - self.soft_dim
        else:
            self.hard_dim = self.stab_idx
            self.soft_dim = self.m - self.hard_dim

        self.hard = slice(0, self.hard_dim)
        self.soft = slice(self.hard_dim, self.m)

        x = cp.Variable(self.n)
        r = cp.Variable()

        A_param = cp.Parameter((self.m, self.n))
        b_param = cp.Parameter(self.m)
        d_param = cp.Parameter(self.m, nonneg=True)

        cons = [r >= 0]

        if self.hard_dim > 0:
            cons.append(
                A_param[:self.hard_dim, :] @ x
                + cp.multiply(d_param[:self.hard_dim], r)
                <= b_param[:self.hard_dim]
            )

        if self.soft_dim > 0:
            s = cp.Variable(self.soft_dim)
            cons += [
                s >= 0,
                A_param[self.hard_dim:, :] @ x
                + cp.multiply(d_param[self.hard_dim:], r)
                <= b_param[self.hard_dim:] + s,
            ]
            slack_term = cp.sum_squares(s)
            variables = [x, r, s]
        else:
            s = None
            slack_term = 0
            variables = [x, r]

        if self.add_nonnegativity:
            cons.append(x >= 0)

        obj = cp.Maximize(self.radius_weight * r - self.slack_penalty * slack_term)

        prob = cp.Problem(obj, cons)
        self.cheby_layer = CvxpyLayer(
            prob,
            parameters=[A_param, b_param, d_param],
            variables=variables,
        )

    @staticmethod
    def _alpha_map_batched(x_hat: Tensor, A: Tensor, b_eff: Tensor, x0: Tensor, eps_inside: float) -> Tensor:
        d = x_hat - x0
        Ax0 = torch.matmul(A, x0.unsqueeze(-1)).squeeze(-1)
        Ad = torch.matmul(A, d.unsqueeze(-1)).squeeze(-1)
        slack = (b_eff - Ax0).clamp(min=0.0)

        inf = torch.tensor(float("inf"), device=x_hat.device, dtype=x_hat.dtype)
        alpha_i = torch.where(Ad > 0, slack / (Ad + 1e-12), inf)
        alpha = torch.amin(alpha_i, dim=-1, keepdim=True)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))
        alpha = torch.clamp(alpha - eps_inside, 0.0, 1.0)
        return x0 + alpha * d

    def forward(self, x_hat: Tensor, A: Tensor, b: Tensor, var_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        if b.dim() not in (2, 3) or A.dim() not in (3, 4):
            raise ValueError("Invalid dimensions: b dim 2/3, A dim 3/4 expected.")

        b_BS = b.unsqueeze(1) if b.dim() == 2 else b
        A_BS = A.unsqueeze(1) if A.dim() == 3 else A
        x_BS = x_hat.unsqueeze(1) if x_hat.dim() == 2 else x_hat

        B, S, m, n = A_BS.shape
        if m != self.m or n != self.n:
            raise ValueError(f"Configured (m={self.m}, n={self.n}) but got (m={m}, n={n}).")

        d_BS = torch.norm(A_BS, dim=-1).clamp(min=1e-12)  # [B,S,m]

        A_f = A_BS.reshape(-1, m, n)
        b_f = b_BS.reshape(-1, m)
        d_f = d_BS.reshape(-1, m)

        solver_args = {"verbose": self.cvx_verbose}
        if self.cvx_solver is not None:
            solver_args["solver"] = self.cvx_solver

        x0_f, r_f, s_f = self.cheby_layer(A_f, b_f, d_f, solver_args=solver_args)
        x0 = x0_f.view(B, S, n)
        s = s_f.view(B, S, self.soft_dim)

        # Build relaxed RHS for alpha-map: b_eff = b + s on soft rows
        b_eff = b_BS.clone()
        b_eff[..., self.soft] = b_eff[..., self.soft] + s

        A_BS, b_eff = self._normalize_constraints(A_BS, b_eff)
        out = self._alpha_map_batched(x_BS, A_BS, b_eff, x0, self.eps_inside)

        if self.enforce_nonneg:
            out = out.clamp_min(0.0)

        if var_mask is not None:
            vm = var_mask.unsqueeze(1) if var_mask.dim() == 2 else var_mask
            out = out * vm

        return out.squeeze(1) if b.dim() == 2 else out


class FrankWolfePolicyImprovement(NormalizedProjection):
    """
    FAIRNESS UPDATE:
      - No artificial upper bounds.
      - Lower is nonnegativity (default 0).
      - Feasible anchor uses the SAME soft slack semantics as CvxpyProjectionLayer.
      - LMO is solved over the hard feasible set (no slack) because FW requires a convex feasible set.
        (If hard set is empty, FW is undefined; you should fall back to projection mode.)
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

        if self.stab_idx < 0:
            self.soft_dim = abs(self.stab_idx)
            self.hard_dim = self.m - self.soft_dim
        else:
            self.hard_dim = self.stab_idx
            self.soft_dim = self.m - self.hard_dim

        # -----------------------------
        # (1) Anchor problem with slackness
        # -----------------------------
        x = cp.Variable(self.n)

        x_raw_param = cp.Parameter(self.n)
        A_param = cp.Parameter((self.m, self.n))
        b_param = cp.Parameter(self.m)
        lower_param = cp.Parameter(self.n)

        anchor_cons = [x >= lower_param]

        if self.hard_dim > 0:
            anchor_cons.append(A_param[:self.hard_dim, :] @ x <= b_param[:self.hard_dim])

        if self.soft_dim > 0:
            s = cp.Variable(self.soft_dim)
            anchor_cons += [
                A_param[self.hard_dim:, :] @ x <= b_param[self.hard_dim:] + s,
                s >= 0,
            ]
            slack_term = self.slack_penalty * cp.sum_squares(s)
        else:
            slack_term = 0

        # force all params to be present in the problem
        anchor_dummy = 0 * (
            cp.sum(x_raw_param) + cp.sum(A_param) + cp.sum(b_param) + cp.sum(lower_param)
        )

        anchor_obj = cp.Minimize(
            0.5 * cp.sum_squares(x - x_raw_param) + slack_term + anchor_dummy
        )

        anchor_problem = cp.Problem(anchor_obj, anchor_cons)
        self.anchor_layer = CvxpyLayer(
            anchor_problem,
            parameters=[x_raw_param, A_param, b_param, lower_param],
            variables=[x],
        )

        # -----------------------------
        # (2) LMO over hard feasible set only
        # -----------------------------
        c = cp.Variable(self.n)

        g_param = cp.Parameter(self.n)
        A2_param = cp.Parameter((self.m, self.n))
        b2_param = cp.Parameter(self.m)
        lower2_param = cp.Parameter(self.n)

        lmo_cons = [c >= lower2_param]
        lmo_cons.append(A2_param @ c <= b2_param)

        # force all params to be present
        lmo_dummy = 0 * (
            cp.sum(g_param) + cp.sum(A2_param) + cp.sum(b2_param) + cp.sum(lower2_param)
        )

        lmo_obj = cp.Minimize(-g_param @ c + lmo_dummy)
        lmo_problem = cp.Problem(lmo_obj, lmo_cons)

        self.lmo_layer = CvxpyLayer(
            lmo_problem,
            parameters=[g_param, A2_param, b2_param, lower2_param],
            variables=[c],
        )

    def _broadcast_lower(self, lower: Optional[Tensor], ref: Tensor) -> Tensor:
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
            upper: Optional[Tensor] = None,
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

        try:
            x_feas_f, = self.anchor_layer(x_f, A_f, b_f, lo_f)
        except (DiffcpSolverError, RuntimeError, ValueError) as e:
            warnings.warn(f"Anchor solve failed; falling back to lower-clamped raw action. Error: {e}")
            x_feas_f = torch.maximum(x_f, lo_f)

        x_feas = x_feas_f.view(B, S, n)
        if detach:
            x_feas = x_feas.detach()
        return x_feas.squeeze(1) if x.dim() == 2 else x_feas

    def lmo(
            self,
            g: Tensor,
            A: Tensor,
            b: Tensor,
            lower: Optional[Tensor] = None,
            upper: Optional[Tensor] = None,
            detach: bool = True,
            fallback: Optional[Tensor] = None,
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

        try:
            c_f, = self.lmo_layer(g_f, A_f, b_f, lo_f)
        except (DiffcpSolverError, RuntimeError, ValueError) as e:
            warnings.warn(f"LMO solve failed; falling back to anchor/no-op direction. Error: {e}")
            if fallback is None:
                c_f = lo_f
            else:
                fb_BS = fallback.unsqueeze(1) if fallback.dim() == 2 else fallback
                c_f = fb_BS.reshape(-1, n)

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
            s,
            lower: Optional[Tensor] = None,
            upper: Optional[Tensor] = None,
            detach_solvers: bool = True,
            mode: str = "auto",
    ) -> Tensor:
        x_feas = self.feasible_anchor(x_raw, A, b, lower=lower, upper=None, detach=detach_solvers)

        if mode not in ("auto", "proj", "fw"):
            raise ValueError(f"Unknown mode={mode}")
        if mode == "proj":
            return x_feas

        want_fw = (
                mode == "fw"
                or (mode == "auto" and self.training and torch.is_grad_enabled() and critic_fn is not None)
        )
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
        q = q_out["state_action_value"] if "state_action_value" in q_out else q_out["state_value"]
        g = torch.autograd.grad(q.sum(), x_for_grad, create_graph=False, allow_unused=True)[0]
        if g is None:
            return x_feas
        g = g.detach()

        A, b = self._normalize_constraints(A, b)
        c = self.lmo(g, A, b, lower=lower, upper=None, detach=detach_solvers, fallback=x_feas, )
        x_fw = (1.0 - self.alpha) * x_feas + self.alpha * c
        return x_fw


class ProjectionFactory:
    _class_map = {
        "linear_violation": InnerConvexViolationProjection,
        "linear_violation_policy_clipping": InnerConvexViolationProjection,
        "inner_convex_violation": InnerConvexViolationProjection,
        "inner_convex_violation_alpha": InnerConvexViolationProjection,
        "alpha_chebyshev": AlphaChebyshevProjection,
        "convex_program": CvxpyProjectionLayer,
        "convex_program_policy_clipping": CvxpyProjectionLayer,
        "frank_wolfe": FrankWolfePolicyImprovement,
    }

    @staticmethod
    def create_class(class_type: str, kwargs: dict) -> nn.Module:
        if class_type in ProjectionFactory._class_map:
            return ProjectionFactory._class_map[class_type](**kwargs)
        return EmptyLayer()
