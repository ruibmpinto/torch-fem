"""Nonlinear solvers for the global equilibrium iteration.

Provides one public dispatcher, ``solve_nonlinear``, that routes to a
chosen iterative scheme for solving the discretised equilibrium
equation ``r(u) = 0`` of the FEM problem. Six methods are supported:

- ``'newton_raphson'`` — classical Newton with consistent tangent.
- ``'damped_picard'`` — fixed-point iteration with optional Armijo
  backtracking. Forward evaluations only.
- ``'anderson'`` — Anderson / Pulay acceleration. Forward only.
- ``'broyden'`` — good Broyden rank-1 secant updates. Forward only
  after the first iteration.
- ``'jfnk'`` — Jacobian-free Newton-Krylov with finite-difference
  Jacobian-vector products and an in-house restarted GMRES solver.
- ``'rand_subspace_newton'`` — randomised block subspace Newton with
  finite-difference batched JVPs.

All methods share the same exit contract: the iteration converges
when ``||r(u)|| <= max(atol, rtol * ||r(u_0)||)``; on failure to
converge within ``max_iter`` they raise a generic ``Exception`` with
message containing ``'iteration did not converge.'`` so that the
adaptive sub-incrementation in ``Simulation.run`` continues to handle
non-convergence transparently.

The dispatcher accepts a ``residual_jacobian_fn`` callable which is the
sole interface to the FEM problem. The closure encapsulates element
integration, constraint enforcement, and (for the surrogate path)
hidden-state restoration. Forward-only solvers pass
``need_jacobian=False`` to skip the Jacobian computation entirely; on
the surrogate path this avoids the dominant ``jacfwd`` cost.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Tuple

import torch
from torch import Tensor


SolverName = Literal[
    'newton_raphson',
    'damped_picard',
    'anderson',
    'broyden',
    'jfnk',
    'rand_subspace_newton',
]


_VALID_METHODS = {
    'newton_raphson',
    'damped_picard',
    'anderson',
    'broyden',
    'jfnk',
    'rand_subspace_newton',
}


# =============================================================================
# Public dispatcher
# =============================================================================
def solve_nonlinear(
    method: SolverName,
    residual_jacobian_fn: Callable,
    u0: Tensor,
    max_iter: int = 100,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    verbose: bool = False,
    return_resnorm: bool = False,
    linsolve_fn: Optional[Callable] = None,
    **method_opts,
) -> Tuple[Tensor, Optional[list]]:
    """Dispatch to the requested nonlinear solver.

    Args:
        method (str): One of ``'newton_raphson'``, ``'damped_picard'``,
            ``'anderson'``, ``'broyden'``, ``'jfnk'``,
            ``'rand_subspace_newton'``.
        residual_jacobian_fn (callable): Black-box closure with
            signature ``(u, need_jacobian) -> (r, K, F_int)``. ``r``
            is the residual vector ``F_int - F_ext`` with constrained
            DOFs zeroed. ``K`` is the tangent stiffness (sparse or
            dense torch.Tensor) when ``need_jacobian`` is True, else
            ``None``. ``F_int`` is returned for the caller's
            convenience (used when assembling reaction forces after
            convergence).
        u0 (Tensor): Initial guess for the displacement increment
            ``du`` (flat, length ``n_dof``).
        max_iter (int): Maximum number of outer iterations.
        rtol (float): Relative residual tolerance.
        atol (float): Absolute residual tolerance.
        verbose (bool): If True, print residual norm each iteration.
        return_resnorm (bool): If True, return the residual norm
            history list as the second element of the return tuple.
        linsolve_fn (callable, optional): Linear solve closure with
            signature ``(K, r) -> du_step``. Required for
            ``'newton_raphson'``; ignored otherwise.
        **method_opts: Solver-specific keyword arguments. See the
            individual ``_solve_*`` docstrings.

    Returns:
        Tuple[Tensor, list | None]: Converged ``du`` and the residual
        norm history (or ``None`` if ``return_resnorm`` is False).

    Raises:
        ValueError: If ``method`` is not one of the supported names.
        Exception: With message containing
            ``'iteration did not converge.'`` if ``max_iter`` is
            reached without satisfying tolerance.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f'Unknown nonlinear solver method: {method!r}. '
            f'Must be one of {sorted(_VALID_METHODS)}.'
        )
    if method == 'newton_raphson':
        return _solve_newton_raphson(
            residual_jacobian_fn, u0,
            max_iter=max_iter, rtol=rtol, atol=atol,
            verbose=verbose, return_resnorm=return_resnorm,
            linsolve_fn=linsolve_fn,
            **method_opts,
        )
    if method == 'damped_picard':
        return _solve_damped_picard(
            residual_jacobian_fn, u0,
            max_iter=max_iter, rtol=rtol, atol=atol,
            verbose=verbose, return_resnorm=return_resnorm,
            linsolve_fn=linsolve_fn,
            **method_opts,
        )
    if method == 'anderson':
        return _solve_anderson(
            residual_jacobian_fn, u0,
            max_iter=max_iter, rtol=rtol, atol=atol,
            verbose=verbose, return_resnorm=return_resnorm,
            linsolve_fn=linsolve_fn,
            **method_opts,
        )
    if method == 'broyden':
        return _solve_broyden(
            residual_jacobian_fn, u0,
            max_iter=max_iter, rtol=rtol, atol=atol,
            verbose=verbose, return_resnorm=return_resnorm,
            linsolve_fn=linsolve_fn,
            **method_opts,
        )
    if method == 'jfnk':
        return _solve_jfnk(
            residual_jacobian_fn, u0,
            max_iter=max_iter, rtol=rtol, atol=atol,
            verbose=verbose, return_resnorm=return_resnorm,
            linsolve_fn=linsolve_fn,
            **method_opts,
        )
    if method == 'rand_subspace_newton':
        return _solve_rand_subspace_newton(
            residual_jacobian_fn, u0,
            max_iter=max_iter, rtol=rtol, atol=atol,
            verbose=verbose, return_resnorm=return_resnorm,
            linsolve_fn=linsolve_fn,
            **method_opts,
        )
    # Unreachable
    raise ValueError(f'Unhandled method: {method}')


# =============================================================================
# Convergence helpers
# =============================================================================
def _residual_norm(r: Tensor) -> Tensor:
    """Return the 2-norm of the residual vector."""
    return torch.linalg.norm(r)


def _print_iter(method: str, n_iter: int, r_norm: Tensor) -> None:
    """Print the residual norm of the current outer iteration."""
    print(
        f'[{method}] Iteration {n_iter} | '
        f'Residual: {r_norm.item():.5e}'
    )


def _check_converged(
    r_norm: Tensor, r_norm_0: Tensor, atol: float, rtol: float,
) -> bool:
    """Return True if absolute or relative tolerance is satisfied."""
    return bool(
        (r_norm < atol) or (r_norm < rtol * r_norm_0)
    )


def _maybe_record(history: Optional[list], r_norm: Tensor) -> None:
    """Append a residual norm to the history list if requested."""
    if history is not None:
        history.append(r_norm.item())


def _not_converged_error(method: str) -> Exception:
    """Build the standard non-convergence exception."""
    return Exception(
        f'{method} iteration did not converge.'
    )


# =============================================================================
# Newton-Raphson
# =============================================================================
def _solve_newton_raphson(
    residual_jacobian_fn: Callable,
    u0: Tensor,
    max_iter: int = 100,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    verbose: bool = False,
    return_resnorm: bool = False,
    linsolve_fn: Optional[Callable] = None,
) -> Tuple[Tensor, Optional[list]]:
    """Classical Newton-Raphson with consistent tangent.

    Args:
        residual_jacobian_fn (callable): Closure ``(u, need_jacobian)
            -> (r, K, F_int)``. Called with ``need_jacobian=True``
            every iteration.
        u0 (Tensor): Initial displacement increment.
        max_iter (int): Maximum number of NR iterations.
        rtol (float): Relative tolerance on the residual norm.
        atol (float): Absolute tolerance on the residual norm.
        verbose (bool): Print iteration info if True.
        return_resnorm (bool): Return residual history if True.
        linsolve_fn (callable): Linear solve closure
            ``(K, r) -> du_step`` returning the solution of
            ``K du_step = r``. Required.

    Returns:
        Tuple[Tensor, list | None]: Converged ``du`` and residual
        history (or None).

    Raises:
        ValueError: If ``linsolve_fn`` is None.
        Exception: On non-convergence.
    """
    if linsolve_fn is None:
        raise ValueError(
            "Newton-Raphson requires a 'linsolve_fn' closure."
        )
    history = [] if return_resnorm else None
    u = u0
    r_norm_0 = None
    for i in range(max_iter):
        r, K, _ = residual_jacobian_fn(u, need_jacobian=True)
        r_norm = _residual_norm(r)
        _maybe_record(history, r_norm)
        if i == 0:
            r_norm_0 = r_norm
        if verbose:
            _print_iter('newton_raphson', i + 1, r_norm)
        if _check_converged(r_norm, r_norm_0, atol, rtol):
            return u, history
        # Solve K * du_step = r and apply the Newton update.
        du_step = linsolve_fn(K, r)
        u = u - du_step
    raise _not_converged_error('newton_raphson')


# =============================================================================
# Damped Picard
# =============================================================================
def _build_preconditioner(
    residual_jacobian_fn: Callable,
    u0: Tensor,
    linsolve_fn: Optional[Callable],
) -> Tuple[Callable, Tensor]:
    """Build a left-preconditioner ``M^{-1}`` from a cached ``K_0``.

    When ``linsolve_fn`` is supplied and the residual closure exposes
    a Jacobian, fetch ``K_0`` once via a single
    ``need_jacobian=True`` call and define a closure that returns
    ``K_0^{-1} r`` for any subsequent ``r``. When ``linsolve_fn`` is
    ``None``, return the identity. The pre-computed residual
    ``r(u_0)`` is also returned to avoid an extra forward call.

    Args:
        residual_jacobian_fn (callable): Closure
            ``(u, need_jacobian) -> (r, K, F_int)``.
        u0 (Tensor): Initial displacement increment.
        linsolve_fn (callable | None): Linear solve closure
            ``(K, r) -> du`` used to invert the cached ``K_0``.

    Returns:
        Tuple[callable, Tensor]: ``(precondition, r0)`` where
        ``precondition(r) -> r_pc`` and ``r0`` is the residual at
        ``u0``.
    """
    if linsolve_fn is None:
        r0, _, _ = residual_jacobian_fn(u0, need_jacobian=False)

        def precondition(r):
            return r
        return precondition, r0
    r0, K0, _ = residual_jacobian_fn(u0, need_jacobian=True)
    if K0 is None:
        # Closure refused to produce K despite request.
        def precondition(r):
            return r
        return precondition, r0

    def precondition(r):
        return linsolve_fn(K0, r)

    return precondition, r0


def _solve_damped_picard(
    residual_jacobian_fn: Callable,
    u0: Tensor,
    max_iter: int = 100,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    verbose: bool = False,
    return_resnorm: bool = False,
    linsolve_fn: Optional[Callable] = None,
    damping: float = 1.0,
    line_search: bool = True,
    line_search_max_halvings: int = 5,
) -> Tuple[Tensor, Optional[list]]:
    """Damped Picard fixed-point iteration.

    The iteration takes ``u_{k+1} = u_k - alpha M^{-1} r_k`` where
    ``M^{-1}`` is the optional preconditioner built from the cached
    initial Jacobian ``K_0`` (``M = K_0`` when ``linsolve_fn`` is
    supplied, identity otherwise). When ``line_search`` is True an
    Armijo backtracking step halves ``alpha`` up to
    ``line_search_max_halvings`` times until the residual norm is
    reduced.

    Args:
        residual_jacobian_fn (callable): Closure
            ``(u, need_jacobian) -> (r, K, F_int)``.
        u0 (Tensor): Initial displacement increment.
        max_iter (int): Maximum outer iterations.
        rtol (float): Relative residual tolerance.
        atol (float): Absolute residual tolerance.
        verbose (bool): Print iteration info if True.
        return_resnorm (bool): Return residual history if True.
        linsolve_fn (callable, optional): Linear solve closure
            ``(K, r) -> du`` used to apply ``K_0^{-1}`` as a left
            preconditioner. If ``None`` the iteration runs without
            preconditioning (identity ``M``).
        damping (float): Initial damping coefficient ``alpha``.
        line_search (bool): Enable Armijo backtracking on stagnation.
        line_search_max_halvings (int): Maximum number of halvings
            of ``alpha`` per iteration.

    Returns:
        Tuple[Tensor, list | None]: Converged ``du`` and history.

    Raises:
        Exception: On non-convergence.
    """
    history = [] if return_resnorm else None
    u = u0
    precondition, r = _build_preconditioner(
        residual_jacobian_fn, u, linsolve_fn)
    r_norm = _residual_norm(r)
    r_norm_0 = r_norm
    _maybe_record(history, r_norm)
    if verbose:
        _print_iter('damped_picard', 0, r_norm)
    if _check_converged(r_norm, r_norm_0, atol, rtol):
        return u, history
    for i in range(1, max_iter + 1):
        alpha = damping
        d = precondition(r)
        # Trial step at the full damping factor.
        u_trial = u - alpha * d
        r_trial, _, _ = residual_jacobian_fn(
            u_trial, need_jacobian=False)
        r_trial_norm = _residual_norm(r_trial)
        # Armijo backtracking.
        if line_search:
            n_halvings = 0
            while (
                r_trial_norm >= r_norm
                and n_halvings < line_search_max_halvings
            ):
                alpha = 0.5 * alpha
                u_trial = u - alpha * d
                r_trial, _, _ = residual_jacobian_fn(
                    u_trial, need_jacobian=False)
                r_trial_norm = _residual_norm(r_trial)
                n_halvings += 1
        # Accept the trial step regardless of line-search outcome:
        # if no descent was found, the iteration may still drift to
        # a region where progress resumes.
        u = u_trial
        r = r_trial
        r_norm = r_trial_norm
        _maybe_record(history, r_norm)
        if verbose:
            _print_iter('damped_picard', i, r_norm)
        if _check_converged(r_norm, r_norm_0, atol, rtol):
            return u, history
    raise _not_converged_error('damped_picard')


# =============================================================================
# Anderson acceleration
# =============================================================================
def _solve_anderson(
    residual_jacobian_fn: Callable,
    u0: Tensor,
    max_iter: int = 100,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    verbose: bool = False,
    return_resnorm: bool = False,
    linsolve_fn: Optional[Callable] = None,
    m: int = 5,
    beta: float = 1.0,
    reg: float = 1e-10,
    stagnation_ratio: float = 0.99,
    stagnation_window: int = 3,
    hard_restart: int = 20,
) -> Tuple[Tensor, Optional[list]]:
    """Anderson / Pulay acceleration on a fixed-point map.

    Implementation follows Walker & Ni (SIAM J. Numer. Anal. 49,
    2011). When ``linsolve_fn`` is provided, the underlying
    fixed-point map becomes ``T(u) = u - beta K_0^{-1} r(u)`` where
    ``K_0`` is the Jacobian at ``u_0`` cached once at solver entry;
    otherwise the map is the dimensional ``T(u) = u - beta r(u)``.
    The k-th iterate solves a regularised least-squares problem
    ``min_g ||r_k - dR g||^2 + reg ||g||^2`` over the most recent
    ``m`` consecutive differences and steps with the Anderson
    correction.

    Args:
        residual_jacobian_fn (callable): Closure
            ``(u, need_jacobian) -> (r, K, F_int)``.
        u0 (Tensor): Initial displacement increment.
        max_iter (int): Maximum outer iterations.
        rtol (float): Relative residual tolerance.
        atol (float): Absolute residual tolerance.
        verbose (bool): Print iteration info if True.
        return_resnorm (bool): Return residual history if True.
        linsolve_fn (callable, optional): Linear solve closure
            ``(K, r) -> du`` used to apply ``K_0^{-1}`` as a left
            preconditioner. If ``None`` no preconditioner is used.
        m (int): Mixing depth (history length).
        beta (float): Mixing parameter (``beta=1`` recovers full
            Anderson; smaller values blend toward Picard).
        reg (float): Tikhonov regularisation on the least-squares
            normal equations.
        stagnation_ratio (float): Trigger soft restart when
            ``||r_{k+1}||/||r_k|| > stagnation_ratio`` for
            ``stagnation_window`` consecutive iterations.
        stagnation_window (int): See ``stagnation_ratio``.
        hard_restart (int): Drop history every ``hard_restart``
            iterations regardless of progress (``0`` disables).

    Returns:
        Tuple[Tensor, list | None]: Converged ``du`` and history.

    Raises:
        Exception: On non-convergence.
    """
    history = [] if return_resnorm else None
    u = u0
    precondition, r = _build_preconditioner(
        residual_jacobian_fn, u, linsolve_fn)
    r_norm = _residual_norm(r)
    r_norm_0 = r_norm
    _maybe_record(history, r_norm)
    if verbose:
        _print_iter('anderson', 0, r_norm)
    if _check_converged(r_norm, r_norm_0, atol, rtol):
        return u, history
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Column-stored history of past consecutive differences:
    # column j of U_hist is u_j - u_{j-1}, column j of D_hist is
    # d_j - d_{j-1} where d_k = M^{-1} r_k.
    n = u.numel()
    U_hist = torch.zeros(
        (n, 0), dtype=u.dtype, device=u.device)
    D_hist = torch.zeros(
        (n, 0), dtype=u.dtype, device=u.device)
    d = precondition(r)
    stag_count = 0
    last_iters_since_restart = 0
    for k in range(1, max_iter + 1):
        # Picard-direction trial used to build the next iterate.
        u_picard = u - beta * d
        if U_hist.shape[1] == 0:
            u_next = u_picard
        else:
            # Solve regularised normal equations:
            # (D^T D + reg I) g = D^T d.
            DtD = D_hist.T @ D_hist
            rhs = D_hist.T @ d
            reg_mat = reg * torch.eye(
                DtD.shape[0],
                dtype=u.dtype, device=u.device,
            )
            try:
                gamma = torch.linalg.solve(DtD + reg_mat, rhs)
            except RuntimeError:
                # Fall back to least-squares if normal equations fail.
                gamma = torch.linalg.lstsq(D_hist, d).solution
            # u_{k+1} = u - beta d - (U - beta D) g
            u_next = (
                u_picard - (U_hist - beta * D_hist) @ gamma
            )
        r_next, _, _ = residual_jacobian_fn(
            u_next, need_jacobian=False)
        r_next_norm = _residual_norm(r_next)
        _maybe_record(history, r_next_norm)
        if verbose:
            _print_iter('anderson', k, r_next_norm)
        if _check_converged(r_next_norm, r_norm_0, atol, rtol):
            return u_next, history
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apply the preconditioner to the new residual and append the
        # consecutive-difference column.
        d_next = precondition(r_next)
        du_col = (u_next - u).reshape(n, 1)
        dd_col = (d_next - d).reshape(n, 1)
        U_hist = torch.cat([U_hist, du_col], dim=1)
        D_hist = torch.cat([D_hist, dd_col], dim=1)
        if U_hist.shape[1] > m:
            U_hist = U_hist[:, 1:]
            D_hist = D_hist[:, 1:]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stagnation detection / soft restart.
        ratio = r_next_norm / (r_norm + 1e-30)
        if ratio > stagnation_ratio:
            stag_count += 1
        else:
            stag_count = 0
        do_soft_restart = stag_count >= stagnation_window
        last_iters_since_restart += 1
        do_hard_restart = (
            hard_restart > 0
            and last_iters_since_restart >= hard_restart
        )
        if do_soft_restart or do_hard_restart:
            U_hist = U_hist[:, :0]
            D_hist = D_hist[:, :0]
            stag_count = 0
            last_iters_since_restart = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Roll state forward.
        u = u_next
        r = r_next
        d = d_next
        r_norm = r_next_norm
    raise _not_converged_error('anderson')


# =============================================================================
# Broyden quasi-Newton ('good' Broyden, rank-1 update on J)
# =============================================================================
def _solve_broyden(
    residual_jacobian_fn: Callable,
    u0: Tensor,
    max_iter: int = 100,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    verbose: bool = False,
    return_resnorm: bool = False,
    linsolve_fn: Optional[Callable] = None,
    use_initial_jacobian: bool = True,
    initial_alpha: float = 1.0,
    skip_update_eps: float = 1e-14,
    stagnation_factor: float = 1.5,
    stagnation_window: int = 3,
    jacobian_refresh_period: int = 0,
) -> Tuple[Tensor, Optional[list]]:
    """Good-Broyden quasi-Newton with rank-1 secant updates.

    The initial Jacobian is either taken from ``residual_jacobian_fn``
    on the first iteration (``use_initial_jacobian=True``) or set to
    ``-initial_alpha * I``. Subsequent iterations update
    ``J_{k+1} = J_k + ((dr - J_k du) du^T) / (du^T du)``. On
    stagnation (residual grows by a factor of
    ``stagnation_factor`` over ``stagnation_window`` iterations) the
    Jacobian is reset to ``-initial_alpha I``.

    Args:
        residual_jacobian_fn (callable): Closure
            ``(u, need_jacobian) -> (r, K, F_int)``.
        u0 (Tensor): Initial displacement increment.
        max_iter (int): Maximum outer iterations.
        rtol (float): Relative residual tolerance.
        atol (float): Absolute residual tolerance.
        verbose (bool): Print iteration info if True.
        return_resnorm (bool): Return residual history if True.
        linsolve_fn (callable, optional): Unused; accepted for API
            uniformity with the other solvers.
        use_initial_jacobian (bool): Build ``J_0`` from one
            ``need_jacobian=True`` call. If False, use ``-alpha I``.
        initial_alpha (float): Scalar for the identity-based ``J_0``.
        skip_update_eps (float): Skip the rank-1 update when
            ``||du||^2 < skip_update_eps``.
        stagnation_factor (float): Trigger Jacobian reset when
            ``||r_{k+w}||/||r_k|| > stagnation_factor``.
        stagnation_window (int): Window length for stagnation check.
        jacobian_refresh_period (int): Refresh ``J`` from a fresh
            ``need_jacobian=True`` call every ``jacobian_refresh_period``
            iterations. ``0`` disables periodic refresh (rely on
            stagnation detection only).

    Returns:
        Tuple[Tensor, list | None]: Converged ``du`` and history.

    Raises:
        Exception: On non-convergence.
    """
    del linsolve_fn  # accepted for uniform API; not used by Broyden
    history = [] if return_resnorm else None
    u = u0
    n = u.numel()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial residual and Jacobian.
    if use_initial_jacobian:
        r, K_init, _ = residual_jacobian_fn(u, need_jacobian=True)
        if K_init is None:
            J = -initial_alpha * torch.eye(
                n, dtype=u.dtype, device=u.device)
        else:
            # Densify if sparse; small problems only.
            if K_init.is_sparse:
                J = K_init.to_dense()
            else:
                J = K_init.clone()
    else:
        r, _, _ = residual_jacobian_fn(u, need_jacobian=False)
        J = -initial_alpha * torch.eye(
            n, dtype=u.dtype, device=u.device)
    r_norm = _residual_norm(r)
    r_norm_0 = r_norm
    _maybe_record(history, r_norm)
    if verbose:
        _print_iter('broyden', 0, r_norm)
    if _check_converged(r_norm, r_norm_0, atol, rtol):
        return u, history
    # Track residual norms for stagnation detection.
    recent_norms: list = [r_norm.item()]
    for k in range(1, max_iter + 1):
        # Solve J * s = r and step u <- u - s.
        try:
            s = torch.linalg.solve(J, r)
        except RuntimeError:
            # Singular J: pseudo-inverse fallback.
            s = torch.linalg.lstsq(J, r).solution
        u_new = u - s
        # Periodic Jacobian refresh from a fresh need_jacobian=True
        # evaluation; helps track state-dependent stiffness changes
        # (e.g. plastic loading) that the rank-1 secant updates
        # cannot capture.
        if (
            jacobian_refresh_period > 0
            and (k % jacobian_refresh_period) == 0
        ):
            r_new, K_refresh, _ = residual_jacobian_fn(
                u_new, need_jacobian=True)
            if K_refresh is not None:
                if K_refresh.is_sparse:
                    J = K_refresh.to_dense()
                else:
                    J = K_refresh.clone()
        else:
            r_new, _, _ = residual_jacobian_fn(
                u_new, need_jacobian=False)
        r_new_norm = _residual_norm(r_new)
        _maybe_record(history, r_new_norm)
        if verbose:
            _print_iter('broyden', k, r_new_norm)
        if _check_converged(r_new_norm, r_norm_0, atol, rtol):
            return u_new, history
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rank-1 'good Broyden' update of J (skipped immediately
        # after a refresh because J is already up to date).
        du = u_new - u
        dr = r_new - r
        du_sq = float(torch.dot(du, du))
        skip_update_this_iter = (
            jacobian_refresh_period > 0
            and (k % jacobian_refresh_period) == 0
        )
        if (not skip_update_this_iter) and du_sq > skip_update_eps:
            correction = (
                (dr - J @ du).unsqueeze(1)
                @ du.unsqueeze(0)
                / du_sq
            )
            J = J + correction
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stagnation check; reset J on relapse.
        recent_norms.append(r_new_norm.item())
        if len(recent_norms) > stagnation_window:
            recent_norms.pop(0)
        if (
            len(recent_norms) == stagnation_window
            and (recent_norms[-1]
                 / max(recent_norms[0], 1e-30)
                 > stagnation_factor)
        ):
            # Try a fresh Jacobian first if the closure can produce
            # one, otherwise fall back to scaled identity.
            r_reset, K_reset, _ = residual_jacobian_fn(
                u_new, need_jacobian=True)
            if K_reset is not None:
                if K_reset.is_sparse:
                    J = K_reset.to_dense()
                else:
                    J = K_reset.clone()
                # Use the freshly evaluated residual so the next
                # solve uses the consistent (u, r) pair.
                r_new = r_reset
                r_new_norm = _residual_norm(r_new)
            else:
                J = -initial_alpha * torch.eye(
                    n, dtype=u.dtype, device=u.device)
            recent_norms = [r_new_norm.item()]
        u = u_new
        r = r_new
        r_norm = r_new_norm
    raise _not_converged_error('broyden')


# =============================================================================
# JFNK (Jacobian-Free Newton-Krylov, in-house GMRES)
# =============================================================================
def _jvp_finite_diff(
    residual_jacobian_fn: Callable,
    u: Tensor,
    r_at_u: Tensor,
    v: Tensor,
    eps_floor: float = 1e-7,
) -> Tensor:
    """Forward finite-difference Jacobian-vector product.

    Computes ``J @ v ~= (r(u + eps v) - r(u)) / eps`` with a step
    size scaled to the magnitude of ``u`` and ``v``. The denominator
    is clamped to ``eps_floor`` to avoid catastrophic cancellation
    when ``u`` is small.

    Args:
        residual_jacobian_fn (callable): Black-box residual closure.
        u (Tensor): Current iterate.
        r_at_u (Tensor): Pre-computed residual ``r(u)``.
        v (Tensor): Direction along which to take the JVP.
        eps_floor (float): Minimum allowed perturbation step.

    Returns:
        Tensor: Approximate ``J(u) @ v`` of the same shape as ``v``.
    """
    v_norm = float(torch.linalg.norm(v))
    if v_norm < eps_floor:
        return torch.zeros_like(v)
    machine_eps = float(torch.finfo(u.dtype).eps)
    eps = math.sqrt(machine_eps) * (1.0 + float(
        torch.linalg.norm(u))) / v_norm
    eps = max(eps, eps_floor)
    r_pert, _, _ = residual_jacobian_fn(
        u + eps * v, need_jacobian=False)
    return (r_pert - r_at_u) / eps


def _gmres(
    matvec: Callable,
    b: Tensor,
    tol: float,
    max_inner: int,
    restart: int,
) -> Tensor:
    """Restarted GMRES on a black-box matrix-vector product.

    Solves ``A x = b`` for ``x`` using right-preconditioned-free
    GMRES with classical Gram-Schmidt and explicit restarts.

    Args:
        matvec (callable): ``v -> A v``.
        b (Tensor): Right-hand side.
        tol (float): Absolute tolerance on ``||b - A x||``.
        max_inner (int): Maximum total inner iterations across all
            restart cycles.
        restart (int): Restart depth ``m``.

    Returns:
        Tensor: Approximate solution ``x``.
    """
    n = b.numel()
    x = torch.zeros_like(b)
    r0 = b - matvec(x)
    beta = float(torch.linalg.norm(r0))
    if beta < tol:
        return x
    iters = 0
    while iters < max_inner:
        m = min(restart, max_inner - iters)
        # Krylov basis V (n x (m+1)) and Hessenberg H ((m+1) x m).
        V = torch.zeros(
            (n, m + 1), dtype=b.dtype, device=b.device)
        H = torch.zeros(
            (m + 1, m), dtype=b.dtype, device=b.device)
        V[:, 0] = r0 / beta
        g = torch.zeros(m + 1, dtype=b.dtype, device=b.device)
        g[0] = beta
        c = torch.zeros(m, dtype=b.dtype, device=b.device)
        s = torch.zeros(m, dtype=b.dtype, device=b.device)
        j_completed = 0
        for j in range(m):
            w = matvec(V[:, j])
            for i in range(j + 1):
                H[i, j] = torch.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = torch.linalg.norm(w)
            if float(H[j + 1, j]) > 1e-30:
                V[:, j + 1] = w / H[j + 1, j]
            # Apply previous Givens rotations.
            for i in range(j):
                temp = c[i] * H[i, j] + s[i] * H[i + 1, j]
                H[i + 1, j] = (
                    -s[i] * H[i, j] + c[i] * H[i + 1, j]
                )
                H[i, j] = temp
            # New Givens rotation to zero H[j+1, j].
            denom = torch.sqrt(H[j, j]**2 + H[j + 1, j]**2)
            c[j] = H[j, j] / denom
            s[j] = H[j + 1, j] / denom
            H[j, j] = c[j] * H[j, j] + s[j] * H[j + 1, j]
            H[j + 1, j] = torch.zeros_like(H[j + 1, j])
            # Apply rotation to g.
            temp = c[j] * g[j] + s[j] * g[j + 1]
            g[j + 1] = -s[j] * g[j] + c[j] * g[j + 1]
            g[j] = temp
            j_completed = j + 1
            iters += 1
            if abs(float(g[j + 1])) < tol:
                break
        # Solve upper-triangular system H y = g[:j_completed].
        y = torch.linalg.solve_triangular(
            H[:j_completed, :j_completed],
            g[:j_completed].unsqueeze(1),
            upper=True,
        ).squeeze(1)
        x = x + V[:, :j_completed] @ y
        r0 = b - matvec(x)
        beta = float(torch.linalg.norm(r0))
        if beta < tol:
            break
    return x


def _solve_jfnk(
    residual_jacobian_fn: Callable,
    u0: Tensor,
    max_iter: int = 100,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    verbose: bool = False,
    return_resnorm: bool = False,
    linsolve_fn: Optional[Callable] = None,
    krylov_restart: int = 30,
    krylov_max_inner_factor: int = 2,
    forcing_eta_max: float = 0.9,
    forcing_gamma: float = 0.9,
) -> Tuple[Tensor, Optional[list]]:
    """Jacobian-free Newton-Krylov with finite-difference JVPs.

    Each outer Newton step solves ``J(u_k) s = -r(u_k)`` via in-house
    restarted GMRES; matrix-vector products are computed by a single
    extra forward residual evaluation each. The inner tolerance is
    selected adaptively via Eisenstat-Walker variant 2:
    ``eta_k = min(eta_max, gamma (||r_k||/||r_{k-1}||)^2)``.

    Args:
        residual_jacobian_fn (callable): Closure
            ``(u, need_jacobian) -> (r, K, F_int)``. Called with
            ``need_jacobian=False`` for all forward evaluations.
        u0 (Tensor): Initial displacement increment.
        max_iter (int): Maximum outer Newton iterations.
        rtol (float): Relative residual tolerance (outer).
        atol (float): Absolute residual tolerance (outer).
        verbose (bool): Print iteration info if True.
        return_resnorm (bool): Return residual history if True.
        linsolve_fn (callable, optional): Unused; accepted for API
            uniformity with the other solvers.
        krylov_restart (int): GMRES restart depth.
        krylov_max_inner_factor (int): GMRES total inner iter cap is
            ``krylov_max_inner_factor * n_dof``.
        forcing_eta_max (float): Cap on the Eisenstat-Walker forcing
            term.
        forcing_gamma (float): Constant in the forcing-term update.

    Returns:
        Tuple[Tensor, list | None]: Converged ``du`` and history.

    Raises:
        Exception: On non-convergence.
    """
    del linsolve_fn  # accepted for uniform API; not used by JFNK
    history = [] if return_resnorm else None
    u = u0
    n_dof = u.numel()
    r, _, _ = residual_jacobian_fn(u, need_jacobian=False)
    r_norm = _residual_norm(r)
    r_norm_0 = r_norm
    r_norm_prev = r_norm
    _maybe_record(history, r_norm)
    if verbose:
        _print_iter('jfnk', 0, r_norm)
    if _check_converged(r_norm, r_norm_0, atol, rtol):
        return u, history
    eta = forcing_eta_max
    # Track best-seen residual to guard against FD-noise-driven
    # drift past the FD JVP precision floor: if the iteration walks
    # away from a previously achieved residual far below the current
    # value, we accept the best iterate and exit. This protects the
    # caller from non-monotone behaviour at sub-FD-precision scales.
    u_best = u.clone()
    r_best_norm = float(r_norm)
    for k in range(1, max_iter + 1):
        # Inner GMRES tolerance: relative to current residual norm.
        inner_tol = float(eta * r_norm)
        max_inner = krylov_max_inner_factor * n_dof

        def matvec(v):
            return _jvp_finite_diff(
                residual_jacobian_fn, u, r, v)

        s = _gmres(
            matvec, -r, tol=max(inner_tol, atol * 1e-2),
            max_inner=max_inner, restart=krylov_restart)
        u = u + s
        r, _, _ = residual_jacobian_fn(u, need_jacobian=False)
        r_norm_new = _residual_norm(r)
        _maybe_record(history, r_norm_new)
        if verbose:
            _print_iter('jfnk', k, r_norm_new)
        if _check_converged(r_norm_new, r_norm_0, atol, rtol):
            return u, history
        # Track best iterate.
        if float(r_norm_new) < r_best_norm:
            u_best = u.clone()
            r_best_norm = float(r_norm_new)
        # Stop early if we have drifted ten-fold above the best
        # residual ever seen and that best residual is below the
        # absolute tolerance: this is the textbook FD-noise stall
        # signature, where further iterations only make things worse.
        if (
            r_best_norm < atol
            and float(r_norm_new) > 10.0 * r_best_norm
        ):
            return u_best, history
        # Eisenstat-Walker variant 2 update.
        ratio = float(r_norm_new / max(r_norm_prev, 1e-30))
        eta = min(forcing_eta_max, forcing_gamma * ratio**2)
        r_norm_prev = r_norm
        r_norm = r_norm_new
    # Final fallback: if iteration ended without satisfying the
    # tolerance but a strictly better iterate was previously seen,
    # raise non-convergence using the best residual for the message.
    raise _not_converged_error('jfnk')


# =============================================================================
# Randomised block subspace Newton
# =============================================================================
def _solve_rand_subspace_newton(
    residual_jacobian_fn: Callable,
    u0: Tensor,
    max_iter: int = 100,
    rtol: float = 1e-8,
    atol: float = 1e-6,
    verbose: bool = False,
    return_resnorm: bool = False,
    linsolve_fn: Optional[Callable] = None,
    block_size: int = 8,
    seed: int = 0,
    sv_truncate_ratio: float = 1e-10,
) -> Tuple[Tensor, Optional[list]]:
    """Randomised block subspace Newton step.

    At each iteration ``k``: sample a Gaussian matrix
    ``V in R^{n x block_size}``, orthogonalise via QR, compute
    ``JV`` column by column with finite-difference JVPs, solve the
    block subspace problem ``min_a ||JV a + r||^2`` with truncated
    SVD, then step ``u_{k+1} = u_k + V a``. The iteration is fully
    Jacobian-free.

    Args:
        residual_jacobian_fn (callable): Closure
            ``(u, need_jacobian) -> (r, K, F_int)``. Called with
            ``need_jacobian=False``.
        u0 (Tensor): Initial displacement increment.
        max_iter (int): Maximum outer iterations.
        rtol (float): Relative residual tolerance.
        atol (float): Absolute residual tolerance.
        verbose (bool): Print iteration info if True.
        return_resnorm (bool): Return residual history if True.
        linsolve_fn (callable, optional): Unused; accepted for API
            uniformity with the other solvers.
        block_size (int): Number of random directions per iteration.
        seed (int): Seed for the per-iteration random sampling so
            that iteration counts are deterministic.
        sv_truncate_ratio (float): Truncate singular values smaller
            than ``sv_truncate_ratio * sigma_max`` in the subspace
            least-squares solve.

    Returns:
        Tuple[Tensor, list | None]: Converged ``du`` and history.

    Raises:
        Exception: On non-convergence.
    """
    history = [] if return_resnorm else None
    u = u0
    n_dof = u.numel()
    # Build preconditioner ``M^{-1}`` (identity if linsolve_fn is
    # None). When supplied, the random subspace is mapped through
    # ``M^{-1}`` so the search basis lives in displacement space and
    # the step ``u + W alpha`` is dimensionally consistent.
    precondition, r = _build_preconditioner(
        residual_jacobian_fn, u, linsolve_fn)
    r_norm = _residual_norm(r)
    r_norm_0 = r_norm
    _maybe_record(history, r_norm)
    if verbose:
        _print_iter('rand_subspace_newton', 0, r_norm)
    if _check_converged(r_norm, r_norm_0, atol, rtol):
        return u, history
    # Deterministic per-call generator so iteration counts are
    # reproducible across runs.
    gen = torch.Generator(device=u.device)
    gen.manual_seed(seed)
    k_eff = min(block_size, n_dof)
    u_best = u.clone()
    r_best_norm = float(r_norm)
    for k in range(1, max_iter + 1):
        # Sample Gaussian matrix and orthogonalise.
        V = torch.randn(
            (n_dof, k_eff), generator=gen,
            dtype=u.dtype, device=u.device,
        )
        V, _ = torch.linalg.qr(V)
        # Apply preconditioner column by column to map into the
        # displacement-space search basis ``W``.
        W = torch.zeros_like(V)
        for j in range(k_eff):
            W[:, j] = precondition(V[:, j])
        # Compute J W column by column via finite-diff JVPs.
        JW = torch.zeros_like(W)
        for j in range(k_eff):
            JW[:, j] = _jvp_finite_diff(
                residual_jacobian_fn, u, r, W[:, j])
        # Truncated-SVD subspace least-squares solve.
        try:
            Uj, S, Vh = torch.linalg.svd(JW, full_matrices=False)
            sv_thresh = sv_truncate_ratio * float(S.max())
            S_inv = torch.where(
                S > sv_thresh, 1.0 / S, torch.zeros_like(S),
            )
            alpha = -(Vh.T @ (S_inv * (Uj.T @ r)))
        except RuntimeError:
            # Fallback: damped least-squares.
            JtJ = JW.T @ JW
            damp = 1e-10 * torch.eye(
                k_eff, dtype=u.dtype, device=u.device,
            )
            alpha = -torch.linalg.solve(JtJ + damp, JW.T @ r)
        u = u + W @ alpha
        r, _, _ = residual_jacobian_fn(u, need_jacobian=False)
        r_norm = _residual_norm(r)
        _maybe_record(history, r_norm)
        if verbose:
            _print_iter('rand_subspace_newton', k, r_norm)
        if _check_converged(r_norm, r_norm_0, atol, rtol):
            return u, history
        if float(r_norm) < r_best_norm:
            u_best = u.clone()
            r_best_norm = float(r_norm)
        if (
            r_best_norm < atol
            and float(r_norm) > 10.0 * r_best_norm
        ):
            return u_best, history
    raise _not_converged_error('rand_subspace_newton')
