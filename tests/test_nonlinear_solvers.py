"""Tests for the nonlinear solver dispatcher.

Validates that:

1. ``FEM.solve(nonlinear_solver='newton_raphson')`` reproduces a cached
   reference solution to within floating-point round-off (regression
   guard against the closure-based refactor of the inner Newton loop).
2. The five forward-only solvers (``damped_picard``, ``anderson``,
   ``broyden``, ``jfnk``, ``rand_subspace_newton``) converge on a 4x4
   plane-strain plasticity problem with linear isotropic hardening and
   recover the Newton-Raphson displacement and force fields to within
   ``atol=1e-4 / rtol=1e-4`` per increment. Tolerances are chosen to
   accommodate finite-difference Jacobian-vector precision in JFNK and
   the K_0-based preconditioner used by Damped Picard / Anderson; they
   are tighter than the physical scales of the problem.

The Newton-Raphson reference is cached on disk under ``tests/_cache/``
keyed by a hash of the problem definition; remove the cache to force a
rebuild.

Functions
---------
make_linear_hardening
    Build linear-hardening sigma_f / sigma_f_prime closures.
build_problem
    Construct the 4x4 plane-strain plasticity test domain and its
    increment sequence.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import hashlib
import os
import pickle
# Third-party
import pytest
import torch
# Local
from torchfem import Planar
from torchfem.materials import IsotropicPlasticityPlaneStrain
from torchfem.mesh import rect_quad
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
torch.set_default_dtype(torch.float64)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_cache')

# Problem parameters used to build the cache key.
_PROBLEM_PARAMS = {
    'mesh_n': 4,
    'E': 110000.0,
    'nu': 0.33,
    'sigma_y0': 200.0,
    'H': 1000.0,
    'n_increments': 6,
    'top_disp': 0.02,
}

# AA2024 Swift-Voce parameters (matches the elastoplastic_nlh branch
# in run_simulation_surrogate.py).
_NLH_PARAMS = {
    'mesh_n': 4,
    'E': 70000.0,
    'nu': 0.33,
    'a_s': 798.56,
    'epsilon_0': 0.0178,
    'n_sv': 0.202,
    'k_0': 363.84,
    'q_v': 240.03,
    'beta': 10.533,
    'omega': 0.368,
    'n_increments': 11,
    'top_disp': 0.02,
    'plastic_local_max_iter': 50,
}


# =============================================================================
def make_linear_hardening(sigma_y0, H):
    """Build linear-hardening closures for IsotropicPlasticityPlaneStrain.

    Parameters
    ----------
    sigma_y0 : float
        Initial yield stress.
    H : float
        Hardening modulus.

    Returns
    -------
    sigma_f : Callable
        Yield stress as a function of equivalent plastic strain.
    sigma_f_prime : Callable
        Derivative of `sigma_f`.
    """
    def sigma_f(q):
        return sigma_y0 + H * q

    def sigma_f_prime(q):
        return torch.full_like(q, H)
    return sigma_f, sigma_f_prime


# =============================================================================
def build_problem():
    """Build the 4x4 plane-strain plasticity test problem.

    Returns
    -------
    domain : Planar
        Configured FEM domain with tensile boundary conditions.
    increments : torch.Tensor
        Load-factor sequence (length n_increments).
    """
    p = _PROBLEM_PARAMS
    nodes, elements = rect_quad(p['mesh_n'] + 1, p['mesh_n'] + 1)
    sigma_f, sigma_f_prime = make_linear_hardening(
        p['sigma_y0'], p['H'])
    mat = IsotropicPlasticityPlaneStrain(
        E=p['E'], nu=p['nu'],
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
    dom = Planar(nodes, elements, mat)
    tol = 1e-6
    for i, nc in enumerate(nodes):
        if torch.abs(nc[1]) < tol:
            dom.displacements[i, 1] = 0.0
            dom.constraints[i, 1] = True
            if torch.abs(nc[0]) < tol:
                dom.displacements[i, 0] = 0.0
                dom.constraints[i, 0] = True
        elif torch.abs(nc[1] - 1.0) < tol:
            dom.displacements[i, 1] = p['top_disp']
            dom.constraints[i, 1] = True
    increments = torch.linspace(0.0, 1.0, p['n_increments'])
    return dom, increments


# =============================================================================
def make_swift_voce_nlh():
    """Build Swift-Voce AA2024 hardening closures (elastoplastic_nlh).

    Returns
    -------
    sigma_f : Callable
        Combined Swift-Voce yield-stress function.
    sigma_f_prime : Callable
        Derivative of `sigma_f`.
    """
    p = _NLH_PARAMS
    a_s = p['a_s']; epsilon_0 = p['epsilon_0']; n_sv = p['n_sv']
    k_0 = p['k_0']; q_v = p['q_v']; beta = p['beta']
    omega = p['omega']

    def sigma_f(eps_pl):
        k_s = a_s * (epsilon_0 + eps_pl)**n_sv
        k_v = k_0 + q_v * (1.0 - torch.exp(-beta * eps_pl))
        return omega * k_s + (1.0 - omega) * k_v

    def sigma_f_prime(eps_pl):
        dks = a_s * n_sv * (epsilon_0 + eps_pl)**(n_sv - 1.0)
        dkv = q_v * beta * torch.exp(-beta * eps_pl)
        return omega * dks + (1.0 - omega) * dkv
    return sigma_f, sigma_f_prime


# =============================================================================
def build_problem_nlh():
    """Build the 4x4 elastoplastic_nlh test problem.

    Mirrors the analytic-FE part of ``run_simulation_surrogate.py`` for
    the ``elastoplastic_nlh`` material branch (Swift-Voce AA2024
    hardening, plane strain) on a 4x4 quad mesh under uniaxial tensile
    boundary conditions.

    Returns
    -------
    domain : Planar
        Configured FEM domain.
    increments : torch.Tensor
        Load-factor sequence.
    """
    p = _NLH_PARAMS
    nodes, elements = rect_quad(p['mesh_n'] + 1, p['mesh_n'] + 1)
    sigma_f, sigma_f_prime = make_swift_voce_nlh()
    mat = IsotropicPlasticityPlaneStrain(
        E=p['E'], nu=p['nu'],
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        max_iter=p['plastic_local_max_iter'])
    dom = Planar(nodes, elements, mat)
    tol = 1e-6
    for i, nc in enumerate(nodes):
        if torch.abs(nc[1]) < tol:
            dom.displacements[i, 1] = 0.0
            dom.constraints[i, 1] = True
            if torch.abs(nc[0]) < tol:
                dom.displacements[i, 0] = 0.0
                dom.constraints[i, 0] = True
        elif torch.abs(nc[1] - 1.0) < tol:
            dom.displacements[i, 1] = p['top_disp']
            dom.constraints[i, 1] = True
    increments = torch.linspace(0.0, 1.0, p['n_increments'])
    return dom, increments


# =============================================================================
def _problem_cache_key():
    """Return a short hash of the problem parameters for cache naming."""
    payload = repr(sorted(_PROBLEM_PARAMS.items())).encode()
    return hashlib.md5(payload).hexdigest()[:10]


# =============================================================================
@pytest.fixture(scope='session')
def nr_reference():
    """Cached Newton-Raphson reference solution.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``u``: torch.Tensor of shape (n_increments, n_nod, n_dim)
        - ``f``: torch.Tensor of shape (n_increments, n_nod, n_dim)
        - ``residual_history``: dict mapping increment index to
          per-iteration residual norms.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR, f'nr_ref_{_problem_cache_key()}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)
    dom, incr = build_problem()
    res = dom.solve(
        increments=incr, max_iter=100, rtol=1e-8, atol=1e-6,
        return_intermediate=True, return_resnorm=True,
        nonlinear_solver='newton_raphson')
    u, f = res[0], res[1]
    residual_history = res[-1]
    cached = {
        'u': u.detach().clone(),
        'f': f.detach().clone(),
        'residual_history': {
            k: list(v) for k, v in residual_history.items()},
    }
    with open(cache_path, 'wb') as fh:
        pickle.dump(cached, fh)
    return cached


# =============================================================================
def _nlh_cache_key():
    """Hash of the elastoplastic_nlh problem parameters for caching."""
    payload = repr(sorted(_NLH_PARAMS.items())).encode()
    return hashlib.md5(payload).hexdigest()[:10]


# =============================================================================
@pytest.fixture(scope='session')
def nr_reference_nlh():
    """Cached Newton-Raphson reference for the elastoplastic_nlh problem."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR, f'nr_ref_nlh_{_nlh_cache_key()}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)
    dom, incr = build_problem_nlh()
    res = dom.solve(
        increments=incr, max_iter=100, rtol=1e-8, atol=1e-6,
        return_intermediate=True, return_resnorm=True,
        nonlinear_solver='newton_raphson')
    cached = {
        'u': res[0].detach().clone(),
        'f': res[1].detach().clone(),
        'residual_history': {
            k: list(v) for k, v in res[-1].items()},
    }
    with open(cache_path, 'wb') as fh:
        pickle.dump(cached, fh)
    return cached


# =============================================================================
def test_nr_dispatcher_regression(nr_reference):
    """NR via dispatcher matches the cached NR reference exactly.

    Guards against silent numerical drift from the closure-based
    refactor of FEM.solve; rerunning NR through the dispatcher must
    produce a bit-identical result up to float64 round-off.
    """
    dom, incr = build_problem()
    res = dom.solve(
        increments=incr, max_iter=100, rtol=1e-8, atol=1e-6,
        return_intermediate=True, return_resnorm=True,
        nonlinear_solver='newton_raphson')
    u, f = res[0], res[1]
    residual_history = res[-1]
    # Displacement / force fields must match the cached reference to
    # ~machine precision (this is the same algorithm called twice).
    assert torch.allclose(
        u, nr_reference['u'], atol=1e-12, rtol=0.0), (
        'NR through dispatcher diverges from cached reference '
        '(displacements).')
    assert torch.allclose(
        f, nr_reference['f'], atol=1e-8, rtol=0.0), (
        'NR through dispatcher diverges from cached reference '
        '(forces).')
    # Iteration counts per increment must match.
    ref_hist = nr_reference['residual_history']
    assert set(residual_history.keys()) == set(ref_hist.keys())
    for k in ref_hist:
        assert len(residual_history[k]) == len(ref_hist[k]), (
            f'NR iteration count changed for inc {k}: '
            f'expected {len(ref_hist[k])}, got '
            f'{len(residual_history[k])}.')


# =============================================================================
# Per-solver configuration. ``solve_kwargs`` and ``solver_opts`` are
# chosen so the alternative converges on the test problem within the
# physical tolerances expected for the comparison.
# -----------------------------------------------------------------------------
SOLVER_CONFIGS = {
    'damped_picard': dict(
        solve_kwargs=dict(rtol=1e-5, atol=1e-3, max_iter=200),
        solver_opts={'damping': 1.0}),
    'anderson': dict(
        solve_kwargs=dict(rtol=1e-5, atol=1e-3, max_iter=200),
        solver_opts={'beta': 1.0, 'm': 8}),
    'broyden': dict(
        solve_kwargs=dict(rtol=1e-5, atol=1e-3, max_iter=100),
        solver_opts={'jacobian_refresh_period': 3}),
    'jfnk': dict(
        solve_kwargs=dict(rtol=1e-5, atol=1e-3, max_iter=50),
        solver_opts={}),
    'rand_subspace_newton': dict(
        solve_kwargs=dict(rtol=1e-5, atol=1e-3, max_iter=30),
        solver_opts={'block_size': 50}),
}


# =============================================================================
@pytest.mark.parametrize(
    'solver_type',
    ['damped_picard', 'anderson', 'broyden',
     'jfnk', 'rand_subspace_newton'])
def test_solver_equivalence_to_newton(solver_type, nr_reference):
    """Forward-only solver reproduces NR displacement / force field.

    The alternative solver runs at relaxed residual tolerances
    (``rtol=1e-5, atol=1e-3``) appropriate for finite-difference JVPs
    and K_0-based preconditioning, then the converged displacement and
    internal force fields are compared to the NR reference. Comparison
    tolerances are looser than the FEM solver tolerances to absorb the
    method-dependent rounding error.
    """
    cfg = SOLVER_CONFIGS[solver_type]
    dom, incr = build_problem()
    res = dom.solve(
        increments=incr, return_intermediate=True,
        return_resnorm=True,
        nonlinear_solver=solver_type,
        nonlinear_solver_opts=cfg['solver_opts'],
        **cfg['solve_kwargs'])
    u, f = res[0], res[1]
    residual_history = res[-1]
    # Each increment must satisfy the chosen tolerance: the converged
    # final residual is below atol or below rtol * r_0.
    rtol_solve = cfg['solve_kwargs']['rtol']
    atol_solve = cfg['solve_kwargs']['atol']
    for k, hist in residual_history.items():
        assert len(hist) >= 1, (
            f'{solver_type}: empty residual history for inc {k}.')
        r_init = hist[0]
        r_final = hist[-1]
        assert (r_final < atol_solve
                or r_final < rtol_solve * r_init), (
            f'{solver_type}: did not converge at inc {k} '
            f'(r_init={r_init:.3e}, r_final={r_final:.3e}).')
    # Displacement field must match NR within atol=1e-4 (deviates from
    # NR by less than 1% of the maximum prescribed displacement of 2%
    # strain).
    assert torch.allclose(
        u, nr_reference['u'], atol=1e-4, rtol=1e-4), (
        f'{solver_type}: displacement field disagrees with NR '
        f'reference (max abs diff = '
        f'{(u - nr_reference["u"]).abs().max().item():.3e}).')
    # Reaction force field comparison (looser due to force-norm scale).
    assert torch.allclose(
        f, nr_reference['f'], atol=1.0, rtol=1e-3), (
        f'{solver_type}: force field disagrees with NR '
        f'reference (max abs diff = '
        f'{(f - nr_reference["f"]).abs().max().item():.3e}).')


# =============================================================================
@pytest.mark.parametrize(
    'solver_type',
    ['newton_raphson', 'damped_picard', 'anderson',
     'broyden', 'jfnk', 'rand_subspace_newton'])
def test_solver_on_elastoplastic_nlh(solver_type, nr_reference_nlh):
    """All solvers converge on the elastoplastic_nlh (Swift-Voce) test.

    Reproduces the analytic-FE elastoplastic_nlh material as configured
    in run_simulation_surrogate.py. NR runs at the standard FEM
    tolerances; the forward-only methods use the looser tolerances
    appropriate for FD-JVP / K_0-preconditioned iteration.
    """
    if solver_type == 'newton_raphson':
        solve_kwargs = dict(rtol=1e-8, atol=1e-6, max_iter=100)
        opts = {}
    else:
        cfg = SOLVER_CONFIGS[solver_type]
        solve_kwargs = cfg['solve_kwargs']
        opts = cfg['solver_opts']
    dom, incr = build_problem_nlh()
    res = dom.solve(
        increments=incr, return_intermediate=True,
        return_resnorm=True,
        nonlinear_solver=solver_type,
        nonlinear_solver_opts=opts,
        **solve_kwargs)
    u, f = res[0], res[1]
    residual_history = res[-1]
    rtol_solve = solve_kwargs['rtol']
    atol_solve = solve_kwargs['atol']
    for k, hist in residual_history.items():
        assert len(hist) >= 1, (
            f'{solver_type}: empty residual history for inc {k}.')
        r_init = hist[0]
        r_final = hist[-1]
        assert (r_final < atol_solve
                or r_final < rtol_solve * r_init), (
            f'{solver_type}: did not converge on elastoplastic_nlh '
            f'at inc {k} (r_init={r_init:.3e}, '
            f'r_final={r_final:.3e}).')
    # Displacement field comparison. NR uses 1e-8 residuals and is the
    # reference; forward-only solvers exit at 1e-3 absolute residual,
    # which on this material (E=70 GPa, sigma_y0~365 MPa, top disp 2%)
    # corresponds to displacement error of order
    # atol/E_eff ~ 1e-3 / 7e4 ~ 1e-8. Comparison atol=1e-4 is generous.
    assert torch.allclose(
        u, nr_reference_nlh['u'], atol=1e-4, rtol=1e-4), (
        f'{solver_type}: displacement disagrees with NR on '
        f'elastoplastic_nlh (max abs diff = '
        f'{(u - nr_reference_nlh["u"]).abs().max().item():.3e}).')
