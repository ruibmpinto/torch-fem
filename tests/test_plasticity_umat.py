"""Tests for UMAT-style plasticity materials.

Covers single-element and multi-element equivalence between the
analytic Simo-Taylor consistent tangent (parent classes
`IsotropicPlasticityPlaneStrain`, `IsotropicPlasticity3D`) and the
forward-difference perturbation tangent used by the UMAT subclasses.

Classes
-------
(none — pytest functional-style)

Functions
---------
make_linear_hardening
    Build linear-hardening `sigma_f` / `sigma_f_prime` closures.
make_swift_voce
    Build Swift-Voce nonlinear hardening closures (AA2024).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from math import sqrt
# Third-party
import pytest
import torch
# Local
from torchfem.materials import (
    IsotropicPlasticityPlaneStrain,
    IsotropicPlasticityPlaneStrainUMAT,
    IsotropicPlasticity3D,
    IsotropicPlasticity3DUMAT,
)
# =============================================================================
#
# =============================================================================
torch.set_default_dtype(torch.float64)


# =============================================================================
def make_linear_hardening(sigma_y=250.0, H=1000.0):
    """Build linear-hardening closures.

    Parameters
    ----------
    sigma_y : float, default=250.0
        Initial yield stress.
    H : float, default=1000.0
        Linear hardening modulus.

    Returns
    -------
    sigma_f : Callable
        Yield-stress function of equivalent plastic strain.
    sigma_f_prime : Callable
        Derivative of `sigma_f`.
    """
    def sigma_f(q):
        return sigma_y + H * q

    def sigma_f_prime(q):
        return torch.full_like(q, H)
    return sigma_f, sigma_f_prime


# =============================================================================
def make_swift_voce():
    """Build Swift-Voce AA2024 hardening closures.

    Returns
    -------
    e_young : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_f : Callable
        Yield-stress function.
    sigma_f_prime : Callable
        Derivative of `sigma_f`.
    """
    e_young = 70000.0
    nu = 0.33
    a_s = 798.56
    epsilon_0 = 0.0178
    n_exp = 0.202
    k_0 = 363.84
    q_v = 240.03
    beta = 10.533
    omega = 0.368

    def sigma_f(eps_pl):
        k_s = a_s * (epsilon_0 + eps_pl)**n_exp
        k_v = k_0 + q_v * (1.0 - torch.exp(-beta * eps_pl))
        return omega * k_s + (1.0 - omega) * k_v

    def sigma_f_prime(eps_pl):
        dks = a_s * n_exp * (epsilon_0 + eps_pl)**(n_exp - 1.0)
        dkv = q_v * beta * torch.exp(-beta * eps_pl)
        return omega * dks + (1.0 - omega) * dkv
    return e_young, nu, sigma_f, sigma_f_prime


# =============================================================================
def _make_mat_2d(is_analytical_tangent, n_elem=1,
                 hardening='swift_voce', **kwargs):
    """Build a vectorised 2D plane-strain UMAT material."""
    if hardening == 'swift_voce':
        e_young, nu, sigma_f, sigma_f_prime = make_swift_voce()
    elif hardening == 'linear':
        e_young = 70000.0
        nu = 0.33
        sigma_f, sigma_f_prime = make_linear_hardening(**kwargs)
    else:
        raise ValueError(hardening)
    mat = IsotropicPlasticityPlaneStrainUMAT(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        is_analytical_tangent=is_analytical_tangent)
    return mat.vectorize(n_elem)


# =============================================================================
def _make_parent_2d(n_elem=1, hardening='swift_voce', **kwargs):
    """Build a vectorised 2D parent plane-strain material."""
    if hardening == 'swift_voce':
        e_young, nu, sigma_f, sigma_f_prime = make_swift_voce()
    elif hardening == 'linear':
        e_young = 70000.0
        nu = 0.33
        sigma_f, sigma_f_prime = make_linear_hardening(**kwargs)
    else:
        raise ValueError(hardening)
    mat = IsotropicPlasticityPlaneStrain(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
    return mat.vectorize(n_elem)


# =============================================================================
def _make_mat_3d(is_analytical_tangent, n_elem=1,
                 hardening='swift_voce', **kwargs):
    """Build a vectorised 3D UMAT material."""
    if hardening == 'swift_voce':
        e_young, nu, sigma_f, sigma_f_prime = make_swift_voce()
    elif hardening == 'linear':
        e_young = 70000.0
        nu = 0.33
        sigma_f, sigma_f_prime = make_linear_hardening(**kwargs)
    else:
        raise ValueError(hardening)
    mat = IsotropicPlasticity3DUMAT(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        is_analytical_tangent=is_analytical_tangent)
    return mat.vectorize(n_elem)


# =============================================================================
def _make_parent_3d(n_elem=1, hardening='swift_voce', **kwargs):
    """Build a vectorised 3D parent material."""
    if hardening == 'swift_voce':
        e_young, nu, sigma_f, sigma_f_prime = make_swift_voce()
    elif hardening == 'linear':
        e_young = 70000.0
        nu = 0.33
        sigma_f, sigma_f_prime = make_linear_hardening(**kwargs)
    else:
        raise ValueError(hardening)
    mat = IsotropicPlasticity3D(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
    return mat.vectorize(n_elem)


# =============================================================================
def _zeros_2d(n_elem):
    """Zero sigma/state/de0 tensors for a batch of 2D material points."""
    sigma = torch.zeros(n_elem, 2, 2)
    state = torch.zeros(n_elem, 2)
    de0 = torch.zeros(n_elem, 2, 2)
    F = torch.eye(2).expand(n_elem, 2, 2).clone()
    return sigma, state, de0, F


# =============================================================================
def _zeros_3d(n_elem):
    """Zero sigma/state/de0 tensors for a batch of 3D material points."""
    sigma = torch.zeros(n_elem, 3, 3)
    state = torch.zeros(n_elem, 1)
    de0 = torch.zeros(n_elem, 3, 3)
    F = torch.eye(3).expand(n_elem, 3, 3).clone()
    return sigma, state, de0, F


# =============================================================================
# Single-element elastic-step tests
# =============================================================================
def test_elastic_step_2d_both_modes_match_C():
    """Elastic step below yield: both modes give ddsdde = C."""
    mat_an = _make_mat_2d(is_analytical_tangent=True, n_elem=1)
    mat_pt = _make_mat_2d(is_analytical_tangent=False, n_elem=1)
    sigma, state, de0, F = _zeros_2d(1)
    H_inc = 1e-5 * torch.eye(2).unsqueeze(0)
    s_an, st_an, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    s_pt, st_pt, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # No yielding expected -> ddsdde equals elastic C
    assert torch.allclose(dd_an, mat_an.C)
    assert torch.allclose(dd_pt, mat_pt.C)
    assert torch.allclose(s_an, s_pt)
    assert torch.allclose(st_an, st_pt)


# -----------------------------------------------------------------------------
def test_elastic_step_3d_both_modes_match_C():
    """Elastic step below yield (3D): both modes give ddsdde = C."""
    mat_an = _make_mat_3d(is_analytical_tangent=True, n_elem=1)
    mat_pt = _make_mat_3d(is_analytical_tangent=False, n_elem=1)
    sigma, state, de0, F = _zeros_3d(1)
    H_inc = 1e-5 * torch.eye(3).unsqueeze(0)
    s_an, st_an, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    s_pt, st_pt, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    assert torch.allclose(dd_an, mat_an.C)
    assert torch.allclose(dd_pt, mat_pt.C)
    assert torch.allclose(s_an, s_pt)
    assert torch.allclose(st_an, st_pt)


# =============================================================================
# Single-element plastic-step tests
# =============================================================================
def test_plastic_step_2d_linear_hardening_agrees():
    """Linear hardening: analytic and perturbation ddsdde agree."""
    mat_an = _make_mat_2d(is_analytical_tangent=True, n_elem=1,
                          hardening='linear')
    mat_pt = _make_mat_2d(is_analytical_tangent=False, n_elem=1,
                          hardening='linear')
    sigma, state, de0, F = _zeros_2d(1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Uniaxial strain well above yield
    H_inc = torch.tensor([[[0.02, 0.0], [0.0, 0.0]]])
    _, _, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    _, _, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    err = (dd_pt - dd_an).abs().max().item()
    assert err < 1.0, f'ddsdde abs err {err:.3e}'


# -----------------------------------------------------------------------------
@pytest.mark.parametrize('eps_pl', [0.001, 0.01, 0.05])
def test_plastic_step_2d_swift_voce_agrees(eps_pl):
    """Swift-Voce nonlinear: perturbation matches analytic."""
    mat_an = _make_mat_2d(is_analytical_tangent=True, n_elem=1)
    mat_pt = _make_mat_2d(is_analytical_tangent=False, n_elem=1)
    sigma, state, de0, F = _zeros_2d(1)
    H_inc = torch.tensor([[[eps_pl, 0.0], [0.0, 0.0]]])
    _, _, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    _, _, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    err_abs = (dd_pt - dd_an).abs().max().item()
    err_rel = err_abs / (dd_an.abs().max().item() + 1e-30)
    assert err_rel < 1e-3, (
        f'ddsdde rel err {err_rel:.3e} abs {err_abs:.3e}')


# -----------------------------------------------------------------------------
@pytest.mark.parametrize('eps_pl', [0.005, 0.02])
def test_plastic_step_3d_swift_voce_agrees(eps_pl):
    """3D Swift-Voce: perturbation matches analytic."""
    mat_an = _make_mat_3d(is_analytical_tangent=True, n_elem=1)
    mat_pt = _make_mat_3d(is_analytical_tangent=False, n_elem=1)
    sigma, state, de0, F = _zeros_3d(1)
    H_inc = torch.zeros(1, 3, 3)
    H_inc[0, 0, 0] = eps_pl
    _, _, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    _, _, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    err_abs = (dd_pt - dd_an).abs().max().item()
    err_rel = err_abs / (dd_an.abs().max().item() + 1e-30)
    assert err_rel < 1e-3, (
        f'3D ddsdde rel err {err_rel:.3e} abs {err_abs:.3e}')


# =============================================================================
# Determinism test
# =============================================================================
def test_return_map_deterministic_2d():
    """Repeated calls on same inputs yield identical outputs."""
    mat = _make_mat_2d(is_analytical_tangent=False, n_elem=1)
    sigma, state, de0, F = _zeros_2d(1)
    de = 0.5 * (torch.tensor([[[0.02, 0.0], [0.0, 0.0]]])
                + torch.tensor([[[0.02, 0.0], [0.0, 0.0]]])
                .transpose(-1, -2))
    out = [mat._return_map(de, sigma, state, de0)
           for _ in range(4)]
    for s_i, st_i, fm_i in out[1:]:
        assert torch.equal(s_i, out[0][0])
        assert torch.equal(st_i, out[0][1])
        assert torch.equal(fm_i, out[0][2])


# =============================================================================
# Multi-element tests
# =============================================================================
def test_multi_element_all_elastic_2d():
    """All elastic: perturbation short-circuits to C."""
    n_elem = 20
    mat_pt = _make_mat_2d(is_analytical_tangent=False, n_elem=n_elem)
    sigma, state, de0, F = _zeros_2d(n_elem)
    H_inc = 1e-5 * torch.eye(2).expand(n_elem, 2, 2).clone()
    _, _, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    assert torch.allclose(dd_pt, mat_pt.C)


# -----------------------------------------------------------------------------
def test_multi_element_uniform_plastic_2d():
    """All yielding: per-point ddsdde matches analytic."""
    n_elem = 20
    mat_an = _make_mat_2d(is_analytical_tangent=True, n_elem=n_elem)
    mat_pt = _make_mat_2d(is_analytical_tangent=False, n_elem=n_elem)
    sigma, state, de0, F = _zeros_2d(n_elem)
    H_inc = torch.zeros(n_elem, 2, 2)
    H_inc[:, 0, 0] = 0.02
    s_an, st_an, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    s_pt, st_pt, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    assert torch.allclose(s_an, s_pt, atol=1e-10)
    assert torch.allclose(st_an, st_pt, atol=1e-10)
    err = (dd_pt - dd_an).abs().max().item()
    denom = dd_an.abs().max().item() + 1e-30
    assert err / denom < 1e-3, f'rel err {err/denom:.3e}'


# -----------------------------------------------------------------------------
def test_multi_element_mixed_loading_2d():
    """Mixed elastic/plastic: per-integration-point equivalence."""
    n_elem = 40
    mat_an = _make_mat_2d(is_analytical_tangent=True, n_elem=n_elem)
    mat_pt = _make_mat_2d(is_analytical_tangent=False, n_elem=n_elem)
    sigma, state, de0, F = _zeros_2d(n_elem)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Linear ramp: half elastic, half plastic
    eps_vals = torch.linspace(1e-5, 0.03, n_elem)
    H_inc = torch.zeros(n_elem, 2, 2)
    H_inc[:, 0, 0] = eps_vals
    s_an, st_an, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    s_pt, st_pt, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute fm from parent code to split elastic/plastic
    assert torch.allclose(s_an, s_pt, atol=1e-10)
    assert torch.allclose(st_an, st_pt, atol=1e-10)
    err = (dd_pt - dd_an).abs().max().item()
    denom = dd_an.abs().max().item() + 1e-30
    assert err / denom < 1e-3, f'rel err {err/denom:.3e}'


# -----------------------------------------------------------------------------
def test_multi_element_all_elastic_3d():
    """3D all elastic: perturbation short-circuits to C."""
    n_elem = 10
    mat_pt = _make_mat_3d(is_analytical_tangent=False, n_elem=n_elem)
    sigma, state, de0, F = _zeros_3d(n_elem)
    H_inc = 1e-5 * torch.eye(3).expand(n_elem, 3, 3).clone()
    _, _, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    assert torch.allclose(dd_pt, mat_pt.C)


# -----------------------------------------------------------------------------
def test_multi_element_uniform_plastic_3d():
    """3D all yielding: per-point ddsdde matches analytic."""
    n_elem = 10
    mat_an = _make_mat_3d(is_analytical_tangent=True, n_elem=n_elem)
    mat_pt = _make_mat_3d(is_analytical_tangent=False, n_elem=n_elem)
    sigma, state, de0, F = _zeros_3d(n_elem)
    H_inc = torch.zeros(n_elem, 3, 3)
    H_inc[:, 0, 0] = 0.02
    s_an, st_an, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    s_pt, st_pt, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    assert torch.allclose(s_an, s_pt, atol=1e-10)
    assert torch.allclose(st_an, st_pt, atol=1e-10)
    err = (dd_pt - dd_an).abs().max().item()
    denom = dd_an.abs().max().item() + 1e-30
    assert err / denom < 1e-3, f'3D rel err {err/denom:.3e}'


# -----------------------------------------------------------------------------
def test_multi_element_mixed_loading_3d():
    """3D mixed elastic/plastic: per-point equivalence."""
    n_elem = 20
    mat_an = _make_mat_3d(is_analytical_tangent=True, n_elem=n_elem)
    mat_pt = _make_mat_3d(is_analytical_tangent=False, n_elem=n_elem)
    sigma, state, de0, F = _zeros_3d(n_elem)
    eps_vals = torch.linspace(1e-5, 0.03, n_elem)
    H_inc = torch.zeros(n_elem, 3, 3)
    H_inc[:, 0, 0] = eps_vals
    s_an, st_an, dd_an = mat_an.step(H_inc, F, sigma, state, de0)
    s_pt, st_pt, dd_pt = mat_pt.step(H_inc, F, sigma, state, de0)
    assert torch.allclose(s_an, s_pt, atol=1e-10)
    assert torch.allclose(st_an, st_pt, atol=1e-10)
    err = (dd_pt - dd_an).abs().max().item()
    denom = dd_an.abs().max().item() + 1e-30
    assert err / denom < 1e-3, f'3D rel err {err/denom:.3e}'


# =============================================================================
# Full-simulation multi-element equivalence (2D, small mesh)
# =============================================================================
def test_full_simulation_2d_agrees_with_parent():
    """Small tensile solve: UMAT analytic mode matches parent."""
    from torchfem import Planar
    from torchfem.mesh import rect_quad
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build mesh
    mesh_n = 4
    nodes, elements = rect_quad(mesh_n + 1, mesh_n + 1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build non-vectorised materials so Planar can size them per mesh
    e_young, nu, sigma_f, sigma_f_prime = make_swift_voce()
    mat_parent = IsotropicPlasticityPlaneStrain(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
    mat_umat_an = IsotropicPlasticityPlaneStrainUMAT(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        is_analytical_tangent=True)

    def build_domain(mat):
        dom = Planar(nodes, elements, mat)
        # Tension BCs
        tol = 1e-6
        for i, nc in enumerate(nodes):
            if torch.abs(nc[1]) < tol:
                dom.displacements[i, 1] = 0.0
                dom.constraints[i, 1] = True
                if torch.abs(nc[0]) < tol:
                    dom.displacements[i, 0] = 0.0
                    dom.constraints[i, 0] = True
            elif torch.abs(nc[1] - 1.0) < tol:
                dom.displacements[i, 1] = 0.02
                dom.constraints[i, 1] = True
        return dom
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve both and compare converged displacement field
    u_parent = build_domain(mat_parent).solve(
        increments=torch.linspace(0.0, 1.0, 6), rtol=1e-8)[0]
    u_umat_an = build_domain(mat_umat_an).solve(
        increments=torch.linspace(0.0, 1.0, 6), rtol=1e-8)[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # UMAT analytic mode should reproduce parent exactly
    err = (u_umat_an - u_parent).abs().max().item()
    denom = u_parent.abs().max().item() + 1e-30
    assert err / denom < 1e-10, (
        f'UMAT-analytic vs parent disp rel err {err/denom:.3e}')


# -----------------------------------------------------------------------------
def test_full_simulation_2d_perturbation_agrees_with_parent():
    """4x4 mesh tensile: UMAT perturbation ~= parent analytic."""
    from torchfem import Planar
    from torchfem.mesh import rect_quad
    mesh_n = 4
    nodes, elements = rect_quad(mesh_n + 1, mesh_n + 1)
    e_young, nu, sigma_f, sigma_f_prime = make_swift_voce()
    mat_parent = IsotropicPlasticityPlaneStrain(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
    mat_umat_pt = IsotropicPlasticityPlaneStrainUMAT(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        is_analytical_tangent=False)

    def build_domain(mat):
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
                dom.displacements[i, 1] = 0.02
                dom.constraints[i, 1] = True
        return dom

    u_parent = build_domain(mat_parent).solve(
        increments=torch.linspace(0.0, 1.0, 6), rtol=1e-8)[0]
    u_umat_pt = build_domain(mat_umat_pt).solve(
        increments=torch.linspace(0.0, 1.0, 6), rtol=1e-6)[0]
    err = (u_umat_pt - u_parent).abs().max().item()
    denom = u_parent.abs().max().item() + 1e-30
    assert err / denom < 1e-4, (
        f'UMAT-pert vs parent disp rel err {err/denom:.3e}')
