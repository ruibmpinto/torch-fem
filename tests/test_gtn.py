"""Unit tests for the GTN porous-plasticity custom material.

Tensor convention: torch-fem native full tensors — stress and strain
increments of shape (..., 3, 3), stiffness of shape (..., 3, 3, 3, 3).
All GTN parameters are passed explicitly; there are no defaults.
"""

import pytest
import torch

from torchfem.custom_materials.gtn import (
    GTN3D,
    f_star,
    gtn_flow_normal,
    gtn_return_map,
    gtn_yield,
    nucleation_intensity,
    stress_invariants,
)
from torchfem.materials import IsotropicElasticity3D, IsotropicPlasticity3D

# Use double precision throughout: the bisection tolerance on the
# yield function is far below float32 resolution. Scoped per test
# and restored afterwards so other test modules are unaffected.
@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def linear_flow_stress(peeq):
    # Simple linear matrix hardening used across all tests.
    return 100.0 + 10.0 * peeq


def linear_flow_stress_prime(peeq):
    # Analytic slope of the linear hardening (for the J2 oracle).
    return 10.0 * torch.ones_like(peeq)


def make_params(**overrides):
    # Canonical Tvergaard-style test parameters; every key explicit.
    params = {
        "q1": 1.5,
        "q2": 1.0,
        "fc": 0.15,
        "ff": 0.25,
        "f_n": 0.04,
        "eps_n": 0.3,
        "s_n": 0.1,
        "tol": 1.0e-10,
        "max_iter": 200,
    }
    params.update(overrides)
    return params


def elastic_stiffness(n_elem):
    # Vectorized isotropic elastic stiffness from torch-fem itself.
    material = IsotropicElasticity3D(1000.0, 0.3).vectorize(n_elem)
    return material.C


def uniaxial_strain_increment(n_elem, magnitude):
    # Uniaxial strain increment along x, batched over elements.
    de = torch.zeros(n_elem, 3, 3)
    de[:, 0, 0] = magnitude
    return de


def shear_strain_increment(n_elem, magnitude):
    # Pure (traceless) shear strain increment in the xy plane.
    de = torch.zeros(n_elem, 3, 3)
    de[:, 0, 1] = magnitude
    de[:, 1, 0] = magnitude
    return de


def run_return_map(de, f0, params, sigma_n=None, peeq_n=None,
                   dlam_warm=None):
    # Convenience wrapper: zero initial state unless overridden.
    n_elem = de.shape[0]
    C = elastic_stiffness(n_elem)
    if sigma_n is None:
        sigma_n = torch.zeros(n_elem, 3, 3)
    if peeq_n is None:
        peeq_n = torch.zeros(n_elem)
    if dlam_warm is None:
        dlam_warm = torch.zeros(n_elem)
    f_n = f0 * torch.ones(n_elem)
    return gtn_return_map(
        sigma_n, peeq_n, f_n, dlam_warm, de, C,
        linear_flow_stress, params)


# =====================================================================
# Parameter validation: no silent defaults, unknown keys rejected.
# =====================================================================
def test_params_missing_key_raises():
    params = make_params()
    del params["fc"]
    de = uniaxial_strain_increment(2, 1.0e-3)
    with pytest.raises(ValueError, match="fc"):
        run_return_map(de, 0.0, params)


def test_params_unknown_key_raises():
    params = make_params(bogus=1.0)
    de = uniaxial_strain_increment(2, 1.0e-3)
    with pytest.raises(ValueError, match="bogus"):
        run_return_map(de, 0.0, params)


# =====================================================================
# Pure functions.
# =====================================================================
def test_stress_invariants():
    # Hand-built stress with known hydrostatic and von Mises values.
    sigma = torch.zeros(1, 3, 3)
    sigma[0, 0, 0] = 3.0
    sigma_h, sigma_eq, s_dev = stress_invariants(sigma)
    assert torch.allclose(sigma_h, torch.tensor([1.0]))
    assert torch.allclose(sigma_eq, torch.tensor([3.0]))
    # Deviator of uniaxial stress: diag(2, -1, -1).
    expected_dev = torch.diag(torch.tensor([2.0, -1.0, -1.0]))
    assert torch.allclose(s_dev[0], expected_dev)


def test_fstar_mapping():
    q1, fc, ff = 1.5, 0.15, 0.25
    # Below fc the mapping is the identity.
    f = torch.tensor([0.0, 0.10, 0.15])
    assert torch.allclose(f_star(f, q1, fc, ff), f)
    # Above fc the mapping is a linear ramp with the standard slope.
    f_mid = torch.tensor([0.20])
    slope = (1.0 / q1 - fc) / (ff - fc)
    expected = fc + slope * (f_mid - fc)
    assert torch.allclose(f_star(f_mid, q1, fc, ff), expected)
    # At and beyond ff the mapping is clamped at the ultimate 1/q1.
    f_hi = torch.tensor([0.25, 0.40])
    ult = torch.tensor([1.0 / q1, 1.0 / q1])
    assert torch.allclose(f_star(f_hi, q1, fc, ff), ult)


def test_gtn_yield_dense_limit_is_j2():
    # With fs = 0 the GTN surface reduces to (sigma_eq/sigma_y)^2 - 1.
    sigma_eq = torch.tensor([50.0, 100.0, 200.0])
    sigma_h = torch.tensor([500.0, -500.0, 0.0])
    sigma_y = torch.tensor([100.0, 100.0, 100.0])
    fs = torch.zeros(3)
    phi = gtn_yield(sigma_eq, sigma_h, sigma_y, fs, 1.5, 1.0)
    expected = (sigma_eq / sigma_y) ** 2 - 1.0
    assert torch.allclose(phi, expected)


def test_nucleation_gaussian_peak():
    f_n, eps_n, s_n = 0.04, 0.3, 0.1
    # The intensity peaks at eps_n with the standard Gaussian height.
    peak = nucleation_intensity(torch.tensor([eps_n]), f_n, eps_n, s_n)
    expected = f_n / (s_n * torch.sqrt(
        torch.tensor(2.0 * torch.pi)))
    assert torch.allclose(peak, expected.reshape(1))
    # Far from eps_n the intensity vanishes.
    far = nucleation_intensity(torch.tensor([5.0]), f_n, eps_n, s_n)
    assert far.item() < 1.0e-12


def test_flow_normal_unit_and_volumetric():
    # Deviatoric trial state with porosity: the normal has both a
    # deviatoric and a volumetric part and unit Frobenius norm.
    sigma = torch.zeros(1, 3, 3)
    sigma[0, 0, 0] = 150.0
    sigma_h, _, s_dev = stress_invariants(sigma)
    sigma_y = torch.tensor([100.0])
    fs = torch.tensor([0.05])
    n = gtn_flow_normal(s_dev, sigma_h, sigma_y, fs, 1.5, 1.0)
    norm = torch.linalg.norm(n, dim=(-1, -2))
    assert torch.allclose(norm, torch.ones(1))
    # tr(n) > 0 under tension with fs > 0 (volumetric growth part).
    assert n[0].diagonal().sum().item() > 0.0
    # With fs = 0 the volumetric part vanishes: tr(n) = 0.
    n0 = gtn_flow_normal(
        s_dev, sigma_h, sigma_y, torch.zeros(1), 1.5, 1.0)
    assert abs(n0[0].diagonal().sum().item()) < 1.0e-12


# =====================================================================
# Return map behavior.
# =====================================================================
def test_elastic_step_unchanged():
    # A sub-yield increment returns the elastic trial stress exactly,
    # with no porosity change and no plastic strain.
    params = make_params()
    de = uniaxial_strain_increment(3, 1.0e-5)
    out = run_return_map(de, 0.01, params)
    C = elastic_stiffness(3)
    sigma_trial = torch.einsum("...ijkl,...kl->...ij", C, de)
    assert torch.allclose(out["sigma"], sigma_trial)
    assert torch.all(out["dpeeq_m"] == 0.0)
    assert torch.allclose(out["f"], 0.01 * torch.ones(3))


def test_dense_limit_matches_j2():
    # f0 = 0 and f_n = 0 reduce GTN to von Mises plasticity; the
    # torch-fem J2 return map is the oracle. Work conjugacy with the
    # updated stress makes ep exactly the J2 equivalent plastic
    # strain increment (see module docstring).
    params = make_params(f_n=0.0)
    n_elem = 4
    de = uniaxial_strain_increment(n_elem, 2.0e-1)
    out = run_return_map(de, 0.0, params)
    # torch-fem J2 oracle with the identical hardening callable.
    j2 = IsotropicPlasticity3D(
        1000.0, 0.3, linear_flow_stress, linear_flow_stress_prime,
        abstol=1.0e-12, max_iter=50).vectorize(n_elem)
    F = torch.eye(3).expand(n_elem, 3, 3)
    sigma0 = torch.zeros(n_elem, 3, 3)
    state0 = torch.zeros(n_elem, 1)
    de0 = torch.zeros(n_elem, 3, 3)
    sigma_j2, state_j2, _ = j2.step(de.clone(), F, sigma0, state0, de0)
    assert torch.allclose(out["sigma"], sigma_j2, rtol=1.0e-6,
                          atol=1.0e-6)
    assert torch.allclose(out["dpeeq_m"], state_j2[:, 0],
                          rtol=1.0e-6, atol=1.0e-10)
    # Porosity stays at machine-precision zero in the dense
    # limit (the deviatoric normal has trace O(1e-17)).
    assert torch.all(out["f"].abs() < 1.0e-12)


def test_hydrostatic_yield_with_porosity():
    # Pure hydrostatic tension: dense J2 never yields (the trial
    # deviator is zero), while GTN with porosity activates through
    # the cosh term. Under strain-driven pure hydrostatic overshoot
    # the porous response is GTN's cavitation instability (void
    # growth inflates the yield function faster than pressure
    # relief deflates it), which the return map refuses loudly
    # instead of accepting the zero-dissipation collapse state.
    params = make_params(f_n=0.0)
    n_elem = 1
    de = torch.zeros(n_elem, 3, 3)
    # Triaxial strain just above the porous yield onset
    # (sigma_h = 200 vs onset 172.6 for f = 0.05).
    de[:, 0, 0] = de[:, 1, 1] = de[:, 2, 2] = 8.0e-2
    out_dense = run_return_map(de, 0.0, params)
    assert torch.all(out_dense["dpeeq_m"] == 0.0)
    with pytest.raises(RuntimeError, match="cavitation"):
        run_return_map(de, 0.05, params)


def test_growth_sign():
    params = make_params(f_n=0.0)
    f0 = 0.05
    # Uniaxial tension (positive hydrostatic stress) grows porosity.
    de_tension = uniaxial_strain_increment(1, 2.0e-1)
    out_tension = run_return_map(de_tension, f0, params)
    assert out_tension["f"].item() > f0
    # Pure shear (zero hydrostatic stress) leaves porosity unchanged:
    # the volumetric part of the flow normal vanishes.
    de_shear = shear_strain_increment(1, 8.0e-2)
    out_shear = run_return_map(de_shear, f0, params)
    assert out_shear["dpeeq_m"].item() > 0.0
    assert abs(out_shear["f"].item() - f0) < 1.0e-12


def test_nucleation_growth():
    # Dense matrix with nucleation only: porosity appears with
    # plastic strain and integrates the Gaussian intensity.
    params = make_params()
    de = shear_strain_increment(1, 8.0e-2)
    out = run_return_map(de, 0.0, params)
    dep = out["dpeeq_m"].item()
    assert dep > 0.0
    # One-increment integral of the strain-controlled nucleation.
    a_end = nucleation_intensity(
        out["dpeeq_m"], params["f_n"], params["eps_n"],
        params["s_n"])
    assert out["f"].item() > 0.0
    assert out["f"].item() == pytest.approx(
        (a_end * out["dpeeq_m"]).item(), rel=1.0e-6)
    # Disabling nucleation keeps the material dense under shear.
    params_off = make_params(f_n=0.0)
    out_off = run_return_map(de, 0.0, params_off)
    assert torch.all(out_off["f"] == 0.0)


def test_bisection_convergence_and_warm_start():
    params = make_params()
    de = uniaxial_strain_increment(2, 2.0e-1)
    out = run_return_map(de, 0.02, params)
    # The converged yield function is below the requested tolerance.
    assert torch.all(out["phi"].abs() <= params["tol"])
    # A warm-started repeat of the same increment converges in fewer
    # bisection iterations.
    out_warm = run_return_map(
        de, 0.02, params, dlam_warm=out["dlam"])
    assert out_warm["n_iter"] < out["n_iter"]


def test_max_iter_raises():
    # An unreachable tolerance with a tiny iteration budget must
    # raise loudly - never exit the loop silently.
    params = make_params(tol=1.0e-30, max_iter=3)
    de = uniaxial_strain_increment(1, 2.0e-1)
    with pytest.raises(RuntimeError):
        run_return_map(de, 0.02, params)


# =====================================================================
# Material-class wrapper.
# =====================================================================
def test_gtn3d_step_matches_return_map():
    # The Material-interface wrapper must reproduce the raw return
    # map: state layout [peeq_m, f, ep_warm].
    params = make_params()
    n_elem = 3
    material = GTN3D(
        1000.0, 0.3, linear_flow_stress, params).vectorize(n_elem)
    de = uniaxial_strain_increment(n_elem, 2.0e-1)
    F = torch.eye(3).expand(n_elem, 3, 3)
    sigma0 = torch.zeros(n_elem, 3, 3)
    state0 = torch.zeros(n_elem, 3)
    state0[:, 1] = 0.02
    de0 = torch.zeros(n_elem, 3, 3)
    sigma_new, state_new, ddsdde = material.step(
        de.clone(), F, sigma0, state0, de0)
    out = run_return_map(de, 0.02, params)
    assert torch.allclose(sigma_new, out["sigma"])
    assert torch.allclose(state_new[:, 0], out["dpeeq_m"])
    assert torch.allclose(state_new[:, 1], out["f"])
    assert ddsdde.shape == (n_elem, 3, 3, 3, 3)


def test_gtn3d_vectorize_twice_raises():
    # Double vectorization must raise - not print-and-return.
    params = make_params()
    material = GTN3D(1000.0, 0.3, linear_flow_stress, params)
    vectorized = material.vectorize(4)
    with pytest.raises(RuntimeError):
        vectorized.vectorize(4)


def test_gtn3d_requires_all_arguments():
    # No constructor defaults: omitting gtn_params is a TypeError.
    with pytest.raises(TypeError):
        GTN3D(1000.0, 0.3, linear_flow_stress)


def test_far_overshot_trial_stays_finite():
    # A trial state hundreds of yield stresses beyond the surface
    # (a far-overshot global Newton iterate) must stay finite: the
    # cosh/sinh arguments are clamped, so the return map either
    # converges or raises - it never produces NaN.
    params = make_params(f_n=0.0)
    sigma_y = torch.tensor([100.0])
    sigma_h = torch.tensor([5.0e5])
    fs = torch.tensor([0.05])
    phi = gtn_yield(torch.tensor([1.0e5]), sigma_h, sigma_y, fs,
                    1.5, 1.0)
    assert torch.all(torch.isfinite(phi))
    sigma = torch.zeros(1, 3, 3)
    sigma[0, 0, 0] = 6.0e5
    sigma[0, 1, 1] = sigma[0, 2, 2] = 4.5e5
    _, _, s_dev = stress_invariants(sigma)
    n = gtn_flow_normal(s_dev, sigma_h, sigma_y, fs, 1.5, 1.0)
    assert torch.all(torch.isfinite(n))
    assert torch.allclose(
        torch.linalg.norm(n, dim=(-1, -2)), torch.ones(1))
