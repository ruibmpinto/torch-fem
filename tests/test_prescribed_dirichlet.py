"""Prescribed-displacement increments through the Newton predictor.

Regression tests for the admissible predictor in ``FEM.solve``: a
prescribed-displacement increment used to be written into the unknown
before the constitutive model was evaluated, so the whole increment
landed in the element row at the constrained boundary (a strain jump
of span/element-size times the nominal step). A path-dependent
material with a strict local solver — GTN — rightly refuses that
field, killing the solve before any convergence check ran.

The tests pin down, on a bar whose boundary layer amplifies the
nominal step tenfold:

1. the raw-jump strain state refuses at the return-map level (the
   previously fatal condition, kept as the counterfactual);
2. the same increment through ``FEM.solve`` now converges, plastifies
   and produces the homogeneous field a uniform bar must have;
3. a purely elastic prescribed-displacement increment converges at
   the predictor itself (single residual evaluation);
4. ``r_norm_ref`` is refused for every non-Newton solver.
"""

import pytest
import torch

from torchfem import Solid
from torchfem.custom_materials.gtn import GTN3D, gtn_return_map
from torchfem.materials import IsotropicElasticity3D
from torchfem.mesh import cube_hexa
from torchfem.nonlinear_solvers import solve_nonlinear

# Double precision, scoped per test as in test_gtn.py.
@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def flow_stress(peeq):
    # Linear matrix hardening; yield strain is 2/1000 = 2e-3, so the
    # nominal step below is plastic while the 10x boundary jump of
    # the old scheme is far beyond the radial-return bracket.
    return 2.0 + 1.0 * peeq


def make_params():
    # Canonical test parameters, every key explicit.
    return {
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


def build_bar(material):
    # 10x2x2-element unit-length bar under eighth-symmetry uniaxial
    # tension: u_x = 0 on x = 0, u_y = 0 on y = 0, u_z = 0 on z = 0,
    # and the pulled face x = 1 carries the prescribed x-increment.
    # Ten element rows along the axis set the boundary-layer
    # amplification of the old scheme to 10x.
    nodes, elements = cube_hexa(11, 3, 3, 1.0, 0.2, 0.2)
    bar = Solid(nodes, elements, material)
    tol = 1.0e-12
    pulled = nodes[:, 0] > 1.0 - tol
    bar.constraints[nodes[:, 0] < tol, 0] = True
    bar.constraints[nodes[:, 1] < tol, 1] = True
    bar.constraints[nodes[:, 2] < tol, 2] = True
    bar.constraints[pulled, 0] = True
    return bar, pulled


def test_raw_jump_strain_refuses_at_return_map():
    # The counterfactual: the strain state the OLD scheme fed the
    # material — the whole 0.004 nominal step compressed into one
    # 0.1-long element row, laterally constrained by undeformed
    # neighbours — has no root along the frozen flow normal.
    n_elem = 4
    de = torch.zeros(n_elem, 3, 3)
    de[:, 0, 0] = 0.04
    material = IsotropicElasticity3D(1000.0, 0.3).vectorize(n_elem)
    with pytest.raises(RuntimeError, match="radial-return cap"):
        gtn_return_map(
            torch.zeros(n_elem, 3, 3), torch.zeros(n_elem),
            0.03 * torch.ones(n_elem), torch.zeros(n_elem), de,
            material.C, flow_stress, make_params())


def test_gtn_prescribed_displacement_step_converges():
    # The same increment through FEM.solve: the predictor hands the
    # material the smooth linearized field, the solve converges and
    # the bar plastifies homogeneously.
    material = GTN3D(1000.0, 0.3, flow_stress, make_params())
    bar, pulled = build_bar(material)
    bar.displacements[pulled, 0] = 0.004
    # Initial state [peeq, f, dlam_warm] with the porous start the
    # counterfactual above refuses at the raw-jump strain.
    state0 = torch.zeros(bar.n_int, bar.n_elem, 3)
    state0[..., 1] = 0.03
    u, f, sigma, defgrad, state = bar.solve(
        increments=torch.tensor([0.0, 1.0]), max_iter=50,
        rtol=1.0e-8, atol=0.0,
        aggregate_integration_points=False, aggregate_state=False,
        initial_state=state0)
    # The pulled face carries the full prescribed increment.
    assert torch.allclose(
        u[-1][pulled, 0], torch.full_like(u[-1][pulled, 0], 0.004))
    # The step is genuinely plastic and grows the porosity.
    peeq = state[-1][..., 0]
    void = state[-1][..., 1]
    assert bool((peeq > 1.0e-4).all())
    assert bool((void > 0.03).all())
    # A uniform bar in tension is homogeneous: the axial stress
    # spread over all integration points stays within 1e-6 of its
    # mean, and the mean sits at the porosity-softened flow stress
    # (below the dense yield stress of 2.0, above the elastic
    # regime).
    sigma_xx = sigma[-1][..., 0, 0]
    mean = sigma_xx.mean()
    assert 1.8 < float(mean) < 2.0
    assert float((sigma_xx - mean).abs().max() / mean) < 1.0e-6


def test_elastic_prescribed_displacement_converges_at_predictor():
    # For a linear problem the predictor IS the solution: the solve
    # must converge at the first measured residual (one entry in the
    # history) and reproduce the homogeneous uniaxial-stress state.
    material = IsotropicElasticity3D(1000.0, 0.3)
    bar, pulled = build_bar(material)
    bar.displacements[pulled, 0] = 0.001
    u, f, sigma, defgrad, state, resnorm = bar.solve(
        increments=torch.tensor([0.0, 1.0]), max_iter=50,
        rtol=1.0e-8, atol=0.0,
        aggregate_integration_points=False, aggregate_state=False,
        return_resnorm=True)
    assert len(resnorm[1]) == 1
    # Homogeneous uniaxial stress: sigma_xx = E * eps everywhere.
    sigma_xx = sigma[-1][..., 0, 0]
    assert torch.allclose(
        sigma_xx, torch.full_like(sigma_xx, 1000.0 * 0.001),
        rtol=1.0e-9)


@pytest.mark.parametrize(
    "method",
    ["damped_picard", "anderson", "broyden", "jfnk",
     "rand_subspace_newton"])
def test_r_norm_ref_rejected_for_non_newton(method):
    # The external reference is a Newton-only contract; every other
    # method refuses it instead of silently ignoring it.
    def residual_fn(u, need_jacobian=True):
        return u, None, u

    with pytest.raises(ValueError, match="newton_raphson"):
        solve_nonlinear(
            method=method, residual_jacobian_fn=residual_fn,
            u0=torch.zeros(3), r_norm_ref=1.0)
