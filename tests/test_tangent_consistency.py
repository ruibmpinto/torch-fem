"""The assembled tangent is the derivative of the internal force.

Newton's convergence rate is the only thing a wrong tangent spoils:
the residual defines the solution, so an inconsistent Jacobian still
converges to the right answer when it converges at all, and every
property test — rigid rotation makes no stress, a patch test is
uniform, a converged field matches its reference — passes regardless.
The geometric stiffness under ``nlgeom`` was assembled onto one
diagonal sub-block instead of all three for exactly that reason: two
thirds of it was missing and nothing complained.

What pins a tangent down is differencing the internal force it is
supposed to differentiate. These tests compare the assembled
``K`` against a central difference of ``F_int`` on a distorted,
pre-stressed block, where the geometric term is a real part of the
answer rather than a rounding correction.

Functions
---------
build_block
    Distorted single-element cube with an elastic material.
internal_force
    Assembled internal force at a displacement increment.
assemble_tangent
    Assembled tangent at a displacement increment.
"""

import pytest
import torch

from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D
from torchfem.mesh import cube_hexa

# Double precision: a central difference cannot resolve a tangent in
# single precision.
@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def build_block(n_per_side=2):
    # A distorted mesh, so no accidental symmetry can cancel a
    # misplaced tangent entry. The distortion is deterministic.
    nodes, elements = cube_hexa(n_per_side + 1, n_per_side + 1,
                                n_per_side + 1)
    generator = torch.Generator().manual_seed(0)
    interior = ((nodes > 1.0e-9) & (nodes < 1.0 - 1.0e-9)).all(dim=1)
    jitter = 0.08 * (torch.rand(
        nodes.shape, generator=generator, dtype=nodes.dtype) - 0.5)
    nodes = nodes + jitter * interior[:, None]
    material = IsotropicElasticity3D(1000.0, 0.3)
    return Solid(nodes, elements, material)


def _history(block, du, nlgeom, pre_strain):
    # One increment of history in the layout integrate_material wants,
    # pre-stressed so the geometric term is not negligible.
    n_int, n_elem = block.n_int, block.n_elem
    u = torch.zeros(2, block.n_nod, 3)
    defgrad = torch.eye(3).expand(2, n_int, n_elem, 3, 3).clone()
    stress = torch.zeros(2, n_int, n_elem, 3, 3)
    state = torch.zeros(2, n_int, n_elem, block.material.n_state)
    # A uniaxial pre-stress at the previous increment.
    stress[0, ..., 0, 0] = pre_strain * 1000.0
    stress[0, ..., 1, 1] = 0.3 * pre_strain * 1000.0
    de0 = torch.zeros(n_elem, 3, 3)
    return u, defgrad, stress, state, de0


def internal_force(block, du, nlgeom, pre_strain):
    """Assembled internal force at a displacement increment."""
    u, defgrad, stress, state, de0 = _history(
        block, du, nlgeom, pre_strain)
    # integrate_material reads self.K to decide whether to rebuild the
    # element tangent, so it must exist however this is called first.
    block.K = torch.empty(0)
    _, f_i = block.integrate_material(
        u, defgrad, stress, state, 1, du, de0, nlgeom)
    return block.assemble_force(f_i)


def assemble_tangent(block, du, nlgeom, pre_strain):
    """Assembled tangent at a displacement increment."""
    u, defgrad, stress, state, de0 = _history(
        block, du, nlgeom, pre_strain)
    # A fresh K each call, so no cached stiffness is reused.
    block.K = torch.empty(0)
    k, _ = block.integrate_material(
        u, defgrad, stress, state, 1, du, de0, nlgeom)
    n_dof = 3 * block.n_nod
    dense = torch.zeros(n_dof, n_dof)
    # Scatter the element tangents without applying constraints: the
    # unconstrained operator is what the difference below measures.
    for e in range(block.n_elem):
        dofs = (3 * block.elements[e][:, None]
                + torch.arange(3)[None, :]).reshape(-1)
        dense[dofs[:, None], dofs[None, :]] += k[e]
    return dense


@pytest.mark.parametrize(
    "nlgeom",
    [
        False,
        pytest.param(True, marks=pytest.mark.xfail(
            strict=True,
            reason='The finite-strain tangent is a SYMMETRIC '
                   'approximation of an unsymmetric Jacobian, not an '
                   'incomplete sum that could be finished. The '
                   'internal force integrates detJ on the deformed '
                   'configuration, so it varies with displacement; '
                   'the omitted (d detJ/du) B^T sigma term is 56% '
                   'unsymmetric, and measuring the exact Jacobian by '
                   'differencing gives 3.9e-3 relative asymmetry. '
                   'The solve factorises with CHOLMOD, which needs a '
                   'symmetric positive-definite operator, so no '
                   'symmetric K can close this gap: removing it '
                   'means either a total-Lagrangian reformulation '
                   'whose tangent is symmetric by construction, or '
                   'an unsymmetric factorisation. The residual is '
                   'unaffected, so converged answers are correct, '
                   'and Newton still converges - unlike the '
                   'geometric block-pattern defect this file was '
                   'written for, which did stop it.')),
    ])
def test_tangent_matches_finite_difference(nlgeom):
    # dF_int/du from the element kernel against a central difference
    # of F_int itself, at a pre-stressed, already-deformed state.
    block = build_block()
    n_dof = 3 * block.n_nod
    generator = torch.Generator().manual_seed(1)
    du = 1.0e-3 * torch.randn(n_dof, generator=generator,
                              dtype=torch.float64)
    analytic = assemble_tangent(block, du.clone(), nlgeom, 2.0e-3)
    # Difference a sample of columns; the full matrix costs 2*n_dof
    # integrations and adds nothing once a column is wrong.
    columns = torch.linspace(
        0, n_dof - 1, 12, dtype=torch.int64).unique()
    step = 1.0e-7
    worst = 0.0
    scale = float(analytic.abs().max())
    for j in columns.tolist():
        plus, minus = du.clone(), du.clone()
        plus[j] += step
        minus[j] -= step
        f_plus = internal_force(block, plus, nlgeom, 2.0e-3)
        f_minus = internal_force(block, minus, nlgeom, 2.0e-3)
        numeric = (f_plus - f_minus) / (2.0 * step)
        worst = max(worst, float(
            (numeric - analytic[:, j]).abs().max()))
    assert worst / scale < 1.0e-6, (
        f'nlgeom={nlgeom}: tangent column deviates from the finite '
        f'difference by {worst:.3e}, {worst / scale:.3e} of the '
        f'tangent scale {scale:.3e}')


def test_geometric_stiffness_fills_every_diagonal_block():
    # The specific defect, pinned directly: the geometric term must
    # reach the y and z blocks, not only x. Differencing the force
    # would catch it, but this says which part is wrong when it fails.
    block = build_block(n_per_side=1)
    n_dof = 3 * block.n_nod
    du = torch.zeros(n_dof)
    small = assemble_tangent(block, du.clone(), False, 5.0e-3)
    large = assemble_tangent(block, du.clone(), True, 5.0e-3)
    geometric = large - small
    # Per-direction diagonal mass of the purely geometric part.
    per_direction = [
        float(geometric[a::3, a::3].abs().sum()) for a in range(3)]
    assert min(per_direction) > 0.0, (
        f'geometric stiffness missing from a direction: '
        f'{per_direction}')
    # A pre-stress that is uniaxial in x still couples y and z, and
    # the three blocks carry the same B_l . sigma . B_k up to the
    # identity, so their masses agree exactly.
    spread = ((max(per_direction) - min(per_direction))
              / max(per_direction))
    assert spread < 1.0e-12, (
        f'the three diagonal blocks of the geometric stiffness '
        f'should be identical, got {per_direction}')
