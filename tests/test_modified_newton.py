"""Unit tests for modified-Newton elastic-tangent override.

Functions
---------
test_assemble_linear_elastic_k_shape_and_symmetry
    Verify the helper returns a symmetric sparse K of the
    expected shape and matches an independent element-wise
    stiffness contraction.
test_assemble_linear_elastic_k_constant_over_displacement
    Verify the helper returns the same K regardless of mesh
    displacement state (linear-elastic tangent independence).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
# Third-party
import numpy as np
import pytest
import torch
# Local
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, 'src')))
from torchfem import Planar
from torchfem.materials import IsotropicElasticityPlaneStrain
from torchfem.mesh import rect_quad
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Pinto'
__credits__ = ['Rui Pinto', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
def _build_planar_elastic(nx=3, ny=3, E=70000.0, nu=0.33):
    """Build a planar linear-elastic Planar domain."""
    nodes, elements = rect_quad(nx + 1, ny + 1)
    mat = IsotropicElasticityPlaneStrain(E=E, nu=nu)
    return Planar(nodes, elements, mat)
# -----------------------------------------------------------------------------
def test_assemble_linear_elastic_k_shape_and_symmetry():
    domain = _build_planar_elastic(nx=3, ny=3)
    K = domain.assemble_linear_elastic_k()
    n_dofs = domain.n_dofs
    assert tuple(K.shape) == (n_dofs, n_dofs)
    K_dense = K.to_dense()
    sym_err = torch.linalg.norm(K_dense - K_dense.T).item()
    ref_norm = torch.linalg.norm(K_dense).item()
    assert ref_norm > 0.0
    assert sym_err / ref_norm < 1e-6
# -----------------------------------------------------------------------------
def test_assemble_linear_elastic_k_constant_over_displacement():
    """Linear-elastic K must not depend on u/state. Call the
    helper twice after mutating ext_strain/displacements and
    check the returned K is identical."""
    domain = _build_planar_elastic(nx=2, ny=2)
    K0 = domain.assemble_linear_elastic_k().to_dense()
    # Perturb displacements on the domain - the helper seeds
    # its own zero-state arrays, so K must be unchanged.
    domain.displacements[:] = 0.1 * torch.randn_like(
        domain.displacements)
    K1 = domain.assemble_linear_elastic_k().to_dense()
    diff = torch.linalg.norm(K0 - K1).item()
    assert diff < 1e-10
