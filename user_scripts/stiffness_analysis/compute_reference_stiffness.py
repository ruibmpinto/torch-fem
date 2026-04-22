"""Compute reference boundary stiffness via static condensation.

Creates an isolated NxN quad4 patch, assembles the full
unconstrained stiffness matrix, then computes the Schur
complement to obtain the condensed boundary stiffness.
This is the exact FE counterpart of the surrogate's
per-patch boundary Jacobian.

Functions
---------
compute_reference_stiffness
    Assemble and condense K for one patch resolution.
"""
#
#                                                          Modules
# =============================================================================
# Standard
import os
import pathlib
import sys

import numpy as np

# Third-party
import torch

# Local
from torchfem import Planar
from torchfem.materials import IsotropicElasticityPlaneStrain
from torchfem.mesh import rect_quad

#
#                                                 Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira'
__status__ = 'Development'


# =============================================================================
def compute_reference_stiffness(
        patch_size_x, patch_size_y,
        mesh_nx, mesh_ny,
        E, nu, output_dir):
    """Compute reference condensed boundary stiffness.

    Parameters
    ----------
    patch_size_x : int
        Number of elements in x per patch.
    patch_size_y : int
        Number of elements in y per patch.
    mesh_nx : int
        Total elements in x in the global mesh.
    mesh_ny : int
        Total elements in y in the global mesh.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    output_dir : str
        Directory where the .npy file is saved.
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Patch physical size (unit-square global domain)
    Lx = patch_size_x / mesh_nx
    Ly = patch_size_y / mesh_ny
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create isolated patch mesh
    Nx = patch_size_x + 1
    Ny = patch_size_y + 1
    nodes, elements = rect_quad(Nx, Ny, Lx=Lx, Ly=Ly)
    material = IsotropicElasticityPlaneStrain(
        E=E, nu=nu)
    domain = Planar(nodes, elements, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble unconstrained global K
    n_inc = 2
    u = torch.zeros(
        n_inc, domain.n_nod, domain.n_dim)
    F = torch.zeros(
        n_inc, domain.n_int, domain.n_elem,
        domain.n_stress, domain.n_stress)
    F[:] = torch.eye(domain.n_stress)
    stress = torch.zeros(
        n_inc, domain.n_int, domain.n_elem,
        domain.n_stress, domain.n_stress)
    state = torch.zeros(
        n_inc, domain.n_int, domain.n_elem,
        domain.material.n_state)
    du = torch.zeros(domain.n_dofs)
    de0 = torch.zeros(
        domain.n_elem,
        domain.n_stress, domain.n_stress)
    domain.K = torch.empty(0)

    k_elem, _ = domain.integrate_material(
        u, F, stress, state, 1, du, de0,
        nlgeom=False)
    con = torch.tensor([], dtype=torch.int32)
    K_full = domain.assemble_stiffness(
        k_elem, con).to_dense()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Identify boundary and interior nodes
    bd_nodes = set()
    for i in range(Nx):
        for j in range(Ny):
            if (i == 0 or i == Nx - 1
                    or j == 0 or j == Ny - 1):
                bd_nodes.add(i * Ny + j)
    bd_nodes = sorted(bd_nodes)
    int_nodes = sorted(
        set(range(Nx * Ny)) - set(bd_nodes))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Map nodes to DOFs
    bd_dofs = []
    for n in bd_nodes:
        bd_dofs.extend([2 * n, 2 * n + 1])
    int_dofs = []
    for n in int_nodes:
        int_dofs.extend([2 * n, 2 * n + 1])
    bd_dofs = torch.tensor(bd_dofs)
    int_dofs = torch.tensor(int_dofs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Static condensation (Schur complement)
    # Upcast to float64 to avoid numerical artifacts in
    # the solve (K_II can be ill-conditioned in float32).
    K_full = K_full.to(torch.float64)
    if len(int_dofs) > 0:
        K_BB = K_full[bd_dofs][:, bd_dofs]
        K_BI = K_full[bd_dofs][:, int_dofs]
        K_IB = K_full[int_dofs][:, bd_dofs]
        K_II = K_full[int_dofs][:, int_dofs]
        X = torch.linalg.solve(K_II, K_IB)
        K_cond = K_BB - K_BI @ X
    else:
        # No interior nodes (e.g. 1x1 patch)
        K_cond = K_full
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save as float32 (matches surrogate output format)
    os.makedirs(output_dir, exist_ok=True)
    patch_str = f'{patch_size_x}x{patch_size_y}'
    fname = f'stiffness_ref_{patch_str}.npy'
    fpath = os.path.join(output_dir, fname)
    np.save(fpath, K_cond.to(torch.float32).numpy())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print diagnostics
    eigs = torch.linalg.eigvalsh(K_cond)
    sym_err = torch.norm(
        K_cond - K_cond.T) / torch.norm(K_cond)
    print(f'Patch {patch_str}: '
          f'shape={K_cond.shape}, '
          f'sym_err={sym_err:.2e}, '
          f'eig_min={eigs[0]:.4e}, '
          f'eig_max={eigs[-1]:.4e}, '
          f'cond={eigs[-1] / eigs[0]:.4e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save eigenvalues as float64
    eig_fname = f'eigenvalues_ref_{patch_str}.npy'
    eig_fpath = os.path.join(output_dir, eig_fname)
    np.save(eig_fpath, eigs.numpy())
    print(f'  Saved to {fpath}')
    print(f'  Eigenvalues saved to {eig_fpath}')
    return K_cond
# =============================================================================


if __name__ == '__main__':
    # Material parameters
    E = 110000.0
    nu = 0.33
    # =========================================================
    # Configurations: (mesh_nx, mesh_ny, patch_x, patch_y,
    #                  n_time_inc)
    configs = [
        (4, 4, 1, 1, 10),
        (3, 3, 3, 3, 4),
        (4, 4, 4, 4, 4),
        (8, 8, 4, 4, 4),
        (10, 10, 5, 5, 4),
        (12, 12, 6, 6, 4),
        (14, 14, 7, 7, 4),
        (16, 16, 8, 8, 4),
        (16, 16, 2, 2, 4),
    ]
    # =========================================================
    torch_fem_root = str(
        pathlib.Path(__file__).parents[1])
    results_base = os.path.join(
        torch_fem_root, 'results')

    stiffness_dir = os.path.join(
        results_base, 'elastic', '2d', 'quad4',
        'stiffness_reference')

    for (mx, my, px, py, _n_inc) in configs:
        compute_reference_stiffness(
            patch_size_x=px,
            patch_size_y=py,
            mesh_nx=mx,
            mesh_ny=my,
            E=E, nu=nu,
            output_dir=stiffness_dir)
