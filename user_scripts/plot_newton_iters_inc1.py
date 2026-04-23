"""Plot per-iteration trial state for increment 1 of Newton-Raphson.

Replicates the first load increment of
``run_simulation_surrogate.py`` (elastoplastic_nlh, quad4,
10x10 mesh) and plots the trial displacement, internal
force, and residual fields at each Newton iteration. Also
draws the Dirichlet constraints and prescribed
displacements.

Output:
    torch-fem/results/diagnostic/newton_inc1/
        iter{k}_disp.png      : deformed mesh + ||u_trial||
        iter{k}_force.png     : deformed mesh + ||F_int||
        iter{k}_residual.png  : deformed mesh + ||residual||
        bcs.png               : BC markers and prescribed
                                displacement arrows
"""
#
#                                                        Modules
# =============================================================================
# Standard
import os
import sys
import pathlib
# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
# Local
torch_fem_src = str(pathlib.Path(__file__).parents[1] / 'src')
if torch_fem_src not in sys.path:
    sys.path.insert(0, torch_fem_src)

os.environ.setdefault('TORCHFEM_IMPORT_GRAPHORGE', '0')

from torchfem import Planar
from torchfem.materials import IsotropicPlasticityPlaneStrain
from torchfem.mesh import rect_quad
from torchfem.sparse import CachedSolve, sparse_solve
#
#                                             Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto ' \
             '(rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
#
torch.set_default_dtype(torch.float64)
# -----------------------------------------------------------------------------
def build_domain_and_bcs():
    """Build the same plane-strain problem as the main run.

    Returns
    -------
    domain : Planar
        Finite-element domain with material and BCs set.
    mesh_info : dict
        Dict with mesh_nx, mesh_ny for plot sizing.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Material: Swift-Voce hardening, AA2024 (plane strain)
    e_young = 70000
    nu = 0.33
    a_s = 798.56
    epsilon_0 = 0.0178
    n_sv = 0.202
    k_0 = 363.84
    q_v = 240.03
    beta = 10.533
    omega = 0.368

    def k_s(eps_pl):
        return a_s * (epsilon_0 + eps_pl)**n_sv

    def k_v(eps_pl):
        return (k_0
                + q_v * (1.0 - torch.exp(-beta * eps_pl)))

    def sigma_f(eps_pl):
        return (omega * k_s(eps_pl)
                + (1.0 - omega) * k_v(eps_pl))

    def sigma_f_prime(eps_pl):
        dks = (a_s * n_sv
               * (epsilon_0 + eps_pl)**(n_sv - 1.0))
        dkv = (q_v * beta
               * torch.exp(-beta * eps_pl))
        return (omega * dks
                + (1.0 - omega) * dkv)

    material = IsotropicPlasticityPlaneStrain(
        E=e_young, nu=nu, sigma_f=sigma_f,
        sigma_f_prime=sigma_f_prime, max_iter=50)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Mesh: 10x10 quad4 on unit square
    mesh_nx = 10
    mesh_ny = 10
    nodes, elements = rect_quad(mesh_nx + 1, mesh_ny + 1)
    domain = Planar(nodes, elements, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Simple tension BCs: bottom fixed in y, bottom-left
    # corner fixed in x+y, top prescribed uy=0.05
    tol = 1e-6
    for i, node_coord in enumerate(nodes):
        if torch.abs(node_coord[1]) < tol:
            domain.displacements[i, 1] = 0.0
            domain.constraints[i, 1] = True
            if torch.abs(node_coord[0]) < tol:
                domain.displacements[i, 0] = 0.0
                domain.constraints[i, 0] = True
        elif torch.abs(node_coord[1] - 1.0) < tol:
            domain.displacements[i, 1] = 0.05
            domain.constraints[i, 1] = True
    return domain, {'mesh_nx': mesh_nx,
                    'mesh_ny': mesh_ny}
# -----------------------------------------------------------------------------
def plot_field(domain, u_trial, node_property, title,
               outfile, cmap='viridis', vmin=None,
               vmax=None, bcs=True):
    """Plot deformed mesh colored by node_property.

    Parameters
    ----------
    domain : Planar
        Domain to plot.
    u_trial : torch.Tensor
        Displacement field of shape (n_nodes, 2).
    node_property : torch.Tensor
        Scalar at each node of shape (n_nodes,).
    title : str
        Figure title.
    outfile : str
        Path to save the PNG.
    cmap : str, default='viridis'
        Matplotlib colormap.
    vmin : {float, None}, default=None
        Lower colorbar bound.
    vmax : {float, None}, default=None
        Upper colorbar bound.
    bcs : bool, default=True
        Whether to draw BC markers.
    """
    domain.plot(
        u=u_trial,
        node_property=node_property,
        title=title,
        colorbar=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        bcs=bcs,
    )
    plt.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def plot_bcs(domain, outfile):
    """Plot undeformed mesh with Dirichlet markers and
    prescribed displacement arrows.

    Parameters
    ----------
    domain : Planar
        Domain to plot.
    outfile : str
        Path to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # Draw mesh edges
    for element in domain.elements:
        nd = domain.nodes[element[:4]]
        xs = list(nd[:, 0].numpy()) + [float(nd[0, 0])]
        ys = list(nd[:, 1].numpy()) + [float(nd[0, 1])]
        ax.plot(xs, ys, color='black', linewidth=0.8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dirichlet constraints (open triangles)
    for i, c in enumerate(domain.constraints):
        x = float(domain.nodes[i, 0])
        y = float(domain.nodes[i, 1])
        if c[0]:
            ax.plot(x - 0.01, y, '>', color='red',
                    markersize=8, zorder=10)
        if c[1]:
            ax.plot(x, y - 0.01, '^', color='red',
                    markersize=8, zorder=10)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prescribed nonzero displacements as arrows (full inc)
    for i, d in enumerate(domain.displacements):
        if not (domain.constraints[i, 0]
                or domain.constraints[i, 1]):
            continue
        if torch.norm(d) <= 0.0:
            continue
        x = float(domain.nodes[i, 0])
        y = float(domain.nodes[i, 1])
        dx = float(d[0])
        dy = float(d[1])
        ax.arrow(x, y, dx, dy,
                 width=0.003, color='blue',
                 length_includes_head=True,
                 zorder=11)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Dirichlet BCs (red) and prescribed '
                 'displacements (blue arrows, full load)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(outfile, dpi=180,
                bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def run_increment_1_with_plots(out_dir):
    """Hand-run Newton iterations of increment 1 and plot.

    Parameters
    ----------
    out_dir : str
        Output directory for PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build problem
    domain, _ = build_domain_and_bcs()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot BCs
    plot_bcs(domain, os.path.join(out_dir, 'bcs.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Allocate Newton-loop state (mirroring solve())
    increments = torch.linspace(0.0, 1.0, 51)
    N = len(increments)
    B = domain.compute_B()
    con = torch.nonzero(
        domain.constraints.ravel(),
        as_tuple=False).ravel()
    u = torch.zeros(N, domain.n_nod, domain.n_dim)
    f_hist = torch.zeros(N, domain.n_nod, domain.n_dim)
    stress = torch.zeros(
        N, domain.n_int, domain.n_elem,
        domain.n_stress, domain.n_stress)
    defgrad = torch.zeros(
        N, domain.n_int, domain.n_elem,
        domain.n_stress, domain.n_stress)
    defgrad[:, :, :, :, :] = torch.eye(
        domain.n_stress)
    state = torch.zeros(
        N, domain.n_int, domain.n_elem,
        domain.material.n_state)
    domain.K = torch.empty(0)
    du = torch.zeros_like(domain.nodes).ravel()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Increment 1 only
    n = 1
    inc = increments[n] - increments[n - 1]
    F_ext = increments[n] * domain.forces.ravel()
    DU = inc * domain.displacements.clone().ravel()
    de0 = inc * domain.ext_strain
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect snapshots for shared colorbar scaling later
    snapshots = []
    max_iter = 3
    rtol = 1e-8
    atol = 1e-6
    res_norm0 = None
    for i in range(max_iter):
        du[con] = DU[con]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Snapshot of trial displacement BEFORE
        # integration (shape (n_nodes, n_dim))
        u_trial_pre = (u[n - 1]
                       + du.view(-1, domain.n_dim)
                       ).clone()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        k, f_i = domain.integrate_material(
            u, defgrad, stress, state, n, du,
            de0, False)
        if (domain.K.numel() == 0
                or not domain.material.n_state == 0):
            domain.K = domain.assemble_stiffness(
                k, con)
        F_int = domain.assemble_force(f_i)
        residual = F_int - F_ext
        residual[con] = 0.0
        res_norm = torch.linalg.norm(residual)
        if i == 0:
            res_norm0 = res_norm
        print(f'Iter {i + 1} | residual = '
              f'{res_norm.item():.5e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        snapshots.append({
            'iter': i + 1,
            'u_trial': u_trial_pre,
            'F_int': F_int.view(-1, domain.n_dim),
            'residual': residual.view(-1, domain.n_dim),
            'res_norm': res_norm.item(),
        })
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Convergence check (matches solve())
        if (res_norm < rtol * res_norm0
                or res_norm < atol):
            break
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        du -= sparse_solve(
            domain.K, residual, B, 1e-10, None,
            None, None, CachedSolve(), (i == 0))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shared color ranges across iterations
    u_norms = [
        torch.linalg.norm(s['u_trial'], dim=1)
        for s in snapshots]
    f_norms = [
        torch.linalg.norm(s['F_int'], dim=1)
        for s in snapshots]
    r_norms = [
        torch.linalg.norm(s['residual'], dim=1)
        for s in snapshots]
    u_vmin = min(un.min().item() for un in u_norms)
    u_vmax = max(un.max().item() for un in u_norms)
    f_vmin = min(fn.min().item() for fn in f_norms)
    f_vmax = max(fn.max().item() for fn in f_norms)
    r_vmin = min(rn.min().item() for rn in r_norms)
    r_vmax = max(rn.max().item() for rn in r_norms)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot per iteration
    for s, un, fn, rn in zip(
            snapshots, u_norms, f_norms, r_norms):
        k = s['iter']
        title_u = (f'Iter {k}: '
                   f'$||\\mathbf{{u}}_{{trial}}||$ '
                   f'(res={s["res_norm"]:.3e})')
        plot_field(
            domain, s['u_trial'], un, title_u,
            os.path.join(out_dir,
                         f'iter{k}_disp.png'),
            cmap='viridis',
            vmin=u_vmin, vmax=u_vmax)

        title_f = (f'Iter {k}: '
                   f'$||\\mathbf{{F}}_{{int}}||$')
        plot_field(
            domain, s['u_trial'], fn, title_f,
            os.path.join(out_dir,
                         f'iter{k}_force.png'),
            cmap='plasma',
            vmin=f_vmin, vmax=f_vmax)

        title_r = (f'Iter {k}: '
                   f'$||\\mathbf{{r}}||$ '
                   f'(res={s["res_norm"]:.3e})')
        plot_field(
            domain, s['u_trial'], rn, title_r,
            os.path.join(out_dir,
                         f'iter{k}_residual.png'),
            cmap='magma',
            vmin=r_vmin, vmax=r_vmax)
    print(f'Plots written to {out_dir}')
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    script_dir = os.path.dirname(
        os.path.abspath(__file__))
    torch_fem_root = os.path.dirname(script_dir)
    out_dir = os.path.join(
        torch_fem_root, 'results', 'diagnostic',
        'newton_inc1')
    run_increment_1_with_plots(out_dir)
