"""Plot equivalent plastic strain field over a square mesh.

Runs elastoplastic simulations on Quad1 and Tria1 meshes,
extrapolates Gauss-point epbar to nodes, and produces
three subplots per mesh type:
  (a) Smooth nodal field (tricontourf via node_property)
  (b) Gauss-point scatter overlay
  (c) Nodal scatter overlay
"""
#
#                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# Third-party
import torch
import matplotlib.pyplot as plt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add torch-fem root to path
root_dir = str(pathlib.Path(__file__).parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from torchfem import Planar
from torchfem.materials import IsotropicPlasticityPlaneStrain
from torchfem.mesh import rect_quad
from torchfem.elements import Quad1, Tria1
#
#                                              Authorship & Credits
# =============================================================================
__author__ = 'R. Barreira'
__status__ = 'Development'
# =============================================================================
torch.set_default_dtype(torch.float64)


# =============================================================================
def extrapolate_gauss_to_nodes(
        etype, state_gp, elements, n_nod):
    """Extrapolate Gauss-point scalar to nodes.

    Parameters
    ----------
    etype : Element
        Element type instance.
    state_gp : torch.Tensor
        Shape (n_gp, n_elem).
    elements : torch.Tensor
        Shape (n_elem, n_en).
    n_nod : int
        Total number of nodes.

    Returns
    -------
    nodal_values : torch.Tensor
        Shape (n_nod,).
    """
    N_gp = etype.N(etype.ipoints())
    N_nd = etype.N(etype.npoints())
    E = N_nd @ torch.linalg.pinv(N_gp)
    nodal_elem = E @ state_gp
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_en = elements.shape[1]
    nodal_sum = torch.zeros(n_nod)
    nodal_count = torch.zeros(n_nod)
    for i_node in range(n_en):
        global_ids = elements[:, i_node].long()
        nodal_sum.index_add_(
            0, global_ids, nodal_elem[i_node])
        nodal_count.index_add_(
            0, global_ids,
            torch.ones(elements.shape[0]))
    nodal_count = torch.clamp(nodal_count, min=1.0)
    return nodal_sum / nodal_count


# =============================================================================
def compute_gauss_physical_coords(etype, nodes, elements):
    """Compute physical coordinates of Gauss points.

    Parameters
    ----------
    etype : Element
        Element type instance.
    nodes : torch.Tensor
        Shape (n_nod, 2).
    elements : torch.Tensor
        Shape (n_elem, n_en).

    Returns
    -------
    gauss_coords : torch.Tensor
        Shape (n_elem, n_gp, 2).
    """
    N_gp = etype.N(etype.ipoints())
    nodes_elem = nodes[elements.long()]
    gauss_coords = torch.einsum(
        'gi,eij->egj', N_gp, nodes_elem)
    return gauss_coords


# =============================================================================
def quad_to_tria(elements):
    """Split quad connectivity into triangles.

    Parameters
    ----------
    elements : torch.Tensor
        Shape (n_elem, 4) quad connectivity.

    Returns
    -------
    tri_elements : torch.Tensor
        Shape (2 * n_elem, 3) triangle connectivity.
    """
    t1 = elements[:, [0, 1, 2]]
    t2 = elements[:, [0, 2, 3]]
    return torch.cat([t1, t2], dim=0)


# =============================================================================
def run_and_plot(nodes, elements, etype, material,
                 Lx, Ly, n_inc, fig_name, label):
    """Run simulation and produce 3-panel epbar plot.

    Parameters
    ----------
    nodes : torch.Tensor
        Shape (n_nod, 2).
    elements : torch.Tensor
        Shape (n_elem, n_en).
    etype : Element
        Element type instance.
    material : IsotropicPlasticityPlaneStrain
        Material instance.
    Lx : float
        Domain length in x.
    Ly : float
        Domain length in y.
    n_inc : int
        Number of load increments.
    fig_name : str
        Output figure file name.
    label : str
        Element type label for titles.
    """
    domain = Planar(nodes, elements, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary conditions: indentation-like
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tol = 1e-8
    bottom = nodes[:, 1] < tol
    domain.constraints[bottom, :] = True
    top = nodes[:, 1] > Ly - tol
    top_center = (top
                  & (nodes[:, 0] > 0.3 * Lx)
                  & (nodes[:, 0] < 0.7 * Lx))
    domain.constraints[top_center, 1] = True
    domain.displacements[top_center, 1] = -0.01
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    increments = torch.linspace(0.0, 1.0, n_inc + 1)
    print(f'\n--- Solving {label} mesh ---')
    u, f, sigma, defgrad, alpha = domain.solve(
        increments=increments,
        return_intermediate=True,
        aggregate_integration_points=False,
        verbose=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract epbar at final increment
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    epbar_gp = alpha[-1, :, :, 0]
    epbar_nodes = extrapolate_gauss_to_nodes(
        etype, epbar_gp, elements, nodes.shape[0])
    epbar_elem = epbar_gp.mean(dim=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gauss-point physical coordinates
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gp_epbar = epbar_gp.T.reshape(-1)
    u_final = u[-1]
    pos = nodes + u_final
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Common colorbar range
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vmin = 0.0
    vmax = max(epbar_nodes.max().item(),
               gp_epbar.max().item(),
               epbar_elem.max().item())
    if vmax <= vmin:
        vmax = vmin + 1e-10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = 'plasma'
    # (a) Smooth nodal field
    domain.plot(
        u=u_final, node_property=epbar_nodes,
        cmap=cmap, colorbar=True,
        vmin=vmin, vmax=vmax, bcs=False,
        title=f'({label}) Nodal field (tricontourf)',
        ax=axes[0])
    # (b) Gauss-point scatter
    domain.plot(
        u=u_final, bcs=False,
        title=f'({label}) Gauss-point values',
        ax=axes[1])
    gp_def_coords = compute_gauss_physical_coords(
        etype, pos, elements)
    gp_dx = gp_def_coords[:, :, 0].reshape(-1)
    gp_dy = gp_def_coords[:, :, 1].reshape(-1)
    sc1 = axes[1].scatter(
        gp_dx, gp_dy, c=gp_epbar,
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=10, edgecolors='k', linewidths=0.2,
        zorder=5)
    plt.colorbar(sc1, ax=axes[1])
    # (c) Nodal scatter
    domain.plot(
        u=u_final, bcs=False,
        title=f'({label}) Nodal values (extrapolated)',
        ax=axes[2])
    sc2 = axes[2].scatter(
        pos[:, 0], pos[:, 1], c=epbar_nodes,
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=10, edgecolors='k', linewidths=0.2,
        zorder=5)
    plt.colorbar(sc2, ax=axes[2])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150)
    plt.close(fig)
    print(f'Saved: {fig_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Summary
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'\nepbar range (Gauss pts): '
          f'[{gp_epbar.min():.6f}, '
          f'{gp_epbar.max():.6f}]')
    print(f'epbar range (nodes):     '
          f'[{epbar_nodes.min():.6f}, '
          f'{epbar_nodes.max():.6f}]')
    print(f'epbar range (elem avg):  '
          f'[{epbar_elem.min():.6f}, '
          f'{epbar_elem.max():.6f}]')


# =============================================================================
if __name__ == '__main__':
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Material parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    E_mod = 1000.0
    nu = 0.3
    sigma_y = 10.0
    k_hard = 100.0

    def sigma_f(q):
        return sigma_y + k_hard * q

    def sigma_f_prime(q):
        return k_hard * torch.ones_like(q)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Common parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Nx, Ny = 50, 50
    Lx, Ly = 1.0, 1.0
    n_inc = 20
    nodes, quad_elements = rect_quad(Nx, Ny, Lx, Ly)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Quad1 mesh
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    material_q = IsotropicPlasticityPlaneStrain(
        E_mod, nu, sigma_f, sigma_f_prime)
    run_and_plot(
        nodes, quad_elements, Quad1(), material_q,
        Lx, Ly, n_inc,
        fig_name='epbar_field_quad1.png',
        label='Quad1')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Tria1 mesh (split quads into triangles)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tri_elements = quad_to_tria(quad_elements)
    material_t = IsotropicPlasticityPlaneStrain(
        E_mod, nu, sigma_f, sigma_f_prime)
    run_and_plot(
        nodes, tri_elements, Tria1(), material_t,
        Lx, Ly, n_inc,
        fig_name='epbar_field_tria1.png',
        label='Tria1')
