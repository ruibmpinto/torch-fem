import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
def plot_displacement_field(
        coords_x, coords_y, u_x, u_y, node_coords, u_nodal, filename):
    """Plot displacement field within RVE at grid points.

    Args:
        coords_x (ndarray): X-coordinates grid of shape (res_x, res_y).
        coords_y (ndarray): Y-coordinates grid of shape (res_x, res_y).
        u_x (ndarray): X-displacement field of shape (res_x, res_y).
        u_y (ndarray): Y-displacement field of shape (res_x, res_y).
        node_coords (ndarray): Nodal coordinates of shape (n_nodes, 2).
        u_nodal (ndarray): Nodal displacements of shape (n_nodes, 2).
        filename (str): Output filename for the plot.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(12, 10))

    # Plot 1: Displacement magnitude contour
    u_magnitude = np.sqrt(u_x**2 + u_y**2)
    im1 = ax1.contourf(
        coords_x, coords_y, u_magnitude, levels=20, cmap='viridis')
    ax1.set_title('Displacement Magnitude')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    # Plot element boundary and nodes
    element_x = [
        node_coords[0, 0], node_coords[1, 0], node_coords[2, 0],
        node_coords[3, 0], node_coords[0, 0]]
    element_y = [
        node_coords[0, 1], node_coords[1, 1], node_coords[2, 1],
        node_coords[3, 1], node_coords[0, 1]]
    ax1.plot(element_x, element_y, 'k-', linewidth=2,
             label='Element boundary')
    ax1.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50,
        zorder=5, label='Nodes')

    # Plot 2: X-displacement contour
    im2 = ax2.contourf(coords_x, coords_y, u_x, levels=20,
                       cmap='RdBu_r')
    ax2.set_title('X-Displacement')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    ax2.plot(element_x, element_y, 'k-', linewidth=2)
    ax2.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50, zorder=5)

    # Plot 3: Y-displacement contour
    im3 = ax3.contourf(coords_x, coords_y, u_y, levels=20,
                       cmap='RdBu_r')
    ax3.set_title('Y-Displacement')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    ax3.plot(element_x, element_y, 'k-', linewidth=2)
    ax3.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50, zorder=5)

    # Plot 4: Deformed shape with displacement vectors
    # Show every 5th point for clarity
    stride = 5
    coords_x_sub = coords_x[::stride, ::stride]
    coords_y_sub = coords_y[::stride, ::stride]
    u_x_sub = u_x[::stride, ::stride]
    u_y_sub = u_y[::stride, ::stride]

    # Scale factor for displacement visualization
    max_disp = np.max(np.sqrt(u_x**2 + u_y**2))
    scale_factor = 0.1 / max_disp if max_disp > 0 else 1

    ax4.quiver(
        coords_x_sub, coords_y_sub, u_x_sub, u_y_sub,
        scale_units='xy', scale=1/scale_factor, alpha=0.7, width=0.003)
    ax4.set_title('Displacement Vectors')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.plot(element_x, element_y, 'k-', linewidth=2)
    ax4.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50, zorder=5)

    # Add deformed nodes
    deformed_coords = node_coords + u_nodal
    ax4.scatter(
        deformed_coords[:, 0], deformed_coords[:, 1], c='blue',
        s=50, zorder=5, label='Deformed nodes')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def plot_domain_displacements(domain, u_disp, output_filename):
    """Plot displacement magnitude and components using domain.plot().

    Args:
        domain: FEM domain object with plot() method.
        u_disp (Tensor): Displacement field tensor of shape
            (num_increments, n_nodes, dim).
        output_filename (str): Base filename for output plots. Will be
            modified to create separate files for magnitude and component
            plots.
    """
    # Compute displacement magnitude at nodes
    u_magnitude = torch.sqrt(u_disp[-1, :, 0]**2 + u_disp[-1, :, 1]**2)

    # Plot displacement magnitude
    domain.plot(
        u=u_disp[-1],
        node_property=u_magnitude,
        title='Displacement Magnitude',
        colorbar=True,
        figsize=(6, 6),
        cmap='viridis'
    )
    plt.savefig(
        output_filename.replace('.pkl', '_torchfem_magnitude.png'),
        dpi=300, bbox_inches='tight')
    plt.close()

    # Plot X-displacement
    domain.plot(
        u=u_disp[-1],
        node_property=u_disp[-1, :, 0],
        title='X-Displacement',
        colorbar=True,
        figsize=(6, 6),
        cmap='RdBu_r'
    )
    plt.savefig(
        output_filename.replace('.pkl', '_torchfem_x_displacement.png'),
        dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Y-displacement
    domain.plot(
        u=u_disp[-1],
        node_property=u_disp[-1, :, 1],
        title='Y-Displacement',
        colorbar=True,
        figsize=(6, 6),
        cmap='RdBu_r'
    )
    plt.savefig(
        output_filename.replace('.pkl', '_torchfem_y_displacement.png'),
        dpi=300, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def plot_shape_functions(domain, output_filename):
    """Plot all shape functions over the element.

    Args:
        domain: FEM domain object with nodes, elements, and element
            type attributes.
        output_filename (str): Base filename for output plot. Will be
            modified to append '_shape_functions.png'.
    """
    # Get node coordinates for the single element
    element_nodes = domain.elements[0]
    node_coords = domain.nodes[element_nodes]

    # Create a higher resolution grid
    xi_fine = torch.linspace(-1, 1, 100)
    eta_fine = torch.linspace(-1, 1, 100)
    xi_mesh_fine, eta_mesh_fine = torch.meshgrid(
        xi_fine, eta_fine, indexing='ij')

    # Flatten for shape function evaluation
    xi_flat_fine = xi_mesh_fine.flatten()
    eta_flat_fine = eta_mesh_fine.flatten()
    xi_points_fine = torch.stack([xi_flat_fine, eta_flat_fine], dim=-1)

    # Evaluate shape functions at grid points
    N_fine = domain.etype.N(xi_points_fine)

    # Interpolate physical coordinates
    coords_fine = torch.einsum('ij,jk->ik', N_fine, node_coords)
    coords_fine_x = coords_fine[:, 0].reshape(100, 100)
    coords_fine_y = coords_fine[:, 1].reshape(100, 100)

    # Plot all 4 shape functions
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    shape_func_names = ['N1', 'N2', 'N3', 'N4']

    for i in range(4):
        # Reshape shape function values to grid
        N_i = N_fine[:, i].reshape(100, 100)

        im = axes[i].contourf(
            coords_fine_x.detach().cpu().numpy(),
            coords_fine_y.detach().cpu().numpy(),
            N_i.detach().cpu().numpy(),
            levels=20, cmap='viridis')

        axes[i].set_title(f'Shape Function {shape_func_names[i]}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_aspect('equal')

        plt.colorbar(im, ax=axes[i])

        # Plot element boundary
        element_x = [
            node_coords[0, 0], node_coords[1, 0],
            node_coords[2, 0], node_coords[3, 0],
            node_coords[0, 0]]
        element_y = [
            node_coords[0, 1], node_coords[1, 1],
            node_coords[2, 1], node_coords[3, 1],
            node_coords[0, 1]]
        axes[i].plot(element_x, element_y, 'k-', linewidth=2)

        # Highlight the node where shape function = 1
        axes[i].scatter(
            node_coords[i, 0], node_coords[i, 1],
            c='red', s=100, zorder=5,
            edgecolors='black', linewidth=2)

        # Add other nodes
        other_nodes = [j for j in range(4) if j != i]
        axes[i].scatter(
            node_coords[other_nodes, 0], node_coords[other_nodes, 1],
            c='white', s=50, zorder=5, edgecolors='black')

    plt.tight_layout()
    plt.savefig(
        output_filename.replace('.pkl', '_shape_functions.png'),
        dpi=300, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def _draw_wireframe(ax, nodes_np, elements_np,
                    color='lightgray', linewidth=0.5):
    """Draw element wireframe on an axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    nodes_np : numpy.ndarray(2d)
        Nodal coordinates of shape (n_nodes, 2).
    elements_np : numpy.ndarray(2d)
        Element connectivity of shape (n_elem, 4).
    color : str, default='lightgray'
        Line color.
    linewidth : float, default=0.5
        Line width.
    """

    for elem in elements_np:
        x = nodes_np[elem[[0, 1, 2, 3, 0]], 0]
        y = nodes_np[elem[[0, 1, 2, 3, 0]], 1]
        ax.plot(x, y, color=color, linewidth=linewidth)
# -----------------------------------------------------------------------------
def plot_boundary_overlay(
        nodes, elements, u_surrogate,
        patch_boundary_nodes_dict, output_path):
    """Overlay of reference and deformed boundary nodes.

    Renders element wireframe in the background, open circles
    at reference positions, filled circles at deformed
    positions colored by displacement magnitude, and arrows
    from reference to deformed configuration.

    Parameters
    ----------
    nodes : torch.Tensor
        Reference coordinates of shape (n_nodes, 2).
    elements : torch.Tensor
        Element connectivity of shape (n_elem, 4).
    u_surrogate : torch.Tensor
        Surrogate displacement of shape (n_nodes, 2).
    patch_boundary_nodes_dict : dict
        Mapping patch_id (int) to boundary node indices
        (torch.Tensor of dtype long).
    output_path : str
        File path where the figure is saved.
    """

    nodes_np = nodes.detach().cpu().numpy()
    elements_np = elements.detach().cpu().numpy()
    u_np = u_surrogate.detach().cpu().numpy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect all boundary node indices (unique)
    bd_set = set()
    for idx_tensor in patch_boundary_nodes_dict.values():
        bd_set.update(idx_tensor.tolist())
    bd_idx = np.array(sorted(bd_set))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary node coordinates and displacements
    ref_xy = nodes_np[bd_idx]
    u_bd = u_np[bd_idx]
    def_xy = ref_xy + u_bd
    u_mag = np.sqrt(u_bd[:, 0]**2 + u_bd[:, 1]**2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(8, 8))
    # Wireframe
    _draw_wireframe(ax, nodes_np, elements_np)
    # Reference positions (open circles)
    ax.scatter(
        ref_xy[:, 0], ref_xy[:, 1],
        s=30, facecolors='none', edgecolors='gray',
        linewidths=0.8, zorder=3,
        label='Reference')
    # Deformed positions (filled, colored by |u|)
    sc = ax.scatter(
        def_xy[:, 0], def_xy[:, 1],
        s=30, c=u_mag, cmap='viridis',
        edgecolors='black', linewidths=0.4,
        zorder=4, label='Deformed')
    # Arrows reference -> deformed
    ax.quiver(
        ref_xy[:, 0], ref_xy[:, 1],
        u_bd[:, 0], u_bd[:, 1],
        angles='xy', scale_units='xy', scale=1,
        color='tab:red', alpha=0.6, width=0.003,
        zorder=2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.colorbar(
        sc, ax=ax,
        label=r'$||\tilde{\mathbf{u}}||$')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def plot_boundary_panels(
        nodes, elements, u_surrogate,
        patch_boundary_nodes_dict, output_path,
        u_reference=None):
    """Side-by-side panels of boundary node displacements.

    Left: boundary nodes at reference positions colored by
    surrogate |u|. Center: boundary nodes at deformed
    positions colored by surrogate |u|. Right (if u_reference
    is provided): boundary nodes colored by
    |u_surrogate - u_reference|.

    Parameters
    ----------
    nodes : torch.Tensor
        Reference coordinates of shape (n_nodes, 2).
    elements : torch.Tensor
        Element connectivity of shape (n_elem, 4).
    u_surrogate : torch.Tensor
        Surrogate displacement of shape (n_nodes, 2).
    patch_boundary_nodes_dict : dict
        Mapping patch_id (int) to boundary node indices
        (torch.Tensor of dtype long).
    output_path : str
        File path where the figure is saved.
    u_reference : {torch.Tensor, None}, default=None
        Reference displacement of shape (n_nodes, 2).
        When provided, the error panel is included.
    """

    nodes_np = nodes.detach().cpu().numpy()
    elements_np = elements.detach().cpu().numpy()
    u_srg_np = u_surrogate.detach().cpu().numpy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect all boundary node indices (unique)
    bd_set = set()
    for idx_tensor in patch_boundary_nodes_dict.values():
        bd_set.update(idx_tensor.tolist())
    bd_idx = np.array(sorted(bd_set))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ref_xy = nodes_np[bd_idx]
    u_bd = u_srg_np[bd_idx]
    def_xy = ref_xy + u_bd
    u_mag = np.sqrt(u_bd[:, 0]**2 + u_bd[:, 1]**2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_panels = 3 if u_reference is not None else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 6))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel 1: reference positions, colored by |u_srg|
    _draw_wireframe(axes[0], nodes_np, elements_np)
    sc0 = axes[0].scatter(
        ref_xy[:, 0], ref_xy[:, 1],
        s=30, c=u_mag, cmap='viridis',
        edgecolors='black', linewidths=0.4,
        vmin=u_mag.min(), vmax=u_mag.max(),
        zorder=4)
    axes[0].set_aspect('equal')
    axes[0].set_title(
        r'Reference config — $||\mathbf{u}_{srg}||$')
    plt.colorbar(sc0, ax=axes[0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel 2: deformed positions, colored by |u_srg|
    _draw_wireframe(axes[1], nodes_np, elements_np)
    sc1 = axes[1].scatter(
        def_xy[:, 0], def_xy[:, 1],
        s=30, c=u_mag, cmap='viridis',
        edgecolors='black', linewidths=0.4,
        vmin=u_mag.min(), vmax=u_mag.max(),
        zorder=4)
    axes[1].set_aspect('equal')
    axes[1].set_title(
        r'Deformed config — $||\mathbf{u}_{srg}||$')
    plt.colorbar(sc1, ax=axes[1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel 3: error (only if reference available)
    if u_reference is not None:
        u_ref_np = u_reference.detach().cpu().numpy()
        u_err = u_srg_np[bd_idx] - u_ref_np[bd_idx]
        err_mag = np.sqrt(
            u_err[:, 0]**2 + u_err[:, 1]**2)
        _draw_wireframe(
            axes[2], nodes_np, elements_np)
        sc2 = axes[2].scatter(
            ref_xy[:, 0], ref_xy[:, 1],
            s=30, c=err_mag, cmap='Reds',
            edgecolors='black', linewidths=0.4,
            zorder=4)
        axes[2].set_aspect('equal')
        axes[2].set_title(
            r'$||\mathbf{u}_{srg}'
            r' - \mathbf{u}_{ref}||$')
        plt.colorbar(sc2, ax=axes[2])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def plot_boundary_error_overlay(
        nodes, elements, u_surrogate, u_reference,
        patch_boundary_nodes_dict, output_path):
    """Overlay of displacement error arrows at boundary nodes.

    Renders element wireframe in the background, boundary
    nodes colored by error magnitude, and arrows showing the
    displacement error vector (u_surrogate - u_reference).

    Parameters
    ----------
    nodes : torch.Tensor
        Reference coordinates of shape (n_nodes, 2).
    elements : torch.Tensor
        Element connectivity of shape (n_elem, 4).
    u_surrogate : torch.Tensor
        Surrogate displacement of shape (n_nodes, 2).
    u_reference : torch.Tensor
        Reference displacement of shape (n_nodes, 2).
    patch_boundary_nodes_dict : dict
        Mapping patch_id (int) to boundary node indices
        (torch.Tensor of dtype long).
    output_path : str
        File path where the figure is saved.
    """

    nodes_np = nodes.detach().cpu().numpy()
    elements_np = elements.detach().cpu().numpy()
    u_srg_np = u_surrogate.detach().cpu().numpy()
    u_ref_np = u_reference.detach().cpu().numpy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect all boundary node indices (unique)
    bd_set = set()
    for idx_tensor in patch_boundary_nodes_dict.values():
        bd_set.update(idx_tensor.tolist())
    bd_idx = np.array(sorted(bd_set))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ref_xy = nodes_np[bd_idx]
    u_err = u_srg_np[bd_idx] - u_ref_np[bd_idx]
    err_mag = np.sqrt(
        u_err[:, 0]**2 + u_err[:, 1]**2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_wireframe(ax, nodes_np, elements_np)
    # Boundary nodes colored by error magnitude
    sc = ax.scatter(
        ref_xy[:, 0], ref_xy[:, 1],
        s=30, c=err_mag, cmap='Reds',
        edgecolors='black', linewidths=0.4,
        zorder=4)
    # Error arrows
    ax.quiver(
        ref_xy[:, 0], ref_xy[:, 1],
        u_err[:, 0], u_err[:, 1],
        angles='xy', scale_units='xy', scale=1,
        color='tab:blue', alpha=0.6, width=0.003,
        zorder=3)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.colorbar(
        sc, ax=ax,
        label=(r'$||\tilde{\mathbf{u}}'
               r' - \mathbf{u}_{ref}||$'))
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
# =============================================================================
