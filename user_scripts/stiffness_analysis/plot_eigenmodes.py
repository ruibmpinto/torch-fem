"""Plot the 3 lowest eigenmodes of surrogate stiffness.

Computes the boundary stiffness matrix K for each trained
GNN surrogate model via jacfwd, then plots the 3 lowest
eigenmodes (rigid body modes: 2 translations + 1 rotation
in 2D).

Functions
---------
load_model
    Load a trained Graphorge model.
build_boundary_graph
    Build boundary node coords and edge connectivity.
compute_surrogate_stiffness
    Compute boundary K at zero displacement via jacfwd.
plot_eigenmodes
    Plot the 3 lowest eigenmodes as deformed boundary.
"""
#
#                                                          Modules
# =============================================================================
# Standard
import os
import sys
import copy
import pathlib

# Add graphorge to sys.path
graphorge_path = str(
    pathlib.Path(__file__).parents[3]
    / 'graphorge_material_patches' / 'src')
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)

# Third-party
import torch
import torch.func as torch_func
import matplotlib.pyplot as plt
import numpy as np

# Enable graphorge imports in torchfem
os.environ['TORCHFEM_IMPORT_GRAPHORGE'] = '1'

# Local
from graphorge.gnn_base_model.model.gnn_model \
    import GNNEPDBaseModel
from graphorge.gnn_base_model.data.graph_data \
    import GraphData
from graphorge.projects.material_patches \
    .gnn_model_tools.gen_graphs_files \
    import (get_elem_size_dims,
            get_mesh_connected_nodes)
from torchfem.base import (
    forward_graph, compute_edge_features)

# Matplotlib defaults
plt.rcParams.update({
    'text.usetex': False,
    'font.size': 12,
    'axes.titlesize': 16,
    'figure.dpi': 360,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.figsize': (6, 6),
    'lines.linewidth': 1.5,
})
#
#                                                 Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto']
__status__ = 'Development'
# =============================================================================
#
torch.set_default_dtype(torch.float64)


# =============================================================================
def load_model(model_directory, device_type='cpu'):
    """Load a trained Graphorge model.

    Parameters
    ----------
    model_directory : str
        Path to the trained model directory.
    device_type : str, default='cpu'
        Device for inference.

    Returns
    -------
    model : GNNEPDBaseModel
        Loaded model in eval mode.
    """

    model = GNNEPDBaseModel.init_model_from_file(
        model_directory)
    model.set_device(device_type)
    model.load_model_state(
        load_model_state='best',
        is_remove_posterior=False)
    model.eval()
    return model


# =============================================================================
def build_boundary_graph(
        mesh_nx, mesh_ny, edge_type='all'):
    """Build boundary node coords and edges for a patch.

    Parameters
    ----------
    mesh_nx : int
        Elements in x direction.
    mesh_ny : int
        Elements in y direction.
    edge_type : str, default='all'
        Edge connectivity type ('all' or 'bd').

    Returns
    -------
    coords : torch.Tensor
        Boundary node coordinates, shape (n_bd, 2).
    edges_indexes : torch.Tensor
        Edge connectivity, shape (2, n_edges).
    """

    n_dim = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build full grid node matrix
    mesh_nodes_matrix = np.zeros(
        (mesh_nx + 1, mesh_ny + 1), dtype=int)
    all_coords = np.zeros(
        ((mesh_nx + 1) * (mesh_ny + 1), n_dim))
    node_idx = 0
    for i in range(mesh_nx + 1):
        for j in range(mesh_ny + 1):
            mesh_nodes_matrix[i, j] = node_idx
            all_coords[node_idx, 0] = (
                i / mesh_nx)
            all_coords[node_idx, 1] = (
                j / mesh_ny)
            node_idx += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Identify perimeter nodes
    bd_node_indices = []
    for i in range(mesh_nx + 1):
        for j in range(mesh_ny + 1):
            if (i == 0 or i == mesh_nx
                    or j == 0 or j == mesh_ny):
                bd_node_indices.append(
                    mesh_nodes_matrix[i, j])
    bd_node_indices = sorted(bd_node_indices)
    boundary_node_set = set(bd_node_indices)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary coordinates
    bd_coords = all_coords[bd_node_indices]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Map original to sequential boundary index
    orig_to_bd = {
        nid: pos for pos, nid
        in enumerate(bd_node_indices)}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Edge connectivity
    patch_dim = [1.0, 1.0]
    n_elem_per_dim = [mesh_nx, mesh_ny]
    if edge_type == 'all':
        radius_mult = 20.0
    elif edge_type == 'bd':
        radius_mult = 0.8
    else:
        raise RuntimeError(
            f'Unknown edge_type: {edge_type}')
    connect_radius = radius_mult * np.sqrt(
        np.sum([x**2 for x in get_elem_size_dims(
            patch_dim, n_elem_per_dim, n_dim)]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get all mesh edges, filter to boundary
    connected_all = get_mesh_connected_nodes(
        n_dim, mesh_nodes_matrix)
    connected_bd = []
    for n1, n2 in connected_all:
        if (n1 in boundary_node_set
                and n2 in boundary_node_set):
            connected_bd.append(
                (orig_to_bd[n1], orig_to_bd[n2]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build edges via GraphData
    gnn_data = GraphData(
        n_dim=n_dim, nodes_coords=bd_coords)
    edges_mesh = GraphData.get_edges_indexes_mesh(
        tuple(connected_bd))
    gnn_data.set_graph_edges_indexes(
        connect_radius=connect_radius,
        edges_indexes_mesh=edges_mesh)
    edges_indexes = torch.tensor(
        np.transpose(copy.deepcopy(
            gnn_data.get_graph_edges_indexes())),
        dtype=torch.long)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    coords = torch.tensor(
        bd_coords, dtype=torch.float64)
    return coords, edges_indexes


# =============================================================================
def compute_surrogate_stiffness(
        model, mesh_nx, mesh_ny,
        edge_type='all',
        edge_feature_type=None):
    """Compute boundary stiffness at zero displacement.

    Parameters
    ----------
    model : GNNEPDBaseModel
        Loaded surrogate model.
    mesh_nx : int
        Patch elements in x.
    mesh_ny : int
        Patch elements in y.
    edge_type : str, default='all'
        Edge type for graph construction.
    edge_feature_type : tuple
        Edge feature types.

    Returns
    -------
    K : torch.Tensor
        Boundary stiffness, shape (n_dof, n_dof).
    coords : torch.Tensor
        Boundary node coords, shape (n_bd, 2).
    """

    n_dim = 2
    coords, edges_indexes = build_boundary_graph(
        mesh_nx, mesh_ny, edge_type=edge_type)
    n_bd = coords.shape[0]
    n_dof = n_bd * n_dim
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Auto-detect edge_feature_type from model
    if edge_feature_type is None:
        n_edge_in = model._n_edge_in
        if n_edge_in == n_dim:
            edge_feature_type = ('edge_vector',)
        else:
            edge_feature_type = (
                'edge_vector', 'rel_disp')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Zero displacement (undeformed reference)
    disp_zero = torch.zeros_like(coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Patch physical size (unit square)
    L = torch.tensor([1.0, 1.0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(disp_boundary):
        """Forward closure for jacfwd."""

        disp_mean = disp_boundary.mean(
            dim=0, keepdim=True)
        disp_centered = disp_boundary - disp_mean
        disp_scaled = disp_centered / L
        pred_scaled = forward_graph(
            model=model,
            disps=disp_scaled,
            coords_ref=coords,
            edges_indexes=edges_indexes,
            n_dim=n_dim,
            edge_feature_type=edge_feature_type)
        pred_real = pred_scaled * L
        return pred_real, pred_real.detach()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Jacobian = stiffness matrix
    jacobian, _ = torch_func.jacfwd(
        forward, has_aux=True)(disp_zero)
    K = jacobian.view(n_dof, n_dof)
    return K, coords


# =============================================================================
def _perimeter_order(coords_np):
    """Order boundary nodes along the perimeter.

    Traces bottom -> right -> top -> left edges of
    the unit square boundary to produce a closed
    polygon ordering.

    Parameters
    ----------
    coords_np : numpy.ndarray(2d)
        Boundary node coordinates, shape (n_nodes, 2).

    Returns
    -------
    order : list[int]
        Node indices in perimeter order (closed).
    """

    tol = 1e-10
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Bottom edge: y == y_min, sort by x ascending
    bottom = [i for i in range(len(x))
              if abs(y[i] - y_min) < tol]
    bottom.sort(key=lambda i: x[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Right edge: x == x_max, sort by y ascending
    # Exclude corners already in bottom/top
    right = [i for i in range(len(x))
             if (abs(x[i] - x_max) < tol
                 and abs(y[i] - y_min) > tol
                 and abs(y[i] - y_max) > tol)]
    right.sort(key=lambda i: y[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Top edge: y == y_max, sort by x descending
    top = [i for i in range(len(x))
           if abs(y[i] - y_max) < tol]
    top.sort(key=lambda i: -x[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Left edge: x == x_min, sort by y descending
    # Exclude corners already in bottom/top
    left = [i for i in range(len(x))
            if (abs(x[i] - x_min) < tol
                and abs(y[i] - y_min) > tol
                and abs(y[i] - y_max) > tol)]
    left.sort(key=lambda i: -y[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Concatenate edges and close the polygon
    order = bottom + right + top + left
    order.append(order[0])
    return order


# =============================================================================
def plot_eigenmodes(
        K, coords, n_modes=3, scale=0.15,
        title='', output_path=None):
    """Plot the lowest eigenmodes as deformed boundary.

    Parameters
    ----------
    K : torch.Tensor
        Boundary stiffness (n_dof, n_dof).
    coords : torch.Tensor
        Boundary node reference coords (n_nodes, 2).
    n_modes : int, default=3
        Number of modes to plot.
    scale : float, default=0.15
        Displacement amplification factor.
    title : str, default=''
        Figure suptitle.
    output_path : {str, None}, default=None
        Path to save figure. None = show only.
    """

    n_dim = 2
    n_nodes = coords.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Eigendecomposition (raw K, lower triangle)
    eigs, vecs = torch.linalg.eigh(K)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Take lowest n_modes
    eigs_low = eigs[:n_modes]
    vecs_low = vecs[:, :n_modes]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    coords_np = coords.detach().numpy()
    perim = _perimeter_order(coords_np)
    fig, axes = plt.subplots(
        1, n_modes, figsize=(5 * n_modes, 5))
    if n_modes == 1:
        axes = [axes]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for m in range(n_modes):
        ax = axes[m]
        mode = vecs_low[:, m].detach().numpy()
        mode_2d = mode.reshape(n_nodes, n_dim)
        # Normalise mode to unit max displacement
        max_disp = np.max(np.abs(mode_2d))
        if max_disp > 1e-15:
            mode_2d = mode_2d / max_disp
        # Deformed positions
        x_def = coords_np + scale * mode_2d
        disp_mag = np.linalg.norm(
            mode_2d, axis=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reference boundary outline (dashed)
        ax.plot(
            coords_np[perim, 0],
            coords_np[perim, 1],
            '--', color='grey', linewidth=1.0,
            zorder=0)
        # Deformed boundary outline (solid)
        ax.plot(
            x_def[perim, 0],
            x_def[perim, 1],
            '-', color='black', linewidth=1.2,
            zorder=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reference nodes
        ax.plot(
            coords_np[:, 0], coords_np[:, 1],
            'o', color='lightgrey', markersize=4,
            zorder=2)
        # Deformed nodes with displacement color
        sc = ax.scatter(
            x_def[:, 0], x_def[:, 1],
            c=disp_mag, cmap='coolwarm',
            s=30, zorder=3, edgecolors='k',
            linewidths=0.3)
        # Displacement arrows
        for i in range(n_nodes):
            ax.annotate(
                '',
                xy=(x_def[i, 0], x_def[i, 1]),
                xytext=(coords_np[i, 0],
                        coords_np[i, 1]),
                arrowprops=dict(
                    arrowstyle='->',
                    color='grey', lw=0.8))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_aspect('equal')
        ax.set_title(
            f'Mode {m}, '
            r'$\lambda$'
            f' = {eigs_low[m].item():.2e}',
            fontsize=12)
        ax.set_xlim(
            coords_np[:, 0].min() - 0.25,
            coords_np[:, 0].max() + 0.25)
        ax.set_ylim(
            coords_np[:, 1].min() - 0.25,
            coords_np[:, 1].max() + 0.25)
        plt.colorbar(sc, ax=ax, shrink=0.6)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(
            output_path, bbox_inches='tight')
        print(f'  Saved: {output_path}')
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    script_dir = os.path.dirname(
        os.path.abspath(__file__))
    surrogates_dir = os.path.join(
        os.path.dirname(script_dir),
        'matpatch_surrogates', 'elastic')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Discover all patch resolutions with a model dir
    resolutions = []
    for entry in sorted(os.listdir(surrogates_dir)):
        parts = entry.split('x')
        if len(parts) != 2:
            continue
        nx, ny = int(parts[0]), int(parts[1])
        entry_path = os.path.join(
            surrogates_dir, entry)
        # Standard layout: NxN/model/
        model_dir = os.path.join(
            entry_path, 'model')
        if os.path.isdir(model_dir):
            resolutions.append(
                (nx, ny, model_dir, None))
        # Variant layout: NxN/model_<variant>/
        for sub in sorted(os.listdir(entry_path)):
            if (sub.startswith('model_')
                    and os.path.isdir(
                        os.path.join(
                            entry_path, sub))):
                sub_path = os.path.join(
                    entry_path, sub)
                variant = sub[len('model_'):]
                resolutions.append(
                    (nx, ny, sub_path, variant))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    labels = []
    for r in resolutions:
        lbl = f'{r[0]}x{r[1]}'
        if r[3] is not None:
            lbl += f' ({r[3]})'
        labels.append(lbl)
    print(f'Found {len(resolutions)} models: '
          f'{labels}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    edge_type = 'all'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for nx, ny, model_dir, variant in resolutions:
        patch_str = f'{nx}x{ny}'
        variant_str = (f' ({variant})'
                       if variant else '')
        print(f'\nProcessing patch '
              f'{patch_str}{variant_str}...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load model
        model = load_model(model_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute stiffness (auto-detect edge features)
        K, coords = compute_surrogate_stiffness(
            model, nx, ny,
            edge_type=edge_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Print eigenvalue summary
        eigs = torch.linalg.eigvalsh(K)
        print(f'  K shape: {K.shape}')
        print(f'  3 lowest eigenvalues: '
              f'{eigs[:3].tolist()}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot
        fname = f'eigenmodes_{patch_str}'
        if variant:
            fname += f'_{variant}'
        fname += '.png'
        output_path = os.path.join(
            model_dir, fname)
        title_str = f'Patch {patch_str}'
        if variant:
            title_str += f' ({variant})'
        title_str += ' - 3 Lowest Eigenmodes'
        plot_eigenmodes(
            K, coords, n_modes=3, scale=0.15,
            title=title_str,
            output_path=output_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\nDone.')
