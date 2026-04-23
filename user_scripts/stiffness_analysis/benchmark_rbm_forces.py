"""Benchmark RBM force response of elastic surrogate models.

For each model size, applies rigid body motion displacement
fields (translation x, translation y, rotation) and plots
predicted force arrows on boundary nodes. Near-zero forces
indicate correct RBM removal.

Functions
---------
rbm_translation
    Compute rigid body translation displacement field.
rbm_rotation
    Compute rigid body rotation displacement field.
run_benchmark
    Run RBM force benchmark for all model sizes.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import copy
import pathlib

# Add graphorge and torchfem to sys.path
graphorge_path = str(
    pathlib.Path(__file__).parents[3]
    / 'graphorge_material_patches' / 'src')
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)
torchfem_path = str(
    pathlib.Path(__file__).parents[2] / 'src')
if torchfem_path not in sys.path:
    sys.path.insert(0, torchfem_path)

# Third-party
import torch
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
# Import forward_graph via importlib to bypass
# torchfem __init__ (Python 3.9 union type issue)
import importlib.util
_base_spec = importlib.util.spec_from_file_location(
    'torchfem.base',
    os.path.join(torchfem_path, 'torchfem', 'base.py'),
    submodule_search_locations=[])
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules['torchfem.base'] = _base_mod
_base_spec.loader.exec_module(_base_mod)
forward_graph = _base_mod.forward_graph


#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'R. Barreira'
__status__ = 'Prototype'


# =============================================================================
#
# =============================================================================
SURROGATES_ROOT = str(
    pathlib.Path(__file__).parents[1]
    / 'matpatch_surrogates' / 'elastic')

FIGURES_DIR = str(
    pathlib.Path(__file__).parent / 'figures')

# Matplotlib defaults
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'figure.dpi': 150,
})

# Models to benchmark: {patch_resolution: [model_names]}
# Edit this dictionary to add/remove models.
MODELS_TO_TEST = {
    1: ['model', 'model_edgestiffness'],
    2: ['model'],
    3: ['model'],
    4: ['model'],
    5: ['model_reference', 'model_edgestiffness',
        'model_spdstiffness'],
    6: ['model'],
    7: ['model'],
    8: ['model', 'model_reference',
        'model_relu_no_bias', 'model_rbmdelete'],
}


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
def rbm_translation(ref_coords, tx, ty):
    """Compute rigid body translation displacement field.

    Parameters
    ----------
    ref_coords : numpy.ndarray(2d)
        Reference node coordinates, shape (n_nodes, 2).
    tx : float
        Translation in x.
    ty : float
        Translation in y.

    Returns
    -------
    disps : numpy.ndarray(2d)
        Displacement field, shape (n_nodes, 2).
    """

    disps = np.zeros_like(ref_coords)
    disps[:, 0] = tx
    disps[:, 1] = ty
    return disps


# =============================================================================
def rbm_rotation(ref_coords, theta):
    """Compute rigid body rotation displacement field.

    Linearized CCW rotation about the mesh centroid.

    Parameters
    ----------
    ref_coords : numpy.ndarray(2d)
        Reference node coordinates, shape (n_nodes, 2).
    theta : float
        Rotation angle in radians (counter-clockwise).

    Returns
    -------
    disps : numpy.ndarray(2d)
        Displacement field, shape (n_nodes, 2).
    """

    centroid = ref_coords.mean(axis=0)
    rel = ref_coords - centroid
    disps = np.zeros_like(ref_coords)
    disps[:, 0] = -theta * rel[:, 1]
    disps[:, 1] = theta * rel[:, 0]
    return disps


# =============================================================================
def get_edge_feature_type(model):
    """Derive edge_feature_type from model n_edge_in.

    Parameters
    ----------
    model : GNNEPDBaseModel
        Loaded model.

    Returns
    -------
    edge_feature_type : tuple[str]
        Edge feature names for forward_graph.
    """

    n_edge_in = getattr(model, '_n_edge_in', 2)
    n_dim = 2
    n_features = n_edge_in // n_dim
    if n_features == 1:
        return ('edge_vector',)
    elif n_features == 2:
        return ('edge_vector', 'relative_disp')
    else:
        raise RuntimeError(
            f'Unexpected n_edge_in={n_edge_in}')


# =============================================================================
def predict_forces(
        model, disps_np, coords, edges_indexes,
        edge_feature_type):
    """Run forward_graph and return forces as numpy.

    Parameters
    ----------
    model : GNNEPDBaseModel
        Loaded model in eval mode.
    disps_np : numpy.ndarray(2d)
        Displacement field, shape (n_nodes, 2).
    coords : torch.Tensor
        Reference coordinates, shape (n_nodes, 2).
    edges_indexes : torch.Tensor
        Edge connectivity, shape (2, n_edges).
    edge_feature_type : tuple[str]
        Edge feature names for forward_graph.

    Returns
    -------
    forces : numpy.ndarray(2d)
        Predicted forces, shape (n_nodes, 2).
    """

    n_dim = 2
    disps = torch.tensor(
        disps_np, dtype=torch.float64)
    with torch.no_grad():
        result = forward_graph(
            model, disps, coords, edges_indexes,
            n_dim,
            edge_feature_type=edge_feature_type)
    # forward_graph returns tuple if stepwise
    if isinstance(result, tuple):
        forces = result[0]
    else:
        forces = result
    return forces.detach().numpy()


# =============================================================================
def plot_rbm_forces(
        label, coords_np, forces_dict, out_path):
    """Plot force arrows for all RBM modes.

    Parameters
    ----------
    label : str
        Model label for figure title.
    coords_np : numpy.ndarray(2d)
        Boundary node coordinates, shape (n_nodes, 2).
    forces_dict : dict
        Keys are mode names, values are force arrays.
    out_path : str
        Output figure path.
    """

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.5))
    mode_names = list(forces_dict.keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for ax, mode_name in zip(axes, mode_names):
        forces = forces_dict[mode_name]
        magnitudes = np.linalg.norm(forces, axis=1)
        max_mag = magnitudes.max()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Mesh outline
        ax.plot(
            [0, 1, 1, 0, 0], [0, 0, 1, 1, 0],
            color='0.7', linewidth=0.8, zorder=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Boundary nodes
        ax.scatter(
            coords_np[:, 0], coords_np[:, 1],
            s=12, c='k', zorder=2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Force quiver
        if max_mag > 0:
            q = ax.quiver(
                coords_np[:, 0], coords_np[:, 1],
                forces[:, 0], forces[:, 1],
                magnitudes,
                cmap='hot_r', zorder=3,
                scale=max_mag * 15,
                width=0.005)
            plt.colorbar(q, ax=ax, shrink=0.7)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_title(
            f'{mode_name}\nmax |F| = {max_mag:.2e}')
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig.suptitle(
        f'{label} - RBM force response',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# =============================================================================
def run_benchmark():
    """Run RBM force benchmark for all models."""

    os.makedirs(FIGURES_DIR, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RBM mode definitions
    rbm_modes = {
        'Trans X': lambda c: rbm_translation(
            c, 0.5, 0.0),
        'Trans Y': lambda c: rbm_translation(
            c, 0.0, 0.3),
        'Rotation': lambda c: rbm_rotation(c, 0.05),
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('RBM Force Benchmark')
    print('=' * 60)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for nx in sorted(MODELS_TO_TEST.keys()):
        model_names = MODELS_TO_TEST[nx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build graph once per resolution
        coords, edges_indexes = build_boundary_graph(
            nx, nx)
        coords_np = coords.numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for model_name in model_names:
            model_dir = os.path.join(
                SURROGATES_ROOT,
                f'{nx}x{nx}', model_name)
            if not os.path.isdir(model_dir):
                print(
                    f'\n{nx}x{nx}/{model_name}: '
                    f'not found, skip')
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            label = f'{nx}x{nx}/{model_name}'
            print(f'\n{label}:')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Load model
            model = load_model(model_dir)
            rbm_flag = getattr(
                model,
                '_is_rigid_body_removal', False)
            edge_ft = get_edge_feature_type(model)
            print(
                f'  is_rigid_body_removal'
                f' = {rbm_flag}')
            print(
                f'  edge_feature_type'
                f' = {edge_ft}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Evaluate each RBM mode
            forces_dict = {}
            for mode_name, disp_fn in (
                    rbm_modes.items()):
                disps_np = disp_fn(coords_np)
                forces = predict_forces(
                    model, disps_np, coords,
                    edges_indexes, edge_ft)
                max_f = np.linalg.norm(
                    forces, axis=1).max()
                mean_f = np.linalg.norm(
                    forces, axis=1).mean()
                print(
                    f'  {mode_name:10s}: '
                    f'max |F| = {max_f:.4e}, '
                    f'mean |F| = {mean_f:.4e}')
                forces_dict[mode_name] = forces
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot
            fig_name = (
                f'rbm_forces_{nx}x{nx}'
                f'_{model_name}.png')
            out_path = os.path.join(
                FIGURES_DIR, fig_name)
            plot_rbm_forces(
                label, coords_np,
                forces_dict, out_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\nDone.')


# =============================================================================
if __name__ == '__main__':
    run_benchmark()
