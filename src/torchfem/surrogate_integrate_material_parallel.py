from abc import ABC, abstractmethod
from typing import Literal, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

from torchfem.elements import Element, Quad1, Quad2, Hexa1, Hexa2
from torchfem.materials import Material
from torchfem.sparse import CachedSolve, sparse_solve

import torch_geometric.data as pyg_data
from graphorge.gnn_base_model.model.gnn_model import GNNEPDBaseModel
from graphorge.gnn_base_model.data.graph_data import GraphData
from graphorge.projects.material_patches.gnn_model_tools.gen_graphs_files \
    import (
    get_elem_size_dims, get_mesh_connected_nodes, 
    extract_fem_data_from_pkl)
from graphorge.projects.material_patches.gnn_model_tools.features import (
    GNNPatchFeaturesGenerator)
from graphorge.gnn_base_model.model.custom_layers import (
    compute_stiffness_matrix)


class PatchDataset(Dataset):
    """Dataset for parallel processing of material patches."""
    
    def __init__(self, patch_indices, fem_instance, model, u, n, du, 
                 is_stepwise=False, is_converged=False, hidden_states=None):
        """Initialize dataset with patch processing data.
        
        Args:
            patch_indices: List of patch indices to process
            fem_instance: FEM instance with elements, nodes, etype, n_dim
            model: Graphorge GNN model
            u: Displacement history tensor
            n: Current increment number
            du: Displacement increment
            is_stepwise: Whether to use stepwise RNN mode
            is_converged: Convergence flag for stepwise mode
            hidden_states: Hidden states dict for stepwise mode
        """
        self.patch_indices = patch_indices
        self.fem = fem_instance
        self.model = model
        self.u = u
        self.n = n
        self.du = du
        self.is_stepwise = is_stepwise
        self.is_converged = is_converged
        self.hidden_states = hidden_states
    
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        """Process a single patch and return results."""
        idx_patch = self.patch_indices[idx]
        forces, stiffness, hidden_states = self._process_patch(idx_patch)
        return {
            'idx': idx_patch,
            'forces': forces,
            'stiffness': stiffness,
            'hidden_states': hidden_states
        }
    
    def _process_patch(self, idx_patch):
        """Complete patch processing logic."""
        # Get patch identifier for stepwise mode
        patch_id = f"patch_{idx_patch}"
        
        # Get material patch nodes
        elem_nodes = self.fem.elements[idx_patch]  
        # Extract material patch nodal coordinates and displacements
        elem_coords_ref = self.fem.nodes[elem_nodes]
        # Current displacement at increment n
        # Enable gradients for stiffness matrix computation
        elem_u_current = self.u[self.n, elem_nodes, :].clone().requires_grad_(True)
        # Convert to numpy for GraphData (reference coordinates)
        node_coords_init = elem_coords_ref.detach().numpy()
        
        # Nodal coordinates
        nodes_coords_hist = (elem_coords_ref + elem_u_current).detach(
            ).numpy()[:, :, np.newaxis]
        # Nodal displacements
        nodes_disps_hist = elem_u_current.detach().numpy(
            )[:, :, np.newaxis]
        
        # Forces history (not available, set to zeros)
        nodes_int_forces_hist = np.zeros_like(nodes_disps_hist)
        
        # Extract data in format expected by Graphorge - patch dimensions
        dim = self.fem.n_dim
        if dim == 2:
            patch_dim = [1.0, 1.0]
            n_elem_per_dim = [1, 1]
        elif dim == 3:
            patch_dim = [1.0, 1.0, 1.0]
            n_elem_per_dim = [1, 1, 1]

        # Create mesh matrix for single element
        n_nod = self.fem.etype.nodes
        if isinstance(self.fem.etype, Quad1):
            mesh_nodes_matrix = np.array([[0, 1], [2, 3]])
        else:
            # For other elements, create connectivity based on nodes
            mesh_nodes_matrix = np.arange(n_nod).reshape(-1, 1)
        
        # Instantiate GNN-based material patch graph data
        gnn_patch_data = GraphData(
            n_dim=dim, nodes_coords=node_coords_init)
        
        # Set connectivity radius based on finite element size
        connect_radius = 4 * np.sqrt(np.sum([x**2 for x in 
            get_elem_size_dims(patch_dim, n_elem_per_dim, dim)]))
        
        # Get boundary node information
        # Note: for single element all nodes are boundary nodes
        bd_node_indices = list(range(n_nod))
        boundary_node_set = set(bd_node_indices)
        
        # Create mapping from original to boundary indices
        original_to_boundary_idx = {node_id: position for position, 
                                  node_id in enumerate(bd_node_indices)}
        
        # Get finite element mesh edges for all nodes
        connected_nodes_all = get_mesh_connected_nodes(
            dim, mesh_nodes_matrix)
        
        # Filter connected_nodes to only include boundary node pairs
        connected_nodes_boundary = []
        for node1, node2 in connected_nodes_all:
            if node1 in boundary_node_set and node2 in boundary_node_set:
                boundary_node1 = original_to_boundary_idx[node1]
                boundary_node2 = original_to_boundary_idx[node2]
                connected_nodes_boundary.append(
                    (boundary_node1, boundary_node2))
        
        # Organize connected nodes as tuples
        connected_nodes = tuple(connected_nodes_boundary)
        edges_indexes_mesh = GraphData.get_edges_indexes_mesh(
            connected_nodes)
        
        # Set GNN-based material patch graph edges
        gnn_patch_data.set_graph_edges_indexes(
            connect_radius=connect_radius,
            edges_indexes_mesh=edges_indexes_mesh)
        
        # Create features generator following Graphorge approach
        features_generator = GNNPatchFeaturesGenerator(
            n_dim=self.fem.n_dim,
            nodes_coords_hist=nodes_coords_hist,
            edges_indexes=gnn_patch_data.get_graph_edges_indexes(),
            nodes_disps_hist=nodes_disps_hist,
            nodes_int_forces_hist=nodes_int_forces_hist)
        
        # Build node feature matrix
        node_features_matrix = \
            features_generator.build_nodes_features_matrix(
            features=('coord_hist', 'disp_hist'), 
            n_time_steps=1)
        
        # Build edge feature matrix
        edge_features_matrix = \
            features_generator.build_edges_features_matrix(
            features=('edge_vector', 'relative_disp'), 
            n_time_steps=1)
        
        # Set GNN-based material patch graph node and edges features
        gnn_patch_data.set_node_features_matrix(node_features_matrix)
        gnn_patch_data.set_edge_features_matrix(edge_features_matrix)
        
        # Get PyG homogeneous graph data object
        pyg_graph = gnn_patch_data.get_torch_data_object()
        
        # Get input features and normalize
        node_features_in, edge_features_in, global_features_in, \
            edges_indexes = (self.model.get_input_features_from_graph(
                pyg_graph, is_normalized=True))
        
        # Model prediction: stepwise vs non-stepwise mode
        hidden_states_out = None
        # Stepwise mode: single step prediction with hidden state tracking
        if self.is_stepwise:
            print(f"DEBUG: Processing {patch_id} (idx={idx_patch})")
            # Step - forward pass for forces
            node_features_out, _, _, hidden_states_out = self.model.step(
                node_features_in=node_features_in,
                edge_features_in=edge_features_in,
                global_features_in=global_features_in,
                edges_indexes=edges_indexes,
                is_converged=self.is_converged)
            print(f'DEBUG: {patch_id} hidden_states: {hidden_states_out}')
        # Prediction without hidden state tracking (FFNNs)
        else:
            # Forward pass for forces
            node_features_out, _, _ = self.model(
                node_features_in=node_features_in,
                edge_features_in=edge_features_in,
                global_features_in=global_features_in,
                edges_indexes=edges_indexes,
                batch_vector=None)
        
        # Denormalize output to get real forces
        node_forces_real = self.model.data_scaler_transform(
            tensor=node_features_out,
            features_type='node_features_out',
            mode='denormalize')
        
        # Compute stiffness matrix
        stiffness_matrix = compute_stiffness_matrix(
            model=self.model,
            node_features_in=node_features_in,
            edge_features_in=edge_features_in,
            global_features_in=global_features_in,
            edges_indexes=edges_indexes,
            batch_vector=None,
            n_dim=self.fem.n_dim)
        
        return node_forces_real.flatten(), stiffness_matrix, hidden_states_out


def surrogate_integrate_material_dataloader(
    fem_instance, model, u: Tensor, n: int, du: Tensor,
    is_stepwise: bool = False,
    is_converged: bool = False,
    patch_ids: Tensor = None,
    hidden_states: dict = None,
    num_workers: int = 4,
) -> Tuple[Tensor, Tensor]:
    """Parallel surrogate integration using PyTorch DataLoader.
    
    Args:
        fem_instance: FEM instance with elements, nodes, etype, n_dim
        model: Pre-loaded Graphorge material patch model
        u: Displacement history tensor
        n: Current increment number
        du: Displacement increment tensor
        is_stepwise: Whether to use stepwise RNN mode
        is_converged: Convergence flag for stepwise mode
        patch_ids: Tensor of patch IDs to process
        hidden_states: Hidden states dict for stepwise mode
        num_workers: Number of parallel workers
        
    Returns:
        Tuple[Tensor, Tensor]: Element stiffness matrices and force vectors
    """
    # Update current displacement
    u[n] = u[n - 1] + du.view((-1, fem_instance.n_dim))
    
    # Initialize nodal force and stiffness
    n_nod = fem_instance.etype.nodes
    n_dof_elem = fem_instance.n_dim * n_nod
    f = torch.zeros(fem_instance.n_elem, n_dof_elem)
    k = torch.zeros((fem_instance.n_elem, n_dof_elem, n_dof_elem))
    
    # Determine which elements to process based on patch_ids
    if patch_ids is not None:
        patch_indices = patch_ids.tolist()
    else:
        patch_indices = list(range(fem_instance.n_elem))
    
    # Create dataset and dataloader
    dataset = PatchDataset(patch_indices, fem_instance, model, u, n, du, 
                          is_stepwise, is_converged, hidden_states)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, 
                           collate_fn=lambda x: x[0])  # No batching
    
    # Collect hidden states if stepwise
    hidden_states_out = {} if is_stepwise else None
    
    # Parallel execution and assembly
    for result in dataloader:
        idx_patch = result['idx']
        f[idx_patch] = result['forces']
        k[idx_patch] = result['stiffness']
        
        if is_stepwise and result['hidden_states'] is not None:
            hidden_states_out.update(result['hidden_states'])
    
    if is_stepwise:
        return k, f, hidden_states_out
    else:    
        return k, f


def surrogate_integrate_material_joblib(
    fem_instance, model, u: Tensor, n: int, du: Tensor,
    is_stepwise: bool = False,
    is_converged: bool = False,
    patch_ids: Tensor = None,
    hidden_states: dict = None,
    n_jobs: int = -1,
) -> Tuple[Tensor, Tensor]:
    """Parallel surrogate integration using joblib.
    
    Args:
        fem_instance: FEM instance with elements, nodes, etype, n_dim
        model: Pre-loaded Graphorge material patch model
        u: Displacement history tensor
        n: Current increment number
        du: Displacement increment tensor
        is_stepwise: Whether to use stepwise RNN mode
        is_converged: Convergence flag for stepwise mode
        patch_ids: Tensor of patch IDs to process
        hidden_states: Hidden states dict for stepwise mode
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        Tuple[Tensor, Tensor]: Element stiffness matrices and force vectors
    """
    # Update current displacement
    u[n] = u[n - 1] + du.view((-1, fem_instance.n_dim))
    
    # Initialize nodal force and stiffness
    n_nod = fem_instance.etype.nodes
    n_dof_elem = fem_instance.n_dim * n_nod
    f = torch.zeros(fem_instance.n_elem, n_dof_elem)
    k = torch.zeros((fem_instance.n_elem, n_dof_elem, n_dof_elem))
    
    # Determine which elements to process based on patch_ids
    if patch_ids is not None:
        patch_indices = patch_ids.tolist()
    else:
        patch_indices = list(range(fem_instance.n_elem))
    
    def process_patch(idx_patch):
        """Process a single patch and return results."""
        # Create temporary dataset instance for processing
        temp_dataset = PatchDataset([idx_patch], fem_instance, model, u, n, du,
                                   is_stepwise, is_converged, hidden_states)
        
        # Process the patch
        result = temp_dataset[0]  # Get first (and only) item
        return result
    
    # Parallel execution
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_patch)(idx) for idx in patch_indices
    )
    
    # Collect hidden states if stepwise
    hidden_states_out = {} if is_stepwise else None
    
    # Assembly phase
    for result in results:
        idx_patch = result['idx']
        f[idx_patch] = result['forces']
        k[idx_patch] = result['stiffness']
        
        if is_stepwise and result['hidden_states'] is not None:
            hidden_states_out.update(result['hidden_states'])
    
    if is_stepwise:
        return k, f, hidden_states_out
    else:    
        return k, f


# Example usage:
"""
# For DataLoader approach:
k, f = surrogate_integrate_material_dataloader(
    fem_instance=self,
    model=model,
    u=u,
    n=n,
    du=du,
    is_stepwise=is_stepwise,
    is_converged=is_converged,
    patch_ids=patch_ids,
    hidden_states=hidden_states_dict,
    num_workers=4
)

# For joblib approach:
k, f = surrogate_integrate_material_joblib(
    fem_instance=self,
    model=model,
    u=u,
    n=n,
    du=du,
    is_stepwise=is_stepwise,
    is_converged=is_converged,
    patch_ids=patch_ids,
    hidden_states=hidden_states_dict,
    n_jobs=-1
)
"""