from abc import ABC, abstractmethod
from typing import Literal, Tuple
import numpy as np
import torch
from torch import Tensor
import copy
import cProfile
import pstats

from .elements import Element, Quad1, Quad2, Hexa1, Hexa2
from .materials import Material
from .sparse import CachedSolve, sparse_solve

import os
import functools

is_import_graphorge = (
    os.environ.get('TORCHFEM_IMPORT_GRAPHORGE', '0') == '1')

if is_import_graphorge:
    import torch_geometric.data as pyg_data
    from graphorge.gnn_base_model.model.gnn_model \
        import GNNEPDBaseModel
    from graphorge.gnn_base_model.data.graph_data \
        import GraphData
    from graphorge.projects.material_patches \
        .gnn_model_tools.gen_graphs_files \
        import (get_elem_size_dims,
                get_mesh_connected_nodes)
    from graphorge.projects.material_patches \
        .gnn_model_tools.features import (
            GNNPatchFeaturesGenerator)
    from graphorge.gnn_base_model.model.custom_layers \
        import (compute_stiffness_matrix,
                extract_forces,
                extract_displacements,
                compute_edge_features,
                reconstruct_graph_with_displacements,
                remove_rigid_body_motion)
    import torch.func as torch_func


class FEM(ABC):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a general finite element problem.

        Args:
            nodes (Tensor): Nodal coordinates of shape (n_nodes, n_dim).
            elements (Tensor): Element connectivity of shape
                (n_elements, n_nodes_per_element).
            material (Material): Material model instance.

        Note:
            Automatically vectorizes non-vectorized materials for efficient
            computation across all elements.
        """
        # Store nodes and elements
        self.nodes = nodes
        self.elements = elements
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute problem size
        self.n_dofs = torch.numel(self.nodes)
        self.n_nod = nodes.shape[0]
        self.n_dim = nodes.shape[1]
        self.n_elem = len(self.elements)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize load variables
        self._forces = torch.zeros_like(nodes)
        self._displacements = torch.zeros_like(nodes)
        self._constraints = torch.zeros_like(nodes, dtype=torch.bool)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute mapping from local to global indices
        idx = (self.n_dim * self.elements).unsqueeze(-1) + \
            torch.arange(self.n_dim)
        self.idx = idx.reshape(self.n_elem, -1).to(torch.int32)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorize material
        if material.is_vectorized:
            self.material = material
        else:
            self.material = material.vectorize(self.n_elem)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize types
        self.n_stress: int
        self.n_int: int
        self.ext_strain: Tensor
        self.etype: Element
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cached solve for sparse linear systems
        self.cached_solve = CachedSolve()
    # -------------------------------------------------------------------------
    @property
    def forces(self) -> Tensor:
        """Get nodal forces.

        Returns:
            Tensor: Nodal forces of shape (n_nodes, n_dim).
        """
        return self._forces
    # -------------------------------------------------------------------------
    @forces.setter
    def forces(self, value: Tensor):
        """Set nodal forces with validation.

        Args:
            value (Tensor): Nodal forces tensor of shape (n_nodes, n_dim).
                Must be floating-point type.

        Raises:
            ValueError: If shape doesn't match nodes.
            TypeError: If not floating-point type.
        """
        if not value.shape == self.nodes.shape:
            raise ValueError("Forces must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Forces must be a floating-point tensor.")
        self._forces = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @property
    def displacements(self) -> Tensor:
        """Get nodal displacements.

        Returns:
            Tensor: Nodal displacements of shape (n_nodes, n_dim).
        """
        return self._displacements
    # -------------------------------------------------------------------------
    @displacements.setter
    def displacements(self, value: Tensor):
        """Set nodal displacements with validation.

        Args:
            value (Tensor): Nodal displacements tensor of shape
                (n_nodes, n_dim). Must be floating-point type.

        Raises:
            ValueError: If shape doesn't match nodes.
            TypeError: If not floating-point type.
        """
        if not value.shape == self.nodes.shape:
            raise ValueError("Displacements must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Displacements must be a floating-point tensor.")
        self._displacements = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @property
    def constraints(self) -> Tensor:
        """Get boundary condition constraints.

        Returns:
            Tensor: Boolean tensor of shape (n_nodes, n_dim) where True
                indicates constrained degrees of freedom.
        """
        return self._constraints
    # -------------------------------------------------------------------------
    @constraints.setter
    def constraints(self, value: Tensor):
        """Set boundary condition constraints with validation.

        Args:
            value (Tensor): Boolean tensor of shape (n_nodes, n_dim) where
                True indicates constrained degrees of freedom.

        Raises:
            ValueError: If shape doesn't match nodes.
            TypeError: If not boolean type.
        """
        if not value.shape == self.nodes.shape:
            raise ValueError("Constraints must have the same shape as nodes.")
        if value.dtype != torch.bool:
            raise TypeError("Constraints must be a boolean tensor.")
        self._constraints = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @abstractmethod
    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate element shape functions and derivatives.

        Args:
            xi (Tensor): Natural coordinates for evaluation points.
            u (Tensor | float): Displacement field. Defaults to 0.0.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Shape functions, derivatives,
                and Jacobian determinant.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        """Compute element stiffness matrix.

        Args:
            detJ (Tensor): Jacobian determinant at integration points.
            BCB (Tensor): B^T * C * B matrix product.

        Returns:
            Tensor: Element stiffness matrix.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        """Compute element internal force vector.

        Args:
            detJ (Tensor): Jacobian determinant at integration points.
            B (Tensor): Strain-displacement matrix.
            S (Tensor): Stress tensor.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def plot(self, u: float | Tensor = 0.0, **kwargs):
        """Plot finite element mesh and results.

        Args:
            u (float | Tensor): Displacement field for deformed
                configuration. Defaults to 0.0.
            **kwargs: Additional plotting parameters.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    def compute_B(self) -> Tensor:
        """
        Compute null space matrix representing rigid body modes.
        """
        if self.n_dim == 3:
            B = torch.zeros((self.n_dofs, 6))
            B[0::3, 0] = 1
            B[1::3, 1] = 1
            B[2::3, 2] = 1
            B[1::3, 3] = -self.nodes[:, 2]
            B[2::3, 3] = self.nodes[:, 1]
            B[0::3, 4] = self.nodes[:, 2]
            B[2::3, 4] = -self.nodes[:, 0]
            B[0::3, 5] = -self.nodes[:, 1]
            B[1::3, 5] = self.nodes[:, 0]
        else:
            B = torch.zeros((self.n_dofs, 3))
            B[0::2, 0] = 1
            B[1::2, 1] = 1
            B[1::2, 2] = -self.nodes[:, 0]
            B[0::2, 2] = self.nodes[:, 1]
        return B
    # -------------------------------------------------------------------------
    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain.
        
        Returns:
            Tensor: Element stiffness matrices of shape (n_elem, n_dof_elem, 
                n_dof_elem).
        """
        u = torch.zeros_like(self.nodes)
        F = torch.zeros(2, self.n_int, self.n_elem, self.n_stress, 
                        self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        F[:, :, :, :, :] = torch.eye(self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        s = torch.zeros(2, self.n_int, self.n_elem, self.n_stress, 
                        self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        a = torch.zeros(2, self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros_like(self.nodes)
        de0 = torch.zeros(self.n_elem, self.n_stress, self.n_stress)
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        k, _ = self.integrate_material(u, F, s, a, 1, du, de0, False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return k
    # -------------------------------------------------------------------------
    def integrate_material(
        self,
        u: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        n: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix.
        
        Args:
            u (Tensor): Displacement history of shape (n_increments, n_nodes, 
                n_dim).
            F (Tensor): Deformation gradient history of shape (n_increments, 
                n_int, n_elem, n_stress, n_stress).
            stress (Tensor): Stress history of shape (n_increments, n_int, 
                n_elem, n_stress, n_stress).
            state (Tensor): Material state variable history of shape 
                (n_increments, n_int, n_elem, n_state).
            n (int): Current increment number.
            du (Tensor): Displacement increment vector of shape (n_dofs,).
            de0 (Tensor): External strain increment of shape (n_elem, 
                n_stress, n_stress).
            nlgeom (bool): Whether to use nonlinear geometry.
            
        Returns:
            Tuple[Tensor, Tensor]: Element stiffness matrices of shape 
                (n_elem, n_dof_elem, n_dof_elem) and internal force vectors
                of shape (n_elem, n_dof_elem).
        """
        # Compute updated configuration
        u_trial = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape displacement increment
        du = du.view(-1, self.n_dim)[self.elements].reshape(
            self.n_elem, -1, self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize nodal force and stiffness
        n_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dim * n_nod)
        k = torch.zeros((self.n_elem, self.n_dim * n_nod, self.n_dim * n_nod))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i, (w, xi) in enumerate(zip(self.etype.iweights(), 
                                        self.etype.ipoints())):
            # Compute gradient operators
            _, B0, detJ0 = self.eval_shape_functions(xi)
            if nlgeom:
                # Compute updated gradient operators in deformed configuration
                _, B, detJ = self.eval_shape_functions(xi, u_trial)
            else:
                # Use initial gradient operators
                B = B0
                detJ = detJ0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute displacement gradient increment
            H_inc = torch.einsum("...ij,...jk->...ik", B0, du)
            # Update deformation gradient
            F[n, i] = F[n - 1, i] + H_inc
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Evaluate material response
            stress[n, i], state[n, i], ddsdde = self.material.step(
                H_inc, F[n - 1, i], stress[n - 1, i], state[n - 1, i], de0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element internal forces
            force_contrib = self.compute_f(detJ, B, stress[n, i].clone())
            f += w * force_contrib.reshape(-1, self.n_dim * n_nod)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                # Material stiffness
                BCB = torch.einsum(
                    "...ijpq,...qk,...il->...ljkp", ddsdde, B, B)
                BCB = BCB.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, BCB)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if nlgeom:
                # Geometric stiffness
                BSB = torch.einsum(
                    "...iq,...qk,...il->...lk", stress[n, i].clone(), B, B
                )
                zeros = torch.zeros_like(BSB)
                kg = torch.stack([BSB] + (self.n_dim - 1) * [zeros], dim=-1)
                kg = kg.reshape(-1, n_nod, self.n_dim * n_nod).unsqueeze(-2)
                zeros = torch.zeros_like(kg)
                kg = torch.stack([kg] + (self.n_dim - 1) * [zeros], dim=-2)
                kg = kg.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, kg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return k, f
    # -----------------------------------------------------------------
    def _integrate_fe_subset(
            self, fe_indices, u, defgrad, stress,
            state, n, du, de0, nlgeom):
        """Integrate a subset of elements using FE.

        Temporarily swaps mesh attributes to the FE
        subset, calls integrate_material + assembly,
        then restores originals.

        Args:
            fe_indices (Tensor): Element indices for
                standard FE integration.
            u (Tensor): Displacement history.
            defgrad (Tensor): Deformation gradient.
            stress (Tensor): Stress history.
            state (Tensor): State variable history.
            n (int): Current increment number.
            du (Tensor): Displacement increment.
            de0 (Tensor): External strain increment.
            nlgeom (bool): Nonlinear geometry flag.

        Returns:
            Tuple[Tensor, Tensor]: Raw global sparse
                stiffness (no constraints applied) and
                global force vector.
        """
        # Subset state arrays (axis 2 = elements)
        defgrad_fe = defgrad[:, :, fe_indices].clone()
        stress_fe = stress[:, :, fe_indices].clone()
        state_fe = state[
            :, :, fe_indices].clone()
        de0_fe = de0[fe_indices]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save original attributes
        orig_elements = self.elements
        orig_idx = self.idx
        orig_n_elem = self.n_elem
        orig_material = self.material
        orig_ext_strain = self.ext_strain
        orig_K = self.K
        orig_thickness = getattr(
            self, 'thickness', None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Swap to FE subset
        self.elements = orig_elements[fe_indices]
        n_dim = self.n_dim
        idx = (n_dim * self.elements).unsqueeze(
            -1) + torch.arange(n_dim)
        self.idx = idx.reshape(
            len(fe_indices), -1).to(torch.int32)
        self.n_elem = len(fe_indices)
        if orig_material.is_vectorized:
            self.material = copy.copy(
                orig_material)
            self.material.C = (
                orig_material.C[fe_indices])
        self.ext_strain = orig_ext_strain[
            fe_indices]
        if orig_thickness is not None:
            self.thickness = orig_thickness[
                fe_indices]
        # Force stiffness recomputation
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Integrate
        k_fe, f_fe = self.integrate_material(
            u, defgrad_fe, stress_fe, state_fe,
            n, du, de0_fe, nlgeom)
        # Assemble raw K (no constraint elimination)
        con_empty = torch.tensor(
            [], dtype=torch.int32)
        K_fe = self.assemble_stiffness(
            k_fe, con_empty)
        F_fe = self.assemble_force(f_fe)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Restore original attributes
        self.elements = orig_elements
        self.idx = orig_idx
        self.n_elem = orig_n_elem
        self.material = orig_material
        self.ext_strain = orig_ext_strain
        self.K = orig_K
        if orig_thickness is not None:
            self.thickness = orig_thickness
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write back updated state arrays
        defgrad[:, :, fe_indices] = defgrad_fe
        stress[:, :, fe_indices] = stress_fe
        state[:, :, fe_indices] = state_fe
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return K_fe, F_fe
    # -----------------------------------------------------------------
    def _load_Graphorge_model(self, model_directory: str,
                              device_type: str = 'cpu'):
        """Load and configure Graphorge material patch model.
        
        Args:
            model_directory (str): Path to the directory containing the trained 
                Graphorge model files.
            device_type (str, optional): Device type for model execution 
                ('cpu' or 'cuda'). Defaults to 'cpu'.
                
        Returns:
            GNNEPDBaseModel: Loaded and configured Graphorge model ready for 
                inference with material patch predictions.
                
        Example:
            >>> model = fem_instance._load_Graphorge_model(
            ...     model_directory='/path/to/trained/model',
            ...     device_type='cpu')
        """     
        # Initialize model from directory
        model = GNNEPDBaseModel.init_model_from_file(model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device
        model.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load best model state
        _ = model.load_model_state(
            load_model_state='best', is_remove_posterior=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set to evaluation mode
        model.eval()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Deactivate equilibrium layers (incompatible with
        # jacfwd dual numbers in pinv)
        # model._is_force_moment_equilibrium = False
        # model._is_force_equilibrium = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model
    # -------------------------------------------------------------------------  
    def integrate_field(self, field: Tensor | None = None) -> Tensor:
        """Integrate scalar field over elements.
        
        Args:
            field (Tensor, optional): Scalar field values at nodes of shape 
                (n_nodes,). If None, integrates unity to compute volumes. 
                Defaults to None.
                
        Returns:
            Tensor: Integrated values for each element of shape (n_elem,).
                If field is None, returns element volumes.
        """
        # Default field is ones to integrate volume
        if field is None:
            field = torch.ones(self.n_nod)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Integrate
        res = torch.zeros(len(self.elements))
        for w, xi in zip(self.etype.iweights(), self.etype.ipoints()):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Evalute shale functions
            N, B, detJ = self.eval_shape_functions(xi)
            f = field[self.elements, None].squeeze() @ N
            res += w * f * detJ
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return res
    # -------------------------------------------------------------------------
    def assemble_stiffness(self, k: Tensor, con: Tensor) -> Tensor:
        """Assemble global stiffness matrix from element contributions.
        Args:
            k (Tensor): Element stiffness matrices of shape 
                (n_elem, n_dof_elem, n_dof_elem).
            con (Tensor): Global DOF indices of constrained degrees of freedom
                of shape (n_constraints,).
                
        Returns:
            Tensor: Assembled sparse global stiffness matrix of shape 
                (n_dofs, n_dofs) in COO format.
        """
        # Initialize sparse matrix
        size = (self.n_dofs, self.n_dofs)
        K = torch.empty(size, layout=torch.sparse_coo)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build matrix in chunks to prevent excessive memory usage
        chunks = 4
        for idx, k_chunk in zip(torch.chunk(self.idx, chunks), 
                                torch.chunk(k, chunks)):
            # Ravel indices and values
            chunk_size = idx.shape[0]
            col = idx.unsqueeze(1).expand(chunk_size, self.idx.shape[1], 
                                          -1).ravel()
            row = idx.unsqueeze(-1).expand(chunk_size, -1, 
                                           self.idx.shape[1]).ravel()
            indices = torch.stack([row, col], dim=0)
            values = k_chunk.ravel()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Eliminate and replace constrained dofs
            ci = torch.isin(idx, con)
            mask_col = ci.unsqueeze(1).expand(chunk_size, self.idx.shape[1], 
                                              -1).ravel()
            mask_row = (
                ci.unsqueeze(-1).expand(chunk_size, -1, 
                                        self.idx.shape[1]).ravel()
            )
            mask = ~(mask_col | mask_row)
            diag_index = torch.stack((con, con), dim=0)
            diag_value = torch.ones_like(con, dtype=k.dtype)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate
            indices = torch.cat((indices[:, mask], diag_index), dim=1)
            values = torch.cat((values[mask], diag_value), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble stiffness matrix as a sparse coo tensor
            K += torch.sparse_coo_tensor(indices, values, size=size).coalesce()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return K.coalesce()
    # -------------------------------------------------------------------------
    def assemble_force(self, f: Tensor) -> Tensor:
        """Assemble global force vector from element contributions.
        
        Args:
            f (Tensor): Element force vectors of shape (n_elem, n_dof_elem).
            
        Returns:
            Tensor: Assembled global force vector of shape (n_dofs,).
        """

        # Initialize force vector
        F = torch.zeros((self.n_dofs))

        # Ravel indices and values
        indices = self.idx.ravel()
        values = f.ravel()

        return F.index_add_(0, indices, values)
    # -------------------------------------------------------------------------
    def _apply_constraints_sparse(self, K, con):
        """Eliminate constrained DOFs from sparse K.

        Removes entries in constrained rows/columns and
        adds identity diagonal entries for constrained
        DOFs. Mirrors the constraint logic in
        assemble_stiffness.

        Parameters
        ----------
        K : torch.sparse_coo_tensor
            Global stiffness matrix in COO format.
        con : torch.Tensor
            Indices of constrained DOFs.

        Returns
        -------
        K_con : torch.sparse_coo_tensor
            Stiffness matrix with constraints applied.
        """
        K = K.coalesce()
        indices = K.indices()
        values = K.values()
        rows = indices[0]
        cols = indices[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Mask out constrained rows and columns
        mask_row = torch.isin(rows, con)
        mask_col = torch.isin(cols, con)
        mask = ~(mask_row | mask_col)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Identity diagonal for constrained DOFs
        diag_idx = torch.stack((con, con), dim=0)
        diag_val = torch.ones(
            len(con), dtype=values.dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rebuild sparse tensor
        new_indices = torch.cat(
            (indices[:, mask], diag_idx), dim=1)
        new_values = torch.cat(
            (values[mask], diag_val), dim=0)
        return torch.sparse_coo_tensor(
            new_indices, new_values, K.shape
        ).coalesce()
    # -------------------------------------------------------------------------
    def solve(
        self,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 100,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        device: str | None = None,
        return_intermediate: bool = True,
        aggregate_integration_points: bool = True,
        aggregate_state: bool = None,
        use_cached_solve: bool = False,
        nlgeom: bool = False,
        return_volumes: bool = False,
        return_resnorm: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Solve the FEM problem with the Newton-Raphson method.

        Args:
            increments (Tensor): Load increment stepping.
            max_iter (int): Maximum number of iterations during Newton-Raphson.
            rtol (float): Relative tolerance for Newton-Raphson convergence.
            atol (float): Absolute tolerance for Newton-Raphson convergence.
            stol (float): Solver tolerance for iterative methods.
            verbose (bool): Print iteration information.
            method (str): Method for linear solve 
                ('spsolve','minres','cg','pardiso').
            device (str): Device to run the linear solve on.
            return_intermediate (bool): Return intermediate values if True.
            aggregate_integration_points (bool): Aggregate integration 
                points if True.
            use_cached_solve (bool): Use cached solve, e.g. in topology 
                optimization.
            nlgeom (bool): Use nonlinear geometry if True.
            return_volumes (bool): Return element volumes for each 
                increment if True.
            return_resnorm (bool): Return residual norm history for each 
                increment if True.

        Returns:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Final 
                    displacements,
                    internal forces, stress, deformation gradient, and 
                    material state.
                If return_volumes=True, also returns element volumes with shape
                (num_increments, num_elem, 1).
                If return_resnorm=True, also returns residual history dict with
                increment numbers as keys and lists of residual norms as values.

        """
        # Number of increments
        N = len(increments)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Null space rigid body modes for AMG preconditioner
        B = self.compute_B()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dim)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        stress = torch.zeros(N, self.n_int, self.n_elem,
                             self.n_stress, self.n_stress)
        defgrad = torch.zeros(N, self.n_int, self.n_elem,
                              self.n_stress, self.n_stress)
        defgrad[:, :, :, :, :] = torch.eye(self.n_stress)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize volumes if requested
        if return_volumes:
            volumes = torch.zeros(N, self.n_elem)
            # Compute initial volume for increment 0
            # Default field is ones to integrate volume
            volumes[0] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual norm history if requested
        if return_resnorm:
            residual_history = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global stiffness matrix
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]
            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.ext_strain
            # Newton-Raphson iterations
            # Initialize residual list for this increment if requested
            if return_resnorm:
                residual_history[n] = []
            
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Element-wise integration
                # profiler = cProfile.Profile()
                # profiler.enable()
                k, f_i = self.integrate_material(
                    u, defgrad, stress, state, n, du, de0, nlgeom
                )
                # profiler.disable()
                # print(f"\n=== integrate_material PROFILE (increment {n}) ===")
                # stats = pstats.Stats(profiler)
                # stats.sort_stats('cumulative').print_stats(10)
                if self.K.numel() == 0 or not self.material.n_state == 0 or \
                    nlgeom:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_i)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save residual norm to history if requested
                if return_resnorm:
                    residual_history[n].append(res_norm.item())
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save initial residual
                if i == 0:
                    res_norm0 = res_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Print iteration information
                if verbose:
                    print(f"Increment {n} | Iteration {i+1} | "
                          f"Residual: {res_norm:.5e}")
                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Use cached solve from previous iteration if available
                if i == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()
                # Only update cache on first iteration
                update_cache = i == 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve for displacement increment
                du -= sparse_solve(
                    self.K,
                    residual,
                    B,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check convergence
            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element volumes if requested
            if return_volumes:
                volumes[n] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Aggregate integration points as mean
        if aggregate_state is None:
            aggregate_state = aggregate_integration_points
        if aggregate_integration_points:
            defgrad = defgrad.mean(dim=1)
            stress = stress.mean(dim=1)
        if aggregate_state:
            state = state.mean(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Squeeze outputs
        stress = stress.squeeze()
        defgrad = defgrad.squeeze()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        result = [u, f, stress, defgrad, state]
        # Return intermediate states
        if not return_intermediate:
            result = [x[-1] for x in result]
        # Return volumes
        if return_volumes:
            result.append(volumes if return_intermediate else volumes[-1])
        # Return residual norm
        if return_resnorm:
            result.append(residual_history)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(result)
    # -------------------------------------------------------------------------
    def integrate_material_msc(
        self,
        msc_model,
        u: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        n: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
        scaler_hydrostatic: float,
        scaler_deviatoric: float,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations using MSC surrogate model.
        
        Args:
            msc_model: Loaded MSC PyTorch model.
            u (Tensor): Displacement history of shape (n_increments, n_nodes, 
                n_dim).
            F (Tensor): Deformation gradient history of shape (n_increments, 
                n_int, n_elem, n_stress, n_stress).
            stress (Tensor): Stress history of shape (n_increments, n_int, 
                n_elem, n_stress, n_stress).
            state (Tensor): Material state variable history of shape 
                (n_increments, n_int, n_elem, n_state).
            n (int): Current increment number.
            du (Tensor): Displacement increment vector of shape (n_dofs,).
            de0 (Tensor): External strain increment of shape (n_elem, 
                n_stress, n_stress).
            nlgeom (bool): Whether to use nonlinear geometry.
            scaler_hydrostatic (float): Scaling factor for hydrostatic stress.
            scaler_deviatoric (float): Scaling factor for deviatoric stress.
            
        Returns:
            Tuple[Tensor, Tensor]: Element stiffness matrices of shape 
                (n_elem, n_dof_elem, n_dof_elem) and internal force vectors
                of shape (n_elem, n_dof_elem).
        """
        
        # Compute updated configuration
        u_trial = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape displacement increment
        du = du.view(-1, self.n_dim)[self.elements].reshape(
            self.n_elem, -1, self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize nodal force and stiffness
        n_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dim * n_nod)
        k = torch.zeros((self.n_elem, self.n_dim * n_nod, self.n_dim * n_nod))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i, (w, xi) in enumerate(zip(self.etype.iweights(), 
                                        self.etype.ipoints())):
            # Compute gradient operators
            _, B0, detJ0 = self.eval_shape_functions(xi)
            if nlgeom:
                # Compute updated gradient operators in deformed configuration
                _, B, detJ = self.eval_shape_functions(xi, u_trial)
            else:
                # Use initial gradient operators
                B = B0
                detJ = detJ0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute displacement gradient increment
            H_inc = torch.einsum("...ij,...jk->...ik", B0, du)
            # Update deformation gradient
            # F[n, i] = F[n - 1, i] + H_inc
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute increment strain tensor from displacement gradient
            # For small strain: epsilon = 0.5*(H + H^T)
            delta_eps = 0.5 * (H_inc + H_inc.transpose(-1, -2)).requires_grad_(
                True)
            # Prepare strains in torch-fem order for gradient computation
            delta_eps_msc = torch.zeros(self.n_elem, 6)
            delta_eps_msc[:, 0] = delta_eps[:, 0, 0]  # eps_11
            delta_eps_msc[:, 1] = delta_eps[:, 0, 1]  # eps_12
            if self.n_dim == 3:
                delta_eps_msc[:, 2] = delta_eps[:, 0, 2]  # eps_13
                delta_eps_msc[:, 3] = delta_eps[:, 1, 1]  # eps_22
                delta_eps_msc[:, 4] = delta_eps[:, 1, 2]  # eps_23
                delta_eps_msc[:, 5] = delta_eps[:, 2, 2]  # eps_33
            else:
                delta_eps_msc[:, 2] = delta_eps[:, 1, 1]  # eps_22
            
            # Reshape for MSC input (seq_len=1, features=6)
            eps_msc_input = delta_eps_msc.unsqueeze(1)
            breakpoint()
            # Extract hidden states from previous step
            # state[n-1, i] stores states per integration point
            hidden_states = state[n - 1, i]
            
            # Single forward pass with gradients enabled
            sigma_pred, new_hidden_states = msc_model(
                eps_msc_input, hidden_states)

            # Store new hidden states
            # state[n, i] stores states per integration point
            state[n, i] = new_hidden_states

            # Apply denormalization to get stress in MSC order
            # sigma_pred is (batch, 6), need to add time dimension
            sigma_denorm = sigma_pred.clone()
            sigma_denorm[:, 0] *= scaler_hydrostatic
            sigma_denorm[:, 1:] *= scaler_deviatoric
            breakpoint()
            # Reorder denormalized stress to torch-fem order
            sigma_torchfem = torch.zeros(self.n_elem, 6)
            sigma_torchfem[:, 0] = (sigma_denorm[:, 1] + 
                                     sigma_denorm[:, 0])  # sigma_11
            sigma_torchfem[:, 1] = sigma_denorm[:, 3]  # sigma_12
            if self.n_dim == 3:
                sigma_torchfem[:, 2] = sigma_denorm[:, 4]  # sigma_13
                sigma_torchfem[:, 3] = (sigma_denorm[:, 2] + 
                                         sigma_denorm[:, 0])  # sigma_22
                sigma_torchfem[:, 4] = sigma_denorm[:, 5]  # sigma_23
                sigma_torchfem[:, 5] = (3*sigma_denorm[:, 0] - 
                                         sigma_torchfem[:, 0] - 
                                         sigma_torchfem[:, 3])  # sigma_33
            else:
                sigma_torchfem[:, 2] = (sigma_denorm[:, 2] + 
                                         sigma_denorm[:, 0])  # sigma_22
            breakpoint()
            # Update stress tensor for this integration point
            if self.n_dim == 3:
                stress[n, i, :, 0, 0] = sigma_torchfem[:, 0]  # sigma_11
                stress[n, i, :, 0, 1] = sigma_torchfem[:, 1]  # sigma_12
                stress[n, i, :, 0, 2] = sigma_torchfem[:, 2]  # sigma_13
                stress[n, i, :, 1, 0] = sigma_torchfem[:, 1]  # sigma_21
                stress[n, i, :, 1, 1] = sigma_torchfem[:, 3]  # sigma_22
                stress[n, i, :, 1, 2] = sigma_torchfem[:, 4]  # sigma_23
                stress[n, i, :, 2, 0] = sigma_torchfem[:, 2]  # sigma_31
                stress[n, i, :, 2, 1] = sigma_torchfem[:, 4]  # sigma_32
                stress[n, i, :, 2, 2] = sigma_torchfem[:, 5]  # sigma_33
            else:
                stress[n, i, :, 0, 0] = sigma_torchfem[:, 0]  # sigma_11
                stress[n, i, :, 0, 1] = sigma_torchfem[:, 1]  # sigma_12
                stress[n, i, :, 1, 0] = sigma_torchfem[:, 1]  # sigma_21
                stress[n, i, :, 1, 1] = sigma_torchfem[:, 2]  # sigma_22
            
            # Compute ddsdde via autograd as 4th-order tensor
            # ddsdde[e,i_idx,j_idx,k,l] = 
            # \partial sigma__ij/\partial \varepsilon_kl
            ddsdde = torch.zeros(self.n_elem, 3, 3, 3, 3)
            for i_idx in range(3):
                for j_idx in range(3):
                    grad_output = torch.zeros(self.n_elem, 3, 3)
                    grad_output[:, i_idx, j_idx] = 1.0

                    grads = torch.autograd.grad(
                        outputs=stress[n, i],
                        inputs=delta_eps,
                        grad_outputs=grad_output,
                        retain_graph=True,
                        create_graph=False)[0]
                    # grads has shape (n_elem, 3, 3)
                    ddsdde[:, i_idx, j_idx, :, :] = grads
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element internal forces
            force_contrib = self.compute_f(detJ, B, stress[n, i].clone())
            f += w * force_contrib.reshape(-1, self.n_dim * n_nod)
            breakpoint()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                # Material stiffness
                BCB = torch.einsum(
                    "...ijpq,...qk,...il->...ljkp", ddsdde, B, B)
                BCB = BCB.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, BCB)
            breakpoint()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if nlgeom:
                # Geometric stiffness
                BSB = torch.einsum(
                    "...iq,...qk,...il->...lk", stress[n, i].clone(), B, B
                )
                zeros = torch.zeros_like(BSB)
                kg = torch.stack([BSB] + (self.n_dim - 1) * [zeros], dim=-1)
                kg = kg.reshape(-1, n_nod, self.n_dim * n_nod).unsqueeze(-2)
                zeros = torch.zeros_like(kg)
                kg = torch.stack([kg] + (self.n_dim - 1) * [zeros], dim=-2)
                kg = kg.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, kg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return k, f
    # -------------------------------------------------------------------------
    def solve_msc(
        self,
        msc_model,
        msc_variables: int = 7,
        scaler_hydrostatic: float = 1375.297984380115,
        scaler_deviatoric: float = 324.7645473652983,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 100,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        device: str | None = None,
        return_intermediate: bool = True,
        aggregate_integration_points: bool = True,
        use_cached_solve: bool = False,
        nlgeom: bool = False,
        return_volumes: bool = False,
        return_resnorm: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Solve the FEM problem using the MSC surrogate model.

        Args:
            msc_variables (int): Number of MSC hidden state variables.
            scaler_hydrostatic (float): Scaling factor for hydrostatic
                stress.
            scaler_deviatoric (float): Scaling factor for deviatoric
                stress.
            increments (Tensor): Load increment stepping.
            max_iter (int): Maximum number of iterations during
                Newton-Raphson.
            rtol (float): Relative tolerance for Newton-Raphson
                convergence.
            atol (float): Absolute tolerance for Newton-Raphson
                convergence.
            stol (float): Solver tolerance for iterative methods.
            verbose (bool): Print iteration information.
            method (str): Method for linear solve 
                ('spsolve','minres','cg','pardiso').
            device (str): Device to run the linear solve on.
            return_intermediate (bool): Return intermediate values if True.
            aggregate_integration_points (bool): Aggregate integration 
                points if True.
            use_cached_solve (bool): Use cached solve, e.g. in topology 
                optimization.
            nlgeom (bool): Use nonlinear geometry if True.
            return_volumes (bool): Return element volumes for each 
                increment if True.
            return_resnorm (bool): Return residual norm history for each 
                increment if True.

        Returns:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Final 
                    displacements,
                    internal forces, stress, deformation gradient, and 
                    material state.
                If return_volumes=True, also returns element volumes with shape
                (num_increments, num_elem, 1).
                If return_resnorm=True, also returns residual history dict with
                increment numbers as keys and lists of residual norms as values.

        """
        # Number of increments
        N = len(increments)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Null space rigid body modes for AMG preconditioner
        B = self.compute_B()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dim)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        stress = torch.zeros(N, self.n_int, self.n_elem,
                             self.n_stress, self.n_stress)
        defgrad = torch.zeros(N, self.n_int, self.n_elem,
                              self.n_stress, self.n_stress)
        defgrad[:, :, :, :, :] = torch.eye(self.n_stress)
        state = torch.zeros(N, self.n_int, self.n_elem, msc_variables)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize volumes if requested
        if return_volumes:
            volumes = torch.zeros(N, self.n_elem)
            # Compute initial volume for increment 0
            # Default field is ones to integrate volume
            volumes[0] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual norm history if requested
        if return_resnorm:
            residual_history = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global stiffness matrix
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]
            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.ext_strain
            # Newton-Raphson iterations
            # Initialize residual list for this increment if requested
            if return_resnorm:
                residual_history[n] = []
            
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Element-wise integration with MSC surrogate
                k, f_i = self.integrate_material_msc(
                    msc_model, u, defgrad, stress, state, n, du, de0, nlgeom,
                    scaler_hydrostatic, scaler_deviatoric
                )
                if self.K.numel() == 0 or not msc_variables == 0 or nlgeom:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_i)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)
                breakpoint()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save residual norm to history if requested
                if return_resnorm:
                    residual_history[n].append(res_norm.item())
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save initial residual
                if i == 0:
                    res_norm0 = res_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Print iteration information
                if verbose:
                    print(f"Increment {n} | Iteration {i+1} | "
                          f"Residual: {res_norm:.5e}")
                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Use cached solve from previous iteration if available
                if i == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()
                # Only update cache on first iteration
                update_cache = i == 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve for displacement increment
                du -= sparse_solve(
                    self.K,
                    residual,
                    B,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check convergence
            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element volumes if requested
            if return_volumes:
                volumes[n] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Aggregate integration points as mean
        if aggregate_integration_points:
            defgrad = defgrad.mean(dim=1)
            stress = stress.mean(dim=1)
            state = state.mean(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Squeeze outputs
        stress = stress.squeeze()
        defgrad = defgrad.squeeze()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        result = [u, f, stress, defgrad, state]
        # Return intermediate states
        if not return_intermediate:
            result = [x[-1] for x in result]
        # Return volumes
        if return_volumes:
            result.append(volumes if return_intermediate else volumes[-1])
        # Return residual norm
        if return_resnorm:
            result.append(residual_history)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(result)
    # -------------------------------------------------------------------------
    def _build_graph_topology(self, patch_id,
                              patch_elem_per_dim=None,
                              edge_type='all'):
        """Build and store graph topology for a patch.

        Parameters
        ----------
        patch_id : int
            Material patch identifier.
        patch_elem_per_dim : list, default=None
            Elements per dimension, e.g. [2, 2].
            Defaults to [1, 1].
        edge_type : {'all', 'bd'}, default='all'
            Edge connectivity type. 'all' uses
            radius_mult=20.0, 'bd' uses 0.8.
            Must match the training configuration.
        """
        # Get material patch boundary nodes
        boundary_node_ids = self.patch_bd_nodes[patch_id]
        boundary_coords_ref = self.nodes[boundary_node_ids]
        node_coords_init = boundary_coords_ref.detach().numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract patch dimensions from bounding box of boundary coordinates
        dim = self.n_dim
        coords_min = boundary_coords_ref.min(dim=0).values
        coords_max = boundary_coords_ref.max(dim=0).values
        patch_dim_tensor = coords_max - coords_min
        patch_dim = patch_dim_tensor.tolist()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elements per dimension from parameter or default to [1, 1] / [1]
        if patch_elem_per_dim is None:
            if dim == 2:
                n_elem_per_dim = [1, 1]
            elif dim == 3:
                n_elem_per_dim = [1, 1, 1]
        else:
            n_elem_per_dim = patch_elem_per_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Material patches always use Quad4 (linear) elements
        elem_order = 1
        # Set mesh dimensions from n_elem_per_dim
        mesh_nx = n_elem_per_dim[0]
        mesh_ny = n_elem_per_dim[1]
        mesh_nz = n_elem_per_dim[2] if dim == 3 else 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build mesh_nodes_matrix for Quad4-based patches
        # For a patch with mesh_nx x mesh_ny elements, the boundary nodes
        # form a (mesh_nx+1) x (mesh_ny+1) grid, but we only keep boundary
        n_nod = self.etype.nodes
        if dim == 2:
            # Linear elements: (mesh_nx+1)x(mesh_ny+1) nodes
            mesh_nodes_matrix = np.zeros(
                (mesh_nx + 1, mesh_ny + 1), dtype=int)
            node_idx = 0
            for i in range(mesh_nx + 1):
                for j in range(mesh_ny + 1):
                    mesh_nodes_matrix[i, j] = node_idx
                    node_idx += 1
        elif dim == 3:
            # Linear elements: (mesh_nx+1)x(mesh_ny+1)x(mesh_nz+1) nodes
            mesh_nodes_matrix = np.zeros(
                (mesh_nx + 1, mesh_ny + 1, mesh_nz + 1), dtype=int)
            node_idx = 0
            for i in range(mesh_nx + 1):
                for j in range(mesh_ny + 1):
                    for k in range(mesh_nz + 1):
                        mesh_nodes_matrix[i, j, k] = node_idx
                        node_idx += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate GNN-based material patch graph data
        gnn_patch_data = GraphData(
            n_dim=dim, nodes_coords=node_coords_init)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set connectivity radius based on element size
        if edge_type == 'all':
            radius_mult = 20.0
        elif edge_type == 'bd':
            radius_mult = 0.8
        else:
            raise RuntimeError(
                f'Unknown edge_type: {edge_type}')
        connect_radius = radius_mult * np.sqrt(
            np.sum([x**2 for x in get_elem_size_dims(
                patch_dim, n_elem_per_dim, dim)]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Identify perimeter nodes from local mesh
        # (matches training code in gen_graphs_files.py)
        if dim == 2:
            bd_node_indices = []
            for i in range(mesh_nx + 1):
                for j in range(mesh_ny + 1):
                    if (i == 0 or i == mesh_nx
                            or j == 0
                            or j == mesh_ny):
                        bd_node_indices.append(
                            mesh_nodes_matrix[i, j])
            bd_node_indices = sorted(bd_node_indices)
        elif dim == 3:
            bd_node_indices = []
            for i in range(mesh_nx + 1):
                for j in range(mesh_ny + 1):
                    for k in range(mesh_nz + 1):
                        if (i == 0
                                or i == mesh_nx
                                or j == 0
                                or j == mesh_ny
                                or k == 0
                                or k == mesh_nz):
                            bd_node_indices.append(
                                mesh_nodes_matrix[
                                    i, j, k])
            bd_node_indices = sorted(bd_node_indices)
        boundary_node_set = set(bd_node_indices)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Map local perimeter index to sequential
        # boundary position
        original_to_boundary_idx = {
            node_id: position for position,
            node_id in enumerate(bd_node_indices)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get finite element mesh edges for all nodes
        connected_nodes_all = get_mesh_connected_nodes(
            dim, mesh_nodes_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Filter connected_nodes to only include boundary node pairs
        connected_nodes_boundary = []
        for node1, node2 in connected_nodes_all:
            if node1 in boundary_node_set and node2 in boundary_node_set:
                boundary_node1 = original_to_boundary_idx[node1]
                boundary_node2 = original_to_boundary_idx[node2]
                connected_nodes_boundary.append(
                    (boundary_node1, boundary_node2))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Organize connected nodes as tuples
        connected_nodes = tuple(connected_nodes_boundary)
        edges_indexes_mesh = GraphData.get_edges_indexes_mesh(
            connected_nodes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based material patch graph edges
        gnn_patch_data.set_graph_edges_indexes(
            connect_radius=connect_radius,
            edges_indexes_mesh=edges_indexes_mesh)
        edges_indexes = torch.tensor(
            np.transpose(
                copy.deepcopy(gnn_patch_data.get_graph_edges_indexes())),
                dtype=torch.long)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._edges_indexes[f"patch_{patch_id}"] = edges_indexes
    # -------------------------------------------------------------------------
    def surrogate_integrate_material(
        self, model, u: Tensor, n: int,
        du: Tensor,
        is_stepwise: bool = False,
        is_state_variable: bool = False,
        patch_ids: Tensor = None,
        hidden_states: dict = None,
        state_variables: dict = None,
        edge_feature_type: tuple = ('edge_vector',),
        model_cache: dict = None,
        patch_resolution: dict = None,
        is_jacfwd_parallel: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Perform surrogate integration using Graphorge material patch model.

        Args:
            model (GNNEPDBaseModel): Pre-loaded Graphorge material patch model
                configured for inference.
            u (Tensor): Displacement history tensor of shape (n_increments,
                n_nodes, n_dim). Only current displacement u[n] is used for
                graph construction.
            n (int): Current increment number (0-indexed).
            du (Tensor): Displacement increment tensor of shape (n_dofs,).
                Used to update current displacement configuration.
            is_stepwise (bool): Whether to use stepwise RNN mode with
                hidden state tracking between steps. Defaults to False.
            patch_ids (Tensor): Patch IDs to process.
            hidden_states (dict): Hidden states for RNN mode.
            is_jacfwd_parallel (bool): If True, use jacfwd (all tangent
                vectors simultaneously, high memory). If False, build
                Jacobian column-by-column via jvp (constant memory,
                slower). Defaults to True.

        Returns:
            Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, dict]: A tuple
                containing:
                - K_global (Tensor): Global stiffness matrix (sparse COO)
                  of shape (n_dof_global, n_dof_global).
                - F_global (Tensor): Global internal force vector of shape
                  (n_dof_global,).
                - hidden_states_out (dict, optional): Updated hidden states
                  for each patch (only returned in stepwise mode).

        Note:
            This method constructs graphs using only boundary nodes of each
            patch. Internal nodes are constrained to zero displacement and do
            not contribute to stiffness or forces.
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize patch-specific hidden states output
        hidden_states_out = {}
        # Initialize state variable outputs
        state_var_out = {}
        state_var_trial = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update current displacement
        u[n] = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global force and stiffness assembly
        n_dof_global = self.n_nod * self.n_dim
        F_global = torch.zeros(n_dof_global)

        # Sparse stiffness assembly (COO format)
        k_indices_list = []
        k_values_list = []
        # Per-patch boundary stiffness matrices
        patch_stiffness_dict = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # OLD: Initialize nodal force and stiffness
        # n_nod = self.etype.nodes
        # n_dof_elem = self.n_dim * n_nod
        # f = torch.zeros(self.n_elem, n_dof_elem)
        # k = torch.zeros((self.n_elem, n_dof_elem, n_dof_elem))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine which patches to process based on patch_ids
        if patch_ids is not None:
            # patch_ids contains the actual patch IDs to process
            # Convert to list for iteration
            patch_indices = patch_ids.tolist()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over patches
        for idx_patch in patch_indices:
            # Select model for this patch
            if (model_cache is not None
                    and patch_resolution is not None):
                res = patch_resolution[idx_patch]
                model = model_cache[res]
            elif model_cache is not None:
                model = model_cache['default']
            # Get patch identifier for stepwise mode
            patch_id = f"patch_{idx_patch}"
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material patch boundary nodes
            boundary_node_ids = self.patch_bd_nodes[idx_patch]
            n_boundary = len(boundary_node_ids)
            n_dof_boundary = n_boundary * self.n_dim
            # Current displacement at boundary nodes
            boundary_u_current = u[n, boundary_node_ids, :].clone()
            # Extract boundary node coordinates
            boundary_coords_ref = self.nodes[boundary_node_ids]
            # Get pre-computed edges_indexes for this patch
            edges_indexes = self._edges_indexes[patch_id]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DEBUG: node-ordering consistency check
            # (once per patch_id per process).
            #
            # Works regardless of edge_type ('all' or
            # 'bd'): tests whether
            # boundary_coords_ref[k] coincides with
            # the k-th position of the training-time
            # column-major local pseudo-grid. That is
            # the convention used by
            # _create_mesh_nodes_matrix in
            # run_simulation.py and
            # _build_graph_topology in this file
            # (both iterate i over x, j over y, so
            # local-id k = i*(ny+1) + j).
            _dbg_seen = getattr(
                self, '_ordering_debug_seen', None)
            if _dbg_seen is None:
                _dbg_seen = set()
                self._ordering_debug_seen = _dbg_seen
            if patch_id not in _dbg_seen:
                _dbg_seen.add(patch_id)
                _coords = (boundary_coords_ref
                           .detach().cpu().numpy())
                _bd_ids = (boundary_node_ids
                           .detach().cpu().numpy())
                _n = len(_coords)
                # Shift to local-patch origin
                _loc = _coords - _coords.min(axis=0)
                # Recover element size per axis from
                # unique coord values on the perimeter
                _ux = np.unique(np.round(
                    _loc[:, 0], 10))
                _uy = np.unique(np.round(
                    _loc[:, 1], 10))
                _dx = float(np.min(np.diff(_ux))) \
                    if len(_ux) > 1 else 1.0
                _dy = float(np.min(np.diff(_uy))) \
                    if len(_uy) > 1 else 1.0
                _i = np.round(_loc[:, 0] / _dx) \
                    .astype(int)
                _j = np.round(_loc[:, 1] / _dy) \
                    .astype(int)
                _ny = int(_j.max())
                # Column-major key: k = i*(ny+1) + j
                _key = _i * (_ny + 1) + _j
                # Permutation that, applied to the
                # current boundary_node_ids order,
                # yields column-major training order:
                #   perm[k] = index in current order
                #             of the k-th column-
                #             major position.
                _perm = np.argsort(_key)
                _is_id = bool(np.array_equal(
                    _perm, np.arange(_n)))
                _span = _coords.max(axis=0) \
                    - _coords.min(axis=0)
                print(
                    f'[ORDER-DBG] {patch_id} '
                    f'n_bd={_n} span={_span} '
                    f'dx={_dx:.4e} dy={_dy:.4e}')
                print(
                    f'[ORDER-DBG] {patch_id} '
                    f'colmajor_perm_is_identity='
                    f'{_is_id}')
                if not _is_id:
                    _n_fixed = int(
                        (_perm == np.arange(_n))
                        .sum())
                    print(
                        f'[ORDER-DBG] {patch_id} '
                        f'ORDERING MISMATCH: '
                        f'boundary_node_ids order '
                        f'!= training col-major '
                        f'order. Fixed points: '
                        f'{_n_fixed}/{_n}. perm='
                        f'{_perm.tolist()}')
                    _bd_global = _bd_ids.tolist()
                    _bd_colmaj = _bd_ids[_perm] \
                        .tolist()
                    print(
                        f'[ORDER-DBG] {patch_id} '
                        f'bd_ids (global-sorted) = '
                        f'{_bd_global}')
                    print(
                        f'[ORDER-DBG] {patch_id} '
                        f'bd_ids in col-major '
                        f'order = {_bd_colmaj}')
                else:
                    print(
                        f'[ORDER-DBG] {patch_id} '
                        f'ordering OK: '
                        f'boundary_node_ids '
                        f'already in col-major '
                        f'order')
                # Edge-feature sanity print for
                # edge_type=='all' (dense graph):
                # report the edge-vector spread so
                # user can sanity-check that edge
                # attrs at inference are consistent
                # with training coord range.
                _ei = edges_indexes.detach() \
                    .cpu().numpy()
                _src = _coords[_ei[0]]
                _dst = _coords[_ei[1]]
                _len = np.linalg.norm(
                    _src - _dst, axis=1)
                print(
                    f'[ORDER-DBG] {patch_id} '
                    f'n_edges={_ei.shape[1]} '
                    f'edge_len min={_len.min():.4e}'
                    f' max={_len.max():.4e} '
                    f'mean={_len.mean():.4e} '
                    f'(note: with edge_type=all '
                    f'the graph is ~dense, so edge-'
                    f'length spread is expected)')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # OLD: Get material patch nodes (all nodes, not just boundary)
            # elem_nodes = self.elements[idx_patch]
            # # Current displacement at increment n
            # elem_u_current = u[n, elem_nodes, :].clone()
            # # Extract material patch nodal coordinates
            # elem_coords_ref = self.nodes[elem_nodes]
            # # Get pre-computed edges_indexes for this patch
            # edges_indexes = self._edges_indexes[patch_id]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            def detach_hidden_states(states):
                if isinstance(states, dict):
                    return {
                        k: detach_hidden_states(v)
                        for k, v in states.items()}
                elif isinstance(states, list):
                    return [
                        detach_hidden_states(item)
                        for item in states]
                elif torch.is_tensor(states):
                    return states.detach()
                else:
                    return states
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            def _load_hidden_into_model(mdl, h):
                """Set hidden states on all model
                sub-components."""
                epd = mdl._gnn_epd_model
                epd._hidden_states = h
                if 'encoder' in h:
                    epd._encoder._hidden_states = \
                        h['encoder']
                if 'processor' in h:
                    epd._processor._hidden_states \
                        = h['processor']
                    for li, layer in enumerate(
                            epd._processor
                            ._processor):
                        lk = f'layer_{li}'
                        if lk in h['processor']:
                            layer._hidden_states \
                                = h['processor'][lk]
                if 'decoder' in h:
                    epd._decoder._hidden_states = \
                        h['decoder']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Snapshot h0 for this patch so it can be
            # restored before every forward/jvp call
            # (each call mutates model hidden states).
            _h0_snapshot = None
            if (is_stepwise and hidden_states
                    and patch_id in hidden_states):
                _h0_snapshot = copy.deepcopy(
                    detach_hidden_states(
                        hidden_states[patch_id]))
                _load_hidden_into_model(
                    model,
                    copy.deepcopy(_h0_snapshot))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            def _restore_h0():
                """Reload h0 into model so the next
                forward call sees the correct hidden
                state."""
                if _h0_snapshot is not None:
                    _load_hidden_into_model(
                        model,
                        copy.deepcopy(
                            detach_hidden_states(
                                _h0_snapshot)))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Retrieve epbar_prev for state variable mode
            epbar_prev = None
            if (is_state_variable
                    and state_variables is not None
                    and patch_id in state_variables):
                epbar_prev = state_variables[
                    patch_id].detach().clone()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute patch coordinate rescaling to [0,1]^d
            # GNN trained on patches with coords in [0,1]^d; rescale inputs
            coords_min = boundary_coords_ref.min(dim=0).values
            coords_max = boundary_coords_ref.max(dim=0).values
            L = coords_max - coords_min  # patch physical size per dimension
            L = torch.clamp(L, min=1e-12)  # avoid division by zero
            coords_scaled = (boundary_coords_ref - coords_min) / L
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Model prediction: stepwise vs non-stepwise mode
            hidden_states_trial = None
            def forward(disp_boundary):
                """Forward function for gradient
                computation.

                Rescales inputs to [0,1]^d for GNN
                inference, then rescales forces back to
                physical space. jacfwd differentiates
                through the full chain so stiffness is
                automatically correct.

                For state variable mode, epbar_prev is a
                closure constant (not differentiated).
                The Jacobian is w.r.t. disps only.
                """
                nonlocal hidden_states_trial
                disp_scaled = disp_boundary / L
                if is_state_variable:
                    # State variable mode: FFNN
                    # epbar_prev captured as closure
                    # constant — not differentiated
                    epbar_sc = epbar_prev / L[:1]
                    pred_scaled = forward_graph(
                        model=model,
                        disps=disp_scaled,
                        coords_ref=coords_scaled,
                        edges_indexes=edges_indexes,
                        n_dim=self.n_dim,
                        edge_feature_type=(
                            edge_feature_type),
                        epbar_prev=epbar_sc)
                    # Split output: forces | delta_raw
                    n_d = self.n_dim
                    forces_sc = pred_scaled[:, :n_d]
                    delta_raw = pred_scaled[:, n_d:]
                    # Monotonicity: softplus
                    delta_epbar = torch.nn.functional \
                        .softplus(delta_raw)
                    epbar_new = (
                        epbar_prev
                        + delta_epbar * L[:1])
                    # Store trial state variable
                    state_var_trial[patch_id] = \
                        epbar_new.detach()
                    # Return forces only
                    pred_real = forces_sc * L
                    return (pred_real,
                            pred_real.detach())
                elif is_stepwise:
                    pred_scaled, hs_out = (
                        forward_graph(
                            model=model,
                            disps=disp_scaled,
                            coords_ref=coords_scaled,
                            edges_indexes=edges_indexes,
                            n_dim=self.n_dim,
                            edge_feature_type=(
                                edge_feature_type)))
                    hidden_states_trial = hs_out
                else:
                    pred_scaled = forward_graph(
                        model=model,
                        disps=disp_scaled,
                        coords_ref=coords_scaled,
                        edges_indexes=edges_indexes,
                        n_dim=self.n_dim,
                        edge_feature_type=(
                            edge_feature_type))
                pred_real = pred_scaled * L
                return pred_real, pred_real.detach()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute Jacobian: d(forces_boundary)/d(u_boundary)
            if is_jacfwd_parallel:
                # Parallel: all tangent vectors at once
                _restore_h0()
                (jacobian, boundary_forces) = \
                    torch_func.jacfwd(
                        forward,
                        has_aux=True)(
                        boundary_u_current)
                k_boundary = jacobian.view(
                    n_dof_boundary, n_dof_boundary)
                f_boundary = boundary_forces.flatten()
            else:
                # Sequential: one jvp per DOF
                # Restore h0 before force computation
                _restore_h0()
                boundary_forces, _ = forward(
                    boundary_u_current)
                f_boundary = (
                    boundary_forces.flatten().detach())
                # Save hidden_states_trial from the
                # force eval — jvp calls would
                # overwrite it with drifted values.
                _hs_trial_saved = hidden_states_trial
                cols = []
                for i in range(n_dof_boundary):
                    # Restore h0 before each jvp so
                    # every column is linearized at
                    # the same operating point.
                    _restore_h0()
                    tangent = torch.zeros(
                        n_dof_boundary,
                        dtype=boundary_u_current.dtype)
                    tangent[i] = 1.0
                    tangent = tangent.view(
                        n_boundary, self.n_dim)
                    _, jvp_col = torch_func.jvp(
                        lambda u: forward(u)[0],
                        (boundary_u_current,),
                        (tangent,))
                    cols.append(
                        jvp_col.flatten().detach())
                k_boundary = torch.stack(
                    cols, dim=1)
                # Restore the correct hidden states
                hidden_states_trial = _hs_trial_saved
            # Store per-patch boundary stiffness
            patch_stiffness_dict[idx_patch] = \
                k_boundary.detach().clone()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DEBUG: post-Jacobian diagnostics, once
            # per patch per process. Tests:
            #   - which model / resolution is used
            #   - magnitudes of u, f, K at iter 1
            #   - K symmetry (conservative elasticity
            #     => K should be symmetric)
            #   - zero-input forward: f(0) ≈ 0 ?
            #   - FD-Jacobian vs jacfwd on col 0 only
            _solve_seen = getattr(
                self, '_solve_debug_seen', None)
            if _solve_seen is None:
                _solve_seen = set()
                self._solve_debug_seen = _solve_seen
            if patch_id not in _solve_seen:
                _solve_seen.add(patch_id)
                _res_tag = (
                    patch_resolution[idx_patch]
                    if (model_cache is not None
                        and patch_resolution
                        is not None) else 'default')
                _mdl_repr = type(model).__name__
                _u_norm = float(torch.norm(
                    boundary_u_current))
                _f_norm = float(torch.norm(
                    f_boundary))
                _K = k_boundary.detach()
                _K_fro = float(torch.norm(_K))
                _K_sym_err = float(torch.norm(
                    _K - _K.T)
                    / max(_K_fro, 1e-30))
                _L_val = L.detach().cpu().numpy()
                print(
                    f'[SOLVE-DBG] {patch_id} '
                    f'res={_res_tag} '
                    f'model={_mdl_repr} L={_L_val}')
                print(
                    f'[SOLVE-DBG] {patch_id} '
                    f'||u||={_u_norm:.4e} '
                    f'||f||={_f_norm:.4e} '
                    f'||K||_F={_K_fro:.4e} '
                    f'||K-K^T||/||K||={_K_sym_err:.3e}')
                # Zero-input sanity
                _zero_u = torch.zeros_like(
                    boundary_u_current)
                _restore_h0()
                _f0, _ = forward(_zero_u)
                _f0_norm = float(torch.norm(_f0))
                print(
                    f'[SOLVE-DBG] {patch_id} '
                    f'||f(u=0)||={_f0_norm:.4e} '
                    f'(should be ~0 for a well-'
                    f'trained surrogate with no '
                    f'prior plastic history)')
                # FD vs jacfwd for column 0
                _eps = 1e-5 * max(
                    _u_norm, 1.0)
                _pert_p = boundary_u_current.clone()
                _pert_m = boundary_u_current.clone()
                _pert_p[0, 0] = (
                    _pert_p[0, 0] + _eps)
                _pert_m[0, 0] = (
                    _pert_m[0, 0] - _eps)
                _restore_h0()
                _fp, _ = forward(_pert_p)
                _restore_h0()
                _fm, _ = forward(_pert_m)
                _fd_col0 = (
                    (_fp - _fm).flatten().detach()
                    / (2.0 * _eps))
                _K_col0 = _K[:, 0]
                _diff = float(torch.norm(
                    _fd_col0 - _K_col0))
                _ref = float(torch.norm(_fd_col0))
                _rel = _diff / max(_ref, 1e-30)
                print(
                    f'[SOLVE-DBG] {patch_id} '
                    f'FD vs jacfwd col0: '
                    f'||FD||={_ref:.4e} '
                    f'||jacfwd||='
                    f'{float(torch.norm(_K_col0)):.4e} '
                    f'||diff||={_diff:.4e} '
                    f'rel={_rel:.3e} '
                    f'(rel<<1 => K matches ∂f/∂u; '
                    f'rel~O(1) => K is broken)')
            # # OLD: Compute Jacobian for all element nodes
            # (jacobian, node_output) = torch_func.jacfwd(
            #     forward, has_aux=True)(elem_u_current)
            # node_forces_trial = node_output.flatten()
            # # Reshape Jacobian to stiffness matrix format
            # stiffness_matrix = jacobian.view(n_dof_elem, n_dof_elem)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble boundary forces to global force vector
            # Vectorized: map local DOFs to global DOFs
            _dim_range = torch.arange(
                self.n_dim, device=boundary_node_ids.device)
            global_dofs = (
                boundary_node_ids.unsqueeze(1) * self.n_dim
                + _dim_range).flatten()
            F_global.index_add_(0, global_dofs, f_boundary)

            # Collect boundary stiffness entries (vectorized)
            rows = global_dofs.unsqueeze(1).expand(
                -1, n_dof_boundary).flatten()
            cols = global_dofs.unsqueeze(0).expand(
                n_dof_boundary, -1).flatten()
            k_indices_list.append(
                torch.stack([rows, cols]))
            k_values_list.append(
                k_boundary.flatten())

            # Store updated hidden states for stepwise mode
            if is_stepwise and hidden_states_trial is not None:
                hidden_states_out[patch_id] = hidden_states_trial
            # Store trial state variables
            if (is_state_variable
                    and patch_id in state_var_trial):
                state_var_out[patch_id] = \
                    state_var_trial[patch_id]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # OLD: Store element forces and stiffness
            # f[idx_patch] = node_forces_trial.flatten()
            # k[idx_patch] = stiffness_matrix
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build global sparse stiffness matrix
        k_indices_tensor = torch.cat(
            k_indices_list, dim=1)
        k_values_tensor = torch.cat(k_values_list)
        K_global = torch.sparse_coo_tensor(
            k_indices_tensor, k_values_tensor,
            (n_dof_global, n_dof_global)
        ).coalesce()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # OLD: Return element-level arrays
        # if is_stepwise:
        #     return k, f, hidden_states_out
        # else:
        #     return k, f
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_stepwise:
            return (K_global, F_global,
                    hidden_states_out,
                    patch_stiffness_dict)
        elif is_state_variable:
            return (K_global, F_global,
                    state_var_out,
                    patch_stiffness_dict)
        else:
            return (K_global, F_global,
                    patch_stiffness_dict)
    # -------------------------------------------------------------------------
    def solve_matpatch(
        self,
        is_mat_patch: Tensor,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 100,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        device: str | None = None,
        return_intermediate: bool = True,
        aggregate_integration_points: bool = True,
        use_cached_solve: bool = False,
        nlgeom: bool = False,
        return_volumes: bool = False,
        is_stepwise: bool = False,
        model_directory: str | dict | None = None,
        return_resnorm: bool = False,
        patch_boundary_nodes: dict | None = None,
        patch_elem_per_dim: list | dict | None = None,
        patch_resolution: dict | None = None,
        edge_type: str = 'all',
        edge_feature_type: tuple = ('edge_vector',),
        is_export_stiffness: bool = False,
        stiffness_output_dir: str | None = None,
        patch_size_label: str | None = None,
        is_jacfwd_parallel: bool = False,
        is_state_variable: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Solve the FEM problem with material patches and the surrogate model.

        Args:
            is_mat_patch (Tensor): Element-wise flag indicating material patch
                usage of shape (n_elem,). Values: 0 = use standard
                integration, >0 = material patch ID for surrogate integration.
            increments (Tensor): Load increment stepping of shape 
                (n_increments,). Defaults to torch.tensor([0.0, 1.0]).
            max_iter (int): Maximum number of iterations during Newton-Raphson.
                Defaults to 100.
            rtol (float): Relative tolerance for Newton-Raphson convergence.
                Defaults to 1e-8.
            atol (float): Absolute tolerance for Newton-Raphson convergence.
                Defaults to 1e-6.
            stol (float): Solver tolerance for iterative methods. 
                Defaults to 1e-10.
            verbose (bool): Whether to print iteration information. 
                Defaults to False.
            method (str, optional): Method for linear solve ('spsolve', 
                'minres', 'cg', 'pardiso'). Defaults to None for automatic 
                selection.
            device (str, optional): Device to run the linear solve on. 
                Defaults to None.
            return_intermediate (bool): Whether to return intermediate values. 
                Defaults to True.
            aggregate_integration_points (bool): Whether to aggregate 
                integration points. Defaults to True.
            use_cached_solve (bool): Whether to use cached solve for 
                optimization. Defaults to False.
            nlgeom (bool): Whether to use nonlinear geometry. 
                Defaults to False.
            return_volumes (bool): Whether to return element volumes for each 
                increment. Defaults to False.
            is_stepwise (bool): Whether to use stepwise RNN mode for 
                surrogate integration. Defaults to False.
            model_directory (str, optional): Path to trained Graphorge model. 
                If None, uses default path. Defaults to None.
            return_resnorm (bool): Whether to return residual norm history. 
                Defaults to False.

        Returns:
            Tuple[Tensor, ...]: If return_volumes=False and return_resnorm=False, 
                returns 5-tuple of (displacements, forces, stress, 
                deformation_gradient, state). If return_volumes=True, returns 
                6-tuple with volumes added. If return_resnorm=True, returns 
                additional residual_history dict as last element.
                If return_intermediate=True, returns full history arrays.
                If return_intermediate=False, returns only final values.
                
        Raises:
            ValueError: If is_mat_patch shape doesn't match number of elements.
            Exception: If Newton-Raphson iteration fails to converge.
        """
        # Validate is_mat_patch tensor
        if is_mat_patch.shape[0] != self.n_elem:
            raise ValueError(
                f'is_mat_patch shape '
                f'{is_mat_patch.shape} must match '
                f'number of elements {self.n_elem}')
        # Mutual exclusion
        assert not (is_stepwise and is_state_variable), \
            'is_stepwise and is_state_variable are ' \
            'mutually exclusive'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Number of increments
        N = len(increments)
        # Null space rigid body modes for AMG preconditioner
        # B = self.compute_B()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dim)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        stress = torch.zeros(N, self.n_int, self.n_elem,
                             self.n_stress, self.n_stress)
        defgrad = torch.zeros(N, self.n_int, self.n_elem,
                              self.n_stress, self.n_stress)
        defgrad[:, :, :, :, :] = torch.eye(self.n_stress)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize volumes if requested
        if return_volumes:
            volumes = torch.zeros(N, self.n_elem)
            # Compute initial volume for increment 0
            volumes[0] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual norm history if requested
        if return_resnorm:
            residual_history = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global stiffness matrix
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Classify elements: FE vs surrogate
        fe_mask = (is_mat_patch == -1)
        patch_mask = (is_mat_patch >= 0)
        has_fe = torch.any(fe_mask).item()
        has_surr = torch.any(patch_mask).item()
        fe_indices = None
        if has_fe:
            fe_indices = torch.where(fe_mask)[0]
        model_cache = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load surrogate model(s)
        if torch.any(patch_mask):
            dev = device if device else 'cpu'
            if isinstance(model_directory, dict):
                # Multi-resolution: dict keyed by
                # resolution tuple -> model path
                for res_key, path in (
                        model_directory.items()):
                    model_cache[res_key] = (
                        self._load_Graphorge_model(
                            model_directory=path,
                            device_type=dev))
            else:
                # Single model (backward compat)
                model_cache['default'] = (
                    self._load_Graphorge_model(
                        model_directory=model_directory,
                        device_type=dev))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Enable stepwise mode on all cached models
            if is_stepwise:
                for m in model_cache.values():
                    m._save_time_series_attrs()
                    m.set_rnn_mode(is_stepwise=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Cache fast affine scaler parameters
            for m in model_cache.values():
                if hasattr(m, 'prepare_fast_scalers'):
                    m.prepare_fast_scalers()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize patch IDs and topology
            patch_ids = torch.unique(
                is_mat_patch[is_mat_patch >= 0])
            self._edges_indexes = {}
            self.patch_bd_nodes = {}
            if patch_boundary_nodes is not None:
                self.patch_bd_nodes = (
                    patch_boundary_nodes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build graph topology and hidden states
            # per patch (resolution-aware)
            hidden_states_dict = {}
            for pid in patch_ids:
                pid_int = pid.item()
                # Resolve per-patch elem_per_dim
                if isinstance(
                        patch_elem_per_dim, dict):
                    epd = patch_elem_per_dim[pid_int]
                else:
                    epd = patch_elem_per_dim
                self._build_graph_topology(
                    pid_int, epd,
                    edge_type=edge_type)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Select model for this patch's
                # resolution to get n_message_steps
                if patch_resolution is not None:
                    res = patch_resolution[pid_int]
                    m = model_cache[res]
                else:
                    m = model_cache['default']
                n_msg = m._n_message_steps
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Create processor hidden states
                processor_hidden = {}
                for li in range(n_msg):
                    processor_hidden[
                        f'layer_{li}'] = {
                        'node': None,
                        'edge': None,
                        'global': None}
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                hidden_states_dict[
                    f'patch_{pid_int}'] = {
                    'encoder': {
                        'node': None,
                        'edge': None,
                        'global': None},
                    'processor': processor_hidden,
                    'decoder': {
                        'node': None,
                        'edge': None,
                        'global': None}}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize state variable dict (epbar)
            state_var_dict = {}
            if is_state_variable:
                for pid in patch_ids:
                    pid_int = pid.item()
                    pk = f'patch_{pid_int}'
                    n_bd = len(
                        self.patch_bd_nodes[pid_int])
                    state_var_dict[pk] = \
                        torch.zeros(n_bd, 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]
            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.ext_strain
            # Initialize residual norm list for this increment if requested
            if return_resnorm:
                residual_history[n] = []
            # Newton-Raphson iterations
            _step_scale = 1.0
            _prev_res = None
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FE integration (regular elements)
                K_fe = None
                F_fe = None
                if has_fe:
                    K_fe, F_fe = (
                        self._integrate_fe_subset(
                            fe_indices, u, defgrad,
                            stress, state, n, du,
                            de0, nlgeom))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Surrogate integration (patch elements)
                K_surr = None
                F_surr = None
                patch_k_dict = {}
                hidden_state_out = None
                state_var_out = None
                if has_surr:
                    patch_ids = torch.unique(
                        is_mat_patch[patch_mask])
                    surr_kwargs = dict(
                        model=None,
                        u=u, n=n, du=du,
                        is_stepwise=is_stepwise,
                        is_state_variable=(
                            is_state_variable),
                        patch_ids=patch_ids,
                        edge_feature_type=
                            edge_feature_type,
                        model_cache=model_cache,
                        patch_resolution=
                            patch_resolution,
                        is_jacfwd_parallel=
                            is_jacfwd_parallel)
                    if is_stepwise:
                        surr_kwargs[
                            'hidden_states'] = (
                            hidden_states_dict)
                        (K_surr, F_surr,
                         hidden_state_out,
                         patch_k_dict) = \
                            self \
                            .surrogate_integrate_material(
                            **surr_kwargs)
                    elif is_state_variable:
                        surr_kwargs[
                            'state_variables'] = (
                            state_var_dict)
                        (K_surr, F_surr,
                         state_var_out,
                         patch_k_dict) = \
                            self \
                            .surrogate_integrate_material(
                            **surr_kwargs)
                    else:
                        (K_surr, F_surr,
                         patch_k_dict) = \
                            self \
                            .surrogate_integrate_material(
                            **surr_kwargs)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Combine FE and surrogate contributions
                if has_fe and has_surr:
                    K_combined = (
                        K_fe + K_surr).coalesce()
                    F_int = F_fe + F_surr
                elif has_fe:
                    K_combined = K_fe
                    F_int = F_fe
                else:
                    K_combined = K_surr
                    F_int = F_surr
                # Apply constraint elimination
                self.K = (
                    self._apply_constraints_sparse(
                        K_combined, con))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)
                # Save residual norm to history if requested
                if return_resnorm:
                    residual_history[n].append(res_norm.item())
                # Save initial residual for relative error
                if i == 0:
                    res_norm0 = res_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Print iteration information
                if verbose:
                    print(
                        f'Increment {n} | '
                        f'Iteration {i+1} | '
                        f'Residual: {res_norm:.5e}')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    # Update patch-specific hidden states
                    # with converged values
                    if is_stepwise:
                        for pid in patch_ids:
                            patch_key = \
                                f"patch_{pid.item()}"
                            hidden_states_dict[
                                patch_key] = \
                                hidden_state_out[
                                    patch_key]
                    # Update state variables on
                    # convergence
                    if (is_state_variable
                            and state_var_out):
                        for pk in state_var_out:
                            state_var_dict[pk] = \
                                state_var_out[pk]
                    # Save converged per-patch stiffness
                    if (is_export_stiffness
                            and stiffness_output_dir):
                        import os
                        for pid, k_mat in \
                                patch_k_dict.items():
                            fname = (
                                f'stiffness_'
                                f'{patch_size_label}'
                                f'_id{pid}'
                                f'_inc{n}.npy')
                            np.save(
                                os.path.join(
                                    stiffness_output_dir,
                                    fname),
                                k_mat.cpu().numpy())
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Use cached solve from previous iteration if available
                if i == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()
                # Only update cache on first iteration
                update_cache = i == 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve for displacement increment
                # Adaptive step damping: if residual
                # grew, halve the step; if it shrank,
                # recover toward full step.
                if (_prev_res is not None
                        and res_norm > 1.01 * _prev_res):
                    _step_scale = max(
                        0.01, _step_scale * 0.5)
                elif _prev_res is not None:
                    _step_scale = min(
                        1.0, _step_scale * 1.1)
                _prev_res = res_norm.item()
                # Break early on NaN
                if torch.isnan(res_norm):
                    break
                delta_u = sparse_solve(
                    self.K,
                    residual,
                    None,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )
                du -= _step_scale * delta_u
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Raise an Exception if not converged
            if (res_norm > rtol * res_norm0
                    and res_norm > atol):
                raise Exception(
                    'Newton-Raphson iteration '
                    'did not converge.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element volumes if requested
            if return_volumes:
                volumes[n] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Disable stepwise mode on all cached models
        if is_stepwise:
            for m in model_cache.values():
                m.set_rnn_mode(is_stepwise=False)
                m._restore_time_series_attrs()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Aggregate integration points as mean
        if aggregate_integration_points:
            defgrad = defgrad.mean(dim=1)
            stress = stress.mean(dim=1)
            state = state.mean(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Squeeze outputs
        stress = stress.squeeze()
        defgrad = defgrad.squeeze()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        result = [u, f, stress, defgrad, state]
        # Return intermediate states
        if not return_intermediate:
            result = [x[-1] for x in result]
        # Return volumes
        if return_volumes:
            result.append(volumes if return_intermediate else volumes[-1])
        # Return residual norm
        if return_resnorm:
            result.append(residual_history)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(result)
# =============================================================================
def compute_edge_features(coords_hist, disps_hist,
                          edges_indexes, n_dim,
                          n_edge_in=None):
        """Compute edge features for GNN inference.

        Computes edge_vector and optionally relative_disp,
        depending on the model's expected n_edge_in.

        Parameters
        ----------
        coords_hist : torch.Tensor
            Current node coordinates,
            shape (n_nodes, n_time_steps*n_dim).
        disps_hist : torch.Tensor
            Node displacements,
            shape (n_nodes, n_time_steps*n_dim).
        edges_indexes : torch.Tensor
            Edge connectivity, shape (2, n_edges).
        n_dim : int
            Spatial dimensions.
        n_edge_in : {int, None}, default=None
            Number of edge input features expected by model.
            If n_dim: only edge_vector.
            If 2*n_dim: edge_vector + relative_disp.
            If None: include both.

        Returns
        -------
        edge_features : torch.Tensor
            Shape (n_edges, n_edge_in).
        """
        edge_sources = edges_indexes[0]
        edge_targets = edges_indexes[1]
        n_edges = edge_sources.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        n_coords_features = coords_hist.shape[1]
        n_time_steps = n_coords_features // n_dim
        edge_vector_size = n_time_steps * n_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine which features to include
        include_rel_disp = True
        if n_edge_in is not None:
            if n_edge_in == edge_vector_size:
                include_rel_disp = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if include_rel_disp:
            n_edge_features = 2 * edge_vector_size
        else:
            n_edge_features = edge_vector_size
        edge_features = torch.zeros(
            n_edges, n_edge_features,
            device=coords_hist.device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorized edge feature computation
        edge_features[:, :edge_vector_size] = (
            coords_hist[edge_sources]
            - coords_hist[edge_targets])
        if include_rel_disp:
            edge_features[:, edge_vector_size:] = (
                disps_hist[edge_sources]
                - disps_hist[edge_targets])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_features
# =============================================================================
def forward_graph(
        model, disps, coords_ref, edges_indexes, n_dim,
        batch_vector=None, global_features_in=None,
        edge_feature_type=('edge_vector',),
        epbar_prev=None):
        """Forward pass through GNN model with graph data reconstruction.

        Args:
            model: Trained GNN model for material patch prediction.
            disps (Tensor): Nodal displacements of shape (n_nodes, n_dim).
            coords_ref (Tensor): Reference nodal coordinates of shape
                (n_nodes, n_dim).
            edges_indexes (Tensor): Edge connectivity of shape (2, n_edges).
            n_dim (int): Spatial dimension (2 or 3).
            batch_vector (Tensor, optional): Batch vector for multiple
                patches. Defaults to None.
            global_features_in (Tensor, optional): Global features.
                Defaults to None.

        Returns:
            Tensor: Predicted nodal forces of shape (n_nodes, n_dim).
        """
        coords = coords_ref + disps
        if epbar_prev is not None:
            node_features_in = torch.cat(
                [coords, disps, epbar_prev], dim=1)
        else:
            node_features_in = torch.cat(
                [coords, disps], dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove rigid body motions from input features
        # (must happen before normalization to match
        # training preprocessing)
        if getattr(model, '_is_rigid_body_removal', False):
            node_features_in = remove_rigid_body_motion(
                model, node_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Derive n_edge_in from user-specified features
        n_edge_in = len(edge_feature_type) * n_dim
        # Recompute edge features based on updated coords
        edge_features_in = compute_edge_features(
            coords, disps, edges_indexes, n_dim,
            n_edge_in=n_edge_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Forward pass with updated features
        # Use fast affine scalers if available
        _fast = getattr(model, '_fast_scalers_ready', False)
        # Stepwise forward mode
        if model._is_stepwise:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated node features
            if _fast:
                n_b = node_features_in.shape[-1]
                node_features_norm = (
                    node_features_in
                    * model._fs_node_feat_in_s[:n_b]
                    + model._fs_node_feat_in_o[:n_b])
            else:
                node_features_norm = (
                    model.stepwise_data_scaler_transform(
                        tensor=node_features_in,
                        features_type='node_features_in',
                        mode='normalize'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated edge features
            if _fast:
                n_b = edge_features_in.shape[-1]
                edge_features_norm = (
                    edge_features_in
                    * model._fs_edge_feat_in_s[:n_b]
                    + model._fs_edge_feat_in_o[:n_b])
            else:
                edge_features_norm = (
                    model.stepwise_data_scaler_transform(
                        tensor=edge_features_in,
                        features_type='edge_features_in',
                        mode='normalize'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Use stepwise forward method
            node_output_norm, _, _, hidden_states_out = \
                model.step(
                    node_features_in=node_features_norm,
                    edge_features_in=edge_features_norm,
                    global_features_in=global_features_in,
                    edges_indexes=edges_indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Denormalize output to get real forces
            if _fast:
                n_b = node_output_norm.shape[-1]
                node_output_real = (
                    node_output_norm
                    * model._fs_node_feat_out_is[:n_b]
                    + model._fs_node_feat_out_io[:n_b])
            else:
                node_output_real = (
                    model.stepwise_data_scaler_transform(
                        tensor=node_output_norm,
                        features_type='node_features_out',
                        mode='denormalize'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return node_output_real, hidden_states_out
        # Regular forward mode
        else:
            # Normalize updated node features
            if _fast:
                node_features_norm = (
                    node_features_in
                    * model._fs_node_feat_in_s
                    + model._fs_node_feat_in_o)
            else:
                node_features_norm = (
                    model.data_scaler_transform(
                        tensor=node_features_in,
                        features_type='node_features_in',
                        mode='normalize'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated edge features
            if _fast:
                edge_features_norm = (
                    edge_features_in
                    * model._fs_edge_feat_in_s
                    + model._fs_edge_feat_in_o)
            else:
                edge_features_norm = (
                    model.data_scaler_transform(
                        tensor=edge_features_in,
                        features_type='edge_features_in',
                        mode='normalize'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Use regular forward method
            node_output_norm, _, _ = model(
                node_features_in=node_features_norm,
                edge_features_in=edge_features_norm,
                global_features_in=global_features_in,
                edges_indexes=edges_indexes,
                batch_vector=batch_vector)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Denormalize output to get real forces
            if _fast:
                node_output_real = (
                    node_output_norm
                    * model._fs_node_feat_out_is
                    + model._fs_node_feat_out_io)
            else:
                node_output_real = (
                    model.data_scaler_transform(
                        tensor=node_output_norm,
                        features_type='node_features_out',
                        mode='denormalize'))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return node_output_real