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

import torch_geometric.data as pyg_data
from graphorge.gnn_base_model.model.gnn_model import GNNEPDBaseModel
from graphorge.gnn_base_model.data.graph_data import GraphData
from graphorge.projects.material_patches.gnn_model_tools.gen_graphs_files \
    import (get_elem_size_dims, get_mesh_connected_nodes)
from graphorge.projects.material_patches.gnn_model_tools.features import (
    GNNPatchFeaturesGenerator)
from graphorge.gnn_base_model.model.custom_layers import (
    compute_stiffness_matrix, extract_forces,
    extract_displacements, compute_edge_features,
    reconstruct_graph_with_displacements)
import torch.func as torch_func
import functools


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
    # ------------------------------------------------------------------------- 
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
        self, model, u: Tensor, n: int, du: Tensor,
        is_stepwise: bool = False,
        patch_ids: Tensor = None,
        hidden_states: dict = None,
        edge_feature_type: tuple = ('edge_vector',),
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update current displacement
        u[n] = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global force and stiffness assembly
        n_dof_global = self.n_nod * self.n_dim
        F_global = torch.zeros(n_dof_global)

        # Sparse stiffness assembly (COO format)
        k_indices = []
        k_values = []
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
                    return {k: detach_hidden_states(v) for k, v in states.items()}
                elif isinstance(states, list):
                    return [detach_hidden_states(item) for item in states]
                elif torch.is_tensor(states):
                    return states.detach()
                else:
                    return states
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Load patch-specific hidden states into model before inference
            if is_stepwise and hidden_states and patch_id in hidden_states:
                # patch_hidden = copy.deepcopy(hidden_states[patch_id])
                patch_hidden = copy.deepcopy(
                    detach_hidden_states(hidden_states[patch_id]))
                model._gnn_epd_model._hidden_states = patch_hidden
                # Set the model's hidden states to this patch's states
                if 'encoder' in patch_hidden:
                    model._gnn_epd_model._encoder._hidden_states = \
                        patch_hidden['encoder']
                if 'processor' in patch_hidden:
                    model._gnn_epd_model._processor._hidden_states = \
                        patch_hidden['processor']
                    for i, layer in enumerate(
                        model._gnn_epd_model._processor._processor):
                        layer_key = f'layer_{i}'
                        if layer_key in patch_hidden['processor']:
                            layer._hidden_states = \
                                patch_hidden['processor'][layer_key]
                if 'decoder' in patch_hidden:
                    model._gnn_epd_model._decoder._hidden_states = \
                        patch_hidden['decoder']
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
                """Forward function for gradient computation.

                Rescales inputs to [0,1]^d for GNN inference, then rescales
                forces back to physical space. jacfwd differentiates through
                the full chain so stiffness is automatically correct.
                """
                nonlocal hidden_states_trial
                # Rescale displacements by patch size
                disp_scaled = disp_boundary / L
                if is_stepwise:
                    pred_scaled, hidden_states_out = forward_graph(
                        model=model,
                        disps=disp_scaled,
                        coords_ref=coords_scaled,
                        edges_indexes=edges_indexes,
                        n_dim=self.n_dim,
                        edge_feature_type=edge_feature_type)
                    hidden_states_trial = hidden_states_out
                else:
                    pred_scaled = forward_graph(
                        model=model,
                        disps=disp_scaled,
                        coords_ref=coords_scaled,
                        edges_indexes=edges_indexes,
                        n_dim=self.n_dim,
                        edge_feature_type=edge_feature_type)
                # GNN predicts forces in scaled space; unscale to physical
                pred_real = pred_scaled * L
                return pred_real, pred_real.detach()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute Jacobian: d(forces_boundary)/d(u_boundary)
            (jacobian, boundary_forces) = torch_func.jacfwd(
                forward, has_aux=True)(boundary_u_current)
            # Stiffness shape: (n_boundary*dim, n_boundary*dim)
            k_boundary = jacobian.view(n_dof_boundary, n_dof_boundary)
            f_boundary = boundary_forces.flatten()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # OLD: Compute Jacobian for all element nodes
            # (jacobian, node_output) = torch_func.jacfwd(
            #     forward, has_aux=True)(elem_u_current)
            # node_forces_trial = node_output.flatten()
            # # Reshape Jacobian to stiffness matrix format
            # stiffness_matrix = jacobian.view(n_dof_elem, n_dof_elem)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble boundary forces to global force vector
            for local_i, global_node in enumerate(boundary_node_ids):
                for d in range(self.n_dim):
                    global_dof = global_node * self.n_dim + d
                    local_dof = local_i * self.n_dim + d
                    F_global[global_dof] += f_boundary[local_dof]

            # Collect boundary stiffness entries for sparse assembly
            for local_i, global_node_i in enumerate(boundary_node_ids):
                for d_i in range(self.n_dim):
                    global_dof_i = global_node_i * self.n_dim + d_i
                    local_dof_i = local_i * self.n_dim + d_i

                    for local_j, global_node_j in enumerate(
                            boundary_node_ids):
                        for d_j in range(self.n_dim):
                            global_dof_j = global_node_j * self.n_dim + d_j
                            local_dof_j = local_j * self.n_dim + d_j

                            k_indices.append(
                                [global_dof_i, global_dof_j])
                            k_values.append(
                                k_boundary[local_dof_i, local_dof_j])

            # Store updated hidden states for stepwise mode
            if is_stepwise and hidden_states_trial is not None:
                hidden_states_out[patch_id] = hidden_states_trial
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # OLD: Store element forces and stiffness
            # f[idx_patch] = node_forces_trial.flatten()
            # k[idx_patch] = stiffness_matrix
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build global sparse stiffness matrix
        k_indices_tensor = torch.tensor(k_indices, dtype=torch.long).T
        k_values_tensor = torch.tensor(k_values)
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
            return K_global, F_global, hidden_states_out
        else:
            return K_global, F_global
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
        model_directory: str | None = None,
        return_resnorm: bool = False,
        patch_boundary_nodes: dict | None = None,
        patch_elem_per_dim: list | None = None,
        edge_type: str = 'all',
        edge_feature_type: tuple = ('edge_vector',),
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
            raise ValueError(f'is_mat_patch shape {is_mat_patch.shape} ' + \
                             f'must match number of elements {self.n_elem}')
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
        # Initialize material patch data structures for surrogate integration
        patch_mask = is_mat_patch >= 0
        model = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load surrogate model
        if torch.any(patch_mask):
            # Use provided model directory or default path
            if model_directory is None:
                model_directory = (
                    "/Users/rbarreira/Desktop/machine_learning/material_patches/"
                    "graphorge_material_patches/src/graphorge/projects/"
                    "material_patches/elastic/2d/quad4/mesh1x1/ninc1/"
                    "26.1_force_equilibrium_npath10000/reference/3_model")
            model = self._load_Graphorge_model(
                model_directory=model_directory,
                device_type=device if device else 'cpu')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Enable stepwise mode
            # Temporarily set attributes for single-step mode (_n_time_* = 1)
            if is_stepwise:
                # Store original attributes for stepwise data scaler access
                # orig_attrs = 
                model._save_time_series_attrs()
                # model._original_time_attrs = orig_attrs
                # Enable stepwise mode
                model.set_rnn_mode(is_stepwise=True)
            
            # Initialize patch_id dictionary with None values
            patch_ids = torch.unique(is_mat_patch[is_mat_patch >= 0])
            # Initialize edges_indexes dict and patch boundary nodes dict
            self._edges_indexes = {}
            self.patch_bd_nodes = {}
            # Populate patch boundary nodes from input parameter
            if patch_boundary_nodes is not None:
                self.patch_bd_nodes = patch_boundary_nodes
            breakpoint()
            # Initialize hidden states dict before loop to accumulate all patches
            hidden_states_dict = {}
            for pid in patch_ids:
                pid_int = pid.item()
                # Build graph topology for this material patch
                self._build_graph_topology(
                    pid_int, patch_elem_per_dim,
                    edge_type=edge_type)
                # Initialize hidden states structure for GNN model
                # Same structure as graphorge:
                # - encoder: for encoding layers
                # - processor: for message passing layers (layer_0, layer_1, etc.)
                # - decoder: for decoding layers
                # Each layer can have node, edge, global hidden states
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get number of message passing steps from model if available
                n_message_steps = model._n_message_steps
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Create processor hidden states for each layer
                processor_hidden = {}
                for layer_idx in range(n_message_steps):
                    processor_hidden[f'layer_{layer_idx}'] = {
                        'node': None,
                        'edge': None,
                        'global': None
                    }
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Accumulate hidden states for each patch
                hidden_states_dict[f"patch_{pid_int}"] = {
                    'encoder': {
                        'node': None,
                        'edge': None,
                        'global': None
                    },
                    'processor': processor_hidden,
                    'decoder': {
                        'node': None,
                        'edge': None,
                        'global': None
                    }
                }
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
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # # OLD: Element-wise integration with material patch support
                # # Initialize element stiffness and force arrays
                # n_nod = self.etype.nodes
                # k = torch.zeros((self.n_elem, self.n_dim * n_nod,
                #                  self.n_dim * n_nod))
                # f_i = torch.zeros(self.n_elem, self.n_dim * n_nod)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Surrogate integration returns global K and F
                patch_mask = is_mat_patch >= 0
                if torch.any(patch_mask):
                    # Get unique patch IDs and process all at once
                    patch_ids = torch.unique(is_mat_patch[patch_mask])
                    # Call surrogate integration (returns global K and F)
                    if is_stepwise:
                        K_raw, F_int, hidden_state_out = \
                            self.surrogate_integrate_material(
                            model, u, n, du,
                            is_stepwise=is_stepwise,
                            patch_ids=patch_ids,
                            hidden_states=hidden_states_dict,
                            edge_feature_type=edge_feature_type)
                    else:
                        K_raw, F_int = \
                            self.surrogate_integrate_material(
                            model, u, n, du,
                            is_stepwise=is_stepwise,
                            patch_ids=patch_ids,
                            edge_feature_type=edge_feature_type)
                    # Apply constraint elimination
                    self.K = \
                        self._apply_constraints_sparse(
                            K_raw, con)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # # OLD: Assemble results for all patches
                # k[patch_mask] = k_surr[patch_mask]
                # f_i[patch_mask] = f_surr[patch_mask]
                # # OLD: Assemble global stiffness matrix and internal force
                # if self.K.numel() == 0 or not self.material.n_state == 0 or \
                #     nlgeom:
                #     self.K = self.assemble_stiffness(k, con)
                # F_int = self.assemble_force(f_i)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Print iteration information
                if verbose:
                    print(f"Increment {n} | Iteration {i+1} | "
                          f"Residual: {res_norm:.5e}")
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    # Update patch-specific hidden states with converged values
                    if is_stepwise:
                        # Update patch-specific hidden states 
                        # with converged values
                        # hidden_state_out contains the updated states for 
                        # processed patches
                        for pid in patch_ids:
                            patch_key = f"patch_{pid.item()}"
                            # Update the patch-specific hidden states
                            hidden_states_dict[patch_key] = hidden_state_out[
                                patch_key]
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
                    None,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Raise an Exception if the model did not converge
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Disable stepwise mode after computation
        if is_stepwise:
            model.set_rnn_mode(is_stepwise=False)
            model._restore_time_series_attrs()
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
        current_coords_hist = coords_hist
        for k in range(n_edges):
            i = edge_sources[k]
            j = edge_targets[k]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # edge_vector: coords[i] - coords[j]
            for t in range(n_time_steps):
                s = t * n_dim
                e = (t + 1) * n_dim
                edge_features[k, s:e] = (
                    current_coords_hist[i, s:e]
                    - current_coords_hist[j, s:e])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # relative_disp: disps[i] - disps[j]
            if include_rel_disp:
                for t in range(n_time_steps):
                    s = t * n_dim
                    e = (t + 1) * n_dim
                    offset = edge_vector_size + s
                    edge_features[k, offset:offset
                                  + n_dim] = (
                        disps_hist[i, s:e]
                        - disps_hist[j, s:e])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_features
# =============================================================================
def forward_graph(
        model, disps, coords_ref, edges_indexes, n_dim,
        batch_vector=None, global_features_in=None,
        edge_feature_type=('edge_vector',)):
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
        node_features_in = torch.cat([coords, disps], dim=1)
        # Derive n_edge_in from user-specified features
        n_edge_in = len(edge_feature_type) * n_dim
        # Recompute edge features based on updated coords
        edge_features_in = compute_edge_features(
            coords, disps, edges_indexes, n_dim,
            n_edge_in=n_edge_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Forward pass with updated features
        # Stepwise forward mode
        if model._is_stepwise:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated node features
            node_features_norm = model.stepwise_data_scaler_transform(
                tensor=node_features_in,
                features_type='node_features_in',
                mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated edge features
            edge_features_norm = model.stepwise_data_scaler_transform(
                tensor=edge_features_in,
                features_type='edge_features_in',
                mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Use stepwise forward method
            node_output_norm, _, _, hidden_states_out = model.step(
                node_features_in=node_features_norm,
                edge_features_in=edge_features_norm,
                global_features_in=global_features_in,
                edges_indexes=edges_indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Denormalize output to get real forces
            node_output_real = model.stepwise_data_scaler_transform(
                tensor=node_output_norm,
                features_type='node_features_out',
                mode='denormalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return node_output_real, hidden_states_out
        # Regular forward mode
        else:
            # Normalize updated node features
            node_features_norm = model.data_scaler_transform(
                tensor=node_features_in,
                features_type='node_features_in',
                mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated edge features
            edge_features_norm = model.data_scaler_transform(
                tensor=edge_features_in,
                features_type='edge_features_in',
                mode='normalize')
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
            node_output_real = model.data_scaler_transform(
                tensor=node_output_norm,
                features_type='node_features_out',
                mode='denormalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return node_output_real