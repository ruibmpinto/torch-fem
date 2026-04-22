"""Run FEM simulation and boundary stiffness extraction.

Supports both regular and irregular material patches. Computes
boundary-DOF stiffness matrices via static condensation and
automatic differentiation for comparison.

Classes
-------
Simulation
    Material patch finite element simulation.

Functions
---------
run_simulation
    Convenience wrapper for backward compatibility.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import pathlib
import pickle as pkl
# Third-party
import numpy as np
import torch
import torch.func as torch_func
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add graphorge to sys.path
graphorge_path = str(
    pathlib.Path(__file__).parents[2] /
    "graphorge_material_patches" / "src")
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Local
from torchfem import Solid, Planar
from torchfem.materials import (
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
    IsotropicPlasticityPlaneStrain,
    IsotropicPlasticityPlaneStress,
    IsotropicElasticity3D,
    IsotropicPlasticity3D,
    IsotropicHencky3D,
    Hyperelastic3D,
    IsotropicHenckyPlaneStrain
)
from torchfem.mesh import cube_hexa, rect_quad
from torchfem.elements import (
    linear_to_quadratic, Hexa1r, Quad1r
)
from utils.boundary_conditons import (
    prescribe_disps_by_coords
)
from utils.mechanical_quantities import (
    compute_strain_energy_density
)
from utils.plotting import (
    plot_displacement_field,
    plot_domain_displacements,
    plot_shape_functions
)
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.set_default_dtype(torch.float64)


# =============================================================================
def extrapolate_gauss_to_nodes(
        etype, state_gp, elements, n_nod):
    """Extrapolate Gauss-point scalar field to nodes.

    Uses shape-function-based superconvergent
    extrapolation: E = N(nodes) @ pinv(N(gauss)).
    Works for all element types regardless of whether
    n_gauss == n_nodes.

    Parameters
    ----------
    etype : Element
        Element type instance.
    state_gp : torch.Tensor
        Gauss-point values, shape (n_gp, n_elem).
    elements : torch.Tensor
        Element connectivity, shape (n_elem, n_nodes).
    n_nod : int
        Total number of nodes in the mesh.

    Returns
    -------
    nodal_values : torch.Tensor
        Shape (n_nod,). Contribution-averaged nodal
        values.
    """
    # Evaluate shape functions at Gauss points
    # and at node natural coordinates
    N_gp = etype.N(etype.ipoints())    # (n_gp, n_en)
    N_nd = etype.N(etype.npoints())    # (n_en, n_en)
    # Extrapolation matrix: (n_en, n_gp)
    E = N_nd @ torch.linalg.pinv(N_gp)
    # Extrapolate: (n_en, n_gp) @ (n_gp, n_elem)
    # -> (n_en, n_elem)
    nodal_elem = E @ state_gp
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Scatter-add to global nodal array
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
    # Average contributions from shared nodes
    nodal_count = torch.clamp(nodal_count, min=1.0)
    return nodal_sum / nodal_count
# =============================================================================
def _assemble_raw_stiffness(domain):
    """Assemble global stiffness without constraint elimination.

    Initialises state arrays, calls integrate_material once
    with zero displacements, and scatters element stiffness
    into a dense global matrix.

    Parameters
    ----------
    domain : FEM
        torchfem domain (Planar or Solid).

    Returns
    -------
    K : torch.Tensor
        Dense global stiffness, shape (n_dofs, n_dofs).
    """
    n_dofs = domain.n_dofs
    n_stress = domain.n_stress
    n_int = domain.n_int
    n_elem = domain.n_elem
    n_state = domain.material.n_state
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Ensure K cache is initialised (integrate_material
    # checks self.K.numel() to decide recomputation)
    domain.K = torch.empty(0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # State arrays matching integrate_material signature
    N = 2
    u = torch.zeros(N, domain.n_nod, domain.n_dim)
    stress = torch.zeros(
        N, n_int, n_elem, n_stress, n_stress)
    defgrad = torch.zeros(
        N, n_int, n_elem, n_stress, n_stress)
    defgrad[:, :, :, :, :] = torch.eye(n_stress)
    state = torch.zeros(N, n_int, n_elem, n_state)
    du = torch.zeros(n_dofs)
    de0 = torch.zeros(n_elem, n_stress, n_stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    k_elem, _ = domain.integrate_material(
        u, defgrad, stress, state,
        n=1, du=du, de0=de0, nlgeom=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Scatter into dense global matrix
    K = torch.zeros(n_dofs, n_dofs)
    idx = domain.idx.long()
    for e in range(n_elem):
        dof_e = idx[e]
        K[dof_e.unsqueeze(1), dof_e.unsqueeze(0)] += (
            k_elem[e])
    return K


# =============================================================================
def _static_condensation(K, bd_dofs, int_dofs):
    """Condense interior DOFs out of global stiffness.

    Computes K_cond = K_bb - K_bi @ K_ii^{-1} @ K_ib.

    Parameters
    ----------
    K : torch.Tensor
        Full dense stiffness, shape (n_dofs, n_dofs).
    bd_dofs : torch.LongTensor
        Boundary DOF indices.
    int_dofs : torch.LongTensor
        Interior DOF indices.

    Returns
    -------
    K_cond : torch.Tensor
        Condensed boundary stiffness,
        shape (n_bd_dofs, n_bd_dofs).
    """
    K_bb = K[bd_dofs][:, bd_dofs]
    if len(int_dofs) == 0:
        return K_bb
    K_bi = K[bd_dofs][:, int_dofs]
    K_ib = K[int_dofs][:, bd_dofs]
    K_ii = K[int_dofs][:, int_dofs]
    K_cond = K_bb - K_bi @ torch.linalg.solve(K_ii, K_ib)
    return K_cond


# =============================================================================
def _stiffness_ad(K_full, bd_dofs, int_dofs):
    """Compute boundary stiffness via AD through solve.

    Differentiates f_b(u_b) where interior DOFs are
    eliminated by solving K_ii @ u_i = -K_ib @ u_b,
    then f_b = K_bb @ u_b + K_bi @ u_i.

    Parameters
    ----------
    K_full : torch.Tensor
        Full dense stiffness, shape (n_dofs, n_dofs).
    bd_dofs : torch.LongTensor
        Boundary DOF indices.
    int_dofs : torch.LongTensor
        Interior DOF indices.

    Returns
    -------
    K_ad : torch.Tensor
        Boundary stiffness from AD,
        shape (n_bd_dofs, n_bd_dofs).
    """
    K_bb = K_full[bd_dofs][:, bd_dofs]
    if len(int_dofs) == 0:
        return K_bb
    K_bi = K_full[bd_dofs][:, int_dofs]
    K_ib = K_full[int_dofs][:, bd_dofs]
    K_ii = K_full[int_dofs][:, int_dofs]

    def boundary_forces(u_b):
        """Boundary forces given boundary displacements."""
        rhs = -K_ib @ u_b
        u_i = torch.linalg.solve(K_ii, rhs)
        f_b = K_bb @ u_b + K_bi @ u_i
        return f_b

    n_bd = len(bd_dofs)
    K_ad = torch_func.jacfwd(boundary_forces)(
        torch.zeros(n_bd))
    return K_ad


# =============================================================================
class Simulation:
    """Material patch finite element simulation.

    Manages FEM simulation workflow for material patches
    including mesh generation, material definition, boundary
    condition application, system solving, result
    post-processing, and boundary stiffness extraction.

    Attributes
    ----------
    _label_to_idx : dict
        Mapping from 1-based patch node labels to 0-based
        torchfem node indices.
    _boundary_node_indices : list[int]
        0-based node indices of boundary nodes.
    _boundary_dofs : torch.LongTensor
        Global DOF indices of boundary nodes.
    _interior_dofs : torch.LongTensor
        Global DOF indices of interior nodes.

    Methods
    -------
    run()
        Execute complete simulation workflow.
    compute_boundary_stiffness()
        Compute boundary stiffness via static condensation
        and AD.
    """

    def __init__(
        self,
        element_type='quad4',
        material_behavior='elastic',
        patch_idx=0,
        num_increments=1,
        mesh_nx=3,
        mesh_ny=3,
        mesh_nz=3,
        is_red_int=False,
        is_save=False,
        is_compute_stiffness=False,
        is_save_avg_epbar=False,
        is_save_nodal_epbar=False,
        is_adaptive_timestepping=True,
        adaptive_max_subdiv=8,
        filepath='/Users/rbarreira/Desktop/'
                 'machine_learning/'
                 'material_patches/_data/'
    ):
        """Initialize simulation parameters.

        Parameters
        ----------
        element_type : str, default='quad4'
            Element type ('quad4', 'hex8', etc.)
        material_behavior : str, default='elastic'
            Material model ('elastic',
            'elastoplastic_nlh', etc.)
        patch_idx : int, default=0
            Material patch index.
        num_increments : int, default=1
            Number of load increments.
        mesh_nx : int, default=3
            Elements in x-direction.
        mesh_ny : int, default=3
            Elements in y-direction.
        mesh_nz : int, default=3
            Elements in z-direction (3D only).
        is_red_int : bool, default=False
            Use reduced integration.
        is_save : bool, default=False
            Save simulation output to disk.
        is_compute_stiffness : bool, default=False
            Compute and save boundary stiffness matrices.
        is_save_avg_epbar : bool, default=False
            Save volume-averaged equivalent plastic strain
            per element in output pickle.
        is_save_nodal_epbar : bool, default=False
            Save per-node equivalent plastic strain time
            series in output pickle (extrapolated from
            Gauss points).
        is_adaptive_timestepping : bool, default=True
            Enable adaptive sub-incrementation with
            retry-and-downsample on Newton-Raphson
            failure. If True, on convergence failure
            `run()` re-solves with 2x, 4x, ... refinement
            of the load factor sequence (up to
            `adaptive_max_subdiv`) and downsamples solver
            outputs back to the original time points. If
            False, a single solve attempt is made with
            the base increments; convergence failures
            propagate as exceptions.
        adaptive_max_subdiv : int, default=8
            Maximum subdivision factor used when
            `is_adaptive_timestepping` is True. The retry
            sequence is the powers of 2 in
            [1, adaptive_max_subdiv] (e.g. 8 gives
            1, 2, 4, 8; 16 gives 1, 2, 4, 8, 16; 1
            disables refinement). Must be an integer
            >= 1. Ignored when
            `is_adaptive_timestepping` is False.
        filepath : str
            Base path to material patch input data.
        """
        self.element_type = element_type
        self.material_behavior = material_behavior
        self.patch_idx = patch_idx
        self.num_increments = num_increments
        self.mesh_nx = mesh_nx
        self.mesh_ny = mesh_ny
        self.mesh_nz = mesh_nz
        self.is_red_int = is_red_int
        self.is_save = is_save
        self.is_compute_stiffness = is_compute_stiffness
        self.is_save_avg_epbar = is_save_avg_epbar
        self.is_save_nodal_epbar = is_save_nodal_epbar
        self.is_adaptive_timestepping = (
            is_adaptive_timestepping)
        if adaptive_max_subdiv < 1:
            raise ValueError(
                f'adaptive_max_subdiv must be >= 1, '
                f'got {adaptive_max_subdiv}.')
        self.adaptive_max_subdiv = adaptive_max_subdiv
        self.filepath = filepath
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine element order and dimension
        self._element_properties()
        # Setup file paths
        self._setup_paths()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data structures
        self.matpatch = None
        self.domain = None
        self.simulation_data = None
        self.nodes_constrained = None
        self._label_to_idx = None
        self._boundary_node_indices = None
        self._boundary_dofs = None
        self._interior_dofs = None

    # -------------------------------------------------------------------------
    def _element_properties(self):
        """Determine element order and problem dimension."""
        if self.element_type in [
                'quad4', 'tri3', 'tetra4', 'hex8']:
            self.elem_order = 1
        elif self.element_type in [
                'quad8', 'tri6', 'tetra10', 'hex20']:
            self.elem_order = 2
        if self.element_type in [
                'quad4', 'tri3', 'quad8', 'tri6']:
            self.dim = 2
        elif self.element_type in [
                'tetra4', 'hex8', 'tetra10', 'hex20']:
            self.dim = 3

    # -------------------------------------------------------------------------
    def _setup_paths(self):
        """Setup input and output file paths."""
        out_filepath = (
            '/Users/rbarreira/Desktop/machine_learning/'
            'material_patches/')
        is_irregular = 'irregular' in self.filepath
        if self.dim == 2:
            if is_irregular:
                self.dir_path = (
                    f'{out_filepath}_data/'
                    f'{self.material_behavior}/'
                    f'{self.dim}d/{self.element_type}/'
                    f'irregular/'
                    f'mesh{self.mesh_nx}x{self.mesh_ny}/'
                    f'ninc{self.num_increments}/')
            else:
                self.dir_path = (
                    f'{out_filepath}_data/'
                    f'{self.material_behavior}/'
                    f'{self.dim}d/{self.element_type}/'
                    f'mesh{self.mesh_nx}x{self.mesh_ny}/'
                    f'ninc{self.num_increments}/')
            self.input_filename = (
                f'{self.filepath}'
                f'material_patches_generation_'
                f'{self.dim}d_{self.element_type}_mesh_'
                f'{self.mesh_nx}x{self.mesh_ny}/'
                f'material_patch_{self.patch_idx}/'
                f'material_patch/'
                f'material_patch_attributes.pkl')
        elif self.dim == 3:
            if self.element_type not in [
                    'tetra4', 'hex8', 'tetra10', 'hex20']:
                raise ValueError(
                    f'Invalid element type for '
                    f'{self.dim}d problem!')
            self.dir_path = (
                f'{out_filepath}_data/'
                f'{self.material_behavior}/'
                f'{self.dim}d/{self.element_type}/'
                f'mesh{self.mesh_nx}x{self.mesh_ny}'
                f'x{self.mesh_nz}/'
                f'ninc{self.num_increments}/')
            self.input_filename = (
                f'{self.filepath}'
                f'material_patches_generation_'
                f'{self.dim}d_{self.element_type}_'
                f'mesh{self.mesh_nx}x{self.mesh_ny}'
                f'x{self.mesh_nz}/'
                f'material_patch_{self.patch_idx}/'
                f'material_patch/'
                f'material_patch_attributes.pkl')
        self.output_filename = (
            f'{self.dir_path}'
            f'matpatch_idx{self.patch_idx}.pkl')
        os.makedirs(self.dir_path, exist_ok=True)

    # -------------------------------------------------------------------------
    def load_material_patch(self):
        """Load material patch data from file."""
        with open(self.input_filename, 'rb') as file:
            self.matpatch = pkl.load(file)

    # -------------------------------------------------------------------------
    def create_material(self):
        """Create material model based on material_behavior.

        Returns
        -------
        material : Material
            Material model instance.
        """
        if self.material_behavior == 'elastic':
            e_young = 110000
            nu = 0.33
            if self.dim == 2:
                return IsotropicElasticityPlaneStrain(
                    E=e_young, nu=nu)
            elif self.dim == 3:
                return IsotropicElasticity3D(
                    E=e_young, nu=nu)
        elif self.material_behavior == 'hyperelastic':
            e_young = 20000
            nu = 0.33
            lmbda = (e_young * nu
                     / ((1. + nu) * (1. - 2. * nu)))
            mu = e_young / (2. * (1. + nu))

            def psi(F):
                """Neo-Hookean strain energy density."""
                C = F.transpose(-1, -2) @ F
                logJ = 0.5 * torch.logdet(C)
                return (
                    mu / 2 * (torch.trace(C) - 3.0)
                    - mu * logJ
                    + lmbda / 2 * logJ**2)

            if self.dim == 2:
                return IsotropicHenckyPlaneStrain(
                    E=e_young, nu=nu)
            elif self.dim == 3:
                return Hyperelastic3D(psi)
        elif self.material_behavior == 'elastoplastic_lh':
            e_young = 70000
            nu = 0.33
            sigma_y = 100.0
            hardening_modulus = 100.0

            def sigma_f(eps_pl):
                """Linear hardening function."""
                return sigma_y + hardening_modulus * eps_pl

            def sigma_f_prime(_eps_pl):
                """Derivative of linear hardening."""
                return hardening_modulus

            if self.dim == 2:
                return IsotropicPlasticityPlaneStrain(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)
            elif self.dim == 3:
                return IsotropicPlasticity3D(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)
        elif self.material_behavior == 'elastoplastic_nlh':
            e_young = 70000
            nu = 0.33
            # Swift-Voce hardening parameters for AA2024
            a_s = 798.56
            epsilon_0 = 0.0178
            n = 0.202
            k_0 = 363.84
            q_v = 240.03
            beta = 10.533
            omega = 0.368

            def k_s(eps_pl):
                """Swift hardening component."""
                return a_s * (epsilon_0 + eps_pl)**n

            def k_v(eps_pl):
                """Voce hardening component."""
                return (k_0 + q_v
                        * (1.0 - torch.exp(
                            -beta * eps_pl)))

            def sigma_f(eps_pl):
                """Combined Swift-Voce hardening."""
                return (omega * k_s(eps_pl)
                        + (1.0 - omega) * k_v(eps_pl))

            def sigma_f_prime(eps_pl):
                """Derivative of Swift-Voce hardening."""
                dks = (a_s * n
                       * (epsilon_0 + eps_pl)**(n - 1.0))
                dkv = (q_v * beta
                       * torch.exp(-beta * eps_pl))
                return (omega * dks
                        + (1.0 - omega) * dkv)

            if self.dim == 2:
                return IsotropicPlasticityPlaneStrain(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)
            elif self.dim == 3:
                return IsotropicPlasticity3D(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)
        else:
            raise ValueError(
                f'Unknown material behavior: '
                f'{self.material_behavior}')

    # -------------------------------------------------------------------------
    def _build_mesh_from_patch(self, material):
        """Build FE mesh from patch pkl data.

        Derives node coordinates and element connectivity
        from mesh_nodes_coords_ref and mesh_nodes_matrix.
        Works for both regular and irregular patches.

        Parameters
        ----------
        material : Material
            Material model instance.

        Returns
        -------
        domain : FEM
            torchfem domain (Planar or Solid).
        """
        coords_ref = self.matpatch['mesh_nodes_coords_ref']
        mnm = self.matpatch['mesh_nodes_matrix']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build label-to-index mapping and node tensor
        sorted_labels = sorted(
            coords_ref.keys(), key=int)
        label_to_idx = {
            int(lbl): idx
            for idx, lbl in enumerate(sorted_labels)}
        self._label_to_idx = label_to_idx
        nodes = torch.tensor(np.array(
            [coords_ref[lbl] for lbl in sorted_labels]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Derive element connectivity from mesh_nodes_matrix
        nx, ny = (mnm.shape[0] - 1, mnm.shape[1] - 1)
        elements = []
        for i in range(nx):
            for j in range(ny):
                n0 = label_to_idx[int(mnm[i, j])]
                n1 = label_to_idx[int(mnm[i + 1, j])]
                n3 = label_to_idx[
                    int(mnm[i + 1, j + 1])]
                n2 = label_to_idx[
                    int(mnm[i, j + 1])]
                elements.append([n0, n1, n3, n2])
        elements = torch.tensor(
            elements, dtype=torch.long)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.dim == 2:
            domain = Planar(nodes, elements, material)
            if (self.is_red_int
                    and self.element_type == 'quad4'):
                domain.etype = Quad1r()
                domain.n_int = len(
                    domain.etype.iweights())
            return domain
        elif self.dim == 3:
            domain = Solid(nodes, elements, material)
            if (self.is_red_int
                    and self.element_type == 'hex8'):
                domain.etype = Hexa1r()
                domain.n_int = len(
                    domain.etype.iweights())
            return domain

    # -------------------------------------------------------------------------
    def create_mesh(self, material):
        """Create finite element mesh and domain.

        Uses patch pkl data when available (both regular
        and irregular). Falls back to rect_quad/cube_hexa
        when patch data is not loaded.

        Parameters
        ----------
        material : Material
            Material model instance.

        Returns
        -------
        domain : FEM
            torchfem domain (Planar or Solid).
        """
        if (self.matpatch is not None
                and 'mesh_nodes_matrix' in self.matpatch
                and 'mesh_nodes_coords_ref'
                in self.matpatch):
            return self._build_mesh_from_patch(material)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fallback: generate regular mesh
        if self.dim == 2:
            nodes, elements = rect_quad(
                self.mesh_nx + 1, self.mesh_ny + 1)
            if self.elem_order == 2:
                nodes, elements = \
                    linear_to_quadratic(
                        nodes, elements)
            domain = Planar(nodes, elements, material)
            if (self.is_red_int
                    and self.element_type == 'quad4'):
                domain.etype = Quad1r()
                domain.n_int = len(
                    domain.etype.iweights())
            return domain
        elif self.dim == 3:
            nodes, elements = cube_hexa(
                self.mesh_nx + 1,
                self.mesh_ny + 1,
                self.mesh_nz + 1)
            if self.elem_order == 2:
                nodes, elements = \
                    linear_to_quadratic(
                        nodes, elements)
            domain = Solid(nodes, elements, material)
            if (self.is_red_int
                    and self.element_type == 'hex8'):
                domain.etype = Hexa1r()
                domain.n_int = len(
                    domain.etype.iweights())
            return domain

    # -------------------------------------------------------------------------
    def _identify_boundary_interior_dofs(self):
        """Identify boundary and interior DOF indices.

        Boundary nodes are those with entries in
        mesh_boundary_nodes_disps. All other nodes are
        interior.
        """
        bd_disps = self.matpatch[
            'mesh_boundary_nodes_disps']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Map boundary labels to 0-based node indices
        bd_node_indices = []
        for lbl in bd_disps.keys():
            if self._label_to_idx is not None:
                idx = self._label_to_idx[int(lbl)]
            else:
                idx = self.node_label_to_torchfem_idx[
                    int(lbl)]
            bd_node_indices.append(idx)
        bd_node_indices = sorted(bd_node_indices)
        self._boundary_node_indices = bd_node_indices
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build DOF index arrays
        bd_dofs = []
        for node_idx in bd_node_indices:
            for d in range(self.dim):
                bd_dofs.append(
                    node_idx * self.dim + d)
        all_dofs = set(range(self.domain.n_dofs))
        int_dofs = sorted(all_dofs - set(bd_dofs))
        self._boundary_dofs = torch.tensor(
            bd_dofs, dtype=torch.long)
        self._interior_dofs = torch.tensor(
            int_dofs, dtype=torch.long)

    # -------------------------------------------------------------------------
    def compute_boundary_stiffness(self):
        """Compute boundary stiffness via both methods.

        Assembles the raw global stiffness matrix, then
        extracts the boundary-DOF sub-matrix via:
        1. Static condensation
        2. Automatic differentiation through the solve

        Stores results in self.simulation_data and prints
        comparison metrics.
        """
        self._identify_boundary_interior_dofs()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble raw global K
        K_full = _assemble_raw_stiffness(self.domain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Method 1: static condensation
        K_sc = _static_condensation(
            K_full, self._boundary_dofs,
            self._interior_dofs)
        # Method 2: AD through solve
        K_ad = _stiffness_ad(
            K_full, self._boundary_dofs,
            self._interior_dofs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Comparison metrics
        norm_sc = torch.linalg.norm(K_sc)
        diff_norm = torch.linalg.norm(K_sc - K_ad)
        rel_error = (diff_norm / norm_sc).item()
        sym_error_sc = (
            torch.linalg.norm(K_sc - K_sc.T)
            / norm_sc).item()
        sym_error_ad = (
            torch.linalg.norm(K_ad - K_ad.T)
            / torch.linalg.norm(K_ad)).item()
        n_bd = len(self._boundary_dofs)
        n_int = len(self._interior_dofs)
        # print(
        #     f'  Boundary stiffness: '
        #     f'{n_bd}x{n_bd} '
        #     f'({n_bd // self.dim} bd nodes, '
        #     f'{n_int // self.dim} int nodes)')
        # print(
        #     f'  SC vs AD rel error: '
        #     f'{rel_error:.2e}')
        # print(
        #     f'  Symmetry SC: {sym_error_sc:.2e}, '
        #     f'AD: {sym_error_ad:.2e}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store results
        if self.simulation_data is None:
            self.simulation_data = {}
        if rel_error < 1e-10:
            self.simulation_data['K_boundary'] = (
                K_sc.detach().numpy())
        else:
            self.simulation_data[
                'K_boundary_static_cond'] = (
                K_sc.detach().numpy())
            self.simulation_data[
                'K_boundary_ad'] = (
                K_ad.detach().numpy())
        self.simulation_data[
            'boundary_node_indices'] = (
            self._boundary_node_indices)
        self.simulation_data[
            'boundary_dofs'] = (
            self._boundary_dofs.numpy())
        self.simulation_data[
            'interior_dofs'] = (
            self._interior_dofs.numpy())

    # -------------------------------------------------------------------------
    def initialize_data_structures(self):
        """Initialize simulation data storage structures."""
        self.simulation_data = {
            'bd_nodes_coords': {},
            'bd_nodes_disps_time_series': {},
            'bd_nodes_forces_time_series': {},
            'stress_avg': {},
            'strain_energy_density': {},
        }
        if (self.material_behavior in [
                'elastoplastic_lh',
                'elastoplastic_nlh']):
            if self.is_save_avg_epbar:
                self.simulation_data[
                    'epsilon_pl_eq'] = {}
            if self.is_save_nodal_epbar:
                self.simulation_data[
                    'bd_nodes_epbar_time_series'] = {}
        self._build_node_mapping()
        self._initialize_boundary_nodes()
        self._copy_data()

    # -------------------------------------------------------------------------
    def _build_node_mapping(self):
        """Build mapping from patch labels to mesh indices.

        When mesh was built from patch data,
        _label_to_idx is already set. Otherwise falls back
        to coordinate-distance matching.
        """
        if self._label_to_idx is not None:
            self.node_label_to_torchfem_idx = (
                self._label_to_idx)
            return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fallback: coordinate matching
        self.node_label_to_torchfem_idx = {}
        nodes = self.domain.nodes
        for node_label in (
                self.matpatch[
                    'mesh_nodes_coords_ref'].keys()):
            node_coords_matp = self.matpatch[
                'mesh_nodes_coords_ref'][node_label]
            ref_point = torch.tensor(node_coords_matp)
            distances = torch.sqrt(
                torch.sum(
                    (nodes - ref_point)**2, axis=1))
            closest_idx = torch.argmin(distances).item()
            if distances[closest_idx] >= 1e-6:
                break
            self.node_label_to_torchfem_idx[
                int(node_label)] = closest_idx

    # -------------------------------------------------------------------------
    def _initialize_boundary_nodes(self):
        """Initialize boundary node data structures."""
        for node_label in (
                self.matpatch[
                    'mesh_boundary_nodes_disps'].keys()):
            closest_idx = (
                self.node_label_to_torchfem_idx[
                    int(node_label)])
            self.simulation_data[
                'bd_nodes_coords'][closest_idx] = (
                self.matpatch[
                    'mesh_nodes_coords_ref'][node_label])
            self.simulation_data[
                'bd_nodes_disps_time_series'][
                closest_idx] = []
            self.simulation_data[
                'bd_nodes_forces_time_series'][
                closest_idx] = []

    # -------------------------------------------------------------------------
    def _copy_data(self):
        """Copy material patch data to simulation_data."""
        excluded_fields = {
            'mesh_boundary_nodes_disps',
            'load_factor_time_series',
            'mesh_nodes_coords_ref',
            'mesh_boundary_nodes_disps_time'}
        for key, value in self.matpatch.items():
            if key not in excluded_fields:
                if key == 'mesh_nodes_matrix':
                    self.simulation_data[
                        'mesh_nodes_matrix'] = (
                        self._create_mesh_nodes_matrix())
                else:
                    self.simulation_data[key] = value

    # -------------------------------------------------------------------------
    def _create_mesh_nodes_matrix(self):
        """Create mesh nodes matrix based on mesh."""
        if self.dim == 2:
            if self.elem_order == 1:
                mesh_nodes_matrix = np.zeros(
                    (self.mesh_nx + 1,
                     self.mesh_ny + 1), dtype=int)
                node_idx = 0
                for i in range(self.mesh_nx + 1):
                    for j in range(self.mesh_ny + 1):
                        mesh_nodes_matrix[i, j] = (
                            node_idx)
                        node_idx += 1
            else:
                mesh_nodes_matrix = np.zeros(
                    (2 * self.mesh_nx + 1,
                     2 * self.mesh_ny + 1), dtype=int)
                node_idx = 0
                for i in range(2 * self.mesh_nx + 1):
                    for j in range(
                            2 * self.mesh_ny + 1):
                        mesh_nodes_matrix[i, j] = (
                            node_idx)
                        node_idx += 1
            return torch.tensor(mesh_nodes_matrix)
        elif self.dim == 3:
            if self.elem_order == 1:
                mesh_nodes_matrix = np.zeros(
                    (self.mesh_nx + 1,
                     self.mesh_ny + 1,
                     self.mesh_nz + 1), dtype=int)
                node_idx = 0
                for i in range(self.mesh_nx + 1):
                    for j in range(self.mesh_ny + 1):
                        for k in range(
                                self.mesh_nz + 1):
                            mesh_nodes_matrix[
                                i, j, k] = node_idx
                            node_idx += 1
            else:
                mesh_nodes_matrix = np.zeros(
                    (2 * self.mesh_nx + 1,
                     2 * self.mesh_ny + 1,
                     2 * self.mesh_nz + 1), dtype=int)
                node_idx = 0
                for i in range(2 * self.mesh_nx + 1):
                    for j in range(
                            2 * self.mesh_ny + 1):
                        for k in range(
                                2 * self.mesh_nz + 1):
                            mesh_nodes_matrix[
                                i, j, k] = node_idx
                            node_idx += 1
            return torch.tensor(mesh_nodes_matrix)

    # -------------------------------------------------------------------------
    def apply_boundary_conditions(self):
        """Apply boundary conditions from material patch."""
        _, self.nodes_constrained = (
            prescribe_disps_by_coords(
                domain=self.domain,
                data=self.matpatch,
                dim=self.dim))

    # -------------------------------------------------------------------------
    def _get_base_increments(self):
        """Return base increment sequence from patch.

        Returns
        -------
        increments : torch.Tensor
            Load factor sequence, length
            num_increments + 1.
        """
        if self.num_increments == 1:
            return torch.linspace(0.0, 1.0, 2)
        return torch.tensor(
            self.matpatch['load_factor_time_series'])

    # -------------------------------------------------------------------------
    @staticmethod
    def _refine_increments(increments, n_subdiv):
        """Subdivide each interval between load factors.

        Inserts n_subdiv - 1 equally-spaced points
        within each consecutive pair. Original points
        remain at stride n_subdiv in the refined
        sequence, so downsampling the solver output by
        ``[::n_subdiv]`` recovers values at the
        original time points.

        Parameters
        ----------
        increments : torch.Tensor
            Original increments, shape (N0,).
        n_subdiv : int
            Number of subintervals per original
            segment. n_subdiv=1 returns input
            unchanged.

        Returns
        -------
        refined : torch.Tensor
            Refined increments, shape
            ((N0 - 1) * n_subdiv + 1,).
        """
        if n_subdiv == 1:
            return increments
        pieces = []
        for i in range(len(increments) - 1):
            a = increments[i].item()
            b = increments[i + 1].item()
            for k in range(n_subdiv):
                pieces.append(
                    a + (b - a) * k / n_subdiv)
        pieces.append(increments[-1].item())
        return torch.tensor(
            pieces, dtype=increments.dtype)

    # -------------------------------------------------------------------------
    def solve(self, increments=None):
        """Solve the FEM system.

        Parameters
        ----------
        increments : {torch.Tensor, None}, default=None
            Optional override of the increment
            sequence. If None, uses base increments
            from the patch.

        Returns
        -------
        results : tuple
            (u_disp, f_int, sigma_out, def_grad,
             alpha_out, vol_elem).
        """
        if increments is None:
            increments = self._get_base_increments()
        return self.domain.solve(
            increments=increments,
            return_intermediate=True,
            aggregate_integration_points=True,
            aggregate_state=False,
            return_volumes=True)

    # -------------------------------------------------------------------------
    def postprocess_results(
            self, u_disp, f_int, sigma_out, def_grad,
            alpha_out, vol_elem):
        """Postprocess and store simulation results.

        Parameters
        ----------
        u_disp : torch.Tensor
            Nodal displacements.
        f_int : torch.Tensor
            Internal forces.
        sigma_out : torch.Tensor
            Stress tensor (aggregated over GPs).
        def_grad : torch.Tensor
            Deformation gradient (aggregated over GPs).
        alpha_out : torch.Tensor
            State variables (un-aggregated, per GP).
        vol_elem : torch.Tensor
            Element volumes.
        """
        # Extrapolate epbar from un-aggregated state
        # (n_int, n_elem per increment) before averaging
        nodal_epbar = None
        if (self.is_save_nodal_epbar
                and self.material_behavior in [
                    'elastoplastic_lh',
                    'elastoplastic_nlh']):
            n_inc = alpha_out.shape[0]
            n_nod = self.domain.n_nod
            nodal_epbar = torch.zeros(n_inc, n_nod)
            for t in range(n_inc):
                # state_gp: (n_int, n_elem)
                state_gp = alpha_out[t, :, :, 0]
                nodal_epbar[t] = (
                    extrapolate_gauss_to_nodes(
                        self.domain.etype,
                        state_gp,
                        self.domain.elements,
                        n_nod))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Aggregate state over integration points
        alpha_avg = alpha_out.mean(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stress and def_grad already aggregated by solver
        vol_weights = self._compute_volume_weights(
            vol_elem)
        sigma_out_avg = self._compute_stress_avg(
            sigma_out, vol_weights)
        alpha_out_avg = (
            self._compute_plastic_strain_avg(
                alpha_avg, vol_weights))
        strain_energy_density = (
            compute_strain_energy_density(
                sigma_out, def_grad,
                self.material_behavior, self.dim))
        total_strain_energy = (
            strain_energy_density * vol_weights
        ).sum(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._store_results(
            u_disp, f_int, sigma_out_avg,
            alpha_out_avg, total_strain_energy,
            nodal_epbar=nodal_epbar)

    # -------------------------------------------------------------------------
    def _compute_volume_weights(self, vol_elem):
        """Compute volume-weighted normalization."""
        total_volume = vol_elem.sum(
            dim=1, keepdim=True)
        return vol_elem / total_volume

    # -------------------------------------------------------------------------
    def _compute_stress_avg(
            self, sigma_out, vol_weights):
        """Compute volume-weighted stress average."""
        vol_weights_expanded = (
            vol_weights.unsqueeze(-1).unsqueeze(-1))
        if self.num_increments == 1:
            return (sigma_out[-1, :, :]
                    * vol_weights_expanded)
        else:
            return sigma_out * vol_weights_expanded

    # -------------------------------------------------------------------------
    def _compute_plastic_strain_avg(
            self, alpha_out, vol_weights):
        """Compute volume-weighted plastic strain avg."""
        if self.material_behavior in [
                'elastoplastic_lh',
                'elastoplastic_nlh']:
            return (
                (alpha_out[:, :, 0] * vol_weights)
                .sum(dim=1))
        return None

    # -------------------------------------------------------------------------
    def _store_results(
            self, u_disp, f_int, sigma_out_avg,
            alpha_out_avg, total_strain_energy,
            nodal_epbar=None):
        """Store all simulation results.

        Parameters
        ----------
        u_disp : torch.Tensor
            Nodal displacements.
        f_int : torch.Tensor
            Internal forces.
        sigma_out_avg : torch.Tensor
            Volume-weighted stress average.
        alpha_out_avg : torch.Tensor
            Volume-weighted plastic strain average.
        total_strain_energy : torch.Tensor
            Total strain energy.
        nodal_epbar : torch.Tensor, default=None
            Nodal epbar, shape (N_inc, n_nod).
        """
        for idx_node in range(u_disp.shape[1]):
            if idx_node in self.nodes_constrained:
                if self.num_increments == 1:
                    self.simulation_data[
                        'bd_nodes_disps_time_series'][
                        idx_node] = (
                        u_disp[-1, idx_node, :])
                    self.simulation_data[
                        'bd_nodes_forces_time_series'][
                        idx_node] = (
                        f_int[-1, idx_node, :])
                else:
                    self.simulation_data[
                        'bd_nodes_disps_time_series'][
                        idx_node] = (
                        u_disp[:, idx_node, :])
                    self.simulation_data[
                        'bd_nodes_forces_time_series'][
                        idx_node] = (
                        f_int[:, idx_node, :])
                # Store per-node epbar time series
                if nodal_epbar is not None:
                    if self.num_increments == 1:
                        self.simulation_data[
                            'bd_nodes_epbar_time_series'
                            ][idx_node] = (
                            nodal_epbar[
                                -1, idx_node])
                    else:
                        self.simulation_data[
                            'bd_nodes_epbar_time_series'
                            ][idx_node] = (
                            nodal_epbar[
                                :, idx_node])
        if self.num_increments == 1:
            self.simulation_data['stress_avg'] = (
                sigma_out_avg[-1, 0])
            self.simulation_data[
                'strain_energy_density'] = (
                total_strain_energy[-1])
            if (self.is_save_avg_epbar
                    and self.material_behavior in [
                        'elastoplastic_lh',
                        'elastoplastic_nlh']):
                self.simulation_data[
                    'epsilon_pl_eq'] = (
                    alpha_out_avg[-1])
        else:
            for idx_time in range(
                    self.num_increments + 1):
                self.simulation_data[
                    'strain_energy_density'][
                    idx_time] = (
                    total_strain_energy[idx_time])
                if (self.is_save_avg_epbar
                        and self.material_behavior
                        in ['elastoplastic_lh',
                            'elastoplastic_nlh']):
                    self.simulation_data[
                        'epsilon_pl_eq'][
                        idx_time] = (
                        alpha_out_avg[idx_time])

    # -------------------------------------------------------------------------
    def save_results(self):
        """Save simulation results to pickle file."""
        try:
            with open(self.output_filename, 'wb') as f:
                pkl.dump(
                    self.simulation_data, f,
                    protocol=pkl.HIGHEST_PROTOCOL)
        except Exception as excp:
            print(
                f'Error saving simulation data: {excp}')

    # -------------------------------------------------------------------------
    def run(self):
        """Execute complete simulation workflow.

        If ``self.is_adaptive_timestepping`` is True,
        applies retry-and-downsample on Newton-Raphson
        failure: re-solves with 2x, 4x, 8x refinement
        of the load factor sequence and downsamples
        back to the original time points. If False,
        performs a single solve attempt at the base
        increments and propagates convergence failures.

        Raises
        ------
        RuntimeError
            If adaptive timestepping is enabled and
            Newton-Raphson fails at all subdivision
            levels (1x, 2x, 4x, 8x).
        Exception
            If adaptive timestepping is disabled and
            Newton-Raphson fails on the single solve.
        """
        self.load_material_patch()
        material = self.create_material()
        self.domain = self.create_mesh(material)
        self.initialize_data_structures()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stiffness extraction (before BCs/solve)
        if self.is_compute_stiffness:
            self.compute_boundary_stiffness()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.apply_boundary_conditions()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve (with optional adaptive sub-increment)
        if not self.is_adaptive_timestepping:
            results = self.solve()
        else:
            base_incr = self._get_base_increments()
            # Build powers-of-2 sequence in
            # [1, adaptive_max_subdiv]
            subdiv_seq = []
            k = 1
            while k <= self.adaptive_max_subdiv:
                subdiv_seq.append(k)
                k *= 2
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            results = None
            used_subdiv = 1
            last_exc = None
            for n_subdiv in subdiv_seq:
                incr = self._refine_increments(
                    base_incr, n_subdiv)
                try:
                    results = self.solve(
                        increments=incr)
                    used_subdiv = n_subdiv
                    if n_subdiv > 1:
                        print(
                            f'  patch {self.patch_idx}:'
                            f' converged with '
                            f'{n_subdiv}x '
                            f'subincrementation')
                    break
                except Exception as excp:
                    last_exc = excp
            if results is None:
                raise RuntimeError(
                    f'Newton-Raphson failed for patch '
                    f'{self.patch_idx} up to '
                    f'{subdiv_seq[-1]}x '
                    f'subincrementation: {last_exc}')
            # Downsample refined results to base pts
            if used_subdiv > 1:
                results = tuple(
                    r[::used_subdiv] for r in results)
        self.postprocess_results(*results)
        if self.is_save:
            self.save_results()


# =============================================================================
def run_simulation(
    element_type='quad4',
    material_behavior='elastic',
    patch_idx=0,
    num_increments=1,
    mesh_nx=3,
    mesh_ny=3,
    mesh_nz=3,
    is_red_int=False,
    is_save=False,
    is_compute_stiffness=False,
    filepath='/Users/rbarreira/Desktop/machine_learning/'
             'material_patches/_data/'
):
    """Run FEM simulation for material patch analysis.

    Convenience function wrapping Simulation class.

    Parameters
    ----------
    element_type : str
        Finite element type.
    material_behavior : str
        Material constitutive model.
    patch_idx : int
        Material patch identifier index.
    num_increments : int
        Number of load increments.
    mesh_nx : int
        Number of elements in x-direction.
    mesh_ny : int
        Number of elements in y-direction.
    mesh_nz : int
        Number of elements in z-direction.
    is_red_int : bool
        Use reduced integration.
    is_save : bool
        Save simulation output.
    is_compute_stiffness : bool
        Compute boundary stiffness matrices.
    filepath : str
        Base path to material patch input data.
    """
    sim = Simulation(
        element_type=element_type,
        material_behavior=material_behavior,
        patch_idx=patch_idx,
        num_increments=num_increments,
        mesh_nx=mesh_nx,
        mesh_ny=mesh_ny,
        mesh_nz=mesh_nz,
        is_red_int=is_red_int,
        is_save=is_save,
        is_compute_stiffness=is_compute_stiffness,
        filepath=filepath,
    )
    sim.run()


# =============================================================================
if __name__ == '__main__':
    filepath = '/Volumes/Expansion/material_patches_data/'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test 1: 1x1 irregular patch (all boundary, no
    # interior DOFs -- K_cond == full K)
    print('=== 1x1 patch ===')
    run_simulation(
        element_type='quad4',
        material_behavior='elastic',
        num_increments=1,
        patch_idx=0,
        filepath=filepath,
        mesh_nx=1,
        mesh_ny=1,
        is_compute_stiffness=True,
        is_save=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test 2: 5x5 irregular patch (20 boundary, 16
    # interior -- static condensation active)
    print('=== 5x5 irregular patch ===')
    run_simulation(
        element_type='quad4',
        material_behavior='elastic',
        num_increments=1,
        patch_idx=0,
        filepath=filepath,
        mesh_nx=5,
        mesh_ny=5,
        is_compute_stiffness=True,
        is_save=False)
