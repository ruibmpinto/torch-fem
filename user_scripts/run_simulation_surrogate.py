"""User script: Run FEM simulation with material patch surrogate models."""
#
#                                                                       Modules
# =============================================================================
# Standard
import cProfile
import os
import pathlib
import pickle as pkl
import pstats
import signal
import sys
import time

import psutil
import scalene

# Add graphorge to sys.path
graphorge_path = str(pathlib.Path(__file__).parents[2] \
                     / "graphorge" / "src")
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch

# Enable graphorge imports in torchfem
os.environ['TORCHFEM_IMPORT_GRAPHORGE'] = '1'

# Local
from utils.boundary_conditons import prescribe_disps_by_coords
from utils.plotting import (
    plot_boundary_error_overlay,
    plot_boundary_overlay,
    plot_boundary_panels,
)

from torchfem import Planar, Solid
from torchfem.elements import linear_to_quadratic
from torchfem.materials import (
    Hyperelastic3D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicHenckyPlaneStrain,
    IsotropicPlasticity3D,
    IsotropicPlasticityPlaneStrain,
)
from torchfem.mesh import cube_hexa, rect_quad

# Matplotlib.pyplot default parameters
plt.rcParams.update({
    "text.usetex": False,
    "font.size": 12,
    "axes.titlesize": 16,
    "figure.dpi": 360,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.figsize": (6, 6),
    "lines.linewidth": 1.5
})
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
#
torch.set_default_dtype(torch.float64)
# -----------------------------------------------------------------------------
def run_simulation_surrogate(
        element_type='quad4',
        material_behavior='elastic',
        mesh_nx=1, mesh_ny=1, mesh_nz=1,
        patch_size_x=1, patch_size_y=1,
        patch_size_z=1,
        edge_type='all',
        edge_feature_type=('edge_vector',),
        fe_border=0, patch_zones=None,
        is_adaptive_timestepping=False,
        adaptive_max_subdiv=8,
        *, model_name):
    """Run simulation with Graphorge surrogate model.

    Parameters
    ----------
    patch_zones : list[dict], default=None
        List of zone dicts, each with keys:
        - 'region': (row_start, row_end, col_start, col_end)
        - 'patch_size': (nx, ny)
        When None, a single zone is built from patch_size_x/y
        covering the full surrogate region.
    is_adaptive_timestepping : bool, default=False
        If True, wrap the solve in a retry-and-subdivide loop:
        on Newton-Raphson failure, re-solve with 2x, 4x, ...
        refinement of the load-factor sequence (up to
        `adaptive_max_subdiv`) and downsample tensor results
        back to the original time points. If False, perform a
        single solve; a convergence failure propagates as an
        exception.
    adaptive_max_subdiv : int, default=8
        Maximum subdivision factor. The retry sequence is the
        powers of 2 in [1, adaptive_max_subdiv]. Ignored when
        `is_adaptive_timestepping` is False.
    model_name : {str, list[str]}
        Sub-directory name under
        matpatch_surrogates/{material_behavior}/{NxN}/ from
        which to load the surrogate model. Required, keyword-
        only. When patch_zones contains multiple zones, pass a
        list of length len(patch_zones) with one model name
        per zone, in zone order. Zones that share the same
        patch resolution must carry the same model name.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Monitor memory and time
    process = psutil.Process(os.getpid())
    start_time = time.time()

    def print_status(location):
        current_time = time.time()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"[{location}] Time: {current_time - start_time:.2f}s, "
              f"Memory: {memory_mb:.1f}MB")

    print_status("START")

    # Set up signal handler to catch termination
    def signal_handler(signum, frame):
        print(f"\n[SIGNAL] Caught signal {signum}")
        print_status("SIGNAL_CAUGHT")
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    # Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Determine element order and dimension
    if element_type in ['quad4', 'tri3', 'tetra4', 'hex8']:
        elem_order = 1
    elif element_type in ['quad8', 'tri6', 'tetra10', 'hex20']:
        elem_order = 2

    if element_type in ['quad4', 'tri3', 'quad8', 'tri6']:
        dim = 2
    elif element_type in ['tetra4', 'hex8', 'tetra10', 'hex20']:
        dim = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Default model path if not provided
    # Models stored in matpatch_surrogates/{behavior}/{NxN}/model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    torch_fem_root = os.path.dirname(script_dir)
    results_base = os.path.join(torch_fem_root, 'results')
    surrogates_dir = os.path.join(script_dir, 'matpatch_surrogates')
    # Build model path(s) -- single or multi-resolution
    # (deferred until after patch_zones are resolved)
    # model_directory_map set below
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Constitutive law
    if material_behavior == 'elastic':

        e_young = 110000
        nu = 0.33

        if dim == 2:
            material = IsotropicElasticityPlaneStrain(E=e_young, nu=nu)
        elif dim == 3:
            material = IsotropicElasticity3D(E=e_young, nu=nu)
    elif material_behavior == 'hyperelastic':

        e_young = 20000
        nu = 0.33

        lmbda = e_young * nu / ((1. + nu) * (1. - 2. * nu))
        mu = e_young / (2. * (1. + nu))

        def psi(F):
            """
            Neo-Hookean strain energy density function.
            """
            # Compute the right Cauchy-Green deformation tensor
            C = F.transpose(-1, -2) @ F
            # Stable computation of the logarithm of the determinant
            logJ = 0.5 * torch.logdet(C)
            return (mu / 2 * (torch.trace(C) - 3.0) - mu * logJ +
                    lmbda / 2 * logJ**2)

        if dim == 2:
            material = IsotropicHenckyPlaneStrain(E=e_young, nu=nu)
        elif dim == 3:
            # material = IsotropicHencky3D(
            #     E=e_young, nu=nu, n_state=0)
            material = Hyperelastic3D(psi)

    elif material_behavior == 'elastoplastic':

        e_young = 210000
        nu = 0.33

        sigma_y = 100.0
        hardening_modulus = 100.0

        # Hardening function
        def sigma_f(q):
            return sigma_y + hardening_modulus * q

        # Derivative of the hardening function
        def sigma_f_prime(q):
            return hardening_modulus

        if dim == 2:
            material = IsotropicPlasticityPlaneStrain(
                E=e_young, nu=nu, sigma_f=sigma_f,
                sigma_f_prime=sigma_f_prime)
        elif dim == 3:
            material = IsotropicPlasticity3D(
                E=e_young, nu=nu, sigma_f=sigma_f,
                sigma_f_prime=sigma_f_prime)

    elif material_behavior == 'elastoplastic_nlh':

        e_young = 70000
        nu = 0.33

        # Swift-Voce hardening parameters for AA2024
        a_s = 798.56    # MPa
        epsilon_0 = 0.0178
        n_sv = 0.202
        k_0 = 363.84    # MPa
        q_v = 240.03     # MPa
        beta = 10.533
        omega = 0.368

        def k_s(eps_pl):
            """Swift hardening component."""
            return a_s * (epsilon_0 + eps_pl)**n_sv

        def k_v(eps_pl):
            """Voce hardening component."""
            return (
                k_0
                + q_v * (1.0 - torch.exp(-beta * eps_pl)))

        def sigma_f(eps_pl):
            """Combined Swift-Voce hardening."""
            return (
                omega * k_s(eps_pl)
                + (1.0 - omega) * k_v(eps_pl))

        def sigma_f_prime(eps_pl):
            """Derivative of Swift-Voce hardening."""
            dks = (
                a_s * n_sv
                * (epsilon_0 + eps_pl)**(n_sv - 1.0))
            dkv = (
                q_v * beta
                * torch.exp(-beta * eps_pl))
            return (
                omega * dks
                + (1.0 - omega) * dkv)

        if dim == 2:
            material = IsotropicPlasticityPlaneStrain(
                E=e_young, nu=nu, sigma_f=sigma_f,
                sigma_f_prime=sigma_f_prime,
                max_iter=50)
        elif dim == 3:
            material = IsotropicPlasticity3D(
                E=e_young, nu=nu, sigma_f=sigma_f,
                sigma_f_prime=sigma_f_prime,
                max_iter=50)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Geometry & mesh
    if dim == 2:
        # rect_quad takes number of nodes per direction
        nodes, elements = rect_quad(mesh_nx + 1, mesh_ny + 1)
        if elem_order == 2:
            nodes, elements = linear_to_quadratic(nodes, elements)
        # Create domain
        domain = Planar(nodes, elements, material)
    elif dim == 3:
        # cube_hexa takes number of nodes per direction
        nodes, elements = cube_hexa(mesh_nx + 1, mesh_ny + 1, mesh_nz + 1)
        if elem_order == 2:
            nodes, elements = linear_to_quadratic(nodes, elements)
        # Create domain
        domain = Solid(nodes, elements, material)
    # Define material patch flag - map elements to patches
    num_elements = elements.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build patch_zones if not provided (backward compat)
    if patch_zones is None:
        surr_nx = mesh_nx - 2 * fe_border
        surr_ny = mesh_ny - 2 * fe_border
        patch_zones = [{
            'region': (
                fe_border, fe_border + surr_ny,
                fe_border, fe_border + surr_nx),
            'patch_size': (
                patch_size_x, patch_size_y),
        }]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize model_name to a per-zone list parallel to patch_zones.
    if isinstance(model_name, str):
        if len(patch_zones) != 1:
            raise ValueError(
                f'model_name must be a list of length {len(patch_zones)} '
                f'(one entry per patch zone) when patch_zones contains '
                f'multiple zones; got a single string.')
        model_names_per_zone = [model_name]
    else:
        if len(model_name) != len(patch_zones):
            raise ValueError(
                f'model_name list length {len(model_name)} does not match '
                f'number of patch zones {len(patch_zones)}.')
        model_names_per_zone = list(model_name)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Classify elements into patches (multi-zone)
    patch_id_counter = 0
    patch_resolution = {}
    model_name_by_res = {}
    is_mat_patch = torch.full(
        (num_elements,), -1, dtype=torch.int)
    for zone_idx, zone in enumerate(patch_zones):
        r0, r1, c0, c1 = zone['region']
        psx, psy = zone['patch_size']
        zone_nx = (c1 - c0) // psx
        zone_ny = (r1 - r0) // psy
        # Bind the resolution to a model name; reject mismatches so the
        # same resolution cannot pull from two different checkpoints.
        zone_model_name = model_names_per_zone[zone_idx]
        if (psx, psy) in model_name_by_res:
            if model_name_by_res[(psx, psy)] != zone_model_name:
                raise ValueError(
                    f'Patch resolution ({psx}, {psy}) is bound to '
                    f'model_name {model_name_by_res[(psx, psy)]!r} from '
                    f'an earlier zone but zone {zone_idx} requests '
                    f'{zone_model_name!r}.')
        else:
            model_name_by_res[(psx, psy)] = zone_model_name
        for elem_idx in range(num_elements):
            ei = elem_idx // mesh_nx
            ej = elem_idx % mesh_nx
            if r0 <= ei < r1 and c0 <= ej < c1:
                pi = (ei - r0) // psy
                pj = (ej - c0) // psx
                pid = patch_id_counter + pi * zone_nx + pj
                is_mat_patch[elem_idx] = pid
                patch_resolution[pid] = (psx, psy)
        patch_id_counter += zone_nx * zone_ny
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build model directory map from unique resolutions
    unique_res = set(patch_resolution.values())
    is_multi_res = len(unique_res) > 1
    if is_multi_res:
        model_directory_map = {
            res: os.path.join(
                surrogates_dir, material_behavior,
                f'{res[0]}x{res[1]}', model_name_by_res[res])
            for res in unique_res}
        patch_resolution_arg = patch_resolution
    else:
        res = list(unique_res)[0]
        model_directory_map = os.path.join(
            surrogates_dir, material_behavior,
            f'{res[0]}x{res[1]}', model_name_by_res[res])
        patch_resolution_arg = None
    # Stable per-run suffix used to keep different model variants from
    # overwriting each other's output directories.
    output_model_tag = '_'.join(sorted(set(model_names_per_zone)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build patch_elem_per_dim map
    if is_multi_res:
        patch_elem_per_dim_map = {
            pid: list(res)
            for pid, res
            in patch_resolution.items()}
    else:
        res = list(unique_res)[0]
        patch_elem_per_dim_map = list(res)

    # Identify boundary nodes of material patches
    # Build node-to-patches mapping (only patch elems)
    node_to_patches = {}
    for elem_idx in range(num_elements):
        patch_id = is_mat_patch[elem_idx].item()
        if patch_id < 0:
            continue
        elem_nodes = elements[
            elem_idx, :4].tolist()
        for node_id in elem_nodes:
            if node_id not in node_to_patches:
                node_to_patches[node_id] = set()
            node_to_patches[node_id].add(patch_id)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect nodes belonging to FE elements
    fe_nodes = set()
    for elem_idx in range(num_elements):
        if is_mat_patch[elem_idx].item() == -1:
            for nid in elements[
                    elem_idx, :4].tolist():
                fe_nodes.add(nid)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Identify external boundary nodes
    external_boundary_nodes = set()
    for i, node_coord in enumerate(nodes):
        tol = 1e-6
        if dim == 2:
            on_boundary = (
                torch.abs(node_coord[0]) < tol
                or torch.abs(
                    node_coord[0] - 1.0) < tol
                or torch.abs(
                    node_coord[1]) < tol
                or torch.abs(
                    node_coord[1] - 1.0) < tol)
        elif dim == 3:
            on_boundary = (
                torch.abs(node_coord[0]) < tol
                or torch.abs(
                    node_coord[0] - 1.0) < tol
                or torch.abs(
                    node_coord[1]) < tol
                or torch.abs(
                    node_coord[1] - 1.0) < tol
                or torch.abs(
                    node_coord[2]) < tol
                or torch.abs(
                    node_coord[2] - 1.0) < tol)
        if on_boundary:
            external_boundary_nodes.add(i)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Patch boundary nodes: shared between patches,
    # on external boundary, or touching FE region
    patch_boundary_nodes = []
    for node_id in node_to_patches:
        patches = node_to_patches[node_id]
        is_boundary = (
            len(patches) > 1
            or node_id in external_boundary_nodes
            or node_id in fe_nodes)
        if is_boundary:
            patch_boundary_nodes.append(node_id)
    patch_boundary_nodes = torch.tensor(
        sorted(patch_boundary_nodes),
        dtype=torch.long)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Internal patch nodes: in a patch, not boundary,
    # not part of any FE element
    all_patch_nodes = set(node_to_patches.keys())
    patch_internal_nodes = (
        all_patch_nodes
        - set(patch_boundary_nodes.tolist())
        - fe_nodes)
    patch_internal_nodes = torch.tensor(
        sorted(list(patch_internal_nodes)),
        dtype=torch.long)

    # Build patch_boundary_nodes_dict:
    # patch_id -> boundary node indices
    patch_boundary_nodes_dict = {}
    n_patches = patch_id_counter
    for patch_id in range(n_patches):
        patch_boundary_set = set()
        for node_id, patches in (
                node_to_patches.items()):
            if patch_id in patches:
                if (len(patches) > 1
                        or node_id
                        in external_boundary_nodes
                        or node_id in fe_nodes):
                    patch_boundary_set.add(
                        node_id)
        patch_boundary_nodes_dict[
            patch_id] = torch.tensor(
            sorted(list(patch_boundary_set)),
            dtype=torch.long)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary conditions
    # # OLD: Load boundary conditions from material patch file (1x1 only)
    # bc_filepath = (
    #     '/Users/rbarreira/Desktop/machine_learning/material_patches/'
    #     '_input_material_patches/material_patches_generation_2d_quad4_'
    #     'mesh_1x1/material_patch_0/material_patch/'
    #     'material_patch_attributes.pkl')
    # with open(bc_filepath, 'rb') as file:
    #     matpatch = pkl.load(file)
    # _, nodes_constrained = prescribe_disps_by_coords(
    #     domain=domain, data=matpatch, dim=dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Simple tension test BCs
    tol = 1e-6
    for i, node_coord in enumerate(nodes):
        # Bottom edge: fix y
        if torch.abs(node_coord[1]) < tol:
            domain.displacements[i, 1] = 0.0
            domain.constraints[i, 1] = True
            # Bottom-left corner: also fix x
            if torch.abs(node_coord[0]) < tol:
                domain.displacements[i, 0] = 0.0
                domain.constraints[i, 0] = True
        # Top edge: prescribe vertical displacement
        elif torch.abs(node_coord[1] - 1.0) < tol:
            domain.displacements[i, 1] = 0.05
            domain.constraints[i, 1] = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute reference solution WITHOUT internal node constraints
    # (all nodes free to move)
    print_status("BEFORE_SOLVE")
    if material_behavior in ('elastoplastic', 'elastoplastic_nlh'):
        increments_ref = torch.linspace(0.0, 1.0, 51)
    else:
        increments_ref = torch.linspace(0.0, 1.0, 11)

    # Profile solve method
    profiler_solve = cProfile.Profile()
    profiler_solve.enable()
    u_ref, f_ref, _, _, state_ref, res_hist_ref = \
        domain.solve(
            increments=increments_ref, rtol=1e-8,
            verbose=True, return_resnorm=True)
    profiler_solve.disable()
    print("\n=== SOLVE METHOD PROFILE ===")
    stats_solve = pstats.Stats(profiler_solve)
    stats_solve.sort_stats('cumulative').print_stats(15)
    print_status("AFTER_SOLVE")

    # Plot reference solution
    u_ref_final = u_ref[-1]
    # Compute reference colorbar limits
    u_ref_norm = torch.norm(u_ref_final, dim=1)
    vmin_ref = u_ref_norm.min().item()
    vmax_ref = u_ref_norm.max().item()

    domain.plot(
        u=u_ref_final,
        node_property=u_ref_norm,
        title=r'Reference: $||\mathbf{u}||_{2}$ (no internal '
              r'constraints)',
        colorbar=True,
        cmap='viridis',
        vmin=vmin_ref,
        vmax=vmax_ref
    )
    mesh_str = f"{mesh_nx}x{mesh_ny}"
    patch_str = f"{patch_size_x}x{patch_size_y}"
    if dim == 3:
        mesh_str += f"x{mesh_nz}"
        patch_str += f"x{patch_size_z}"
    n_increments = len(increments_ref) - 1
    output_dir = os.path.join(
        results_base, material_behavior, f"{dim}d", element_type,
        f"mesh_{mesh_str}", f"patch_{patch_str}",
        f"n_time_inc_{n_increments}")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir, "disp_field_ref.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Reference solution saved to {plot_path}")

    # Save reference results
    results_ref = {
        'displacements': u_ref.detach().cpu().numpy(),
        'forces': f_ref.detach().cpu().numpy(),
    }
    ref_file = os.path.join(
        output_dir, "results_reference.pkl")
    with open(ref_file, 'wb') as fh:
        pkl.dump(results_ref, fh)
    print(f"Reference results saved to {ref_file}")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save reference residual and state to subdirectory
    ref_results_dir = os.path.join(
        output_dir, 'reference_results')
    os.makedirs(ref_results_dir, exist_ok=True)
    # Save residual history as CSV files
    for inc, norms in res_hist_ref.items():
        res0 = norms[0] if norms[0] > 0 else 1.0
        rows = np.column_stack((
            np.arange(len(norms)),
            np.array(norms),
            np.array(norms) / res0))
        np.savetxt(
            os.path.join(
                ref_results_dir,
                f'residual_inc{inc}.csv'),
            rows,
            delimiter=',',
            header='iteration,absolute,relative',
            comments='')
    print(f'Reference residuals saved to '
          f'{ref_results_dir}')
    # Save equivalent plastic strain as pkl
    epbar_file = os.path.join(
        ref_results_dir, 'epbar.pkl')
    with open(epbar_file, 'wb') as fh:
        pkl.dump(
            state_ref.detach().cpu().numpy(), fh)
    print(f'Reference epbar saved to {epbar_file}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save reference constraint state before locking internal
    # nodes (needed for plotting reference fields later)
    constraints_ref = domain.constraints.clone()
    displacements_bc_ref = domain.displacements.clone()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Constrain internal patch nodes (zero displacement, all DOFs)
    # for surrogate solve
    for node_id in patch_internal_nodes:
        domain.displacements[node_id, :] = 0.0
        domain.constraints[node_id, :] = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solver
    # Create more increments for elastoplastic simulation
    if material_behavior in (
            'elastoplastic', 'elastoplastic_nlh'):
        increments = torch.linspace(0.0, 1.0, 51)
        is_stepwise = True
    else:
        # One step for elastic sim
        # increments = torch.tensor([0.0, 1.0])
        # is_stepwise = False
        increments = torch.linspace(0.0, 1.0, 11)
        # RNN-like behavior
        is_stepwise = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mesh_str = f"{mesh_nx}x{mesh_ny}"
    patch_str = f"{patch_size_x}x{patch_size_y}"
    if dim == 3:
        mesh_str += f"x{mesh_nz}"
        patch_str += f"x{patch_size_z}"
    n_increments = len(increments) - 1

    output_dir = os.path.join(
        results_base, material_behavior, f"{dim}d", element_type,
        f"mesh_{mesh_str}", f"patch_{patch_str}",
        f"n_time_inc_{n_increments}", output_model_tag)
    os.makedirs(output_dir, exist_ok=True)

    # Plot mesh colored by material patch ID
    domain.plot(
        element_property=is_mat_patch.float(),
        colorbar=True,
        title='Material Patches',
        bcs=False
    )
    # Add patch internal nodes (red)
    plt.scatter(
        nodes[patch_internal_nodes, 0].numpy(),
        nodes[patch_internal_nodes, 1].numpy(),
        c='red',
        s=20,
        marker='x',
        zorder=11
    )
    # Add patch boundary nodes (black)
    plt.scatter(
        nodes[patch_boundary_nodes, 0].numpy(),
        nodes[patch_boundary_nodes, 1].numpy(),
        c='blue',
        s=50,
        marker='o',
        zorder=10
    )

    plot_path = os.path.join(output_dir, "material_patches.png")
    plt.savefig(plot_path)
    plt.close()


    # breakpoint()


    print_status("BEFORE_SOLVE_MATPATCH")
    # Profile solve_matpatch method
    profiler_matpatch = cProfile.Profile()
    profiler_matpatch.enable()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build output directory paths for stiffness saving
    mesh_str = f'{mesh_nx}x{mesh_ny}'
    if is_multi_res:
        patch_str = '_'.join(
            f'{r[0]}x{r[1]}'
            for r in sorted(unique_res))
    else:
        res0 = list(unique_res)[0]
        patch_str = f'{res0[0]}x{res0[1]}'
    if dim == 3:
        mesh_str += f'x{mesh_nz}'
    n_increments = len(increments) - 1
    output_dir = os.path.join(
        results_base, material_behavior, f'{dim}d', element_type,
        f'mesh_{mesh_str}', f'patch_{patch_str}',
        f'n_time_inc_{n_increments}', output_model_tag)
    os.makedirs(output_dir, exist_ok=True)
    stiffness_dir = os.path.join(output_dir, 'stiffness')
    os.makedirs(stiffness_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build base solve kwargs shared by all retry attempts
    solve_kwargs = dict(
        is_mat_patch=is_mat_patch, max_iter=100, rtol=1e-8,
        verbose=True, return_intermediate=True, return_volumes=False,
        return_resnorm=True, is_stepwise=is_stepwise,
        model_directory=model_directory_map,
        patch_boundary_nodes=patch_boundary_nodes_dict,
        patch_elem_per_dim=patch_elem_per_dim_map,
        patch_resolution=patch_resolution_arg,
        edge_type=edge_type, edge_feature_type=edge_feature_type,
        is_export_stiffness=True, stiffness_output_dir=stiffness_dir,
        patch_size_label=patch_str)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve (with optional adaptive sub-increment)
    if not is_adaptive_timestepping:
        u, f, _, _, _, residual_history = domain.solve_matpatch(
            increments=increments, **solve_kwargs)
    else:
        subdiv_seq = []
        k_sub = 1
        while k_sub <= adaptive_max_subdiv:
            subdiv_seq.append(k_sub)
            k_sub *= 2
        last_exc = None
        results = None
        used_subdiv = 1
        for n_subdiv in subdiv_seq:
            if n_subdiv == 1:
                incr = increments
            else:
                refined = [increments[0]]
                for j in range(len(increments) - 1):
                    a, b = increments[j], increments[j + 1]
                    for s in range(1, n_subdiv + 1):
                        refined.append(a + (b - a) * s / n_subdiv)
                incr = torch.tensor(refined, dtype=increments.dtype)
            try:
                results = domain.solve_matpatch(
                    increments=incr, **solve_kwargs)
                used_subdiv = n_subdiv
                if n_subdiv > 1:
                    print(f'  converged with {n_subdiv}x '
                          f'subincrementation')
                break
            except Exception as excp:
                last_exc = excp
                print(f'  {n_subdiv}x failed: {excp}')
        if results is None:
            raise RuntimeError(
                f'Newton-Raphson failed up to {subdiv_seq[-1]}x '
                f'subincrementation: {last_exc}')
        if used_subdiv > 1:
            results = tuple(
                r[::used_subdiv] if isinstance(r, torch.Tensor) else r
                for r in results)
        u, f, _, _, _, residual_history = results
    profiler_matpatch.disable()
    print("\n=== SOLVE_MATPATCH METHOD PROFILE ===")
    stats_matpatch = pstats.Stats(profiler_matpatch)
    stats_matpatch.sort_stats('cumulative').print_stats(15)
    # Stop profiling and print results
    # profiler.disable()
    print_status("AFTER_SOLVE_MATPATCH")

    # print("\n" + "="*60)
    # print("PROFILING RESULTS - TOP 15 BOTTLENECKS")
    # print("="*60)
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative').print_stats(15)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save residual history to CSV
    residual_dir = os.path.join(output_dir, 'residual')
    os.makedirs(residual_dir, exist_ok=True)
    for inc, norms in residual_history.items():
        res0 = norms[0] if norms[0] > 0 else 1.0
        rows = np.column_stack((
            np.arange(len(norms)),
            np.array(norms),
            np.array(norms) / res0))
        np.savetxt(
            os.path.join(
                residual_dir,
                f'residual_inc{inc}.csv'),
            rows,
            delimiter=',',
            header='iteration,absolute,relative',
            comments='')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save results
    results = {
        'displacements': u.detach().cpu().numpy(),
        'forces': f.detach().cpu().numpy(),
        'model_directory': model_directory_map,
        'model_name': model_name,
        'material_patch_ids': is_mat_patch.detach().cpu().numpy()}

    output_file = os.path.join(output_dir, "results.pkl")
    with open(output_file, 'wb') as file_handle:
        pkl.dump(results, file_handle)

    print(f"Results saved to {output_file}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot boundary conditions
    domain.plot()

    plot_path = os.path.join(output_dir, "reference_configuration.png")
    plt.savefig(plot_path)
    plt.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot displacement field
    # Get final displacement field
    # Shape: (n_nodes, n_dim)
    u_final = u[-1]

    domain.plot(
        u=u_final,
        node_property=torch.norm(u_final, dim=1),
        title=r'Surrogate: $||\mathbf{u}||_{2}$',
        colorbar=True,
        cmap='viridis',
        vmin=vmin_ref,
        vmax=vmax_ref
    )

    plot_path = os.path.join(output_dir, "disp_field_srg.png")
    plt.savefig(plot_path)
    plt.close()


    u_diff = torch.abs(u[-1] - u_ref[-1])

    domain.plot(
        u=u_diff,
        node_property=torch.norm(u_diff, dim=1),
        title=r'$||\mathbf{u}_{srg} - '
              r'\mathbf{u}_{ref}||_{2}$',
        colorbar=True,
        cmap='viridis',
        vmin=vmin_ref,
        vmax=vmax_ref
    )

    plot_path = os.path.join(output_dir, "disp_field_diff.png")
    plt.savefig(plot_path)
    plt.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot force fields (surrogate and reference)
    f_final = f[-1]
    f_ref_final = f_ref[-1]
    # Shared colorbar limits
    f_all = torch.cat([
        torch.norm(f_final, dim=1),
        torch.norm(f_ref_final, dim=1)])
    vmin_f = f_all.min().item()
    vmax_f = f_all.max().item()

    domain.plot(
        u=u_final,
        node_property=torch.norm(f_final, dim=1),
        title=r'Surrogate: $||\mathbf{f}||_{2}$',
        colorbar=True,
        cmap='plasma',
        vmin=vmin_f,
        vmax=vmax_f)
    plot_path = os.path.join(
        output_dir, 'force_field_srg.png')
    plt.savefig(plot_path)
    plt.close()

    # Restore reference constraints so domain.plot renders
    # without internal-node BC markers
    constraints_srg = domain.constraints.clone()
    displacements_bc_srg = domain.displacements.clone()
    domain.constraints = constraints_ref
    domain.displacements = displacements_bc_ref

    domain.plot(
        u=u_ref_final,
        node_property=torch.norm(f_ref_final, dim=1),
        title=r'Reference: $||\mathbf{f}||_{2}$',
        colorbar=True,
        cmap='plasma',
        vmin=vmin_f,
        vmax=vmax_f)
    plot_path = os.path.join(
        output_dir, 'force_field_ref.png')
    plt.savefig(plot_path)
    plt.close()

    # Restore surrogate constraints
    domain.constraints = constraints_srg
    domain.displacements = displacements_bc_srg

    f_diff = torch.abs(f[-1] - f_ref[-1])

    domain.plot(
        u=u_diff,
        node_property=torch.norm(f_diff, dim=1),
        title=r'$||\mathbf{f}_{pred} - \mathbf{f}_{ref}||_{2}$',
        colorbar=True,
        cmap='viridis'
    )

    plot_path = os.path.join(output_dir, "force_field_difference.png")
    plt.savefig(plot_path)
    plt.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary-only displacement plots
    plot_boundary_overlay(
        nodes=domain.nodes,
        elements=domain.elements,
        u_surrogate=u[-1],
        patch_boundary_nodes_dict=
            patch_boundary_nodes_dict,
        output_path=os.path.join(
            output_dir,
            'boundary_disp_overlay.png'))

    plot_boundary_panels(
        nodes=domain.nodes,
        elements=domain.elements,
        u_surrogate=u[-1],
        patch_boundary_nodes_dict=
            patch_boundary_nodes_dict,
        output_path=os.path.join(
            output_dir,
            'boundary_disp_panels.png'),
        u_reference=u_ref[-1])

    plot_boundary_error_overlay(
        nodes=domain.nodes,
        elements=domain.elements,
        u_surrogate=u[-1],
        u_reference=u_ref[-1],
        patch_boundary_nodes_dict=
            patch_boundary_nodes_dict,
        output_path=os.path.join(
            output_dir,
            'boundary_error_overlay.png'))
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Multi-resolution: 8x8 mesh, no FE border
    # Top-left 4x4 block  -> one 4x4 patch
    # Top-right 4x4 block -> four 2x2 patches
    # Bot-left 4x4 block  -> four 2x2 patches
    # Bot-right 4x4 block -> one 4x4 patch
    run_simulation_surrogate(
        element_type='quad4',
        material_behavior='elastoplastic_nlh',
        mesh_nx=10,
        mesh_ny=10,
        edge_type='all',
        edge_feature_type=(
            'edge_vector', 'relative_disp'),
        # fe_border=0,
        patch_size_x=1,
        patch_size_y=1,
        is_adaptive_timestepping=False,
        adaptive_max_subdiv=8,
        model_name='model_potentialhead',
        # patch_zones=[
        #     # Top-left 4x4: rows 0-3, cols 0-3
        #     {'region': (0, 4, 0, 4),
        #      'patch_size': (4, 4)},
        #     # Top-right 4x4: rows 0-3, cols 4-7
        #     {'region': (0, 4, 4, 8),
        #      'patch_size': (2, 2)},
        #     # Bot-left 4x4: rows 4-7, cols 0-3
        #     {'region': (4, 8, 0, 4),
        #      'patch_size': (2, 2)},
        #     # Bot-right 4x4: rows 4-7, cols 4-7
        #     {'region': (4, 8, 4, 8),
        #      'patch_size': (4, 4)},
        # ]
        )
