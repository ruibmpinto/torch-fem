"""Timing analysis: FE material integration vs GNN surrogate.

Measures wall-clock time of integrate_material (elastic,
hyperelastic) and surrogate_integrate_material (GNN +
jacfwd) for patch sizes 1x1 through 8x8. Writes results
to timing_results.csv.

Notes
-----
Each patch size uses mesh = patch_size * 4 so that
every run has exactly 16 patches, making timing
comparable across sizes.
"""
#
#                                                          Modules
# =============================================================================
# Standard
import csv
import os
import sys
import pathlib
import time
import functools

# Add graphorge to sys.path
graphorge_path = str(
    pathlib.Path(__file__).parents[3]
    / 'graphorge_material_patches' / 'src')
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)

# Third-party
import torch
import numpy as np

# Enable graphorge imports in torchfem
os.environ['TORCHFEM_IMPORT_GRAPHORGE'] = '1'

# Local
from torchfem import Planar
from torchfem.materials import (
    IsotropicElasticityPlaneStrain,
    IsotropicHenckyPlaneStrain,
    IsotropicPlasticityPlaneStrainUMAT,
)
from torchfem.mesh import rect_quad
#
#                                                Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto']
__status__ = 'Development'
# =============================================================================
#
torch.set_default_dtype(torch.float64)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USER_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
SURROGATES_DIR = os.path.join(
    USER_SCRIPTS_DIR, 'matpatch_surrogates')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'timing_results.csv')

N_DIM = 2
N_WARMUP = 2
PATCH_SIZES = list(range(1, 8))
INCREMENTS = torch.linspace(0.0, 1.0, 11)

# Per-model configuration from model_summary.dat
# and model_init_file.pkl inspection
MODEL_CONFIG = {
    1: {
        'edge_type': 'bd',
        'edge_feature_type': ('edge_vector',),
    },
    2: {
        'edge_type': 'all',
        'edge_feature_type': ('edge_vector', 'rel_disp'),
    },
    3: {
        'edge_type': 'all',
        'edge_feature_type': ('edge_vector', 'rel_disp'),
    },
    4: {
        'edge_type': 'all',
        'edge_feature_type': ('edge_vector', 'rel_disp'),
    },
    5: {
        'edge_type': 'all',
        'edge_feature_type': ('edge_vector',),
    },
    6: {
        'edge_type': 'all',
        'edge_feature_type': ('edge_vector', 'rel_disp'),
    },
    7: {
        'edge_type': 'all',
        'edge_feature_type': ('edge_vector', 'rel_disp'),
    },
    # 8: {
    #     'edge_type': 'all',
    #     'edge_feature_type': ('edge_vector', 'rel_disp'),
    # },
}

# =============================================================================
def make_swift_voce_aa2024():
    """Build Swift-Voce AA2024 hardening law.

    Returns
    -------
    e_young : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_f : Callable
        Yield-stress function of equivalent plastic strain.
    sigma_f_prime : Callable
        Derivative of `sigma_f`.

    Notes
    -----
    Parameters match
    `user_scripts/run_simulation.py:508-542` so reference FE
    solves produce the same stress response as the GNN training
    data.
    """
    e_young = 70000.0
    nu = 0.33
    a_s = 798.56
    epsilon_0 = 0.0178
    n_exp = 0.202
    k_0 = 363.84
    q_v = 240.03
    beta = 10.533
    omega = 0.368

    def sigma_f(eps_pl):
        k_s = a_s * (epsilon_0 + eps_pl)**n_exp
        k_v = k_0 + q_v * (
            1.0 - torch.exp(-beta * eps_pl))
        return omega * k_s + (1.0 - omega) * k_v

    def sigma_f_prime(eps_pl):
        dks = a_s * n_exp * (epsilon_0 + eps_pl)**(n_exp - 1.0)
        dkv = q_v * beta * torch.exp(-beta * eps_pl)
        return omega * dks + (1.0 - omega) * dkv
    return e_young, nu, sigma_f, sigma_f_prime


# =============================================================================
def _get_model_dir(patch_size):
    """Return model directory for a given patch size.

    Parameters
    ----------
    patch_size : int
        Patch side length.

    Returns
    -------
    model_dir : str
        Path to model directory.
    """
    ps = f'{patch_size}x{patch_size}'
    base = os.path.join(SURROGATES_DIR, 'elastic', ps)
    model_dir = os.path.join(base, 'model')
    if not os.path.isdir(model_dir):
        model_dir = os.path.join(
            base, 'model_edgestiffness')
    return model_dir


# =============================================================================
def _setup_domain(mesh_n, material):
    """Create a 2D quad4 domain with tension BCs.

    Parameters
    ----------
    mesh_n : int
        Number of elements per side.
    material : object
        Material instance.

    Returns
    -------
    domain : Planar
        FE domain with BCs applied.
    nodes : torch.Tensor
        Nodal coordinates.
    """
    nodes, elements = rect_quad(
        mesh_n + 1, mesh_n + 1)
    domain = Planar(nodes, elements, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Simple tension BCs
    tol = 1e-6
    for i, nc in enumerate(nodes):
        if torch.abs(nc[1]) < tol:
            domain.displacements[i, 1] = 0.0
            domain.constraints[i, 1] = True
            if torch.abs(nc[0]) < tol:
                domain.displacements[i, 0] = 0.0
                domain.constraints[i, 0] = True
        elif torch.abs(nc[1] - 1.0) < tol:
            domain.displacements[i, 1] = 0.05
            domain.constraints[i, 1] = True
    return domain, nodes


# =============================================================================
def _build_patch_data(mesh_n, patch_size, nodes):
    """Build patch assignment and boundary node data.

    Parameters
    ----------
    mesh_n : int
        Number of elements per side.
    patch_size : int
        Patch side length in elements.
    nodes : torch.Tensor
        Nodal coordinates.

    Returns
    -------
    is_mat_patch : torch.Tensor
        Element-to-patch mapping.
    patch_boundary_nodes_dict : dict
        Patch ID to boundary node indices.
    patch_elem_per_dim_map : dict
        Patch ID to [nx, ny] element counts.
    n_patches : int
        Total number of patches.
    patch_internal : set
        Internal patch node IDs.
    """
    num_elements = mesh_n * mesh_n
    n_patches_per_side = mesh_n // patch_size
    n_patches = n_patches_per_side ** 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assign elements to patches
    is_mat_patch = torch.full(
        (num_elements,), -1, dtype=torch.int)
    for elem_idx in range(num_elements):
        ei = elem_idx // mesh_n
        ej = elem_idx % mesh_n
        pi = ei // patch_size
        pj = ej // patch_size
        pid = pi * n_patches_per_side + pj
        is_mat_patch[elem_idx] = pid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build connectivity: node -> set of patch IDs
    n_nodes_per_side = mesh_n + 1
    node_to_patches = {}
    for elem_idx in range(num_elements):
        pid = is_mat_patch[elem_idx].item()
        ei = elem_idx // mesh_n
        ej = elem_idx % mesh_n
        elem_nodes = [
            ei * n_nodes_per_side + ej,
            ei * n_nodes_per_side + ej + 1,
            (ei + 1) * n_nodes_per_side + ej,
            (ei + 1) * n_nodes_per_side + ej + 1,
        ]
        for nid in elem_nodes:
            node_to_patches.setdefault(
                nid, set()).add(pid)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # External boundary nodes
    tol = 1e-6
    external_boundary = set()
    for i, nc in enumerate(nodes):
        if (torch.abs(nc[0]) < tol
                or torch.abs(nc[0] - 1.0) < tol
                or torch.abs(nc[1]) < tol
                or torch.abs(nc[1] - 1.0) < tol):
            external_boundary.add(i)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Patch boundary nodes
    patch_boundary_nodes_dict = {}
    for pid in range(n_patches):
        bnd = set()
        for nid, patches in node_to_patches.items():
            if pid in patches:
                if (len(patches) > 1
                        or nid
                        in external_boundary):
                    bnd.add(nid)
        patch_boundary_nodes_dict[pid] = (
            torch.tensor(
                sorted(bnd), dtype=torch.long))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Internal patch nodes
    all_patch_nodes = set(node_to_patches.keys())
    all_boundary = set()
    for v in patch_boundary_nodes_dict.values():
        all_boundary.update(v.tolist())
    patch_internal = all_patch_nodes - all_boundary
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Patch element dimensions
    patch_elem_per_dim_map = {
        pid: [patch_size, patch_size]
        for pid in range(n_patches)}
    return (
        is_mat_patch, patch_boundary_nodes_dict,
        patch_elem_per_dim_map, n_patches,
        patch_internal)


# =============================================================================
def _make_timer(original_method):
    """Wrap a method to record call durations.

    Parameters
    ----------
    original_method : callable
        Method to wrap.

    Returns
    -------
    wrapper : callable
        Wrapped method.
    timings : list
        List that accumulates elapsed times.
    """
    timings = []

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_method(*args, **kwargs)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
        return result

    return wrapper, timings


# =============================================================================
def time_fe(material, material_name, mesh_n):
    """Time FE integrate_material calls.

    Parameters
    ----------
    material : object
        Material instance.
    material_name : str
        Label for CSV output.
    mesh_n : int
        Elements per side.

    Returns
    -------
    row : dict
        CSV row with timing results.
    """
    domain, _ = _setup_domain(mesh_n, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instrument integrate_material
    orig = domain.integrate_material
    wrapper, timings = _make_timer(orig)
    domain.integrate_material = wrapper
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve
    domain.solve(increments=INCREMENTS, rtol=1e-8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Skip warmup calls
    t = (timings[N_WARMUP:]
         if len(timings) > N_WARMUP
         else timings)
    arr = np.array(t)
    return {
        'method': 'fe',
        'material': material_name,
        'patch_size': '',
        'mesh_size': mesh_n,
        'n_boundary_dofs': 0,
        'mean_time_s': f'{arr.mean():.6e}',
        'std_time_s': f'{arr.std():.6e}',
        'n_calls': len(t),
    }


# =============================================================================
def time_surrogate(patch_size):
    """Time surrogate_integrate_material for one size.

    Parameters
    ----------
    patch_size : int
        Patch side length.

    Returns
    -------
    row : dict
        CSV row with timing results.
    """
    mesh_n = patch_size * 2
    e_young = 110000
    nu = 0.33
    material = IsotropicElasticityPlaneStrain(
        E=e_young, nu=nu)
    domain, nodes = _setup_domain(mesh_n, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build patch data
    (is_mat_patch, patch_bnd_dict,
     patch_elem_map, n_patches,
     patch_internal) = _build_patch_data(
        mesh_n, patch_size, nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Constrain internal patch nodes
    for nid in patch_internal:
        domain.displacements[nid, :] = 0.0
        domain.constraints[nid, :] = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model directory and per-model config
    model_dir = _get_model_dir(patch_size)
    cfg = MODEL_CONFIG[patch_size]
    edge_type = cfg['edge_type']
    edge_feature_type = cfg['edge_feature_type']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instrument surrogate_integrate_material
    orig = domain.surrogate_integrate_material
    wrapper, timings = _make_timer(orig)
    domain.surrogate_integrate_material = wrapper
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary DOFs for one patch: 4p nodes * 2 DOFs
    n_bnd_dofs = 4 * patch_size * N_DIM
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve — no iteration limit
    ps_str = f'{patch_size}x{patch_size}'
    try:
        domain.solve_matpatch(
            is_mat_patch=is_mat_patch,
            increments=INCREMENTS,
            max_iter=200,
            rtol=1e-8,
            verbose=True,
            return_intermediate=True,
            return_volumes=False,
            return_resnorm=True,
            is_stepwise=False,
            model_directory=model_dir,
            patch_boundary_nodes=patch_bnd_dict,
            patch_elem_per_dim=patch_elem_map,
            edge_type=edge_type,
            edge_feature_type=edge_feature_type,
            is_export_stiffness=False,
            patch_size_label=ps_str,
        )
    except Exception as e:
        print(f'  [WARN] {ps_str} solve failed: {e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Skip warmup calls
    t = (timings[N_WARMUP:]
         if len(timings) > N_WARMUP
         else timings)
    if len(t) == 0:
        t = timings
    arr = np.array(t) if len(t) > 0 else np.array(
        [float('nan')])
    return {
        'method': 'surrogate',
        'material': 'elastic',
        'patch_size': ps_str,
        'mesh_size': mesh_n,
        'n_boundary_dofs': n_bnd_dofs,
        'mean_time_s': f'{arr.mean():.6e}',
        'std_time_s': f'{arr.std():.6e}',
        'n_calls': len(t),
    }


# =============================================================================
def _get_model_dir_nlh(patch_size):
    """Return elastoplastic_nlh model directory for a patch size.

    Parameters
    ----------
    patch_size : int
        Patch side length.

    Returns
    -------
    model_dir : str
        Path to trained GNN model directory.
    """
    ps = f'{patch_size}x{patch_size}'
    base = os.path.join(
        SURROGATES_DIR, 'elastoplastic_nlh', ps)
    model_dir = os.path.join(base, 'model')
    return model_dir


# =============================================================================
def time_surrogate_elastoplastic_nlh(patch_size):
    """Time surrogate_integrate_material on elastoplastic_nlh.

    Parameters
    ----------
    patch_size : int
        Patch side length.

    Returns
    -------
    row : dict
        CSV row with timing results.
    """
    mesh_n = patch_size * 2
    e_young, nu, sigma_f, sigma_f_prime = make_swift_voce_aa2024()
    material = IsotropicPlasticityPlaneStrainUMAT(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        is_analytical_tangent=True)
    domain, nodes = _setup_domain(mesh_n, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build patch data
    (is_mat_patch, patch_bnd_dict,
     patch_elem_map, n_patches,
     patch_internal) = _build_patch_data(
        mesh_n, patch_size, nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Constrain internal patch nodes
    for nid in patch_internal:
        domain.displacements[nid, :] = 0.0
        domain.constraints[nid, :] = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model directory and per-model config. Values match
    # matpatch_surrogates/elastoplastic_nlh/1x1/model_summary.dat.
    model_dir = _get_model_dir_nlh(patch_size)
    edge_type = 'all'
    edge_feature_type = ('edge_vector', 'relative_disp')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instrument surrogate_integrate_material
    orig = domain.surrogate_integrate_material
    wrapper, timings = _make_timer(orig)
    domain.surrogate_integrate_material = wrapper
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary DOFs for one patch: 4p nodes * 2 DOFs
    n_bnd_dofs = 4 * patch_size * N_DIM
    ps_str = f'{patch_size}x{patch_size}'
    try:
        domain.solve_matpatch(
            is_mat_patch=is_mat_patch,
            increments=INCREMENTS,
            max_iter=200,
            rtol=1e-8,
            verbose=True,
            return_intermediate=True,
            return_volumes=False,
            return_resnorm=True,
            is_stepwise=True,
            model_directory=model_dir,
            patch_boundary_nodes=patch_bnd_dict,
            patch_elem_per_dim=patch_elem_map,
            edge_type=edge_type,
            edge_feature_type=edge_feature_type,
            is_export_stiffness=False,
            patch_size_label=ps_str,
        )
    except Exception as e:
        print(f'  [WARN] nlh {ps_str} solve failed: {e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Skip warmup calls
    t = (timings[N_WARMUP:]
         if len(timings) > N_WARMUP
         else timings)
    if len(t) == 0:
        t = timings
    arr = (np.array(t) if len(t) > 0
           else np.array([float('nan')]))
    return {
        'method': 'surrogate',
        'material': 'elastoplastic_nlh',
        'patch_size': ps_str,
        'mesh_size': mesh_n,
        'n_boundary_dofs': n_bnd_dofs,
        'mean_time_s': f'{arr.mean():.6e}',
        'std_time_s': f'{arr.std():.6e}',
        'n_calls': len(t),
    }


# =============================================================================
def time_fe_elastoplastic_nlh(mesh_n, is_analytical_tangent):
    """Time FE integrate_material on elastoplastic_nlh.

    Parameters
    ----------
    mesh_n : int
        Elements per side.
    is_analytical_tangent : bool
        Tangent strategy for the plasticity material.

    Returns
    -------
    row : dict
        CSV row with timing results.
    """
    e_young, nu, sigma_f, sigma_f_prime = make_swift_voce_aa2024()
    material = IsotropicPlasticityPlaneStrainUMAT(
        E=e_young, nu=nu,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        is_analytical_tangent=is_analytical_tangent)
    domain, _ = _setup_domain(mesh_n, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instrument integrate_material
    orig = domain.integrate_material
    wrapper, timings = _make_timer(orig)
    domain.integrate_material = wrapper
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve
    domain.solve(increments=INCREMENTS, rtol=1e-8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Skip warmup calls
    t = (timings[N_WARMUP:]
         if len(timings) > N_WARMUP
         else timings)
    arr = np.array(t)
    label = ('fe_analytic_tangent' if is_analytical_tangent
            else 'fe_perturbation_tangent')
    return {
        'method': label,
        'material': 'elastoplastic_nlh',
        'patch_size': '',
        'mesh_size': mesh_n,
        'n_boundary_dofs': 0,
        'mean_time_s': f'{arr.mean():.6e}',
        'std_time_s': f'{arr.std():.6e}',
        'n_calls': len(t),
    }


# =============================================================================
def main():
    """Run all timing measurements and write CSV."""
    rows = []
    fieldnames = [
        'method', 'material', 'patch_size',
        'mesh_size', 'n_boundary_dofs',
        'mean_time_s', 'std_time_s', 'n_calls']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FE timing: mesh = 4*p for each patch_size
    e_young_el = 110000
    nu_el = 0.33
    e_young_he = 20000
    nu_he = 0.33
    for p in PATCH_SIZES:
        mesh_n = p * 2
        print(f'[FE elastic] mesh {mesh_n}x{mesh_n}')
        mat_el = IsotropicElasticityPlaneStrain(
            E=e_young_el, nu=nu_el)
        rows.append(
            time_fe(mat_el, 'elastic', mesh_n))
        print(
            f'[FE hyperelastic] '
            f'mesh {mesh_n}x{mesh_n}')
        mat_he = IsotropicHenckyPlaneStrain(
            E=e_young_he, nu=nu_he)
        rows.append(
            time_fe(mat_he, 'hyperelastic', mesh_n))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Surrogate timing (elastic)
    for p in PATCH_SIZES:
        print(f'[Surrogate elastic] patch {p}x{p}')
        rows.append(time_surrogate(p))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Elastoplastic_nlh fairness benchmark: patch_1x1 only
    mesh_n_nlh = 2
    print(
        f'[FE elastoplastic_nlh analytic] '
        f'mesh {mesh_n_nlh}x{mesh_n_nlh}')
    rows.append(
        time_fe_elastoplastic_nlh(
            mesh_n_nlh, is_analytical_tangent=True))
    print(
        f'[FE elastoplastic_nlh perturbation] '
        f'mesh {mesh_n_nlh}x{mesh_n_nlh}')
    rows.append(
        time_fe_elastoplastic_nlh(
            mesh_n_nlh, is_analytical_tangent=False))
    print('[Surrogate elastoplastic_nlh] patch 1x1')
    rows.append(time_surrogate_elastoplastic_nlh(1))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Results written to {OUTPUT_CSV}')


# =============================================================================
if __name__ == '__main__':
    main()
