"""Benchmark Jacobian methods for elastoplastic_nlh GNN surrogate.

Compares wall-clock time of four Jacobian computation strategies on
the boundary stiffness matrix dR/du for each trained patch model in
matpatch_surrogates/elastoplastic_nlh/.

Methods
-------
jacfwd_par
    Production baseline: torch.func.jacfwd, all tangent vectors at
    once (vmap-parallelized).
jacrev
    torch.func.jacrev (reverse-mode AD via vmap).
jvp_loop
    Column-by-column torch.func.jvp.
fd_fwd
    One-sided finite differences in float64.

Functions
---------
detach_hidden_states
    Recursively detach tensors inside a hidden-state container.
load_hidden_into_model
    Restore a hidden-state snapshot onto the GNN model.
benchmark_jacobian_methods
    Time the four variants on a single forward closure.
make_benchmark_wrapper
    Wrap surrogate_integrate_material to benchmark on the first
    patch of every call.
run_one_patch_size
    Run solve_matpatch with the wrapper for one patch size.
main
    Iterate over PATCH_SIZES, write CSV, print summary.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import csv
import os
import pathlib
import sys
import time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
graphorge_path = str(
    pathlib.Path(SCRIPT_DIR).parents[2]
    / 'graphorge' / 'src')
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
os.environ['TORCHFEM_IMPORT_GRAPHORGE'] = '1'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
import torch
from torch import func as torch_func
# Local
from run_timing import (
    _build_patch_data,
    _get_model_dir_nlh,
    _setup_domain,
)
from torchfem.base import forward_graph
from torchfem.materials import IsotropicElasticityPlaneStrain
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto']
__status__ = 'Development'
# =============================================================================
#
torch.set_default_dtype(torch.float64)

PATCH_SIZES = [1, 2, 5, 8]
N_DIM = 2
EDGE_TYPE = 'all'
EDGE_FEATURE_TYPE = ('edge_vector', 'relative_disp')
N_REPS = 20
N_WARMUP_REPS = 5
OUTPUT_CSV = os.path.join(
    SCRIPT_DIR, 'jacobian_benchmark_results.csv')

METHOD_LABELS = ('fwd', 'jacfwd_par', 'jacrev',
                 'jvp_loop', 'fd_fwd')


# =============================================================================
def detach_hidden_states(states):
    """Recursively detach tensors in a hidden-state container.

    Parameters
    ----------
    states : {dict, list, torch.Tensor, None}
        Hidden-state structure returned by the GNN.

    Returns
    -------
    detached : same structure
        With every tensor detached from its computation graph.
    """
    if isinstance(states, dict):
        return {k: detach_hidden_states(v)
                for k, v in states.items()}
    if isinstance(states, list):
        return [detach_hidden_states(item) for item in states]
    if torch.is_tensor(states):
        return states.detach()
    return states


# =============================================================================
def load_hidden_into_model(model, hidden):
    """Push a hidden-state snapshot back onto the model components.

    Parameters
    ----------
    model : GNNEPDBaseModel
        Surrogate GNN model with stepwise mode enabled.
    hidden : dict
        Hidden-state dict matching the schema of
        `model._gnn_epd_model._hidden_states`.
    """
    epd = model._gnn_epd_model
    epd._hidden_states = hidden
    if 'encoder' in hidden:
        epd._encoder._hidden_states = hidden['encoder']
    if 'processor' in hidden:
        epd._processor._hidden_states = hidden['processor']
        for li, layer in enumerate(
                epd._processor._processor):
            lk = f'layer_{li}'
            if lk in hidden['processor']:
                layer._hidden_states = (
                    hidden['processor'][lk])
    if 'decoder' in hidden:
        epd._decoder._hidden_states = hidden['decoder']


# =============================================================================
def _restore_factory(model, h0_snapshot, is_stepwise):
    """Build a closure that restores the hidden-state snapshot.

    Parameters
    ----------
    model : GNNEPDBaseModel
        Surrogate model.
    h0_snapshot : {dict, None}
        Detached hidden state captured before benchmarking.
    is_stepwise : bool
        Whether the model is in stepwise mode.

    Returns
    -------
    restore : callable
        Reloads h0 onto the model. No-op if h0 is None.
    """
    def restore():
        if is_stepwise and h0_snapshot is not None:
            load_hidden_into_model(
                model,
                copy.deepcopy(
                    detach_hidden_states(h0_snapshot)))
    return restore


# =============================================================================
def _time_method(fn, n_reps, n_warmup):
    """Time a callable repeatedly and return statistics.

    Parameters
    ----------
    fn : callable
        Zero-argument function whose call is timed.
    n_reps : int
        Number of timed repetitions to record.
    n_warmup : int
        Number of warmup calls to discard.

    Returns
    -------
    times : list[float]
        Per-call wall-clock seconds of timed reps.
    last_value : object
        Return value of the final call.
    """
    for _ in range(n_warmup):
        last_value = fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        last_value = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times, last_value


# =============================================================================
def benchmark_jacobian_methods(
        forward_aux, boundary_u, n_boundary, n_dim,
        restore_h0, is_stepwise):
    """Benchmark four Jacobian variants on a forward closure.

    Parameters
    ----------
    forward_aux : callable
        Maps boundary displacement (n_boundary, n_dim) to predicted
        force (n_boundary, n_dim). Returns either a Tensor or
        (Tensor, aux) depending on `is_stepwise`.
    boundary_u : torch.Tensor
        Operating-point displacement of shape (n_boundary, n_dim).
    n_boundary : int
        Number of boundary nodes.
    n_dim : int
        Spatial dimension.
    restore_h0 : callable
        Restores model hidden state to the operating point.
    is_stepwise : bool
        If True, `forward_aux` returns (pred, hidden); use has_aux
        in jacfwd/jacrev.

    Returns
    -------
    results : dict
        Per-method dict with 'times' (list[float]) and 'K'
        (torch.Tensor) entries.
    """
    n_dof = n_boundary * n_dim
    has_aux = is_stepwise

    def fwd_only(u):
        out = forward_aux(u)
        return out[0] if has_aux else out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 0. Forward only (no Jacobian)
    def call_fwd():
        restore_h0()
        with torch.no_grad():
            return fwd_only(boundary_u).detach().clone()
    times_fwd, f_fwd = _time_method(
        call_fwd, N_REPS, N_WARMUP_REPS)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. jacfwd parallel
    def call_jacfwd_par():
        restore_h0()
        if has_aux:
            j, _ = torch_func.jacfwd(
                forward_aux, has_aux=True)(boundary_u)
        else:
            j = torch_func.jacfwd(forward_aux)(boundary_u)
        return j.view(n_dof, n_dof).detach().clone()
    times_par, k_par = _time_method(
        call_jacfwd_par, N_REPS, N_WARMUP_REPS)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. jacrev
    def call_jacrev():
        restore_h0()
        if has_aux:
            j, _ = torch_func.jacrev(
                forward_aux, has_aux=True)(boundary_u)
        else:
            j = torch_func.jacrev(forward_aux)(boundary_u)
        return j.view(n_dof, n_dof).detach().clone()
    times_rev, k_rev = _time_method(
        call_jacrev, N_REPS, N_WARMUP_REPS)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. jvp loop
    def call_jvp_loop():
        cols = []
        for i in range(n_dof):
            restore_h0()
            tangent = torch.zeros(
                n_dof, dtype=boundary_u.dtype)
            tangent[i] = 1.0
            tangent = tangent.view(n_boundary, n_dim)
            _, jvp_col = torch_func.jvp(
                fwd_only, (boundary_u,), (tangent,))
            cols.append(jvp_col.flatten().detach())
        return torch.stack(cols, dim=1)
    times_jvp, k_jvp = _time_method(
        call_jvp_loop, N_REPS, N_WARMUP_REPS)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. FD forward (one-sided), float64
    eps = (np.sqrt(np.finfo(np.float64).eps)
           * max(boundary_u.abs().max().item(), 1.0))

    def call_fd_fwd():
        with torch.no_grad():
            restore_h0()
            f0 = fwd_only(boundary_u).flatten()
            cols = []
            u_flat = boundary_u.flatten().clone()
            for i in range(n_dof):
                restore_h0()
                u_pert_flat = u_flat.clone()
                u_pert_flat[i] += eps
                u_pert = u_pert_flat.view(
                    n_boundary, n_dim)
                f_pert = fwd_only(u_pert).flatten()
                cols.append((f_pert - f0) / eps)
            return torch.stack(cols, dim=1)
    times_fd, k_fd = _time_method(
        call_fd_fwd, N_REPS, N_WARMUP_REPS)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return {
        'fwd': {'times': times_fwd, 'K': None},
        'jacfwd_par': {'times': times_par, 'K': k_par},
        'jacrev': {'times': times_rev, 'K': k_rev},
        'jvp_loop': {'times': times_jvp, 'K': k_jvp},
        'fd_fwd': {'times': times_fd, 'K': k_fd},
    }


# =============================================================================
def _init_hidden_states(model):
    """Build an empty per-patch hidden-state dict for stepwise mode.

    Parameters
    ----------
    model : GNNEPDBaseModel
        Surrogate with stepwise mode enabled.

    Returns
    -------
    hidden : dict
        Nested dict with 'encoder', 'processor' (one subkey per
        message-passing layer), and 'decoder' entries, each holding
        `{'node': None, 'edge': None, 'global': None}`.
    """
    n_msg = model._n_message_steps
    processor_hidden = {
        f'layer_{li}': {
            'node': None, 'edge': None, 'global': None}
        for li in range(n_msg)}
    return {
        'encoder': {
            'node': None, 'edge': None, 'global': None},
        'processor': processor_hidden,
        'decoder': {
            'node': None, 'edge': None, 'global': None},
    }


# =============================================================================
def run_one_patch_size(patch_size):
    """Benchmark Jacobian methods on one patch of one trained model.

    No FE solver is invoked. The domain is used only to build the
    graph topology for the first patch via `_build_graph_topology`
    (mirroring the setup inside `solve_matpatch`, base.py:1880-1890).

    Parameters
    ----------
    patch_size : int
        Patch side length (1, 2, 5, or 8).

    Returns
    -------
    summary : dict
        Per-method timing statistics plus correctness diagnostics.
    """
    mesh_n = patch_size * 2
    # Placeholder material: only required by the Planar constructor;
    # FE integration is never triggered in this benchmark.
    material = IsotropicElasticityPlaneStrain(
        E=110000.0, nu=0.33)
    domain, nodes = _setup_domain(mesh_n, material)
    (is_mat_patch, patch_bnd_dict, patch_elem_map,
     n_patches, patch_internal) = _build_patch_data(
        mesh_n, patch_size, nodes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load surrogate and configure for stepwise inference
    model_dir = _get_model_dir_nlh(patch_size)
    ps_str = f'{patch_size}x{patch_size}'
    model = domain._load_Graphorge_model(
        model_directory=model_dir, device_type='cpu')
    model._save_time_series_attrs()
    model.set_rnn_mode(is_stepwise=True)
    if hasattr(model, 'prepare_fast_scalers'):
        model.prepare_fast_scalers()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build graph topology for first patch only (mirror base.py:1868-1890)
    domain._edges_indexes = {}
    domain.patch_bd_nodes = patch_bnd_dict
    idx_patch = 0
    domain._build_graph_topology(
        idx_patch, patch_elem_map[idx_patch],
        edge_type=EDGE_TYPE)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary node layout + coordinate normalization
    bnd_ids = domain.patch_bd_nodes[idx_patch]
    n_boundary = len(bnd_ids)
    boundary_coords_ref = domain.nodes[bnd_ids]
    edges_indexes = domain._edges_indexes[
        f'patch_{idx_patch}']
    coords_min = boundary_coords_ref.min(dim=0).values
    coords_max = boundary_coords_ref.max(dim=0).values
    L = torch.clamp(coords_max - coords_min, min=1e-12)
    coords_scaled = (
        boundary_coords_ref - coords_min) / L
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Operating point: zero boundary displacement at t=0
    boundary_u = torch.zeros(
        n_boundary, N_DIM, dtype=torch.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial hidden-state snapshot (empty dict == step-0 state)
    h0 = _init_hidden_states(model)
    load_hidden_into_model(model, copy.deepcopy(h0))
    restore_h0 = _restore_factory(model, h0, True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward closure matching surrogate_integrate_material
    # (base.py:1552-1599), stepwise branch (has_aux=True).
    def forward_aux(disp_boundary):
        disp_scaled = disp_boundary / L
        pred_scaled, _ = forward_graph(
            model=model,
            disps=disp_scaled,
            coords_ref=coords_scaled,
            edges_indexes=edges_indexes,
            n_dim=domain.n_dim,
            edge_feature_type=EDGE_FEATURE_TYPE)
        pred_real = pred_scaled * L
        return pred_real, pred_real.detach()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bench = benchmark_jacobian_methods(
        forward_aux=forward_aux,
        boundary_u=boundary_u,
        n_boundary=n_boundary,
        n_dim=domain.n_dim,
        restore_h0=restore_h0,
        is_stepwise=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_dof = n_boundary * N_DIM
    summary = {
        'patch_size': ps_str,
        'n_dof_boundary': n_dof,
    }
    k_ref = bench['jacfwd_par']['K']
    for label in METHOD_LABELS:
        ts = bench[label]['times']
        arr = np.array(ts)
        summary[f'{label}_mean_s'] = float(arr.mean())
        summary[f'{label}_std_s'] = float(arr.std())
        summary[f'{label}_n'] = int(arr.size)
        k_alt = bench[label]['K']
        if k_alt is None:
            summary[f'{label}_relerr'] = float('nan')
        else:
            num = (k_alt - k_ref).norm().item()
            den = max(k_ref.norm().item(), 1e-30)
            summary[f'{label}_relerr'] = num / den
    return summary


# =============================================================================
def main():
    """Run benchmark across PATCH_SIZES, write CSV, print summary."""
    rows = []
    for p in PATCH_SIZES:
        print(f'\n=== Benchmarking patch size {p}x{p} '
              f'(n_dof={4*p*N_DIM}) ===')
        try:
            summary = run_one_patch_size(p)
        except Exception as e:
            print(f'  [ERROR] failed for {p}x{p}: {e}')
            continue
        rows.append(summary)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for label in METHOD_LABELS:
            mean_s = summary[f'{label}_mean_s']
            std_s = summary[f'{label}_std_s']
            n = summary[f'{label}_n']
            relerr = summary[f'{label}_relerr']
            print(f'  {label:>11s}: '
                  f'{mean_s*1e3:8.3f} ms '
                  f'+- {std_s*1e3:7.3f} ms  '
                  f'(n={n:3d}, relerr={relerr:.2e})')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fieldnames = ['patch_size', 'n_dof_boundary']
    for label in METHOD_LABELS:
        fieldnames.extend([
            f'{label}_mean_s',
            f'{label}_std_s',
            f'{label}_n',
            f'{label}_relerr'])
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote {OUTPUT_CSV}')


# =============================================================================
if __name__ == '__main__':
    main()
