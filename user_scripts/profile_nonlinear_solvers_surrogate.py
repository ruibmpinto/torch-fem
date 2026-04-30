"""Benchmark the six nonlinear solvers on the elastoplastic_nlh surrogate.

Builds the same matpatch problem used by ``run_simulation_surrogate.py``
for ``material_behavior='elastoplastic_nlh'`` with uniform 1x1 patches
on a 10x10 quad mesh, then runs ``domain.solve_matpatch(...)`` once per
selected nonlinear solver and records the four metrics:

- ``time_total``: wall-clock from the entry into ``solve_matpatch`` to
  return.
- ``n_iter_total``: sum of iteration counts across all increments.
- ``time_per_iter``: ``time_total / max(n_iter_total, 1)``.
- ``order_of_conv``: median per-increment order of convergence
  estimated by a linear fit of ``log(r_{k+1})`` against ``log(r_k)``
  over iterations satisfying ``0.1 * r_0 > r_k > 10 * atol``.

The surrogate is loaded from
``user_scripts/matpatch_surrogates/elastoplastic_nlh/1x1/model_reference``
by default. Override with ``--model-name``.

Usage
-----
::

    source activate env_torchfem
    cd torch-fem
    python user_scripts/profile_nonlinear_solvers_surrogate.py \\
        --n-increments 10 --n-reps 1

Outputs ``profile_nonlinear_solvers_surrogate_results.csv`` (one row
per ``(solver, n_increments)`` aggregated over repetitions) and
``profile_nonlinear_solvers_surrogate_orders.csv`` (per-increment
order of convergence values).

Functions
---------
build_surrogate_problem
    Construct the 10x10 mesh, 1x1 patch topology, model_directory map
    and prescribed boundary conditions; mirrors the setup in
    run_simulation_surrogate.py.
fit_order_of_convergence
    Estimate p from a residual sequence (window-restricted log fit).
classify_order
    Map p to a categorical label.
run_one
    Execute one ``(solver, repetition)`` measurement.
main
    CLI entry point.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import argparse
import csv
import math
import os
import pathlib
import statistics
import sys
import time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add graphorge and torch-fem source paths.
_torch_fem_root = pathlib.Path(__file__).resolve().parents[1]
_graphorge_root = _torch_fem_root.parent / 'graphorge' / 'src'
for _p in (_torch_fem_root / 'src', _graphorge_root):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
# Enable graphorge imports inside torchfem.
os.environ['TORCHFEM_IMPORT_GRAPHORGE'] = '1'
# Third-party
import numpy as np
import torch
# Local
from torchfem import Planar
from torchfem.materials import IsotropicPlasticityPlaneStrain
from torchfem.mesh import rect_quad
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
torch.set_default_dtype(torch.float64)

SOLVERS_DEFAULT = [
    'newton_raphson', 'damped_picard', 'anderson',
    'broyden', 'jfnk', 'rand_subspace_newton',
]

SOLVER_CONFIGS = {
    'newton_raphson': dict(
        rtol=1e-8, atol=1e-6, max_iter=100, opts={}),
    'damped_picard': dict(
        rtol=1e-5, atol=1e-3, max_iter=200,
        opts={'damping': 1.0}),
    'anderson': dict(
        rtol=1e-5, atol=1e-3, max_iter=200,
        opts={'beta': 1.0, 'm': 8}),
    'broyden': dict(
        rtol=1e-5, atol=1e-3, max_iter=100,
        opts={'jacobian_refresh_period': 3}),
    'jfnk': dict(
        rtol=1e-5, atol=1e-3, max_iter=50,
        opts={}),
    'rand_subspace_newton': dict(
        rtol=1e-5, atol=1e-3, max_iter=30,
        opts={'block_size': 50}),
}


# =============================================================================
def make_swift_voce():
    """Build the Swift-Voce AA2024 hardening closures."""
    a_s = 798.56
    epsilon_0 = 0.0178
    n_sv = 0.202
    k_0 = 363.84
    q_v = 240.03
    beta = 10.533
    omega = 0.368

    def sigma_f(eps_pl):
        k_s = a_s * (epsilon_0 + eps_pl)**n_sv
        k_v = k_0 + q_v * (1.0 - torch.exp(-beta * eps_pl))
        return omega * k_s + (1.0 - omega) * k_v

    def sigma_f_prime(eps_pl):
        dks = a_s * n_sv * (epsilon_0 + eps_pl)**(n_sv - 1.0)
        dkv = q_v * beta * torch.exp(-beta * eps_pl)
        return omega * dks + (1.0 - omega) * dkv
    return sigma_f, sigma_f_prime


# =============================================================================
def build_surrogate_problem(
        mesh_nx=10, mesh_ny=10, patch_size_x=1, patch_size_y=1,
        model_name='model_reference', top_disp=0.05):
    """Build the elastoplastic_nlh / 1x1 patch matpatch problem.

    Mirrors the setup in ``run_simulation_surrogate.py`` for
    ``material_behavior='elastoplastic_nlh'`` with a single patch zone
    covering the full mesh at 1x1 resolution.

    Parameters
    ----------
    mesh_nx, mesh_ny : int
        Number of elements per direction.
    patch_size_x, patch_size_y : int
        Patch size in elements (1x1 = one quad per patch).
    model_name : str
        Sub-directory under
        ``user_scripts/matpatch_surrogates/elastoplastic_nlh/{NxN}/``
        from which to load the surrogate model.
    top_disp : float
        Prescribed vertical displacement on the top edge.

    Returns
    -------
    setup : dict
        Keys: ``domain``, ``is_mat_patch``, ``model_directory``,
        ``patch_boundary_nodes_dict``, ``patch_elem_per_dim``,
        ``patch_resolution``, ``edge_feature_type``, ``is_stepwise``.
    """
    nodes, elements = rect_quad(mesh_nx + 1, mesh_ny + 1)
    sigma_f, sigma_f_prime = make_swift_voce()
    material = IsotropicPlasticityPlaneStrain(
        E=70000.0, nu=0.33,
        sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
        max_iter=50)
    domain = Planar(nodes, elements, material)
    num_elements = elements.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Single zone covering the full mesh at the requested patch size.
    fe_border = 0
    psx, psy = patch_size_x, patch_size_y
    surr_nx = mesh_nx - 2 * fe_border
    surr_ny = mesh_ny - 2 * fe_border
    zone_nx = surr_nx // psx
    zone_ny = surr_ny // psy
    is_mat_patch = torch.full((num_elements,), -1, dtype=torch.int)
    patch_resolution = {}
    for elem_idx in range(num_elements):
        ei = elem_idx // mesh_nx
        ej = elem_idx % mesh_nx
        if (fe_border <= ei < fe_border + surr_ny
                and fe_border <= ej < fe_border + surr_nx):
            pi = (ei - fe_border) // psy
            pj = (ej - fe_border) // psx
            pid = pi * zone_nx + pj
            is_mat_patch[elem_idx] = pid
            patch_resolution[pid] = (psx, psy)
    n_patches = zone_nx * zone_ny
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Single resolution -> single model directory.
    surrogates_dir = (
        _torch_fem_root / 'user_scripts' / 'matpatch_surrogates')
    model_directory = str(
        surrogates_dir / 'elastoplastic_nlh'
        / f'{psx}x{psy}' / model_name)
    if not os.path.isdir(model_directory):
        raise FileNotFoundError(
            f'Surrogate model directory not found: {model_directory}')
    patch_elem_per_dim = [psx, psy]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-patch boundary node lists. Mirrors the algorithm in
    # run_simulation_surrogate.py.
    node_to_patches = {}
    for elem_idx in range(num_elements):
        pid = is_mat_patch[elem_idx].item()
        if pid < 0:
            continue
        for nid in elements[elem_idx, :4].tolist():
            node_to_patches.setdefault(nid, set()).add(pid)
    fe_nodes = set()
    for elem_idx in range(num_elements):
        if is_mat_patch[elem_idx].item() == -1:
            for nid in elements[elem_idx, :4].tolist():
                fe_nodes.add(nid)
    external_boundary_nodes = set()
    tol = 1e-6
    for i, nc in enumerate(nodes):
        if (torch.abs(nc[0]) < tol
                or torch.abs(nc[0] - 1.0) < tol
                or torch.abs(nc[1]) < tol
                or torch.abs(nc[1] - 1.0) < tol):
            external_boundary_nodes.add(i)
    patch_boundary_nodes_dict = {}
    for pid in range(n_patches):
        bset = set()
        for nid, patches in node_to_patches.items():
            if pid not in patches:
                continue
            if (len(patches) > 1
                    or nid in external_boundary_nodes
                    or nid in fe_nodes):
                bset.add(nid)
        patch_boundary_nodes_dict[pid] = torch.tensor(
            sorted(bset), dtype=torch.long)
    # Internal patch nodes (constrained to zero for the surrogate).
    all_patch_nodes = set(node_to_patches.keys())
    boundary_union = set()
    for s in patch_boundary_nodes_dict.values():
        boundary_union.update(s.tolist())
    patch_internal_nodes = (
        all_patch_nodes - boundary_union - fe_nodes)
    patch_internal_nodes = torch.tensor(
        sorted(patch_internal_nodes), dtype=torch.long)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # External tensile boundary conditions.
    for i, nc in enumerate(nodes):
        if torch.abs(nc[1]) < tol:
            domain.displacements[i, 1] = 0.0
            domain.constraints[i, 1] = True
            if torch.abs(nc[0]) < tol:
                domain.displacements[i, 0] = 0.0
                domain.constraints[i, 0] = True
        elif torch.abs(nc[1] - 1.0) < tol:
            domain.displacements[i, 1] = top_disp
            domain.constraints[i, 1] = True
    # Lock internal patch nodes for the surrogate solve.
    for nid in patch_internal_nodes:
        domain.displacements[nid, :] = 0.0
        domain.constraints[nid, :] = True
    return {
        'domain': domain,
        'is_mat_patch': is_mat_patch,
        'model_directory': model_directory,
        'patch_boundary_nodes_dict': patch_boundary_nodes_dict,
        'patch_elem_per_dim': patch_elem_per_dim,
        'patch_resolution_arg': None,
        'edge_feature_type': ('edge_vector', 'relative_disp'),
        'is_stepwise': True,
        'n_patches': n_patches,
        'n_dof': domain.n_nod * domain.n_dim,
    }


# =============================================================================
def fit_order_of_convergence(residual_history, atol):
    """Linear-fit slope of log(r_{k+1}) vs log(r_k) over the meat window."""
    rh = list(residual_history)
    if len(rh) < 4:
        return float('nan')
    r0 = rh[0]
    if r0 <= 0:
        return float('nan')
    upper = 0.1 * r0
    lower = max(10.0 * atol, 1e-30)
    pairs = []
    for k in range(len(rh) - 1):
        rk, rk1 = rh[k], rh[k + 1]
        if rk <= 0 or rk1 <= 0:
            continue
        if not (lower < rk < upper):
            continue
        pairs.append((math.log(rk), math.log(rk1)))
    if len(pairs) < 3:
        return float('nan')
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    slope, _ = np.polyfit(xs, ys, deg=1)
    return float(slope)


# =============================================================================
def classify_order(p):
    """Categorical label for the order of convergence."""
    if math.isnan(p):
        return '?'
    if p <= 1.2:
        return 'linear'
    if p < 1.8:
        return 'superlinear'
    return 'quadratic'


# =============================================================================
def run_one(setup, solver_name, n_increments, seed):
    """Run one timed surrogate solve and return the metrics record.

    Parameters
    ----------
    setup : dict
        Output of ``build_surrogate_problem``.
    solver_name : str
    n_increments : int
        Number of load increments (sequence ``[0, ..., 1]`` of length
        ``n_increments + 1``).
    seed : int
        Random seed for stochastic solvers.

    Returns
    -------
    dict
        Metrics record.
    """
    cfg = SOLVER_CONFIGS[solver_name]
    increments = torch.linspace(0.0, 1.0, n_increments + 1)
    torch.manual_seed(seed)
    opts = dict(cfg['opts'])
    if solver_name == 'rand_subspace_newton':
        opts['seed'] = seed
        # block_size defaults to 8; tune relative to the boundary DOFs.
        # 1x1 patches expose few boundary DOFs, so leave the default.
    domain = setup['domain']
    start = time.perf_counter()
    converged = True
    residual_history = {}
    err = ''
    try:
        res = domain.solve_matpatch(
            is_mat_patch=setup['is_mat_patch'],
            increments=increments,
            max_iter=cfg['max_iter'],
            rtol=cfg['rtol'], atol=cfg['atol'],
            verbose=False,
            return_intermediate=True,
            return_resnorm=True,
            return_volumes=False,
            is_stepwise=setup['is_stepwise'],
            model_directory=setup['model_directory'],
            patch_boundary_nodes=setup['patch_boundary_nodes_dict'],
            patch_elem_per_dim=setup['patch_elem_per_dim'],
            patch_resolution=setup['patch_resolution_arg'],
            edge_type='all',
            edge_feature_type=setup['edge_feature_type'],
            is_export_stiffness=False,
            stiffness_output_dir=None,
            patch_size_label=None,
            is_jacfwd_parallel=False,
            nonlinear_solver=solver_name,
            nonlinear_solver_opts=opts,
        )
        residual_history = res[-1]
    except Exception as excp:
        converged = False
        err = f'{type(excp).__name__}: {excp}'
    elapsed = time.perf_counter() - start
    n_iter_total = sum(len(v) for v in residual_history.values())
    final_res_norm = float('nan')
    if residual_history:
        last_inc = max(residual_history.keys())
        last_hist = residual_history[last_inc]
        if last_hist:
            final_res_norm = float(last_hist[-1])
    per_inc_orders = []
    for k in sorted(residual_history.keys()):
        per_inc_orders.append(
            (k, fit_order_of_convergence(
                residual_history[k], cfg['atol'])))
    if per_inc_orders:
        finite = [p for _, p in per_inc_orders if not math.isnan(p)]
        order_of_conv = (
            float(statistics.median(finite)) if finite
            else float('nan'))
    else:
        order_of_conv = float('nan')
    return {
        'solver': solver_name,
        'n_increments': n_increments,
        'n_dof': setup['n_dof'],
        'n_patches': setup['n_patches'],
        'n_iter_total': int(n_iter_total),
        'time_total': float(elapsed),
        'time_per_iter': float(
            elapsed / max(n_iter_total, 1)),
        'order_of_conv': order_of_conv,
        'order_label': classify_order(order_of_conv),
        'converged': converged,
        'final_res_norm': final_res_norm,
        'per_inc_orders': per_inc_orders,
        'error': err,
    }


# =============================================================================
def aggregate(records):
    """Aggregate repeated runs into mean/std per solver."""
    grouped = {}
    for rec in records:
        grouped.setdefault(rec['solver'], []).append(rec)
    rows = []
    for solver, reps in grouped.items():
        time_totals = [r['time_total'] for r in reps]
        time_per_iter = [r['time_per_iter'] for r in reps]
        ref = reps[0]
        rows.append({
            'solver': solver,
            'n_increments': ref['n_increments'],
            'n_dof': ref['n_dof'],
            'n_patches': ref['n_patches'],
            'n_iter_total': ref['n_iter_total'],
            'time_total_mean_s': float(statistics.mean(time_totals)),
            'time_total_std_s': (
                float(statistics.stdev(time_totals))
                if len(time_totals) > 1 else 0.0),
            'time_per_iter_mean_s': (
                float(statistics.mean(time_per_iter))),
            'time_per_iter_std_s': (
                float(statistics.stdev(time_per_iter))
                if len(time_per_iter) > 1 else 0.0),
            'order_of_conv': ref['order_of_conv'],
            'order_label': ref['order_label'],
            'converged': all(r['converged'] for r in reps),
            'final_res_norm': ref['final_res_norm'],
            'error': ref['error'],
        })
    rows.sort(key=lambda r: r['solver'])
    return rows


# =============================================================================
def write_summary_csv(rows, out_path):
    """Write aggregated rows to CSV."""
    fieldnames = [
        'solver', 'n_increments', 'n_dof', 'n_patches',
        'n_iter_total', 'time_total_mean_s', 'time_total_std_s',
        'time_per_iter_mean_s', 'time_per_iter_std_s',
        'order_of_conv', 'order_label', 'converged',
        'final_res_norm', 'error',
    ]
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
def write_per_increment_orders_csv(records, out_path):
    """Write per-increment order-of-convergence values to CSV."""
    seen = set()
    fieldnames = ['solver', 'increment', 'order_of_conv']
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            if rec['solver'] in seen:
                continue
            seen.add(rec['solver'])
            for inc, p in rec['per_inc_orders']:
                writer.writerow({
                    'solver': rec['solver'],
                    'increment': inc,
                    'order_of_conv': p,
                })


# =============================================================================
def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            'Benchmark the six nonlinear solvers on the '
            'elastoplastic_nlh surrogate (1x1 patches, 10x10 mesh, '
            'model_reference checkpoint).'))
    parser.add_argument(
        '--solvers', nargs='+', default=SOLVERS_DEFAULT,
        choices=SOLVERS_DEFAULT,
        help='Solvers to benchmark.')
    parser.add_argument(
        '--n-increments', type=int, default=10,
        help='Number of load increments. The reference simulation '
             'uses 50; smaller values run faster.')
    parser.add_argument(
        '--n-reps', type=int, default=1,
        help='Repetitions per solver for time stats. Each rep is '
             'expensive; default 1.')
    parser.add_argument(
        '--mesh-nx', type=int, default=10)
    parser.add_argument(
        '--mesh-ny', type=int, default=10)
    parser.add_argument(
        '--patch-size-x', type=int, default=1)
    parser.add_argument(
        '--patch-size-y', type=int, default=1)
    parser.add_argument(
        '--model-name', type=str, default='model_reference')
    parser.add_argument(
        '--top-disp', type=float, default=0.05)
    parser.add_argument(
        '--out-dir', default=os.path.dirname(
            os.path.abspath(__file__)),
        help='Directory for output CSVs.')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(
        args.out_dir,
        'profile_nonlinear_solvers_surrogate_results.csv')
    orders_path = os.path.join(
        args.out_dir,
        'profile_nonlinear_solvers_surrogate_orders.csv')
    print('Building surrogate problem ...')
    setup = build_surrogate_problem(
        mesh_nx=args.mesh_nx, mesh_ny=args.mesh_ny,
        patch_size_x=args.patch_size_x,
        patch_size_y=args.patch_size_y,
        model_name=args.model_name,
        top_disp=args.top_disp)
    print(
        f'  domain n_dof = {setup["n_dof"]}, '
        f'n_patches = {setup["n_patches"]}, '
        f'increments = {args.n_increments}')
    print(f'  model = {setup["model_directory"]}')
    all_records = []
    for solver in args.solvers:
        print(f'-> {solver} ({args.n_reps} rep(s))')
        for rep in range(args.n_reps):
            # Each rep needs a freshly built domain because the solve
            # mutates ``self.K`` and stepwise hidden state across the
            # increment loop. Rebuild from scratch.
            setup = build_surrogate_problem(
                mesh_nx=args.mesh_nx, mesh_ny=args.mesh_ny,
                patch_size_x=args.patch_size_x,
                patch_size_y=args.patch_size_y,
                model_name=args.model_name,
                top_disp=args.top_disp)
            rec = run_one(
                setup, solver, args.n_increments, seed=rep)
            all_records.append(rec)
            status = 'OK' if rec['converged'] else 'FAIL'
            tag = (
                f't={rec["time_total"]:.2f}s '
                f'iters={rec["n_iter_total"]} '
                f'r_final={rec["final_res_norm"]:.3e}')
            extra = (
                '' if rec['converged']
                else f' err={rec["error"]}')
            print(f'   rep {rep}: {status} {tag}{extra}')
    rows = aggregate(all_records)
    write_summary_csv(rows, summary_path)
    write_per_increment_orders_csv(all_records, orders_path)
    print()
    print(f'{"solver":<22} {"iters":>6} {"time(s)":>10} '
          f'{"t/iter(s)":>10} {"order":>6} {"label":>12} '
          f'{"conv":>6}')
    print('-' * 80)
    for row in rows:
        print(
            f'{row["solver"]:<22} '
            f'{row["n_iter_total"]:>6d} '
            f'{row["time_total_mean_s"]:>10.4f} '
            f'{row["time_per_iter_mean_s"]:>10.5f} '
            f'{row["order_of_conv"]:>6.2f} '
            f'{row["order_label"]:>12} '
            f'{str(row["converged"]):>6}')
    print()
    print(f'Summary CSV: {summary_path}')
    print(f'Per-inc order CSV: {orders_path}')


# =============================================================================
if __name__ == '__main__':
    main()
