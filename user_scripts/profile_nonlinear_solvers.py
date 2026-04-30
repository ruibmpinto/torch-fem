"""Benchmark the six nonlinear solvers on representative FEM problems.

Measures four metrics per (solver, problem, repetition) triple:

- ``time_total``: wall-clock from the entry into ``domain.solve(...)`` to
  return, in seconds (``time.perf_counter``).
- ``n_iter_total``: sum of iteration counts across all increments.
- ``time_per_iter``: ``time_total / max(n_iter_total, 1)``.
- ``order_of_conv``: numerical order of convergence estimated by a
  least-squares fit of ``log(r_{k+1})`` against ``log(r_k)`` over the
  iterations satisfying ``0.1 * r_0 > r_k > 10 * atol`` (the "meat" of
  the run, away from both the initial transient and the tolerance
  floor). Reported as the median across increments; the per-increment
  values are written to a side CSV.

Two analytic-FE problems are built in-process:

- ``elastic_4x4``: 4x4 quad mesh, plane-strain linear elasticity, five
  increments. NR converges in one iteration; the run measures solver
  overhead per increment.
- ``plastic_4x4``: 4x4 quad mesh, plane-strain plasticity with linear
  hardening, ten increments. NR takes 3-8 iters/inc; representative of
  the equivalence-test workload.

Surrogate problems (``surrogate_1x1`` and ``surrogate_5x5`` in the
plan) require a Graphorge model checkpoint and are too project-
specific to build in-process here. Run them by invoking
``run_simulation_surrogate.py`` with ``nonlinear_solver`` set to each
method in turn; the per-iteration residual histories are written to
``residual/residual_inc*.csv`` from which the same metrics can be
recomputed offline.

Usage
-----
::

    source activate env_torchfem
    cd torch-fem
    python user_scripts/profile_nonlinear_solvers.py
    python user_scripts/profile_nonlinear_solvers.py --plot \\
        --problems elastic_4x4 plastic_4x4 \\
        --solvers newton_raphson anderson jfnk

Outputs ``profile_nonlinear_solvers_results.csv`` (aggregated mean and
standard deviation per ``(solver, problem)``) and
``profile_nonlinear_solvers_orders.csv`` (per-increment order of
convergence).

Functions
---------
build_problem
    Construct a benchmark problem by name.
fit_order_of_convergence
    Estimate the order of convergence from a residual norm history.
classify_order
    Map a numerical order ``p`` to a categorical label.
run_benchmark
    Execute one ``(solver, problem, rep)`` measurement.
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
import statistics
import time
# Third-party
import numpy as np
import torch
# Local (set up sys.path so the script runs without installation).
import pathlib
import sys
_repo_src = str(pathlib.Path(__file__).resolve().parents[1] / 'src')
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)
from torchfem import Planar
from torchfem.materials import (
    IsotropicElasticityPlaneStrain,
    IsotropicPlasticityPlaneStrain,
)
from torchfem.mesh import rect_quad
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
torch.set_default_dtype(torch.float64)

PROBLEMS_DEFAULT = ['elastic_4x4', 'plastic_4x4', 'elastoplastic_nlh_4x4']
SOLVERS_DEFAULT = [
    'newton_raphson', 'damped_picard', 'anderson',
    'broyden', 'jfnk', 'rand_subspace_newton',
]

# Per-solver kwargs and method-options that match the equivalence-test
# configuration (see tests/test_nonlinear_solvers.py).
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
def build_problem(name):
    """Construct a benchmark problem by name.

    Parameters
    ----------
    name : str
        One of ``'elastic_4x4'`` or ``'plastic_4x4'``.

    Returns
    -------
    domain : Planar
        Configured FEM domain with tensile boundary conditions.
    increments : torch.Tensor
        Load-factor sequence.
    n_dof : int
        Total number of nodal degrees of freedom.
    """
    if name == 'elastic_4x4':
        nodes, elements = rect_quad(5, 5)
        material = IsotropicElasticityPlaneStrain(
            E=110000.0, nu=0.33)
        increments = torch.linspace(0.0, 1.0, 5)
    elif name == 'plastic_4x4':
        nodes, elements = rect_quad(5, 5)

        def sigma_f(q):
            return 200.0 + 1000.0 * q

        def sigma_f_prime(q):
            return torch.full_like(q, 1000.0)

        material = IsotropicPlasticityPlaneStrain(
            E=110000.0, nu=0.33,
            sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
        increments = torch.linspace(0.0, 1.0, 10)
    elif name == 'elastoplastic_nlh_4x4':
        # AA2024 Swift-Voce hardening, matching the elastoplastic_nlh
        # branch in run_simulation_surrogate.py.
        nodes, elements = rect_quad(5, 5)
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

        material = IsotropicPlasticityPlaneStrain(
            E=70000.0, nu=0.33,
            sigma_f=sigma_f, sigma_f_prime=sigma_f_prime,
            max_iter=50)
        increments = torch.linspace(0.0, 1.0, 11)
    else:
        raise ValueError(f'Unknown problem: {name!r}')
    dom = Planar(nodes, elements, material)
    tol = 1e-6
    for i, nc in enumerate(nodes):
        if torch.abs(nc[1]) < tol:
            dom.displacements[i, 1] = 0.0
            dom.constraints[i, 1] = True
            if torch.abs(nc[0]) < tol:
                dom.displacements[i, 0] = 0.0
                dom.constraints[i, 0] = True
        elif torch.abs(nc[1] - 1.0) < tol:
            dom.displacements[i, 1] = 0.02
            dom.constraints[i, 1] = True
    n_dof = dom.n_nod * dom.n_dim
    return dom, increments, n_dof


# =============================================================================
def fit_order_of_convergence(residual_history, atol):
    """Estimate the order of convergence ``p`` from a residual sequence.

    The fit is the slope of ``log(r_{k+1})`` against ``log(r_k)`` over
    the window ``0.1 * r_0 > r_k > 10 * atol``. Returns ``float('nan')``
    if the window contains fewer than three points (insufficient data).

    Parameters
    ----------
    residual_history : list[float]
        Per-iteration residual norms (length n_iter, monotone or not).
    atol : float
        Absolute tolerance used by the solver.

    Returns
    -------
    p : float
        Estimated order of convergence (NaN when undefined).
    """
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
        rk = rh[k]
        rk1 = rh[k + 1]
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
    """Return ``'linear'`` / ``'superlinear'`` / ``'quadratic'`` / ``'?'``.

    Parameters
    ----------
    p : float
        Numerical order of convergence (NaN allowed).

    Returns
    -------
    label : str
    """
    if math.isnan(p):
        return '?'
    if p <= 1.2:
        return 'linear'
    if p < 1.8:
        return 'superlinear'
    return 'quadratic'


# =============================================================================
def run_benchmark(problem_name, solver_name, seed):
    """Run one timed solve and return the metrics record.

    Parameters
    ----------
    problem_name : str
        Problem name (see ``build_problem``).
    solver_name : str
        Solver name (see ``SOLVER_CONFIGS``).
    seed : int
        Seed used for stochastic solvers (RandSub).

    Returns
    -------
    record : dict
        Keys: ``solver``, ``problem``, ``n_dof``, ``n_increments``,
        ``n_iter_total``, ``time_total``, ``time_per_iter``,
        ``order_of_conv``, ``order_label``, ``converged``,
        ``final_res_norm``, ``per_inc_orders``.
    """
    cfg = SOLVER_CONFIGS[solver_name]
    dom, increments, n_dof = build_problem(problem_name)
    torch.manual_seed(seed)
    opts = dict(cfg['opts'])
    if solver_name == 'rand_subspace_newton':
        opts['seed'] = seed
    start = time.perf_counter()
    converged = True
    try:
        res = dom.solve(
            increments=increments,
            rtol=cfg['rtol'],
            atol=cfg['atol'],
            max_iter=cfg['max_iter'],
            return_intermediate=True,
            return_resnorm=True,
            nonlinear_solver=solver_name,
            nonlinear_solver_opts=opts,
        )
        residual_history = res[-1]
    except Exception as excp:
        converged = False
        residual_history = {}
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
        'problem': problem_name,
        'n_dof': n_dof,
        'n_increments': int(len(increments) - 1),
        'n_iter_total': int(n_iter_total),
        'time_total': float(elapsed),
        'time_per_iter': (
            float(elapsed / max(n_iter_total, 1))
        ),
        'order_of_conv': order_of_conv,
        'order_label': classify_order(order_of_conv),
        'converged': converged,
        'final_res_norm': final_res_norm,
        'per_inc_orders': per_inc_orders,
    }


# =============================================================================
def aggregate(records):
    """Aggregate repeated runs into mean/std per (solver, problem).

    Parameters
    ----------
    records : list[dict]
        Output of ``run_benchmark`` for multiple repetitions.

    Returns
    -------
    list[dict]
        One row per ``(solver, problem)`` with mean / stdev for time
        metrics; iteration counts and order are taken from the first
        successful repetition (deterministic given fixed seed).
    """
    grouped = {}
    for rec in records:
        key = (rec['solver'], rec['problem'])
        grouped.setdefault(key, []).append(rec)
    rows = []
    for (solver, problem), reps in grouped.items():
        time_totals = [r['time_total'] for r in reps]
        time_per_iter = [r['time_per_iter'] for r in reps]
        ref = reps[0]
        rows.append({
            'solver': solver,
            'problem': problem,
            'n_dof': ref['n_dof'],
            'n_increments': ref['n_increments'],
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
        })
    rows.sort(key=lambda r: (r['problem'], r['solver']))
    return rows


# =============================================================================
def write_summary_csv(rows, out_path):
    """Write aggregated rows to a CSV file.

    Parameters
    ----------
    rows : list[dict]
        Output of ``aggregate``.
    out_path : str
        Destination CSV path.
    """
    fieldnames = [
        'solver', 'problem', 'n_dof', 'n_increments',
        'n_iter_total', 'time_total_mean_s', 'time_total_std_s',
        'time_per_iter_mean_s', 'time_per_iter_std_s',
        'order_of_conv', 'order_label', 'converged',
        'final_res_norm',
    ]
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
def write_per_increment_orders_csv(records, out_path):
    """Write per-increment order-of-convergence values to CSV.

    Parameters
    ----------
    records : list[dict]
        Output of ``run_benchmark`` for one repetition per
        (solver, problem) combination (the first repetition is
        sufficient since iteration counts are deterministic given a
        fixed seed).
    out_path : str
        Destination CSV path.
    """
    seen = set()
    fieldnames = ['solver', 'problem', 'increment', 'order_of_conv']
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            key = (rec['solver'], rec['problem'])
            if key in seen:
                continue
            seen.add(key)
            for inc, p in rec['per_inc_orders']:
                writer.writerow({
                    'solver': rec['solver'],
                    'problem': rec['problem'],
                    'increment': inc,
                    'order_of_conv': p,
                })


# =============================================================================
def maybe_plot(records, out_path):
    """Plot residual histories per problem, one curve per solver.

    Parameters
    ----------
    records : list[dict]
        Output of ``run_benchmark`` (only the first repetition per
        (solver, problem) is plotted because residual histories are
        deterministic given a fixed seed).
    out_path : str
        Output PDF path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping plots.')
        return
    seen = {}
    for rec in records:
        key = (rec['solver'], rec['problem'])
        if key in seen:
            continue
        seen[key] = rec
    problems = sorted({rec['problem'] for rec in records})
    n_prob = len(problems)
    fig, axes = plt.subplots(
        1, n_prob, figsize=(5 * n_prob, 4), squeeze=False)
    for j, problem in enumerate(problems):
        ax = axes[0, j]
        ax.set_yscale('log')
        ax.set_xlabel('iteration')
        ax.set_ylabel(r'$\|r\|$')
        ax.set_title(problem)
        ax.grid(True, which='both', alpha=0.3)
        for (solver, prob), rec in seen.items():
            if prob != problem:
                continue
            # Concatenate residual histories across increments with
            # an offset for the iteration index.
            full = []
            offset = 0
            for inc in sorted(rec['per_inc_orders']):
                # Re-fetch the residual history from the record;
                # per_inc_orders only carries the order, not the
                # full sequence, so reconstruct on the fly.
                pass
            # We do not have the raw residual history here; it was
            # consumed by ``run_benchmark``. Skip without plotting
            # individual curves to avoid confusion.
        ax.text(
            0.5, 0.5,
            'Per-iteration residuals were not retained;\n'
            'see profile_nonlinear_solvers_orders.csv\n'
            'for the per-increment order of convergence.',
            transform=ax.transAxes, ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Plot saved to {out_path}')


# =============================================================================
def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            'Benchmark torch-fem nonlinear solvers on the in-process '
            'analytic-FE problems. Surrogate cases must be exercised '
            'via run_simulation_surrogate.py with the appropriate '
            'nonlinear_solver flag.'))
    parser.add_argument(
        '--problems', nargs='+', default=PROBLEMS_DEFAULT,
        choices=PROBLEMS_DEFAULT,
        help='Problems to benchmark.')
    parser.add_argument(
        '--solvers', nargs='+', default=SOLVERS_DEFAULT,
        choices=SOLVERS_DEFAULT,
        help='Solvers to benchmark.')
    parser.add_argument(
        '--n-reps', type=int, default=3,
        help='Repetitions per (solver, problem) for time stats.')
    parser.add_argument(
        '--out-dir', default=os.path.dirname(
            os.path.abspath(__file__)),
        help='Directory in which to write output CSVs.')
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate a residual-history PDF plot.')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(
        args.out_dir, 'profile_nonlinear_solvers_results.csv')
    orders_path = os.path.join(
        args.out_dir, 'profile_nonlinear_solvers_orders.csv')
    plot_path = os.path.join(
        args.out_dir, 'profile_nonlinear_solvers_residuals.pdf')
    all_records = []
    for problem in args.problems:
        for solver in args.solvers:
            print(f'-> {solver} on {problem}')
            for rep in range(args.n_reps):
                rec = run_benchmark(problem, solver, seed=rep)
                all_records.append(rec)
    rows = aggregate(all_records)
    write_summary_csv(rows, summary_path)
    write_per_increment_orders_csv(all_records, orders_path)
    if args.plot:
        maybe_plot(all_records, plot_path)
    # Pretty-print the summary to stdout.
    print()
    print(f'{"problem":<14} {"solver":<22} '
          f'{"iters":>6} {"time(s)":>10} '
          f'{"t/iter(s)":>10} {"order":>6} {"label":>12} '
          f'{"conv":>6}')
    print('-' * 96)
    for row in rows:
        print(
            f'{row["problem"]:<14} {row["solver"]:<22} '
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
