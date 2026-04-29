"""Plot Jacobian-method scaling from jacobian_benchmark_results.csv.

Reads per-patch-size timings produced by `jacobian_benchmark.py`
and plots wall-clock cost vs patch side length for the four
Jacobian variants (jacfwd_par, jacrev, jvp_loop, fd_fwd) on a
semilog y-axis.

Functions
---------
main
    Load CSV, build arrays, render plot.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import csv
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto']
__status__ = 'Development'
# =============================================================================
#
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    SCRIPT_DIR, 'jacobian_benchmark_results.csv')
OUTPUT_PDF = os.path.join(
    SCRIPT_DIR, 'jacobian_scaling_plot.pdf')

METHODS = (
    ('fwd', 'forward pass only',
     'v-', '#ff7f0e'),
    ('jacfwd_par', 'torch.func.jacfwd (parallel)',
     '^-', '#2ca02c'),
    ('jacrev', 'torch.func.jacrev',
     'o-', '#1f77b4'),
    ('jvp_loop', 'torch.func.jvp (col-by-col)',
     's-', '#9467bd'),
    ('fd_fwd', 'finite differences (fwd)',
     'd-', '#d62728'),
)

plt.rcParams.update({
    'text.usetex': False,
    'font.size': 12,
    'axes.titlesize': 16,
    'figure.dpi': 360,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (6.5, 6),
    'lines.linewidth': 1.5,
})


# =============================================================================
def main():
    """Read CSV and produce Jacobian scaling plot."""
    with open(CSV_PATH, 'r') as f:
        rows = list(csv.DictReader(f))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort rows by patch side length p = int(patch_size.split('x')[0])
    rows.sort(
        key=lambda r: int(r['patch_size'].split('x')[0]))
    p_sizes = np.array(
        [int(r['patch_size'].split('x')[0]) for r in rows])
    n_dof = np.array(
        [int(r['n_dof_boundary']) for r in rows])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots()
    for key, label, style, color in METHODS:
        mean_s = np.array(
            [float(r[f'{key}_mean_s']) for r in rows])
        std_s = np.array(
            [float(r[f'{key}_std_s']) for r in rows])
        ax.errorbar(
            p_sizes, mean_s, yerr=std_s,
            fmt=style, color=color, label=label,
            capsize=3)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dual-axis labels: show n_dof corresponding to each p
    ax.set_yscale('log')
    ax.set_xlabel('Patch side length $p$ '
                  r'(boundary DOFs $n=8p$)')
    ax.set_ylabel('Jacobian time [s]')
    ax.set_title(
        r'dR/du cost: jacfwd vs jacrev vs jvp vs FD')
    ax.set_xticks(p_sizes.tolist())
    xtick_labels = [
        f'{p}\n(n={d})'
        for p, d in zip(p_sizes.tolist(), n_dof.tolist())]
    ax.set_xticklabels(xtick_labels)
    ax.legend(loc='best')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF)
    fig.savefig(
        OUTPUT_PDF.replace('.pdf', '.png'), dpi=200)
    print(f'Plot saved to {OUTPUT_PDF}')
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    main()
