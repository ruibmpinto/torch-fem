"""Plot scaling comparison from timing_results.csv.

Reads timing data and plots FE (elastic, hyperelastic)
vs GNN surrogate cost as a function of patch size.

Notes
-----
FE timings are for the full mesh (mesh = patch_size*4).
Surrogate timings are for surrogate_integrate_material
which processes all 16 patches per call.
To compare per-patch cost:
  - FE per-patch = FE total / n_patches
  - Surrogate per-patch = surrogate total / n_patches
"""
#
#                                                          Modules
# =============================================================================
# Standard
import csv
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
#
#                                                Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto']
__status__ = 'Development'
# =============================================================================
#
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'timing_results.csv')
OUTPUT_PDF = os.path.join(SCRIPT_DIR, 'scaling_plot.pdf')

N_PATCHES = 16  # patch_size * 4 mesh => 4x4 = 16 patches

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
    """Read CSV and produce scaling plot."""
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parse CSV
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Organize data
    fe_elastic = {}
    fe_hyper = {}
    surrogate = {}
    for r in rows:
        mesh_n = int(r['mesh_size'])
        mean_t = float(r['mean_time_s'])
        if r['method'] == 'fe':
            p = mesh_n // 2
            per_patch = mean_t / N_PATCHES
            if r['material'] == 'elastic':
                fe_elastic[p] = per_patch
            elif r['material'] == 'hyperelastic':
                fe_hyper[p] = per_patch
        elif r['method'] == 'surrogate':
            p = mesh_n // 2
            per_patch = mean_t / N_PATCHES
            surrogate[p] = per_patch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build arrays sorted by patch size
    ps_el = sorted(fe_elastic.keys())
    ps_he = sorted(fe_hyper.keys())
    ps_su = sorted(surrogate.keys())
    t_el = np.array(
        [fe_elastic[p] for p in ps_el])
    t_he = np.array(
        [fe_hyper[p] for p in ps_he])
    t_su = np.array(
        [surrogate[p] for p in ps_su])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot
    fig, ax = plt.subplots()
    ax.semilogy(
        ps_el, t_el, 'o-',
        color='#1f77b4', label='FE elastic (2D)')
    ax.semilogy(
        ps_he, t_he, 's-',
        color='#d62728',
        label='FE Hencky (2D)')
    ax.semilogy(
        ps_su, t_su, '^-',
        color='#2ca02c', label='GNN surrogate')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Annotations
    ax.set_xlabel('Patch side length $p$')
    ax.set_ylabel('Time per patch [s]')
    ax.set_title(
        'Material integration cost: FE vs surrogate')
    ax.set_xticks(list(range(1, 9)))
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
