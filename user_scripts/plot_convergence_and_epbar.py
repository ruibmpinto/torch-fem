"""Plot convergence comparison and mean equivalent plastic strain.

Reads reference and surrogate results from the output directory
produced by run_simulation_surrogate.py.

Usage
-----
    python plot_convergence_and_epbar.py
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle as pkl
import glob

# Third-party
import numpy as np
import matplotlib.pyplot as plt

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'R. Barreira'
__credits__ = ['R. Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Results directory
results_dir = os.path.join(
    os.path.dirname(__file__), os.pardir,
    'results', 'elastoplastic_nlh', '2d', 'quad4',
    'mesh_10x10', 'patch_1x1', 'n_time_inc_50')
results_dir = os.path.normpath(results_dir)

# Output directory for plots
figures_dir = os.path.join(results_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)


def _load_residual_csvs(residual_dir):
    """Load residual history from CSV files in a directory.

    Parameters
    ----------
    residual_dir : str
        Directory containing residual_inc*.csv files.

    Returns
    -------
    res_hist : dict
        Dictionary mapping increment number to list of
        absolute residual norms.
    """
    csv_files = sorted(
        glob.glob(
            os.path.join(
                residual_dir, 'residual_inc*.csv')),
        key=lambda x: int(
            os.path.basename(x)
            .replace('residual_inc', '')
            .replace('.csv', '')))
    res_hist = {}
    for csv_file in csv_files:
        inc = int(
            os.path.basename(csv_file)
            .replace('residual_inc', '')
            .replace('.csv', ''))
        data = np.genfromtxt(
            csv_file, delimiter=',', skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        res_hist[inc] = data[:, 1].tolist()
    return res_hist


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Convergence comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_convergence(results_dir, figures_dir):
    """Plot convergence comparison between reference and surrogate.

    Parameters
    ----------
    results_dir : str
        Path to results directory.
    figures_dir : str
        Path to output figures directory.
    """
    # Load reference residual history from CSV files
    ref_residual_dir = os.path.join(
        results_dir, 'reference_results')
    res_hist_ref = _load_residual_csvs(ref_residual_dir)
    if not res_hist_ref:
        print('Reference residual CSVs not found in '
              f'{ref_residual_dir}. Re-run '
              'run_simulation_surrogate.py.')
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Surrogate residual history from CSV files
    residual_dir = os.path.join(results_dir, 'residual')
    res_hist_srg = _load_residual_csvs(residual_dir)
    if not res_hist_srg:
        print('Surrogate residual CSVs not found in '
              f'{residual_dir}.')
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build increment-based arrays: iterations within each
    # increment are spread as fractional offsets
    def build_increment_arrays(res_hist):
        """Build increment-based x and residual arrays.

        Parameters
        ----------
        res_hist : dict
            Dictionary mapping increment number to list of
            residual norms.

        Returns
        -------
        x_vals : list[float]
            Increment number with fractional offset for
            iterations within each increment.
        residuals : list[float]
            Corresponding residual norms.
        """
        x_vals = []
        residuals = []
        for inc in sorted(res_hist.keys()):
            norms = res_hist[inc]
            n_iter = len(norms)
            for i, norm in enumerate(norms):
                offset = i / max(n_iter, 1)
                x_vals.append(inc + offset)
                residuals.append(norm)
        return x_vals, residuals
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x_ref, res_ref = build_increment_arrays(
        res_hist_ref)
    x_srg, res_srg = build_increment_arrays(
        res_hist_srg)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Detect onset of plasticity from reference epbar
    plasticity_inc = None
    epbar_file = os.path.join(
        results_dir, 'reference_results', 'epbar.pkl')
    if os.path.isfile(epbar_file):
        with open(epbar_file, 'rb') as fh:
            state = pkl.load(fh)
        # epbar is first state variable
        epbar = state[:, :, 0]
        mean_epbar = np.mean(epbar, axis=1)
        # First increment where mean epbar > 0
        nonzero = np.where(mean_epbar > 1e-12)[0]
        if len(nonzero) > 0:
            plasticity_inc = nonzero[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(x_ref, res_ref,
                color='#2c7bb6', linewidth=0.8,
                alpha=0.9, label='Reference FEM')
    ax.semilogy(x_srg, res_srg,
                color='#d7191c', linewidth=0.8,
                alpha=0.9,
                label='Material patch surrogate')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Vertical line at onset of plasticity
    if plasticity_inc is not None:
        ax.axvline(
            x=plasticity_inc, color='#555555',
            linestyle='--', linewidth=1.0,
            alpha=0.7,
            label=f'Onset of plasticity '
                  f'(inc {plasticity_inc})')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax.set_xlabel('Increment number')
    ax.set_ylabel('Residual norm')
    ax.set_title(
        'Convergence: Reference FEM vs Material Patch '
        'Surrogate')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_path = os.path.join(
        figures_dir, 'convergence_comparison.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f'Convergence plot saved to {plot_path}')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Iterations per increment comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_iterations_per_increment(results_dir, figures_dir):
    """Plot number of iterations per increment for both solvers.

    Parameters
    ----------
    results_dir : str
        Path to results directory.
    figures_dir : str
        Path to output figures directory.
    """
    # Load reference residual history from CSVs
    ref_residual_dir = os.path.join(
        results_dir, 'reference_results')
    res_hist_ref = _load_residual_csvs(ref_residual_dir)
    if not res_hist_ref:
        print('No reference residual history available.')
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load surrogate residual history from CSVs
    residual_dir = os.path.join(results_dir, 'residual')
    res_hist_srg = _load_residual_csvs(residual_dir)
    if not res_hist_srg:
        print('No surrogate residual history available.')
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build iterations per increment
    incs_ref = sorted(res_hist_ref.keys())
    iters_ref = [len(res_hist_ref[i]) for i in incs_ref]
    incs_srg = sorted(res_hist_srg.keys())
    iters_srg = [len(res_hist_srg[i]) for i in incs_srg]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.35
    x_ref = np.arange(len(incs_ref))
    x_srg = np.arange(len(incs_srg))
    ax.bar(x_ref - bar_width / 2, iters_ref,
           bar_width, color='#2c7bb6', alpha=0.8,
           label='Reference FEM')
    ax.bar(x_srg + bar_width / 2, iters_srg,
           bar_width, color='#d7191c', alpha=0.8,
           label='Material patch surrogate')
    ax.set_xlabel('Load increment')
    ax.set_ylabel('Newton-Raphson iterations')
    ax.set_title('Iterations per increment')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_path = os.path.join(
        figures_dir, 'iterations_per_increment.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f'Iterations per increment saved to {plot_path}')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Mean equivalent plastic strain evolution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_mean_epbar(results_dir, figures_dir):
    """Plot evolution of mean equivalent plastic strain.

    Parameters
    ----------
    results_dir : str
        Path to results directory.
    figures_dir : str
        Path to output figures directory.
    """
    epbar_file = os.path.join(
        results_dir, 'reference_results', 'epbar.pkl')
    if not os.path.isfile(epbar_file):
        print('epbar.pkl not found. Re-run '
              'run_simulation_surrogate.py.')
        return
    with open(epbar_file, 'rb') as fh:
        state = pkl.load(fh)
    n_inc = state.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Equivalent plastic strain is first state variable
    epbar = state[:, :, 0]
    # Mean over all elements per increment
    mean_epbar = np.mean(epbar, axis=1)
    # Max over all elements per increment
    max_epbar = np.max(epbar, axis=1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pseudo-time (load increments normalized to [0, 1])
    pseudo_time = np.linspace(0.0, 1.0, n_inc)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pseudo_time, mean_epbar,
            color='#2c7bb6', linewidth=1.5,
            label=r'Mean $\bar{\varepsilon}^p$')
    ax.plot(pseudo_time, max_epbar,
            color='#d7191c', linewidth=1.5,
            linestyle='--',
            label=r'Max $\bar{\varepsilon}^p$')
    ax.set_xlabel('Pseudo-time (load fraction)')
    ax.set_ylabel(
        r'Equivalent plastic strain $\bar{\varepsilon}^p$')
    ax.set_title(
        'Evolution of equivalent plastic strain '
        '(reference FEM)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Second x-axis: increment number
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    n_ticks = min(n_inc, 11)
    tick_incs = np.linspace(0, n_inc - 1, n_ticks,
                            dtype=int)
    tick_positions = pseudo_time[tick_incs]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_incs)
    ax2.set_xlabel('Increment number')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig.tight_layout()
    plot_path = os.path.join(
        figures_dir, 'mean_epbar_evolution.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f'Plastic strain plot saved to {plot_path}')


# =============================================================================
if __name__ == '__main__':
    plot_convergence(results_dir, figures_dir)
    plot_iterations_per_increment(results_dir, figures_dir)
    plot_mean_epbar(results_dir, figures_dir)
