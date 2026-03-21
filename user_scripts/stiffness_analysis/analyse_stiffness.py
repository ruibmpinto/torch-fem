"""Surrogate stiffness matrix analysis.

Loads surrogate boundary stiffness matrices for all
n_time_inc_10 configurations, computes quality metrics,
and produces diagnostic plots.

Functions
---------
load_surrogate_stiffness
    Load all surrogate .npy files for one config.
load_reference_stiffness
    Load reference stiffness .npy for one resolution.
compute_matrix_metrics
    Compute scalar quality metrics for one matrix.
"""
#
#                                                          Modules
# =============================================================================
# Standard
import os
import re
import glob
import pathlib

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

#
#                                                 Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira'
__status__ = 'Development'


# =============================================================================
#
# =============================================================================
def load_surrogate_stiffness(stiffness_dir, patch_str):
    """Load all surrogate stiffness matrices for one config.

    Parameters
    ----------
    stiffness_dir : str
        Path to the stiffness/ directory.
    patch_str : str
        Patch resolution string, e.g. '3x3'.

    Returns
    -------
    K_dict : dict
        Keys are (patch_id, increment) tuples.
        Values are numpy 2d arrays (float32).
    """

    pattern = os.path.join(
        stiffness_dir,
        f'stiffness_{patch_str}_id*_inc*.npy')
    files = glob.glob(pattern)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    regex = re.compile(
        rf'stiffness_{patch_str}'
        r'_id(\d+)_inc(\d+)\.npy')
    K_dict = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        m = regex.match(fname)
        if m is None:
            continue
        pid = int(m.group(1))
        inc = int(m.group(2))
        K_dict[(pid, inc)] = np.load(fpath)
    return K_dict


# -------------------------------------------------------------------------
def load_cholesky_stiffness(cholesky_dir, patch_str):
    """Load Cholesky-learned stiffness for one resolution.

    Parameters
    ----------
    cholesky_dir : str
        Path to cholesky_spd/results_cholesky/ directory.
    patch_str : str
        Patch resolution string, e.g. '3x3'.

    Returns
    -------
    K_chol : numpy.ndarray or None
        Learned stiffness matrix, or None if not found.
    """

    fpath = os.path.join(
        cholesky_dir,
        f'patch_{patch_str}',
        f'stiffness_learned_{patch_str}.npy')
    if os.path.isfile(fpath):
        return np.load(fpath)
    return None


# -------------------------------------------------------------------------
def load_reference_stiffness(ref_dir, patch_str):
    """Load reference stiffness matrix for one resolution.

    Parameters
    ----------
    ref_dir : str
        Path to stiffness_reference/ directory.
    patch_str : str
        Patch resolution string, e.g. '3x3'.

    Returns
    -------
    K_ref : numpy.ndarray or None
        Reference stiffness matrix, or None if not found.
    """

    fpath = os.path.join(
        ref_dir, f'stiffness_ref_{patch_str}.npy')
    if os.path.isfile(fpath):
        return np.load(fpath)
    return None


# -------------------------------------------------------------------------
def compute_matrix_metrics(K):
    """Compute scalar quality metrics for one matrix.

    Parameters
    ----------
    K : numpy.ndarray(2d)
        Stiffness matrix.

    Returns
    -------
    metrics : dict
        Keys: sym_err, eig_min, eig_max, n_neg_eig,
        frac_neg_eig, cond, trace_per_dof, frob_per_dof.
    """

    n_dof = K.shape[0]
    K_64 = K.astype(np.float64)
    frob = np.linalg.norm(K_64)
    sym_err = (np.linalg.norm(K_64 - K_64.T) / frob
               if frob > 0 else 0.0)
    eigs = np.linalg.eigvalsh(K_64)
    n_neg = int(np.sum(eigs < 0))
    eig_min = float(eigs[0])
    eig_max = float(eigs[-1])
    cond = (eig_max / eig_min
            if abs(eig_min) > 0 else np.inf)
    return {
        'sym_err': sym_err,
        'eig_min': eig_min,
        'eig_max': eig_max,
        'n_neg_eig': n_neg,
        'frac_neg_eig': n_neg / n_dof,
        'cond': cond,
        'trace_per_dof': np.trace(K_64) / n_dof,
        'frob_per_dof': frob / n_dof,
    }


# -------------------------------------------------------------------------
def plot_symmetry_deviation(all_metrics, resolutions,
                            figures_dir):
    """Boxplot of symmetry deviation vs resolution.

    Parameters
    ----------
    all_metrics : dict
        Keys are patch_str, values are lists of metric
        dicts (one per patch_id/increment pair).
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    """

    data = []
    for res in resolutions:
        vals = [m['sym_err'] for m in all_metrics[res]]
        data.append(vals)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=resolutions)
    ax.set_xlabel('Patch resolution')
    ax.set_ylabel(
        r'$\|\mathbf{K} - \mathbf{K}^T\|_F'
        r' / \|\mathbf{K}\|_F$')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'symmetry_deviation.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_condition_number(all_metrics, ref_metrics,
                          resolutions, figures_dir):
    """Boxplot of condition number vs resolution.

    Parameters
    ----------
    all_metrics : dict
        Keys are patch_str, values are lists of metric
        dicts.
    ref_metrics : dict
        Keys are patch_str, values are metric dicts for
        the reference matrix (or None).
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    """

    data = []
    for res in resolutions:
        vals = [abs(m['cond']) for m in all_metrics[res]]
        data.append(vals)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=resolutions)
    # Overlay reference condition numbers
    for i, res in enumerate(resolutions):
        if res in ref_metrics and ref_metrics[res] is not None:
            ref_cond = abs(ref_metrics[res]['cond'])
            ax.plot(i + 1, ref_cond, 'ro',
                    markersize=10, zorder=20)
    ax.set_xlabel('Patch resolution')
    ax.set_ylabel('Condition number')
    ax.set_yscale('log')
    # Legend for reference marker
    ax.plot([], [], 'ro', markersize=10,
            label='Reference')
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'condition_number.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_increment_variation(all_K, resolutions,
                             figures_dir):
    """Bar chart of max increment-to-increment variation.

    Parameters
    ----------
    all_K : dict
        Keys are patch_str, values are dicts
        {(patch_id, inc): np.ndarray}.
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    """

    means = []
    stds = []
    for res in resolutions:
        K_dict = all_K[res]
        patch_ids = sorted(set(
            pid for pid, _ in K_dict.keys()))
        variations = []
        for pid in patch_ids:
            incs = sorted(
                inc for p, inc in K_dict.keys()
                if p == pid)
            if len(incs) < 2:
                continue
            K1 = K_dict[(pid, incs[0])].astype(
                np.float64)
            norm_K1 = np.linalg.norm(K1)
            if norm_K1 == 0:
                continue
            max_var = 0.0
            for inc in incs[1:]:
                Kn = K_dict[(pid, inc)].astype(
                    np.float64)
                rel = np.linalg.norm(Kn - K1) / norm_K1
                if rel > max_var:
                    max_var = rel
            variations.append(max_var)
        means.append(np.mean(variations)
                     if variations else 0.0)
        stds.append(np.std(variations)
                    if variations else 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(resolutions))
    ax.bar(x, means, yerr=stds, capsize=4,
           color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_xlabel('Patch resolution')
    ax.set_ylabel(
        r'$\max_n \|\mathbf{K}_n - \mathbf{K}_1\|_F'
        r' / \|\mathbf{K}_1\|_F$')
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'increment_variation.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_patch_deviation(all_K, resolutions, figures_dir):
    """Patch-to-patch deviation at each increment.

    Parameters
    ----------
    all_K : dict
        Keys are patch_str, values are dicts
        {(patch_id, inc): np.ndarray}.
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    for res in resolutions:
        K_dict = all_K[res]
        increments = sorted(set(
            inc for _, inc in K_dict.keys()))
        patch_ids = sorted(set(
            pid for pid, _ in K_dict.keys()))
        if len(patch_ids) < 2:
            continue
        mean_devs = []
        for inc in increments:
            matrices = []
            for pid in patch_ids:
                if (pid, inc) in K_dict:
                    matrices.append(
                        K_dict[(pid, inc)].astype(
                            np.float64))
            if len(matrices) < 2:
                mean_devs.append(0.0)
                continue
            # Pairwise relative deviations
            devs = []
            for i in range(len(matrices)):
                norm_i = np.linalg.norm(matrices[i])
                if norm_i == 0:
                    continue
                for j in range(i + 1, len(matrices)):
                    rel = (np.linalg.norm(
                        matrices[i] - matrices[j])
                        / norm_i)
                    devs.append(rel)
            mean_devs.append(
                np.mean(devs) if devs else 0.0)
        ax.plot(increments, mean_devs, 'o-',
                label=res, markersize=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax.set_xlabel('Increment')
    ax.set_ylabel(
        r'Mean pairwise '
        r'$\|\mathbf{K}_i - \mathbf{K}_j\|_F'
        r' / \|\mathbf{K}_i\|_F$')
    ax.legend(title='Resolution', fontsize=8,
              ncol=2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'patch_deviation.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_increment_drift(all_K, resolutions, figures_dir):
    """Per-patch drift relative to first increment.

    Parameters
    ----------
    all_K : dict
        Keys are patch_str, values are dicts
        {(patch_id, inc): np.ndarray}.
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    """

    n_res = len(resolutions)
    n_cols = 4
    n_rows = int(np.ceil(n_res / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharey=True,
        squeeze=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx, res in enumerate(resolutions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        K_dict = all_K[res]
        patch_ids = sorted(set(
            pid for pid, _ in K_dict.keys()))
        increments = sorted(set(
            inc for _, inc in K_dict.keys()))
        for pid in patch_ids:
            if (pid, increments[0]) not in K_dict:
                continue
            K1 = K_dict[(pid, increments[0])].astype(
                np.float64)
            norm_K1 = np.linalg.norm(K1)
            if norm_K1 == 0:
                continue
            drifts = []
            for inc in increments:
                if (pid, inc) not in K_dict:
                    drifts.append(np.nan)
                    continue
                Kn = K_dict[(pid, inc)].astype(
                    np.float64)
                drifts.append(
                    np.linalg.norm(Kn - K1)
                    / norm_K1)
            ax.plot(increments, drifts, '-',
                    alpha=0.5, linewidth=0.8)
        ax.set_title(res, fontsize=10)
        ax.set_xlabel('Increment', fontsize=8)
        if col == 0:
            ax.set_ylabel(
                r'$\|\mathbf{K}_n - \mathbf{K}_1\|_F'
                r' / \|\mathbf{K}_1\|_F$',
                fontsize=8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Hide unused subplots
    for idx in range(n_res, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'increment_drift.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_eigenvalue_spectra(all_K, ref_stiffness,
                            resolutions, figures_dir,
                            cholesky_stiffness=None):
    """Eigenvalue spectra per resolution.

    Parameters
    ----------
    all_K : dict
        Keys are patch_str, values are dicts
        {(patch_id, inc): np.ndarray}.
    ref_stiffness : dict
        Keys are patch_str, values are np.ndarray or None.
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    cholesky_stiffness : dict or None, default=None
        Keys are patch_str, values are np.ndarray or None.
        If provided, overlay Cholesky eigenvalues.
    """

    n_res = len(resolutions)
    n_cols = 4
    n_rows = int(np.ceil(n_res / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx, res in enumerate(resolutions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        K_dict = all_K[res]
        # Overlay all surrogate spectra
        for (pid, inc), K in K_dict.items():
            eigs = np.linalg.eigvalsh(
                K.astype(np.float64))
            ax.plot(range(len(eigs)), np.sort(eigs),
                    '-o', alpha=0.15, color='b',
                    linewidth=0.6)
        # Legend for matpatch marker
        ax.plot([], [], 'bo', markersize=5,
            label='matpatch')
        # Reference spectrum
        K_ref = ref_stiffness.get(res)
        if K_ref is not None:
            eigs_ref = np.linalg.eigvalsh(
                K_ref.astype(np.float64))
            ax.plot(range(len(eigs_ref)),
                    np.sort(eigs_ref),
                    'r--', linewidth=1.5,
                    label='Reference')
        # Cholesky spectrum
        if cholesky_stiffness is not None:
            K_chol = cholesky_stiffness.get(res)
            if K_chol is not None:
                eigs_chol = np.linalg.eigvalsh(
                    K_chol.astype(np.float64))
                ax.plot(
                    range(len(eigs_chol)),
                    np.sort(eigs_chol),
                    'g-.', linewidth=1.5,
                    label='Cholesky')
        ax.legend(fontsize=7)
        ax.set_title(res, fontsize=10)
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Eigenvalue', fontsize=8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx in range(n_res, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'eigenvalue_spectra.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_eigenvalue_spectra_zoom(all_K, ref_stiffness,
                                 resolutions,
                                 figures_dir,
                                 n_eig=8,
                                 cholesky_stiffness=None):
    """Zoom on the first n_eig eigenvalues.

    Parameters
    ----------
    all_K : dict
        Keys are patch_str, values are dicts
        {(patch_id, inc): np.ndarray}.
    ref_stiffness : dict
        Keys are patch_str, values are np.ndarray or None.
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    n_eig : int, default=8
        Number of smallest eigenvalues to show.
    cholesky_stiffness : dict or None, default=None
        Keys are patch_str, values are np.ndarray or None.
        If provided, overlay Cholesky eigenvalues.
    """

    n_res = len(resolutions)
    n_cols = 4
    n_rows = int(np.ceil(n_res / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx, res in enumerate(resolutions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        K_dict = all_K[res]
        # Overlay all surrogate spectra (first n_eig)
        for (pid, inc), K in K_dict.items():
            eigs = np.sort(np.linalg.eigvalsh(
                K.astype(np.float64)))
            ax.plot(
                range(min(n_eig, len(eigs))),
                eigs[:n_eig],
                '-o', alpha=0.15, color='b',
                linewidth=0.6, markersize=3)
        # Legend for matpatch marker
        ax.plot([], [], 'bo', markersize=5,
                label='matpatch')
        # Reference spectrum (first n_eig)
        K_ref = ref_stiffness.get(res)
        if K_ref is not None:
            eigs_ref = np.sort(np.linalg.eigvalsh(
                K_ref.astype(np.float64)))
            ax.plot(
                range(min(n_eig, len(eigs_ref))),
                eigs_ref[:n_eig],
                'r--o', linewidth=1.5,
                markersize=4,
                label='Reference')
        # Cholesky spectrum (first n_eig)
        if cholesky_stiffness is not None:
            K_chol = cholesky_stiffness.get(res)
            if K_chol is not None:
                eigs_chol = np.sort(
                    np.linalg.eigvalsh(
                        K_chol.astype(np.float64)))
                ax.plot(
                    range(min(n_eig, len(eigs_chol))),
                    eigs_chol[:n_eig],
                    'g-.o', linewidth=1.5,
                    markersize=4,
                    label='Cholesky')
        ax.legend(fontsize=7)
        ax.set_title(res, fontsize=10)
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Eigenvalue', fontsize=8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx in range(n_res, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'eigenvalue_spectra_zoom.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_eigenvalue_spectra_individual(
        all_K, ref_stiffness, resolutions,
        figures_dir, n_eig=8,
        cholesky_stiffness=None):
    """Individual patch eigenvalue spectra (no averaging).

    Each patch_id is a distinct colour; increments of the
    same patch share the colour.

    Parameters
    ----------
    all_K : dict
        Keys are patch_str, values are dicts
        {(patch_id, inc): np.ndarray}.
    ref_stiffness : dict
        Keys are patch_str, values are np.ndarray or None.
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    n_eig : int, default=8
        Number of smallest eigenvalues to show.
    cholesky_stiffness : dict or None, default=None
        Keys are patch_str, values are np.ndarray or None.
    """

    n_res = len(resolutions)
    n_cols = 4
    n_rows = int(np.ceil(n_res / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False)
    cmap = plt.cm.tab10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx, res in enumerate(resolutions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        K_dict = all_K[res]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Group by patch_id
        patch_ids = sorted(set(
            pid for pid, _ in K_dict.keys()))
        for ip, pid in enumerate(patch_ids):
            c = cmap(ip % 10)
            incs = sorted(
                inc for p, inc in K_dict.keys()
                if p == pid)
            for ii, inc in enumerate(incs):
                K = K_dict[(pid, inc)]
                eigs = np.sort(
                    np.linalg.eigvalsh(
                        K.astype(np.float64)))
                ne = min(n_eig, len(eigs))
                lbl = (f'patch {pid}'
                       if ii == 0 else None)
                ax.plot(
                    range(ne), eigs[:ne],
                    '-o', color=c, alpha=0.6,
                    linewidth=0.8, markersize=3,
                    label=lbl)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reference
        K_ref = ref_stiffness.get(res)
        if K_ref is not None:
            eigs_ref = np.sort(
                np.linalg.eigvalsh(
                    K_ref.astype(np.float64)))
            ne = min(n_eig, len(eigs_ref))
            ax.plot(
                range(ne), eigs_ref[:ne],
                'r--', linewidth=2.0, zorder=10,
                label='Reference')
        # Cholesky
        if cholesky_stiffness is not None:
            K_chol = cholesky_stiffness.get(res)
            if K_chol is not None:
                eigs_chol = np.sort(
                    np.linalg.eigvalsh(
                        K_chol.astype(np.float64)))
                ne = min(n_eig, len(eigs_chol))
                ax.plot(
                    range(ne), eigs_chol[:ne],
                    'k-.', linewidth=2.0,
                    zorder=10,
                    label='Cholesky')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Zero line
        ax.axhline(0, color='grey', linewidth=0.5,
                   linestyle=':')
        ax.legend(fontsize=5, ncol=2)
        ax.set_title(res, fontsize=10)
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Eigenvalue', fontsize=8)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx in range(n_res, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            figures_dir,
            'eigenvalue_spectra_individual.png'),
        dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------
def plot_stiffness_heatmaps(all_K, ref_stiffness,
                            resolutions, figures_dir):
    """Heatmaps: reference, surrogate, relative error.

    Parameters
    ----------
    all_K : dict
        Keys are patch_str, values are dicts
        {(patch_id, inc): np.ndarray}.
    ref_stiffness : dict
        Keys are patch_str, values are np.ndarray or None.
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    """

    for res in resolutions:
        K_ref = ref_stiffness.get(res)
        if K_ref is None:
            continue
        K_ref_64 = K_ref.astype(np.float64)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Mean surrogate across all patches and increments
        K_dict = all_K[res]
        matrices = [K.astype(np.float64)
                    for K in K_dict.values()]
        K_srg = np.mean(matrices, axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Relative error (element-wise, two-tier)
        # Where |K_ref| is significant: element-wise
        # relative error. Where |K_ref| ~ 0: normalize
        # by max(|K_ref|) to measure surrogate leakage.
        abs_ref = np.abs(K_ref_64)
        K_ref_max = abs_ref.max()
        sig_mask = abs_ref > 1e-3 * K_ref_max
        denom = np.where(sig_mask, abs_ref, K_ref_max)
        rel_err = np.abs(K_srg - K_ref_64) / denom
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 5))
        # (a) Reference
        vmax = max(np.abs(K_ref_64).max(),
                   np.abs(K_srg).max())
        norm = TwoSlopeNorm(
            vmin=-vmax, vcenter=0, vmax=vmax)
        im0 = axes[0].imshow(
            K_ref_64, cmap='RdBu_r', norm=norm,
            aspect='equal')
        axes[0].set_title(
            r'$\mathbf{K}_{ref}$' + f' ({res})')
        fig.colorbar(im0, ax=axes[0], shrink=0.8)
        # (b) Surrogate (mean)
        im1 = axes[1].imshow(
            K_srg, cmap='RdBu_r', norm=norm,
            aspect='equal')
        axes[1].set_title(
            r'$\tilde{\mathbf{K}}$'
            + f' ({res}, mean)')
        fig.colorbar(im1, ax=axes[1], shrink=0.8)
        # (c) Relative error
        im2 = axes[2].imshow(
            rel_err, cmap='hot_r', aspect='equal')
        axes[2].set_title(
            r'$|\tilde{\mathbf{K}}'
            r' - \mathbf{K}_{ref}|'
            r' / \mathrm{denom}$'
            + f' ({res})')
        fig.colorbar(im2, ax=axes[2], shrink=0.8)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                figures_dir,
                f'stiffness_heatmap_{res}.png'),
            dpi=200)
        plt.close(fig)


# -------------------------------------------------------------------------
def plot_cross_resolution_invariants(
        all_metrics, ref_metrics, resolutions,
        figures_dir):
    """Cross-resolution scalar invariants.

    Parameters
    ----------
    all_metrics : dict
        Keys are patch_str, values are lists of metric
        dicts.
    ref_metrics : dict
        Keys are patch_str, values are metric dicts
        (or None).
    resolutions : list[str]
        Ordered resolution labels.
    figures_dir : str
        Output directory for figures.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(resolutions))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # trace / n_dof
    for i, res in enumerate(resolutions):
        vals = [m['trace_per_dof']
                for m in all_metrics[res]]
        axes[0].scatter(
            np.full(len(vals), i), vals,
            alpha=1, s=10, color='b')
    # Legend for matpatch marker
    axes[0].plot([], [], 'bo', markersize=5,
            label='matpatch')
    ref_trace = []
    ref_x = []
    for i, res in enumerate(resolutions):
        rm = ref_metrics.get(res)
        if rm is not None:
            ref_trace.append(rm['trace_per_dof'])
            ref_x.append(i)
    if ref_trace:
        axes[0].scatter(
            ref_x, ref_trace, marker='o',
            s=10, color='red', zorder=5,
            label='reference')
        axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(resolutions)
    axes[0].set_xlabel('Patch resolution')
    axes[0].set_ylabel(
        r'$\mathrm{tr}(\mathbf{K}) / n_{dof}$')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ||K||_F / n_dof
    for i, res in enumerate(resolutions):
        vals = [m['frob_per_dof']
                for m in all_metrics[res]]
        axes[1].scatter(
            np.full(len(vals), i), vals,
            alpha=0.3, s=10, color='b')
    # Legend for matpatch marker
    axes[1].plot([], [], 'bo', markersize=5,
            label='matpatch')
    ref_frob = []
    ref_x = []
    for i, res in enumerate(resolutions):
        rm = ref_metrics.get(res)
        if rm is not None:
            ref_frob.append(rm['frob_per_dof'])
            ref_x.append(i)
    if ref_frob:
        axes[1].scatter(
            ref_x, ref_frob, marker='o',
            s=10, color='red', zorder=5,
            label='reference')
        axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(resolutions)
    axes[1].set_xlabel('Patch resolution')
    axes[1].set_ylabel(
        r'$\|\mathbf{K}\|_F / n_{dof}$')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'cross_resolution_invariants.png'),
        dpi=200)
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    # Flags
    include_cholesky = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Paths
    torch_fem_root = str(
        pathlib.Path(__file__).parents[2])
    project_root = str(
        pathlib.Path(torch_fem_root).parent)
    results_base = os.path.join(
        torch_fem_root, 'results',
        'elastic', '2d', 'quad4')
    ref_dir = os.path.join(
        results_base, 'stiffness_reference')
    cholesky_dir = os.path.join(
        project_root, 'cholesky_spd',
        'results_cholesky')
    figures_dir = os.path.join(
        os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    # =========================================================
    # Configurations (n_time_inc_10 only)
    configs = [
        ('4x4', '1x1'),
        ('8x8', '2x2'),
        ('12x12', '3x3'),
        ('16x16', '4x4'),
        ('20x20', '5x5'),
        ('24x24', '6x6'),
        ('28x28', '7x7'),
        ('32x32', '8x8'),
    ]
    resolutions = [p for _, p in configs]
    # =========================================================
    # Load all data
    print('Loading surrogate stiffness matrices...')
    all_K = {}
    for mesh_str, patch_str in configs:
        stiffness_dir = os.path.join(
            results_base,
            f'mesh_{mesh_str}',
            f'patch_{patch_str}',
            'n_time_inc_10',
            'stiffness')
        K_dict = load_surrogate_stiffness(
            stiffness_dir, patch_str)
        all_K[patch_str] = K_dict
        n_patches = len(set(
            pid for pid, _ in K_dict.keys()))
        n_incs = len(set(
            inc for _, inc in K_dict.keys()))
        print(f'  {patch_str}: {n_patches} patches, '
              f'{n_incs} increments, '
              f'{len(K_dict)} matrices')
    # =========================================================
    # Load reference stiffness
    print('Loading reference stiffness matrices...')
    ref_stiffness = {}
    for res in resolutions:
        ref_stiffness[res] = load_reference_stiffness(
            ref_dir, res)
        status = ('found' if ref_stiffness[res]
                  is not None else 'not found')
        print(f'  {res}: {status}')
    # =========================================================
    # Load Cholesky stiffness (optional)
    cholesky_stiffness = None
    if include_cholesky:
        print('Loading Cholesky stiffness matrices...')
        cholesky_stiffness = {}
        for res in resolutions:
            cholesky_stiffness[res] = \
                load_cholesky_stiffness(
                    cholesky_dir, res)
            status = ('found'
                      if cholesky_stiffness[res]
                      is not None else 'not found')
            print(f'  {res}: {status}')
    # =========================================================
    # Compute metrics
    print('Computing metrics...')
    all_metrics = {}
    ref_metrics = {}
    for res in resolutions:
        metrics_list = []
        for (pid, inc), K in all_K[res].items():
            m = compute_matrix_metrics(K)
            m['patch_id'] = pid
            m['increment'] = inc
            metrics_list.append(m)
        all_metrics[res] = metrics_list
        # Reference metrics
        K_ref = ref_stiffness[res]
        if K_ref is not None:
            ref_metrics[res] = compute_matrix_metrics(
                K_ref)
        else:
            ref_metrics[res] = None
    # =========================================================
    # Generate plots
    print('Generating plots...')
    plot_symmetry_deviation(
        all_metrics, resolutions, figures_dir)
    print('  symmetry_deviation.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_condition_number(
        all_metrics, ref_metrics,
        resolutions, figures_dir)
    print('  condition_number.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_increment_variation(
        all_K, resolutions, figures_dir)
    print('  increment_variation.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_patch_deviation(
        all_K, resolutions, figures_dir)
    print('  patch_deviation.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_increment_drift(
        all_K, resolutions, figures_dir)
    print('  increment_drift.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_eigenvalue_spectra(
        all_K, ref_stiffness, resolutions, figures_dir,
        cholesky_stiffness=cholesky_stiffness)
    print('  eigenvalue_spectra.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_eigenvalue_spectra_zoom(
        all_K, ref_stiffness, resolutions, figures_dir,
        cholesky_stiffness=cholesky_stiffness)
    print('  eigenvalue_spectra_zoom.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_eigenvalue_spectra_individual(
        all_K, ref_stiffness, resolutions, figures_dir,
        cholesky_stiffness=cholesky_stiffness)
    print('  eigenvalue_spectra_individual.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_stiffness_heatmaps(
        all_K, ref_stiffness, resolutions, figures_dir)
    print('  stiffness_heatmap_*.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_cross_resolution_invariants(
        all_metrics, ref_metrics,
        resolutions, figures_dir)
    print('  cross_resolution_invariants.png')
    # =========================================================
    print(f'All figures saved to {figures_dir}')
