"""Full tensor product element stiffness.

Constructs single tensor product quad elements of order
p=1 (quad1) through p=8 (quad8) with all (p+1)^2 nodes
(boundary and interior) and computes the full stiffness
matrix over the complete domain.

Both full and reduced integration variants are computed.
Results saved as .npy files.

Functions
---------
gauss_legendre_01
    Gauss-Legendre points and weights on [0, 1].
plane_strain_tangent
    Fourth-order plane strain elasticity tensor.
"""
#
#                                                          Modules
# =============================================================================
# Standard
import os
import pathlib

# Third-party
import torch
import numpy as np
import matplotlib.pyplot as plt

#
#                                                 Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira'
__status__ = 'Development'


# =============================================================================
def gauss_legendre_01(n):
    """Gauss-Legendre quadrature on [0, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points per direction.

    Returns
    -------
    pts : numpy.ndarray(1d)
        Quadrature point coordinates on [0, 1].
    wts : numpy.ndarray(1d)
        Quadrature weights (sum to 1).
    """

    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * (pts + 1.0)
    wts = 0.5 * wts
    return pts, wts
# =============================================================================


# =============================================================================
def gll_nodes_01(p):
    """Gauss-Lobatto-Legendre nodes on [0, 1].

    GLL nodes on [-1, 1] are the endpoints {-1, 1}
    plus the roots of P'_p(x), where P_p is the
    Legendre polynomial of degree p.

    Parameters
    ----------
    p : int
        Polynomial order (number of nodes = p + 1).

    Returns
    -------
    nodes : numpy.ndarray(1d)
        GLL node coordinates on [0, 1], sorted.
    """

    if p == 1:
        return np.array([0.0, 1.0])
    # Legendre polynomial P_p in coefficient basis
    coeffs = np.zeros(p + 1)
    coeffs[p] = 1.0
    # Derivative P'_p has p-1 interior roots
    deriv = np.polynomial.legendre.legder(coeffs)
    roots = np.polynomial.legendre.legroots(deriv)
    # Add endpoints and map [-1,1] -> [0,1]
    nodes = np.sort(
        np.concatenate([[-1.0], roots, [1.0]]))
    return 0.5 * (nodes + 1.0)
# =============================================================================


# =============================================================================
def plane_strain_tangent(E, nu):
    """Fourth-order plane strain elasticity tensor.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    C : torch.Tensor
        Elasticity tensor of shape (2, 2, 2, 2).
    """

    lbd = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    C = torch.zeros(
        2, 2, 2, 2, dtype=torch.float64)
    C[0, 0, 0, 0] = 2.0 * G + lbd
    C[0, 0, 1, 1] = lbd
    C[1, 1, 0, 0] = lbd
    C[1, 1, 1, 1] = 2.0 * G + lbd
    C[0, 1, 0, 1] = G
    C[0, 1, 1, 0] = G
    C[1, 0, 0, 1] = G
    C[1, 0, 1, 0] = G
    return C
# =============================================================================


# =============================================================================
def _lagrange_1d(xi_nodes, i, xi_eval):
    """Evaluate i-th 1D Lagrange polynomial.

    Parameters
    ----------
    xi_nodes : torch.Tensor
        Node positions of shape (p+1,).
    i : int
        Basis function index (0 to p).
    xi_eval : torch.Tensor
        Evaluation points.

    Returns
    -------
    result : torch.Tensor
        Polynomial values at xi_eval.
    """

    p = len(xi_nodes) - 1
    numerator = torch.ones_like(xi_eval)
    denominator = 1.0
    for j in range(p + 1):
        if i == j:
            continue
        numerator *= (xi_eval - xi_nodes[j])
        denominator *= (xi_nodes[i] - xi_nodes[j])
    return numerator / denominator
# =============================================================================


# =============================================================================
def _lagrange_1d_deriv(xi_nodes, i, xi_eval):
    """Derivative of i-th 1D Lagrange polynomial.

    Uses the analytical derivative of the product
    formula: differentiate each factor in turn, sum.

    Parameters
    ----------
    xi_nodes : torch.Tensor
        Node positions of shape (p+1,).
    i : int
        Basis function index (0 to p).
    xi_eval : torch.Tensor
        Evaluation points.

    Returns
    -------
    result : torch.Tensor
        Derivative values at xi_eval.
    """

    p = len(xi_nodes) - 1
    result = torch.zeros_like(xi_eval)
    for k in range(p + 1):
        if k == i:
            continue
        term = torch.ones_like(xi_eval)
        for m in range(p + 1):
            if m == i or m == k:
                continue
            term *= (xi_eval - xi_nodes[m])
        denom = 1.0
        for m in range(p + 1):
            if m == i:
                continue
            denom *= (xi_nodes[i] - xi_nodes[m])
        result += term / denom
    return result
# =============================================================================


# =============================================================================
def _compute_element_stiffness(
        p, node_indices, C, Lx, Ly, n_gauss,
        xi_nodes_1d=None):
    """Element stiffness for given tensor product nodes.

    Integrates K = int B^T C B det(J) dA over [0,1]^2
    for a rectangular element of size Lx x Ly.
    The Jacobian is constant: J = diag(Lx, Ly).

    Parameters
    ----------
    p : int
        Polynomial order in each direction.
    node_indices : list[tuple[int, int]]
        Active (i, j) indices in the tensor product grid.
    C : torch.Tensor
        Fourth-order elasticity tensor (2, 2, 2, 2).
    Lx : float
        Element size in x direction.
    Ly : float
        Element size in y direction.
    n_gauss : int
        Gauss points per direction.
    xi_nodes_1d : {torch.Tensor, None}, default=None
        1D node positions of shape (p+1,). If None,
        equispaced nodes via torch.linspace(0, 1, p+1).

    Returns
    -------
    K : torch.Tensor
        Stiffness matrix of shape
        (2*n_nodes, 2*n_nodes).
    """

    if xi_nodes_1d is None:
        xi_nodes_1d = torch.linspace(
            0, 1., p + 1, dtype=torch.float64)
    n_nodes = len(node_indices)
    n_dof = 2 * n_nodes
    K = torch.zeros(
        n_dof, n_dof, dtype=torch.float64)
    det_J = Lx * Ly
    pts, wts = gauss_legendre_01(n_gauss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i_gp in range(n_gauss):
        for j_gp in range(n_gauss):
            w = wts[i_gp] * wts[j_gp]
            xi_t = torch.tensor(
                [pts[i_gp]],
                dtype=torch.float64)
            eta_t = torch.tensor(
                [pts[j_gp]],
                dtype=torch.float64)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Pre-compute 1D values and derivatives
            l_xi = torch.zeros(
                p + 1, dtype=torch.float64)
            l_eta = torch.zeros(
                p + 1, dtype=torch.float64)
            dl_xi = torch.zeros(
                p + 1, dtype=torch.float64)
            dl_eta = torch.zeros(
                p + 1, dtype=torch.float64)
            for k in range(p + 1):
                l_xi[k] = _lagrange_1d(
                    xi_nodes_1d, k, xi_t)
                l_eta[k] = _lagrange_1d(
                    xi_nodes_1d, k, eta_t)
                dl_xi[k] = _lagrange_1d_deriv(
                    xi_nodes_1d, k, xi_t)
                dl_eta[k] = _lagrange_1d_deriv(
                    xi_nodes_1d, k, eta_t)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # B matrix: (n_dim, n_nodes)
            B = torch.zeros(
                2, n_nodes, dtype=torch.float64)
            for idx, (i, j) in enumerate(
                    node_indices):
                B[0, idx] = (
                    dl_xi[i] * l_eta[j] / Lx)
                B[1, idx] = (
                    l_xi[i] * dl_eta[j] / Ly)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # K += w * det_J * B^T C B
            BCB = torch.einsum(
                'ijpq,qk,il->ljkp', C, B, B)
            BCB = BCB.reshape(n_dof, n_dof)
            K += w * det_J * BCB
    return K
# =============================================================================


# =============================================================================
def _print_diagnostics(label, K):
    """Print stiffness matrix diagnostics.

    Parameters
    ----------
    label : str
        Description label.
    K : torch.Tensor
        Stiffness matrix.
    """

    eigs = torch.linalg.eigvalsh(K)
    sym_err = (
        torch.norm(K - K.T) / torch.norm(K))
    n_zero = (eigs.abs() < 1e-8).sum().item()
    n_neg = (eigs < -1e-8).sum().item()
    print(f'  {label}:')
    print(f'    shape      = {tuple(K.shape)}')
    print(f'    sym_err    = {sym_err:.2e}')
    print(f'    eig_min    = {eigs[0]:.6e}')
    print(f'    eig_max    = {eigs[-1]:.6e}')
    print(f'    n_zero_eig = {n_zero}')
    print(f'    n_neg_eig  = {n_neg}')
    return eigs
# =============================================================================


# =============================================================================
def plot_eigenvalue_spectra(results_dir, figures_dir,
                            p_max=8, n_gauss_max=9):
    """Eigenvalue spectra per element order.

    One subplot per polynomial order p. Each subplot
    overlays spectra for all quadrature orders, with
    exact integration highlighted.

    Parameters
    ----------
    results_dir : str
        Directory with stiffness .npy files.
    figures_dir : str
        Output directory for figures.
    p_max : int, default=8
        Maximum polynomial order.
    n_gauss_max : int, default=9
        Maximum quadrature order.
    """

    n_cols = 4
    n_rows = int(np.ceil(p_max / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cmap = plt.cm.viridis
    for p in range(1, p_max + 1):
        idx = p - 1
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        n_gauss_exact = p + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for ng in range(1, n_gauss_max + 1):
            fpath = os.path.join(
                results_dir,
                f'stiffness_quad{p}_ng{ng}.npy')
            if not os.path.isfile(fpath):
                continue
            K = np.load(fpath).astype(np.float64)
            eigs = np.sort(
                np.linalg.eigvalsh(K))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if ng == n_gauss_exact:
                ax.plot(
                    range(len(eigs)), eigs,
                    'r-', linewidth=1.8,
                    zorder=10,
                    label=f'ng={ng} (exact)')
            elif ng < n_gauss_exact:
                c = cmap(ng / n_gauss_max)
                ax.plot(
                    range(len(eigs)), eigs,
                    '-', color=c, alpha=0.6,
                    linewidth=0.8,
                    label=f'ng={ng} (under)')
            else:
                ax.plot(
                    range(len(eigs)), eigs,
                    '--', color='grey',
                    alpha=0.4,
                    linewidth=0.6,
                    label=f'ng={ng} (over)')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GLL exact integration overlay
        ng_ex = p + 1
        fpath_gll = os.path.join(
            results_dir,
            f'stiffness_quad{p}_gll_ng{ng_ex}.npy')
        if os.path.isfile(fpath_gll):
            K_gll = np.load(
                fpath_gll).astype(np.float64)
            eigs_gll = np.sort(
                np.linalg.eigvalsh(K_gll))
            ax.plot(
                range(len(eigs_gll)), eigs_gll,
                'b--', linewidth=1.8,
                zorder=9,
                label=f'GLL exact (ng={ng_ex})')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_title(f'quad{p}', fontsize=10)
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Eigenvalue', fontsize=8)
        ax.legend(fontsize=5, ncol=2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx in range(p_max, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'eigenvalue_spectra.png'),
        dpi=200)
    plt.close(fig)
# =============================================================================


# =============================================================================
def plot_eigenvalue_spectra_zoom(
        results_dir, figures_dir,
        p_max=8, n_gauss_max=9, n_eig=8):
    """Zoom on the first n_eig eigenvalues.

    Parameters
    ----------
    results_dir : str
        Directory with stiffness .npy files.
    figures_dir : str
        Output directory for figures.
    p_max : int, default=8
        Maximum polynomial order.
    n_gauss_max : int, default=9
        Maximum quadrature order.
    n_eig : int, default=8
        Number of smallest eigenvalues to show.
    """

    n_cols = 4
    n_rows = int(np.ceil(p_max / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cmap = plt.cm.viridis
    for p in range(1, p_max + 1):
        idx = p - 1
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        n_gauss_exact = p + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for ng in range(1, n_gauss_max + 1):
            fpath = os.path.join(
                results_dir,
                f'stiffness_quad{p}_ng{ng}.npy')
            if not os.path.isfile(fpath):
                continue
            K = np.load(fpath).astype(np.float64)
            eigs = np.sort(
                np.linalg.eigvalsh(K))
            ne = min(n_eig, len(eigs))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if ng == n_gauss_exact:
                ax.plot(
                    range(ne), eigs[:ne],
                    'r-o', linewidth=1.8,
                    markersize=4, zorder=10,
                    label=f'ng={ng} (exact)')
            elif ng < n_gauss_exact:
                c = cmap(ng / n_gauss_max)
                ax.plot(
                    range(ne), eigs[:ne],
                    '-o', color=c, alpha=0.6,
                    linewidth=0.8,
                    markersize=3,
                    label=f'ng={ng} (under)')
            else:
                ax.plot(
                    range(ne), eigs[:ne],
                    '--o', color='grey',
                    alpha=0.4,
                    linewidth=0.6,
                    markersize=3,
                    label=f'ng={ng} (over)')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GLL exact integration overlay
        ng_ex = p + 1
        fpath_gll = os.path.join(
            results_dir,
            f'stiffness_quad{p}_gll_ng{ng_ex}.npy')
        if os.path.isfile(fpath_gll):
            K_gll = np.load(
                fpath_gll).astype(np.float64)
            eigs_gll = np.sort(
                np.linalg.eigvalsh(K_gll))
            ne = min(n_eig, len(eigs_gll))
            ax.plot(
                range(ne), eigs_gll[:ne],
                'b--s', linewidth=1.8,
                markersize=4, zorder=9,
                label=f'GLL exact (ng={ng_ex})')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_title(f'quad{p}', fontsize=10)
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Eigenvalue', fontsize=8)
        ax.legend(fontsize=5, ncol=2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for idx in range(p_max, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir,
                     'eigenvalue_spectra_zoom.png'),
        dpi=200)
    plt.close(fig)
# =============================================================================


# =============================================================================
def plot_cross_resolution_invariants(
        results_dir, figures_dir,
        p_max=8, n_gauss_max=9):
    """Cross-resolution scalar invariants.

    Left: trace / n_dof vs element order.
    Right: ||K||_F / n_dof vs element order.
    Each quadrature order is a separate series.

    Parameters
    ----------
    results_dir : str
        Directory with stiffness .npy files.
    figures_dir : str
        Output directory for figures.
    p_max : int, default=8
        Maximum polynomial order.
    n_gauss_max : int, default=9
        Maximum quadrature order.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis
    orders = list(range(1, p_max + 1))
    labels = [f'quad{p}' for p in orders]
    x = np.arange(len(orders))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for ng in range(1, n_gauss_max + 1):
        trace_vals = []
        frob_vals = []
        x_valid = []
        for i, p in enumerate(orders):
            fpath = os.path.join(
                results_dir,
                f'stiffness_quad{p}_ng{ng}.npy')
            if not os.path.isfile(fpath):
                continue
            K = np.load(fpath).astype(np.float64)
            n_dof = K.shape[0]
            frob = np.linalg.norm(K)
            trace_vals.append(
                np.trace(K) / n_dof)
            frob_vals.append(frob / n_dof)
            x_valid.append(i)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        n_exact = ng  # ng is exact for p = ng - 1
        is_any_exact = (1 <= n_exact <= p_max)
        if ng <= p_max and ng == orders[ng - 1]:
            marker = 'o'
        else:
            marker = 'o'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Color: exact for each p is ng=p+1
        # Use a single color per ng
        c = cmap(ng / n_gauss_max)
        lbl = f'ng={ng}'
        axes[0].plot(
            x_valid, trace_vals,
            '-o', color=c, markersize=4,
            linewidth=0.8, alpha=0.7,
            label=lbl)
        axes[1].plot(
            x_valid, frob_vals,
            '-o', color=c, markersize=4,
            linewidth=0.8, alpha=0.7,
            label=lbl)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Mark exact integration points
    trace_exact = []
    frob_exact = []
    x_exact = []
    for i, p in enumerate(orders):
        ng_ex = p + 1
        fpath = os.path.join(
            results_dir,
            f'stiffness_quad{p}_ng{ng_ex}.npy')
        if not os.path.isfile(fpath):
            continue
        K = np.load(fpath).astype(np.float64)
        n_dof = K.shape[0]
        trace_exact.append(
            np.trace(K) / n_dof)
        frob_exact.append(
            np.linalg.norm(K) / n_dof)
        x_exact.append(i)
    axes[0].plot(
        x_exact, trace_exact,
        'rs', markersize=8, zorder=10,
        label='equi exact')
    axes[1].plot(
        x_exact, frob_exact,
        'rs', markersize=8, zorder=10,
        label='equi exact')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GLL exact integration overlay
    trace_gll = []
    frob_gll = []
    x_gll = []
    for i, p in enumerate(orders):
        ng_ex = p + 1
        fpath = os.path.join(
            results_dir,
            f'stiffness_quad{p}'
            f'_gll_ng{ng_ex}.npy')
        if not os.path.isfile(fpath):
            continue
        K = np.load(fpath).astype(np.float64)
        n_dof = K.shape[0]
        trace_gll.append(
            np.trace(K) / n_dof)
        frob_gll.append(
            np.linalg.norm(K) / n_dof)
        x_gll.append(i)
    axes[0].plot(
        x_gll, trace_gll,
        'bD', markersize=8, zorder=10,
        label='GLL exact')
    axes[1].plot(
        x_gll, frob_gll,
        'bD', markersize=8, zorder=10,
        label='GLL exact')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for ax_i in range(2):
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(labels)
        axes[ax_i].set_xlabel('Element order')
        axes[ax_i].legend(
            fontsize=6, ncol=2)
    axes[0].set_ylabel(
        r'$\mathrm{tr}(\mathbf{K}) / n_{dof}$')
    axes[1].set_ylabel(
        r'$\|\mathbf{K}\|_F / n_{dof}$')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            figures_dir,
            'cross_resolution_invariants.png'),
        dpi=200)
    plt.close(fig)
# =============================================================================


if __name__ == '__main__':
    # Material parameters
    E = 110000.0
    nu = 0.33
    C = plane_strain_tangent(E, nu)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Element parameters
    Lx = 0.25
    Ly = 0.25
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Quadrature sweep: 1 to n_gauss_exact where
    # n_gauss_exact = p_max + 1 = 9 (exact for p=8)
    n_gauss_exact = 9
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Results directory
    torch_fem_root = str(
        pathlib.Path(__file__).parents[1])
    results_dir = os.path.join(
        torch_fem_root, 'results',
        'elastic', '2d', 'single_element')
    os.makedirs(results_dir, exist_ok=True)
    # =========================================================
    for p in range(1, 9):
        all_indices = [
            (i, j)
            for i in range(p + 1)
            for j in range(p + 1)]
        n_nodes = len(all_indices)
        n_dof = 2 * n_nodes
        n_boundary = 4 * p
        n_interior = (p - 1) ** 2
        n_gauss_full = p + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GLL nodes for this order
        gll_np = gll_nodes_01(p)
        xi_gll = torch.tensor(
            gll_np, dtype=torch.float64)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f'quad{p} (p={p}): '
              f'{n_nodes} nodes ({n_boundary} bd, '
              f'{n_interior} int), '
              f'{n_dof} DOFs')
        print(f'  Exact integration: '
              f'{n_gauss_full}x{n_gauss_full}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sweep quadrature orders from 1 to n_gauss_exact
        for ng in range(1, n_gauss_exact + 1):
            # Equispaced
            K_eq = _compute_element_stiffness(
                p, all_indices, C, Lx, Ly, ng)
            # GLL
            K_gll = _compute_element_stiffness(
                p, all_indices, C, Lx, Ly, ng,
                xi_nodes_1d=xi_gll)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Label
            if ng == n_gauss_full:
                tag = 'exact'
            elif ng < n_gauss_full:
                tag = 'under'
            else:
                tag = 'over'
            print(f'  --- equispaced ---')
            _print_diagnostics(
                f'n_gauss={ng}x{ng} ({tag})',
                K_eq)
            print(f'  --- GLL ---')
            _print_diagnostics(
                f'n_gauss={ng}x{ng} ({tag})',
                K_gll)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save equispaced
            fname_eq = (f'stiffness_quad{p}'
                        f'_ng{ng}.npy')
            np.save(
                os.path.join(results_dir, fname_eq),
                K_eq.numpy())
            # Save GLL
            fname_gll = (f'stiffness_quad{p}'
                         f'_gll_ng{ng}.npy')
            np.save(
                os.path.join(
                    results_dir, fname_gll),
                K_gll.numpy())
        print()
    # =========================================================
    # Plots
    figures_dir = os.path.join(
        results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_eigenvalue_spectra(
        results_dir, figures_dir)
    print('  eigenvalue_spectra.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_eigenvalue_spectra_zoom(
        results_dir, figures_dir)
    print('  eigenvalue_spectra_zoom.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_cross_resolution_invariants(
        results_dir, figures_dir)
    print('  cross_resolution_invariants.png')
    # =========================================================
    print(f'All results saved to {results_dir}')
    print(f'All figures saved to {figures_dir}')
