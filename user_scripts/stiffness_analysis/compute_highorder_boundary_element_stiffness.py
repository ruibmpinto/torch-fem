"""Boundary-only tensor product element stiffness.

Constructs a single high-order quad element with nodes only
on the boundary (no interior nodes) and computes its
stiffness matrix via Gauss-Legendre quadrature. Also
computes the statically condensed stiffness from the full
tensor product element for comparison.

Classes
-------
BoundaryElement
    Tensor product element with boundary-only nodes.

Functions
---------
gauss_legendre_01
    Gauss-Legendre points and weights on [0, 1].
plane_strain_tangent
    Fourth-order plane strain elasticity tensor.
static_condensation
    Schur complement reduction to boundary DOFs.
"""
#
#                                                          Modules
# =============================================================================
# Standard
import sys
import pathlib

# Third-party
import torch
import numpy as np

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
def _lagrange_1d(p, i, xi_eval):
    """Evaluate i-th 1D Lagrange polynomial of order p.

    Nodal positions are equispaced on [0, 1].

    Parameters
    ----------
    p : int
        Polynomial order.
    i : int
        Basis function index (0 to p).
    xi_eval : torch.Tensor
        Evaluation points.

    Returns
    -------
    result : torch.Tensor
        Polynomial values at xi_eval.
    """
    xi_nodes = torch.linspace(
        0, 1., p + 1,
        dtype=xi_eval.dtype,
        device=xi_eval.device)
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
def _lagrange_1d_deriv(p, i, xi_eval):
    """Derivative of i-th 1D Lagrange polynomial.

    Uses the analytical derivative of the product
    formula: differentiate each factor in turn, sum.

    Parameters
    ----------
    p : int
        Polynomial order.
    i : int
        Basis function index (0 to p).
    xi_eval : torch.Tensor
        Evaluation points.

    Returns
    -------
    result : torch.Tensor
        Derivative values at xi_eval.
    """
    xi_nodes = torch.linspace(
        0, 1., p + 1,
        dtype=xi_eval.dtype,
        device=xi_eval.device)
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
        p, node_indices, C, Lx, Ly, n_gauss):
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

    Returns
    -------
    K : torch.Tensor
        Stiffness matrix of shape
        (2*n_nodes, 2*n_nodes).
    """
    n_nodes = len(node_indices)
    n_dof = 2 * n_nodes
    K = torch.zeros(
        n_dof, n_dof, dtype=torch.float64)
    det_J = Lx * Ly
    pts, wts = gauss_legendre_01(n_gauss)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                    p, k, xi_t)
                l_eta[k] = _lagrange_1d(
                    p, k, eta_t)
                dl_xi[k] = _lagrange_1d_deriv(
                    p, k, xi_t)
                dl_eta[k] = _lagrange_1d_deriv(
                    p, k, eta_t)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # B matrix: (n_dim, n_nodes)
            # B[0, a] = dN_a/dx = dL_i/dxi * L_j / Lx
            # B[1, a] = dN_a/dy = L_i * dL_j/deta / Ly
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
            # einsum: C_{ijpq} B_{qk} B_{il} -> K_{ljkp}
            BCB = torch.einsum(
                'ijpq,qk,il->ljkp', C, B, B)
            BCB = BCB.reshape(n_dof, n_dof)
            K += w * det_J * BCB
    return K
# =============================================================================


# =============================================================================
def static_condensation(K_full, bd_dofs, int_dofs):
    """Schur complement reduction to boundary DOFs.

    K_bd = K_BB - K_BI K_II^{-1} K_IB

    Parameters
    ----------
    K_full : torch.Tensor
        Full stiffness matrix.
    bd_dofs : torch.Tensor
        Boundary DOF indices.
    int_dofs : torch.Tensor
        Interior DOF indices.

    Returns
    -------
    K_cond : torch.Tensor
        Condensed boundary stiffness.
    """
    if len(int_dofs) == 0:
        return K_full[bd_dofs][:, bd_dofs]
    K_BB = K_full[bd_dofs][:, bd_dofs]
    K_BI = K_full[bd_dofs][:, int_dofs]
    K_IB = K_full[int_dofs][:, bd_dofs]
    K_II = K_full[int_dofs][:, int_dofs]
    X = torch.linalg.solve(K_II, K_IB)
    return K_BB - K_BI @ X
# =============================================================================


# =============================================================================
class BoundaryElement:
    """Tensor product element with boundary-only nodes.

    Constructs Lagrange shape functions of order p on
    [0, 1]^2 and retains only boundary nodes (edges and
    corners). The interior bubble functions are discarded.

    For order p, each edge has p+1 nodes (including the
    two corner nodes shared with adjacent edges). Total
    boundary nodes: 4*p.

    Attributes
    ----------
    _p : int
        Polynomial order.
    _boundary_indices : list[tuple[int, int]]
        Boundary node (i, j) indices in the tensor
        product grid.
    _all_indices : list[tuple[int, int]]
        All node (i, j) indices.
    _interior_indices : list[tuple[int, int]]
        Interior node (i, j) indices.
    _n_boundary_nodes : int
        Number of boundary nodes (4*p).

    Methods
    -------
    evaluate(xi_eta_eval)
        Shape function values at evaluation points.
    evaluate_gradients(xi_eta_eval)
        Parametric derivatives of shape functions.
    boundary_node_coords(Lx, Ly)
        Physical coordinates of boundary nodes.
    compute_stiffness(C, Lx, Ly, n_gauss)
        Boundary-only element stiffness matrix.
    compute_full_stiffness(C, Lx, Ly, n_gauss)
        Full tensor product element stiffness.
    compute_condensed_stiffness(C, Lx, Ly, n_gauss)
        Statically condensed boundary stiffness.
    """

    def __init__(self, p):
        """Constructor.

        Parameters
        ----------
        p : int
            Polynomial order. Each edge gets p+1 nodes
            (including corners).
        """
        if not isinstance(p, int) or p < 1:
            raise RuntimeError(
                'Order p must be a positive integer.')
        self._p = p
        self._boundary_indices = []
        self._all_indices = []
        self._interior_indices = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(p + 1):
            for j in range(p + 1):
                self._all_indices.append((i, j))
                if (i == 0 or i == p
                        or j == 0 or j == p):
                    self._boundary_indices.append(
                        (i, j))
                else:
                    self._interior_indices.append(
                        (i, j))
        self._n_boundary_nodes = len(
            self._boundary_indices)
    # -------------------------------------------------------------------------
    @property
    def p(self):
        """Polynomial order."""
        return self._p
    # -------------------------------------------------------------------------
    @property
    def n_boundary_nodes(self):
        """Number of boundary nodes."""
        return self._n_boundary_nodes
    # -------------------------------------------------------------------------
    @property
    def boundary_indices(self):
        """Boundary node indices (i, j)."""
        return list(self._boundary_indices)
    # -------------------------------------------------------------------------
    def evaluate(self, xi_eta_eval):
        """Evaluate boundary shape functions.

        Parameters
        ----------
        xi_eta_eval : torch.Tensor
            Points of shape (N, 2) in [0, 1]^2.

        Returns
        -------
        N_values : torch.Tensor
            Shape (n_boundary_nodes, N).
        """
        p = self._p
        xi = xi_eta_eval[:, 0]
        eta = xi_eta_eval[:, 1]
        n_pts = xi_eta_eval.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        l_xi = torch.zeros(
            p + 1, n_pts,
            dtype=xi.dtype, device=xi.device)
        l_eta = torch.zeros(
            p + 1, n_pts,
            dtype=xi.dtype, device=xi.device)
        for k in range(p + 1):
            l_xi[k] = _lagrange_1d(p, k, xi)
            l_eta[k] = _lagrange_1d(p, k, eta)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        N_values = torch.zeros(
            self._n_boundary_nodes, n_pts,
            dtype=xi.dtype, device=xi.device)
        for idx, (i, j) in enumerate(
                self._boundary_indices):
            N_values[idx] = l_xi[i] * l_eta[j]
        return N_values
    # -------------------------------------------------------------------------
    def evaluate_gradients(self, xi_eta_eval):
        """Parametric derivatives of shape functions.

        Parameters
        ----------
        xi_eta_eval : torch.Tensor
            Points of shape (N, 2) in [0, 1]^2.

        Returns
        -------
        dN_dxi : torch.Tensor
            Shape (n_boundary_nodes, N).
        dN_deta : torch.Tensor
            Shape (n_boundary_nodes, N).
        """
        p = self._p
        xi = xi_eta_eval[:, 0]
        eta = xi_eta_eval[:, 1]
        n_pts = xi_eta_eval.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        l_xi = torch.zeros(
            p + 1, n_pts,
            dtype=xi.dtype, device=xi.device)
        l_eta = torch.zeros(
            p + 1, n_pts,
            dtype=xi.dtype, device=xi.device)
        dl_xi = torch.zeros(
            p + 1, n_pts,
            dtype=xi.dtype, device=xi.device)
        dl_eta = torch.zeros(
            p + 1, n_pts,
            dtype=xi.dtype, device=xi.device)
        for k in range(p + 1):
            l_xi[k] = _lagrange_1d(p, k, xi)
            l_eta[k] = _lagrange_1d(p, k, eta)
            dl_xi[k] = _lagrange_1d_deriv(
                p, k, xi)
            dl_eta[k] = _lagrange_1d_deriv(
                p, k, eta)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dN_dxi = torch.zeros(
            self._n_boundary_nodes, n_pts,
            dtype=xi.dtype, device=xi.device)
        dN_deta = torch.zeros(
            self._n_boundary_nodes, n_pts,
            dtype=xi.dtype, device=xi.device)
        for idx, (i, j) in enumerate(
                self._boundary_indices):
            dN_dxi[idx] = dl_xi[i] * l_eta[j]
            dN_deta[idx] = l_xi[i] * dl_eta[j]
        return dN_dxi, dN_deta
    # -------------------------------------------------------------------------
    def boundary_node_coords(
            self, Lx=1.0, Ly=1.0):
        """Physical coordinates of boundary nodes.

        Parameters
        ----------
        Lx : float, default=1.0
            Element size in x.
        Ly : float, default=1.0
            Element size in y.

        Returns
        -------
        coords : torch.Tensor
            Shape (n_boundary_nodes, 2).
        """
        p = self._p
        xi_nodes = torch.linspace(
            0, 1., p + 1, dtype=torch.float64)
        coords = torch.zeros(
            self._n_boundary_nodes, 2,
            dtype=torch.float64)
        for idx, (i, j) in enumerate(
                self._boundary_indices):
            coords[idx, 0] = xi_nodes[i] * Lx
            coords[idx, 1] = xi_nodes[j] * Ly
        return coords
    # -------------------------------------------------------------------------
    def compute_stiffness(
            self, C, Lx, Ly, n_gauss):
        """Boundary-only element stiffness matrix.

        Integrates K = int B^T C B det(J) dA using only
        the boundary shape functions. Interior nodes are
        simply absent (not condensed out).

        Parameters
        ----------
        C : torch.Tensor
            Elasticity tensor of shape (2, 2, 2, 2).
        Lx : float
            Element size in x.
        Ly : float
            Element size in y.
        n_gauss : int
            Gauss points per direction.

        Returns
        -------
        K : torch.Tensor
            Shape (2*n_bd, 2*n_bd).
        """
        return _compute_element_stiffness(
            self._p, self._boundary_indices,
            C, Lx, Ly, n_gauss)
    # -------------------------------------------------------------------------
    def compute_full_stiffness(
            self, C, Lx, Ly, n_gauss):
        """Full tensor product element stiffness.

        Uses all (p+1)^2 nodes including interior.

        Parameters
        ----------
        C : torch.Tensor
            Elasticity tensor of shape (2, 2, 2, 2).
        Lx : float
            Element size in x.
        Ly : float
            Element size in y.
        n_gauss : int
            Gauss points per direction.

        Returns
        -------
        K : torch.Tensor
            Shape (2*(p+1)^2, 2*(p+1)^2).
        """
        return _compute_element_stiffness(
            self._p, self._all_indices,
            C, Lx, Ly, n_gauss)
    # -------------------------------------------------------------------------
    def compute_condensed_stiffness(
            self, C, Lx, Ly, n_gauss):
        """Statically condensed boundary stiffness.

        Assembles the full (p+1)^2-node stiffness then
        applies Schur complement to eliminate interior
        DOFs. This yields the exact boundary stiffness
        that accounts for optimal interior response.

        Parameters
        ----------
        C : torch.Tensor
            Elasticity tensor of shape (2, 2, 2, 2).
        Lx : float
            Element size in x.
        Ly : float
            Element size in y.
        n_gauss : int
            Gauss points per direction.

        Returns
        -------
        K_cond : torch.Tensor
            Shape (2*n_bd, 2*n_bd).
        """
        K_full = self.compute_full_stiffness(
            C, Lx, Ly, n_gauss)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        bd_dofs = []
        int_dofs = []
        for idx, (i, j) in enumerate(
                self._all_indices):
            if (i == 0 or i == self._p
                    or j == 0 or j == self._p):
                bd_dofs.extend(
                    [2 * idx, 2 * idx + 1])
            else:
                int_dofs.extend(
                    [2 * idx, 2 * idx + 1])
        bd_dofs = torch.tensor(bd_dofs)
        int_dofs = torch.tensor(int_dofs)
        return static_condensation(
            K_full, bd_dofs, int_dofs)
# =============================================================================


# =============================================================================
def _compare_with_torchfem(
        patch_size, E, nu, Lx, Ly):
    """Compare against torch-fem Quad1 mesh + condensation.

    Parameters
    ----------
    patch_size : int
        Number of Quad1 elements per direction.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    Lx : float
        Patch physical size in x.
    Ly : float
        Patch physical size in y.

    Returns
    -------
    K_cond : torch.Tensor
        Condensed boundary stiffness from Quad1 mesh.
    """
    root = str(pathlib.Path(__file__).parent)
    tfem_src = str(
        pathlib.Path(root) / 'torch-fem' / 'src')
    if tfem_src not in sys.path:
        sys.path.insert(0, tfem_src)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from torchfem import Planar
    from torchfem.mesh import rect_quad
    from torchfem.materials import \
        IsotropicElasticityPlaneStrain
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Nx = patch_size + 1
    Ny = patch_size + 1
    nodes, elements = rect_quad(
        Nx, Ny, Lx=Lx, Ly=Ly)
    material = IsotropicElasticityPlaneStrain(
        E=E, nu=nu)
    domain = Planar(nodes, elements, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble unconstrained stiffness
    n_inc = 2
    u = torch.zeros(
        n_inc, domain.n_nod, domain.n_dim)
    F = torch.zeros(
        n_inc, domain.n_int, domain.n_elem,
        domain.n_stress, domain.n_stress)
    F[:] = torch.eye(domain.n_stress)
    stress = torch.zeros(
        n_inc, domain.n_int, domain.n_elem,
        domain.n_stress, domain.n_stress)
    state = torch.zeros(
        n_inc, domain.n_int, domain.n_elem,
        domain.material.n_state)
    du = torch.zeros(domain.n_dofs)
    de0 = torch.zeros(
        domain.n_elem,
        domain.n_stress, domain.n_stress)
    domain.K = torch.empty(0)
    k_elem, _ = domain.integrate_material(
        u, F, stress, state, 1, du, de0,
        nlgeom=False)
    con = torch.tensor([], dtype=torch.int32)
    K_full = domain.assemble_stiffness(
        k_elem, con).to_dense()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Identify boundary and interior nodes
    bd_nodes = set()
    for i in range(Nx):
        for j in range(Ny):
            if (i == 0 or i == Nx - 1
                    or j == 0 or j == Ny - 1):
                bd_nodes.add(i * Ny + j)
    bd_nodes = sorted(bd_nodes)
    int_nodes = sorted(
        set(range(Nx * Ny)) - set(bd_nodes))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bd_dofs = []
    for n in bd_nodes:
        bd_dofs.extend([2 * n, 2 * n + 1])
    int_dofs = []
    for n in int_nodes:
        int_dofs.extend([2 * n, 2 * n + 1])
    bd_dofs = torch.tensor(bd_dofs)
    int_dofs = torch.tensor(int_dofs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    K_full = K_full.to(torch.float64)
    return static_condensation(
        K_full, bd_dofs, int_dofs)
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
    print(f'{label}:')
    print(f'  shape   = {tuple(K.shape)}')
    print(f'  sym_err = {sym_err:.2e}')
    print(f'  eig_min = {eigs[0]:.6e}')
    print(f'  eig_max = {eigs[-1]:.6e}')
    n_zero = (eigs.abs() < 1e-8).sum().item()
    print(f'  n_zero_eigs = {n_zero}')
    return eigs
# =============================================================================


if __name__ == '__main__':
    # Material parameters
    E = 110000.0
    nu = 0.33
    C = plane_strain_tangent(E, nu)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Element parameters
    p = 4
    Lx = 0.25
    Ly = 0.25
    n_gauss = p + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elem = BoundaryElement(p)
    n_full = (p + 1) ** 2
    n_int = (p - 1) ** 2
    print(f'Order p = {p}')
    print(f'Boundary nodes: {elem.n_boundary_nodes}')
    print(f'Boundary DOFs:  '
          f'{2 * elem.n_boundary_nodes}')
    print(f'Full element nodes: {n_full}')
    print(f'Interior nodes:     {n_int}')
    print(f'Gauss points: {n_gauss}x{n_gauss}')
    print(f'Element size: {Lx} x {Ly}')
    print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Boundary-only stiffness (no interior nodes)
    K_bd = elem.compute_stiffness(
        C, Lx, Ly, n_gauss)
    eigs_bd = _print_diagnostics(
        'Boundary-only stiffness', K_bd)
    print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Full element stiffness
    K_full = elem.compute_full_stiffness(
        C, Lx, Ly, n_gauss)
    _print_diagnostics(
        'Full element stiffness', K_full)
    print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Condensed stiffness (Schur complement)
    K_cond = elem.compute_condensed_stiffness(
        C, Lx, Ly, n_gauss)
    eigs_cond = _print_diagnostics(
        'Condensed stiffness (Schur complement)',
        K_cond)
    print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Compare boundary-only vs condensed
    diff = (torch.norm(K_bd - K_cond)
            / torch.norm(K_cond))
    print(f'Relative diff (bd-only vs condensed): '
          f'{diff:.6e}')
    print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 5. Eigenvalue comparison
    print('Eigenvalue spectrum comparison:')
    print(f'  {"idx":>4s}  '
          f'{"Boundary-only":>18s}  '
          f'{"Condensed":>18s}')
    for i in range(len(eigs_bd)):
        print(f'  {i:4d}  '
              f'{eigs_bd[i]:18.6e}  '
              f'{eigs_cond[i]:18.6e}')
    print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6. Validate against torch-fem (p=1 case)
    print('Validation: p=1 vs torch-fem Quad1 '
          '(1x1 patch)')
    elem_q1 = BoundaryElement(1)
    C_q1 = plane_strain_tangent(E, nu)
    K_q1 = elem_q1.compute_stiffness(
        C_q1, Lx, Ly, n_gauss=2)
    try:
        K_tfem = _compare_with_torchfem(
            1, E, nu, Lx, Ly)
        diff_q1 = (
            torch.norm(
                K_q1 - K_tfem.to(torch.float64))
            / torch.norm(K_tfem))
        print(f'  Relative diff: {diff_q1:.6e}')
    except Exception as e:
        print(f'  torch-fem not available: {e}')
    print()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 7. Compare with torch-fem Quad1 mesh
    # condensation (same boundary node count)
    print(f'Comparison: p={p} condensed vs '
          f'torch-fem {p}x{p} Quad1 mesh condensed')
    try:
        K_tfem_mesh = _compare_with_torchfem(
            p, E, nu, Lx, Ly)
        diff_mesh = (
            torch.norm(
                K_cond - K_tfem_mesh.to(
                    torch.float64))
            / torch.norm(K_tfem_mesh))
        print(f'  Relative diff: {diff_mesh:.6e}')
        print('  (Expected to differ: single p-order '
              'element vs p*p Quad1 mesh)')
    except Exception as e:
        print(f'  torch-fem not available: {e}')
