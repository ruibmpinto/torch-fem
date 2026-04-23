# GLL vs Equispaced Nodes: Stiffness Matrix Analysis

## Setup

Single rectangular element of size $L_x = L_y = 0.25$ on $[0, L_x] \times [0, L_y]$.
Plane strain elasticity with $E = 110\,000$, $\nu = 0.33$.
Tensor product Lagrange basis of order $p$ in each direction,
giving $(p+1)^2$ nodes and $2(p+1)^2$ DOFs.

Two families of nodal distributions on $[0, 1]$:

- **Equispaced**: $\xi_i = i / p$, $i = 0, \dots, p$.
- **GLL** (Gauss-Lobatto-Legendre): $\xi_0 = 0$, $\xi_p = 1$,
  interior nodes at the roots of $P'_p(\xi)$ mapped from
  $[-1, 1]$ to $[0, 1]$.

Both families include the endpoints, so boundary node positions
are identical. Only interior node placement differs.

Gauss-Legendre quadrature with $n_g$ points per direction.
Exact integration of the bilinear form requires $n_g = p + 1$
(integrand is polynomial of degree $2(p-1)$ in each variable;
$n_g$ points integrate exactly up to degree $2 n_g - 1$).

## Results Summary

Max eigenvalue $\lambda_{\max}$ at exact integration ($n_g = p + 1$):

| Order | Equispaced          | GLL                 | Ratio (equi/GLL) |
|-------|---------------------|---------------------|-------------------|
| p=1   | $2.43 \times 10^5$  | $2.43 \times 10^5$  | 1.00              |
| p=2   | $7.94 \times 10^5$  | $7.94 \times 10^5$  | 1.00              |
| p=3   | $1.78 \times 10^6$  | $1.17 \times 10^6$  | 1.52              |
| p=4   | $4.24 \times 10^6$  | $1.37 \times 10^6$  | 3.09              |
| p=5   | $1.19 \times 10^7$  | $1.57 \times 10^6$  | 7.58              |
| p=6   | $4.11 \times 10^7$  | $1.77 \times 10^6$  | 23.2              |
| p=7   | $1.77 \times 10^8$  | $1.99 \times 10^6$  | 89.0              |
| p=8   | $9.39 \times 10^8$  | $2.20 \times 10^6$  | 427               |

Both families produce exactly 3 zero eigenvalues at exact
integration (rigid body modes: 2 translations + 1 rotation).

## Why p=1 and p=2 are Identical

For $p = 1$ the nodes are $\{0, 1\}$ in both cases.

For $p = 2$ the GLL interior node is the root of $P'_2(\xi)$.
Since $P_2(\xi) = \frac{1}{2}(3\xi^2 - 1)$, its derivative is
$P'_2(\xi) = 3\xi$, with root $\xi = 0$, which maps to
$\frac{1}{2}(0 + 1) = 0.5$ on $[0, 1]$ -- the equispaced
midpoint. The two distributions coincide.

## Why Equispaced Max Eigenvalue Grows Exponentially

### Lebesgue constant

For a set of interpolation nodes $\{\xi_i\}_{i=0}^{p}$, the
Lebesgue constant is

$$\Lambda_p = \max_{\xi \in [0,1]} \sum_{i=0}^{p} |L_i(\xi)|$$

where $L_i$ is the $i$-th Lagrange basis polynomial. It
controls the interpolation operator norm:
$\|I_p f\|_\infty \leq \Lambda_p \|f\|_\infty$.

For equispaced nodes, $\Lambda_p$ grows exponentially:

$$\Lambda_p^{\text{equi}} \sim \frac{2^{p+1}}{e \, p \ln p}$$

For GLL nodes, the growth is logarithmic:

$$\Lambda_p^{\text{GLL}} \sim \frac{2}{\pi} \ln(p+1) + C$$

### Connection to the stiffness matrix

The element stiffness matrix is

$$K_{AB} = \int_{\Omega_e} \frac{\partial N_A}{\partial x_i}
\, C_{ijkl} \, \frac{\partial N_B}{\partial x_l}
\; d\Omega$$

where $N_A(\mathbf{x}) = L_i(\xi) L_j(\eta)$ for the
tensor product node $A \leftrightarrow (i, j)$.

For a constant-Jacobian rectangular element the derivatives
separate:

$$\frac{\partial N_A}{\partial x}
= \frac{1}{L_x} L'_i(\xi) L_j(\eta)$$

The max eigenvalue is bounded by

$$\lambda_{\max}(K) \leq \|K\|_2
\leq \|K\|_F
= \left(\sum_{A,B} K_{AB}^2\right)^{1/2}$$

and the individual entries scale with $\|L'_i\|_{L^2}$. The
derivative of a Lagrange basis polynomial satisfies

$$L'_i(\xi) = \sum_{k \neq i}
\frac{\prod_{m \neq i, m \neq k}(\xi - \xi_m)}
{\prod_{m \neq i}(\xi_i - \xi_m)}$$

The denominator contains the product
$\prod_{m \neq i}(\xi_i - \xi_m)$. For equispaced nodes the
factors near the center are $O(1/p)$ each, making the product
exponentially small -- hence the derivatives become
exponentially large. Specifically, for equispaced nodes with
spacing $h = 1/p$:

$$\prod_{m \neq i}(\xi_i - \xi_m)
= \frac{(-1)^{p-i} \, i! \, (p - i)!}{p^p}$$

This is smallest for $i \approx p/2$ (center node), where
Stirling gives

$$i! \, (p - i)! \approx \pi p \left(\frac{p}{2e}\right)^p$$

so the denominator decays as $(1/2)^p$ relative to $p^p$.
The derivative norms therefore grow as $O(2^p)$.

### Eigenvalue scaling

For equispaced nodes the maximum derivative norm scales as

$$\max_i \|L'_i\|_{L^2[0,1]} = O(2^p / \sqrt{p})$$

The stiffness matrix entry magnitudes scale as the square of
the derivative norms (bilinear form), giving

$$\lambda_{\max}^{\text{equi}} = O(4^p / p) \cdot C_{\text{mat}} / L^2$$

where $C_{\text{mat}}$ is the maximum eigenvalue of the
elasticity tensor and $L$ is the element size.

The observed growth from $p = 3$ to $p = 8$ is approximately
$9.39 \times 10^8 / 1.78 \times 10^6 \approx 528$, while
$4^8 / 4^3 / (8/3) = 4^5 / 2.67 \approx 384$. The agreement
is reasonable given the asymptotic nature of the bound.

For GLL nodes the derivatives are bounded:

$$\max_i \|L'_i\|_{L^2[0,1]} = O(p^2)$$

This is because GLL nodes are the zeros of
$(1 - \xi^2) P'_p(\xi)$, which is a Sturm-Liouville
eigenfunction. The resulting Lagrange polynomials satisfy
discrete orthogonality with respect to the GLL quadrature
weights, and their derivatives are controlled by the spectral
differentiation matrix whose norm grows as $O(p^2)$.

This gives

$$\lambda_{\max}^{\text{GLL}} = O(p^4) \cdot C_{\text{mat}} / L^2$$

The observed ratio $2.20 \times 10^6 / 2.43 \times 10^5
\approx 9.0$ from $p = 1$ to $p = 8$, compared to
$8^4 / 1^4 = 4096$, shows the actual growth is much gentler
than the theoretical worst case, because the $O(p^4)$ bound
comes from the spectral differentiation matrix norm, which is a
supremum over all functions, not the specific polynomial
integrand arising in the stiffness form.

## Why GLL Nodes Stay Bounded

Three complementary perspectives:

### 1. Node distribution and potential theory

The asymptotic density of GLL nodes follows the Chebyshev
(arccosine) distribution:

$$\rho(\xi) = \frac{1}{\pi \sqrt{\xi(1-\xi)}}
\quad \text{on } [0, 1]$$

This clusters nodes near the endpoints, where Lagrange
interpolants on a finite interval naturally develop large
oscillations (boundary effect). By placing more nodes near
$\xi = 0$ and $\xi = 1$, GLL nodes suppress the Runge
oscillations that plague equispaced interpolation.

The minimum spacing between consecutive GLL nodes is
$O(1/p^2)$ (at the endpoints), compared to $1/p$ for
equispaced. This endpoint clustering is the key mechanism:
it keeps $\prod_{m \neq i}(\xi_i - \xi_m)$ from becoming
exponentially small for any node $i$.

### 2. Spectral differentiation matrix

The GLL spectral differentiation matrix $D_{ij} = L'_j(\xi_i)$
has the property that its operator norm (largest singular value)
is $O(p^2)$. This is a classical result in spectral methods
(see Canuto et al., Spectral Methods, Theorem 2.4.1). For
equispaced nodes the differentiation matrix norm is $O(2^p)$.

The stiffness matrix is assembled from terms involving $D$
contracted with the elasticity tensor. Since $K$ is a quadratic
form in the derivatives, its spectral radius inherits the
scaling of $\|D\|^2$:

- Equispaced: $\lambda_{\max} \sim \|D\|^2 \sim 4^p$
- GLL: $\lambda_{\max} \sim \|D\|^2 \sim p^4$

### 3. Equivalence to spectral element methods

A tensor product Lagrange element on GLL nodes with GLL
quadrature (not Gauss-Legendre) is precisely the spectral
element method (SEM). In SEM the mass matrix is diagonal
(mass lumping is exact for GLL quadrature applied to the
$N_A N_B$ integrand), and the stiffness matrix inherits the
conditioning of the continuous Laplacian's spectral
discretization.

In the present computation Gauss-Legendre quadrature is used
rather than GLL quadrature, so the mass matrix is not diagonal.
However, the stiffness matrix entries still benefit from the
bounded derivative norms of GLL-based Lagrange polynomials.
The quadrature rule affects accuracy (exact vs under/over
integration) but not the fundamental scaling of the basis
function derivatives.

## Conditioning Implications

The condition number of the stiffness matrix (excluding rigid
body modes) is $\kappa = \lambda_{\max} / \lambda_{\min}^+$
where $\lambda_{\min}^+$ is the smallest positive eigenvalue.

For a single element, $\lambda_{\min}^+$ is determined by the
softest deformation mode and scales as $O(C_{\text{mat}} L^2)$
independent of $p$ (it corresponds to a nearly-uniform shear
of the element). Therefore:

- Equispaced: $\kappa \sim 4^p$
- GLL: $\kappa \sim p^4$

This has direct consequences for iterative solvers. A
conjugate gradient solver on the equispaced stiffness requires
$O(2^p)$ iterations (square root of condition number), while
GLL requires $O(p^2)$.

## Under-Integration Behavior

At exact integration ($n_g = p + 1$), both node types produce
exactly 3 zero eigenvalues. Under-integration introduces
additional spurious zero-energy modes (hourglass modes):

| n_g  | Zero eigenvalues (equi) | Zero eigenvalues (GLL) |
|------|-------------------------|------------------------|
| 1    | $2(p+1)^2 - 3$         | $2(p+1)^2 - 3$        |
| p    | 6                       | 6                      |
| p+1  | 3                       | 3                      |

The count of spurious modes is identical for both node types
at the same quadrature order. This is expected: the number of
constraints imposed by quadrature depends only on the number of
integration points (each point provides 3 independent strain
constraints in 2D), not on the node positions.

However, equispaced nodes at high $p$ with moderate
under-integration can produce **negative eigenvalues** (observed
at p=7 with $n_g = 3, 4$ and p=8 with $n_g = 3, 4, 5, 6, 7$).
GLL nodes do not produce negative eigenvalues at any quadrature
order tested. This is because the equispaced derivative
amplification creates large oscillatory B-matrix entries that,
when integrated with too few quadrature points, can produce
non-physical negative stiffness contributions.

## References

1. Canuto, C., Hussaini, M.Y., Quarteroni, A., Zang, T.A.
   *Spectral Methods: Fundamentals in Single Domains*.
   Springer, 2006.

2. Trefethen, L.N. *Approximation Theory and Approximation
   Practice*. SIAM, 2013. Chapter 15 (Lebesgue constants).

3. Karniadakis, G., Sherwin, S. *Spectral/hp Element Methods
   for Computational Fluid Dynamics*. Oxford, 2005.

4. Pozrikidis, C. *Introduction to Finite and Spectral Element
   Methods Using MATLAB*. Chapman & Hall, 2005.
