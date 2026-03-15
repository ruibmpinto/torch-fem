# Surrogate Stiffness Matrix Analysis Plan

## Setup

- Material: linear elasticity (E=110000, nu=0.33)
- Geometry: square domain meshed with uniform quad4 elements
- 4 material patches per simulation (2x2 patch grid)
- Resolutions: 1x1, 2x2, 3x3, 4x4, 5x5, 6x6, 7x7, 8x8
- Multiple load increments (linear elasticity: tangent should
  be constant across increments)

## Metrics per patch per increment

### Positive definiteness
- Compute eigenvalues of K
- All eigenvalues must be > 0 for a physically valid elastic
  stiffness matrix
- Report fraction of negative eigenvalues (should be 0)

### Symmetry deviation
- `||K - K^T||_F / ||K||_F`
- Measures how far the surrogate Jacobian is from producing
  a valid symmetric tangent
- Exact FE stiffness is symmetric by construction; any
  asymmetry is pure surrogate error

### Condition number
- `lambda_max / lambda_min`
- Indicates numerical health of the linear solve
- Compare across resolutions

### Variation across patches (same increment)
- Since all 4 patches see different boundary conditions,
  spread in eigenvalues quantifies sensitivity to loading
- Report std of eigenvalue spectra across patches

### Variation across increments (same patch)
- For linear elasticity this should be zero
- Any nonzero spread is pure surrogate error
- Quantifies how well the model learned load-independence
- Key metric: `max_n ||K(inc_n) - K(inc_1)||_F / ||K(inc_1)||_F`

## Cross-resolution comparison

Matrix sizes differ across resolutions (different number of
boundary DOFs), so compare scalar invariants:

- `trace(K) / n_dof` — average diagonal stiffness per DOF
- `||K||_F / n_dof` — Frobenius norm per DOF
- Condition number
- Fraction of negative eigenvalues
- Max relative increment-to-increment variation:
  `max_n ||K(inc_n) - K(inc_1)||_F / ||K(inc_1)||_F`

The last metric is the most telling: it directly measures
the surrogate's failure to learn that elasticity has a
constant tangent.

## Notes

- No analytical reference stiffness is available for patches
  larger than 1x1. The surrogate K is the only data point for
  those resolutions.
- Averaging stiffness matrices over increments is not
  meaningful for linear elasticity (the true K is constant;
  averaging just masks surrogate prediction drift).
- Averaging over patches is valid only if patches are
  statistically equivalent. Corner, edge, and interior patches
  see different displacement patterns due to boundary
  conditions.
