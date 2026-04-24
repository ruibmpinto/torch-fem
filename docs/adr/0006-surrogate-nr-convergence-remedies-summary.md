# ADR-0006: Surrogate Newton-Raphson convergence — remedies summary

## Status

Informational. Summarises the implementation and empirical findings
of ADR-0004 (SPD eigenvalue projection) and ADR-0005 (modified Newton
with linear-elastic analytic tangent), both targeting the same
problem on two separate feature branches.

## Problem

The graph-surrogate material-patch integration (ADR-0001) diverges
in Newton-Raphson on patches larger than 1x1 when used inside
`solve_matpatch`.

Observed on `elastoplastic_nlh` with Swift-Voce AA2024 and a
10x10 quad4 mesh:

- 1x1 patch: converges in 3 iterations, residual
  `114.8 -> 3.39 -> 0.00767`.
- 2x2 patch: residual `1.02e+02 -> 9.91e+03` over 13 iterations,
  then singular matrix at iteration 14 (`scipy_spsolve` in
  `torchfem/sparse.py:262`).
- 5x5 patch: residual `76.87 -> 3496.53` over 51 iterations, no
  convergence.

Diagnostics on the per-patch surrogate tangent K (40x40 for 5x5):

- Asymmetry `||K - K.T|| / ||K||` in `0.78-0.86`.
- Fraction of negative eigenvalues ~ `60 %`.
- Condition numbers `4e16 - 2.5e17`.
- Frobenius distance of the symmetric part to the analytic elastic
  tangent: `||K_sym - K_analytic||_F / ||K||_F = 0.96`.
- Force bias `||f_gnn(u = 0)|| = 9.61` (non-zero).
- Autodiff correctness verified: JVP-vs-FD relative error `5e-5`.

The surrogate Jacobian is non-SPD and the symmetric part is
numerically unrelated to the analytic tangent. The GNN maps
boundary displacements to boundary forces as a black-box
integrator; it was not trained with any constraint on the Jacobian,
so `df/du` is unsupervised between training points.

The Newton step `du = -K^{-1} r` requires K to be SPD for
guaranteed descent on the residual norm. On 1x1 (8 DOFs) stochastic
luck occasionally yields a valid direction; on larger patches
(40 DOFs for 5x5) the probability of descent collapses.

## How the two branches aimed to solve it

### Branch 1 — `feature/eigenvalue-projection-tangent` (ADR-0004)

Project the per-patch surrogate tangent onto the nearest SPD
matrix *before* global assembly, so the Newton step is guaranteed
to be a descent direction regardless of residual.

Pipeline:
```
K_sym     = 0.5 * (K + K.T)
lam, Q    = eigh(K_sym)
lam'      = clamp(lam, min = max(eps_rel * |lam|_max, eps_abs))
K_spd     = Q @ diag(lam') @ Q.T
```

Addresses the indefiniteness that pure symmetrisation (tried
previously, Obs #1953) did not remove.

### Branch 2 — `feature/modified-newton-elastic-tangent` (ADR-0005)

Replace the surrogate tangent entirely with a precomputed analytic
linear-elastic stiffness `K_analytic` on the same mesh (modified
Newton). Residual `r` still comes from the surrogate force.
`K_analytic` is SPD by construction, so the Newton step is a valid
descent direction irrespective of the surrogate's behaviour.
Quadratic convergence is sacrificed; linear convergence acceptable.

## Implementation summary

### Branch 1: `feature/eigenvalue-projection-tangent` (pushed, `f202f0e`)

Eigenvalue-projection SPD post-processing applied per patch before
global assembly.

- `_project_spd` helper in `base.py`:
  `K <- Q * max(Lambda, eps) * Q^T` with relative floor
  `eps_rel * |lambda|_max` and absolute floor `1e-10`.
- `surrogate_integrate_material` / `solve_matpatch`: kwargs
  `tangent_postproc in {'none', 'sym', 'spd'}`,
  `spd_eps_rel = 1e-6`. No env vars.
- Per-iteration diagnostics
  `{n_neg, lambda_min, lambda_max, ||K_spd - K_sym||_F / ||K_sym||_F}`
  accumulated in `self._spd_iter_history`; CSV dumped at end of
  increment on both convergence and `max_iter` exit.
- 4/4 unit tests in `tests/test_spd_projection.py`.
- ADR-0004.

### Main branch (commit `7e7cbec`)

Adaptive sub-incrementation added to `run_simulation_surrogate`:
kwargs `is_adaptive_timestepping`, `adaptive_max_subdiv = 8`;
retry/subdivide loop mirrors `run_simulation.py:1383-1423`.

### Branch 2: `feature/modified-newton-elastic-tangent` (not pushed)

Modified Newton with analytic linear-elastic tangent.

- `FEM.assemble_linear_elastic_k()`: full-mesh sparse elastic K
  via `_integrate_fe_subset` with zero-state inputs.
- `solve_matpatch` kwargs
  `nr_tangent in {'surrogate', 'elastic_analytic'}`, `K_elastic`.
  When `'elastic_analytic'`: substitute `K_combined` with
  `K_elastic` before constraint elimination; residual still
  surrogate.
- Twin `Planar` / `Solid` domain built in
  `run_simulation_surrogate` with
  `IsotropicElasticityPlaneStrain(E = e_young, nu = nu)`;
  assembled once.
- 2/2 unit tests in `tests/test_modified_newton.py`.
- ADR-0005.

## Findings — 5x5 patch residual trajectories

| Method                              | Iter 1 | Best               | Terminal (iter ~100) | Behaviour                                 |
|-------------------------------------|--------|--------------------|----------------------|-------------------------------------------|
| Surrogate K (baseline)              | 76.87  | 76.87              | NaN (singular)       | monotonic divergence to 3496 over 51 iter |
| Surrogate K + SPD projection        | 76.87  | 76.87              | ~3360                | bounded plateau, never converges          |
| Elastic K, interior pinned (K_bb)   | 76.87  | **5.28 @ iter 81** | 7.33                 | slow steady descent, slow upturn          |
| Elastic K, interior free (Schur)    | 76.87  | 13.82 @ iter 47    | 71.63                | faster descent then sharp divergence      |

## Interpretation

1. **Branch 1 (SPD projection)** eliminates indefiniteness. Removes
   the singular-matrix crash seen with raw surrogate K, gives a
   bounded residual. Does not converge because the symmetric part
   of the learned tangent is ~96 % wrong in Frobenius norm vs the
   analytic elastic tangent; the Newton step descends on the wrong
   functional.

2. **Branch 2, interior pinned (K_bb)** is the version currently
   checked in. Smaller effective step per iteration because K_bb
   is stiffer than the condensed boundary stiffness. Consequence:
   slow but steady descent from 76.87 to **5.28** over 81
   iterations — best residual achieved so far. After iteration 81
   the residual creeps upward to 7.33 by iter 100.

3. **Branch 2, interior free (Schur complement)** predicted to
   give the physically correct effective boundary stiffness
   matching how the surrogate was trained. Observed: faster initial
   descent (reached 13.82 by iter 47 vs iter 81 for pinned) but
   then pronounced divergence — residual climbs back to 71.63 at
   iter 99. Step size is better-matched initially; once u enters a
   region where the learned force field is strongly non-monotone,
   the larger Schur-complemented step overshoots harder than the
   conservative K_bb step.

4. **Both elastic variants fail to converge.** Diagnosis confirmed:
   the surrogate residual is not the gradient of a convex
   potential.
    - `f_gnn(u = 0) = 9.61` (non-zero bias; no zero-residual state).
    - 71-86 % antisymmetric Jacobian (non-conservative force
      field).
    - Modified Newton's "guaranteed descent" property requires
      residual monotonicity in u, which the GNN does not provide.

5. **Trade-off between K_bb and Schur.** K_bb gives a smaller
   step -> tolerates local non-monotonicity longer (lower minimum
   residual) but never reaches tolerance. Schur gives the right
   magnitude for a linear patch but is too aggressive in the
   plastic regime (larger divergence). Neither closes the gap to
   1e-2.

## Remaining remedies (not implemented)

- **Armijo / backtracking line search inside NR.** Guarantees
  monotone residual descent per iteration regardless of step
  direction quality. Easiest next fix.
- **Adaptive sub-incrementation.** Already wired via
  `is_adaptive_timestepping = True, adaptive_max_subdiv = 8`. Keeps
  each increment inside the linearisation basin. The 5x5 runs in
  the table above were not performed with it — re-run with it on
  for direct comparison.
- **Tangent-supervised training loss** (remedy 2 in
  `tasks/todo.md`). Adds `lambda * ||K_gnn - K_analytic||_F^2` to
  the surrogate loss. Constrains the Jacobian directly.
- **Option D energy head** (ADR-0003,
  `architecture/potential-head`). Architectural fix: predict
  scalar Phi; F and K derived so K is symmetric by Schwarz.
  Deferred.

## Conclusion

Branch 2 in the current (pinned-interior) form is the
best-performing remedy tested, reducing the 5x5 residual floor
from ~3360 to **5.28** — a factor of ~630 improvement — but still
does not meet the `< 1e-2` acceptance criterion. Root cause is
confirmed as the non-conservative, biased surrogate force field;
no post-hoc tangent treatment can fix that without touching the
model itself (tangent-supervised loss, or energy-head
architecture).

## References

- ADR-0001: graph surrogate for material patches.
- ADR-0002: hidden-state snapshot for Jacobian consistency
  (`64d74d1`).
- ADR-0003: energy head for symmetric tangent (deferred;
  `architecture/potential-head`).
- ADR-0004: eigenvalue projection SPD post-processing
  (`feature/eigenvalue-projection-tangent`, `f202f0e`).
- ADR-0005: modified Newton with elastic analytic tangent
  (`feature/modified-newton-elastic-tangent`).
- `tasks/todo.md`: remedies 3 (eigenvalue projection) and 5
  (modified Newton) with acceptance criteria.
- Memory: `project_surrogate_nr_descent_direction.md`.
