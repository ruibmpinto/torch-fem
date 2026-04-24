# ADR-0004: Eigenvalue-projection SPD post-processing for the surrogate tangent

## Status

Proposed, on branch `feature/eigenvalue-projection-tangent`.

## Context

The GNN surrogate produces a per-patch boundary tangent `K` that is
not SPD. Measurements on 5x5 elastoplastic_nlh patches:

- `||K - K.T|| / ||K||` in 0.78-0.86,
- fraction of negative eigenvalues ~ 60 %,
- `||K_sym - K_analytic||_F / ||K||_F` ~ 0.96.

Newton-Raphson with this `K` diverges on 2x2 and 5x5 patches (e.g.
residual 76.87 -> 3496.53 over 51 iterations on 5x5, singular matrix
on 2x2). The correct descent direction requires `K` to be SPD.

ADR-0002 fixed hidden-state drift during sequential jvp so Jacobian
columns are linearised at the same RNN operating point. That reduced
the 5x5 residual envelope from `76.87 -> 3496` to `114.8 -> 478.76`
but Newton still diverges.

ADR-0003 proposed a long-term energy-head architecture (Option D) with
symmetric tangent by construction. Implementation is non-trivial and
deferred.

Post-hoc symmetrisation (`K <- 0.5 * (K + K.T)`) was tested in an
earlier branch. It removes asymmetry but leaves indefiniteness:
residual still diverges (101 -> 5480 over 7 iter). Eigenvalue
clamping has not been tried previously.

## Decision

Apply an SPD projection at the per-patch level after the tangent is
computed, before assembly into the global sparse `K`:

    K_sym    = 0.5 * (K + K.T)
    lam, Q   = eigh(K_sym)
    floor    = max(eps_rel * max(|lam|), eps_abs)
    lam'     = clamp(lam, min=floor)
    K_spd    = Q @ diag(lam') @ Q.T

Per-patch boundary tangents are dense and small (8x8 for 1x1, 16x16
for 2x2, 40x40 for 5x5), so the eigendecomposition cost is
negligible compared with the surrogate forward/jvp evaluations.

Defaults: `eps_rel = 1e-6`, `eps_abs = 1e-10`.

## Exposed controls

Added as keyword arguments to `FEM.surrogate_integrate_material` and
`FEM.solve_matpatch`, and plumbed through `run_simulation_surrogate`
as function parameters:

- `tangent_postproc` : `{'none', 'sym', 'spd'}`, default `'none'`.
- `spd_eps_rel` : `float`, default `1e-6`.

No environment variables, no module-level uppercase constants.
Legacy env var `TORCHFEM_K_POSTPROC` is not reintroduced.

## Instrumentation

When `tangent_postproc == 'spd'`, per-patch diagnostics are collected
on each Newton iteration into `self._last_spd_diagnostics` (keyed by
patch id):

- `n_neg`  : number of negative eigenvalues pre-clamp,
- `lambda_min`, `lambda_max` : spectrum bounds pre-clamp,
- `frob_ratio` : `||K_spd - K_sym||_F / ||K_sym||_F`.

On convergence, if `stiffness_output_dir` is set, the diagnostics from
the last (converged) iteration of the increment are written to
`<stiffness_output_dir>/spd_diag_<patch_size>_inc<n>.csv`.

## Consequences

- Newton step is guaranteed to be a descent direction (K_spd is
  SPD by construction), removing the indefiniteness failure mode.
- The symmetric part of K remains the learned one, which memory and
  diagnostics indicate is ~96 % wrong in Frobenius norm vs. the
  analytic elastic tangent. Quadratic convergence is not recovered;
  residual may stall above tolerance.
- The stall magnitude quantifies the remaining wrongness of the
  symmetric part and is the trigger for either remedy 2
  (tangent-supervised loss) or the ADR-0003 energy head.
- Cost: one `torch.linalg.eigh` on a ~40x40 dense tensor per patch
  per NR iteration. Negligible versus the surrogate forward/jvp.

## Rejected alternatives

- Global SPD projection on the assembled sparse K. Eigendecomposition
  on the global K would be `O(n_dof^3)` per iteration. Per-patch
  dense eigh is `O(n_bd^3)` with `n_bd <= 80`, orders of magnitude
  cheaper.
- Re-enabling the removed `TORCHFEM_K_POSTPROC` env var. Module-level
  uppercase global rejected per code-style convention; replaced by
  constructor/call-site kwargs.
- Mean-force removal (previously paired with symmetrisation). Not
  reintroduced; it masks f(u=0) bias without addressing the
  descent-direction problem.

## Verification

Unit tests: `torch-fem/tests/test_spd_projection.py`.

End-to-end acceptance (per tasks/todo.md):

1. 1x1 regression: baseline residual trajectory preserved within
   1e-6 rel. difference with `tangent_postproc='spd'`.
2. 2x2: no singular matrix; residual monotonically non-increasing
   over 10 iter.
3. 5x5: residual monotonically non-increasing for >= 10 iter;
   final rel. residual < 1e-2 by iter 50. Stall is acceptable and
   its magnitude is to be recorded in a follow-up ADR.

## References

- ADR-0002: hidden-state snapshot for Jacobian consistency.
- ADR-0003: energy head for symmetric tangent (deferred).
- Memory observation #1953: symmetrisation alone insufficient.
- Memory observation #1951: 60 % negative eigenvalues,
  condition numbers 4e16-2.5e17.
