# ADR-0005: Modified Newton with linear-elastic analytic tangent

## Status

Proposed, on branch `feature/modified-newton-elastic-tangent`.

## Context

Branch 1 (eigenvalue projection, ADR-0004) guarantees a symmetric
positive-definite surrogate tangent but the learned tangent's
symmetric part remains ~96 % wrong in Frobenius norm versus the
analytic elastic tangent (Obs #2075, #2115). On 2x2 and 5x5 patches
Newton-Raphson still stalls / oscillates (e.g. 5x5:
`76.9 -> 3360` over 51 iterations) because the descent direction
`-K_spd^{-1} r` does not correspond to the residual-norm descent
direction.

Remedy 5 from `tasks/todo.md`: use the analytic linear-elastic
stiffness `K_analytic` for the Newton step direction while keeping
the surrogate force in the residual. `K_analytic` is SPD by
construction. Guaranteed descent on the residual-norm surrogate.
Quadratic convergence is lost; linear convergence is acceptable.

## Decision

Add a `nr_tangent` control to `solve_matpatch` with choices
`{'surrogate', 'elastic_analytic'}`:

- `'surrogate'` (default): unchanged behaviour. Uses the GNN tangent
  (possibly post-processed via ADR-0004's `tangent_postproc='spd'`).
- `'elastic_analytic'`: caller supplies a precomputed sparse
  `K_elastic` (built once per simulation from a linear-elastic
  twin domain); `solve_matpatch` replaces the per-iteration
  `K_combined` with `K_elastic` before constraint elimination.
  Residual `F_int - F_ext` still uses surrogate force.

`K_elastic` is assembled by `FEM.assemble_linear_elastic_k()`, a new
helper that wraps `_integrate_fe_subset` with zero-state inputs.
It is called once on a twin domain constructed from the same mesh
and `IsotropicElasticityPlaneStrain`/`IsotropicElasticity3D`
material with the current E and nu.

For the elastoplastic_nlh configuration (AA2024, E = 70 GPa,
nu = 0.33) the twin is `Planar(nodes, elements,
IsotropicElasticityPlaneStrain(E=70000, nu=0.33))`.

## Exposed controls

- `FEM.solve_matpatch(..., nr_tangent='surrogate', K_elastic=None)`.
- `run_simulation_surrogate(..., nr_tangent='surrogate')`.

Validation raises if `nr_tangent='elastic_analytic'` and
`K_elastic` is `None`. When `'elastic_analytic'` is selected the
surrogate Jacobian is still computed (but discarded at assembly);
a follow-up could skip the jvp loop entirely for a large speedup.

## Consequences

- Guaranteed descent: `K_analytic` is SPD and condition number
  comparable to a standard linear-elastic problem (order 1e3 for
  simple tension geometries), so the Newton step never amplifies
  the residual.
- Convergence rate: linear at best. The larger the plastic strain,
  the more `K_analytic` differs from the true elastoplastic
  tangent; the residual stalls at a plateau determined by this
  mismatch.
- Cost: one `_integrate_fe_subset` assembly at simulation start
  (O(n_elem) work), then no per-iteration Jacobian cost beyond
  what the surrogate already pays.
- No change to surrogate training, architecture, or the
  `_h0_snapshot` fix from ADR-0002.

## Rejected alternatives

- Computing `K_analytic` per iteration from the elastoplastic
  algorithmic tangent: would recover quadratic convergence but
  defeats the purpose (we'd be running the conventional solver).
- Hybrid K = alpha * K_surr + (1 - alpha) * K_elastic: adds a
  tuning knob without a principled choice of alpha.
- Using K_spd from ADR-0004 combined with a backtracking line
  search: addresses step size, not direction correctness; does
  not solve the descent-cone problem on 5x5.

## Verification

- `tests/test_modified_newton.py`: checks
  `assemble_linear_elastic_k` shape, symmetry (1e-6 relative),
  and displacement-state independence.
- `tests/test_spd_projection.py`: unchanged, still green.
- Acceptance from `tasks/todo.md`: on 5x5 elastoplastic_nlh,
  residual monotonically non-increasing to < 1e-2 within 200 NR
  iterations, no singular matrix. Record stall level if present.

## References

- ADR-0002: hidden-state snapshot.
- ADR-0003: energy head (deferred).
- ADR-0004: eigenvalue projection SPD post-processing.
- `tasks/todo.md` remedies 3 and 5.
- Memory: `project_surrogate_nr_descent_direction.md`.
