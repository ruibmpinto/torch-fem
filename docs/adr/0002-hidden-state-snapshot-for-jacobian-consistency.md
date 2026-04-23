# ADR-0002: Hidden-State Snapshot in `surrogate_integrate_material`

* **Status:** Accepted
* **Date:** 2026-04-22
* **Supersedes:** —
* **Related:** ADR-0001 (GNN surrogate), ADR-0003 (5x5 energy-head)

## Context and Problem Statement

The stepwise (RNN) surrogate path in `surrogate_integrate_material`
builds the boundary tangent stiffness `K = dF/du` either via
`torch.func.jacfwd` (parallel) or column-by-column via
`torch.func.jvp` (sequential). The GNN's hidden states
(`_gnn_epd_model._encoder/_processor/_decoder._hidden_states`) are
**mutated on every forward pass** when `is_stepwise=True` — each
forward call advances the RNN one timestep and overwrites the stored
hidden states in place.

For a single `jacfwd` pass this is benign: `jacfwd` evaluates the
model once for the forward value, then re-runs forward passes on
perturbed inputs internally (each starting from the same model state
at function-entry). However for the sequential-jvp loop the model
state is **not** re-initialised between columns:

```
state h0 → forward(u0)      → forces   (state drifts to h1)
state h1 → jvp(forward, e1) → col 1    (state drifts to h2)
state h2 → jvp(forward, e2) → col 2    (state drifts to h3)
...
```

Each column of `K` is the directional derivative evaluated at a
**different operating point** `h_i`. Assembled together they do not
form a coherent Jacobian of any single function. Newton-Raphson then
takes the "update" `Δu = -K⁻¹ F` — which points nowhere sensible —
and the residual grows instead of shrinking.

### Observed symptom (elastoplastic_nlh 1x1, mesh 10×10, 50 inc)

| Version                    | Inc 1 residuals          |
|----------------------------|--------------------------|
| Prior converging run       | 114.8 → 3.39 → 7.67e-03  |
| Post-regression (this ADR) | 114.8 → 478.76 → diverge |

The initial residual `‖F(u=0)‖ = 114.8` matches exactly, confirming
`F` is computed correctly. Only `K` is wrong.

## Decision Drivers

* Sequential-jvp path must be usable (constant memory; `jacfwd`
  memory-blows for large patches).
* Parallel `jacfwd` path must also be safe against any future RNN
  changes that leak state into auxiliary pre-traces.
* Stepwise RNN mode is required for path-dependent plasticity — the
  hidden state carry *between* converged increments must remain
  correct.
* Node-ordering was ruled out as a cause (FD-vs-jacfwd agreement
  `5e-5`, obs #1972).

## Considered Options

* **Option A:** Disable stepwise mode (`is_stepwise=False`).
  Rejected — loses temporal state needed for elastoplasticity.
* **Option B:** Disable sequential-jvp (always use `jacfwd`).
  Rejected — memory infeasible for 5x5 and larger patches.
* **Option C:** Detach hidden states inside the GNN forward pass so
  they never appear as closure state. Rejected — the detachment was
  already there; the bug is state *mutation*, not autograd leakage.
* **Option D:** Snapshot `h0` once per patch, restore before every
  forward/jvp call so every column is linearised at the same
  operating point.

## Decision Outcome

Chosen option: **D — h0 snapshot and restore**.

### Implementation (src/torchfem/base.py)

Inside the per-patch loop of `surrogate_integrate_material`:

1. `_load_hidden_into_model(mdl, h)` helper writes a hidden-state
   dict into `mdl._gnn_epd_model._encoder / _processor (all layers)
   / _decoder._hidden_states`.
2. `_h0_snapshot = copy.deepcopy(detach_hidden_states(
   hidden_states[patch_id]))` — captured once when the patch is
   entered, detached and deep-copied so subsequent in-place mutations
   of the model's state dict cannot corrupt it.
3. `_restore_h0()` closure reloads the snapshot into the model. It
   is called:
   * Before `torch_func.jacfwd` in the parallel branch.
   * Before the force evaluation in the sequential branch.
   * Before **every** `torch_func.jvp` inside the column loop.
4. `_hs_trial_saved = hidden_states_trial` preserves the trial
   hidden state produced by the force evaluation. The jvp columns
   overwrite `hidden_states_trial` with drifted values, so it is
   reset to `_hs_trial_saved` after the loop. On Newton convergence
   this saved state is then stored into `hidden_states_dict[
   patch_id]` by `solve_matpatch`, giving the correct carry for the
   next increment.

### Positive Consequences

* `K` is now a consistent Jacobian of `forward` at `h0`.
* Sequential-jvp path converges identically to `jacfwd` parallel
  path up to floating-point round-off.
* The fix is local to `surrogate_integrate_material`; `solve_matpatch`
  and the GNN architecture are untouched.

### Negative Consequences

* One extra `copy.deepcopy(detach_hidden_states(...))` per jvp
  column (≈ `n_dof_boundary` times per patch per Newton iteration).
  For 1x1 this is 8 extra copies of a small dict; for 5x5 it is 40.
  Measured overhead is `<2%` of a Newton iteration — dominated by
  the jvp itself.
* The fix depends on the private layout of `GNNEPDBaseModel`
  (`_gnn_epd_model._encoder._hidden_states`, etc.). If graphorge
  renames these attributes, the helper must be updated.

### Provenance

The fix was originally prototyped in commit `40ebc35` (branch
`debug/node_ordering`, 2026-04-22 00:21) and refined in `0384e8a`
after the per-column K/F post-hoc patches were removed. The
`debug/node_ordering` branch was **never merged to main**, so the
same regression silently reappeared on `main` once the 1x1 model
was restored. Recovered and re-applied in commit `64d74d1`.

### Related claude-mem observations

* **#999 (2026-04-07)** — original refactor extracting the h0
  snapshot mechanism.
* **#1089 (2026-04-07)** — full surrogate_integrate_material /
  hidden state management description.
* **#1325, #1326, #1329 (2026-04-10/11)** — autodiff vs state
  mutation analysis.
* **#1925 (2026-04-21)** — confirmation that stepwise mode updates
  hidden states unconditionally with no convergence gating.
* **#1927 (2026-04-21)** — reference converging run used as ground
  truth (`results/elastoplastic_nlh/2d/quad4/mesh_10x10/patch_1x1/
  n_time_inc_50/`).
