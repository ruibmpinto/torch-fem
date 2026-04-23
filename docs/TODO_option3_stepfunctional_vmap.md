# TODO: Option 3 — `step_functional` + `torch.func.grad` + `vmap`

Status: not implemented. Option 1 (stepwise autograd with `detach_hidden_states`) lives on `graphorge/architecture/potential-head` and unblocks training in the short term. Option 3 is the long-term target because the same primitive produces forces at training time and `K = Hess(Φ)` at inference.

## Why this exists

The current `loss_nature = 'force_from_potential'` path (graphorge `training.py`) and the planned `forward_graph` energy-head branch in `torch-fem/src/torchfem/base.py` call autograd against a GNN that mutates its RNN hidden state during `forward`. Two consequences:

1. `torch.func.hessian` / `torch.func.grad` refuse to transform modules that mutate state. A functional variant is required.
2. Keeping state-threading explicit (`h_prev` as an argument rather than attribute) makes symmetry provable line by line: the algorithmic tangent is `∂²Φ/∂u_t²` with `h_{t-1}` a frozen closure constant.

## What to build

### graphorge side

Add a pure-function variant of one RNN step on `GNNEPDBaseModel`:

```
def step_functional(self, node_features_in, edge_features_in,
                    global_features_in, edges_indexes,
                    hidden_states_in, batch_vector=None):
    """Stateless single-step forward.

    Returns (phi, hidden_states_out). Does NOT read or write
    self._hidden_states. All RNN cells take their previous
    state through the hidden_states_in dict and emit the next
    state in hidden_states_out.
    """
```

Requirements:
- No in-place writes to any `_hidden_states` attribute on the
  module tree. The existing `forward` / `step` paths keep
  reading/writing `_hidden_states`; `step_functional` must use
  a local dict threaded through encoder / each GIN processor
  layer / decoder.
- `phi` has shape `(n_graphs,)` (scalar per graph after sum-pool).
- `hidden_states_out` mirrors the nested structure of the
  current `self._hidden_states` dict so that the caller can
  feed it back in at `t+1`.

### Training loop (graphorge)

Replace the stepwise Python loop currently in `training.py`:

```
from torch.func import grad as func_grad

def loss_step(u_t, coord_t, edge_t, global_t,
              h_prev, edges_indexes, batch_vector, model):
    x_t = torch.cat([coord_t, u_t], dim=1)
    phi, _ = model.step_functional(
        x_t, edge_t, global_t, edges_indexes,
        hidden_states_in=h_prev, batch_vector=batch_vector)
    return phi.sum()

f_of_u = func_grad(loss_step, argnums=0)

h = None  # initial hidden state
f_list = []
for t in range(n_time):
    u_t = disp_hist[:, n_dim*t:n_dim*(t+1)]
    coord_t = coord_hist[:, n_dim*t:n_dim*(t+1)]
    edge_t = edge_hist[:, n_edge_in*t:n_edge_in*(t+1)]
    f_t = f_of_u(u_t, coord_t, edge_t, None,
                 h, edges_indexes, batch_vector, model)
    f_list.append(f_t)
    # Re-forward to advance hidden state, detached so next
    # step's backward graph stops here.
    with torch.no_grad():
        _, h = model.step_functional(
            torch.cat([coord_t, u_t], dim=1),
            edge_t, None, edges_indexes,
            hidden_states_in=h, batch_vector=batch_vector)
    h = _detach_tree(h)
f_packed = torch.cat(f_list, dim=1)
```

Payoff over Option 1:
- `h` is a plain dict carried in Python, not mutated state — no risk of cross-sample leakage if a batch is interrupted.
- Same `step_functional` composes with `torch.func.hessian` at inference time.

### Batched variant (vmap over graphs in the mini-batch)

If per-step kernels are launch-bound on GPU, `torch.func.vmap` the per-step call over the graph batch dimension:

```
phi_of_u = torch.func.vmap(step_functional, in_dims=(0, 0, 0, 0, 0, None, None))
```

Not vmap-able over the time axis because `h` has a sequential dependency.

### torch-fem inference (`base.py`)

Same function, different caller. At inference:

```
K = torch.func.hessian(
    lambda u: model.step_functional(
        build_x(coord, u), edge_t, None,
        edges_indexes, h_prev_detached, batch_vector)[0].sum())(u_bd)
```

Chain-rule rescale from normalized-input Hessian to physical-input Hessian via `K_phys = a² · K_norm` with `a = 2/(scale_max - scale_min)` per dimension (MinMax scaler assumed in this branch; StandardScaler uses `a = 1/std`).

## Correctness checks to include when landing

1. Unit test: `‖K − K.T‖ / ‖K‖ < 1e-10` on a randomly initialized energy-head model at a random displacement and random detached hidden state. Must pass without requiring `.symm()`.
2. Unit test: `step_functional` and current batched `forward` produce identical `phi` per step up to floating-point noise, given the same hidden-state history. Catches any drift in the state-threading rewrite.
3. Single-sample overfit: `force_from_potential` loss on one sample converges below `1e-4` within 5000 steps with stepwise autograd and Option 3 within the same budget.
4. `torch.func.hessian` runs on `step_functional` without raising over module mutation — the regression test for "we successfully decoupled state from the module".

## Known correctness concern orthogonal to this option

The normalization chain rule is not implemented in the current graphorge `force_from_potential` branch. `u_t` passed to autograd is the normalized displacement, so `∂Φ/∂u_norm = F_phys · scale_u`, which does not equal `F_norm = (F_phys − mean_F)/scale_F`. The loss currently compares mismatched quantities. Fix path: either
- differentiate in normalized input and rescale to `F_norm` via `F_norm = (∂Φ/∂u_norm) · (scale_u/scale_F) − mean_F/scale_F` (MinMax) before the MSE, or
- denormalize `u` inside the forward, compute `Φ` in physical space, and normalize the target side only.

Resolve this together with Option 3; `step_functional` is the natural place to carry the rescale explicitly.

## Files involved

- `graphorge/src/graphorge/gnn_base_model/model/gnn_model.py` — add `step_functional`.
- `graphorge/src/graphorge/gnn_base_model/model/gnn_epd_model.py` — stateless encoder/processor/decoder call paths.
- `graphorge/src/graphorge/gnn_base_model/model/gnn_architectures.py` — GIN layer stateless variant.
- `graphorge/src/graphorge/gnn_base_model/train/training.py` — swap stepwise loop body.
- `torch-fem/src/torchfem/base.py` — energy-head inference branch in `forward_graph`.

## References

- ADR-0003 energy-head-for-symmetric-tangent (this repo, `docs/adr/`).
- Memory obs #2066 (plan) and #2115 (decision) for the original spec.
- Option 1 landed on graphorge branch `architecture/potential-head` in `training.py` under `loss_nature == 'force_from_potential'`.
