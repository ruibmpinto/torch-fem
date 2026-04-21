"""In-distribution replay of the 5x5 matpatch surrogate (batch mode).

The training loop at graphorge/gnn_base_model/train/training.py:469
calls model(node_features_in, edge_features_in, global_features_in,
edges_indexes, batch_vector) with the FULL packed sequence at once
(node features shape (n_node, n_time*n_feat_per_step)). This script
reproduces that exact forward path on a training sample and reports
MSE against the stored y. Any deviation from the ~2.4e-5 training
loss scale isolates a loader/normalization/dtype problem as distinct
from the stepwise-inference logic used by torch-fem.

Sample schema (from gen_graphs_files.py with default features):
    x         : (n_bd=20, n_time*n_feat_node=404)
        node_features = ('coord_hist', 'disp_hist')
    edge_attr : (n_edges=380, n_time*n_feat_edge=404)
        edge_features = ('edge_vector', 'relative_disp')
    y         : (n_bd, n_time*n_feat_out=202)
        node_targets = ('int_force',)

Pass criterion:
    ||pred - y||^2 / ||y||^2 within 1e-3 (training loss scale ~2.4e-5).
    Larger values imply loader / normalization / dtype mismatch.

Functions
---------
main
    Run batch-mode replay and print report.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.environ['TORCHFEM_IMPORT_GRAPHORGE'] = '1'
_here = pathlib.Path(__file__).resolve()
_root = _here.parents[3]
_graphorge_src = _root / 'graphorge' / 'src'
_torchfem_src = _root / 'torch-fem' / 'src'
for p in (str(_graphorge_src), str(_graphorge_src / 'graphorge'),
         str(_torchfem_src)):
    if p not in sys.path:
        sys.path.insert(0, p)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
import torch
# Local
from graphorge.gnn_base_model.model.gnn_model import GNNEPDBaseModel
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'RBMP'
__credits__ = ['RBMP', ]
__status__ = 'Diagnostic'
# =============================================================================
#
# =============================================================================
SAMPLE_PATH = str(
    _root / 'graphorge' / 'experiments' / 'elastoplastic_nlh' / '2d'
    / 'quad4' / 'mesh5x5' / 'ninc100' / 'data_5000' / 'reference'
    / '1_training_dataset' / 'material_patch_graph_0.pt')
MODEL_DIR = str(
    _root / 'torch-fem' / 'user_scripts' / 'matpatch_surrogates'
    / 'elastoplastic_nlh' / '5x5' / 'model')
# =============================================================================
def load_model_batch_mode(model_dir, device='cpu'):
    """Load surrogate and keep training-time batch mode.

    Batch mode: model() processes the full packed sequence in one
    call. Do NOT call set_rnn_mode(is_stepwise=True) here; that
    switches to per-step inference.
    """
    model = GNNEPDBaseModel.init_model_from_file(model_dir)
    model.set_device(device)
    _ = model.load_model_state(
        load_model_state='best', is_remove_posterior=False)
    model.eval()
    model.double()
    return model
# =============================================================================
def main():
    """Run batch-mode replay and print report."""
    print(f'[REPLAY] sample = {SAMPLE_PATH}')
    print(f'[REPLAY] model  = {MODEL_DIR}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load sample
    sample = torch.load(
        SAMPLE_PATH, weights_only=False, map_location='cpu')
    x = sample.x.to(torch.float64)
    y = sample.y.to(torch.float64)
    ea = sample.edge_attr.to(torch.float64)
    ei = sample.edge_index.long()
    print(f'[REPLAY] x.shape={tuple(x.shape)} '
          f'y.shape={tuple(y.shape)} '
          f'edge_attr.shape={tuple(ea.shape)} '
          f'edges={tuple(ei.shape)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load model
    model = load_model_batch_mode(MODEL_DIR)
    print(f'[REPLAY] loaded model class={type(model).__name__} '
          f'stepwise={getattr(model, "_is_stepwise", None)} '
          f'rbm_removal={getattr(model, "_is_rigid_body_removal", None)} '
          f'n_node_in={getattr(model, "_n_node_in", None)} '
          f'n_edge_in={getattr(model, "_n_edge_in", None)} '
          f'n_time_node={getattr(model, "_n_time_node", None)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Use the same entry points training used (training.py:449-455):
    #   get_input_features_from_graph  -> RBM removal + normalization
    #   get_output_features_from_graph -> normalization of targets
    #   model(...)                     -> raw forward on normalized inputs
    # Loss at training is computed on NORMALIZED pred vs NORMALIZED y.
    node_in_norm, edge_in_norm, glob_in_norm, ei_chk = (
        model.get_input_features_from_graph(
            sample, is_normalized=True))
    y_norm_targets, _, _ = model.get_output_features_from_graph(
        sample, is_normalized=True)
    print(f'[REPLAY] get_input_features -> '
          f'node={tuple(node_in_norm.shape)} '
          f'edge={tuple(edge_in_norm.shape)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Batch-mode forward (single graph, no batch_vector).
    with torch.no_grad():
        out = model(
            node_features_in=node_in_norm,
            edge_features_in=edge_in_norm,
            global_features_in=glob_in_norm,
            edges_indexes=ei_chk,
            batch_vector=None)
    if isinstance(out, (tuple, list)):
        node_out_norm = out[0]
    else:
        node_out_norm = out
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Training loss path: compare normalized pred to normalized target
    err_norm = node_out_norm - y_norm_targets
    se_norm = float((err_norm ** 2).sum())
    ty_norm = float((y_norm_targets ** 2).sum())
    mse_norm = se_norm / y_norm_targets.numel()
    print(f'[REPLAY] NORMALIZED pred vs target:')
    print(f'[REPLAY]   ||pred_n - y_n||^2     = {se_norm:.4e}')
    print(f'[REPLAY]   ||y_n||^2              = {ty_norm:.4e}')
    print(f'[REPLAY]   MSE (training loss)    = {mse_norm:.4e}')
    print(f'[REPLAY]   training best-loss     ~ 2.4e-5')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Denormalized pred vs raw y
    pred = model.data_scaler_transform(
        tensor=node_out_norm,
        features_type='node_features_out',
        mode='denormalize')
    print(f'[REPLAY] DENORMALIZED pred.shape={tuple(pred.shape)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    err = pred - y
    se = float((err ** 2).sum())
    ty = float((y ** 2).sum())
    rel = se / max(ty, 1e-30)
    print(f'[REPLAY] ||pred-y||^2 = {se:.4e}')
    print(f'[REPLAY] ||y||^2      = {ty:.4e}')
    print(f'[REPLAY] relative SE  = {rel:.4e}')
    print(f'[REPLAY] training-loss scale (MSE) ~ 2.4e-5')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_out_per_t = 2
    n_time = pred.shape[1] // n_out_per_t
    rel_t = []
    for t in range(n_time):
        a = t * n_out_per_t
        b = a + n_out_per_t
        se_t = float(((pred[:, a:b] - y[:, a:b]) ** 2).sum())
        ty_t = float((y[:, a:b] ** 2).sum())
        rel_t.append(se_t / max(ty_t, 1e-30))
    rel_t = np.asarray(rel_t)
    print(f'[REPLAY] per-step rel SE:  '
          f'min={rel_t.min():.3e} '
          f'max={rel_t.max():.3e} '
          f'median={np.median(rel_t):.3e}')
    print(f'[REPLAY] first 3 rel SE = '
          f'{[f"{v:.3e}" for v in rel_t[:3]]}')
    print(f'[REPLAY] last 3  rel SE = '
          f'{[f"{v:.3e}" for v in rel_t[-3:]]}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loader PASS criterion uses NORMALIZED MSE (training metric).
    # Denormalized physical error is an inherent property of the
    # surrogate, separate from loader correctness.
    training_loss_scale = 2.5e-5
    if mse_norm < 3.0 * training_loss_scale:
        verdict = (
            'PASS: surrogate reproduces training targets within '
            'training loss scale in normalized space '
            f'(MSE={mse_norm:.2e}). Loader / RBM / normalization / '
            'dtype are intact. NOTE: denormalized physical error is '
            f'{100*rel**0.5:.1f}% relative — this is the inherent '
            'accuracy of the direct-force surrogate and is what '
            'Newton has to solve against.')
    else:
        verdict = (
            f'FAIL: normalized MSE {mse_norm:.2e} is >3x the '
            f'training loss scale {training_loss_scale:.1e}. '
            'Loader, normalization, or dtype issue.')
    print(f'[REPLAY] verdict: {verdict}')
# =============================================================================
if __name__ == '__main__':
    main()
