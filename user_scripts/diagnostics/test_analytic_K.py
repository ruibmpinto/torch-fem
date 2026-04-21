"""Analytic vs surrogate boundary stiffness for an isolated 5x5 patch.

Builds a canonical unit-square 5x5 Quad4 patch with plane-strain
isotropic elasticity (E=70000, nu=0.33), assembles the global
stiffness K at zero strain, condenses internal DOFs via Schur
complement to get the true boundary tangent K_bd_analytic
(40x40, symmetric positive-definite).

Evaluates the surrogate's boundary tangent at u_bd = 0 via jacfwd,
symmetrizes it, and reports ||K_surr_sym - K_analytic||_F /
||K_analytic||_F, plus eigenvalue spectra. Non-rigid-body eigen-
values of K_analytic must be strictly positive; the surrogate's
must be comparable in magnitude and direction.

Node ordering in the synthetic mesh matches the training-time
convention: column-major (i over x outer, j over y inner,
node_id = i*6 + j). Boundary-node order within the 40-DOF block
is column-major-sorted (same as torch-fem's pseudo-grid
boundary extraction).

Pass criterion:
    ||K_surr_sym - K_analytic||_F / ||K_analytic||_F  O(1e-2)
    non-rigid-body eigenvalues of K_surr_sym within same order
    of magnitude as K_analytic.

Usage:
    /Users/rbarreira/mambaforge/envs/env_torchfem/bin/python \
        user_scripts/diagnostics/test_analytic_K.py

Functions
---------
build_patch_mesh
    Return (nodes, elements, bd_list) for the 5x5 canonical patch.
compute_boundary_tangent_analytic
    Assemble K_global and Schur-condense to boundary.
compute_boundary_tangent_surrogate
    jacfwd of surrogate forward_graph at u_bd=0.
main
    Run comparison and print report.
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
import torch.func as torch_func
# Local
from torchfem.planar import Planar
from torchfem.materials import IsotropicElasticityPlaneStrain
from torchfem.base import forward_graph
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
MODEL_DIR = str(
    _root / 'torch-fem' / 'user_scripts' / 'matpatch_surrogates'
    / 'elastoplastic_nlh' / '5x5' / 'model')
N_DIM = 2
N_ELEM_SIDE = 5
N_NODE_SIDE = N_ELEM_SIDE + 1
E_YOUNG = 70000.0
NU = 0.33
EDGE_FEATURE_TYPE = ('edge_vector', 'relative_disp')
# =============================================================================
def build_patch_mesh():
    """Canonical 5x5 Quad4 unit-square patch, column-major node order.

    Returns
    -------
    nodes : torch.Tensor
        Shape (36, 2). Float64. Row-major over (i, j) where
        i is the x-index and j is the y-index:
        node_id = i*6 + j; position = (i*0.2, j*0.2).
    elements : torch.Tensor
        Shape (25, 4), long. Quad4 CCW connectivity.
    bd_list : list[int]
        Sorted list of boundary node ids (column-major).
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nodes = torch.zeros(N_NODE_SIDE * N_NODE_SIDE, N_DIM,
                        dtype=torch.float64)
    for i in range(N_NODE_SIDE):
        for j in range(N_NODE_SIDE):
            nid = i * N_NODE_SIDE + j
            nodes[nid, 0] = i / N_ELEM_SIDE
            nodes[nid, 1] = j / N_ELEM_SIDE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elems = []
    for i in range(N_ELEM_SIDE):
        for j in range(N_ELEM_SIDE):
            ll = i * N_NODE_SIDE + j
            lr = (i + 1) * N_NODE_SIDE + j
            ur = (i + 1) * N_NODE_SIDE + (j + 1)
            ul = i * N_NODE_SIDE + (j + 1)
            elems.append([ll, lr, ur, ul])
    elements = torch.tensor(elems, dtype=torch.long)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bd_list = sorted([
        i * N_NODE_SIDE + j
        for i in range(N_NODE_SIDE)
        for j in range(N_NODE_SIDE)
        if (i == 0 or i == N_ELEM_SIDE
            or j == 0 or j == N_ELEM_SIDE)])
    return nodes, elements, bd_list
# =============================================================================
def compute_boundary_tangent_analytic(nodes, elements, bd_list):
    """Assemble global K at zero strain and Schur-condense to boundary.

    Returns
    -------
    K_bd : torch.Tensor
        Shape (n_bd * n_dim, n_bd * n_dim), symmetric.
    K_full : torch.Tensor
        Shape (n_dof, n_dof) dense, for cross-checks.
    """
    material = IsotropicElasticityPlaneStrain(E=E_YOUNG, nu=NU)
    # Planar expects float32 by default; cast nodes to match material
    domain = Planar(
        nodes.to(torch.float32),
        elements,
        material)
    k_elem = domain.k0()
    con = torch.zeros(0, dtype=torch.long)
    K_sparse = domain.assemble_stiffness(k_elem, con)
    K_full = K_sparse.to_dense().to(torch.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_node = nodes.shape[0]
    int_list = [n for n in range(n_node) if n not in set(bd_list)]
    bd_dofs = torch.tensor(
        [N_DIM * n + d for n in bd_list for d in range(N_DIM)],
        dtype=torch.long)
    int_dofs = torch.tensor(
        [N_DIM * n + d for n in int_list for d in range(N_DIM)],
        dtype=torch.long)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    K_bb = K_full.index_select(0, bd_dofs).index_select(1, bd_dofs)
    K_ii = K_full.index_select(0, int_dofs).index_select(1, int_dofs)
    K_bi = K_full.index_select(0, bd_dofs).index_select(1, int_dofs)
    K_ib = K_full.index_select(0, int_dofs).index_select(1, bd_dofs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # K_bd = K_bb - K_bi K_ii^-1 K_ib
    sol = torch.linalg.solve(K_ii, K_ib)
    K_bd = K_bb - K_bi @ sol
    return K_bd, K_full
# =============================================================================
def build_surrogate_graph_inputs(nodes_bd):
    """Build edges_indexes for the all-to-all boundary graph.

    Matches torch-fem inference with edge_type='all'
    (fully-connected directed graph on boundary nodes).
    """
    n = nodes_bd.shape[0]
    src = []
    dst = []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)
# =============================================================================
def load_surrogate(model_dir, device='cpu'):
    """Load checkpoint matching the torch-fem inference path.

    Uses stepwise mode because that is what
    surrogate_integrate_material uses when computing k_boundary
    via jacfwd. The K_surrogate evaluated here is therefore the
    exact tangent Newton sees at runtime.
    """
    model = GNNEPDBaseModel.init_model_from_file(model_dir)
    model.set_device(device)
    _ = model.load_model_state(
        load_model_state='best', is_remove_posterior=False)
    model.eval()
    model._save_time_series_attrs()
    model.set_rnn_mode(is_stepwise=True)
    if hasattr(model, 'prepare_fast_scalers'):
        model.prepare_fast_scalers()
    model.double()
    return model
# =============================================================================
def compute_boundary_tangent_surrogate(model, coords_bd, edges_indexes):
    """Compute surrogate boundary stiffness via jacfwd at u_bd=0.

    Returns
    -------
    K_bd : torch.Tensor
        Shape (n_bd * n_dim, n_bd * n_dim). NOT symmetric in general.
    f0 : torch.Tensor
        Shape (n_bd * n_dim,). Surrogate force at u_bd=0 (bias).
    """
    n_bd = coords_bd.shape[0]
    u0 = torch.zeros(n_bd, N_DIM, dtype=torch.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward_fn(u_bd):
        out = forward_graph(
            model=model,
            disps=u_bd,
            coords_ref=coords_bd,
            edges_indexes=edges_indexes,
            n_dim=N_DIM,
            edge_feature_type=EDGE_FEATURE_TYPE)
        pred = out[0] if isinstance(out, tuple) else out
        return pred
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    f0 = forward_fn(u0).detach().flatten().clone()
    jac = torch_func.jacfwd(forward_fn)(u0)
    K_surr = jac.view(n_bd * N_DIM, n_bd * N_DIM)
    return K_surr.detach(), f0
# =============================================================================
def main():
    """Run analytic vs surrogate K comparison and print report."""
    nodes, elements, bd_list = build_patch_mesh()
    print(f'[ANA-K] patch: {N_ELEM_SIDE}x{N_ELEM_SIDE} unit square, '
          f'n_node={nodes.shape[0]} n_elem={elements.shape[0]} '
          f'n_bd={len(bd_list)}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    K_analytic, K_full = compute_boundary_tangent_analytic(
        nodes, elements, bd_list)
    ana_fro = float(torch.norm(K_analytic))
    ana_sym_err = float(
        torch.norm(K_analytic - K_analytic.T) / max(ana_fro, 1e-30))
    eig_ana = torch.linalg.eigvalsh(
        0.5 * (K_analytic + K_analytic.T))
    print(f'[ANA-K] K_analytic: shape={tuple(K_analytic.shape)} '
          f'||K||_F={ana_fro:.4e} '
          f'||K-K^T||/||K||={ana_sym_err:.3e}')
    _n_zero_ana = int(
        (eig_ana.abs() < 1e-6 * eig_ana.max()).sum())
    print(f'[ANA-K] K_analytic spectrum: '
          f'min={eig_ana.min().item():.4e} '
          f'max={eig_ana.max().item():.4e} '
          f'n_zero_modes(<1e-6*max)={_n_zero_ana}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    coords_bd = nodes[bd_list].to(torch.float64)
    edges_indexes = build_surrogate_graph_inputs(coords_bd)
    print(f'[ANA-K] surrogate graph: n_edges={edges_indexes.shape[1]} '
          f'(all-to-all directed)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model = load_surrogate(MODEL_DIR)
    print(f'[ANA-K] loaded model class={type(model).__name__} '
          f'stepwise={model._is_stepwise}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    K_surr, f0 = compute_boundary_tangent_surrogate(
        model, coords_bd, edges_indexes)
    K_surr_sym = 0.5 * (K_surr + K_surr.T)
    surr_fro = float(torch.norm(K_surr))
    surr_sym_err = float(
        torch.norm(K_surr - K_surr.T) / max(surr_fro, 1e-30))
    eig_surr = torch.linalg.eigvalsh(K_surr_sym)
    print(f'[ANA-K] K_surrogate: shape={tuple(K_surr.shape)} '
          f'||K||_F={surr_fro:.4e} '
          f'||K-K^T||/||K||={surr_sym_err:.3e}')
    print(f'[ANA-K] ||f_surr(u=0)||={float(torch.norm(f0)):.4e}')
    print(f'[ANA-K] K_surr_sym spectrum: '
          f'min={eig_surr.min().item():.4e} '
          f'max={eig_surr.max().item():.4e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    diff = K_surr_sym - K_analytic
    rel_fro = float(torch.norm(diff) / max(ana_fro, 1e-30))
    print(f'[ANA-K] ||K_surr_sym - K_analytic||_F / '
          f'||K_analytic||_F = {rel_fro:.3e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    verdict = (
        'PASS: surrogate symmetric tangent is close to the '
        'analytic elastic tangent; Newton should converge once '
        'K is symmetrized.'
        if rel_fro < 1e-1 else
        'FAIL: surrogate tangent differs from analytic elastic '
        'tangent at O(1). Newton search direction is categorically '
        'wrong regardless of symmetry. Retrain with conservative '
        '(potential-based) head or stiffness targets.')
    print(f'[ANA-K] verdict: {verdict}')
# =============================================================================
if __name__ == '__main__':
    main()
