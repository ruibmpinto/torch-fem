"""Unit tests for _project_spd in torchfem.base.

Classes
-------
(no classes — plain pytest functions)

Functions
---------
test_project_spd_clamps_negative_eigenvalues
    Verify negative eigenvalues are lifted to the positive floor.
test_project_spd_preserves_already_spd
    Verify an SPD matrix passes through with negligible distortion.
test_project_spd_symmetrises_asymmetric_input
    Verify output is symmetric even when input is not.
test_project_spd_diag_counts_negatives
    Verify the returned diagnostic dict reports the correct
    negative-eigenvalue count.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
# Third-party
import numpy as np
import pytest
import torch
# Local
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, 'src')))
from torchfem.base import _project_spd
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Pinto'
__credits__ = ['Rui Pinto', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
def _random_symmetric_with_negative_eigs(n, n_neg, seed=0):
    """Build a symmetric matrix with a prescribed number of
    negative eigenvalues."""
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n,
                                       dtype=torch.float64))
    lam = torch.linspace(-3.0, 5.0, n, dtype=torch.float64)
    lam[:n_neg] = torch.linspace(-3.0, -0.1, n_neg,
                                 dtype=torch.float64)
    lam[n_neg:] = torch.linspace(0.1, 5.0, n - n_neg,
                                 dtype=torch.float64)
    return (q * lam.unsqueeze(0)) @ q.T, lam
# -----------------------------------------------------------------------------
def test_project_spd_clamps_negative_eigenvalues():
    k, _ = _random_symmetric_with_negative_eigs(
        n=8, n_neg=3, seed=0)
    k_spd, diag = _project_spd(k, eps_rel=1e-6)
    lam_out = torch.linalg.eigvalsh(k_spd)
    floor = max(1e-6 * k.abs().max().item(), 1e-10)
    assert lam_out.min().item() >= floor - 1e-10
    assert diag['n_neg'] == 3
    err_sym = torch.linalg.norm(k_spd - k_spd.T).item()
    assert err_sym < 1e-10
# -----------------------------------------------------------------------------
def test_project_spd_preserves_already_spd():
    torch.manual_seed(42)
    a = torch.randn(10, 10, dtype=torch.float64)
    k = a @ a.T + 1e-1 * torch.eye(10,
                                   dtype=torch.float64)
    k_spd, diag = _project_spd(k, eps_rel=1e-6)
    rel_err = (
        torch.linalg.norm(k_spd - k).item()
        / torch.linalg.norm(k).item())
    assert rel_err < 1e-10
    assert diag['n_neg'] == 0
    assert diag['frob_ratio'] < 1e-10
# -----------------------------------------------------------------------------
def test_project_spd_symmetrises_asymmetric_input():
    torch.manual_seed(1)
    k = torch.randn(12, 12, dtype=torch.float64)
    assert torch.linalg.norm(k - k.T).item() > 1e-6
    k_spd, _ = _project_spd(k, eps_rel=1e-6)
    err_sym = torch.linalg.norm(k_spd - k_spd.T).item()
    assert err_sym < 1e-10
# -----------------------------------------------------------------------------
def test_project_spd_diag_counts_negatives():
    k, lam = _random_symmetric_with_negative_eigs(
        n=16, n_neg=7, seed=3)
    _, diag = _project_spd(k, eps_rel=1e-6)
    assert diag['n_neg'] == 7
    assert diag['lambda_min'] == pytest.approx(
        lam.min().item(), rel=1e-6)
    assert diag['lambda_max'] == pytest.approx(
        lam.max().item(), rel=1e-6)
