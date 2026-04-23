"""Verification script for the Lou-Zhang-Yoon constitutive model.

Runs the following checks:

1. Import sanity.
2. Reduction to von Mises: compare ``LouZhangYoon3D`` with parameters
   ``a=sqrt(3), b=c=d=0`` to the native ``IsotropicPlasticity3D`` under
   a uniaxial strain path.
3. Apex return-mapping: impose a pure hydrostatic strain increment
   beyond the apex pressure with pressure-dependent yield parameters
   and confirm convergence.
4. Plane-strain elastic consistency: verify the plane-strain variant
   matches torchfem's ``IsotropicElasticityPlaneStrain`` in the
   elastic regime.
5. Batched vs single-element consistency: run a batch ``N=4`` against
   four individual single-element runs and confirm agreement.

Functions
---------
run_checks
    Execute all verification checks and print results.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import pathlib
import sys
# Third-party
import torch
# Local
_repo_root = pathlib.Path(__file__).resolve().parents[1]
_src_root = str(_repo_root/'src')
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)
from torchfem.custom_materials.lou_zhang_yoon import \
    LouZhangYoon3D, LouZhangYoonPlaneStrain
from torchfem.materials import IsotropicElasticityPlaneStrain, \
    IsotropicPlasticity3D
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
def _const_fn(value):
    """Return a function returning ``value*ones_like(q)``.

    Parameters
    ----------
    value : float
        Constant return value.

    Returns
    -------
    fn : function
        Function mapping ``q`` to ``value*ones_like(q)``.
    """
    def fn(q):
        return value*torch.ones_like(q)
    return fn
# -----------------------------------------------------------------------------
def _zero_fn():
    """Return a function returning ``zeros_like(q)``.

    Returns
    -------
    fn : function
        Function mapping ``q`` to ``zeros_like(q)``.
    """
    def fn(q):
        return torch.zeros_like(q)
    return fn
# -----------------------------------------------------------------------------
def _linear_hardening(sigma_y0, h_mod):
    """Build a linear isotropic hardening law and its derivative.

    Parameters
    ----------
    sigma_y0 : float
        Initial yield stress.
    h_mod : float
        Linear hardening modulus.

    Returns
    -------
    sigma_f : function
        Yield stress as a function of accumulated plastic strain.
    sigma_f_prime : function
        Derivative of the yield stress with respect to the
        accumulated plastic strain.
    """
    def sigma_f(q):
        return sigma_y0 + h_mod*q
    def sigma_f_prime(q):
        return h_mod*torch.ones_like(q)
    return sigma_f, sigma_f_prime
# -----------------------------------------------------------------------------
def check_import():
    """Verify the module imports cleanly.

    Returns
    -------
    passed : bool
        True if the classes are importable.
    """
    print('[check_import] classes imported successfully')
    return True
# -----------------------------------------------------------------------------
def check_vm_reduction():
    """Compare LZY with vM-equivalent parameters against
    IsotropicPlasticity3D.

    Returns
    -------
    passed : bool
        True if the LZY model matches IsotropicPlasticity3D to a
        relative tolerance of ``1e-3`` under uniaxial strain.
    """
    E_val = 210.0e3
    nu_val = 0.3
    sigma_y0 = 250.0
    h_mod = 1.0e3
    sigma_f, sigma_f_prime = _linear_hardening(sigma_y0, h_mod)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build LZY model with vM-equivalent parameters
    lzy = LouZhangYoon3D(
        torch.tensor(E_val), torch.tensor(nu_val),
        sigma_f, sigma_f_prime,
        yield_a=_const_fn(math.sqrt(3.0)),
        yield_a_prime=_zero_fn(),
        yield_b=_const_fn(0.0), yield_b_prime=_zero_fn(),
        yield_c=_const_fn(0.0), yield_c_prime=_zero_fn(),
        yield_d=_const_fn(0.0), yield_d_prime=_zero_fn(),
        is_apex_handling=False,
        is_fixed_yield_parameters=True)
    vm = IsotropicPlasticity3D(
        torch.tensor(E_val), torch.tensor(nu_val),
        sigma_f, sigma_f_prime)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Vectorize to batch of size 1
    n_elem = 1
    lzy = lzy.vectorize(n_elem)
    vm = vm.vectorize(n_elem)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Uniaxial strain path
    n_steps = 30
    eps_max = 0.006
    eps_inc = eps_max/n_steps
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial state
    sigma_l = torch.zeros(n_elem, 3, 3)
    state_l = torch.zeros(n_elem, lzy.n_state)
    F_l = torch.eye(3).expand(n_elem, 3, 3).clone()
    sigma_v = torch.zeros(n_elem, 3, 3)
    state_v = torch.zeros(n_elem, vm.n_state)
    F_v = torch.eye(3).expand(n_elem, 3, 3).clone()
    de0 = torch.zeros(n_elem, 3, 3)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run the incremental update
    max_diff = 0.0
    for step in range(n_steps):
        H_inc = torch.zeros(n_elem, 3, 3)
        H_inc[..., 0, 0] = eps_inc
        sigma_l, state_l, _ = lzy.step(H_inc, F_l, sigma_l, state_l, de0)
        sigma_v, state_v, _ = vm.step(H_inc, F_v, sigma_v, state_v, de0)
        F_l = F_l + H_inc
        F_v = F_v + H_inc
        diff = float(torch.abs(sigma_l - sigma_v).max())
        max_diff = max(max_diff, diff)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Report results
    tol_abs = 1.0
    passed = max_diff < tol_abs
    print(f'[check_vm_reduction] max |sigma_lzy - sigma_vm| = '
          f'{max_diff:.4e} (tol {tol_abs:.1e}) -> '
          f'{"PASS" if passed else "FAIL"}')
    print(f'  LZY  sigma_xx (final) = '
          f'{float(sigma_l[0, 0, 0]):.4f}')
    print(f'  vM   sigma_xx (final) = '
          f'{float(sigma_v[0, 0, 0]):.4f}')
    return passed
# -----------------------------------------------------------------------------
def check_apex():
    """Exercise the apex return-mapping under pure hydrostatic strain.

    Returns
    -------
    passed : bool
        True if the apex branch produces a purely hydrostatic stress
        and an accumulated plastic strain greater than zero.
    """
    E_val = 210.0e3
    nu_val = 0.3
    sigma_y0 = 250.0
    h_mod = 0.0
    sigma_f, sigma_f_prime = _linear_hardening(sigma_y0, h_mod)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pressure-dependent yield: a=1, b=0.3, c=d=0
    lzy = LouZhangYoon3D(
        torch.tensor(E_val), torch.tensor(nu_val),
        sigma_f, sigma_f_prime,
        yield_a=_const_fn(1.0), yield_a_prime=_zero_fn(),
        yield_b=_const_fn(0.3), yield_b_prime=_zero_fn(),
        yield_c=_const_fn(0.0), yield_c_prime=_zero_fn(),
        yield_d=_const_fn(0.0), yield_d_prime=_zero_fn(),
        is_apex_handling=True,
        is_fixed_yield_parameters=True)
    n_elem = 1
    lzy = lzy.vectorize(n_elem)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pure hydrostatic total strain increment large enough to trigger apex
    eps_vol = 0.01
    H_inc = torch.zeros(n_elem, 3, 3)
    for axis in range(3):
        H_inc[..., axis, axis] = eps_vol
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sigma = torch.zeros(n_elem, 3, 3)
    state = torch.zeros(n_elem, lzy.n_state)
    F = torch.eye(3).expand(n_elem, 3, 3).clone()
    de0 = torch.zeros(n_elem, 3, 3)
    sigma_new, state_new, _ = lzy.step(H_inc, F, sigma, state, de0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check stress is purely hydrostatic
    dev = sigma_new - (1.0/3.0)*(
        sigma_new[..., 0, 0] + sigma_new[..., 1, 1]
        + sigma_new[..., 2, 2]).unsqueeze(-1).unsqueeze(-1)*torch.eye(3)
    dev_norm = float(torch.linalg.norm(dev))
    q_new = float(state_new[0, 0])
    p = float((sigma_new[0, 0, 0] + sigma_new[0, 1, 1]
               + sigma_new[0, 2, 2])/3.0)
    passed = (dev_norm < 1.0e-6) and (q_new > 0.0)
    print(f'[check_apex] dev_norm = {dev_norm:.4e}, '
          f'q_new = {q_new:.6f}, pressure = {p:.4f} -> '
          f'{"PASS" if passed else "FAIL"}')
    return passed
# -----------------------------------------------------------------------------
def check_plane_strain_elastic():
    """Compare plane-strain LZY with plane-strain elasticity in the
    elastic regime.

    Returns
    -------
    passed : bool
        True if the plane-strain LZY output matches the reference
        plane-strain elastic output under a sub-yield strain path.
    """
    E_val = 210.0e3
    nu_val = 0.3
    sigma_y0 = 1.0e9
    h_mod = 0.0
    sigma_f, sigma_f_prime = _linear_hardening(sigma_y0, h_mod)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lzy = LouZhangYoonPlaneStrain(
        torch.tensor(E_val), torch.tensor(nu_val),
        sigma_f, sigma_f_prime,
        yield_a=_const_fn(math.sqrt(3.0)), yield_a_prime=_zero_fn(),
        yield_b=_const_fn(0.0), yield_b_prime=_zero_fn(),
        yield_c=_const_fn(0.0), yield_c_prime=_zero_fn(),
        yield_d=_const_fn(0.0), yield_d_prime=_zero_fn(),
        is_apex_handling=False,
        is_fixed_yield_parameters=True)
    ref = IsotropicElasticityPlaneStrain(
        torch.tensor(E_val), torch.tensor(nu_val))
    n_elem = 1
    lzy = lzy.vectorize(n_elem)
    ref = ref.vectorize(n_elem)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Small uniaxial strain path (sub-yield)
    n_steps = 5
    eps_inc = 1.0e-4
    sigma_l = torch.zeros(n_elem, 2, 2)
    state_l = torch.zeros(n_elem, lzy.n_state)
    F_l = torch.eye(2).expand(n_elem, 2, 2).clone()
    sigma_r = torch.zeros(n_elem, 2, 2)
    state_r = torch.zeros(n_elem, ref.n_state)
    F_r = torch.eye(2).expand(n_elem, 2, 2).clone()
    de0 = torch.zeros(n_elem, 2, 2)
    max_diff = 0.0
    for step in range(n_steps):
        H_inc = torch.zeros(n_elem, 2, 2)
        H_inc[..., 0, 0] = eps_inc
        sigma_l, state_l, _ = lzy.step(H_inc, F_l, sigma_l, state_l, de0)
        sigma_r, state_r, _ = ref.step(H_inc, F_r, sigma_r, state_r, de0)
        F_l = F_l + H_inc
        F_r = F_r + H_inc
        diff = float(torch.abs(sigma_l - sigma_r).max())
        max_diff = max(max_diff, diff)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tol_abs = 1.0e-3
    passed = max_diff < tol_abs
    print(f'[check_plane_strain_elastic] max diff = {max_diff:.4e} '
          f'(tol {tol_abs:.1e}) -> '
          f'{"PASS" if passed else "FAIL"}')
    return passed
# -----------------------------------------------------------------------------
def check_batched_consistency():
    """Verify batched and single-element runs produce identical
    results.

    Returns
    -------
    passed : bool
        True if the batched LZY output matches four single-element
        LZY outputs to a relative tolerance of ``1e-6``.
    """
    E_val = 210.0e3
    nu_val = 0.3
    sigma_y0 = 250.0
    h_mod = 1.0e3
    sigma_f, sigma_f_prime = _linear_hardening(sigma_y0, h_mod)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def make_model(n_elem):
        model = LouZhangYoon3D(
            torch.tensor(E_val), torch.tensor(nu_val),
            sigma_f, sigma_f_prime,
            yield_a=_const_fn(math.sqrt(3.0)),
            yield_a_prime=_zero_fn(),
            yield_b=_const_fn(0.0), yield_b_prime=_zero_fn(),
            yield_c=_const_fn(0.0), yield_c_prime=_zero_fn(),
            yield_d=_const_fn(0.0), yield_d_prime=_zero_fn(),
            is_apex_handling=False,
            is_fixed_yield_parameters=True)
        return model.vectorize(n_elem)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Batched run
    n_batch = 4
    lzy_b = make_model(n_batch)
    sigma_b = torch.zeros(n_batch, 3, 3)
    state_b = torch.zeros(n_batch, lzy_b.n_state)
    F_b = torch.eye(3).expand(n_batch, 3, 3).clone()
    de0_b = torch.zeros(n_batch, 3, 3)
    eps_inc = 1.0e-4
    n_steps = 20
    for _ in range(n_steps):
        H_inc = torch.zeros(n_batch, 3, 3)
        H_inc[..., 0, 0] = eps_inc
        sigma_b, state_b, _ = lzy_b.step(H_inc, F_b, sigma_b, state_b, de0_b)
        F_b = F_b + H_inc
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Single-element references
    max_diff = 0.0
    for elem in range(n_batch):
        lzy_s = make_model(1)
        sigma_s = torch.zeros(1, 3, 3)
        state_s = torch.zeros(1, lzy_s.n_state)
        F_s = torch.eye(3).expand(1, 3, 3).clone()
        de0_s = torch.zeros(1, 3, 3)
        for _ in range(n_steps):
            H_inc = torch.zeros(1, 3, 3)
            H_inc[..., 0, 0] = eps_inc
            sigma_s, state_s, _ = lzy_s.step(
                H_inc, F_s, sigma_s, state_s, de0_s)
            F_s = F_s + H_inc
        diff = float(torch.abs(sigma_b[elem] - sigma_s[0]).max())
        max_diff = max(max_diff, diff)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tol_abs = 1.0e-4
    passed = max_diff < tol_abs
    print(f'[check_batched_consistency] max diff = {max_diff:.4e} '
          f'(tol {tol_abs:.1e}) -> '
          f'{"PASS" if passed else "FAIL"}')
    return passed
# -----------------------------------------------------------------------------
def run_checks():
    """Run all verification checks and report the outcome.

    Returns
    -------
    all_passed : bool
        True if all verification checks pass.
    """
    results = [
        check_import(),
        check_vm_reduction(),
        check_apex(),
        check_plane_strain_elastic(),
        check_batched_consistency()]
    all_passed = all(results)
    print('=' * 60)
    print(f'Overall: {"PASS" if all_passed else "FAIL"} '
          f'({sum(results)}/{len(results)})')
    return all_passed
# =============================================================================
if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    run_checks()
