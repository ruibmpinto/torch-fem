"""Gurson-Tvergaard-Needleman (GTN) porous-plasticity material.

This module implements the GTN porous-plasticity model with void
growth and strain-controlled nucleation for isotropic materials
under small strains. The yield surface couples the porosity into
the plastic response (void softening), in contrast to decoupled
kinematic void-growth estimates. The coalescence acceleration is
included through the standard effective-porosity mapping f*(f).
The q3 = q1**2 convention is adopted throughout.

Tensor convention: torch-fem native full tensors everywhere —
stress and strain increments of shape ``(..., 3, 3)``, elastic
stiffness and algorithmic tangent of shape ``(..., 3, 3, 3, 3)``.
No Voigt representation is used in this module.

The return mapping follows the bracketed-bisection scheme of the
explicit VUMAT reference implementation by Irfan Habeeb CN
(github.com/irfancn/Abaqus-VUMAT-Gurson_GTN): the scalar unknown is
the matrix equivalent plastic-strain increment ``ep``; the flow
direction is frozen at the elastic trial state; the bracket is
expanded multiplicatively until it contains the root and then
bisected until the yield function is below tolerance. Two explicit
deviations from the reference are made for implicit-solver use:

1. The bisection unknown is the tensor plastic multiplier
   ``dlambda`` along the frozen normal (the reference bisects the
   matrix strain increment ``ep`` directly). The matrix increment
   follows from the plastic-work equivalence with the UPDATED
   stress,
   ``(1 - f) sigma_y ep = (sigma_trial : n - (n : C : n) dlambda)
   dlambda``,
   solved per evaluation by a short fixed-point iteration. The
   yield value is then a continuous function of the unknown up to
   the radial-return cap
   ``dlambda <= (sigma_trial : n) / (n : C : n)``, so a root is
   bracketable even under strong porosity growth (bisecting ``ep``
   has no root there: the work parabola caps the absorbable matrix
   strain). In the dense limit ``ep`` is exactly the J2 equivalent
   plastic-strain increment (the reference divides the work by
   ``sigma_old : n``, valid only for the small increments of an
   explicit code and singular on the first increment).
2. The volumetric part of the flow normal uses the analytic
   derivative of the yield function,
   ``q1 q2 f* sinh(1.5 q2 sigma_h / sigma_y) / sigma_y`` per
   diagonal component (the reference carries an extra factor of 3;
   since the normal is renormalized, this changes the direction
   mix between the deviatoric and volumetric parts).

Nonconvergence of the bisection raises ``RuntimeError`` — the
local solve never exits silently. All GTN parameters are passed
explicitly through a validated dictionary; there are no defaults.

Functions
---------
validate_gtn_params
    Strict validation of the GTN parameter dictionary.
stress_invariants
    Hydrostatic stress, von Mises stress and deviator.
f_star
    Effective (coalescence-accelerated) porosity mapping.
gtn_yield
    GTN yield function with q3 = q1**2.
nucleation_intensity
    Strain-controlled Gaussian nucleation intensity.
gtn_flow_normal
    Unit flow normal from the trial state.
gtn_return_map
    Element-vectorized bracketed-bisection return mapping.

Classes
-------
GTN3D
    torch-fem Material wrapper around the return mapping.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math

# Third-party
import torch

# Local
from ..materials import IsotropicElasticity3D

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira Morais Pinto', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
def validate_gtn_params(gtn_params):
    """Validate the GTN parameter dictionary strictly.

    Every key is required and unknown keys are rejected: there are
    no silent defaults anywhere in this module.

    Parameters
    ----------
    gtn_params : dict
        Required keys: ``q1``, ``q2``, ``fc``, ``ff``, ``f_n``,
        ``eps_n``, ``s_n``, ``tol``, ``max_iter``.

    Returns
    -------
    gtn_params : dict
        The validated dictionary (returned unchanged).

    Raises
    ------
    ValueError
        If any required key is missing or any unknown key is
        present; the offending keys are named in the message.
    """
    # The exact required key set: nothing optional, nothing extra.
    required = {'q1', 'q2', 'fc', 'ff', 'f_n', 'eps_n', 's_n',
                'tol', 'max_iter'}
    provided = set(gtn_params.keys())
    # Missing keys are an error naming each absent parameter.
    missing = sorted(required - provided)
    if missing:
        raise ValueError(
            f'gtn_params is missing required keys: {missing}.')
    # Unknown keys are an error naming each unexpected parameter.
    unknown = sorted(provided - required)
    if unknown:
        raise ValueError(
            f'gtn_params contains unknown keys: {unknown}.')
    return gtn_params
# =============================================================================
def stress_invariants(sigma):
    """Hydrostatic stress, von Mises stress and deviator.

    Parameters
    ----------
    sigma : torch.Tensor
        Cauchy stress of shape ``(..., 3, 3)``.

    Returns
    -------
    sigma_h : torch.Tensor
        Hydrostatic stress ``tr(sigma)/3`` of shape ``(...,)``.
    sigma_eq : torch.Tensor
        Von Mises equivalent stress of shape ``(...,)``.
    s_dev : torch.Tensor
        Stress deviator of shape ``(..., 3, 3)``.
    """
    # Hydrostatic part from the trace.
    sigma_h = (sigma[..., 0, 0] + sigma[..., 1, 1]
               + sigma[..., 2, 2]) / 3.0
    # Deviator: subtract the hydrostatic part from the diagonal.
    identity = torch.eye(
        3, dtype=sigma.dtype, device=sigma.device)
    s_dev = sigma - sigma_h[..., None, None] * identity
    # Von Mises stress with an additive floor inside the square
    # root: deliberate regularization so the gradient is finite at
    # zero stress (not a hidden fallback).
    j2_floor = 1.0e-30
    s_norm_sq = (s_dev * s_dev).sum(dim=(-1, -2))
    sigma_eq = torch.sqrt(1.5 * s_norm_sq + j2_floor)
    return sigma_h, sigma_eq, s_dev
# =============================================================================
def f_star(f, q1, fc, ff):
    """Effective (coalescence-accelerated) porosity mapping.

    Identity below the critical porosity ``fc``; linear ramp from
    ``fc`` towards the ultimate value ``1/q1`` reached at the
    failure porosity ``ff``; clamped at ``1/q1`` beyond ``ff``.
    Mirrors subroutine ``fnfs`` of the VUMAT reference.

    Parameters
    ----------
    f : torch.Tensor
        Void volume fraction of shape ``(...,)``.
    q1 : float
        Tvergaard parameter.
    fc : float
        Critical porosity at coalescence onset.
    ff : float
        Porosity at final failure.

    Returns
    -------
    fs : torch.Tensor
        Effective porosity of shape ``(...,)``.
    """
    # Ultimate effective porosity at which the surface collapses.
    f_ultimate = 1.0 / q1
    # Linear coalescence ramp between fc and ff.
    ramp = fc + (f_ultimate - fc) * (f - fc) / (ff - fc)
    # Identity below fc, ramp above, clamped at the ultimate value.
    fs = torch.where(f <= fc, f, ramp)
    return torch.clamp(fs, max=f_ultimate)
# =============================================================================
def gtn_yield(sigma_eq, sigma_h, sigma_y, fs, q1, q2):
    """GTN yield function with the q3 = q1**2 convention.

    Phi = (sigma_eq/sigma_y)**2
          + 2 q1 fs cosh(1.5 q2 sigma_h / sigma_y)
          - (1 + (q1 fs)**2).

    Mirrors subroutine ``fngur`` of the VUMAT reference.

    Parameters
    ----------
    sigma_eq : torch.Tensor
        Von Mises equivalent stress of shape ``(...,)``.
    sigma_h : torch.Tensor
        Hydrostatic stress of shape ``(...,)``.
    sigma_y : torch.Tensor
        Matrix flow stress of shape ``(...,)``.
    fs : torch.Tensor
        Effective porosity of shape ``(...,)``.
    q1 : float
        Tvergaard parameter.
    q2 : float
        Tvergaard parameter.

    Returns
    -------
    phi : torch.Tensor
        Yield function value of shape ``(...,)``.
    """
    # Deviatoric, porosity and pressure contributions.
    ratio_sq = (sigma_eq / sigma_y) ** 2
    pressure = 2.0 * q1 * fs * torch.cosh(
        1.5 * q2 * sigma_h / sigma_y)
    return ratio_sq + pressure - (1.0 + (q1 * fs) ** 2)
# =============================================================================
def nucleation_intensity(peeq_m, f_n, eps_n, s_n):
    """Strain-controlled Gaussian nucleation intensity.

    A = f_n / (s_n sqrt(2 pi))
        * exp(-0.5 ((peeq_m - eps_n)/s_n)**2).

    The normalized argument is clamped to ``|arg| <= 10``: a
    deliberate overflow guard taken from the VUMAT reference (which
    clamps the upper side), extended symmetrically.

    Parameters
    ----------
    peeq_m : torch.Tensor
        Matrix equivalent plastic strain of shape ``(...,)``.
    f_n : float
        Volume fraction of void-nucleating particles.
    eps_n : float
        Mean nucleation strain.
    s_n : float
        Standard deviation of the nucleation strain.

    Returns
    -------
    intensity : torch.Tensor
        Nucleation intensity ``A`` of shape ``(...,)``.
    """
    # Normalized distance to the mean nucleation strain, clamped.
    arg = torch.clamp((peeq_m - eps_n) / s_n, min=-10.0, max=10.0)
    # Gaussian intensity with the standard normalization.
    amplitude = f_n / (s_n * math.sqrt(2.0 * math.pi))
    return amplitude * torch.exp(-0.5 * arg * arg)
# =============================================================================
def gtn_flow_normal(s_dev, sigma_h, sigma_y, fs, q1, q2):
    """Unit flow normal of the GTN potential at a given state.

    The unnormalized associative direction is the analytic yield
    gradient
    dPhi/dsigma = 3 s / sigma_y**2
                  + q1 q2 fs sinh(1.5 q2 sigma_h / sigma_y)
                    / sigma_y * I,
    subsequently normalized to unit Frobenius norm. Note: the VUMAT
    reference carries an extra factor of 3 on the volumetric term;
    the analytic derivative is used here (deliberate deviation, see
    module docstring).

    Parameters
    ----------
    s_dev : torch.Tensor
        Stress deviator of shape ``(..., 3, 3)``.
    sigma_h : torch.Tensor
        Hydrostatic stress of shape ``(...,)``.
    sigma_y : torch.Tensor
        Matrix flow stress of shape ``(...,)``.
    fs : torch.Tensor
        Effective porosity of shape ``(...,)``.
    q1 : float
        Tvergaard parameter.
    q2 : float
        Tvergaard parameter.

    Returns
    -------
    normal : torch.Tensor
        Unit flow normal of shape ``(..., 3, 3)``.
    """
    # Deviatoric part of the yield gradient.
    deviatoric = 3.0 * s_dev / (sigma_y ** 2)[..., None, None]
    # Volumetric part of the yield gradient (analytic derivative).
    volumetric = (q1 * q2 * fs * torch.sinh(
        1.5 * q2 * sigma_h / sigma_y) / sigma_y)
    identity = torch.eye(
        3, dtype=s_dev.dtype, device=s_dev.device)
    direction = deviatoric + volumetric[..., None, None] * identity
    # Unit normalization with an additive floor: deliberate
    # regularization against the zero-gradient corner case.
    norm_floor = 1.0e-30
    norm = torch.sqrt(
        (direction * direction).sum(dim=(-1, -2)) + norm_floor)
    return direction / norm[..., None, None]
# =============================================================================
def gtn_return_map(sigma_n, peeq_m_n, f_n, dlam_warm, de_mech, C,
                   flow_stress_fn, gtn_params):
    """Element-vectorized GTN return mapping (bracketed bisection).

    The scalar unknown per element is the tensor plastic multiplier
    ``dlambda`` along the flow normal frozen at the elastic trial
    state. For each candidate ``dlambda`` the matrix equivalent
    plastic-strain increment ``ep`` follows from the plastic-work
    equivalence with the updated stress (see module docstring), the
    porosity is updated by growth plus strain-controlled
    nucleation, and the yield function is re-evaluated; the
    bisection converges when ``|Phi| <= tol``.

    Parameters
    ----------
    sigma_n : torch.Tensor
        Stress at the start of the increment, shape ``(..., 3, 3)``.
    peeq_m_n : torch.Tensor
        Matrix equivalent plastic strain, shape ``(...,)``.
    f_n : torch.Tensor
        Void volume fraction, shape ``(...,)``.
    dlam_warm : torch.Tensor
        Warm-start value for ``dlambda`` (converged value of the
        previous increment; zeros disable), shape ``(...,)``.
    de_mech : torch.Tensor
        Mechanical strain increment, shape ``(..., 3, 3)``.
    C : torch.Tensor
        Elastic stiffness, shape ``(..., 3, 3, 3, 3)``.
    flow_stress_fn : callable
        Matrix flow stress as a function of the matrix equivalent
        plastic strain (pluggable hardening).
    gtn_params : dict
        Validated GTN parameters, see :func:`validate_gtn_params`.

    Returns
    -------
    out : dict
        - ``'sigma'`` : updated stress, shape ``(..., 3, 3)``.
        - ``'dpeeq_m'`` : matrix plastic-strain increment ``ep``,
          shape ``(...,)``.
        - ``'f'`` : updated void volume fraction, shape ``(...,)``.
        - ``'d_eps_p'`` : plastic strain increment tensor, shape
          ``(..., 3, 3)``.
        - ``'phi'`` : converged yield value on plastic elements and
          zero on elastic elements (constraint residual), shape
          ``(...,)``.
        - ``'dlam'`` : converged tensor plastic multiplier (warm
          start for the next increment), shape ``(...,)``.
        - ``'n_iter'`` : int, bisection iterations used.
        - ``'ddsdde'`` : algorithmic tangent, shape
          ``(..., 3, 3, 3, 3)`` (elastic stiffness on elastic
          elements; rank-one continuum approximation with secant
          hardening on plastic elements).

    Raises
    ------
    ValueError
        On an invalid parameter dictionary.
    RuntimeError
        If the bracket cannot contain the root within the strain
        cap or the bisection does not converge within
        ``max_iter`` iterations.
    """
    # Strict parameter validation: no silent defaults.
    validate_gtn_params(gtn_params)
    q1 = gtn_params['q1']
    q2 = gtn_params['q2']
    fc = gtn_params['fc']
    ff = gtn_params['ff']
    f_n_param = gtn_params['f_n']
    eps_n = gtn_params['eps_n']
    s_n = gtn_params['s_n']
    tol = gtn_params['tol']
    max_iter = gtn_params['max_iter']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Elastic trial state.
    sigma_trial = sigma_n + torch.einsum(
        '...ijkl,...kl->...ij', C, de_mech)
    sigma_h_t, sigma_eq_t, s_dev_t = stress_invariants(sigma_trial)
    # Start-of-increment flow stress and effective porosity.
    sigma_y_n = flow_stress_fn(peeq_m_n)
    fs_n = f_star(f_n, q1, fc, ff)
    # Trial yield function and plastic mask.
    phi_trial = gtn_yield(
        sigma_eq_t, sigma_h_t, sigma_y_n, fs_n, q1, q2)
    is_plastic = phi_trial > 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Elastic defaults for every output.
    sigma_new = sigma_trial.clone()
    dpeeq_m = torch.zeros_like(peeq_m_n)
    f_new = f_n.clone()
    d_eps_p = torch.zeros_like(sigma_trial)
    # Constraint residual: zero on elastic elements by definition.
    phi_out = torch.zeros_like(phi_trial)
    # Converged multiplier (warm start for the next increment).
    dlam_out = torch.zeros_like(phi_trial)
    ddsdde = C.clone()
    n_iter_used = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return mapping on the plastic subset.
    if bool(is_plastic.any()):
        pm = is_plastic
        # Plastic-subset views of the trial state.
        sig_t = sigma_trial[pm]
        peeq_p = peeq_m_n[pm]
        f_p = f_n[pm]
        C_p = C[pm]
        # Flow normal frozen at the trial state (reference scheme).
        normal = gtn_flow_normal(
            s_dev_t[pm], sigma_h_t[pm], sigma_y_n[pm], fs_n[pm],
            q1, q2)
        # Scalar contractions reused by every bisection evaluation:
        # b = sigma_trial : n and a = n : C : n.
        b_coef = (sig_t * normal).sum(dim=(-1, -2))
        c_normal = torch.einsum(
            '...ijkl,...kl->...ij', C_p, normal)
        a_coef = (normal * c_normal).sum(dim=(-1, -2))

        # A non-positive trial-stress projection on the flow normal
        # admits no radial return: refuse loudly.
        if bool((b_coef <= 0.0).any()):
            raise RuntimeError(
                'GTN return map: non-positive trial-stress '
                'projection on the flow normal.')
        # Radial-return cap, the tighter of two limits: (i) at
        # dlambda = b/a the updated stress loses its entire
        # projection on the flow normal; (ii) at the deviatoric
        # flip limit the deviatoric stress is driven through zero
        # (over-return), beyond which the frozen-normal update is
        # invalid.
        identity = torch.eye(
            3, dtype=normal.dtype, device=normal.device)
        trace_n_full = (normal[..., 0, 0] + normal[..., 1, 1]
                        + normal[..., 2, 2])
        n_dev = normal - (trace_n_full / 3.0)[..., None, None] \
            * identity
        cn_trace = (c_normal[..., 0, 0] + c_normal[..., 1, 1]
                    + c_normal[..., 2, 2])
        cn_dev = c_normal - (cn_trace / 3.0)[..., None, None] \
            * identity
        flip_num = (s_dev_t[pm] * n_dev).sum(dim=(-1, -2))
        flip_den = (n_dev * cn_dev).sum(dim=(-1, -2))
        # Purely volumetric flow has no deviatoric limit: fall back
        # to the projection cap (explicit choice, not a fallback of
        # convenience: the deviatoric constraint is empty there).
        lam_flip = torch.where(
            flip_den > 1.0e-30, flip_num / flip_den.clamp(
                min=1.0e-30), b_coef / a_coef)
        lam_cap = torch.minimum(b_coef / a_coef, lam_flip)

        def evaluate(dlam):
            # One bisection evaluation at the candidate tensor
            # multiplier dlam: returns the yield value and every
            # dependent quantity.
            #
            # Matrix plastic-strain increment from the plastic-work
            # equivalence with the UPDATED stress,
            #   (1 - f) sigma_y(peeq + ep) ep = (b - a dlam) dlam,
            # solved by a short fixed-point iteration on ep (three
            # passes: the contraction factor is H/sigma_y << 1;
            # deliberate fixed iteration count, not a hidden loop).
            work = (b_coef - a_coef * dlam) * dlam
            ep = work / ((1.0 - f_p) * flow_stress_fn(peeq_p))
            for _ in range(3):
                ep = work / ((1.0 - f_p)
                             * flow_stress_fn(peeq_p + ep))
            # Plastic work is non-negative on the radial path.
            ep = torch.clamp(ep, min=0.0)
            sigma_y = flow_stress_fn(peeq_p + ep)
            # Porosity update: growth from the volumetric plastic
            # flow plus strain-controlled nucleation; the total
            # increment is floored at zero as in the reference.
            trace_n = (normal[..., 0, 0] + normal[..., 1, 1]
                       + normal[..., 2, 2])
            growth = (1.0 - f_p) * dlam * trace_n
            nucleation = nucleation_intensity(
                peeq_p + ep, f_n_param, eps_n, s_n) * ep
            df = torch.clamp(growth + nucleation, min=0.0)
            f_upd = f_p + df
            # Updated stress along the frozen normal.
            sig_upd = sig_t - dlam[..., None, None] * c_normal
            sigma_h_u, sigma_eq_u, _ = stress_invariants(sig_upd)
            fs_upd = f_star(f_upd, q1, fc, ff)
            phi = gtn_yield(
                sigma_eq_u, sigma_h_u, sigma_y, fs_upd, q1, q2)
            return phi, ep, f_upd, sig_upd
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Bracket state: phi(0) = phi_trial > 0 anchors the lower
        # end; the upper end starts at the reference's 1e-5 or at
        # the warm start when one is supplied, both capped at the
        # radial-return limit.
        lam_lo = torch.zeros_like(peeq_p)
        warm = dlam_warm[pm]
        lam_hi = torch.where(
            warm > 0.0, warm,
            torch.full_like(warm, 1.0e-5))
        lam_hi = torch.minimum(lam_hi, lam_cap)
        # Converged-state buffers, frozen once an element converges.
        lam_conv = torch.zeros_like(peeq_p)
        ep_conv = torch.zeros_like(peeq_p)
        phi_conv = torch.zeros_like(peeq_p)
        done = torch.zeros_like(peeq_p, dtype=torch.bool)
        f_conv = f_p.clone()
        sig_conv = sig_t.clone()
        # Probe at the upper end first: expansion phase and warm
        # start are handled by the same loop.
        probe = lam_hi.clone()
        expanding = torch.ones_like(done)
        for iteration in range(max_iter):
            n_iter_used = iteration + 1
            phi_k, ep_k, f_k, sig_k = evaluate(probe)
            # Record newly converged elements and freeze them.
            newly = (~done) & (phi_k.abs() <= tol)
            lam_conv = torch.where(newly, probe, lam_conv)
            ep_conv = torch.where(newly, ep_k, ep_conv)
            phi_conv = torch.where(newly, phi_k, phi_conv)
            f_conv = torch.where(newly, f_k, f_conv)
            sig_conv = torch.where(
                newly[..., None, None], sig_k, sig_conv)
            done = done | newly
            if bool(done.all()):
                break
            # Bracket update by the sign of the yield function:
            # phi >= 0 means the root lies above the probe.
            root_above = phi_k >= 0.0
            # A probe pinned at the radial-return cap with phi > 0
            # has no root along the frozen normal: refuse loudly
            # rather than continue.
            stuck = (~done) & root_above & (probe >= lam_cap)
            if bool(stuck.any()):
                raise RuntimeError(
                    'GTN return map: no root along the frozen '
                    'flow normal within the radial-return cap; '
                    'reduce the increment size.')
            lam_lo = torch.where(
                (~done) & root_above, probe, lam_lo)
            lam_hi = torch.where(
                (~done) & (~root_above), probe, lam_hi)
            # Expansion phase: while the upper end still has
            # phi >= 0, expand it tenfold up to the radial-return
            # cap (reference scheme).
            expanding = expanding & root_above
            grown = torch.minimum(lam_hi * 10.0, lam_cap)
            lam_hi = torch.where(
                (~done) & expanding, grown, lam_hi)
            # Next probe: the expanded upper end while expanding,
            # the bracket midpoint once the root is contained.
            midpoint = 0.5 * (lam_lo + lam_hi)
            probe = torch.where(expanding, lam_hi, midpoint)
            probe = torch.where(done, lam_conv, probe)
        # Nonconvergence is an error, never a silent exit.
        if not bool(done.all()):
            n_open = int((~done).sum())
            raise RuntimeError(
                f'GTN bisection did not converge for {n_open} '
                f'element(s) within max_iter={max_iter} '
                f'(tol={tol}).')
        # A converged plastic state with zero matrix plastic strain
        # is the degenerate cavitation collapse (total stress
        # relaxation at zero dissipation): physically it signals
        # unstable void growth under a too-large high-triaxiality
        # increment. Refuse loudly.
        if bool((ep_conv <= 0.0).any()):
            n_bad = int((ep_conv <= 0.0).sum())
            raise RuntimeError(
                f'GTN return map: cavitation collapse (zero '
                f'plastic dissipation) on {n_bad} element(s); '
                'reduce the increment size.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Scatter the converged plastic results into the outputs.
        sigma_new[pm] = sig_conv
        dpeeq_m[pm] = ep_conv
        f_new[pm] = f_conv
        d_eps_p[pm] = lam_conv[..., None, None] * normal
        phi_out[pm] = phi_conv
        dlam_out[pm] = lam_conv
        # Rank-one continuum tangent approximation with secant
        # hardening from the converged bisection (documented
        # approximation; consistent tangent is the escalation
        # path).
        sigma_y_end = flow_stress_fn(peeq_p + ep_conv)
        h_secant = (sigma_y_end - sigma_y_n[pm]) / torch.clamp(
            ep_conv, min=1.0e-30)
        denom = a_coef + h_secant
        rank_one = torch.einsum(
            '...ij,...kl->...ijkl', c_normal, c_normal)
        ddsdde[pm] = C_p - rank_one / denom[..., None, None,
                                            None, None]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pack outputs.
    return {
        'sigma': sigma_new,
        'dpeeq_m': dpeeq_m,
        'f': f_new,
        'd_eps_p': d_eps_p,
        'phi': phi_out,
        'dlam': dlam_out,
        'n_iter': n_iter_used,
        'ddsdde': ddsdde,
    }
# =============================================================================
class GTN3D(IsotropicElasticity3D):
    """GTN porous-plasticity material under 3D small strains.

    torch-fem Material wrapper around :func:`gtn_return_map` with
    the native full-tensor interface. The internal state layout is
    ``[peeq_m, f, dlam_warm]``: matrix equivalent plastic strain,
    void volume fraction, and the warm-start multiplier for the
    next increment's bisection.

    All constructor arguments are required: there are no default
    tolerances or iteration limits.

    Attributes
    ----------
    E : torch.Tensor
        Young modulus.
    nu : torch.Tensor
        Poisson ratio.
    C : torch.Tensor
        Elastic stiffness of shape ``(..., 3, 3, 3, 3)``.
    n_state : int
        Number of internal state variables (here: 3).
    flow_stress_fn : callable
        Matrix flow stress as a function of the matrix equivalent
        plastic strain.
    gtn_params : dict
        Validated GTN parameter dictionary.

    Methods
    -------
    vectorize(self, n_elem)
        Return a vectorized copy of the material.
    step(self, H_inc, F, sigma, state, de0)
        Perform an incremental state update for a batch of
        elements.
    """
    def __init__(self, E, nu, flow_stress_fn, gtn_params):
        """Constructor.

        Parameters
        ----------
        E : {float, torch.Tensor}
            Young modulus.
        nu : {float, torch.Tensor}
            Poisson ratio.
        flow_stress_fn : callable
            Matrix flow stress as a function of the matrix
            equivalent plastic strain.
        gtn_params : dict
            GTN parameters, validated strictly (see
            :func:`validate_gtn_params`).
        """
        super().__init__(E, nu)
        # Validate at construction so misconfiguration fails early.
        self.gtn_params = validate_gtn_params(gtn_params)
        self.flow_stress_fn = flow_stress_fn
        # State layout: [peeq_m, f, dlam_warm].
        self.n_state = 3
    # -------------------------------------------------------------------------
    def vectorize(self, n_elem):
        """Return a vectorized copy of the material.

        Parameters
        ----------
        n_elem : int
            Number of elements to vectorize the material for.

        Returns
        -------
        material : GTN3D
            New vectorized material instance.

        Raises
        ------
        RuntimeError
            If the material is already vectorized (never a silent
            print-and-return).
        """
        # Double vectorization is a caller error: refuse loudly.
        if self.is_vectorized:
            raise RuntimeError(
                'GTN3D material is already vectorized.')
        E = self.E.repeat(n_elem)
        nu = self.nu.repeat(n_elem)
        return GTN3D(E, nu, self.flow_stress_fn, self.gtn_params)
    # -------------------------------------------------------------------------
    def step(self, H_inc, F, sigma, state, de0):
        """Perform a strain increment with the GTN model.

        Parameters
        ----------
        H_inc : torch.Tensor
            Incremental displacement gradient of shape
            ``(..., 3, 3)``.
        F : torch.Tensor
            Deformation gradient of shape ``(..., 3, 3)``
            (unused under small strains; part of the torch-fem
            Material interface).
        sigma : torch.Tensor
            Stress of shape ``(..., 3, 3)``.
        state : torch.Tensor
            Internal state ``[peeq_m, f, dlam_warm]`` of shape
            ``(..., 3)``.
        de0 : torch.Tensor
            External (e.g. thermal) strain increment of shape
            ``(..., 3, 3)``.

        Returns
        -------
        sigma_new : torch.Tensor
            Updated stress of shape ``(..., 3, 3)``.
        state_new : torch.Tensor
            Updated internal state of shape ``(..., 3)``.
        ddsdde : torch.Tensor
            Algorithmic tangent of shape ``(..., 3, 3, 3, 3)``.
        """
        # Small-strain increment from the displacement gradient.
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        # Mechanical part: total minus external (thermal) strain.
        out = gtn_return_map(
            sigma, state[..., 0], state[..., 1], state[..., 2],
            de - de0, self.C, self.flow_stress_fn,
            self.gtn_params)
        # Updated state: accumulated matrix plastic strain, void
        # fraction, and the warm start for the next increment.
        state_new = state.clone()
        state_new[..., 0] = state[..., 0] + out['dpeeq_m']
        state_new[..., 1] = out['f']
        state_new[..., 2] = out['dlam']
        return out['sigma'], state_new, out['ddsdde']
# =============================================================================
