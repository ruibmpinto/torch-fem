"""Lou-Zhang-Yoon elasto-plastic constitutive model.

This module implements the Lou-Zhang-Yoon (LZY) pressure-dependent
elasto-plastic constitutive model with a general differentiable yield
function defined in terms of the stress invariants and four yield
parameters (a, b, c, d) that may evolve with the accumulated plastic
strain. Isotropic strain hardening is adopted. The apex singularity is
handled by a purely volumetric return-mapping along the hydrostatic
axis.

The infinitesimal strain formulation is assumed. The model exposes the
torchfem material interface ``step(H_inc, F, sigma, state, de0)``
returning ``(sigma_new, state_new, ddsdde)`` with full-tensor shapes
compatible with the batched torchfem solver.

Classes
-------
LouZhangYoon3D
    Lou-Zhang-Yoon constitutive model under 3D small strains.
LouZhangYoonPlaneStrain
    Lou-Zhang-Yoon constitutive model under 2D plane-strain small
    strains.

Notes
-----
The cone-surface return-mapping is formulated as a batched
Newton-Raphson (NR) problem on eight unknowns per yielded element:
the six Kelvin components of the elastic strain, the accumulated
plastic strain, and the incremental plastic multiplier. The apex
return-mapping is a scalar-per-element NR problem on the incremental
volumetric plastic strain.

The algorithmic consistent tangent modulus is computed by implicit
differentiation of the converged NR residual: with
``R(x, e_trial) = 0`` at ``x*``, the sensitivity reads
``dx*/d(e_trial) = J^{-1} @ [I_6; 0; 0]`` because only the elastic-
strain residual depends on ``e_trial`` (with partial ``-I_6``).
Contracting with the elastic stiffness yields the cone tangent in
Kelvin form ``C_ep_k = C_k @ (J^{-1})[:6, :6]``, converted back to
the minor-symmetric fourth-order tensor. On the apex branch the
tangent is purely volumetric
``ddsdde = c_apex * (I otimes I)`` with
``c_apex = K * (J_apex - K)/J_apex`` derived from implicit diff of
the scalar apex residual.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import torch
# Local
from ..materials import IsotropicElasticity3D, \
    IsotropicElasticityPlaneStrain
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
class LouZhangYoon3D(IsotropicElasticity3D):
    """Lou-Zhang-Yoon constitutive model under 3D small strains.

    Pressure-dependent elasto-plastic material model with a general
    differentiable yield function defined in terms of stress
    invariants and four yield parameters that may evolve with the
    accumulated plastic strain. Isotropic strain hardening is
    adopted. The apex singularity of the yield surface is optionally
    handled through a purely volumetric return-mapping.

    Attributes
    ----------
    E : torch.Tensor
        Young modulus.
    nu : torch.Tensor
        Poisson ratio.
    lbd : torch.Tensor
        First Lamé parameter.
    G : torch.Tensor
        Shear modulus (second Lamé parameter).
    K : torch.Tensor
        Bulk modulus.
    C : torch.Tensor
        Fourth-order elastic stiffness tensor stored with shape
        ``(..., 3, 3, 3, 3)``.
    n_state : int
        Number of internal state variables (here: 1).
    is_vectorized : bool
        If True, material properties carry a batch dimension.
    sigma_f : function
        Function returning the yield stress as a function of the
        accumulated plastic strain.
    sigma_f_prime : function
        Function returning the derivative of the yield stress with
        respect to the accumulated plastic strain.
    yield_a, yield_b, yield_c, yield_d : function
        Yield parameter evolution laws, each a function of the
        accumulated plastic strain.
    yield_a_prime, yield_b_prime, yield_c_prime, yield_d_prime : \
function
        Derivatives of the yield parameter evolution laws with
        respect to the accumulated plastic strain.
    is_associative_hardening : bool
        If True, then associative hardening is assumed (admissible
        only for constant yield parameters).
    is_fixed_yield_parameters : bool
        If True, yield parameter derivatives with respect to the
        accumulated plastic strain are skipped (performance
        optimization).
    is_apex_handling : bool
        If True, the apex singularity is handled by a volumetric
        return-mapping. If False, apex handling is disabled.
    apex_switch_tol : float
        Tolerance of the criterion to switch from the cone surface
        return-mapping to the apex return-mapping.
    tolerance : float
        Convergence tolerance for the local return-mapping NR
        iterative procedure.
    max_iter : int
        Maximum number of iterations for the local return-mapping
        NR iterative procedure.

    Methods
    -------
    vectorize(self, n_elem)
        Return a vectorized copy of the material.
    step(self, H_inc, F, sigma, state, de0)
        Perform an incremental state update for a batch of elements.
    """
    def __init__(self, E, nu, sigma_f, sigma_f_prime,
                 yield_a, yield_a_prime, yield_b, yield_b_prime,
                 yield_c, yield_c_prime, yield_d, yield_d_prime,
                 is_associative_hardening=False,
                 is_fixed_yield_parameters=True,
                 is_apex_handling=True, apex_switch_tol=0.05,
                 tolerance=1e-6, max_iter=10):
        """Constitutive model constructor.

        Parameters
        ----------
        E : {float, torch.Tensor}
            Young modulus.
        nu : {float, torch.Tensor}
            Poisson ratio.
        sigma_f : function
            Yield stress as a function of the accumulated plastic
            strain.
        sigma_f_prime : function
            Derivative of the yield stress with respect to the
            accumulated plastic strain.
        yield_a : function
            Yield parameter ``a`` as a function of the accumulated
            plastic strain.
        yield_a_prime : function
            Derivative of the yield parameter ``a`` with respect to
            the accumulated plastic strain.
        yield_b : function
            Yield parameter ``b`` as a function of the accumulated
            plastic strain.
        yield_b_prime : function
            Derivative of the yield parameter ``b`` with respect to
            the accumulated plastic strain.
        yield_c : function
            Yield parameter ``c`` as a function of the accumulated
            plastic strain.
        yield_c_prime : function
            Derivative of the yield parameter ``c`` with respect to
            the accumulated plastic strain.
        yield_d : function
            Yield parameter ``d`` as a function of the accumulated
            plastic strain.
        yield_d_prime : function
            Derivative of the yield parameter ``d`` with respect to
            the accumulated plastic strain.
        is_associative_hardening : bool, default=False
            If True, adopt the associative hardening rule (only
            admissible when yield parameters are constant).
        is_fixed_yield_parameters : bool, default=True
            If True, yield parameter derivatives with respect to the
            accumulated plastic strain are skipped.
        is_apex_handling : bool, default=True
            If True, handle the apex singularity through a
            volumetric return-mapping.
        apex_switch_tol : float, default=0.05
            Tolerance of the apex switch criterion.
        tolerance : float, default=1e-6
            Convergence tolerance for the local NR procedure.
        max_iter : int, default=10
            Maximum number of iterations for the local NR procedure.
        """
        # Initialize elastic attributes
        super().__init__(E, nu)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store hardening laws
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.yield_a = yield_a
        self.yield_a_prime = yield_a_prime
        self.yield_b = yield_b
        self.yield_b_prime = yield_b_prime
        self.yield_c = yield_c
        self.yield_c_prime = yield_c_prime
        self.yield_d = yield_d
        self.yield_d_prime = yield_d_prime
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store algorithm flags and tolerances
        self.is_associative_hardening = is_associative_hardening
        self.is_fixed_yield_parameters = is_fixed_yield_parameters
        self.is_apex_handling = is_apex_handling
        self.apex_switch_tol = apex_switch_tol
        self.tolerance = tolerance
        self.max_iter = max_iter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Single state variable: accumulated plastic strain
        self.n_state = 1
        # Bulk modulus (used by apex return-mapping)
        self.K = self.E/(3.0*(1.0 - 2.0*self.nu))
    # -------------------------------------------------------------------------
    def vectorize(self, n_elem):
        """Return a vectorized copy of the material for ``n_elem``.

        Parameters
        ----------
        n_elem : int
            Number of elements to vectorize the material for.

        Returns
        -------
        material : LouZhangYoon3D
            Vectorized copy of the material.
        """
        if self.is_vectorized:
            print('Material is already vectorized.')
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            return LouZhangYoon3D(
                E, nu, self.sigma_f, self.sigma_f_prime,
                self.yield_a, self.yield_a_prime,
                self.yield_b, self.yield_b_prime,
                self.yield_c, self.yield_c_prime,
                self.yield_d, self.yield_d_prime,
                is_associative_hardening=self.is_associative_hardening,
                is_fixed_yield_parameters=self.is_fixed_yield_parameters,
                is_apex_handling=self.is_apex_handling,
                apex_switch_tol=self.apex_switch_tol,
                tolerance=self.tolerance, max_iter=self.max_iter)
    # -------------------------------------------------------------------------
    def step(self, H_inc, F, sigma, state, de0):
        """Perform an incremental state update in 3D small strains.

        Parameters
        ----------
        H_inc : torch.Tensor
            Incremental displacement gradient with shape
            ``(..., 3, 3)``.
        F : torch.Tensor
            Current deformation gradient with shape ``(..., 3, 3)``
            (unused).
        sigma : torch.Tensor
            Current Cauchy stress tensor with shape ``(..., 3, 3)``.
        state : torch.Tensor
            Internal state variables with shape ``(..., 1)``, storing
            the accumulated plastic strain.
        de0 : torch.Tensor
            External small strain increment with shape
            ``(..., 3, 3)``.

        Returns
        -------
        sigma_new : torch.Tensor
            Updated Cauchy stress tensor with shape ``(..., 3, 3)``.
        state_new : torch.Tensor
            Updated internal state with shape ``(..., 1)``.
        ddsdde : torch.Tensor
            Algorithmic tangent stiffness tensor with shape
            ``(..., 3, 3, 3, 3)``.
        """
        # Compute small strain increment
        de = 0.5*(H_inc.transpose(-1, -2) + H_inc) - de0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Delegate to shared 3D core (returns consistent tangent)
        q_old = state[..., 0]
        stress_new, q_new, _, ddsdde = self._step_3d_core(
            sigma, de, q_old, self.C)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update state variables
        state_new = state.clone()
        state_new[..., 0] = q_new
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_new, state_new, ddsdde
    # -------------------------------------------------------------------------
    def _step_3d_core(self, sigma, de, q_old, C):
        """Perform the core 3D state update shared by all variants.

        Parameters
        ----------
        sigma : torch.Tensor
            Current Cauchy stress tensor with shape ``(..., 3, 3)``.
        de : torch.Tensor
            Small strain increment with shape ``(..., 3, 3)``.
        q_old : torch.Tensor
            Last converged accumulated plastic strain with shape
            ``(...,)``.
        C : torch.Tensor
            Elastic consistent tangent modulus with shape
            ``(..., 3, 3, 3, 3)``.

        Returns
        -------
        stress_new : torch.Tensor
            Updated Cauchy stress tensor with shape ``(..., 3, 3)``.
        q_new : torch.Tensor
            Updated accumulated plastic strain with shape ``(...,)``.
        is_plastic : torch.Tensor
            Boolean mask of yielded elements with shape ``(...,)``.
        ddsdde : torch.Tensor
            Consistent algorithmic tangent modulus with shape
            ``(..., 3, 3, 3, 3)``. Elastic ``C`` on elastic-only
            elements; cone analytical tangent (implicit diff of
            NR) on cone-yielded elements; purely volumetric
            tangent on apex-yielded elements.
        """
        device = sigma.device
        dtype = sigma.dtype
        # Compute trial stress
        sigma_trial = sigma + torch.einsum('...ijkl,...kl->...ij', C, de)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute trial yield parameters and yield stress
        yield_a = self.yield_a(q_old)
        yield_b = self.yield_b(q_old)
        yield_c = self.yield_c(q_old)
        yield_d = self.yield_d(q_old)
        yield_stress = self.sigma_f(q_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute trial stress invariants
        i1_trial, j2_trial, j3_trial = self._invariants(sigma_trial)
        # Compute trial effective stress
        eff_trial = self._effective_stress(
            i1_trial, j2_trial, j3_trial,
            yield_a, yield_b, yield_c, yield_d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check yield function
        yield_function = eff_trial - yield_stress
        is_plastic = yield_function > 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize outputs with elastic trial state
        stress_new = sigma_trial.clone()
        q_new = q_old.clone()
        # Elastic consistent tangent for all elements (will be
        # overwritten on yielded subsets below).
        C_batch = C.expand(sigma.shape[:-2] + C.shape[-4:])
        ddsdde = C_batch.clone()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Early return if no yielded elements
        if not bool(is_plastic.any()):
            return stress_new, q_new, is_plastic, ddsdde
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Partition yielded elements into apex and cone subsets
        p_trial = i1_trial/3.0
        safe_b = torch.where(
            torch.abs(yield_b) < 1e-6,
            torch.tensor(1e-6, device=sigma.device, dtype=sigma.dtype),
            torch.abs(yield_b))
        p_apex = yield_stress/(3.0*yield_a*safe_b)
        if self.is_apex_handling:
            is_apex = is_plastic & (p_trial
                                    > (1.0 - self.apex_switch_tol)*p_apex)
        else:
            is_apex = torch.zeros_like(is_plastic)
        is_cone = is_plastic & torch.logical_not(is_apex)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic trial strain for yielded entries (shared)
        e_trial_strain = self._compute_elastic_trial_strain(
            sigma_trial, C_batch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cone return-mapping (8-equation NR)
        if bool(is_cone.any()):
            cone_stress, cone_q, cone_jac = (
                self._return_mapping_cone(
                    e_trial_strain[is_cone],
                    q_old[is_cone],
                    C_batch[is_cone]))
            stress_new[is_cone] = cone_stress
            q_new[is_cone] = cone_q
            # Consistent cone tangent via implicit diff
            ddsdde[is_cone] = (
                LouZhangYoon3D._build_cone_tangent(
                    cone_jac, C_batch[is_cone]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apex return-mapping (scalar NR)
        if self.is_apex_handling and bool(is_apex.any()):
            apex_stress, apex_q, apex_c = (
                self._return_mapping_apex(
                    p_trial[is_apex],
                    q_old[is_apex],
                    self.K_scalar_for(is_apex)))
            stress_new[is_apex] = apex_stress
            q_new[is_apex] = apex_q
            # Apex tangent: c_apex * (I otimes I)
            soid = torch.eye(3, device=device, dtype=dtype)
            i_dyad_i = torch.einsum(
                'ij,kl->ijkl', soid, soid)
            ddsdde[is_apex] = (
                apex_c
                .unsqueeze(-1).unsqueeze(-1)
                .unsqueeze(-1).unsqueeze(-1)
                * i_dyad_i)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_new, q_new, is_plastic, ddsdde
    # -------------------------------------------------------------------------
    def K_scalar_for(self, mask):
        """Return bulk modulus tensor broadcast to the masked subset.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask selecting the elements of interest.

        Returns
        -------
        bulk_modulus : torch.Tensor
            Bulk modulus evaluated at the masked elements.
        """
        if self.K.dim() == 0:
            n_sel = int(mask.sum().item())
            return self.K.expand(n_sel)
        else:
            return self.K[mask]
    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_elastic_trial_strain(sigma_trial, C):
        """Recover elastic trial strain from trial stress.

        The elastic trial strain is obtained by inverting the elastic
        relation ``sigma_trial = C : eps_e`` over the symmetric
        subspace using the Kelvin notation representation of ``C``.

        Parameters
        ----------
        sigma_trial : torch.Tensor
            Trial stress tensor with shape ``(..., 3, 3)``.
        C : torch.Tensor
            Elastic stiffness tensor with shape ``(..., 3, 3, 3, 3)``.

        Returns
        -------
        e_trial_strain : torch.Tensor
            Elastic trial strain tensor with shape ``(..., 3, 3)``.
        """
        # Convert to Kelvin notation (symmetric 6-vector representation)
        sigma_k = LouZhangYoon3D._sym_to_kelvin(sigma_trial)
        C_k = LouZhangYoon3D._fourth_to_kelvin(C)
        # Solve in Kelvin space: C_k @ e_k = sigma_k
        e_k = torch.linalg.solve(C_k, sigma_k.unsqueeze(-1)).squeeze(-1)
        # Convert back to full 3x3 tensor
        return LouZhangYoon3D._kelvin_to_sym(e_k)
    # -------------------------------------------------------------------------
    @staticmethod
    def _sym_to_kelvin(tensor):
        """Convert a symmetric second-order tensor to Kelvin form.

        The Kelvin ordering is ``[11, 22, 33, 12, 13, 23]`` with
        off-diagonal components scaled by ``sqrt(2)``.

        Parameters
        ----------
        tensor : torch.Tensor
            Symmetric second-order tensor with shape ``(..., 3, 3)``.

        Returns
        -------
        kelvin : torch.Tensor
            Kelvin-form representation with shape ``(..., 6)``.
        """
        sqrt2 = math.sqrt(2.0)
        return torch.stack([
            tensor[..., 0, 0],
            tensor[..., 1, 1],
            tensor[..., 2, 2],
            sqrt2*tensor[..., 0, 1],
            sqrt2*tensor[..., 0, 2],
            sqrt2*tensor[..., 1, 2]], dim=-1)
    # -------------------------------------------------------------------------
    @staticmethod
    def _kelvin_to_sym(kelvin):
        """Convert a Kelvin-form 6-vector to a symmetric second-order
        tensor.

        Parameters
        ----------
        kelvin : torch.Tensor
            Kelvin-form representation with shape ``(..., 6)``.

        Returns
        -------
        tensor : torch.Tensor
            Symmetric second-order tensor with shape ``(..., 3, 3)``.
        """
        inv_sqrt2 = 1.0/math.sqrt(2.0)
        batch_shape = kelvin.shape[:-1]
        tensor = torch.zeros(*batch_shape, 3, 3,
                             device=kelvin.device, dtype=kelvin.dtype)
        tensor[..., 0, 0] = kelvin[..., 0]
        tensor[..., 1, 1] = kelvin[..., 1]
        tensor[..., 2, 2] = kelvin[..., 2]
        tensor[..., 0, 1] = inv_sqrt2*kelvin[..., 3]
        tensor[..., 1, 0] = inv_sqrt2*kelvin[..., 3]
        tensor[..., 0, 2] = inv_sqrt2*kelvin[..., 4]
        tensor[..., 2, 0] = inv_sqrt2*kelvin[..., 4]
        tensor[..., 1, 2] = inv_sqrt2*kelvin[..., 5]
        tensor[..., 2, 1] = inv_sqrt2*kelvin[..., 5]
        return tensor
    # -------------------------------------------------------------------------
    @staticmethod
    def _fourth_to_kelvin(tensor4):
        """Convert a minor-symmetric fourth-order tensor to a Kelvin
        matrix.

        Parameters
        ----------
        tensor4 : torch.Tensor
            Fourth-order tensor with shape ``(..., 3, 3, 3, 3)``
            assumed minor-symmetric (both index pairs).

        Returns
        -------
        kelvin_matrix : torch.Tensor
            Kelvin matrix with shape ``(..., 6, 6)``.
        """
        sqrt2 = math.sqrt(2.0)
        idx_pairs = [(0, 0), (1, 1), (2, 2),
                     (0, 1), (0, 2), (1, 2)]
        factors = [1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2]
        batch_shape = tensor4.shape[:-4]
        kelvin_matrix = torch.zeros(
            *batch_shape, 6, 6,
            device=tensor4.device, dtype=tensor4.dtype)
        for p, (i, j) in enumerate(idx_pairs):
            for qidx, (k, l) in enumerate(idx_pairs):
                kelvin_matrix[..., p, qidx] = (
                    factors[p]*factors[qidx]*tensor4[..., i, j, k, l])
        return kelvin_matrix
    # -------------------------------------------------------------------------
    @staticmethod
    def _kelvin_to_fourth(kelvin_matrix):
        """Convert a Kelvin 6x6 matrix to a minor-symmetric
        fourth-order tensor.

        Inverse of :meth:`_fourth_to_kelvin`. Fills all four
        minor-symmetric positions
        ``(i,j,k,l), (j,i,k,l), (i,j,l,k), (j,i,l,k)``.

        Parameters
        ----------
        kelvin_matrix : torch.Tensor
            Kelvin matrix with shape ``(..., 6, 6)``.

        Returns
        -------
        tensor4 : torch.Tensor
            Minor-symmetric fourth-order tensor with shape
            ``(..., 3, 3, 3, 3)``.
        """
        sqrt2 = math.sqrt(2.0)
        idx_pairs = [(0, 0), (1, 1), (2, 2),
                     (0, 1), (0, 2), (1, 2)]
        factors = [1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2]
        batch_shape = kelvin_matrix.shape[:-2]
        tensor4 = torch.zeros(
            *batch_shape, 3, 3, 3, 3,
            device=kelvin_matrix.device,
            dtype=kelvin_matrix.dtype)
        for p, (i, j) in enumerate(idx_pairs):
            for qidx, (k, l) in enumerate(idx_pairs):
                val = (
                    kelvin_matrix[..., p, qidx]
                    / (factors[p]*factors[qidx]))
                tensor4[..., i, j, k, l] = val
                tensor4[..., j, i, k, l] = val
                tensor4[..., i, j, l, k] = val
                tensor4[..., j, i, l, k] = val
        return tensor4
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_cone_tangent(jacobian, C):
        """Consistent cone-surface tangent via implicit diff.

        Let ``R(x, e_trial) = 0`` at converged ``x* =
        (eps_e_kelvin[6], q, inc_p_mult)`` define the cone NR
        system. Only the first six residual equations depend on
        ``e_trial`` with ``dR_r1/d(e_trial_kelvin) = -I_6``.
        Therefore
        ``dx*/d(e_trial_kelvin) = jacobian^{-1} @ [I_6; 0; 0]``
        and the consistent tangent in Kelvin form reads
        ``C_ep_k = C_k @ (jacobian^{-1})[:6, :6]``.

        Parameters
        ----------
        jacobian : torch.Tensor
            Converged 8x8 NR Jacobian of shape ``(N, 8, 8)``.
        C : torch.Tensor
            Elastic stiffness tensor with shape
            ``(N, 3, 3, 3, 3)``.

        Returns
        -------
        ddsdde : torch.Tensor
            Consistent algorithmic tangent with shape
            ``(N, 3, 3, 3, 3)``.
        """
        device = jacobian.device
        dtype = jacobian.dtype
        n = jacobian.shape[0]
        # RHS: (N, 8, 6) with identity in first 6 rows
        rhs = torch.zeros(
            n, 8, 6, device=device, dtype=dtype)
        rhs[:, :6, :6] = torch.eye(
            6, device=device, dtype=dtype)
        # Solve jacobian @ Z = rhs
        Z = torch.linalg.solve(jacobian, rhs)
        # First six rows: d(eps_e_kelvin)/d(e_trial_kelvin)
        A = Z[:, :6, :]
        # Elastic stiffness in Kelvin (N, 6, 6)
        C_k = LouZhangYoon3D._fourth_to_kelvin(C)
        # d(sigma_kelvin)/d(de_kelvin) = C_k @ A
        ddsdde_k = torch.matmul(C_k, A)
        return LouZhangYoon3D._kelvin_to_fourth(ddsdde_k)
    # -------------------------------------------------------------------------
    @staticmethod
    def _identity_tensors(device, dtype):
        """Build reusable identity and projector tensors.

        Parameters
        ----------
        device : torch.device
            Device on which tensors are allocated.
        dtype : torch.dtype
            Floating-point precision.

        Returns
        -------
        soid : torch.Tensor
            Second-order identity tensor with shape ``(3, 3)``.
        fosym : torch.Tensor
            Symmetric fourth-order identity with shape
            ``(3, 3, 3, 3)``.
        fodevsym : torch.Tensor
            Symmetric fourth-order deviatoric projector with shape
            ``(3, 3, 3, 3)``.
        """
        soid = torch.eye(3, device=device, dtype=dtype)
        i4 = torch.einsum('ij,kl->ijkl', soid, soid)
        i4s = 0.5*(torch.einsum('ik,jl->ijkl', soid, soid)
                   + torch.einsum('il,jk->ijkl', soid, soid))
        fosym = i4s
        fodevsym = i4s - (1.0/3.0)*i4
        return soid, fosym, fodevsym
    # -------------------------------------------------------------------------
    @staticmethod
    def _invariants(stress):
        """Compute stress and deviatoric stress invariants.

        Parameters
        ----------
        stress : torch.Tensor
            Stress tensor with shape ``(..., 3, 3)``.

        Returns
        -------
        i1 : torch.Tensor
            First principal invariant of the stress tensor.
        j2 : torch.Tensor
            Second invariant of the deviatoric stress tensor.
        j3 : torch.Tensor
            Third invariant of the deviatoric stress tensor.
        """
        # First principal invariant of the stress tensor
        i1 = (stress[..., 0, 0] + stress[..., 1, 1]
              + stress[..., 2, 2])
        # Trace of the squared stress tensor
        tr_sq = torch.einsum('...ij,...ji->...', stress, stress)
        # Second principal invariant of the stress tensor
        i2 = 0.5*(i1**2 - tr_sq)
        # Third principal invariant of the stress tensor
        i3 = torch.det(stress)
        # Invariants of the deviatoric stress tensor
        j2 = (1.0/3.0)*i1**2 - i2
        j3 = (2.0/27.0)*i1**3 - (1.0/3.0)*i1*i2 + i3
        return i1, j2, j3
    # -------------------------------------------------------------------------
    @staticmethod
    def _invariants_and_derivatives(stress):
        """Compute stress invariants and their derivatives.

        Parameters
        ----------
        stress : torch.Tensor
            Stress tensor with shape ``(..., 3, 3)``.

        Returns
        -------
        i1 : torch.Tensor
            First principal invariant of the stress tensor.
        j2 : torch.Tensor
            Second invariant of the deviatoric stress tensor.
        j3 : torch.Tensor
            Third invariant of the deviatoric stress tensor.
        di1_dstress : torch.Tensor
            First-order derivative of ``i1`` with respect to stress
            with shape ``(..., 3, 3)``.
        dj2_dstress : torch.Tensor
            First-order derivative of ``j2`` with respect to stress
            with shape ``(..., 3, 3)``.
        dj3_dstress : torch.Tensor
            First-order derivative of ``j3`` with respect to stress
            with shape ``(..., 3, 3)``.
        d2j2_dstress2 : torch.Tensor
            Second-order derivative of ``j2`` with respect to stress
            with shape ``(..., 3, 3, 3, 3)``.
        d2j3_dstress2 : torch.Tensor
            Second-order derivative of ``j3`` with respect to stress
            with shape ``(..., 3, 3, 3, 3)``.
        """
        device = stress.device
        dtype = stress.dtype
        # Build identity and deviatoric projector tensors
        soid, _, fodevsym = LouZhangYoon3D._identity_tensors(device, dtype)
        # Compute invariants
        i1, j2, j3 = LouZhangYoon3D._invariants(stress)
        # Compute deviatoric stress tensor
        dev_stress = stress - (i1/3.0).unsqueeze(-1).unsqueeze(-1)*soid
        # Determinant and inverse of the deviatoric stress tensor
        dev_det = torch.det(dev_stress)
        # Regularized inverse to avoid singularities on deviatoric plane
        dev_inv = torch.linalg.pinv(dev_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Derivative of inverse of symmetric tensor
        d_inv_d_sym = LouZhangYoon3D._fourth_inv_sym(dev_inv)
        # Auxiliary terms
        w6 = torch.einsum('...ij,...ijkl->...kl', dev_inv, fodevsym)
        dw6_dstress = torch.einsum(
            '...abij,...ijkl->...abkl',
            torch.einsum('...mnij,...mnab->...abij', fodevsym, d_inv_d_sym),
            fodevsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # First-order derivatives with respect to stress
        di1_dstress = soid.expand(stress.shape)
        dj2_dstress = dev_stress
        dj3_dstress = dev_det.unsqueeze(-1).unsqueeze(-1)*w6
        # Second-order derivatives with respect to stress
        d2j2_dstress2 = fodevsym.expand(stress.shape[:-2] + fodevsym.shape)
        dyad_w6_dj3 = torch.einsum('...ij,...kl->...ijkl', w6, dj3_dstress)
        d2j3_dstress2 = (dyad_w6_dj3
                         + dev_det.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                         .unsqueeze(-1)*dw6_dstress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return (i1, j2, j3, di1_dstress, dj2_dstress, dj3_dstress,
                d2j2_dstress2, d2j3_dstress2)
    # -------------------------------------------------------------------------
    @staticmethod
    def _fourth_inv_sym(s_inv):
        """Derivative of inverse of symmetric second-order tensor.

        The closed-form expression is
        :math:`\\partial S^{-1}_{kl}/\\partial S_{mn} = -\\tfrac{1}{2}
        (S^{-1}_{km} S^{-1}_{nl} + S^{-1}_{kn} S^{-1}_{ml})`.

        Parameters
        ----------
        s_inv : torch.Tensor
            Inverse of a symmetric second-order tensor with shape
            ``(..., 3, 3)``.

        Returns
        -------
        d_inv_d_sym : torch.Tensor
            Fourth-order tensor with shape ``(..., 3, 3, 3, 3)``
            with ``[..., k, l, m, n]`` storing
            :math:`\\partial S^{-1}_{kl}/\\partial S_{mn}`.
        """
        term1 = torch.einsum('...km,...nl->...klmn', s_inv, s_inv)
        term2 = torch.einsum('...kn,...ml->...klmn', s_inv, s_inv)
        return -0.5*(term1 + term2)
    # -------------------------------------------------------------------------
    @staticmethod
    def _effective_stress(i1, j2, j3, yield_a, yield_b, yield_c, yield_d):
        """Compute the LZY effective stress.

        Parameters
        ----------
        i1 : torch.Tensor
            First principal invariant of the stress tensor.
        j2 : torch.Tensor
            Second invariant of the deviatoric stress tensor.
        j3 : torch.Tensor
            Third invariant of the deviatoric stress tensor.
        yield_a, yield_b, yield_c, yield_d : torch.Tensor
            Yield parameters.

        Returns
        -------
        eff : torch.Tensor
            Effective stress.
        """
        w1 = yield_b*i1
        w4 = j2**3 - yield_c*(j3**2)
        w3 = yield_d*j3
        w5 = torch.sqrt(torch.clamp(w4, min=0.0)) - w3
        # The cubic root of w5 is evaluated preserving sign
        w5_third = torch.sign(w5)*torch.pow(torch.abs(w5), 1.0/3.0)
        return yield_a*(w1 + w5_third)
    # -------------------------------------------------------------------------
    def _return_mapping_cone(self, e_trial_strain, q_old, C):
        """Perform the cone-surface return-mapping for yielded entries.

        Parameters
        ----------
        e_trial_strain : torch.Tensor
            Elastic trial strain with shape ``(N, 3, 3)``.
        q_old : torch.Tensor
            Last converged accumulated plastic strain with shape
            ``(N,)``.
        C : torch.Tensor
            Elastic stiffness tensor with shape ``(N, 3, 3, 3, 3)``.

        Returns
        -------
        stress_new : torch.Tensor
            Updated stress tensor with shape ``(N, 3, 3)``.
        q_new : torch.Tensor
            Updated accumulated plastic strain with shape ``(N,)``.
        jacobian : torch.Tensor
            Converged NR Jacobian at ``x*`` with shape
            ``(N, 8, 8)``, used to build the consistent tangent
            via implicit differentiation.
        """
        device = e_trial_strain.device
        dtype = e_trial_strain.dtype
        n_elem = e_trial_strain.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial iterative guess
        e_strain = e_trial_strain.clone()
        q = q_old.clone()
        inc_p_mult = torch.zeros(n_elem, device=device, dtype=dtype)
        # Initial yield stress (used as reference normalization)
        init_yield_stress = self.sigma_f(
            torch.zeros(n_elem, device=device, dtype=dtype))
        # Small threshold to handle near-zero values
        small = 1e-8
        # Convergence norm of iterative solution vector
        conv_diter_norm = torch.zeros(n_elem, device=device, dtype=dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Newton-Raphson iterative loop
        for nr_iter in range(self.max_iter + 1):
            # Build residuals and Jacobian
            r_vec, jacobian, flow_vector = (
                self._cone_residual_and_jacobian(
                    e_strain, e_trial_strain, q, q_old, inc_p_mult,
                    C, init_yield_stress))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Residual convergence norms
            norm_etrial = torch.linalg.norm(
                e_trial_strain.reshape(n_elem, -1), dim=-1)
            norm_r1 = torch.linalg.norm(r_vec[:, :6], dim=-1)
            conv_r1 = torch.where(norm_etrial < small,
                                  norm_r1,
                                  norm_r1/torch.clamp(norm_etrial, min=small))
            conv_r2 = torch.where(
                torch.abs(q_old) < small,
                torch.abs(r_vec[:, 6]),
                torch.abs(r_vec[:, 6])/torch.clamp(torch.abs(q_old),
                                                   min=small))
            conv_r3 = torch.abs(r_vec[:, 7])
            conv_residual = (conv_r1 + conv_r2 + conv_r3)/3.0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check convergence of all elements
            if nr_iter > 0 and bool(
                    (conv_residual < self.tolerance).all()) and bool(
                    (conv_diter_norm < self.tolerance).all()):
                break
            if nr_iter == self.max_iter:
                break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Solve batched linearized NR system
            d_iter = torch.linalg.solve(jacobian,
                                        (-r_vec).unsqueeze(-1)).squeeze(-1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute convergence norm of iterative solution vector
            norm_factors = torch.cat([
                norm_etrial.unsqueeze(-1).expand(n_elem, 6),
                torch.abs(q_old).unsqueeze(-1).expand(n_elem, 2)], dim=-1)
            scaled_diter = torch.where(norm_factors > small,
                                       d_iter/torch.clamp(norm_factors,
                                                          min=small),
                                       d_iter)
            conv_diter_norm = torch.linalg.norm(scaled_diter, dim=-1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update unknowns
            de_strain = LouZhangYoon3D._kelvin_to_sym(d_iter[:, :6])
            e_strain = e_strain + de_strain
            q = q + d_iter[:, 6]
            inc_p_mult = inc_p_mult + d_iter[:, 7]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute updated stress from updated elastic strain
        stress_new = torch.einsum('...ijkl,...kl->...ij', C, e_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_new, q, jacobian
    # -------------------------------------------------------------------------
    def _cone_residual_and_jacobian(self, e_strain, e_trial_strain, q,
                                    q_old, inc_p_mult, C,
                                    init_yield_stress):
        """Build batched cone-surface NR residuals and Jacobian.

        Parameters
        ----------
        e_strain : torch.Tensor
            Current elastic strain iterate with shape ``(N, 3, 3)``.
        e_trial_strain : torch.Tensor
            Elastic trial strain with shape ``(N, 3, 3)``.
        q : torch.Tensor
            Current accumulated plastic strain iterate with shape
            ``(N,)``.
        q_old : torch.Tensor
            Last converged accumulated plastic strain with shape
            ``(N,)``.
        inc_p_mult : torch.Tensor
            Current incremental plastic multiplier with shape
            ``(N,)``.
        C : torch.Tensor
            Elastic stiffness tensor with shape ``(N, 3, 3, 3, 3)``.
        init_yield_stress : torch.Tensor
            Reference initial yield stress used as normalization,
            with shape ``(N,)``.

        Returns
        -------
        r_vec : torch.Tensor
            Residual vector in Kelvin form with shape ``(N, 8)``.
        jacobian : torch.Tensor
            Batched Jacobian with shape ``(N, 8, 8)``.
        flow_vector : torch.Tensor
            Flow vector (full tensor) with shape ``(N, 3, 3)`` (used
            by callers for diagnostics).
        """
        device = e_strain.device
        dtype = e_strain.dtype
        n_elem = e_strain.shape[0]
        soid, fosym, _ = LouZhangYoon3D._identity_tensors(device, dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute current stress and invariants with derivatives
        stress = torch.einsum('...ijkl,...kl->...ij', C, e_strain)
        (i1, j2, j3, di1_dstress, dj2_dstress, dj3_dstress,
         d2j2_dstress2, d2j3_dstress2) = \
            LouZhangYoon3D._invariants_and_derivatives(stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Current yield stress and hardening modulus
        yield_stress = self.sigma_f(q)
        hard_slope = self.sigma_f_prime(q)
        # Current yield parameters and their derivatives
        yield_a = self.yield_a(q)
        yield_b = self.yield_b(q)
        yield_c = self.yield_c(q)
        yield_d = self.yield_d(q)
        if not self.is_fixed_yield_parameters:
            a_slope = self.yield_a_prime(q)
            b_slope = self.yield_b_prime(q)
            c_slope = self.yield_c_prime(q)
            d_slope = self.yield_d_prime(q)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Auxiliary terms
        yb = yield_b.unsqueeze(-1).unsqueeze(-1)
        yc = yield_c.unsqueeze(-1).unsqueeze(-1)
        yd = yield_d.unsqueeze(-1).unsqueeze(-1)
        ya = yield_a.unsqueeze(-1).unsqueeze(-1)
        w1 = yield_b*i1
        w2 = yield_c*(j3**2)
        w3 = yield_d*j3
        w4 = j2**3 - w2
        w4_safe = torch.clamp(w4, min=1e-16)
        sqrt_w4 = torch.sqrt(w4_safe)
        w5 = sqrt_w4 - w3
        # Cubic root preserving sign for w5
        w5_abs = torch.clamp(torch.abs(w5), min=1e-16)
        w5_third = torch.sign(w5)*torch.pow(w5_abs, 1.0/3.0)
        # Derivatives of auxiliary terms with respect to stress
        dw1_dstress = yb*di1_dstress
        dw2_dstress = 2.0*yc*j3.unsqueeze(-1).unsqueeze(-1)*dj3_dstress
        dw3_dstress = yd*dj3_dstress
        dw4_dstress = (3.0*(j2**2).unsqueeze(-1).unsqueeze(-1)*dj2_dstress
                       - dw2_dstress)
        sqrt_w4_ = sqrt_w4.unsqueeze(-1).unsqueeze(-1)
        dw5_dstress = (0.5/sqrt_w4_)*dw4_dstress - dw3_dstress
        # Second-order derivatives with respect to stress
        dyad_dj3 = torch.einsum('...ij,...kl->...ijkl',
                                dj3_dstress, dj3_dstress)
        d2w2_dstress2 = 2.0*yc.unsqueeze(-1).unsqueeze(-1)*(
            dyad_dj3
            + j3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * d2j3_dstress2)
        d2w3_dstress2 = yd.unsqueeze(-1).unsqueeze(-1)*d2j3_dstress2
        dyad_dj2 = torch.einsum('...ij,...kl->...ijkl',
                                dj2_dstress, dj2_dstress)
        d2w4_dstress2 = (6.0*j2.unsqueeze(-1).unsqueeze(-1)
                         .unsqueeze(-1).unsqueeze(-1)*dyad_dj2
                         + 3.0*(j2**2).unsqueeze(-1).unsqueeze(-1)
                         .unsqueeze(-1).unsqueeze(-1)*d2j2_dstress2
                         - d2w2_dstress2)
        w4_pow32 = (w4_safe**(3.0/2.0)).unsqueeze(-1).unsqueeze(-1)\
            .unsqueeze(-1).unsqueeze(-1)
        sqrt_w4_4 = sqrt_w4.unsqueeze(-1).unsqueeze(-1)\
            .unsqueeze(-1).unsqueeze(-1)
        dyad_dw4 = torch.einsum('...ij,...kl->...ijkl',
                                dw4_dstress, dw4_dstress)
        d2w5_dstress2 = (-0.25*(1.0/w4_pow32)*dyad_dw4
                         + 0.5*(1.0/sqrt_w4_4)*d2w4_dstress2
                         - d2w3_dstress2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Derivatives with respect to q (only if yield params vary)
        if not self.is_fixed_yield_parameters:
            dw1_dq = i1*b_slope
            dw2_dq = (j3**2)*c_slope
            dw3_dq = j3*d_slope
            dw4_dq = -dw2_dq
            dw5_dq = 0.5*(1.0/sqrt_w4)*dw4_dq - dw3_dq
            # Cross second-order derivatives
            d2w1_dqdstress = b_slope.unsqueeze(-1).unsqueeze(-1)*soid
            d2w2_dqdstress = 2.0*j3.unsqueeze(-1).unsqueeze(-1)\
                * c_slope.unsqueeze(-1).unsqueeze(-1)*dj3_dstress
            d2w3_dqdstress = d_slope.unsqueeze(-1).unsqueeze(-1)\
                * dj3_dstress
            d2w4_dqdstress = -d2w2_dqdstress
            d2w5_dqdstress = (
                -0.25*(1.0/w4_pow32.squeeze(-1).squeeze(-1))
                .unsqueeze(-1).unsqueeze(-1)
                * dw4_dq.unsqueeze(-1).unsqueeze(-1)*dw4_dstress
                + 0.5*(1.0/sqrt_w4_)*d2w4_dqdstress
                - d2w3_dqdstress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Effective stress
        eff_stress = yield_a*(w1 + w5_third)
        # Flow vector (gradient of effective stress with respect to stress)
        w5_pow_m2_3 = torch.pow(w5_abs, -2.0/3.0)\
            .unsqueeze(-1).unsqueeze(-1)
        flow_vector = ya*(dw1_dstress + (1.0/3.0)*w5_pow_m2_3*dw5_dstress)
        # Flow vector norm (needed for non-associative hardening)
        flow_k = LouZhangYoon3D._sym_to_kelvin(flow_vector)
        norm_flow = torch.linalg.norm(flow_k, dim=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # First residual (elastic strain equation)
        r1 = (e_strain - e_trial_strain
              + inc_p_mult.unsqueeze(-1).unsqueeze(-1)*flow_vector)
        # Second residual (accumulated plastic strain evolution)
        if self.is_associative_hardening:
            r2 = q - q_old - inc_p_mult
        else:
            r2 = q - q_old - inc_p_mult*math.sqrt(2.0/3.0)*norm_flow
        # Third residual (yield function, normalized)
        r3 = (eff_stress - yield_stress)/init_yield_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Derivative of flow vector with respect to stress
        w5_pow_m5_3 = torch.pow(w5_abs, -5.0/3.0)\
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        w5_pow_m2_3_4 = w5_pow_m2_3.unsqueeze(-1).unsqueeze(-1)
        dyad_dw5 = torch.einsum('...ij,...kl->...ijkl',
                                dw5_dstress, dw5_dstress)
        ya4 = ya.unsqueeze(-1).unsqueeze(-1)
        dflow_dstress = (1.0/3.0)*ya4*(
            -(2.0/3.0)*w5_pow_m5_3*dyad_dw5
            + w5_pow_m2_3_4*d2w5_dstress2)
        # Derivative of flow vector with respect to elastic strain
        dflow_destrain = torch.einsum('...ijkl,...klmn->...ijmn',
                                      dflow_dstress, C)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Derivative of effective stress with respect to elastic strain
        deff_destrain = torch.einsum('...ij,...ijkl->...kl',
                                     flow_vector, C)
        # Derivative of effective stress with respect to q
        if self.is_fixed_yield_parameters:
            deff_dq = torch.zeros_like(q)
        else:
            deff_dq = (a_slope*(w1 + w5_third)
                       + yield_a*(dw1_dq
                                  + (1.0/3.0)
                                  * w5_pow_m2_3.squeeze(-1).squeeze(-1)
                                  * dw5_dq))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Derivative of flow vector with respect to q
        if self.is_fixed_yield_parameters:
            dflow_dq = torch.zeros_like(flow_vector)
        else:
            a_slope4 = a_slope.unsqueeze(-1).unsqueeze(-1)
            w5_pow_m2_3_2 = w5_pow_m2_3
            dw5_dq4 = dw5_dq.unsqueeze(-1).unsqueeze(-1)
            dflow_dq = (a_slope4*(dw1_dstress
                                  + (1.0/3.0)*w5_pow_m2_3_2*dw5_dstress)
                        + ya*(d2w1_dqdstress
                              - (2.0/9.0)
                              * torch.pow(w5_abs, -5.0/3.0)
                              .unsqueeze(-1).unsqueeze(-1)
                              * dw5_dq4*dw5_dstress
                              + (1.0/3.0)*w5_pow_m2_3_2*d2w5_dqdstress))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Residual 1 derivatives
        dr1_destrain = (fosym.expand(n_elem, 3, 3, 3, 3)
                        + inc_p_mult.unsqueeze(-1).unsqueeze(-1)
                        .unsqueeze(-1).unsqueeze(-1)*dflow_destrain)
        dr1_dq = inc_p_mult.unsqueeze(-1).unsqueeze(-1)*dflow_dq
        dr1_dinc = flow_vector
        # Residual 2 derivatives
        if self.is_associative_hardening:
            dr2_destrain = torch.zeros_like(flow_vector)
            dr2_dq = torch.ones(n_elem, device=device, dtype=dtype)
            dr2_dinc = -1.0*torch.ones(n_elem, device=device, dtype=dtype)
        else:
            norm_flow_safe = torch.clamp(norm_flow, min=1e-16)
            factor = -inc_p_mult*math.sqrt(2.0/3.0)/norm_flow_safe
            dr2_destrain = (factor.unsqueeze(-1).unsqueeze(-1)
                            * torch.einsum('...ij,...ijkl->...kl',
                                           flow_vector, dflow_destrain))
            if self.is_fixed_yield_parameters:
                dr2_dq = torch.ones(n_elem, device=device, dtype=dtype)
            else:
                dr2_dq = (1.0 + factor*torch.einsum('...ij,...ij->...',
                                                    flow_vector, dflow_dq))
            dr2_dinc = -math.sqrt(2.0/3.0)*norm_flow
        # Residual 3 derivatives
        dr3_destrain = deff_destrain/init_yield_stress.unsqueeze(-1)\
            .unsqueeze(-1)
        if self.is_fixed_yield_parameters:
            dr3_dq = -hard_slope/init_yield_stress
        else:
            dr3_dq = (deff_dq - hard_slope)/init_yield_stress
        dr3_dinc = torch.zeros(n_elem, device=device, dtype=dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble residuals in Kelvin form
        r1_k = LouZhangYoon3D._sym_to_kelvin(r1)
        r_vec = torch.cat([r1_k, r2.unsqueeze(-1), r3.unsqueeze(-1)], dim=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble Jacobian blocks in Kelvin form
        j11 = LouZhangYoon3D._fourth_to_kelvin(dr1_destrain)
        j12 = LouZhangYoon3D._sym_to_kelvin(dr1_dq).unsqueeze(-1)
        j13 = LouZhangYoon3D._sym_to_kelvin(dr1_dinc).unsqueeze(-1)
        j21 = LouZhangYoon3D._sym_to_kelvin(dr2_destrain).unsqueeze(-2)
        j22 = dr2_dq.unsqueeze(-1).unsqueeze(-1)
        j23 = dr2_dinc.unsqueeze(-1).unsqueeze(-1)
        j31 = LouZhangYoon3D._sym_to_kelvin(dr3_destrain).unsqueeze(-2)
        j32 = dr3_dq.unsqueeze(-1).unsqueeze(-1)
        j33 = dr3_dinc.unsqueeze(-1).unsqueeze(-1)
        # Assemble full 8x8 Jacobian per element
        row1 = torch.cat([j11, j12, j13], dim=-1)
        row2 = torch.cat([j21, j22, j23], dim=-1)
        row3 = torch.cat([j31, j32, j33], dim=-1)
        jacobian = torch.cat([row1, row2, row3], dim=-2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return r_vec, jacobian, flow_vector
    # -------------------------------------------------------------------------
    def _return_mapping_apex(self, p_trial, q_old, K):
        """Perform the apex return-mapping for yielded entries.

        Parameters
        ----------
        p_trial : torch.Tensor
            Trial pressure with shape ``(N,)``.
        q_old : torch.Tensor
            Last converged accumulated plastic strain with shape
            ``(N,)``.
        K : torch.Tensor
            Bulk modulus with shape ``(N,)``.

        Returns
        -------
        stress_new : torch.Tensor
            Updated stress tensor with shape ``(N, 3, 3)``, purely
            hydrostatic.
        q_new : torch.Tensor
            Updated accumulated plastic strain with shape ``(N,)``.
        c_apex : torch.Tensor
            Apex tangent coefficient with shape ``(N,)``. The apex
            tangent reads ``ddsdde = c_apex * (I otimes I)``.
            Derivation:
            ``d(pressure)/d(p_trial) = (jac - K)/jac`` from implicit
            diff of the apex NR residual, and
            ``d(p_trial)/d(de_kl) = K * delta_kl``; since
            ``sigma_ij = pressure * delta_ij``,
            ``c_apex = K * (jac - K)/jac``.
        """
        device = p_trial.device
        dtype = p_trial.dtype
        n_elem = p_trial.shape[0]
        small = 1e-8
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial yield parameters at zero plastic strain
        zero = torch.zeros(n_elem, device=device, dtype=dtype)
        a_init = self.yield_a(zero)
        b_init = self.yield_b(zero)
        # Drucker-Prager-equivalent parameters
        etay = 3.0*a_init*b_init
        xi = (2.0*math.sqrt(3.0)/3.0)*torch.sqrt(
            torch.clamp(1.0 - (1.0/3.0)*etay**2, min=0.0))
        safe_etay = torch.where(torch.abs(etay) < 1e-6,
                                torch.tensor(1e-6, device=device, dtype=dtype),
                                torch.abs(etay))
        alpha = xi/safe_etay
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial iterative guess
        inc_vol_p_strain = torch.zeros(n_elem, device=device, dtype=dtype)
        conv_diter_norm = torch.zeros(n_elem, device=device, dtype=dtype)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Newton-Raphson iterative loop
        for nr_iter in range(self.max_iter + 1):
            q_iter = q_old + alpha*inc_vol_p_strain
            yield_stress = self.sigma_f(q_iter)
            hard_slope = self.sigma_f_prime(q_iter)
            yield_a = self.yield_a(q_iter)
            yield_b = self.yield_b(q_iter)
            safe_yield_b = torch.where(
                torch.abs(yield_b) < 1e-6,
                torch.tensor(1e-6, device=device, dtype=dtype),
                torch.abs(yield_b))
            beta = 1.0/(3.0*yield_a*safe_yield_b)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Residual and Jacobian
            residual = (yield_stress*beta
                        - (p_trial - K*inc_vol_p_strain))
            jacobian = alpha*beta*hard_slope + K
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Convergence check
            safe_yield = torch.where(
                torch.abs(yield_stress) < small,
                torch.tensor(1.0, device=device, dtype=dtype),
                torch.abs(yield_stress))
            conv_residual = torch.abs(residual)/safe_yield
            if nr_iter > 0 and bool(
                    (conv_residual < self.tolerance).all()) and bool(
                    (conv_diter_norm < self.tolerance).all()):
                break
            if nr_iter == self.max_iter:
                break
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Newton-Raphson iteration
            d_iter = -residual/jacobian
            # Convergence norm of iterative solution
            scaled = torch.where(torch.abs(q_old) > small,
                                 d_iter/torch.clamp(torch.abs(q_old),
                                                    min=small),
                                 d_iter)
            conv_diter_norm = torch.abs(scaled)
            # Update unknown
            inc_vol_p_strain = inc_vol_p_strain + d_iter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute pressure and stress
        pressure = p_trial - K*inc_vol_p_strain
        soid = torch.eye(3, device=device, dtype=dtype)
        stress_new = (pressure.unsqueeze(-1).unsqueeze(-1)
                      * soid.expand(n_elem, 3, 3))
        # Accumulated plastic strain
        q_new = q_old + alpha*inc_vol_p_strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apex consistent tangent coefficient
        # jac = alpha*beta*hard_slope + K at convergence
        c_apex = K*(jacobian - K)/torch.clamp(
            jacobian, min=small)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return stress_new, q_new, c_apex
# =============================================================================
#
# =============================================================================
class LouZhangYoonPlaneStrain(LouZhangYoon3D):
    """Lou-Zhang-Yoon constitutive model under 2D plane-strain small
    strains.

    The plane-strain update is performed by embedding the 2D state in
    3D, invoking the 3D core update, and contracting the result back
    to 2D. The out-of-plane elastic strain component
    :math:`\\varepsilon^{e}_{33}` is stored as an additional state
    variable to recover the 3D kinematic description on subsequent
    steps.

    Attributes
    ----------
    C : torch.Tensor
        2D plane-strain elastic stiffness tensor with shape
        ``(..., 2, 2, 2, 2)``.
    C_3d : torch.Tensor
        3D elastic stiffness tensor with shape
        ``(..., 3, 3, 3, 3)`` used internally by the state update.
    n_state : int
        Number of internal state variables (here: 2 ---
        ``[acc_p_strain, e_strain_33]``).
    """
    def __init__(self, E, nu, sigma_f, sigma_f_prime,
                 yield_a, yield_a_prime, yield_b, yield_b_prime,
                 yield_c, yield_c_prime, yield_d, yield_d_prime,
                 is_associative_hardening=False,
                 is_fixed_yield_parameters=True,
                 is_apex_handling=True, apex_switch_tol=0.05,
                 tolerance=1e-6, max_iter=10):
        """Plane-strain constitutive model constructor.

        Parameters
        ----------
        See :class:`LouZhangYoon3D`.
        """
        # Initialize 3D parent (builds 3D elastic stiffness tensor)
        super().__init__(
            E, nu, sigma_f, sigma_f_prime,
            yield_a, yield_a_prime, yield_b, yield_b_prime,
            yield_c, yield_c_prime, yield_d, yield_d_prime,
            is_associative_hardening=is_associative_hardening,
            is_fixed_yield_parameters=is_fixed_yield_parameters,
            is_apex_handling=is_apex_handling,
            apex_switch_tol=apex_switch_tol,
            tolerance=tolerance, max_iter=max_iter)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Preserve 3D stiffness tensor for internal state update
        self.C_3d = self.C
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Overwrite C with a 2D plane-strain stiffness tensor
        plane = IsotropicElasticityPlaneStrain(E, nu)
        self.C = plane.C
        self.Cs = plane.Cs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Additional state: out-of-plane elastic strain component
        self.n_state = 2
    # -------------------------------------------------------------------------
    def vectorize(self, n_elem):
        """Return a vectorized copy of the material for ``n_elem``.

        Parameters
        ----------
        n_elem : int
            Number of elements to vectorize the material for.

        Returns
        -------
        material : LouZhangYoonPlaneStrain
            Vectorized copy of the material.
        """
        if self.is_vectorized:
            print('Material is already vectorized.')
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            return LouZhangYoonPlaneStrain(
                E, nu, self.sigma_f, self.sigma_f_prime,
                self.yield_a, self.yield_a_prime,
                self.yield_b, self.yield_b_prime,
                self.yield_c, self.yield_c_prime,
                self.yield_d, self.yield_d_prime,
                is_associative_hardening=self.is_associative_hardening,
                is_fixed_yield_parameters=self.is_fixed_yield_parameters,
                is_apex_handling=self.is_apex_handling,
                apex_switch_tol=self.apex_switch_tol,
                tolerance=self.tolerance, max_iter=self.max_iter)
    # -------------------------------------------------------------------------
    def step(self, H_inc, F, sigma, state, de0):
        """Perform an incremental state update in 2D plane strain.

        Parameters
        ----------
        H_inc : torch.Tensor
            2D incremental displacement gradient with shape
            ``(..., 2, 2)``.
        F : torch.Tensor
            Current 2D deformation gradient with shape
            ``(..., 2, 2)`` (unused).
        sigma : torch.Tensor
            Current 2D Cauchy stress tensor with shape
            ``(..., 2, 2)``.
        state : torch.Tensor
            Internal state variables with shape ``(..., 2)``, storing
            ``[acc_p_strain, e_strain_33]``.
        de0 : torch.Tensor
            External small strain increment with shape
            ``(..., 2, 2)``.

        Returns
        -------
        sigma_new : torch.Tensor
            Updated 2D Cauchy stress tensor with shape
            ``(..., 2, 2)``.
        state_new : torch.Tensor
            Updated internal state with shape ``(..., 2)``.
        ddsdde : torch.Tensor
            Algorithmic tangent stiffness tensor with shape
            ``(..., 2, 2, 2, 2)``.
        """
        device = sigma.device
        dtype = sigma.dtype
        batch_shape = sigma.shape[:-2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute 2D small strain increment
        de_2d = 0.5*(H_inc.transpose(-1, -2) + H_inc) - de0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract last converged state
        q_old = state[..., 0]
        e33_old = state[..., 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build 3D tensors with out-of-plane components
        de_3d = torch.zeros(*batch_shape, 3, 3,
                            device=device, dtype=dtype)
        de_3d[..., :2, :2] = de_2d
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reconstruct 3D stress from 2D stress plus last-converged
        # out-of-plane component derived from the stored 3D elastic
        # strain field
        sigma_3d = torch.zeros(*batch_shape, 3, 3,
                               device=device, dtype=dtype)
        sigma_3d[..., :2, :2] = sigma
        # Reconstruct out-of-plane stress from elastic relation
        # sigma_33 = nu*(sigma_11 + sigma_22) + E*e33_old
        # derived from sigma = lam*tr(eps_e)*I + 2*mu*eps_e
        tr_s_in = sigma[..., 0, 0] + sigma[..., 1, 1]
        sigma_3d[..., 2, 2] = self.nu*tr_s_in + self.E*e33_old
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Expand 3D stiffness to batch shape
        C_3d = self.C_3d.expand(batch_shape + self.C_3d.shape[-4:])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Invoke 3D core update (returns consistent tangent)
        stress_3d, q_new, _, ddsdde_3d = self._step_3d_core(
            sigma_3d, de_3d, q_old, C_3d)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract in-plane stress and recover out-of-plane elastic strain
        sigma_new = stress_3d[..., :2, :2].clone()
        # e33_new = (sigma_33 - nu*(sigma_11 + sigma_22))/E
        tr_s_in_new = stress_3d[..., 0, 0] + stress_3d[..., 1, 1]
        e33_new = (stress_3d[..., 2, 2] - self.nu*tr_s_in_new)/self.E
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Assemble updated state variables
        state_new = state.clone()
        state_new[..., 0] = q_new
        state_new[..., 1] = e33_new
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plane-strain tangent: in-plane block of 3D tangent.
        # Plane-strain enforces de_33 = 0 kinematically, so the
        # full sensitivity of the in-plane stress to the in-plane
        # strain is captured by the top-left 2x2x2x2 block of the
        # 3D consistent tangent evaluated at de_33 = 0.
        ddsdde = ddsdde_3d[..., :2, :2, :2, :2].contiguous()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return sigma_new, state_new, ddsdde
# =============================================================================
