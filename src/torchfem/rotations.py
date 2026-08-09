import torch
from torch import Tensor


def polar_log(f: Tensor) -> tuple[Tensor, Tensor]:
    """Rotation and logarithmic stretch of a deformation gradient.

    Splits ``f = R U`` with ``R`` orthogonal and ``U`` symmetric
    positive definite, and returns ``R`` together with ``ln U``. Both
    follow from the eigendecomposition of ``C = f^T f``, which is
    exact, batched and differentiable.

    ``ln U`` is objective: superposing a rigid rotation on ``f``
    leaves it unchanged, because the rotation cancels in ``f^T f``.
    For an infinitesimal increment it reduces to the symmetric
    displacement gradient, so the small-strain description is its
    limit rather than a separate model.

    Args:
        f (Tensor): Deformation gradient of shape (..., 3, 3).

    Returns:
        tuple[Tensor, Tensor]: Orthogonal factor and logarithmic
            stretch, both of shape (..., 3, 3).
    """
    cauchy_green = f.transpose(-1, -2) @ f
    eigenvalues, eigenvectors = torch.linalg.eigh(cauchy_green)
    # C is positive definite for any admissible deformation, so a
    # non-positive eigenvalue means the element has inverted.
    if bool((eigenvalues <= 0.0).any()):
        raise ValueError(
            'Right Cauchy-Green tensor is not positive definite: the '
            'deformation gradient is singular or inverted.')
    roots = torch.sqrt(eigenvalues)
    transposed = eigenvectors.transpose(-1, -2)
    stretch_inverse = (eigenvectors @ torch.diag_embed(1.0 / roots)
                       @ transposed)
    stretch_log = (eigenvectors @ torch.diag_embed(torch.log(roots))
                   @ transposed)
    return f @ stretch_inverse, stretch_log


def planar_rotation(phi: float | Tensor) -> Tensor:
    """Create a planar rotation matrix with an angle phi."""
    phi = torch.as_tensor(phi)
    c = torch.cos(phi)
    s = torch.sin(phi)
    return torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-1,
    )


def axis_rotation(axis: Tensor, phi: float | Tensor) -> Tensor:
    """Create a rotation matrix with an angle phi around an axis."""
    phi = torch.as_tensor(phi)
    axis = axis / torch.norm(axis)
    x, y, z = axis
    c = torch.cos(phi)
    s = torch.sin(phi)
    t = 1 - c
    return torch.stack(
        [
            torch.stack(
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                dim=-1,
            ),
            torch.stack(
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                dim=-1,
            ),
            torch.stack(
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
                dim=-1,
            ),
        ],
        dim=-1,
    )


def euler_rotation(rotation_angles: Tensor) -> Tensor:
    """Create a rotation matrix with Euler angles."""
    alpha = rotation_angles[..., 0]
    beta = rotation_angles[..., 1]
    gamma = rotation_angles[..., 2]
    return torch.stack(
        [
            torch.stack(
                [
                    torch.cos(alpha) * torch.cos(beta),
                    torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma)
                    - torch.sin(alpha) * torch.cos(gamma),
                    torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma)
                    + torch.sin(alpha) * torch.sin(gamma),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.sin(alpha) * torch.cos(beta),
                    torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma)
                    + torch.cos(alpha) * torch.cos(gamma),
                    torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma)
                    - torch.cos(alpha) * torch.sin(gamma),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    -torch.sin(beta),
                    torch.cos(beta) * torch.sin(gamma),
                    torch.cos(beta) * torch.cos(gamma),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )


def voigt_stress_rotation(R: Tensor) -> Tensor:
    """Rotation matrix for a stress tensor in Voigt notation."""
    if R.shape[-1] == 2 and R.shape[-2] == 2:
        R00 = R[..., 0, 0]
        R01 = R[..., 0, 1]
        R10 = R[..., 1, 0]
        R11 = R[..., 1, 1]
        return torch.stack(
            [
                torch.stack([R00**2, R10**2, R00 * R10], dim=-1),
                torch.stack([R01**2, R11**2, R01 * R11], dim=-1),
                torch.stack(
                    [2 * R01 * R00, 2 * R11 * R10, R00**2 - R10**2],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
    elif R.shape[-1] == 3 and R.shape[-2] == 3:
        return torch.stack(
            [
                torch.stack(
                    [
                        R[..., 0, 0] ** 2,
                        R[..., 1, 0] ** 2,
                        R[..., 2, 0] ** 2,
                        R[..., 1, 0] * R[..., 2, 0],
                        R[..., 2, 0] * R[..., 0, 0],
                        R[..., 0, 0] * R[..., 1, 0],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        R[..., 0, 1] ** 2,
                        R[..., 1, 1] ** 2,
                        R[..., 2, 1] ** 2,
                        R[..., 1, 1] * R[..., 2, 1],
                        R[..., 2, 1] * R[..., 0, 1],
                        R[..., 0, 1] * R[..., 1, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        R[..., 0, 2] ** 2,
                        R[..., 1, 2] ** 2,
                        R[..., 2, 2] ** 2,
                        R[..., 1, 2] * R[..., 2, 2],
                        R[..., 2, 2] * R[..., 0, 2],
                        R[..., 0, 2] * R[..., 1, 2],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * R[..., 0, 1] * R[..., 0, 2],
                        2 * R[..., 1, 1] * R[..., 1, 2],
                        2 * R[..., 2, 1] * R[..., 2, 2],
                        R[..., 1, 1] * R[..., 2, 2]
                        + R[..., 1, 2] * R[..., 2, 1],
                        R[..., 2, 1] * R[..., 0, 2]
                        + R[..., 2, 2] * R[..., 0, 1],
                        R[..., 0, 1] * R[..., 1, 2]
                        + R[..., 0, 2] * R[..., 1, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * R[..., 0, 2] * R[..., 0, 0],
                        2 * R[..., 1, 2] * R[..., 1, 0],
                        2 * R[..., 2, 2] * R[..., 2, 0],
                        R[..., 1, 2] * R[..., 2, 0]
                        + R[..., 1, 0] * R[..., 2, 2],
                        R[..., 2, 2] * R[..., 0, 0]
                        + R[..., 2, 0] * R[..., 0, 2],
                        R[..., 0, 2] * R[..., 1, 0]
                        + R[..., 0, 0] * R[..., 1, 2],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        2 * R[..., 0, 0] * R[..., 0, 1],
                        2 * R[..., 1, 0] * R[..., 1, 1],
                        2 * R[..., 2, 0] * R[..., 2, 1],
                        R[..., 1, 0] * R[..., 2, 1]
                        + R[..., 1, 1] * R[..., 2, 0],
                        R[..., 2, 0] * R[..., 0, 1]
                        + R[..., 2, 1] * R[..., 0, 0],
                        R[..., 0, 0] * R[..., 1, 1]
                        + R[..., 0, 1] * R[..., 1, 0],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
    else:
        raise ValueError("Invalid shape of R.")


def voigt_strain_rotation(R: Tensor) -> Tensor:
    if R.shape[-1] == 2 and R.shape[-2] == 2:
        R00 = R[..., 0, 0]
        R01 = R[..., 0, 1]
        R10 = R[..., 1, 0]
        R11 = R[..., 1, 1]
        return torch.stack(
            [
                torch.stack([R00**2, R10**2, 2 * R00 * R10], dim=-1),
                torch.stack([R01**2, R11**2, 2 * R01 * R11], dim=-1),
                torch.stack([R01 * R00, R11 * R10, R00**2 - R10**2], dim=-1),
            ],
            dim=-1,
        )
    elif R.shape[-1] == 3 and R.shape[-2] == 3:
        return torch.stack(
            [
                torch.stack(
                    [
                        R[..., 0, 0] ** 2,
                        R[..., 1, 0] ** 2,
                        R[..., 2, 0] ** 2,
                        2 * R[..., 1, 0] * R[..., 2, 0],
                        2 * R[..., 2, 0] * R[..., 0, 0],
                        2 * R[..., 0, 0] * R[..., 1, 0],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        R[..., 0, 1] ** 2,
                        R[..., 1, 1] ** 2,
                        R[..., 2, 1] ** 2,
                        2 * R[..., 1, 1] * R[..., 2, 1],
                        2 * R[..., 2, 1] * R[..., 0, 1],
                        2 * R[..., 0, 1] * R[..., 1, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        R[..., 0, 2] ** 2,
                        R[..., 1, 2] ** 2,
                        R[..., 2, 2] ** 2,
                        2 * R[..., 1, 2] * R[..., 2, 2],
                        2 * R[..., 2, 2] * R[..., 0, 2],
                        2 * R[..., 0, 2] * R[..., 1, 2],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        R[..., 0, 1] * R[..., 0, 2],
                        R[..., 1, 1] * R[..., 1, 2],
                        R[..., 2, 1] * R[..., 2, 2],
                        R[..., 1, 1] * R[..., 2, 2]
                        + R[..., 1, 2] * R[..., 2, 1],
                        R[..., 2, 1] * R[..., 0, 2]
                        + R[..., 2, 2] * R[..., 0, 1],
                        R[..., 0, 1] * R[..., 1, 2]
                        + R[..., 0, 2] * R[..., 1, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        R[..., 0, 2] * R[..., 0, 0],
                        R[..., 1, 2] * R[..., 1, 0],
                        R[..., 2, 2] * R[..., 2, 0],
                        R[..., 1, 2] * R[..., 2, 0]
                        + R[..., 1, 0] * R[..., 2, 2],
                        R[..., 2, 2] * R[..., 0, 0]
                        + R[..., 2, 0] * R[..., 0, 2],
                        R[..., 0, 2] * R[..., 1, 0]
                        + R[..., 0, 0] * R[..., 1, 2],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        R[..., 0, 0] * R[..., 0, 1],
                        R[..., 1, 0] * R[..., 1, 1],
                        R[..., 2, 0] * R[..., 2, 1],
                        R[..., 1, 0] * R[..., 2, 1]
                        + R[..., 1, 1] * R[..., 2, 0],
                        R[..., 2, 0] * R[..., 0, 1]
                        + R[..., 2, 1] * R[..., 0, 0],
                        R[..., 0, 0] * R[..., 1, 1]
                        + R[..., 0, 1] * R[..., 1, 0],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
    else:
        raise ValueError("Invalid shape of R.")
