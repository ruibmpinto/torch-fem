"""
Multi-Point Constraint Example: 3D Periodic Boundary Conditions

Demonstrates periodic BCs on a unit cube subjected to uniaxial strain.
Tests master-slave elimination for coupling opposite faces.
"""

import torch
import numpy as np
from torchfem.solid import Solid
from torchfem.materials import IsotropicElasticity

# Material properties (steel)
E = 200e3  # Young's modulus (MPa)
nu = 0.3   # Poisson's ratio

# Create unit cube mesh
n_elem = 3  # Elements per direction
L = 1.0

# Generate structured hex mesh
n_nodes = n_elem + 1
x = torch.linspace(0, L, n_nodes)
y = torch.linspace(0, L, n_nodes)
z = torch.linspace(0, L, n_nodes)

X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
nodes = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)

# Element connectivity (hex8)
elements = []
for k in range(n_elem):
    for j in range(n_elem):
        for i in range(n_elem):
            n0 = k * n_nodes**2 + j * n_nodes + i
            n1 = k * n_nodes**2 + j * n_nodes + (i + 1)
            n2 = k * n_nodes**2 + (j + 1) * n_nodes + (i + 1)
            n3 = k * n_nodes**2 + (j + 1) * n_nodes + i
            n4 = (k + 1) * n_nodes**2 + j * n_nodes + i
            n5 = (k + 1) * n_nodes**2 + j * n_nodes + (i + 1)
            n6 = (k + 1) * n_nodes**2 + (j + 1) * n_nodes + (i + 1)
            n7 = (k + 1) * n_nodes**2 + (j + 1) * n_nodes + i
            elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

elements = torch.tensor(elements)

# Initialize FEM problem
material = IsotropicElasticity(E=E, nu=nu)
fem = Solid(nodes, elements, material, element_type='hexa')

print(f"Mesh: {nodes.shape[0]} nodes, {elements.shape[0]} elements")
print(f"Total DOFs: {fem.n_dofs}")

# Apply periodic boundary conditions using MPCs
# Match opposite faces: x=0 with x=L, y=0 with y=L, z=0 with z=L

tol = 1e-6

# Find nodes on each face
face_x0 = torch.where(torch.abs(nodes[:, 0]) < tol)[0]
face_xL = torch.where(torch.abs(nodes[:, 0] - L) < tol)[0]
face_y0 = torch.where(torch.abs(nodes[:, 1]) < tol)[0]
face_yL = torch.where(torch.abs(nodes[:, 1] - L) < tol)[0]
face_z0 = torch.where(torch.abs(nodes[:, 2]) < tol)[0]
face_zL = torch.where(torch.abs(nodes[:, 2] - L) < tol)[0]

print(f"\nBoundary nodes:")
print(f"  x=0 face: {len(face_x0)} nodes")
print(f"  x=L face: {len(face_xL)} nodes")
print(f"  y=0 face: {len(face_y0)} nodes")
print(f"  y=L face: {len(face_yL)} nodes")
print(f"  z=0 face: {len(face_z0)} nodes")
print(f"  z=L face: {len(face_zL)} nodes")


def match_face_nodes(face0, faceL, coord_idx0, coord_idx1):
    """Match nodes on opposite faces by sorting on other coordinates."""
    # Sort both faces by the other two coordinates
    coords0 = nodes[face0][:, [coord_idx0, coord_idx1]]
    coordsL = nodes[faceL][:, [coord_idx0, coord_idx1]]

    # Create sorting keys
    key0 = coords0[:, 0] * 1000 + coords0[:, 1]
    keyL = coordsL[:, 0] * 1000 + coordsL[:, 1]

    idx0 = torch.argsort(key0)
    idxL = torch.argsort(keyL)

    return face0[idx0], faceL[idxL]


# Match nodes on opposite faces
x0_matched, xL_matched = match_face_nodes(face_x0, face_xL, 1, 2)
y0_matched, yL_matched = match_face_nodes(face_y0, face_yL, 0, 2)
z0_matched, zL_matched = match_face_nodes(face_z0, face_zL, 0, 1)

# Count unique constraints (accounting for edges and corners)
# Each face pair contributes constraints, but we must avoid over-constraining
# edges and corners

# Strategy: Apply x-periodicity to all x-face nodes
#           Apply y-periodicity to y-face nodes not on x-faces
#           Apply z-periodicity to z-face nodes not on x or y faces

# X-periodicity: all nodes on x=0 and x=L faces
x_periodic_nodes = len(x0_matched)

# Y-periodicity: nodes on y-faces excluding x-faces
y_mask = ~(torch.isin(y0_matched, face_x0) | torch.isin(y0_matched, face_xL))
y_periodic_nodes = y_mask.sum().item()

# Z-periodicity: nodes on z-faces excluding x and y faces
z_mask = ~(torch.isin(z0_matched, face_x0) | torch.isin(z0_matched, face_xL) |
           torch.isin(z0_matched, face_y0) | torch.isin(z0_matched, face_yL))
z_periodic_nodes = z_mask.sum().item()

n_constraints = 3 * (x_periodic_nodes + y_periodic_nodes + z_periodic_nodes)

C = torch.zeros(n_constraints, fem.n_dofs)
d = torch.zeros(n_constraints)

constraint_idx = 0

# X-direction periodicity (all x-face nodes)
for i in range(len(x0_matched)):
    node0 = x0_matched[i].item()
    nodeL = xL_matched[i].item()

    for dof in range(3):
        C[constraint_idx, nodeL * 3 + dof] = 1.0
        C[constraint_idx, node0 * 3 + dof] = -1.0
        constraint_idx += 1

# Y-direction periodicity (y-face nodes not on x-faces)
for i in range(len(y0_matched)):
    if not y_mask[i]:
        continue

    node0 = y0_matched[i].item()
    nodeL = yL_matched[i].item()

    for dof in range(3):
        C[constraint_idx, nodeL * 3 + dof] = 1.0
        C[constraint_idx, node0 * 3 + dof] = -1.0
        constraint_idx += 1

# Z-direction periodicity (z-face nodes not on x or y faces)
for i in range(len(z0_matched)):
    if not z_mask[i]:
        continue

    node0 = z0_matched[i].item()
    nodeL = zL_matched[i].item()

    for dof in range(3):
        C[constraint_idx, nodeL * 3 + dof] = 1.0
        C[constraint_idx, node0 * 3 + dof] = -1.0
        constraint_idx += 1

print(f"\nMPC setup:")
print(f"  X-periodic nodes: {x_periodic_nodes}")
print(f"  Y-periodic nodes: {y_periodic_nodes}")
print(f"  Z-periodic nodes: {z_periodic_nodes}")
print(f"  Total constraints: {n_constraints}")
print(f"  Constraint matrix shape: {C.shape}")

# Set MPCs
fem.set_mpc(C, d)

print(f"  Master DOFs: {len(fem._master_dofs)}")
print(f"  Slave DOFs: {len(fem._slave_dofs)}")

# Apply uniaxial strain in z-direction
# Fix one corner and apply force at opposite corner

# Find corner at (0,0,0)
corner_000 = torch.where(
    (torch.abs(nodes[:, 0]) < tol) &
    (torch.abs(nodes[:, 1]) < tol) &
    (torch.abs(nodes[:, 2]) < tol)
)[0][0].item()

# Find corner at (L,L,L)
corner_LLL = torch.where(
    (torch.abs(nodes[:, 0] - L) < tol) &
    (torch.abs(nodes[:, 1] - L) < tol) &
    (torch.abs(nodes[:, 2] - L) < tol)
)[0][0].item()

fem.constraints[corner_000, :] = True
fem.displacements[corner_000, :] = 0.0

# Apply tensile force in z-direction
fem.forces[corner_LLL, 2] = 500.0

print(f"\nBoundary conditions:")
print(f"  Fixed corner (0,0,0): node {corner_000}")
print(f"  Loaded corner (L,L,L): node {corner_LLL}")

# Solve
print("\nSolving...")
u, f, stress, _, _ = fem.solve(verbose=True)

print(f"\nSolution:")
print(f"  Max displacement: {u.abs().max():.6e}")
print(f"  Max stress: {stress.abs().max():.3f} MPa")

# Verify periodicity on a few node pairs
print("\nVerifying periodic boundary conditions:")
print("X-direction periodicity (first 3 pairs):")
for i in range(min(3, len(x0_matched))):
    node0 = x0_matched[i].item()
    nodeL = xL_matched[i].item()
    u_diff = u[nodeL] - u[node0]
    print(f"  Pair {i}: u_diff = "
          f"[{u_diff[0]:.6e}, {u_diff[1]:.6e}, {u_diff[2]:.6e}]")

print("\nY-direction periodicity (first 3 valid pairs):")
count = 0
for i in range(len(y0_matched)):
    if not y_mask[i]:
        continue
    node0 = y0_matched[i].item()
    nodeL = yL_matched[i].item()
    u_diff = u[nodeL] - u[node0]
    print(f"  Pair {count}: u_diff = "
          f"[{u_diff[0]:.6e}, {u_diff[1]:.6e}, {u_diff[2]:.6e}]")
    count += 1
    if count >= 3:
        break

print("\nTest completed successfully!")
print("Periodic boundary conditions enforced via master-slave elimination.")
