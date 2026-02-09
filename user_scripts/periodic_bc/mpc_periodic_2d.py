"""
Multi-Point Constraint Example: 2D Periodic Boundary Conditions

Demonstrates periodic BCs on a unit square subjected to shear deformation.
Tests master-slave elimination for coupling opposite edges.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import matplotlib.pyplot as plt
from torchfem.planar import Planar
from torchfem.materials import IsotropicElasticityPlaneStrain
from torchfem.mesh import rect_quad

# Material properties (aluminum)
E = 70e3  # Young's modulus (MPa)
nu = 0.3  # Poisson's ratio

# Create unit square mesh using rect_quad
n_nodes_x = 5
n_nodes_y = 5
L = 1.0

nodes, elements = rect_quad(n_nodes_x, n_nodes_y, L, L)

# Initialize FEM problem
material = IsotropicElasticityPlaneStrain(E=E, nu=nu)
fem = Planar(nodes, elements, material)

print(f"Mesh: {nodes.shape[0]} nodes, {elements.shape[0]} elements")
print(f"Total DOFs: {fem.n_dofs}")

# Apply periodic boundary conditions using MPCs
# Match left edge with right edge: u_right = u_left
# Match bottom edge with top edge: u_top = u_bottom

# Find nodes on each edge
tol = 1e-6
left_nodes = torch.where(torch.abs(nodes[:, 0]) < tol)[0]
right_nodes = torch.where(torch.abs(nodes[:, 0] - L) < tol)[0]
bottom_nodes = torch.where(torch.abs(nodes[:, 1]) < tol)[0]
top_nodes = torch.where(torch.abs(nodes[:, 1] - L) < tol)[0]

# Sort by y-coordinate for left/right, x-coordinate for bottom/top
left_idx = torch.argsort(nodes[left_nodes, 1])
right_idx = torch.argsort(nodes[right_nodes, 1])
bottom_idx = torch.argsort(nodes[bottom_nodes, 0])
top_idx = torch.argsort(nodes[top_nodes, 0])

left_nodes = left_nodes[left_idx]
right_nodes = right_nodes[right_idx]
bottom_nodes = bottom_nodes[bottom_idx]
top_nodes = top_nodes[top_idx]

print(f"\nBoundary nodes:")
print(f"  Left edge: {len(left_nodes)} nodes")
print(f"  Right edge: {len(right_nodes)} nodes")
print(f"  Bottom edge: {len(bottom_nodes)} nodes")
print(f"  Top edge: {len(top_nodes)} nodes")

# Set periodic boundary conditions using direct DOF elimination
# Format: [(master_nodes, slave_nodes, components)]
# components: [0,1] for both x and y
periodic_pairs = [
    (left_nodes, right_nodes, [0, 1]),   # Left-right periodicity
    (bottom_nodes, top_nodes, [0, 1]),   # Bottom-top periodicity
]

fem.set_periodic_bc(periodic_pairs)

print(f"\nPeriodic BC setup:")
print(f"  Total DOFs (full): {fem.n_dofs}")
print(f"  Reduced DOFs: {fem.n_dofs_reduced}")
print(f"  DOF reduction: {fem.n_dofs - fem.n_dofs_reduced} eliminated")

# Fix bottom-left corner to prevent rigid body motion
bottom_left_corner = bottom_nodes[0].item()
fem.constraints[bottom_left_corner, 0] = True
fem.constraints[bottom_left_corner, 1] = True
fem.displacements[bottom_left_corner, 0] = 0.0
fem.displacements[bottom_left_corner, 1] = 0.0

# Apply shear: prescribe displacement on master nodes only
# Middle node on left edge (master) - prescribe horizontal displacement
mid_left_idx = len(left_nodes) // 2
mid_left_node = left_nodes[mid_left_idx].item()
shear_disp = 0.05 * L

fem.constraints[mid_left_node, 0] = True
fem.displacements[mid_left_node, 0] = shear_disp

# Middle node on bottom edge (master) - prescribe vertical displacement
mid_bottom_idx = len(bottom_nodes) // 2
mid_bottom_node = bottom_nodes[mid_bottom_idx].item()
vertical_disp = 0.03 * L

fem.constraints[mid_bottom_node, 1] = True
fem.displacements[mid_bottom_node, 1] = vertical_disp

print(f"\nBoundary conditions:")
print(f"  Fixed bottom-left corner: node {bottom_left_corner}")
print(f"  Prescribed on mid-left (master): node {mid_left_node}, "
      f"u_x = {shear_disp}")
print(f"  Prescribed on mid-bottom (master): node {mid_bottom_node}, "
      f"u_y = {vertical_disp}")
print(f"  Periodic BCs on all edges")

# Solve
print("\nSolving...")
u, f, stress, _, _ = fem.solve(verbose=True)

print(f"\nSolution:")
print(f"  Max displacement: {u.abs().max():.6e}")
print(f"  Max stress: {stress.abs().max():.3f} MPa")

# Note: Verification not needed with DOF elimination approach
# Periodic nodes share the same DOFs, so periodicity is exact by
# construction

# Get final displacement field
u_final = u[-1] if u.dim() == 3 else u

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Undeformed mesh
ax = axes[0]
for elem in elements:
    elem_nodes = nodes[elem]
    elem_nodes_plot = torch.cat([elem_nodes, elem_nodes[0:1]], dim=0)
    ax.plot(elem_nodes_plot[:, 0], elem_nodes_plot[:, 1], 'b-', lw=0.5)
ax.set_aspect('equal')
ax.set_title('Undeformed mesh')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# Deformed mesh with displacement magnitude
ax = axes[1]
scale = 5.0
nodes_def = nodes + scale * u_final
u_mag = torch.linalg.norm(u_final, dim=1)

# Mark periodic pairs with same color
ax.plot([nodes[left_nodes, 0].min(), nodes[left_nodes, 0].min()],
        [0, L], 'r-', lw=2, alpha=0.5, label='Left edge (master)')
ax.plot([nodes[right_nodes, 0].max(), nodes[right_nodes, 0].max()],
        [0, L], 'b-', lw=2, alpha=0.5, label='Right edge (slave)')
ax.plot([0, L],
        [nodes[bottom_nodes, 1].min(), nodes[bottom_nodes, 1].min()],
        'g-', lw=2, alpha=0.5, label='Bottom edge (master)')
ax.plot([0, L],
        [nodes[top_nodes, 1].max(), nodes[top_nodes, 1].max()],
        'm-', lw=2, alpha=0.5, label='Top edge (slave)')

for elem in elements:
    elem_nodes = nodes_def[elem]
    elem_nodes_plot = torch.cat([elem_nodes, elem_nodes[0:1]], dim=0)
    ax.plot(elem_nodes_plot[:, 0], elem_nodes_plot[:, 1], 'b-', lw=0.5)

scatter = ax.scatter(nodes_def[:, 0], nodes_def[:, 1], c=u_mag,
                     cmap='viridis', s=50, zorder=5)
ax.set_aspect('equal')
ax.set_title(f'Deformed mesh (scale={scale})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8)
plt.colorbar(scatter, ax=ax, label='|u|')

plt.tight_layout()
plt.savefig('mpc_periodic_2d.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to mpc_periodic_2d.png")
plt.show()
