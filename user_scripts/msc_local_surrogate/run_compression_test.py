import torch
import numpy as np
import os
import sys
import pathlib
import matplotlib.pyplot as plt

# Get the current working directory and build the path
graphorge_path = str(pathlib.Path(os.getcwd()).parents[1] / 
                     "graphorge_material_patches" / "src")
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)

# Add MSC path
msc_path = "/Users/rbarreira/Desktop/machine_learning/msc/src"
if msc_path not in sys.path:
    sys.path.insert(0, msc_path)
# Import MSC model
from minimal_state_cell_c import MSC

from torchfem import Solid
from torchfem.materials import IsotropicPlasticity3D
from torchfem.mesh import cube_hexa
from torchfem.elements import Hexa1r

torch.set_default_dtype(torch.float64)

def run_compression_test():
    # Create unit cube mesh using cube_hexa
    nodes, elements = cube_hexa(Nx=2, Ny=2, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define von Mises plasticity material
    def sigma_f(eps_p):
        """Yield stress as function of plastic strain"""
        sigma_y = 250.0  # Initial yield stress
        H = 1000.0       # Hardening modulus
        return sigma_y + H * eps_p
    
    def sigma_f_prime(eps_p):
        """Derivative of yield stress"""
        H = 1000.0
        return H
    
    material = IsotropicPlasticity3D(
        E=210000.0,
        nu=0.3,
        sigma_f=sigma_f,
        sigma_f_prime=sigma_f_prime
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create solid model
    model = Solid(nodes, elements, material)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary conditions
    # Fix bottom left corner (x=0, y=0, z=0)
    bottom_left_corner = (nodes[:, 0] == 0.0) & (
        nodes[:, 1] == 0.0) & (
        nodes[:, 2] == 0.0)
    model.constraints[bottom_left_corner, :] = True
    # Fix bottom face (z = 0) in z-direction only
    bottom_nodes = nodes[:, 2] == 0.0
    model.constraints[bottom_nodes, 2] = True
    # Apply compression to top face (z = 1.0) - compress by 0.05
    top_nodes = nodes[:, 2] == 1.0
    model.displacements[top_nodes, 2] = -0.05
    model.constraints[top_nodes, 2] = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solve - reference solution
    increments = torch.linspace(0.0, 1.0, 100)
    u_ref, f_ref, _, _, _ = model.solve(rtol=1e-6, atol=1e-6,
                                       increments=increments, nlgeom=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate and load MSC model
    breakpoint()
    msc_model = MSC(input_size=6,
                    output_size=0,
                    output_depth=-1,
                    variables=7,
                    depth=4,
                    width=25,
                    internal_activation='Tanh',
                    activation='Tanh',
                    output_activation='Tanh',
                    return_state=True,
                    return_sequences=False)
    model_path = f'/Users/rbarreira/Desktop/machine_learning/msc/' + \
        f'trained_msc_u7_d4_w25_e1500_batch64_lrinit0.005_lrrate0.03_lrpower0.5'
    msc_model.load_state_dict(torch.load(model_path, weights_only=True))
    msc_model.eval()
    
    # Override element type to use Hexa1r (reduced integration)
    model.etype = Hexa1r()
    # Update integration points
    model.n_int = len(model.etype.iweights())  
    # Solve with MSC
    u_msc, f_msc, _, _, _ = model.solve_msc(
        msc_model=msc_model,
        msc_variables=7,
        increments=increments,
        max_iter=100,
        rtol=1e-8,
        scaler_hydrostatic=1375.297984380115,
        scaler_deviatoric=324.7645473652983,
        verbose=True,
        return_intermediate=True,
        return_volumes=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output results
    output_dir = 'results/cube_compression_msc'
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate forces on bottom face and displacement of top face
    bottom_force_msc = []
    top_displacement = []
    for i in range(len(increments)):
        # Sum forces on bottom nodes (z-direction)
        bottom_force_msc.append(torch.sum(f_msc[i][bottom_nodes, 2]).item())
        
        # Get displacement of first top node
        top_node_idx = torch.where(top_nodes)[0][0]
        top_displacement.append(u_msc[i][top_node_idx, 2].item())
    
    # Plot force vs displacement
    plt.figure(figsize=(6, 6))
    plt.plot(top_displacement, bottom_force_msc, 'r--', label='MSC')
    plt.xlabel('Top Face Displacement')
    plt.ylabel('Sum of Bottom Face Forces (Z-direction)')
    plt.legend()
    plt.savefig(f"{output_dir}/force_displacement_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_compression_test()