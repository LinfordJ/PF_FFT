import taichi as ti
import numpy as np
import os
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu, default_fp=ti.f32)

from multiphase_fft.config import SimulationConfig
from multiphase_fft.solver.spectral_solver import SpectralSolver

def init_3d_phases(N, N_phases):
    phi = np.zeros((N_phases, N[0], N[1], N[2]), dtype=np.float32)
    
    phi[0, ...] = 1.0
    
    np.random.seed(42)
    n_grains = N_phases - 1
    
    for p in range(1, N_phases):
        cx = np.random.randint(10, N[0] - 10)
        cy = np.random.randint(10, N[1] - 10)
        cz = np.random.randint(10, N[2] - 10)
        r = np.random.randint(8, 15)
        
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)
                    if dist < r:
                        phi[p, i, j, k] = 1.0
                        phi[0, i, j, k] = 0.0
                        
    for i in range(N[0]):
        for j in range(N[1]):
            for k in range(N[2]):
                sum_p = np.sum(phi[:, i, j, k])
                if sum_p > 0:
                    phi[:, i, j, k] /= sum_p
                else:
                    phi[0, i, j, k] = 1.0
                    
    return phi

def main():
    N_phases = 5
    N = (64, 64, 64)
    
    cfg = SimulationConfig(
        dim=3,
        N=N,
        L=(64.0, 64.0, 64.0),
        N_phases=N_phases,
        kappa=2.0,
        W=1.0,
        U=1.0,
        bulk_energies=[0.0] * N_phases,
        mobility=1.0,
        dt=0.1,
        equation_type="Allen-Cahn"
    )

    solver = SpectralSolver(cfg)

    print("Initializing 3D phases...")
    phi_init = init_3d_phases(N, N_phases)
    solver.phi.from_numpy(phi_init)

    output_dir = "output_3d"
    os.makedirs(output_dir, exist_ok=True)

    n_steps = 100
    save_freq = 20

    print("Starting 3D MPF FFT Simulation...")

    for step in range(n_steps + 1):
        if step % save_freq == 0:
            phi_np = solver.phi.to_numpy()
            
            slice_idx = N[2] // 2
            rendered_slice = np.zeros((N[0], N[1], 3), dtype=np.float32)
            
            colors = [
                [0.1, 0.1, 0.1], 
                [0.8, 0.2, 0.2], 
                [0.2, 0.8, 0.2], 
                [0.2, 0.2, 0.8], 
                [0.8, 0.8, 0.2]  
            ]
            
            for p in range(N_phases):
                for c in range(3):
                    rendered_slice[:, :, c] += phi_np[p, :, :, slice_idx] * colors[p % len(colors)][c]
                    
            rendered_slice = np.clip(rendered_slice, 0, 1)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(rendered_slice, origin='lower')
            plt.title(f"3D Multi-Phase Field Slice Z={slice_idx} (Step {step})")
            plt.axis('off')
            
            filepath = os.path.join(output_dir, f"phi_3d_slice_{step:04d}.png")
            plt.savefig(filepath)
            plt.close()
            print(f"Saved {filepath}")

        if step < n_steps:
            solver.step()

    print("3D Simulation completed successfully.")

if __name__ == "__main__":
    main()
