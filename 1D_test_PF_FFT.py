import taichi as ti
import numpy as np
import os
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu, default_fp=ti.f32)

from multiphase_fft.config import SimulationConfig
from multiphase_fft.solver.spectral_solver import SpectralSolver

def init_1d_phases(N, N_phases):
    phi = np.zeros((N_phases, N), dtype=np.float32)
    for i in range(N):
        if i < N // 3:
            phi[0, i] = 1.0
        elif i < 2 * N // 3:
            phi[1, i] = 1.0
        else:
            phi[2, i] = 1.0
    return phi

def main():
    N_phases = 3
    N = (256,)
    
    cfg = SimulationConfig(
        dim=1,
        N=N,
        L=(256.0,),
        N_phases=N_phases,
        kappa=2.0,
        W=1.0,
        U=1.0,
        bulk_energies=[0.0, 0.0, 0.0],
        mobility=1.0,
        dt=0.1,
        equation_type="Allen-Cahn"
    )

    solver = SpectralSolver(cfg)

    phi_init = init_1d_phases(N[0], N_phases)
    solver.phi.from_numpy(phi_init)

    output_dir = "output_1d"
    os.makedirs(output_dir, exist_ok=True)

    n_steps = 1000
    save_freq = 100

    print("Starting 1D MPF FFT Simulation...")

    for step in range(n_steps + 1):
        if step % save_freq == 0:
            phi_np = solver.phi.to_numpy()
            
            plt.figure(figsize=(8, 4))
            for p in range(N_phases):
                plt.plot(phi_np[p, :], label=f"Phase {p}")
            plt.ylim(-0.1, 1.1)
            plt.xlabel("Position (x)")
            plt.ylabel("Phase Fraction")
            plt.title(f"1D Multi-Phase Field FFT (Step {step})")
            plt.legend()
            plt.grid(True)
            
            filepath = os.path.join(output_dir, f"phi_1d_{step:04d}.png")
            plt.savefig(filepath)
            plt.close()
            print(f"Saved {filepath}")

        if step < n_steps:
            solver.step()

    print("1D Simulation completed successfully.")

if __name__ == "__main__":
    main()
