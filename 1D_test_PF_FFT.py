import taichi as ti
import numpy as np
import os
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu,fast_math=True)
# ti.init(arch=ti.cpu, default_fp=ti.f32)

from multiphase_fft.config import SimulationConfig
from multiphase_fft.solver.spectral_solver import SpectralSolver

def init_1d_phases(N, N_phases, mode="tanh", eta=2.0):
    phi = np.zeros((N_phases, N), dtype=np.float32)
    
    if mode == "step":
        for i in range(N):
            if i < N // 3:
                phi[0, i] = 1.0
            elif i < 2 * N // 3:
                phi[1, i] = 1.0
            else:
                phi[2, i] = 1.0
    elif mode == "tanh":
        x = np.arange(N)
        
        def tanh_box(x_arr, center, width, interface_eta):
            left = center - width / 2.0
            right = center + width / 2.0
            return 0.5 * (np.tanh((x_arr - left) / (np.sqrt(2) * interface_eta)) - 
                          np.tanh((x_arr - right) / (np.sqrt(2) * interface_eta)))
        
        phi[0, :] = tanh_box(x, 3.0 * N / 8.0, N / 4.0, eta)
        phi[2, :] = tanh_box(x, 5.0 * N / 8.0, N / 4.0, eta)
        phi[1, :] = 1.0 - phi[0, :] - phi[2, :]
        
        phi = np.clip(phi, 0.0, 1.0)
        sum_phi = np.sum(phi, axis=0)
        phi /= (sum_phi + 1e-10)
    elif mode == "uniform_noise":
        phi = np.ones((N_phases, N), dtype=np.float32) / N_phases
        np.random.seed(42)
        fluctuation = 0.01
        noise = np.random.uniform(-fluctuation, fluctuation, size=(N_phases, N)).astype(np.float32)
        phi += noise
        
        sum_phi = np.sum(phi, axis=0)
        phi /= (sum_phi + 1e-10)
        
    return phi

def main():
    N_phases = 3
    N = (256,)
    
    cfg = SimulationConfig(
        dim=1,
        N=N,
        L=(25.6,),
        N_phases=N_phases,
        kappa=4.0,
        W=64.0,
        U=192.0,
        bulk_energies=[0.0, 0.0, 0.0],
        mobility=0.02,
        dt=0.01,
        equation_type="Allen-Cahn"
    )

    solver = SpectralSolver(cfg)

    # mode can be "step" or "tanh"
    phi_init = init_1d_phases(N[0], N_phases, mode="uniform_noise", eta=3.0)
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
