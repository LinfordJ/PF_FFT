import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import sys
sys.path.append(".")
from multiphase_fft.config import SimulationConfig
from multiphase_fft.solver.spectral_solver import SpectralSolver

@ti.data_oriented
class FDSolver2D:
    def __init__(self, N_phases, Nx, Ny, dx, dt, kappa, u, u_3phase, mobility):
        self.N_phases = N_phases
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dt = dt
        self.kappa = kappa
        self.u = u
        self.u_3phase = u_3phase
        self.mobility = mobility

        self.phi = ti.field(dtype=ti.f32, shape=(N_phases, Nx, Ny))
        self.laplacian = ti.field(dtype=ti.f32, shape=(N_phases, Nx, Ny))
        self.forces = ti.field(dtype=ti.f32, shape=(N_phases, Nx, Ny))

    @ti.func
    def get_neighbors(self, i, j):
        im = i - 1 if i - 1 >= 0 else self.Nx - 1
        jm = j - 1 if j - 1 >= 0 else self.Ny - 1
        ip = i + 1 if i + 1 < self.Nx else 0
        jp = j + 1 if j + 1 < self.Ny else 0
        return im, jm, ip, jp

    @ti.kernel
    def compute_laplacian(self):
        for p, i, j in self.phi:
            im, jm, ip, jp = self.get_neighbors(i, j)
            # Correct isotropic 9-point stencil (Ohta scheme)
            # Weights: 2/3 for nearest, 1/6 for next-nearest, -10/3 for center
            lap = (
                (1.0 / 6.0) * (self.phi[p, im, jm] + self.phi[p, ip, jm] + self.phi[p, im, jp] + self.phi[p, ip, jp]) +
                (2.0 / 3.0) * (self.phi[p, im, j] + self.phi[p, ip, j] + self.phi[p, i, jm] + self.phi[p, i, jp]) -
                (10.0 / 3.0) * self.phi[p, i, j]
            ) / (self.dx * self.dx)
            self.laplacian[p, i, j] = lap

    @ti.kernel
    def advance(self):
        for i, j in ti.ndrange(self.Nx, self.Ny):
            sum_force = 0.0
            for p in range(self.N_phases):
                penalty_force = 0.0
                for k1 in range(self.N_phases):
                    for l in range(k1 + 1, self.N_phases):
                        if k1 != p and l != p:
                            penalty_force += ti.abs(self.phi[k1, i, j] * self.phi[l, i, j])
                penalty_force *= ti.math.sign(self.phi[p, i, j])
                penalty_force = -self.u_3phase * penalty_force

                chemical_force = 0.0
                for k1 in range(self.N_phases):
                    if k1 != p:
                        chemical_force += ti.abs(self.phi[k1, i, j])
                chemical_force *= ti.math.sign(self.phi[p, i, j])
                chemical_force = -self.u * chemical_force

                f = chemical_force + self.kappa * self.laplacian[p, i, j] + penalty_force
                self.forces[p, i, j] = f
                sum_force += f

            sum_phi = 0.0
            for p in range(self.N_phases):
                effective_force = self.forces[p, i, j] - sum_force / float(self.N_phases)
                val = self.phi[p, i, j] + effective_force * self.mobility * self.dt
                self.phi[p, i, j] = val
                sum_phi += val

            if sum_phi != 0.0:
                inv_sum = 1.0 / sum_phi
                for p in range(self.N_phases):
                    self.phi[p, i, j] *= inv_sum

    def step(self):
        self.compute_laplacian()
        self.advance()

def generate_initial_phi(N_phases, Nx, Ny, fluctuation):
    np.random.seed(42)
    phi_np = np.ones((N_phases, Nx, Ny), dtype=np.float32) / N_phases
    noise = np.random.uniform(-fluctuation, fluctuation, size=(N_phases, Nx, Ny)).astype(np.float32)
    phi_np += noise
    sum_phi = np.sum(phi_np, axis=0)
    for p in range(N_phases):
        phi_np[p] /= sum_phi
    return phi_np

def calculate_metrics(phi_fd, phi_fft):
    mse = np.mean((phi_fd - phi_fft) ** 2)
    max_diff = np.max(np.abs(phi_fd - phi_fft))
    return mse, max_diff

def plot_comparison(phi_fd, phi_fft, step, output_dir):
    grain_fd = np.argmax(phi_fd, axis=0)
    grain_fft = np.argmax(phi_fft, axis=0)
    
    bound_fd = np.max(phi_fd, axis=0) - np.min(phi_fd, axis=0)
    bound_fft = np.max(phi_fft, axis=0) - np.min(phi_fft, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Comparison at Step {step}", fontsize=16)

    axs[0, 0].imshow(grain_fd, cmap='jet', interpolation='nearest')
    axs[0, 0].set_title("FD: Grains")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(grain_fft, cmap='jet', interpolation='nearest')
    axs[0, 1].set_title("FFT: Grains")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(bound_fd, cmap='gray', interpolation='nearest')
    axs[1, 0].set_title("FD: Boundaries")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(bound_fft, cmap='gray', interpolation='nearest')
    axs[1, 1].set_title("FFT: Boundaries")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"compare_{step:04d}.png"))
    plt.close()

def main():
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    
    Nx, Ny = 128, 128
    L = 12.8
    dx = L / Nx
    N_phases = 10
    dt = 0.005
    kappa = 4.0
    u = 64.0
    u_3phase = 3.0 * u
    mobility = 0.02
    max_steps = 1000
    
    output_dir = "comparison_output"
    os.makedirs(output_dir, exist_ok=True)

    fd_solver = FDSolver2D(N_phases, Nx, Ny, dx, dt, kappa, u, u_3phase, mobility)

    fft_config = SimulationConfig(
        dim=2, N=(Nx, Ny), L=(L, L), N_phases=N_phases, dt=dt,
        kappa=kappa, W=u, U=u_3phase, mobility=mobility,
        bulk_energies=[0.0] * N_phases, equation_type="Allen-Cahn"
    )
    fft_solver = SpectralSolver(fft_config)

    phi_init = generate_initial_phi(N_phases, Nx, Ny, 0.01)
    fd_solver.phi.from_numpy(phi_init)
    fft_solver.phi.from_numpy(phi_init)

    print("Starting Comparison Loop...")
    print(f"{'Step':>6} | {'MSE':>12} | {'Max Diff':>12} | {'FD Time (s)':>12} | {'FFT Time (s)':>12}")
    print("-" * 65)

    metrics_log = []
    
    total_fd_time = 0.0
    total_fft_time = 0.0

    for step in range(max_steps + 1):
        if step % 100 == 0:
            phi_fd_np = fd_solver.phi.to_numpy()
            phi_fft_np = fft_solver.phi.to_numpy()
            
            mse, max_diff = calculate_metrics(phi_fd_np, phi_fft_np)
            print(f"{step:6d} | {mse:12.6f} | {max_diff:12.6f} | {total_fd_time:12.4f} | {total_fft_time:12.4f}")
            
            metrics_log.append((step, mse, max_diff))
            plot_comparison(phi_fd_np, phi_fft_np, step, output_dir)

        if step < max_steps:
            t0 = time.time()
            fd_solver.step()
            ti.sync()
            t1 = time.time()
            total_fd_time += (t1 - t0)
            
            t2 = time.time()
            fft_solver.step()
            ti.sync()
            t3 = time.time()
            total_fft_time += (t3 - t2)

    steps, mses, max_diffs = zip(*metrics_log)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, mses, label='Mean Squared Error', marker='o')
    plt.plot(steps, max_diffs, label='Max Difference', marker='x')
    plt.yscale('log')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Difference (log scale)')
    plt.title('FD vs FFT Deviation over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "metrics.png"))
    plt.close()
    
    print(f"Comparison complete. Results saved to '{output_dir}'.")

if __name__ == "__main__":
    main()