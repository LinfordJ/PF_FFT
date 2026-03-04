import taichi as ti
import numpy as np
import os
import pyvista as pv

ti.init(arch=ti.cpu, default_fp=ti.f32)

from multiphase_fft.config import SimulationConfig
from multiphase_fft.solver.spectral_solver import SpectralSolver

def init_uniform_noise_phi_3d(N, N_phases, fluctuation):
    phi_np = np.ones((N_phases, N[0], N[1], N[2]), dtype=np.float32) / N_phases
    noise = np.random.uniform(-fluctuation, fluctuation, size=(N_phases, N[0], N[1], N[2])).astype(np.float32)
    phi_np += noise
    
    sum_phi = np.sum(phi_np, axis=0)
    for p in range(N_phases):
        phi_np[p] /= sum_phi
        
    return phi_np

def render_pyvista_step(phi_np, step, output_dir):
    N_phases, Nx, Ny, Nz = phi_np.shape

    colors = np.array([
        [0.1, 0.1, 0.1],  
        [0.8, 0.2, 0.2],  
        [0.2, 0.8, 0.2],  
        [0.2, 0.2, 0.8],  
        [0.8, 0.8, 0.2],
        [0.2, 0.8, 0.8],
        [0.8, 0.2, 0.8],
        [1.0, 0.5, 0.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.5, 1.0],
    ])

    rgb_vol = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    for p in range(N_phases):
        for c in range(3):
            rgb_vol[:, :, :, c] += phi_np[p, :, :, :] * colors[p % len(colors)][c]
    rgb_vol = np.clip(rgb_vol, 0, 1)

    grains_vol = np.argmax(phi_np, axis=0).astype(np.float32)
    bounds_vol = np.max(phi_np, axis=0) - np.min(phi_np, axis=0)

    grid = pv.ImageData()
    grid.dimensions = (Nx, Ny, Nz)
    grid.spacing = (1, 1, 1)

    grid.point_data["RGB"] = (rgb_vol.reshape(-1, 3, order='F') * 255).astype(np.uint8)
    grid.point_data["Grains"] = grains_vol.flatten(order='F')
    grid.point_data["Boundaries"] = bounds_vol.flatten(order='F')

    slices = grid.slice_orthogonal(x=Nx//2, y=Ny//2, z=Nz//2)
    outline = grid.outline()

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'

    p = pv.Plotter(shape=(1, 3), window_size=(1800, 600), off_screen=True)

    p.subplot(0, 0)
    p.add_mesh(outline, color="black")
    p.add_mesh(slices, scalars="RGB", rgb=True, show_scalar_bar=False)
    p.add_text(f"1. RGB (Step {step})", font_size=14, position='upper_edge')
    p.camera_position = 'iso'

    p.subplot(0, 1)
    p.add_mesh(outline, color="black")
    p.add_mesh(slices, scalars="Grains", cmap="jet", show_scalar_bar=False)
    p.add_text(f"2. Grains (Step {step})", font_size=14, position='upper_edge')
    p.camera_position = 'iso'

    p.subplot(0, 2)
    p.add_mesh(outline, color="black")
    p.add_mesh(slices, scalars="Boundaries", cmap="bone_r", show_scalar_bar=False)
    p.add_text(f"3. Boundaries (Step {step})", font_size=14, position='upper_edge')
    p.camera_position = 'iso'

    p.link_views()
    
    filepath = os.path.join(output_dir, f"phi_3d_step_{step:04d}.png")
    p.screenshot(filepath)
    p.close()
    print(f"Saved {filepath}")

def main():
    N_phases = 8
    N = (64, 64, 64)
    
    cfg = SimulationConfig(
        dim=3,
        N=N,
        L=(64.0, 64.0, 64.0),
        N_phases=N_phases,
        kappa=4.0,
        W=64.0,
        U=192.0,
        bulk_energies=[0.0] * N_phases,
        mobility=0.02,
        dt=0.01,
        equation_type="Allen-Cahn"
    )

    solver = SpectralSolver(cfg)

    print("Initializing 3D phases with uniform noise...")
    phi_init = init_uniform_noise_phi_3d(N, N_phases, fluctuation=0.01)
    solver.phi.from_numpy(phi_init)

    output_dir = "output_3d"
    os.makedirs(output_dir, exist_ok=True)

    n_steps = 200
    save_freq = 50

    print("Starting 3D MPF FFT Simulation...")

    for step in range(n_steps + 1):
        if step % save_freq == 0:
            phi_np = solver.phi.to_numpy()
            render_pyvista_step(phi_np, step, output_dir)

        if step < n_steps:
            solver.step()

    print("3D Simulation completed successfully.")
    
    np.save(os.path.join(output_dir, "phi_3d_final.npy"), solver.phi.to_numpy())
    print("Saved final 3D numpy array to output_3d/phi_3d_final.npy")

if __name__ == "__main__":
    main()
