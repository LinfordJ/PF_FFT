import taichi as ti
import numpy as np
import os
from multiphase_fft.config import SimulationConfig
from multiphase_fft.solver.spectral_solver import SpectralSolver
from multiphase_fft.visualization.gui import PhaseFieldGUI

@ti.kernel
def init_voronoi_phi(phi: ti.template(), N_phases: ti.i32, Nx: ti.i32, Ny: ti.i32, 
                     seed_x: ti.types.ndarray(), seed_y: ti.types.ndarray(), seed_phase: ti.types.ndarray()):
    num_seeds = seed_x.shape[0]
    for I in ti.grouped(ti.ndrange(*phi.shape[1:])):
        x, y = I[0], I[1]
        
        min_dist = 1e10
        closest_phase = 0
        
        for s in range(num_seeds):
            dx = x - seed_x[s]
            dy = y - seed_y[s]
            if dx > Nx / 2: dx -= Nx
            if dx < -Nx / 2: dx += Nx
            if dy > Ny / 2: dy -= Ny
            if dy < -Ny / 2: dy += Ny
            
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                closest_phase = seed_phase[s]
                
        for i in range(N_phases):
            if i == closest_phase:
                phi[i, I] = 1.0
            else:
                phi[i, I] = 0.0

import scipy.ndimage

def init_random_noise_phi(phi_field, N_phases, Nx, Ny, fluctuation):
    noise_fields = np.zeros((N_phases, Nx, Ny), dtype=np.float32)
    for p in range(N_phases):
        raw_noise = np.random.rand(Nx, Ny)
        noise_fields[p] = scipy.ndimage.gaussian_filter(raw_noise, sigma=4.0)

    dominant_phase = np.argmax(noise_fields, axis=0)
    
    phi_np = np.zeros((N_phases, Nx, Ny), dtype=np.float32)
    for p in range(N_phases):
        phi_np[p] = (dominant_phase == p).astype(np.float32)
        phi_np[p] = scipy.ndimage.gaussian_filter(phi_np[p], sigma=1.0)

    sum_phi = np.sum(phi_np, axis=0)
    for p in range(N_phases):
        phi_np[p] /= (sum_phi + 1e-10)
        
    phi_field.from_numpy(phi_np)

def init_uniform_noise_phi(phi_field, N_phases, Nx, Ny, fluctuation):
    phi_np = np.ones((N_phases, Nx, Ny), dtype=np.float32) / N_phases
    # add uniform noise in [-fluctuation, fluctuation]
    noise = np.random.uniform(-fluctuation, fluctuation, size=(N_phases, Nx, Ny)).astype(np.float32)
    phi_np += noise
    
    # normalize
    sum_phi = np.sum(phi_np, axis=0)
    for p in range(N_phases):
        phi_np[p] /= sum_phi
        
    phi_field.from_numpy(phi_np)

def main():
    ti.init(arch=ti.gpu)
    
    # Parameters mirror polycrystal_MPF_new.py for stability and realistic grain growth.
    config = SimulationConfig(
        dim=2,
        N=(128, 128),
        L=(12.8, 12.8),
        N_phases=20,             
        dt=0.01,
        kappa=4.0,
        W=64.0,
        U=192.0,
        mobility=0.02,
        bulk_energies=[0.0] * 20, 
        equation_type="Allen-Cahn", 
        max_steps=50000
    )
    
    solver = SpectralSolver(config)
    
    # Choose initialization method: "random_noise", "voronoi" or "uniform_noise"
    init_method = "uniform_noise"
    
    if init_method == "voronoi":
        num_seeds = 30
        seed_x = (np.random.rand(num_seeds) * config.N[0]).astype(np.float32)
        seed_y = (np.random.rand(num_seeds) * config.N[1]).astype(np.float32)
        seed_phase = np.random.randint(0, config.N_phases, size=num_seeds).astype(np.int32)
        init_voronoi_phi(solver.phi, config.N_phases, config.N[0], config.N[1], seed_x, seed_y, seed_phase)
    elif init_method == "random_noise":
        init_random_noise_phi(solver.phi, config.N_phases, config.N[0], config.N[1], 0.3)
    elif init_method == "uniform_noise":
        init_uniform_noise_phi(solver.phi, config.N_phases, config.N[0], config.N[1], 0.01)
    
    gui = PhaseFieldGUI(config, solver)
    
    os.makedirs("output", exist_ok=True)
    
    # Run the simulation loop
    step = 0
    time_val = 0.0
    while gui.running and step < config.max_steps:
        solver.step()
        time_val += config.dt
        if step % 200 == 0:
            filename = f"output/frame_combined_{step:05d}.png"
            gui.render(filename=filename, step=step, time_val=time_val)
            print(f"Saved step {step} combined output")
        elif step % 5 == 0:
            gui.render(step=step, time_val=time_val)
            
        step += 1

if __name__ == "__main__":
    main()
