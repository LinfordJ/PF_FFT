import taichi as ti
import numpy as np
from multiphase_fft.config import SimulationConfig
from multiphase_fft.math_utils.taichi_fft import fft_1d_batched, ifft_1d_batched, fft_2d_batched, ifft_2d_batched, fft_3d_batched, ifft_3d_batched
from multiphase_fft.physics.interpolation import PolynomialInterpolation
from multiphase_fft.physics.energy import MultiphaseEnergy

@ti.data_oriented
class SpectralSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.dim = self.cfg.dim
        self.N_phases = self.cfg.N_phases
        self.N = self.cfg.N
        
        self.dx = [self.cfg.L[i] / self.cfg.N[i] for i in range(self.dim)]
        
        self.interp = PolynomialInterpolation()
        self.energy = MultiphaseEnergy(W=self.cfg.W, U=self.cfg.U)
        
        shape = self.N
        
        self.phi = ti.field(dtype=ti.f32, shape=(self.N_phases, *shape))
        self.df_dphi = ti.field(dtype=ti.f32, shape=(self.N_phases, *shape))
        self.bulk_energies = ti.field(dtype=ti.f32, shape=(self.N_phases,))
        
        self.bulk_energies.from_numpy(np.array(self.cfg.bulk_energies, dtype=np.float32))
        
        self.phi_work_k = ti.math.vec2.field(shape=(self.N_phases, *shape))
        self.df_work_k = ti.math.vec2.field(shape=(self.N_phases, *shape))
        self.fft_buffer = ti.math.vec2.field(shape=(self.N_phases, *shape))
        
        self.k2 = ti.field(dtype=ti.f32, shape=shape)
        self.mobility = ti.field(dtype=ti.f32, shape=(self.N_phases,))
        self.mobility.from_numpy(np.array([self.cfg.mobility] * self.N_phases, dtype=np.float32))
        
        self.setup_k2()

    def setup_k2(self):
        k2_np = np.zeros(self.N, dtype=np.float32)
        if self.dim == 1:
            kx = np.fft.fftfreq(self.N[0], d=self.dx[0]) * 2 * np.pi
            k2_np = (kx**2).astype(np.float32)
        elif self.dim == 2:
            kx = np.fft.fftfreq(self.N[0], d=self.dx[0]) * 2 * np.pi
            ky = np.fft.fftfreq(self.N[1], d=self.dx[1]) * 2 * np.pi
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k2_np = (KX**2 + KY**2).astype(np.float32)
        elif self.dim == 3:
            kx = np.fft.fftfreq(self.N[0], d=self.dx[0]) * 2 * np.pi
            ky = np.fft.fftfreq(self.N[1], d=self.dx[1]) * 2 * np.pi
            kz = np.fft.fftfreq(self.N[2], d=self.dx[2]) * 2 * np.pi
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            k2_np = (KX**2 + KY**2 + KZ**2).astype(np.float32)
        self.k2.from_numpy(k2_np)

    @ti.kernel
    def compute_df_and_load(self):
        for I in ti.grouped(ti.ndrange(*self.N)):
            # Precompute sums needed for Steinbach forces
            sum_abs = 0.0
            sum_sq_abs = 0.0
            
            for i in range(self.N_phases):
                p_val = self.phi[i, I]
                abs_val = ti.abs(p_val)
                sum_abs += abs_val
                sum_sq_abs += abs_val * abs_val

            # First pass: compute raw force for each phase
            sum_force = 0.0
            for p in range(self.N_phases):
                phi_p = self.phi[p, I]
                force = self.energy.compute_force(phi_p, sum_abs, sum_sq_abs)
                self.df_dphi[p, I] = force
                sum_force += force

            # Second pass: apply Steinbach coupling and add bulk term
            for p in range(self.N_phases):
                phi_p = self.phi[p, I]
                force = self.df_dphi[p, I]
                effective_force = force - sum_force / float(self.N_phases)
                df_bulk = self.interp.dh(phi_p) * self.bulk_energies[p]
                df = -effective_force + df_bulk
                self.df_dphi[p, I] = df
                self.phi_work_k[p, I] = ti.math.vec2(phi_p, 0.0)
                self.df_work_k[p, I] = ti.math.vec2(df, 0.0)

    @ti.kernel
    def update_work_k_allen_cahn(self, dt: ti.f32, kappa: ti.f32, mobility: ti.f32):
        for I in ti.grouped(ti.ndrange(*self.N)):
            for i in range(self.N_phases):
                m = self.mobility[i]
                denominator = 1.0 + dt * m * kappa * self.k2[I]
                self.phi_work_k[i, I] = (self.phi_work_k[i, I] - dt * m * self.df_work_k[i, I]) / denominator

    @ti.kernel
    def update_work_k_cahn_hilliard(self, dt: ti.f32, kappa: ti.f32, mobility: ti.f32):
        for I in ti.grouped(ti.ndrange(*self.N)):
            k2 = self.k2[I]
            for i in range(self.N_phases):
                m = self.mobility[i]
                denominator = 1.0 + dt * m * kappa * (k2 * k2)
                self.phi_work_k[i, I] = (self.phi_work_k[i, I] - dt * m * k2 * self.df_work_k[i, I]) / denominator

    @ti.kernel
    def save_work_k_and_project(self):
        for I in ti.grouped(ti.ndrange(*self.N)):
            sum_phi = 0.0
            for i in range(self.N_phases):
                val = self.phi_work_k[i, I][0]
                self.phi[i, I] = val
                sum_phi += val
            if sum_phi != 0.0:
                inv_sum = 1.0 / sum_phi
                for i in range(self.N_phases):
                    self.phi[i, I] *= inv_sum

    def step(self):
        self.compute_df_and_load()
        
        if self.dim == 1:
            fft_1d_batched(self.phi_work_k, self.fft_buffer)
            fft_1d_batched(self.df_work_k, self.fft_buffer)
        elif self.dim == 2:
            fft_2d_batched(self.phi_work_k, self.fft_buffer)
            fft_2d_batched(self.df_work_k, self.fft_buffer)
        elif self.dim == 3:
            fft_3d_batched(self.phi_work_k, self.fft_buffer)
            fft_3d_batched(self.df_work_k, self.fft_buffer)
            
        if self.cfg.equation_type == "Allen-Cahn":
            self.update_work_k_allen_cahn(self.cfg.dt, self.cfg.kappa, self.cfg.mobility)
        else:
            self.update_work_k_cahn_hilliard(self.cfg.dt, self.cfg.kappa, self.cfg.mobility)
            
        if self.dim == 1:
            ifft_1d_batched(self.phi_work_k, self.fft_buffer)
        elif self.dim == 2:
            ifft_2d_batched(self.phi_work_k, self.fft_buffer)
        elif self.dim == 3:
            ifft_3d_batched(self.phi_work_k, self.fft_buffer)
            
        self.save_work_k_and_project()
