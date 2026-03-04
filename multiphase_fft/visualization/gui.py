import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from multiphase_fft.config import SimulationConfig

@ti.data_oriented
class PhaseFieldGUI:
    """
    Visualizes the multi-phase simulation in real-time.
    Maps N phases to a 2D RGB image buffer, taking a z-slice for 3D simulations.
    Supports multiple visualization modes.
    """
    def __init__(self, config: SimulationConfig, solver):
        self.cfg = config
        self.solver = solver
        self.dim = self.cfg.dim
        self.N_phases = self.cfg.N_phases
        self.Nx = self.cfg.N[0]
        self.Ny = self.cfg.N[1]
        
        # Take center slice for 3D visualization
        self.z_slice = self.cfg.N[2] // 2 if self.dim == 3 else 0
        
        # Predefined distinct colors for RGB continuous phase mixing
        color_palette = [
            [1.0, 0.0, 0.0], # Red
            [0.0, 1.0, 0.0], # Green
            [0.0, 0.0, 1.0], # Blue
            [1.0, 1.0, 0.0], # Yellow
            [0.0, 1.0, 1.0], # Cyan
            [1.0, 0.0, 1.0]  # Magenta
        ]
        
        # Ensure we have enough colors or wrap around
        extended_palette = [color_palette[i % len(color_palette)] for i in range(self.N_phases)]
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=(self.N_phases,))
        self.colors.from_numpy(np.array(extended_palette, dtype=np.float32))
        
        # Jet colormap for discrete grains visualization
        jet = plt.get_cmap('jet')
        jet_colors = jet(np.linspace(0, 1, self.N_phases))[:, :3]
        self.jet_colors = ti.Vector.field(3, dtype=ti.f32, shape=(self.N_phases,))
        self.jet_colors.from_numpy(jet_colors.astype(np.float32))
        
        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.Nx * 3, self.Ny))
        
        self.window_name = "Multi-Phase (RGB | Grains | Boundaries)"
        self.gui_res = (self.Nx * 3, self.Ny)
        self.gui = ti.GUI(self.window_name, res=self.gui_res)

    @ti.kernel
    def render_all_modes(self):
        for i, j in ti.ndrange(self.Nx, self.Ny):
            max_val = -1e10
            min_val = 1e10
            max_idx = 0
            color_rgb = ti.Vector([0.0, 0.0, 0.0])
            
            for p in range(self.N_phases):
                val = 0.0
                if ti.static(self.dim == 2):
                    val = self.solver.phi[p, i, j]
                else:
                    val = self.solver.phi[p, i, j, ti.cast(self.z_slice, ti.i32)]
                
                clamped_val = ti.max(0.0, ti.min(1.0, val))
                color_rgb += clamped_val * self.colors[p]
                
                if val > max_val: 
                    max_val = val
                    max_idx = p
                if val < min_val: 
                    min_val = val
                    
            self.image_buffer[i, j] = color_rgb
            self.image_buffer[i + self.Nx, j] = self.jet_colors[max_idx]
            
            gap = max_val - min_val
            self.image_buffer[i + 2 * self.Nx, j] = ti.Vector([gap, gap, gap])

    def render(self, filename: str = None, step: int = None, time_val: float = None):
        self.render_all_modes()
        self.gui.set_image(self.image_buffer)
        
        info_text = ""
        if step is not None:
            info_text += f"Step: {step} "
        if time_val is not None:
            info_text += f"| Time: {time_val:.3f}"
            
        if info_text:
            self.gui.text(content=info_text, pos=(0.01, 0.98), font_size=16, color=0xFFFFFF)
            
        self.gui.text(content="RGB", pos=(0.15, 0.05), font_size=16, color=0xFFFFFF)
        self.gui.text(content="Grains", pos=(0.48, 0.05), font_size=16, color=0xFFFFFF)
        self.gui.text(content="Boundaries", pos=(0.80, 0.05), font_size=16, color=0x000000)
            
        if filename:
            self.gui.show(filename)
        else:
            self.gui.show()
        
    @property
    def running(self):
        return self.gui.running
