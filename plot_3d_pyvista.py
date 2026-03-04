import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt

def main():
    print("Loading 3D data...")
    phi_np = np.load("output_3d/phi_3d_final.npy")
    N_phases, Nx, Ny, Nz = phi_np.shape

    colors = np.array([
        [0.1, 0.1, 0.1],  
        [0.8, 0.2, 0.2],  
        [0.2, 0.8, 0.2],  
        [0.2, 0.2, 0.8],  
        [0.8, 0.8, 0.2],  
    ])

    print("Calculating metrics...")
    rgb_vol = np.zeros((Nx, Ny, Nz, 3), dtype=np.float32)
    for p in range(N_phases):
        for c in range(3):
            rgb_vol[:, :, :, c] += phi_np[p, :, :, :] * colors[p % len(colors)][c]
    rgb_vol = np.clip(rgb_vol, 0, 1)

    grains_vol = np.argmax(phi_np, axis=0).astype(np.float32)
    bounds_vol = np.max(phi_np, axis=0) - np.min(phi_np, axis=0)

    print("Building PyVista mesh...")
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

    print("Rendering PyVista plotter...")
    p = pv.Plotter(shape=(1, 3), window_size=(1800, 600), off_screen=True)

    p.subplot(0, 0)
    p.add_mesh(outline, color="black")
    p.add_mesh(slices, scalars="RGB", rgb=True, show_scalar_bar=False)
    p.add_text("1. RGB Mixed Phase", font_size=14, position='upper_edge')
    p.camera_position = 'iso'

    p.subplot(0, 1)
    p.add_mesh(outline, color="black")
    p.add_mesh(slices, scalars="Grains", cmap="jet", show_scalar_bar=False)
    p.add_text("2. Dominant Phase (Grains)", font_size=14, position='upper_edge')
    p.camera_position = 'iso'

    p.subplot(0, 2)
    p.add_mesh(outline, color="black")
    p.add_mesh(slices, scalars="Boundaries", cmap="bone_r", show_scalar_bar=False)
    p.add_text("3. Phase Boundaries", font_size=14, position='upper_edge')
    p.camera_position = 'iso'

    p.link_views()
    
    out_path = "output_3d/pyvista_isometric_views.png"
    p.screenshot(out_path)
    print(f"Saved PyVista rendering to {out_path}")

if __name__ == "__main__":
    main()
