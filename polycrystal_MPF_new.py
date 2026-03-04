"""generate the evoulution process of polycrystal
   polycrystal using multi-phase field model of Steinbach
   (which is a generalization for Ginzburg-Landau equation in multi-phase problems)"""
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt 
from matplotlib import cm
#import pyvista as pv
import sys
import os


@ti.data_oriented
class Polycrystal():

    def __init__(self) -> None:
        # parameters for geometric model
        self.phase_nums = 20
        self.nx = 80; self.ny = 80; self.nz = 80
        self.dt = 0.01
        self.dx = 0.1; self.dy = 0.1; self.dz = 0.1
        self.kappa = 4.  # 4.  # gradient coefficient
        self.u = 64.  # energy barrier, chemical energy coefficient
        self.u_3phase_penalty = 3. * self.u  # energy penalty for 3 phases coexistence
        self.fluctuation = 0.01
        self.mobility = 0.02

        self.phi = ti.Vector.field(self.phase_nums, ti.f64, shape=(self.nx, self.ny, self.nz))
        self.laplacian = ti.Vector.field(self.phase_nums, ti.f64, shape=(self.nx, self.ny, self.nz))

    @ti.kernel
    def initialize(self, ):
        for I in ti.grouped(self.phi):

            for p in range(self.phi[I].n):
                self.phi[I][p] = 1. / self.phase_nums
                
            ### set some initial perturbation for phi
            for p in range(self.phi[I].n):
                self.phi[I][p] = self.phi[I][p] \
                    - self.fluctuation + 2.*self.fluctuation * ti.random(ti.f64)
            ### normalization
            phi_sum = 0.0
            for p in range(self.phi[I].n):
                phi_sum += self.phi[I][p]
            for p in range(self.phi[I].n):
                self.phi[I][p] = self.phi[I][p]/phi_sum
    

    @ti.func
    def neighbor_index(self, i, j, k):
        """
            use periodic boundary condition to get neighbor index
        """
        im = i - 1 if i - 1 >= 0 else self.nx - 1
        jm = j - 1 if j - 1 >= 0 else self.ny - 1
        km = k - 1 if k - 1 >= 0 else self.nz - 1
        ip = i + 1 if i + 1 < self.nx else 0
        jp = j + 1 if j + 1 < self.ny else 0
        kp = k + 1 if k + 1 < self.nz else 0
        return im, jm, km, ip, jp, kp


    @ti.kernel
    def advance(self, ):  # advance a time step
        phi, laplacian, dx, dy, dz, dt, mobility = ti.static(
            self.phi, self.laplacian, self.dx, self.dy, self.dz, self.dt, self.mobility)
        
        ### compute the laplacian
        for i, j, k in phi:
            im, jm, km, ip, jp, kp = self.neighbor_index(i, j, k)
            laplacian[i,j,k] = ( # laplacian of phi
            2*phi[im,jp,kp] + 3*phi[i,jp,kp] + 2*phi[ip,jp,kp] + 3*phi[im,jp,k] + 6*phi[i,jp,k] + 3*phi[ip,jp,k] + 2*phi[im,jp,km] + 3*phi[i,jp,km] + 2*phi[ip,jp,km] +
            3*phi[im,j,kp] + 6*phi[i,j,kp] + 3*phi[ip,j,kp] + 6*phi[im,j,k] + (-88)*phi[i,j,k] + 6*phi[ip,j,k] + 3*phi[im,j,km] + 6*phi[i,j,km] + 3*phi[ip,j,km]+
            2*phi[im,jm,kp] + 3*phi[i,jm,kp] + 2*phi[ip,jm,kp] + 3*phi[im,jm,k] + 6*phi[i,jm,k] + 3*phi[ip,jm,k] + 2*phi[im,jm,km] + 3*phi[i,jm,km] + 2*phi[ip,jm,km]
            )/(26*dx*dx)



        ### compute evolution and advance a time step
        for i, j, k in phi:
            forces = ti.Vector([0. for _ in range(phi[i, j, k].n)])  # forces for different phases
            for p in range(phi[i, j, k].n):

                ### penalty force
                penalty_force = 0.
                for k1 in range(phi[i, j, k].n):
                    for l in range(k1+1, phi.n):
                        if k1 != p and l != p:
                            penalty_force = penalty_force + \
                                ti.abs(phi[i, j, k][k1] * phi[i, j, k][l])
                penalty_force *= ti.math.sign(phi[i, j, k][p])
                penalty_force = -self.u_3phase_penalty * penalty_force

                ### chemical energy force
                chemical_force = 0.
                for k1 in range(phi[i, j, k].n):
                    if k1 != p:
                        chemical_force = chemical_force + ti.abs(phi[i, j, k][k1])
                chemical_force *= ti.math.sign(phi[i, j, k][p])
                chemical_force = -self.u * chemical_force

                ### compute forces
                forces[p] = (
                    chemical_force + self.kappa * laplacian[i, j, k][p] + penalty_force
                )
            
            ### update phi by Steinbach's equation
            for p in range(phi[i, j, k].n):
                effective_force = 0.
                for k1 in range(phi[i, j, k].n):
                    if k1 != p:
                        effective_force = effective_force + forces[p] - forces[k1]
                phi[i, j, k][p] = phi[i, j, k][p] + effective_force/self.phase_nums * mobility * dt

            ### normalize phi
            phi_sum = 0.0
            for p in range(phi[i, j, k].n):
                phi_sum += phi[i, j, k][p]
            for p in range(phi[i, j, k].n):
                phi[i, j, k][p] = phi[i, j, k][p] / phi_sum
            # phi[i, j, k] = phi[i, j, k] / phi[i, j, k].sum()
    

if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    polycrys = Polycrystal()
    
    polycrys.initialize()  

    for times in range(8000):  # 10000
        
        polycrys.advance()
        
        if (times % 64 == 0):

            print('times={}'.format(times))
            print(polycrys.phi[5, 5, 5].to_numpy())
            phi_np = polycrys.phi.to_numpy() 
            x_np = phi_np[phi_np.shape[0]//2, :, 0]
            gap = np.zeros(phi_np.shape[:3])
            polycrys_np = np.zeros(phi_np.shape[:3])
            for i in range(phi_np.shape[0]):
                for j in range(phi_np.shape[1]):
                    for k in range(phi_np.shape[2]):
                        # ============= grain boundary ================
                        gap[i, j, k] = max(phi_np[i, j, k, :]) - min(phi_np[i, j, k, :])
                        # === different grains in different colors ====
                        polycrys_np[i, j, k] = max(range(len(phi_np[i, j, k, :])), 
                                            key=lambda k1: phi_np[i, j, k, k1])  # show different grains

            #=========plot================
            path = "./pictures/"
            if not os.path.exists(path):
                os.makedirs(path)

            values_1 = gap
            values_2 = polycrys_np
            
            # Create the spatial reference
            # grid = pv.Plotter()
            # grid = pv.UniformGrid()
            grid = pv.ImageData()
            
            # Set the grid dimensions: shape + 1 because we want to inject our values on
            #   the CELL data
            grid.dimensions = np.array(values_1.shape) + 1
            
            # Edit the spatial reference
            grid.origin = (0, 0, 0)  # The bottom left corner of the data set
            grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
            
            #========plot grain boundaries=======
            # Add the data values to the cell data
            grid.cell_data["values_1"] = values_1.flatten(order="F")  # Flatten the array!
            
            # Now plot the grid!
            grid.plot(show_edges=False,cmap=cm.jet,off_screen=True,\
                screenshot="{}grain_boundaries{}.png".format(path,times),show_scalar_bar=False)

            #========plot different grains=======
            # Add the data values to the cell data
            grid1 = pv.ImageData()
            grid1.dimensions = np.array(values_1.shape) + 1
            grid1.origin = (0, 0, 0)
            grid1.spacing = (1, 1, 1)
            grid1.cell_data["values_2"] = values_2.flatten(order="F")  # Flatten the array!
            
            # Now plot the grid!
            grid1.plot(show_edges=False,cmap=cm.jet,off_screen=True,\
                screenshot="{}grains{}.png".format(path,times),show_scalar_bar=False)


            # sys.exit()
             





