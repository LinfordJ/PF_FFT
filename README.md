# Documentation for FFT-Based Multi-Phase Field Model

This document provides a detailed explanation of the physical principles, mathematical derivations, numerical discretization schemes, and code architecture of the Multi-Phase Field (MPF) model implemented in the `multiphase_fft` solver. It is intended to serve as a pedagogical guide and a theoretical reference for code maintenance and secondary development.

---

## 1. Physical Principles and Mathematical Model

This program primarily implements the **Steinbach Multi-Phase Field Model**. In a multi-grain or multi-phase evolving system, multiple phase variables $\phi_1, \phi_2, \dots, \phi_N$ exist simultaneously. We must ensure the volume conservation constraint is satisfied at any spatial point:
$$ \sum_{i=1}^{N} \phi_i(r, t) = 1 $$

### 1.1 Free Energy Functional
The total free energy $F$ of the system consists of gradient energy, chemical free energy (hindering dual-phase coexistence), a third-phase penalty term, and bulk free energy:
$$ F = \int_V \left[ f_{grad} + f_{chem} + f_{pen} + f_{bulk} \right] dV $$

The specific forms defined in the code are:
1. **Gradient Energy**: $\frac{\kappa}{2} \sum_{i=1}^N |\nabla \phi_i|^2$, reflecting the surface tension of the phase interfaces.
2. **Chemical Energy**: In the phase-field model, different phases repel each other. To form an interface with a finite width, the repulsive force is based on $W \sum_{i \neq j} |\phi_i \phi_j|$ as a dual-phase penalty.
3. **Penalty Energy (Three-Phase)**: $U \sum_{i<j<k} \phi_i \phi_j \phi_k$. Its purpose is to **strictly prevent three or more phases from coexisting** at the same spatial point in large amounts, preventing physically unreasonable "melting" phenomena. The expanded calculation is located in `energy.py`.
4. **Bulk Energy**: The chemical potential difference driving the phase transformation (e.g., solidification driving force caused by undercooling).

### 1.2 Kinetic Evolution Equation (Steinbach Equation)
To automatically satisfy the $\sum \phi_i = 1$ constraint during evolution, Steinbach proposed the following kinetic equation:
$$ \frac{\partial \phi_i}{\partial t} = \frac{M}{N_{phases}} \sum_{j \neq i} \left( \frac{\delta F}{\delta \phi_j} - \frac{\delta F}{\delta \phi_i} \right) $$

This is equivalent to calculating the absolute thermodynamic driving force $f_i = -\frac{\delta F}{\delta \phi_i}$ for each phase, and then subtracting the average driving force of all phases to obtain the **Effective Force**:
$$ f_{eff, i} = f_i - \frac{1}{N_{phases}} \sum_{j=1}^{N_{phases}} f_j $$
Thus, the equation simplifies to:
$$ \frac{\partial \phi_i}{\partial t} = M \cdot f_{eff, i} $$

---

## 2. Derivation of the Semi-Implicit Spectral Method via FFT

In the traditional Finite Difference (FD) method, the Laplacian operator $\nabla^2 \phi$ is calculated explicitly using neighboring points. For larger time steps $\Delta t$ or higher mobility $M$, explicit schemes easily diverge numerically (violating the CFL condition).
The **Spectral Method** solves the gradient term in the frequency domain using the Fast Fourier Transform (FFT), enabling **unconditionally stable** handling of linear terms.

### 2.1 Frequency Domain Transformation
Let the thermodynamic driving force be $f_i = \kappa \nabla^2 \phi_i + g_i(\phi)$, where $g_i$ contains the nonlinear chemical energy, penalty energy, and bulk energy.
Applying the Fourier transform $\mathcal{F}$ to both sides of the equation, and using the property $\mathcal{F}(\nabla^2 \phi) = -k^2 \hat{\phi}$ (where $k$ is the wave vector), the evolution equation in the frequency domain becomes:
$$ \frac{\partial \hat{\phi}_i}{\partial t} = M \left( -\kappa k^2 \hat{\phi}_i + \hat{g}_{eff, i} \right) $$

### 2.2 Semi-Implicit Time Discretization
To ensure stability, we evaluate the **linear gradient term** at the next time step $n+1$ (implicitly), while evaluating the **nonlinear term** at the current time step $n$ (explicitly):
$$ \frac{\hat{\phi}_i^{n+1} - \hat{\phi}_i^n}{\Delta t} = M \left( -\kappa k^2 \hat{\phi}_i^{n+1} + \hat{g}_{eff, i}^n \right) $$

Rearranging yields the explicit update formula for $\hat{\phi}^{n+1}$:
$$ \hat{\phi}_i^{n+1} = \frac{\hat{\phi}_i^n + \Delta t \cdot M \cdot \hat{g}_{eff, i}^n}{1 + \Delta t \cdot M \cdot \kappa \cdot k^2} $$

**This is the core of the spectral method's stability**: No matter how large $\Delta t$ or $\kappa$ is, the denominator $1 + \Delta t M \kappa k^2$ is always greater than 1, strongly suppressing high-frequency oscillations. This allows us to use time steps orders of magnitude larger than those in the FD model.

---

## 3. Code Architecture and Function Details

The code adopts a highly modular design, relying fundamentally on the `Taichi` framework to accelerate large-scale parallel computations on the GPU. Below is an exhaustive breakdown of every module and its functions.

### 3.1 `main.py` (Simulation Entry Point)
This file acts as the main orchestrator, setting up the configuration, initializing fields, and running the main time loop.
* **`init_random_noise_phi(...)` / `init_uniform_noise_phi(...)`**: Initial condition generators. `init_uniform_noise_phi` fills the domain with uniformly distributed initial phases plus a small random fluctuation ($\pm \text{fluctuation}$). It immediately normalizes them to sum to $1.0$ at every grid point, which accurately reflects a homogeneous high-temperature liquid or heavily disordered state.
* **`init_voronoi_phi(...)`**: An alternative initialization that seeds geometric nucleation points and grows them outwards using Voronoi tessellation logic, effectively creating an immediate polycrystalline structure as a starting point.
* **`main()`**: The main execution block. It constructs the `SimulationConfig`, instantiates the `SpectralSolver`, sets up the `PhaseFieldGUI`, and repeatedly invokes `solver.step()` inside a loop. It also governs when to render and save image frames.

### 3.2 `config.py` (Configuration Module)
* **`SimulationConfig` (Class)**: A Python `dataclass` used to centralize all physical and numerical parameters. 
  * *Parameters handled*: Grid size ($N$), physical domain size ($L$), time step ($dt$), gradient penalty ($\kappa$), chemical energy ($W$), penalty energy ($U$), mobility ($M$), bulk energies, and equation type (Allen-Cahn vs. Cahn-Hilliard).
  * **`__post_init__()`**: An automatic validation function triggered after instantiation to ensure physics constraints are met (e.g., verifying grid matches the 2D/3D dimension, ensuring phase count is $\ge 2$).

### 3.3 `physics/energy.py` (Energy Physics Module)
* **`MultiphaseEnergy` (Class)**: Encapsulates the local nonlinear energy interactions.
  * **`compute_force(phi_p, sum_abs, sum_sq_abs)`**: Calculates the strictly local nonlinear driving force for a specific phase $p$.
    * *Principle*: Instead of nesting loops ($O(N_{phases}^2)$ or $O(N_{phases}^3)$) to calculate chemical repulsions $W \sum_{i \neq p} \phi_i$ and penalties $U \sum_{i<j<k} \phi_i \phi_j \phi_k$, this function utilizes the algebraic properties of the system ($\sum \phi = 1$) via precomputed sums (`sum_abs` and `sum_sq_abs`). It calculates the local non-linear thermodynamic driving force in $O(1)$ time per phase.

### 3.4 `math_utils/taichi_fft.py` (Low-Level Math Module)
This is a custom-built, parallelized Fast Fourier Transform module written purely in Taichi, avoiding the overhead of constantly copying GPU arrays back to the CPU for NumPy/SciPy FFTs.
* **`complex_mul(...)` / `complex_exp(...)`**: Fundamental complex number arithmetic operations for calculating the "twiddle factors" (Euler's formula $e^{i\theta}$).
* **`reverse_bits(...)` / `bit_reversal_permutation...`**: Rearranges the input array indices according to bit-reversed order, a prerequisite step in the Cooley-Tukey FFT algorithm to perform in-place calculations.
* **`compute_fft_..._step`**: The core "butterfly" operations. It iteratively splits the discrete Fourier transform into smaller sub-problems, massively reducing computational complexity from $O(N^2)$ to $O(N \log N)$.
* **`fft_2d_batched(...)` / `ifft_2d_batched(...)`**: The batched 2D implementations. 
  * *Principle*: In a multi-phase system, we have $N_{phases}$ separate 2D grids. This function executes parallel 1D FFTs across the rows, transposes the matrices in memory, and then executes parallel 1D FFTs across the columns, doing this simultaneously for all $N_{phases}$ channels.

### 3.5 `solver/spectral_solver.py` (Core Solver Engine)
This is the heart of the mathematical progression of the phase-field.
* **`__init__(...)`**: Allocates the memory buffers. Fields like `self.phi` (the spatial domain phase field) and `self.phi_work_k` (the complex frequency domain buffer) are instantiated here.
* **`setup_k2()`**: 
  * *Principle*: Generates the frequency domain grid for the Laplacian operator. The spatial derivative $\nabla^2 \phi$ mathematically maps to multiplying by $-|k|^2$ in Fourier space. This function creates the squared wave number grid $k^2 = k_x^2 + k_y^2 + k_z^2$ using standard `np.fft.fftfreq` mapping.
* **`compute_df_and_load()`**: 
  * *Principle*: Calculates the explicit part of the time step. It loops over the spatial grid, calculates the raw thermodynamic force using `energy.compute_force`, applies the Steinbach average projection ($f_{eff, i} = f_i - \frac{1}{N_{phases}} \sum_{j} f_j$) to maintain the sum-to-one constraint, and finally loads both $\phi$ and the effective force into the complex vector buffers `phi_work_k` and `df_work_k`.
* **`update_work_k_allen_cahn(...)`**:
  * *Principle*: The numerical core of the semi-implicit time integration. Executed entirely in the frequency domain, it applies the update formula: $\hat{\phi}_{new} = (\hat{\phi}_{old} + dt \cdot M \cdot \hat{f}_{eff}) / (1 + dt \cdot M \cdot \kappa \cdot k^2)$. The division by $(1+...k^2)$ acts as a massive low-pass filter, neutralizing the high-frequency instabilities normally triggered by explicit gradient calculations.
* **`save_work_k_and_project()`**:
  * *Principle*: Executed after taking the Inverse FFT. It extracts the real-valued numbers from the complex `phi_work_k` arrays, writes them back to the actual `self.phi` spatial arrays, and enforces a final rigid normalization $\phi_i = \phi_i / \sum \phi_j$ to cleanly wipe out any floating-point arithmetic errors accumulated during the forward-and-inverse FFT trips.
* **`step()`**: The master wrapper. It consecutively calls: `compute_df_and_load` $\rightarrow$ `fft_batched` $\rightarrow$ `update_work_k_allen_cahn` $\rightarrow$ `ifft_batched` $\rightarrow$ `save_work_k_and_project`.

### 3.6 `visualization/gui.py` (Visualization Module)
* **`render_all_modes()`**: Computes visualization buffers concurrently for real-time display.
  * **RGB Mode**: Multiplies pre-defined distinct colors by phase fraction arrays and sums them together. Shows smooth blending at the interfaces.
  * **Grains Mode**: Finds the maximum phase index at every pixel (`max_idx`) and maps it to a discrete 'Jet' colormap. Best for visually defining solid polycrystalline boundaries.
  * **Boundaries Mode**: Calculates the difference between the maximum phase fraction and the minimum phase fraction (`max_val - min_val`). Because the bulk phases are uniform (max=1, min=0, diff=1) and boundaries consist of partial fractions, this perfectly highlights the phase interfaces.
* **`render()`**: Pushes the calculated buffer matrix to Taichi's built-in GUI, draws statistical overlays (Step numbers, time), and manages screen refreshing or `.png` file saving.

---

## 4. Parameter Setting Guide

To ensure physically reasonable field evolution, key parameters must be properly constrained:
* `dx` (`L / N`): Spatial discretization step. Typically, the interface width is $\delta \approx \sqrt{\kappa / W}$. To guarantee at least 4~6 grid points inside the interface, ensure $dx < \delta / 2$.
* `dt`: Time step. Although the spectral method makes the linear term unconditionally stable, the nonlinear term is still bounded. If $dt$ is too large, the nonlinear forces will cause local oscillations.
* `U` and `W`: Empirically, the three-phase penalty coefficient $U$ needs to be large enough to prevent a "third phase" from nucleating within a dual-phase interface. It is typically set from `U = 3.0 * W` to `U = 5.0 * W`.

## 5. Future Extensions and Implementations

The current codebase possesses high generality. Future research or engineering can easily build upon it:
1. **Anisotropy**
   * **Implementation**: Change the constant gradient coefficient $\kappa$ into a function $a(\vec{n})$ dependent on the interface normal vector $\vec{n} = \nabla \phi / |\nabla \phi|$. Since it cannot be fully integrated into the implicit denominator in the spectral method, an Operator Splitting strategy is usually adopted: "constant part implicit + anisotropic perturbation part explicit".
2. **Coupling Microelasticity**
   * **Implementation**: Based on Khachaturyan's microelasticity theory. Elastic energy can be solved exactly in the frequency domain via Green's Tensor. This is a **perfect match** with the current FFT architecture. You only need to add a function in `spectral_solver.py` to calculate the elastic stress $\hat{\sigma}$ based on current $\hat{\phi}$, and transform it back to the spatial domain as an extra driving force.
3. **Cahn-Hilliard Equation Extension**
   * **Implementation**: For conserved fields (like concentration), the evolution equation is $\frac{\partial c}{\partial t} = \nabla \cdot (M \nabla \mu)$. In the frequency domain, simply change the denominator to $1 + \Delta t M \kappa k^4$, and multiply the nonlinear driving term by $-k^2$. The `update_work_k_cahn_hilliard` method has already been reserved in the code.
