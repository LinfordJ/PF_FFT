import taichi as ti
import math

vec2 = ti.math.vec2

@ti.func
def complex_mul(a, b):
    """
    Multiply two complex numbers.
    Args:
        a: ti.math.vec2
        b: ti.math.vec2
    Returns:
        ti.math.vec2
    """
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

@ti.func
def complex_exp(theta):
    """
    Compute e^(i*theta) = cos(theta) + i*sin(theta).
    Args:
        theta: float
    Returns:
        ti.math.vec2
    """
    return vec2(ti.cos(theta), ti.sin(theta))

@ti.func
def reverse_bits(n, bits):
    """
    Reverse the bits of an integer.
    Args:
        n: The integer to reverse
        bits: Number of bits
    Returns:
        The bit-reversed integer
    """
    rev = 0
    num = n
    for _ in range(bits):
        rev = (rev << 1) | (num & 1)
        num >>= 1
    return rev

@ti.kernel
def bit_reversal_permutation_1d(data: ti.template(), size: ti.i32, log2_size: ti.i32):
    """
    Perform in-place bit-reversal permutation for 1D FFT.
    """
    for i in range(size):
        rev = reverse_bits(i, log2_size)
        if i < rev:
            temp = data[i]
            data[i] = data[rev]
            data[rev] = temp

@ti.kernel
def compute_fft_1d_step(data: ti.template(), size: ti.i32, step: ti.i32, half_step: ti.i32, sign: ti.f32):
    """
    Compute a single stage of the 1D Radix-2 Cooley-Tukey FFT.
    """
    for k in range(size // 2):
        # Determine the group and index within the group
        group = k // half_step
        idx = k % half_step
        
        i = group * step + idx
        j = i + half_step
        
        # Twiddle factor angle
        theta = sign * 2.0 * math.pi * ti.cast(idx, ti.f32) / ti.cast(step, ti.f32)
        w = complex_exp(theta)
        
        u = data[i]
        t = complex_mul(w, data[j])
        
        data[i] = u + t
        data[j] = u - t

def fft_1d_inplace(data, sign: float):
    """
    Perform in-place 1D FFT or IFFT.
    Args:
        data: A 1D Taichi field or ndarray of ti.math.vec2.
        sign: -1.0 for FFT, 1.0 for IFFT.
    """
    size = data.shape[0]
    log2_size = int(math.log2(size))
    if 2**log2_size != size:
        raise ValueError("FFT size must be a power of 2.")
    
    # 1. Bit-reversal permutation
    bit_reversal_permutation_1d(data, size, log2_size)
    
    # 2. Cooley-Tukey butterfly stages
    step = 2
    while step <= size:
        half_step = step // 2
        compute_fft_1d_step(data, size, step, half_step, sign)
        step *= 2

@ti.kernel
def scale_data_1d(data: ti.template(), scale: ti.f32):
    for i in data:
        data[i] *= scale

def fft_1d(data):
    """1D Forward Fast Fourier Transform."""
    fft_1d_inplace(data, -1.0)

def ifft_1d(data):
    """1D Inverse Fast Fourier Transform."""
    fft_1d_inplace(data, 1.0)
    size = data.shape[0]
    scale_data_1d(data, 1.0 / size)

# --- N-Dimensional FFT ---

@ti.kernel
def transpose_2d(src: ti.template(), dst: ti.template()):
    for i, j in src:
        dst[j, i] = src[i, j]

def fft_2d(data, buffer):
    """
    2D Forward FFT using row-column decomposition.
    Requires a temporary buffer of the same shape/type as `data` for transposition.
    Both `data` and `buffer` should be 2D Taichi fields/ndarrays of vec2.
    """
    shape = data.shape
    if len(shape) != 2:
        raise ValueError("fft_2d expects 2D data.")
    
    # Assuming shape is (N, N) or (N, M) where both are powers of 2.
    n_rows, n_cols = shape
    
    # We can't easily slice multi-dimensional Taichi fields in kernel, 
    # so we write specific kernels that perform 1D FFTs along rows/cols.
    fft_2d_rows(data, n_rows, n_cols, -1.0)
    transpose_2d(data, buffer)
    fft_2d_rows(buffer, n_cols, n_rows, -1.0)
    transpose_2d(buffer, data)

@ti.kernel
def bit_reversal_permutation_2d_rows(data: ti.template(), n_rows: ti.i32, n_cols: ti.i32, log2_cols: ti.i32):
    for row, col in ti.ndrange(n_rows, n_cols):
        rev = reverse_bits(col, log2_cols)
        if col < rev:
            temp = data[row, col]
            data[row, col] = data[row, rev]
            data[row, rev] = temp

@ti.kernel
def compute_fft_2d_rows_step(data: ti.template(), n_rows: ti.i32, n_cols: ti.i32, step: ti.i32, half_step: ti.i32, sign: ti.f32):
    for row, k in ti.ndrange(n_rows, n_cols // 2):
        group = k // half_step
        idx = k % half_step
        
        i = group * step + idx
        j = i + half_step
        
        theta = sign * 2.0 * math.pi * ti.cast(idx, ti.f32) / ti.cast(step, ti.f32)
        w = complex_exp(theta)
        
        u = data[row, i]
        t = complex_mul(w, data[row, j])
        
        data[row, i] = u + t
        data[row, j] = u - t

def fft_2d_rows(data, n_rows, n_cols, sign: float):
    log2_cols = int(math.log2(n_cols))
    bit_reversal_permutation_2d_rows(data, n_rows, n_cols, log2_cols)
    
    step = 2
    while step <= n_cols:
        half_step = step // 2
        compute_fft_2d_rows_step(data, n_rows, n_cols, step, half_step, sign)
        step *= 2

@ti.kernel
def scale_data_2d(data: ti.template(), scale: ti.f32):
    for i, j in data:
        data[i, j] *= scale

def ifft_2d(data, buffer):
    """2D Inverse FFT."""
    shape = data.shape
    n_rows, n_cols = shape
    
    fft_2d_rows(data, n_rows, n_cols, 1.0)
    transpose_2d(data, buffer)
    fft_2d_rows(buffer, n_cols, n_rows, 1.0)
    transpose_2d(buffer, data)
    
    scale_data_2d(data, 1.0 / (n_rows * n_cols))

# --- 3D FFT ---
@ti.kernel
def bit_reversal_permutation_3d_x(data: ti.template(), nx: ti.i32, ny: ti.i32, nz: ti.i32, log2_nx: ti.i32):
    for i, j, k in ti.ndrange(nx, ny, nz):
        rev = reverse_bits(i, log2_nx)
        if i < rev:
            temp = data[i, j, k]
            data[i, j, k] = data[rev, j, k]
            data[rev, j, k] = temp

@ti.kernel
def compute_fft_3d_x_step(data: ti.template(), nx: ti.i32, ny: ti.i32, nz: ti.i32, step: ti.i32, half_step: ti.i32, sign: ti.f32):
    for m, j, k in ti.ndrange(nx // 2, ny, nz):
        group = m // half_step
        idx = m % half_step
        
        i0 = group * step + idx
        i1 = i0 + half_step
        
        theta = sign * 2.0 * math.pi * ti.cast(idx, ti.f32) / ti.cast(step, ti.f32)
        w = complex_exp(theta)
        
        u = data[i0, j, k]
        t = complex_mul(w, data[i1, j, k])
        
        data[i0, j, k] = u + t
        data[i1, j, k] = u - t

def fft_3d_axis_x(data, nx, ny, nz, sign: float):
    log2_nx = int(math.log2(nx))
    bit_reversal_permutation_3d_x(data, nx, ny, nz, log2_nx)
    
    step = 2
    while step <= nx:
        half_step = step // 2
        compute_fft_3d_x_step(data, nx, ny, nz, step, half_step, sign)
        step *= 2

@ti.kernel
def transpose_3d_xy(src: ti.template(), dst: ti.template()):
    """Transpose X and Y axes: dst[y, x, z] = src[x, y, z]"""
    for x, y, z in src:
        dst[y, x, z] = src[x, y, z]

@ti.kernel
def transpose_3d_xz(src: ti.template(), dst: ti.template()):
    """Transpose X and Z axes: dst[z, y, x] = src[x, y, z]"""
    for x, y, z in src:
        dst[z, y, x] = src[x, y, z]

def fft_3d(data, buffer):
    """
    3D Forward FFT.
    Requires a temporary buffer of the same shape/type as `data` for transpositions.
    """
    nx, ny, nz = data.shape
    
    # 1. FFT along X
    fft_3d_axis_x(data, nx, ny, nz, -1.0)
    
    # 2. Transpose X and Y, FFT along new X (original Y)
    transpose_3d_xy(data, buffer)
    fft_3d_axis_x(buffer, ny, nx, nz, -1.0)
    transpose_3d_xy(buffer, data)
    
    # 3. Transpose X and Z, FFT along new X (original Z)
    transpose_3d_xz(data, buffer)
    fft_3d_axis_x(buffer, nz, ny, nx, -1.0)
    transpose_3d_xz(buffer, data)

@ti.kernel
def scale_data_3d(data: ti.template(), scale: ti.f32):
    for i, j, k in data:
        data[i, j, k] *= scale

def ifft_3d(data, buffer):
    """3D Inverse FFT."""
    nx, ny, nz = data.shape
    
    fft_3d_axis_x(data, nx, ny, nz, 1.0)
    
    transpose_3d_xy(data, buffer)
    fft_3d_axis_x(buffer, ny, nx, nz, 1.0)
    transpose_3d_xy(buffer, data)
    
    transpose_3d_xz(data, buffer)
    fft_3d_axis_x(buffer, nz, ny, nx, 1.0)
    transpose_3d_xz(buffer, data)
    
    scale_data_3d(data, 1.0 / (nx * ny * nz))

# --- fftshift and ifftshift ---

@ti.kernel
def fftshift_1d_kernel(src: ti.template(), dst: ti.template(), n: ti.i32, shift: ti.i32):
    for i in range(n):
        dst[(i + shift) % n] = src[i]

def fftshift_1d(src, dst):
    """1D fftshift: shifts zero-frequency component to center."""
    n = src.shape[0]
    shift = n // 2
    fftshift_1d_kernel(src, dst, n, shift)

def ifftshift_1d(src, dst):
    """1D ifftshift: inverse of fftshift."""
    n = src.shape[0]
    shift = (n + 1) // 2
    fftshift_1d_kernel(src, dst, n, shift)

@ti.kernel
def fftshift_2d_kernel(src: ti.template(), dst: ti.template(), nx: ti.i32, ny: ti.i32, shift_x: ti.i32, shift_y: ti.i32):
    for i, j in ti.ndrange(nx, ny):
        dst[(i + shift_x) % nx, (j + shift_y) % ny] = src[i, j]

def fftshift_2d(src, dst):
    """2D fftshift."""
    nx, ny = src.shape
    fftshift_2d_kernel(src, dst, nx, ny, nx // 2, ny // 2)

def ifftshift_2d(src, dst):
    """2D ifftshift."""
    nx, ny = src.shape
    fftshift_2d_kernel(src, dst, nx, ny, (nx + 1) // 2, (ny + 1) // 2)

@ti.kernel
def fftshift_3d_kernel(src: ti.template(), dst: ti.template(), nx: ti.i32, ny: ti.i32, nz: ti.i32, shift_x: ti.i32, shift_y: ti.i32, shift_z: ti.i32):
    for i, j, k in ti.ndrange(nx, ny, nz):
        dst[(i + shift_x) % nx, (j + shift_y) % ny, (k + shift_z) % nz] = src[i, j, k]

def fftshift_3d(src, dst):
    """3D fftshift."""
    nx, ny, nz = src.shape
    fftshift_3d_kernel(src, dst, nx, ny, nz, nx // 2, ny // 2, nz // 2)

def ifftshift_3d(src, dst):
    """3D ifftshift."""
    nx, ny, nz = src.shape
    fftshift_3d_kernel(src, dst, nx, ny, nz, (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2)

# --- BATCHED 2D FFT (for N_phases simultaneously) ---

@ti.kernel
def transpose_2d_batched(src: ti.template(), dst: ti.template()):
    for b, i, j in src:
        dst[b, j, i] = src[b, i, j]

@ti.kernel
def bit_reversal_permutation_2d_rows_batched(data: ti.template(), n_batches: ti.i32, n_rows: ti.i32, n_cols: ti.i32, log2_cols: ti.i32):
    for b, row, col in ti.ndrange(n_batches, n_rows, n_cols):
        rev = reverse_bits(col, log2_cols)
        if col < rev:
            temp = data[b, row, col]
            data[b, row, col] = data[b, row, rev]
            data[b, row, rev] = temp

@ti.kernel
def compute_fft_2d_rows_step_batched(data: ti.template(), n_batches: ti.i32, n_rows: ti.i32, n_cols: ti.i32, step: ti.i32, half_step: ti.i32, sign: ti.f32):
    for b, row, k in ti.ndrange(n_batches, n_rows, n_cols // 2):
        group = k // half_step
        idx = k % half_step
        
        i = group * step + idx
        j = i + half_step
        
        theta = sign * 2.0 * math.pi * ti.cast(idx, ti.f32) / ti.cast(step, ti.f32)
        w = complex_exp(theta)
        
        u = data[b, row, i]
        t = complex_mul(w, data[b, row, j])
        
        data[b, row, i] = u + t
        data[b, row, j] = u - t

def fft_2d_rows_batched(data, n_batches, n_rows, n_cols, sign: float):
    log2_cols = int(math.log2(n_cols))
    bit_reversal_permutation_2d_rows_batched(data, n_batches, n_rows, n_cols, log2_cols)
    
    step = 2
    while step <= n_cols:
        half_step = step // 2
        compute_fft_2d_rows_step_batched(data, n_batches, n_rows, n_cols, step, half_step, sign)
        step *= 2

@ti.kernel
def scale_data_batched(data: ti.template(), scale: ti.f32):
    for I in ti.grouped(data):
        data[I] *= scale

def fft_2d_batched(data, buffer):
    """Batched 2D Forward FFT."""
    n_batches, n_rows, n_cols = data.shape
    
    fft_2d_rows_batched(data, n_batches, n_rows, n_cols, -1.0)
    transpose_2d_batched(data, buffer)
    fft_2d_rows_batched(buffer, n_batches, n_cols, n_rows, -1.0)
    transpose_2d_batched(buffer, data)

def ifft_2d_batched(data, buffer):
    """Batched 2D Inverse FFT."""
    n_batches, n_rows, n_cols = data.shape
    
    fft_2d_rows_batched(data, n_batches, n_rows, n_cols, 1.0)
    transpose_2d_batched(data, buffer)
    fft_2d_rows_batched(buffer, n_batches, n_cols, n_rows, 1.0)
    transpose_2d_batched(buffer, data)
    
    scale_data_batched(data, 1.0 / (n_rows * n_cols))

# --- BATCHED 3D FFT ---

@ti.kernel
def transpose_3d_xy_batched(src: ti.template(), dst: ti.template()):
    for b, i, j, k in src:
        dst[b, j, i, k] = src[b, i, j, k]

@ti.kernel
def transpose_3d_xz_batched(src: ti.template(), dst: ti.template()):
    for b, i, j, k in src:
        dst[b, k, j, i] = src[b, i, j, k]

@ti.kernel
def bit_reversal_permutation_3d_x_batched(data: ti.template(), n_batches: ti.i32, nx: ti.i32, ny: ti.i32, nz: ti.i32, log2_x: ti.i32):
    for b, y, z, x in ti.ndrange(n_batches, ny, nz, nx):
        rev = reverse_bits(x, log2_x)
        if x < rev:
            temp = data[b, x, y, z]
            data[b, x, y, z] = data[b, rev, y, z]
            data[b, rev, y, z] = temp

@ti.kernel
def compute_fft_3d_x_step_batched(data: ti.template(), n_batches: ti.i32, nx: ti.i32, ny: ti.i32, nz: ti.i32, step: ti.i32, half_step: ti.i32, sign: ti.f32):
    for b, y, z, k in ti.ndrange(n_batches, ny, nz, nx // 2):
        group = k // half_step
        idx = k % half_step
        i = group * step + idx
        j = i + half_step
        
        theta = sign * 2.0 * math.pi * ti.cast(idx, ti.f32) / ti.cast(step, ti.f32)
        w = complex_exp(theta)
        
        u = data[b, i, y, z]
        t = complex_mul(w, data[b, j, y, z])
        
        data[b, i, y, z] = u + t
        data[b, j, y, z] = u - t

def fft_3d_x_batched(data, n_batches, nx, ny, nz, sign: float):
    log2_x = int(math.log2(nx))
    bit_reversal_permutation_3d_x_batched(data, n_batches, nx, ny, nz, log2_x)
    step = 2
    while step <= nx:
        half_step = step // 2
        compute_fft_3d_x_step_batched(data, n_batches, nx, ny, nz, step, half_step, sign)
        step *= 2

def fft_3d_batched(data, buffer):
    n_batches, nx, ny, nz = data.shape
    
    # X direction
    fft_3d_x_batched(data, n_batches, nx, ny, nz, -1.0)
    
    # Y direction (transpose X and Y)
    transpose_3d_xy_batched(data, buffer)
    fft_3d_x_batched(buffer, n_batches, ny, nx, nz, -1.0)
    transpose_3d_xy_batched(buffer, data)
    
    # Z direction (transpose X and Z)
    transpose_3d_xz_batched(data, buffer)
    fft_3d_x_batched(buffer, n_batches, nz, ny, nx, -1.0)
    transpose_3d_xz_batched(buffer, data)

def ifft_3d_batched(data, buffer):
    n_batches, nx, ny, nz = data.shape
    
    fft_3d_x_batched(data, n_batches, nx, ny, nz, 1.0)
    
    transpose_3d_xy_batched(data, buffer)
    fft_3d_x_batched(buffer, n_batches, ny, nx, nz, 1.0)
    transpose_3d_xy_batched(buffer, data)
    
    transpose_3d_xz_batched(data, buffer)
    fft_3d_x_batched(buffer, n_batches, nz, ny, nx, 1.0)
    transpose_3d_xz_batched(buffer, data)
    
    scale_data_batched(data, 1.0 / (nx * ny * nz))

# --- BATCHED 1D FFT ---
@ti.kernel
def bit_reversal_permutation_1d_batched(data: ti.template(), n_batches: ti.i32, n: ti.i32, log2_n: ti.i32):
    for b, i in ti.ndrange(n_batches, n):
        rev = reverse_bits(i, log2_n)
        if i < rev:
            temp = data[b, i]
            data[b, i] = data[b, rev]
            data[b, rev] = temp

@ti.kernel
def compute_fft_1d_step_batched(data: ti.template(), n_batches: ti.i32, n: ti.i32, step: ti.i32, half_step: ti.i32, sign: ti.f32):
    for b, k in ti.ndrange(n_batches, n // 2):
        group = k // half_step
        idx = k % half_step
        i = group * step + idx
        j = i + half_step
        
        theta = sign * 2.0 * math.pi * ti.cast(idx, ti.f32) / ti.cast(step, ti.f32)
        w = complex_exp(theta)
        
        u = data[b, i]
        t = complex_mul(w, data[b, j])
        
        data[b, i] = u + t
        data[b, j] = u - t

def fft_1d_batched_core(data, n_batches, n, sign):
    log2_n = int(math.log2(n))
    bit_reversal_permutation_1d_batched(data, n_batches, n, log2_n)
    step = 2
    while step <= n:
        half_step = step // 2
        compute_fft_1d_step_batched(data, n_batches, n, step, half_step, sign)
        step *= 2

def fft_1d_batched(data, buffer=None):
    n_batches, n = data.shape
    fft_1d_batched_core(data, n_batches, n, -1.0)

def ifft_1d_batched(data, buffer=None):
    n_batches, n = data.shape
    fft_1d_batched_core(data, n_batches, n, 1.0)
    scale_data_batched(data, 1.0 / n)
