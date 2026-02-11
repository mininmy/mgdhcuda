# # Multivariate Polynomial Multiplication with GPU (CuPy, Dask, CuDF, Numba)
import numpy as np
import cudf
import dask.array as da 
import cupy as cp
from numba import cuda, uint8, float64
from dask import delayed
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from config_constants import MAX_EXP
from cuda_least_squares import least_squares_gpu, weighted_least_squares_gpu, weighted_least_squares_with_errors_gpu
# --- Constants ---
NVARS = 3
BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 32

@cuda.jit
def multivariateMulArbitrarySizedPolsCUDA(exp_C, exp_keys, exp_A, exp_B,
                                           coeff_C, coeff_A, coeff_B,
                                           nC, nA, nB):
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    tbx0 = tx + bx * BLOCK_SIZE_X
    tby0 = ty + by * BLOCK_SIZE_Y

    tbx = tbx0
    tby = tby0

    offsetx = BLOCK_SIZE_X * cuda.gridDim.x
    offsety = BLOCK_SIZE_Y * cuda.gridDim.y

    Aes = cuda.shared.array(shape=(BLOCK_SIZE_X, NVARS), dtype=uint8)
    Acs = cuda.shared.array(BLOCK_SIZE_X, dtype=float64)

    Bes = cuda.shared.array(shape=(BLOCK_SIZE_Y, NVARS), dtype=uint8)
    Bcs = cuda.shared.array(BLOCK_SIZE_Y, dtype=float64)

    Cexp = cuda.local.array(NVARS, dtype=uint8)

    while tby < nB:
        while tbx < nA:
            if tx < BLOCK_SIZE_X and tbx < nA:
                Acs[tx] = coeff_A[tbx]
                for k in range(NVARS):
                    Aes[tx, k] = exp_A[tbx * NVARS + k]

            if ty < BLOCK_SIZE_Y and tby < nB:
                Bcs[ty] = coeff_B[tby]
                for k in range(NVARS):
                    Bes[ty, k] = exp_B[tby * NVARS + k]

            cuda.syncthreads()

            if tbx < nA and tby < nB:
                Ccoeff = Acs[tx] * Bcs[ty]
                ekey = 0
                c = nA * tby + tbx
                coeff_C[c] = Ccoeff
                for k in range(NVARS):
                    Cexp[k] = Aes[tx, k] + Bes[ty, k]
                    exp_C[c * NVARS + k] = Cexp[k]
                    ekey = MAX_EXP * ekey + Cexp[k]
                exp_keys[c] = ekey

            tbx += offsetx
        tby += offsety
        tbx = tbx0

def launch_cuda_kernel(exp_A, exp_B, coeff_A, coeff_B):
    nA, nB = exp_A.shape[0], exp_B.shape[0]
    nC = nA * nB

    exp_A_flat = exp_A.reshape(-1)
    exp_B_flat = exp_B.reshape(-1)

    d_exp_A = cuda.to_device(exp_A_flat)
    d_exp_B = cuda.to_device(exp_B_flat)
    d_coeff_A = cuda.to_device(coeff_A)
    d_coeff_B = cuda.to_device(coeff_B)

    d_exp_C = cuda.device_array((nC * NVARS,), dtype=np.uint8)
    d_coeff_C = cuda.device_array(nC, dtype=np.float64)
    d_exp_keys = cuda.device_array(nC, dtype=np.uint64)
   
    threadsperblock = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
    blockspergrid_x = (nA + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    blockspergrid_y = (nB + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    blockspergrid = (blockspergrid_x, blockspergrid_y)
            
    multivariateMulArbitrarySizedPolsCUDA[blockspergrid, threadsperblock](
            d_exp_C, d_exp_keys, d_exp_A, d_exp_B,
            d_coeff_C, d_coeff_A, d_coeff_B,
            nC, nA, nB
        )
    cuda.synchronize()
    return np.stack([d_exp_keys.copy_to_host(), d_coeff_C.copy_to_host()], axis=1)

    

