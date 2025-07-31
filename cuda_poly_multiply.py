# # Multivariate Polynomial Multiplication with GPU (CuPy, Dask, CuDF, Numba)

import numpy as np
import cupy as cp
import cudf
import dask.array as da 

from numba import cuda, uint8, float64
from dask import delayed
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from config_constants import MAX_EXP
# --- Constants ---
NVARS = 3
BLOCK_SIZE_X = 8
BLOCK_SIZE_Y = 8

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
    
    


def main():
    # --- Dask cluster ---
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # --- Generate data ---
    nA, nB = 3, 3
    chunk_size = 2

    exp_A = da.from_array(cp.random.randint(0, 2, size=(nA, NVARS), dtype=cp.uint8), chunks=(chunk_size, NVARS))
    exp_B = da.from_array(cp.random.randint(0, 2, size=(nB, NVARS), dtype=cp.uint8), chunks=(chunk_size, NVARS))
    coeff_A = da.from_array(cp.random.randn(nA).astype(cp.float64), chunks=chunk_size)
    coeff_B = da.from_array(cp.random.randn(nB).astype(cp.float64), chunks=chunk_size)

    # --- Lazy block processing ---
    exp_A_blocks = exp_A.to_delayed().flatten()
    exp_B_blocks = exp_B.to_delayed().flatten()
    coeff_A_blocks = coeff_A.to_delayed().flatten()
    coeff_B_blocks = coeff_B.to_delayed().flatten()
    
    results = []
    for i in range(len(exp_A_blocks)):
        for j in range(len(exp_B_blocks)):
            A_e = exp_A_blocks[i]
            B_e = exp_B_blocks[j]
            A_c = coeff_A_blocks[i]
            B_c = coeff_B_blocks[j]
            
            result = delayed(launch_cuda_kernel)(A_e, B_e, A_c, B_c)
            
            results.append(result)

    # --- Compute and gather ---
    print('Computing....')
    futures = client.compute(results)
    computed_results = client.gather(futures)
    cuda.synchronize()
    print('Done computing')

    # --- CuPy conversion and concatenation ---
    cupy_results = []
    for r in computed_results:
        arr = np.asarray(r)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        cupy_results.append(cp.asarray(arr))

    all_results = cp.concatenate(cupy_results, axis=0)

    # --- CuDF reduction ---
    keys = all_results[:, 0].astype(cp.uint64)
    coeffs = all_results[:, 1]

    df = cudf.DataFrame({'key': keys, 'coeff': coeffs})
    cuda.synchronize()
    reduced_df = df.groupby('key').agg({'coeff': 'sum'}).reset_index()
    cuda.synchronize()
    print(reduced_df.head())


if __name__ == "__main__":
    from numba import config
    config.CUDA_DEBUGINFO = 1
    config.CUDA_EXCEPTION_HANDLING = 1

    import multiprocessing

    multiprocessing.freeze_support()  # Good practice for compatibility

    main()
