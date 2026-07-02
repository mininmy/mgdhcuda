# # Multivariate Polynomial Multiplication with GPU (CuPy, Dask, CuDF, Numba)
import numpy as np
import time
import dask.array as da 
import cupy as cp
from numba import cuda, uint8, float64, uint64
from dask import delayed
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from config_constants import MAX_EXP

# --- Constants ---
NVARS = 3
BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 32
@cuda.jit
def multivariateMulCUDA(exp_C, exp_keys, exp_A, exp_B, coeff_C, coeff_A, coeff_B, nA, nB):
    # Прямі індекси в матриці комбінацій
    tbx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tby = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Перевірка меж (обов'язково!)
    if tbx < nA and tby < nB:
        # 1. Множимо коефіцієнти
        c_idx = tby * nA + tbx # Лінійний індекс у масиві результатів
        coeff_C[c_idx] = coeff_A[tbx] * coeff_B[tby]

        # 2. Додаємо степені (NVARS = 3)
        ekey = uint64(0)
        for k in range(NVARS):
            # Читаємо степені прямо з пам'яті (або з shared, якщо додаси)
            e_sum = exp_A[tbx * NVARS + k] + exp_B[tby * NVARS + k]
            exp_C[c_idx * NVARS + k] = e_sum
            
            # Оновлюємо ключ
            ekey = MAX_EXP * ekey + e_sum # MAX_EXP = 1000
            
        exp_keys[c_idx] = ekey
def get_dynamic_block_size(n):
    # Шукаємо найближчу степінь 2: 4, 8, 16 або 32
    if n <= 4: return 4
    if n <= 8: return 8
    if n <= 16: return 16
    return 32 # Максимум для 2D блоку (32*32=1024)

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
    bs_x = get_dynamic_block_size(nA)
    bs_y = get_dynamic_block_size(nB)
    threadsperblock = (bs_x, bs_y)
    blockspergrid = ((nA + bs_x - 1) // bs_x, (nB + bs_y - 1) // bs_y)
    #blockspergrid_x = (nA + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    #blockspergrid_y = (nB + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    #blockspergrid = (blockspergrid_x, blockspergrid_y)
            
    multivariateMulCUDA[blockspergrid, threadsperblock](
            d_exp_C, d_exp_keys, d_exp_A, d_exp_B,
            d_coeff_C, d_coeff_A, d_coeff_B,
            nA, nB
        )
    cuda.synchronize()
    return np.stack([d_exp_keys.copy_to_host(), d_coeff_C.copy_to_host()], axis=1)

    MAX_EXP = 1000

# 1. Генерація випадкових поліномів (500 термів кожен)
def generate_test_poly(n_terms):
    exps = np.random.randint(0, 10, (n_terms, NVARS)).astype(np.uint8)
    coeffs = np.random.uniform(-1, 1, n_terms).astype(np.float64)
    return exps, coeffs

# 2. NumPy версія (CPU)
def multiply_pols_numpy(exp_A, coeff_A, exp_B, coeff_B):
    start = time.perf_counter()
    # Зовнішнє додавання та множення
    new_exps = exp_A[:, np.newaxis, :] + exp_B[np.newaxis, :, :]
    new_coeffs = (coeff_A[:, np.newaxis] * coeff_B[None, :]).flatten()
    new_exps = new_exps.reshape(-1, NVARS)
    
    # Генерація ключів для редукції
    keys = (new_exps[:, 0].astype(np.uint64) * (MAX_EXP**2) + 
            new_exps[:, 1].astype(np.uint64) * MAX_EXP + 
            new_exps[:, 2].astype(np.uint64))
    
    duration = time.perf_counter() - start
    return duration, keys, new_coeffs

# 3. Твій CUDA лончер (з динамічним блоком)
def test_cuda_launch(exp_A, exp_B, coeff_A, coeff_B):
    # Тут має бути твій виклик multivariateMulCUDA
    # Для тесту просто заміряємо час виконання з пересилкою даних
    start = time.perf_counter()
    
    d_exp_A = cuda.to_device(exp_A.flatten())
    d_exp_B = cuda.to_device(exp_B.flatten())
    d_coeff_A = cuda.to_device(coeff_A)
    d_coeff_B = cuda.to_device(coeff_B)
    
    # ... тут запуск кернела ...
    # cuda.synchronize()
    
    duration = time.perf_counter() - start
    return duration

if __name__ == "__main__":
    nA, nB = 100000, 100000
    exp_A, coeff_A = generate_test_poly(nA)
    exp_B, coeff_B = generate_test_poly(nB)

   # print(f"Testing multiplication: {nA} x {nB} = {nA*nB} terms")

    #t_cpu, keys_cpu, coeffs_cpu = multiply_pols_numpy(exp_A, coeff_A, exp_B, coeff_B)
    #print(f"CPU (NumPy) Time: {t_cpu*1000:.2f} ms")

    start = time.perf_counter()
    launch_cuda_kernel(exp_A, exp_B, coeff_A, coeff_B)
    duration = time.perf_counter() - start
    print(f"GPU (CUDA) Time: {duration*1000:.2f} ms")

