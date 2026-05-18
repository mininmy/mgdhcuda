import numpy as np
from numba import cuda, float64
import cupy as cp
from math import isnan   


@cuda.jit
def selective_winner_vector_hash_kernel(
    all_exps, all_coeffs, in_offsets,
    winner_map_sk,
    out_exps, out_coeffs, out_sizes,
    n_vars, n_winners
):
    winner_idx = cuda.blockIdx.x
    if winner_idx >= n_winners: return

    # --- 1. Параметры Хеша ---
    HASH_SIZE = 1024
    EMPTY = 255 # Маркер пустого слота (для uint8)
    
    # Резервируем Shared Memory
    # (HASH_SIZE, 8) для экспонент, (HASH_SIZE) для коэффициентов
    s_exps = cuda.shared.array(shape=(1024, 8), dtype=np.uint8)
    s_vals = cuda.shared.array(shape=(1024), dtype=np.float64)
    s_lock = cuda.shared.array(shape=(1024), dtype=np.int32) # Для атомарного захвата

    s_local_count = cuda.shared.array(1, dtype=np.int32)
    s_overflow = cuda.shared.array(1, dtype=np.int32)

    tid = cuda.threadIdx.x
    
    # Инициализация всей таблицы
    for i in range(tid, HASH_SIZE, cuda.blockDim.x):
        s_lock[i] = 0
        s_vals[i] = 0.0
        for d in range(n_vars):
            s_exps[i, d] = EMPTY
            
    if tid == 0:
        s_local_count[0] = 0
        s_overflow[0] = 0
    cuda.syncthreads()

    # --- 2. Multiply + Vector Hash ---
    s_id, k_id = winner_map_sk[winner_idx, 1], winner_map_sk[winner_idx, 2]
    start_s, nA = in_offsets[s_id], in_offsets[s_id+1] - in_offsets[s_id]
    start_k, nB = in_offsets[k_id], in_offsets[k_id+1] - in_offsets[k_id]

    for idx in range(tid, nA * nB, cuda.blockDim.x):
        i, j = idx % nA, idx // nA
        c_prod = all_coeffs[start_s + i] * all_coeffs[start_k + j]
        if abs(c_prod) < 1e-13: continue

        # Локальный вектор текущих экспонент
        curr_e = cuda.local.array(8, dtype=np.uint8)
        h = np.uint32(0)
        for d in range(n_vars):
            e_sum = all_exps[start_s + i, d] + all_exps[start_k + j, d]
            curr_e[d] = e_sum
            # Lightweight Hash (Jenkins-style)
            h = (h + e_sum) + (h << 10)
            h ^= (h >> 6)
        
        h = (h + (h << 3)) ^ (h >> 11)
        h = (h + (h << 15)) % HASH_SIZE

        inserted = False
        for attempt in range(HASH_SIZE):
            # АТОМАРНЫЙ ЗАХВАТ СЛОТА (Locking)
            # Если 0 -> меняем на 1. Если уже 1 -> проверяем на совпадение.
            res = cuda.atomic.compare_and_swap(s_lock, h, 0, 1)
            
            if res == 0: # Мы первые захватили слот!
                for d in range(n_vars): s_exps[h, d] = curr_e[d]
                cuda.atomic.add(s_vals, h, c_prod)
                inserted = True
                break
            else: # Слот уже занят, проверяем: наш ли это вектор?
                match = True
                for d in range(n_vars):
                    if s_exps[h, d] != curr_e[d]:
                        match = False
                        break
                if match:
                    cuda.atomic.add(s_vals, h, c_prod)
                    inserted = True
                    break
            
            h = (h + 1) % HASH_SIZE # Линейное пробирование
            
        if not inserted: s_overflow[0] = 1

    cuda.syncthreads()

    # --- 3. Parallel Write-Back ---
    base_out = winner_idx * HASH_SIZE
    if s_overflow[0] == 0:
        for i in range(tid, HASH_SIZE, cuda.blockDim.x):
            if s_lock[i] == 1 and abs(s_vals[i]) > 1e-12:
                idx_to_write = cuda.atomic.add(s_local_count, 0, 1)
                for d in range(n_vars):
                    out_exps[base_out + idx_to_write, d] = s_exps[i, d]
                out_coeffs[base_out + idx_to_write] = s_vals[i]

    cuda.syncthreads()
    if tid == 0:
        out_sizes[winner_idx] = -1 if s_overflow[0] == 1 else s_local_count[0]

@cuda.jit(device=True)
def calc_jacobian_row_nd(sp, sdt, ls_grad, slap, 
                         kp, kdt, lk_grad, klap, 
                         u_b, pos, du_i_dx_i, 
                         nu, w_dat, w_div, w_mom, 
                         i_comp, n_dim, A):
    # Попереднє обчислення для економії ~1.8 млрд інструкцій на шар
    sp2 = sp * sp
    kp2 = kp * kp
    sk  = sp * kp
    sk_dt = sdt * kp + sp * kdt
    
    conv_s, conv_k, sk_conv = 0.0, 0.0, 0.0
    dot_grad_sk, dot_grad_ss, dot_grad_kk = 0.0, 0.0, 0.0
    
    for d in range(n_dim):
        gs, gk, ub = ls_grad[d], lk_grad[d], u_b[d, pos]
        conv_s += ub * gs
        conv_k += ub * gk
        sk_conv += ub * (gs * kp + sp * gk)
        dot_grad_sk += gs * gk
        dot_grad_ss += gs * gs
        dot_grad_kk += gk * gk

    sk_lap = slap * kp + sp * klap + 2.0 * dot_grad_sk
    s2_lap = 2.0 * dot_grad_ss + 2.0 * sp * slap
    k2_lap = 2.0 * dot_grad_kk + 2.0 * kp * klap

    # Формування 6 колонок Якобіана (A_j = Data + Div + Mom)
    A[0] = (1.0 * w_dat) + (sp * du_i_dx_i * w_mom)
    A[1] = (sp * w_dat) + (ls_grad[i_comp] * w_div) + ((sdt + conv_s + sp * du_i_dx_i - nu * slap) * w_mom)
    A[2] = (kp * w_dat) + (lk_grad[i_comp] * w_div) + ((kdt + conv_k + kp * du_i_dx_i - nu * klap) * w_mom)
    A[3] = (sk * w_dat) + ((ls_grad[i_comp]*kp + sp*lk_grad[i_comp]) * w_div) + ((sk_dt + sk_conv + sk * du_i_dx_i - nu * sk_lap) * w_mom)
    A[4] = (sp2 * w_dat) + (2.0 * sp * ls_grad[i_comp] * w_div) + ((2.0 * sp * sdt + 2.0 * sp * conv_s + sp2 * du_i_dx_i - nu * s2_lap) * w_mom)
    A[5] = (kp2 * w_dat) + (2.0 * kp * lk_grad[i_comp] * w_div) + ((2.0 * kp * kdt + 2.0 * kp * conv_k + kp2 * du_i_dx_i - nu * k2_lap) * w_mom)



@cuda.jit
def gmdh_flat_fused_kernel(
    all_phi, all_dt, all_grad, all_lap,
    u_b, du_i_dx_i, pg_i, y_t,
    alphas, nu, w_dat, w_div, w_mom,
    s_idx, i_comp, n_dim, curr_n, n_ks, k_map,
    out_XTX, out_XTy, out_mse
):
    # Глобальний індекс роботи (0 ... curr_n * n_ks - 1)
    idx = cuda.grid(1)
    
    if idx < curr_n * n_ks:
        # Розпаковуємо: яку точку (pos) і яку модель K рахує цей потік
        k_local_idx = idx // curr_n
        pos = idx % curr_n
        
        # 1. ЗАВАНТАЖЕННЯ ДАНИХ (Registers)
        # S-база
        sp = all_phi[s_idx, i_comp, pos]
        sdt = all_dt[s_idx, i_comp, pos]
        slap = all_lap[s_idx, i_comp, pos]
        dudx_ii = du_i_dx_i[pos]
        
        # K-модель
        kp = all_phi[k_local_idx, i_comp, pos]
        kdt = all_dt[k_local_idx, i_comp, pos]
        klap = all_lap[k_local_idx, i_comp, pos]

        # Градієнти (Local Arrays)
        ls_grad = cuda.local.array(3, dtype=np.float64)
        lk_grad = cuda.local.array(3, dtype=np.float64)
        for d in range(n_dim): 
            ls_grad[d] = all_grad[s_idx, i_comp, d, pos]
            lk_grad[d] = all_grad[k_local_idx, i_comp, d, pos]

        # 2. РОЗРАХУНОК НЕВЯЗОК (Residuals)
        conv_s = 0.0
        for d in range(n_dim): conv_s += u_b[d, pos] * ls_grad[d]
        
        res_w = (sp - y_t[pos]) * w_dat + ls_grad[i_comp] * w_div + (sdt + conv_s - nu * slap + pg_i[pos]) * w_mom
        b_train, b_eval = -res_w, res_w

        # 3. РОЗРАХУНОК ЯКОБІАНА
        A = cuda.local.array(6, dtype=np.float64)
        calc_jacobian_row_nd(sp, sdt, ls_grad, slap, kp, kdt, lk_grad, klap, 
                             u_b, pos, dudx_ii, nu, w_dat, w_div, w_mom, i_comp, n_dim, A)

        # 4. АКУМУЛЯЦІЯ (Atomic Global)
        k_real_idx = k_map[k_local_idx]
        
        # Тренування
        for r in range(6):
            cuda.atomic.add(out_XTy, (s_idx, k_real_idx, i_comp, r, 0), A[r] * b_train)
            for c in range(r, 6):
                cuda.atomic.add(out_XTX, (s_idx, k_real_idx, i_comp, r, c), A[r] * A[c])
        
        # Валідація
        res_f = b_eval
        for j in range(6): res_f += alphas[k_local_idx, j] * A[j]
        cuda.atomic.add(out_mse, (s_idx, k_real_idx, i_comp), res_f * res_f)
