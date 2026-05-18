"""
cuda_kernels_PIGMDH.py
======================
Canonical kernel set after architecture merge (god_mode 3D grid +
flat_fused correct physics).

KEPT:
  - build_universal_physics_weights_kernel   (unchanged)
  - selective_winner_vector_hash_kernel      (unchanged)
  - calc_jacobian_row_nd                     (unchanged, called by merged kernel)
  - gmdh_merged_kernel                       (NEW — replaces both god_mode and flat)
  - solve_6x6_batched                        (fixed: symmetrise + partial pivot)

DELETED vs previous file:
  - gmdh_god_mode_kernel                     (superseded)
  - gmdh_flat_fused_kernel                   (superseded)
  - calc_jacobian_row_from_registers         (was only used by god_mode)
  - _write_back_to_global_optimized          (was only used by god_mode)
"""

import numpy as np
from numba import cuda, float64
import cupy as cp


# ---------------------------------------------------------------------------
# 1.  SYMBOLIC → NUMERIC WEIGHT BUILDER  (unchanged)
# ---------------------------------------------------------------------------

@cuda.jit
def build_universal_physics_weights_kernel(
    all_exps, all_coeffs, offsets,
    unique_exps,
    W_phi, W_dt, W_grad, W_lap,   # W_grad: (H_SIZE, n_polys, n_dim)
    hash_keys, n_polys, n_vars, time_idx, n_dim, MAX_EXP, H_SIZE
):
    """
    Fills weight tensors from polynomial coefficient/exponent buffers.

    KNOWN BUG (tracked, not fixed here — fix is in the evaluation loop):
      W_lap and W_dt store the *coefficient* of the derivative term but
      unique_exps still holds the original (unreduced) exponents.
      The evaluation loop in _precompute_all_models_fast must therefore
      build a separate V_sub with reduced exponents for W_lap and W_dt.
      See _precompute_all_models_fast for the corrected evaluation.
    """
    tid = cuda.threadIdx.x
    poly_idx = cuda.blockIdx.x
    if poly_idx >= n_polys:
        return

    EMPTY_KEY = np.uint64(0xFFFFFFFFFFFFFFFF)
    start    = offsets[poly_idx]
    n_terms  = offsets[poly_idx + 1] - start

    for i in range(tid, n_terms, cuda.blockDim.x):
        coeff = all_coeffs[start + i]
        if abs(coeff) < 1e-13:
            continue

        # Build deduplication key
        key = np.uint64(0)
        for d in range(n_vars):
            key = key * MAX_EXP + all_exps[start + i, d]

        h = np.int64(key % H_SIZE)
        for _ in range(H_SIZE):
            old = cuda.atomic.compare_and_swap(hash_keys[h:], EMPTY_KEY, key)
            if old == EMPTY_KEY or old == key:
                if old == EMPTY_KEY:
                    for d in range(n_vars):
                        unique_exps[h, d] = all_exps[start + i, d]

                cuda.atomic.add(W_phi, (h, poly_idx), coeff)

                p_t = all_exps[start + i, time_idx]
                if p_t > 0:
                    cuda.atomic.add(W_dt, (h, poly_idx), coeff * p_t)

                for d in range(n_dim):
                    p_d = all_exps[start + i, d]
                    if p_d > 0:
                        cuda.atomic.add(W_grad, (h, poly_idx, d), coeff * p_d)
                    if p_d > 1:
                        cuda.atomic.add(W_lap,  (h, poly_idx), coeff * p_d * (p_d - 1))
                break
            h = np.int64((h + 1) % H_SIZE)


# ---------------------------------------------------------------------------
# 2.  JACOBIAN ROW — register-only, fully unrolled per n_dim
#
#  Key change: no cuda.local.array.  All intermediate values are named
#  scalar registers.  The device function now RETURNS 6 scalars instead
#  of writing into an array argument, so the caller never needs A[].
#
#  Two specialisations: _2d (n_dim=2) and _3d (n_dim=3).
#  The merged kernel dispatches at compile time via the n_dim argument
#  using an if/else that Numba can constant-fold when the kernel is
#  specialised (see gmdh_merged_kernel_2d / _3d below).
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _jacobian_scalars_2d(
    sp, sdt, sg0, sg1, slap,       # S: value, dt, grad[0], grad[1], laplacian
    kp, kdt, kg0, kg1, klap,       # K: same
    ub0, ub1,                      # base-flow velocity components
    dudx_ii,                       # du_i/dx_i  (for linearisation)
    gs_i, gk_i,                    # ls_grad[i_comp], lk_grad[i_comp]
    nu, w_dat, w_div, w_mom
):
    """
    Returns (a0, a1, a2, a3, a4, a5) — all scalars, all in registers.
    Specialised for n_dim = 2.
    """
    sp2   = sp * sp
    kp2   = kp * kp
    sk    = sp * kp
    sk_dt = sdt * kp + sp * kdt

    # Convection sums — fully unrolled, no array needed
    conv_s  = ub0 * sg0 + ub1 * sg1
    conv_k  = ub0 * kg0 + ub1 * kg1
    sk_conv = ub0 * (sg0 * kp + sp * kg0) + ub1 * (sg1 * kp + sp * kg1)

    dot_sk  = sg0 * kg0 + sg1 * kg1
    dot_ss  = sg0 * sg0 + sg1 * sg1
    dot_kk  = kg0 * kg0 + kg1 * kg1

    sk_lap = slap * kp + sp * klap + 2.0 * dot_sk
    s2_lap = 2.0 * dot_ss + 2.0 * sp * slap
    k2_lap = 2.0 * dot_kk + 2.0 * kp * klap

    a0 = 0.0   # constant basis fn: zero physics derivative (pg_i is in b_train)
    a1 = w_dat * sp  + w_div * gs_i + w_mom * (sdt + conv_s - nu * slap)
    a2 = w_dat * kp  + w_div * gk_i + w_mom * (kdt + conv_k - nu * klap)
    a3 = w_dat * sk  + w_div * (gs_i * kp + sp * gk_i) + \
         w_mom * (sk_dt + sk_conv - nu * sk_lap)
    a4 = w_dat * sp2 + w_div * (2.0 * sp * gs_i) + \
         w_mom * (2.0 * sp * sdt + 2.0 * sp * conv_s - nu * s2_lap)
    a5 = w_dat * kp2 + w_div * (2.0 * kp * gk_i) + \
         w_mom * (2.0 * kp * kdt + 2.0 * kp * conv_k - nu * k2_lap)

    return a0, a1, a2, a3, a4, a5


@cuda.jit(device=True)
def _jacobian_scalars_3d(
    sp, sdt, sg0, sg1, sg2, slap,
    kp, kdt, kg0, kg1, kg2, klap,
    ub0, ub1, ub2,
    dudx_ii, gs_i, gk_i,
    nu, w_dat, w_div, w_mom
):
    """
    Returns (a0, a1, a2, a3, a4, a5) — all scalars, all in registers.
    Specialised for n_dim = 3.
    """
    sp2   = sp * sp
    kp2   = kp * kp
    sk    = sp * kp
    sk_dt = sdt * kp + sp * kdt

    conv_s  = ub0*sg0 + ub1*sg1 + ub2*sg2
    conv_k  = ub0*kg0 + ub1*kg1 + ub2*kg2
    sk_conv = ub0*(sg0*kp + sp*kg0) + ub1*(sg1*kp + sp*kg1) + ub2*(sg2*kp + sp*kg2)

    dot_sk  = sg0*kg0 + sg1*kg1 + sg2*kg2
    dot_ss  = sg0*sg0 + sg1*sg1 + sg2*sg2
    dot_kk  = kg0*kg0 + kg1*kg1 + kg2*kg2

    sk_lap = slap * kp + sp * klap + 2.0 * dot_sk
    s2_lap = 2.0 * dot_ss + 2.0 * sp * slap
    k2_lap = 2.0 * dot_kk + 2.0 * kp * klap

    a0 = 0.0
    a1 = w_dat * sp  + w_div * gs_i + w_mom * (sdt + conv_s - nu * slap)
    a2 = w_dat * kp  + w_div * gk_i + w_mom * (kdt + conv_k - nu * klap)
    a3 = w_dat * sk  + w_div * (gs_i * kp + sp * gk_i) + \
         w_mom * (sk_dt + sk_conv - nu * sk_lap)
    a4 = w_dat * sp2 + w_div * (2.0 * sp * gs_i) + \
         w_mom * (2.0 * sp * sdt + 2.0 * sp * conv_s - nu * s2_lap)
    a5 = w_dat * kp2 + w_div * (2.0 * kp * gk_i) + \
         w_mom * (2.0 * kp * kdt + 2.0 * kp * conv_k - nu * k2_lap)

    return a0, a1, a2, a3, a4, a5


# ---------------------------------------------------------------------------
# 3.  MERGED KERNEL — two compiled specialisations: 2-D and 3-D
#
#  Why two entry points instead of one kernel with an if/else on n_dim:
#    Numba traces each kernel once per unique argument *type*, not value.
#    An integer argument does not produce separate compiled paths for
#    n_dim=2 vs n_dim=3 — both end up in the same PTX with a runtime
#    branch, and the compiler cannot eliminate the dead loads.
#    Two entry points guarantee two separate PTX functions with the
#    dead dimension fully constant-folded away.
#
#  The Python dispatcher `gmdh_merged_kernel` below selects at call time.
# ---------------------------------------------------------------------------

@cuda.jit
def _gmdh_merged_2d(
    all_phi, all_dt, all_grad, all_lap,
    all_pg, y_true_all,
    nu, w_dat, w_div, w_mom,
    n_models, n_comp, curr_n, n_ks, k_map,
    out_XTX, out_XTy, out_mse,
    alphas
):
    """
    n_dim = 2 specialisation.  No cuda.local.array anywhere.

    Grid:  x → pos,  y → pair_idx (s*n_ks + k_local),  z → i_comp
    Block: (threads, 1, 1)

    Tensor layouts
    --------------
    all_phi   [n_models, n_comp,       curr_n]
    all_dt    [n_models, n_comp,       curr_n]
    all_grad  [n_models, n_comp, 2,    curr_n]
    all_lap   [n_models, n_comp,       curr_n]
    all_pg    [n_models, n_vars,       curr_n]  (axis-1 = n_vars; kernel reads 0..n_comp-1)
    y_true    [n_comp,                 curr_n]
    k_map     [n_models, n_ks]
    out_XTX   [n_models, n_models, n_comp, 6, 6]
    out_XTy   [n_models, n_models, n_comp, 6, 1]
    out_mse   [n_models, n_models, n_comp]
    alphas    [n_models, n_ks,     n_comp, 6]
    """
    pos      = cuda.grid(1)
    pair_idx = cuda.blockIdx.y
    i_comp   = cuda.blockIdx.z

    if pos >= curr_n or pair_idx >= n_models * n_ks or i_comp >= n_comp:
        return

    s_idx   = pair_idx // n_ks
    k_local = pair_idx  % n_ks
    k_real  = k_map[s_idx, k_local]

    # ---- shared memory reduction (one system per block) ----
    sh_XTX = cuda.shared.array(shape=(6, 6), dtype=float64)
    sh_XTy = cuda.shared.array(shape=(6,),   dtype=float64)
    sh_MSE = cuda.shared.array(shape=(1,),   dtype=float64)

    tid = cuda.threadIdx.x
    for i in range(tid, 36, cuda.blockDim.x):
        sh_XTX[i // 6, i % 6] = 0.0
    for i in range(tid, 6, cuda.blockDim.x):
        sh_XTy[i] = 0.0
    if tid == 0:
        sh_MSE[0] = 0.0
    cuda.syncthreads()

    # ---- register loads (all scalars, no local arrays) ----
    p = np.int64(pos)

    sp   = np.float64(all_phi[s_idx, i_comp, p])
    sdt  = np.float64(all_dt [s_idx, i_comp, p])
    slap = np.float64(all_lap[s_idx, i_comp, p])
    kp   = np.float64(all_phi[k_real, i_comp, p])
    kdt  = np.float64(all_dt [k_real, i_comp, p])
    klap = np.float64(all_lap[k_real, i_comp, p])
    pg_i = np.float64(all_pg[s_idx, i_comp, p])
    y_ti = np.float64(y_true_all[i_comp, p])

    # spatial gradients — 2 × 2 named scalars, zero arrays
    sg0 = np.float64(all_grad[s_idx, i_comp, 0, p])
    sg1 = np.float64(all_grad[s_idx, i_comp, 1, p])
    kg0 = np.float64(all_grad[k_real, i_comp, 0, p])
    kg1 = np.float64(all_grad[k_real, i_comp, 1, p])

    # base-flow velocity from S-model (spatial dims only)
    ub0 = np.float64(all_phi[s_idx, 0, p])
    ub1 = np.float64(all_phi[s_idx, 1, p])

    # du_i/dx_i for linearisation (safe: i_comp < 2 always in 2-D)
    dudx_ii = sg0 if i_comp == 0 else sg1

    # gs_i, gk_i = grad of S and K in the i_comp direction
    gs_i = sg0 if i_comp == 0 else sg1
    gk_i = kg0 if i_comp == 0 else kg1

    # convection of base (used in residual)
    conv_s = ub0 * sg0 + ub1 * sg1

    # ---- physics residual ----
    res_w   = (sp - y_ti) * w_dat \
            + dudx_ii     * w_div \
            + (sdt + conv_s - nu * slap + pg_i) * w_mom
    b_train = -res_w

    # ---- Jacobian row — 6 named scalars, zero arrays ----
    a0, a1, a2, a3, a4, a5 = _jacobian_scalars_2d(
        sp, sdt, sg0, sg1, slap,
        kp, kdt, kg0, kg1, klap,
        ub0, ub1,
        dudx_ii, gs_i, gk_i,
        nu, w_dat, w_div, w_mom
    )

    # ---- accumulate XTX (full 6×6) and XTy into shared ----
    # Fully unrolled outer product — no loop, no array index
    cuda.atomic.add(sh_XTy, 0, a0 * b_train)
    cuda.atomic.add(sh_XTy, 1, a1 * b_train)
    cuda.atomic.add(sh_XTy, 2, a2 * b_train)
    cuda.atomic.add(sh_XTy, 3, a3 * b_train)
    cuda.atomic.add(sh_XTy, 4, a4 * b_train)
    cuda.atomic.add(sh_XTy, 5, a5 * b_train)

    # Row 0
    cuda.atomic.add(sh_XTX, (0,0), a0*a0); cuda.atomic.add(sh_XTX, (0,1), a0*a1)
    cuda.atomic.add(sh_XTX, (0,2), a0*a2); cuda.atomic.add(sh_XTX, (0,3), a0*a3)
    cuda.atomic.add(sh_XTX, (0,4), a0*a4); cuda.atomic.add(sh_XTX, (0,5), a0*a5)
    # Row 1
    cuda.atomic.add(sh_XTX, (1,0), a1*a0); cuda.atomic.add(sh_XTX, (1,1), a1*a1)
    cuda.atomic.add(sh_XTX, (1,2), a1*a2); cuda.atomic.add(sh_XTX, (1,3), a1*a3)
    cuda.atomic.add(sh_XTX, (1,4), a1*a4); cuda.atomic.add(sh_XTX, (1,5), a1*a5)
    # Row 2
    cuda.atomic.add(sh_XTX, (2,0), a2*a0); cuda.atomic.add(sh_XTX, (2,1), a2*a1)
    cuda.atomic.add(sh_XTX, (2,2), a2*a2); cuda.atomic.add(sh_XTX, (2,3), a2*a3)
    cuda.atomic.add(sh_XTX, (2,4), a2*a4); cuda.atomic.add(sh_XTX, (2,5), a2*a5)
    # Row 3
    cuda.atomic.add(sh_XTX, (3,0), a3*a0); cuda.atomic.add(sh_XTX, (3,1), a3*a1)
    cuda.atomic.add(sh_XTX, (3,2), a3*a2); cuda.atomic.add(sh_XTX, (3,3), a3*a3)
    cuda.atomic.add(sh_XTX, (3,4), a3*a4); cuda.atomic.add(sh_XTX, (3,5), a3*a5)
    # Row 4
    cuda.atomic.add(sh_XTX, (4,0), a4*a0); cuda.atomic.add(sh_XTX, (4,1), a4*a1)
    cuda.atomic.add(sh_XTX, (4,2), a4*a2); cuda.atomic.add(sh_XTX, (4,3), a4*a3)
    cuda.atomic.add(sh_XTX, (4,4), a4*a4); cuda.atomic.add(sh_XTX, (4,5), a4*a5)
    # Row 5
    cuda.atomic.add(sh_XTX, (5,0), a5*a0); cuda.atomic.add(sh_XTX, (5,1), a5*a1)
    cuda.atomic.add(sh_XTX, (5,2), a5*a2); cuda.atomic.add(sh_XTX, (5,3), a5*a3)
    cuda.atomic.add(sh_XTX, (5,4), a5*a4); cuda.atomic.add(sh_XTX, (5,5), a5*a5)

    # ---- corrected MSE (sign fix: +) ----
    al0 = np.float64(alphas[s_idx, k_local, i_comp, 0])
    al1 = np.float64(alphas[s_idx, k_local, i_comp, 1])
    al2 = np.float64(alphas[s_idx, k_local, i_comp, 2])
    al3 = np.float64(alphas[s_idx, k_local, i_comp, 3])
    al4 = np.float64(alphas[s_idx, k_local, i_comp, 4])
    al5 = np.float64(alphas[s_idx, k_local, i_comp, 5])
    res_corr = res_w + al0*a0 + al1*a1 + al2*a2 + al3*a3 + al4*a4 + al5*a5
    cuda.atomic.add(sh_MSE, 0, res_corr * res_corr)

    cuda.syncthreads()

    # ---- write-back shared → global ----
    for i in range(tid, 6, cuda.blockDim.x):
        cuda.atomic.add(out_XTy, (s_idx, k_local, i_comp, i, 0), sh_XTy[i])
    for i in range(tid, 36, cuda.blockDim.x):
        r = i // 6; c = i % 6
        cuda.atomic.add(out_XTX, (s_idx, k_local, i_comp, r, c), sh_XTX[r, c])
    if tid == 0:
        cuda.atomic.add(out_mse, (s_idx, k_local, i_comp), sh_MSE[0])


@cuda.jit
def _gmdh_merged_3d(
    all_phi, all_dt, all_grad, all_lap,
    all_pg, y_true_all,
    nu, w_dat, w_div, w_mom,
    n_models, n_comp, curr_n, n_ks, k_map,
    out_XTX, out_XTy, out_mse,
    alphas
):
    """
    n_dim = 3 specialisation.  Identical structure to _2d; one extra
    spatial dimension loaded as named scalars sg2, kg2, ub2.
    """
    pos      = cuda.grid(1)
    pair_idx = cuda.blockIdx.y
    i_comp   = cuda.blockIdx.z

    if pos >= curr_n or pair_idx >= n_models * n_ks or i_comp >= n_comp:
        return

    s_idx   = pair_idx // n_ks
    k_local = pair_idx  % n_ks
    k_real  = k_map[s_idx, k_local]

    sh_XTX = cuda.shared.array(shape=(6, 6), dtype=float64)
    sh_XTy = cuda.shared.array(shape=(6,),   dtype=float64)
    sh_MSE = cuda.shared.array(shape=(1,),   dtype=float64)

    tid = cuda.threadIdx.x
    for i in range(tid, 36, cuda.blockDim.x):
        sh_XTX[i // 6, i % 6] = 0.0
    for i in range(tid, 6, cuda.blockDim.x):
        sh_XTy[i] = 0.0
    if tid == 0:
        sh_MSE[0] = 0.0
    cuda.syncthreads()

    p = np.int64(pos)

    sp   = np.float64(all_phi[s_idx, i_comp, p])
    sdt  = np.float64(all_dt [s_idx, i_comp, p])
    slap = np.float64(all_lap[s_idx, i_comp, p])
    kp   = np.float64(all_phi[k_real, i_comp, p])
    kdt  = np.float64(all_dt [k_real, i_comp, p])
    klap = np.float64(all_lap[k_real, i_comp, p])
    pg_i = np.float64(all_pg[s_idx, i_comp, p])
    y_ti = np.float64(y_true_all[i_comp, p])

    sg0 = np.float64(all_grad[s_idx, i_comp, 0, p])
    sg1 = np.float64(all_grad[s_idx, i_comp, 1, p])
    sg2 = np.float64(all_grad[s_idx, i_comp, 2, p])
    kg0 = np.float64(all_grad[k_real, i_comp, 0, p])
    kg1 = np.float64(all_grad[k_real, i_comp, 1, p])
    kg2 = np.float64(all_grad[k_real, i_comp, 2, p])

    ub0 = np.float64(all_phi[s_idx, 0, p])
    ub1 = np.float64(all_phi[s_idx, 1, p])
    ub2 = np.float64(all_phi[s_idx, 2, p])

    # i_comp-specific scalars via if/else (compile-time branch per specialisation)
    if i_comp == 0:
        dudx_ii = sg0; gs_i = sg0; gk_i = kg0
    elif i_comp == 1:
        dudx_ii = sg1; gs_i = sg1; gk_i = kg1
    else:
        dudx_ii = sg2; gs_i = sg2; gk_i = kg2

    conv_s = ub0*sg0 + ub1*sg1 + ub2*sg2

    res_w   = (sp - y_ti) * w_dat \
            + dudx_ii     * w_div \
            + (sdt + conv_s - nu * slap + pg_i) * w_mom
    b_train = -res_w

    a0, a1, a2, a3, a4, a5 = _jacobian_scalars_3d(
        sp, sdt, sg0, sg1, sg2, slap,
        kp, kdt, kg0, kg1, kg2, klap,
        ub0, ub1, ub2,
        dudx_ii, gs_i, gk_i,
        nu, w_dat, w_div, w_mom
    )

    cuda.atomic.add(sh_XTy, 0, a0*b_train); cuda.atomic.add(sh_XTy, 1, a1*b_train)
    cuda.atomic.add(sh_XTy, 2, a2*b_train); cuda.atomic.add(sh_XTy, 3, a3*b_train)
    cuda.atomic.add(sh_XTy, 4, a4*b_train); cuda.atomic.add(sh_XTy, 5, a5*b_train)

    cuda.atomic.add(sh_XTX, (0,0), a0*a0); cuda.atomic.add(sh_XTX, (0,1), a0*a1)
    cuda.atomic.add(sh_XTX, (0,2), a0*a2); cuda.atomic.add(sh_XTX, (0,3), a0*a3)
    cuda.atomic.add(sh_XTX, (0,4), a0*a4); cuda.atomic.add(sh_XTX, (0,5), a0*a5)
    cuda.atomic.add(sh_XTX, (1,0), a1*a0); cuda.atomic.add(sh_XTX, (1,1), a1*a1)
    cuda.atomic.add(sh_XTX, (1,2), a1*a2); cuda.atomic.add(sh_XTX, (1,3), a1*a3)
    cuda.atomic.add(sh_XTX, (1,4), a1*a4); cuda.atomic.add(sh_XTX, (1,5), a1*a5)
    cuda.atomic.add(sh_XTX, (2,0), a2*a0); cuda.atomic.add(sh_XTX, (2,1), a2*a1)
    cuda.atomic.add(sh_XTX, (2,2), a2*a2); cuda.atomic.add(sh_XTX, (2,3), a2*a3)
    cuda.atomic.add(sh_XTX, (2,4), a2*a4); cuda.atomic.add(sh_XTX, (2,5), a2*a5)
    cuda.atomic.add(sh_XTX, (3,0), a3*a0); cuda.atomic.add(sh_XTX, (3,1), a3*a1)
    cuda.atomic.add(sh_XTX, (3,2), a3*a2); cuda.atomic.add(sh_XTX, (3,3), a3*a3)
    cuda.atomic.add(sh_XTX, (3,4), a3*a4); cuda.atomic.add(sh_XTX, (3,5), a3*a5)
    cuda.atomic.add(sh_XTX, (4,0), a4*a0); cuda.atomic.add(sh_XTX, (4,1), a4*a1)
    cuda.atomic.add(sh_XTX, (4,2), a4*a2); cuda.atomic.add(sh_XTX, (4,3), a4*a3)
    cuda.atomic.add(sh_XTX, (4,4), a4*a4); cuda.atomic.add(sh_XTX, (4,5), a4*a5)
    cuda.atomic.add(sh_XTX, (5,0), a5*a0); cuda.atomic.add(sh_XTX, (5,1), a5*a1)
    cuda.atomic.add(sh_XTX, (5,2), a5*a2); cuda.atomic.add(sh_XTX, (5,3), a5*a3)
    cuda.atomic.add(sh_XTX, (5,4), a5*a4); cuda.atomic.add(sh_XTX, (5,5), a5*a5)

    al0 = np.float64(alphas[s_idx, k_local, i_comp, 0])
    al1 = np.float64(alphas[s_idx, k_local, i_comp, 1])
    al2 = np.float64(alphas[s_idx, k_local, i_comp, 2])
    al3 = np.float64(alphas[s_idx, k_local, i_comp, 3])
    al4 = np.float64(alphas[s_idx, k_local, i_comp, 4])
    al5 = np.float64(alphas[s_idx, k_local, i_comp, 5])
    res_corr = res_w + al0*a0 + al1*a1 + al2*a2 + al3*a3 + al4*a4 + al5*a5
    cuda.atomic.add(sh_MSE, 0, res_corr * res_corr)

    cuda.syncthreads()

    for i in range(tid, 6, cuda.blockDim.x):
        cuda.atomic.add(out_XTy, (s_idx, k_local, i_comp, i, 0), sh_XTy[i])
    for i in range(tid, 36, cuda.blockDim.x):
        r = i // 6; c = i % 6
        cuda.atomic.add(out_XTX, (s_idx, k_local, i_comp, r, c), sh_XTX[r, c])
    if tid == 0:
        cuda.atomic.add(out_mse, (s_idx, k_local, i_comp), sh_MSE[0])


def gmdh_merged_kernel(grid, block, n_dim,
                       all_phi, all_dt, all_grad, all_lap,
                       all_pg, y_true_all,
                       nu, w_dat, w_div, w_mom,
                       n_models, n_comp, curr_n, n_ks, k_map,
                       out_XTX, out_XTy, out_mse,
                       alphas):
    """
    Python dispatcher — selects the compiled 2-D or 3-D PTX specialisation.

    Usage (from _train_and_eval_layer):
        gmdh_merged_kernel(
            grid, block, n_dim=2,
            all_phi=..., ...
        )
    """
    args = (all_phi, all_dt, all_grad, all_lap,
            all_pg, y_true_all,
            nu, w_dat, w_div, w_mom,
            n_models, n_comp, curr_n, n_ks, k_map,
            out_XTX, out_XTy, out_mse, alphas)
    if n_dim == 2:
        _gmdh_merged_2d[grid, block](*args)
    elif n_dim == 3:
        _gmdh_merged_3d[grid, block](*args)
    else:
        raise ValueError(f"gmdh_merged_kernel: n_dim={n_dim} not supported (2 or 3)")


# ---------------------------------------------------------------------------
# 4.  POLYNOMIAL PRODUCT KERNEL  (unchanged)
# ---------------------------------------------------------------------------

@cuda.jit
def selective_winner_vector_hash_kernel(
    all_exps, all_coeffs, in_offsets,
    winner_map_sk,
    out_exps, out_coeffs, out_sizes,
    n_vars, n_winners
):
    """
    Computes monomial products (S*K, S*S, K*K) for winner polynomials.
    Hash-based deduplication in shared memory.
    EMPTY sentinel = 255 → hard constraint: max monomial degree per var = 254.
    """
    winner_idx = cuda.blockIdx.x
    if winner_idx >= n_winners:
        return

    HASH_SIZE = 1024
    EMPTY     = np.uint8(255)

    s_exps  = cuda.shared.array(shape=(1024, 8), dtype=np.uint8)
    s_vals  = cuda.shared.array(shape=(1024,),   dtype=float64)
    s_lock  = cuda.shared.array(shape=(1024,),   dtype=np.int32)
    s_count = cuda.shared.array(shape=(1,),      dtype=np.int32)
    s_ovfl  = cuda.shared.array(shape=(1,),      dtype=np.int32)

    tid = cuda.threadIdx.x

    for i in range(tid, HASH_SIZE, cuda.blockDim.x):
        s_lock[i] = 0
        s_vals[i] = 0.0
        for d in range(n_vars):
            s_exps[i, d] = EMPTY
    if tid == 0:
        s_count[0] = 0
        s_ovfl[0]  = 0
    cuda.syncthreads()

    s_id = winner_map_sk[winner_idx, 1]
    k_id = winner_map_sk[winner_idx, 2]
    start_s = in_offsets[s_id];   nA = in_offsets[s_id + 1] - start_s
    start_k = in_offsets[k_id];   nB = in_offsets[k_id + 1] - start_k

    for idx in range(tid, nA * nB, cuda.blockDim.x):
        i = idx % nA
        j = idx // nA
        c_prod = all_coeffs[start_s + i] * all_coeffs[start_k + j]
        if abs(c_prod) < 1e-13:
            continue

        curr_e = cuda.local.array(8, dtype=np.uint8)
        h = np.uint32(0)
        for d in range(n_vars):
            e_sum    = all_exps[start_s + i, d] + all_exps[start_k + j, d]
            curr_e[d] = e_sum
            h = (h + e_sum) + (h << 10)
            h ^= (h >> 6)
        h = (h + (h << 3)) ^ (h >> 11)
        h = (h + (h << 15)) % HASH_SIZE

        inserted = False
        for _ in range(HASH_SIZE):
            res = cuda.atomic.compare_and_swap(s_lock[h:], 0, 1)
            if res == 0:
                for d in range(n_vars):
                    s_exps[h, d] = curr_e[d]
                cuda.atomic.add(s_vals, h, c_prod)
                inserted = True
                break
            else:
                match = True
                for d in range(n_vars):
                    if s_exps[h, d] != curr_e[d]:
                        match = False
                        break
                if match:
                    cuda.atomic.add(s_vals, h, c_prod)
                    inserted = True
                    break
            h = (h + 1) % HASH_SIZE

        if not inserted:
            s_ovfl[0] = 1

    cuda.syncthreads()

    base_out = winner_idx * HASH_SIZE
    if s_ovfl[0] == 0:
        for i in range(tid, HASH_SIZE, cuda.blockDim.x):
            if s_lock[i] == 1 and abs(s_vals[i]) > 1e-12:
                write_pos = cuda.atomic.add(s_count, 0, 1)
                for d in range(n_vars):
                    out_exps[base_out + write_pos, d] = s_exps[i, d]
                out_coeffs[base_out + write_pos] = s_vals[i]

    cuda.syncthreads()
    if tid == 0:
        out_sizes[winner_idx] = -1 if s_ovfl[0] == 1 else s_count[0]


# ---------------------------------------------------------------------------
# 5.  BATCHED 6×6 SOLVER — shared-memory version, no cuda.local.array
#
#  Why shared memory instead of local (register-spill) arrays:
#    A 6×6 float64 matrix = 288 bytes.  With random loop-indexed access
#    the compiler cannot keep this in registers and spills to local memory
#    (= L1-backed global memory, ~100 cycle latency per access).
#    Moving A and b to shared memory gives explicit L1 placement (~4 cycles)
#    with no spill uncertainty.
#
#  Launch strategy:
#    One block = one linear system.
#    blockDim.x = 1  (single thread per system, no intra-block parallelism
#                     needed — Gauss-Jordan on 6×6 is 216 ops total).
#    Grid = (n_models * n_ks * n_comp, 1, 1)
#
#    With blockDim.x = 1 the SM can schedule many blocks concurrently,
#    giving the same aggregate throughput as a large thread-count launch
#    without the wasted threads from the original design.
#
#  Call from Python:
#    total = n_models * n_ks * n_comp
#    solve_6x6_batched[total, 1](XTX_klocal, XTy_klocal, alphas_out,
#                                n_models, n_ks, n_comp)
#
#    XTX_klocal must be shaped [n_models, n_ks, n_comp, 6, 6]
#    (second axis = k_local, not k_real — re-index on the host first).
# ---------------------------------------------------------------------------

@cuda.jit
def solve_6x6_batched(XTX, XTy, alphas_out, n_models, n_ks, n_comp):
    """
    One block (single thread) → one (s_idx, k_local, i_comp) system.

    Shared memory layout per block:
        sh_A  [6, 6]   — the 6×6 system matrix (lives in L1)
        sh_b  [6]      — right-hand side + solution in-place

    No cuda.local.array anywhere.
    """
    sys_idx = cuda.blockIdx.x        # one block per system
    total   = n_models * n_ks * n_comp
    if sys_idx >= total:
        return

    # Shared arrays — allocated once per block, live in L1
    sh_A = cuda.shared.array(shape=(6, 6), dtype=float64)
    sh_b = cuda.shared.array(shape=(6,),   dtype=float64)

    # Decode flat index
    i_comp  = sys_idx % n_comp
    tmp     = sys_idx // n_comp
    k_local = tmp % n_ks
    s_idx   = tmp // n_ks

    # ---- load ----
    for r in range(6):
        sh_b[r] = XTy[s_idx, k_local, i_comp, r, 0]
        for c in range(6):
            sh_A[r, c] = XTX[s_idx, k_local, i_comp, r, c]

    # ---- symmetrise (safety net) ----
    for r in range(6):
        for c in range(r):
            sh_A[r, c] = sh_A[c, r]

    # ---- Gauss-Jordan with partial pivoting ----
    for i in range(6):

        # Find pivot row in column i
        max_val = abs(sh_A[i, i])
        max_row = i
        for rr in range(i + 1, 6):
            v = abs(sh_A[rr, i])
            if v > max_val:
                max_val = v
                max_row = rr

        # Swap rows i ↔ max_row
        if max_row != i:
            for c in range(6):
                tmp_v          = sh_A[i, c]
                sh_A[i, c]     = sh_A[max_row, c]
                sh_A[max_row, c] = tmp_v
            tmp_v      = sh_b[i]
            sh_b[i]    = sh_b[max_row]
            sh_b[max_row] = tmp_v

        pivot = sh_A[i, i]
        if abs(pivot) < 1e-12:        # rank-deficient column → zero solution row
            sh_b[i] = 0.0
            for c in range(6):
                sh_A[i, c] = 0.0
            continue

        inv_p = 1.0 / pivot
        for c in range(i, 6):
            sh_A[i, c] *= inv_p
        sh_b[i] *= inv_p

        for r in range(6):
            if r == i:
                continue
            f = sh_A[r, i]
            if abs(f) < 1e-15:
                continue
            for c in range(i, 6):
                sh_A[r, c] -= f * sh_A[i, c]
            sh_b[r] -= f * sh_b[i]

    # ---- store ----
    for i in range(6):
        alphas_out[s_idx, k_local, i_comp, i] = sh_b[i]