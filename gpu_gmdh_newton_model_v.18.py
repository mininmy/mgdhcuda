"""
gpu_gmdh_newton_model_v8.py
============================
Conventional monomial basis GMDH.
OOM-safe: runs without pruning at many layers and large top_models.

Key memory fixes applied to this version
-----------------------------------------
1. W_dt / W_grad / W_lap in _eval_poly_set are stub allocations (1×1).
   They are required by the kernel signature but never read afterwards.
   Replaces ~2 GB wasted per call with negligible allocations.

2. POLY_CHUNK=32: polynomials processed in blocks so H_SIZE (hash table)
   is bounded to the unique-monomial count of one chunk, not all n_polys.
   Reduces W_phi allocation ~30× at deep layers.

3. J col 0 = zeros (constant basis function has zero physics derivative).
"""

import warnings
from numba import NumbaPerformanceWarning
import time
import cupy  as cp
import cupyx
import numpy as np
from numba import cuda, float64

from cuda_kernels_PIGMDH import build_universal_physics_weights_kernel
from gpu_polynomial_module import PolynomialGPU

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

PRUNE_THRESH = 0


EXP_BASE = 65536  # base for exponent-tuple hashing; must exceed max exponent
                  # (uint16 max). Constrains n_vars <= 4 so EXP_BASE**n_vars
                  # fits in uint64. Update if you ever push to n_vars > 4.

def _reduce_poly_gpu(exps_gpu, coeffs_gpu, threshold=0):
    if exps_gpu.shape[0] == 0:
        return exps_gpu, coeffs_gpu
    n_terms, n_vars = exps_gpu.shape
    keys = cp.zeros(n_terms, dtype=cp.uint64)
    for i in range(n_vars):
        keys = keys * cp.uint64(EXP_BASE) + exps_gpu[:, i].astype(cp.uint64)
    order       = cp.argsort(keys)
    sorted_keys = keys[order]; sorted_coef = coeffs_gpu[order]; sorted_exps = exps_gpu[order]
    boundary    = cp.empty(n_terms, dtype=cp.bool_)
    boundary[0] = True; boundary[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_ids = cp.cumsum(boundary) - 1
    n_unique  = int(group_ids[-1]) + 1
    red_coeffs = cp.zeros(n_unique, dtype=cp.float64)
    cupyx.scatter_add(red_coeffs, group_ids, sorted_coef)
    unique_pos = cp.where(boundary)[0]; red_exps = sorted_exps[unique_pos]
    keep = cp.abs(red_coeffs) > threshold
    return red_exps[keep], red_coeffs[keep]

def _poly_zero(n_vars):
    return PolynomialGPU(cp.zeros((0, n_vars), dtype=cp.uint16), cp.zeros(0, dtype=cp.float64))
def _poly_constant(value, n_vars):
    return PolynomialGPU(cp.zeros((1, n_vars), dtype=cp.uint16), cp.array([float(value)], dtype=cp.float64))
def _poly_variable(var_idx, n_vars):
    exps = cp.zeros((1, n_vars), dtype=cp.uint16); exps[0, var_idx] = 1
    return PolynomialGPU(exps, cp.ones(1, dtype=cp.float64))
def _poly_scale(p, alpha):
    if p.exponents.shape[0] == 0: return p
    return PolynomialGPU(p.exponents.copy(), p.coeffs * np.float64(alpha))
def _poly_add(p1, p2):
    if p1.exponents.shape[0] == 0: return p2
    if p2.exponents.shape[0] == 0: return p1
    re, rc = _reduce_poly_gpu(cp.concatenate([p1.exponents, p2.exponents], axis=0),
                               cp.concatenate([p1.coeffs,    p2.coeffs]))
    return PolynomialGPU(re, rc)

MUL_MAX_TERMS = 7000
MUL_PRUNE     = 0

def _poly_mul(p1, p2):
    if p1.exponents.shape[0] == 0 or p2.exponents.shape[0] == 0:
        return _poly_zero(p1.nvars)
    p1 = p1.prune(MUL_PRUNE); p2 = p2.prune(MUL_PRUNE)
    if p1.exponents.shape[0] == 0 or p2.exponents.shape[0] == 0:
        return _poly_zero(p1.nvars)
    def _cap(p):
        if p.exponents.shape[0] <= MUL_MAX_TERMS: return p
        p._ensure_cpu_cache()
        order = np.argsort(-np.abs(p._cpu_coeffs))[:MUL_MAX_TERMS]
        return PolynomialGPU(cp.asarray(p._cpu_exps[order]), cp.asarray(p._cpu_coeffs[order]))
    p1, p2 = _cap(p1), _cap(p2)
    p1._ensure_cpu_cache(); p2._ensure_cpu_cache()
    ea, ca = p1._cpu_exps, p1._cpu_coeffs
    eb, cb = p2._cpu_exps, p2._cpu_coeffs
    na, nv = ea.shape; nb = eb.shape[0]
    # int32 sum: two uint16 exponents (max 65535 each) overflow int16.
    prod_exps   = (ea[:, np.newaxis, :].astype(np.int32) +
                   eb[np.newaxis, :, :].astype(np.int32)).reshape(na * nb, nv)
    # Normalise inputs before outer product to prevent float64 overflow.
    # Coefficients can reach 1e200+ at deep layers without pruning;
    # their product would exceed float64 max (~1.8e308).
    # Strategy: divide each input by its max |coeff|, form the product,
    # then rescale back. If the combined scale itself overflows float64,
    # keep the normalised product (preserves coefficient ratios, which is
    # what the LS solver needs; absolute scale is absorbed by alpha).
    _m1 = np.max(np.abs(ca)) if len(ca) > 0 else 1.0
    _m2 = np.max(np.abs(cb)) if len(cb) > 0 else 1.0
    _m1 = _m1 if _m1 > 0 else 1.0
    _m2 = _m2 if _m2 > 0 else 1.0
    _prod_normed = (ca / _m1)[:, np.newaxis] * (cb / _m2)[np.newaxis, :]
    _scale = _m1 * _m2
    with np.errstate(over='ignore'):
        prod_coeffs = (_prod_normed.flatten() * _scale
                       if np.isfinite(_scale)
                       else _prod_normed.flatten())
    # Cap at uint16 max. Above this we can't store the exponent at all, so we
    # have to drop the term — but log loudly, since silent drops at deep layers
    # were the previous bug. With the cap raised from 255 to 65535 this should
    # essentially never trigger except in pathological runs.
    valid = (prod_exps.max(axis=1) <= 65535)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        warnings.warn(
            f"_poly_mul: dropped {n_dropped}/{len(valid)} terms with exponent > 65535",
            stacklevel=2,
        )
    prod_exps = prod_exps[valid].astype(np.uint16); prod_coeffs = prod_coeffs[valid]
    if len(prod_coeffs) == 0: return _poly_zero(nv)
    if len(prod_coeffs) > 4_000_000:
        keys = np.zeros(len(prod_exps), dtype=np.uint64)
        for i in range(nv):
            keys = keys * np.uint64(EXP_BASE) + prod_exps[:, i].astype(np.uint64)
        order = np.argsort(keys)
        sk, sc, se = keys[order], prod_coeffs[order], prod_exps[order]
        boundary = np.empty(len(sk), dtype=bool)
        boundary[0] = True; boundary[1:] = sk[1:] != sk[:-1]
        gids = np.cumsum(boundary) - 1; n_u = int(gids[-1]) + 1
        red_c = np.zeros(n_u); np.add.at(red_c, gids, sc)
        red_e = se[np.where(boundary)[0]]
        keep  = np.abs(red_c) > MUL_PRUNE
        return PolynomialGPU(cp.asarray(red_e[keep]), cp.asarray(red_c[keep]))
    re, rc = _reduce_poly_gpu(cp.asarray(prod_exps), cp.asarray(prod_coeffs))
    return PolynomialGPU(re, rc)


@cuda.jit
def fast_eval_poly_kernel(X, exps, coeffs, offsets, out):
    pos = cuda.grid(1); poly_idx = cuda.blockIdx.y
    N = X.shape[0]; n_vars = X.shape[1]; n_polys = offsets.shape[0] - 1
    if pos >= N or poly_idx >= n_polys: return
    start = offsets[poly_idx]; end = offsets[poly_idx + 1]
    acc = np.float64(0.0)
    for i in range(start, end):
        term = np.float64(coeffs[i])
        for d in range(n_vars):
            e = np.int32(exps[i, d])
            if e == 0: pass
            elif e == 1: term *= np.float64(X[pos, d])
            else:
                x = np.float64(X[pos, d]); xp = x * x; k = np.int32(2)
                while k < e: xp *= x; k += 1
                term *= xp
        acc += term
    out[poly_idx, pos] = acc


class GMDHTrainerGPU:
    def __init__(self, viscosity, chunk_size=500_000, weights=None,
                 top_models=10, prune_thresh=PRUNE_THRESH,
                 qr_sub_size=2000, jac_sys_chunk=512,
                 svd_rcond=1e-10, corr_threshold=1.0):
        self.viscosity      = viscosity
        self.chunk_size     = chunk_size
        self.weights        = weights or {"data": 1.0, "div": 1.0, "mom": 1.0}
        self.top_models     = top_models
        self.prune_thresh   = prune_thresh
        self.qr_sub_size    = qr_sub_size
        self.svd_rcond      = svd_rcond
        self.time_scale     = 1.0
        self.current_models = []
        self.best_model     = []   # best poly per component after training
        self.jac_sys_chunk  = jac_sys_chunk
        self.corr_threshold = corr_threshold

    def _initialize_layer_zero(self):
        models = []
        for v_idx in range(self.n_vars):
            models.append([_poly_variable(v_idx, self.n_vars)
                           for _ in range(self.n_comp)])
        return models

    # ------------------------------------------------------------------
    def _log_best_model(self, best, layer_idx):
        print(f"\n=== BEST MODEL @ Layer {layer_idx} ===")
        print(f"s={best['s']}  k={best['k']}  i={best['i']}  "
              f"combined_err={best['combined_err']:.6e}  fitted_err={best['err']:.6e}")
        print(f"  alpha={best['alpha']}")

    # ------------------------------------------------------------------
    def fit(self, X_cpu, y_target_cpu, n_layers=11):
        X_tr = cp.asarray(X_cpu, dtype=cp.float64)
        y_tr = [cp.asarray(yc, dtype=cp.float64) for yc in y_target_cpu]
        self.n_vars   = X_tr.shape[1]
        self.n_comp   = len(y_tr)
        self.time_idx = self.n_vars - 1
        self.current_models = self._initialize_layer_zero()
        best_rmse = float('inf'); best_layer = -1
        prev_avg_rmse = float('inf')

        for layer in range(n_layers):
            t_start = time.time()
            all_pg_polys = self._compute_pressure_grad_polys()
            candidates, current_rmse = self._train_and_eval_layer(X_tr, y_tr, all_pg_polys)
            t_layer = time.time() - t_start

            # Each candidate changes only one component (ic). Score by:
            #   combined_err = (fitted_err_ic + sum of current RMSE for other
            #                   components from parent model s) / n_comp
            # This accounts for the unchanged components' quality during selection.
            for c in candidates:
                s_idx = c['s']
                ic    = c['i']
                other = sum(float(current_rmse[s_idx, ic2])
                            for ic2 in range(self.n_comp) if ic2 != ic)
                c['combined_err'] = (c['err'] + other) / self.n_comp

            candidates.sort(key=lambda x: x['combined_err'])
            winners = candidates[:self.top_models]

            if not winners:
                print("  → No candidates survived; stopping.")
                break

            layer_rmse = winners[0]['combined_err']
            avg_rmse   = sum(w['combined_err'] for w in winners) / max(len(winners), 1)
            print(f"Layer {layer} | Best combined RMSE: {layer_rmse:.6e} | "
                  f"Avg combined RMSE (top): {avg_rmse:.6e} | "
                  f"Time: {t_layer:.2f}s | n_models: {len(self.current_models)}")

            self._log_best_model(winners[0], layer)

            if avg_rmse > prev_avg_rmse:
                print(f"  → Avg top RMSE increased "
                      f"({prev_avg_rmse:.6e} → {avg_rmse:.6e}). Stopping.")
                break

            best_rmse = layer_rmse; best_layer = layer
            prev_avg_rmse = avg_rmse
            self.current_models = self._assemble(winners)
            cp.get_default_memory_pool().free_all_blocks()

        print(f"Training complete. Best RMSE: {best_rmse:.6e} at layer {best_layer}.")

    # ------------------------------------------------------------------
    def _compute_pressure_grad_polys(self, models=None):
        n_dim = self.n_vars - 1; all_pg = []
        if models is None: models = self.current_models
        for model in models:
            p_grad_i_list = []
            for i in range(n_dim):
                u_i  = model[i]
                pg_i = _poly_scale(u_i.differentiate(self.time_idx), -self.time_scale)
                for j in range(n_dim):
                    u_j = model[j]; du_i_dj = u_i.differentiate(j)
                    if u_j.exponents.shape[0] > 0 and du_i_dj.exponents.shape[0] > 0:
                        pg_i = _poly_add(pg_i, _poly_scale(_poly_mul(u_j, du_i_dj), -1.0))
                for j in range(n_dim):
                    d2 = u_i.differentiate(j).differentiate(j)
                    if d2.exponents.shape[0] > 0:
                        pg_i = _poly_add(pg_i, _poly_scale(d2, self.viscosity))
                p_grad_i_list.append(pg_i.prune(1e-10))
            p_exps_list = []; p_coeffs_list = []
            for i, pg_i in enumerate(p_grad_i_list):
                if pg_i.exponents.shape[0] == 0: continue
                anti_i = pg_i.integrate(i)
                if anti_i.exponents.shape[0] == 0: continue
                anti_i._ensure_cpu_cache()
                ae = anti_i._cpu_exps; ac = anti_i._cpu_coeffs
                m = np.maximum(np.sum(ae[:, :n_dim] > 0, axis=1), 1)
                p_exps_list.append(ae); p_coeffs_list.append(ac / m.astype(np.float64))
            if p_exps_list:
                pe, pc = _reduce_poly_gpu(
                    cp.asarray(np.concatenate(p_exps_list, axis=0).astype(np.uint16)),
                    cp.asarray(np.concatenate(p_coeffs_list, axis=0)))
                p_poly = PolynomialGPU(pe, pc)
            else:
                p_poly = _poly_zero(self.n_vars)
            pg_model = []
            for d in range(n_dim):
                dpd = p_poly.differentiate(d).prune(1e-10); dpd.sync_to_cpu()
                pg_model.append(dpd)
            all_pg.append(pg_model)
        return all_pg

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_correlation_mask(phi_sample, s_gpu, kr_gpu, ic_gpu,
                                  threshold, sys_chunk=1024):
        n_sys = s_gpu.shape[0]; corr = cp.zeros(n_sys, dtype=cp.float64)
        for ss in range(0, n_sys, sys_chunk):
            se = min(ss + sys_chunk, n_sys)
            sp = phi_sample[s_gpu[ss:se],  ic_gpu[ss:se]]
            kp = phi_sample[kr_gpu[ss:se], ic_gpu[ss:se]]
            sp_c = sp - sp.mean(axis=1, keepdims=True)
            kp_c = kp - kp.mean(axis=1, keepdims=True)
            num  = (sp_c * kp_c).sum(axis=1)
            den  = cp.sqrt((sp_c**2).sum(axis=1) * (kp_c**2).sum(axis=1))
            corr[ss:se] = cp.where(den > 1e-12, num / den, cp.float64(0.0))
        return (cp.abs(corr) < cp.float64(threshold)).get()

    # ------------------------------------------------------------------
    def _train_and_eval_layer(self, X, y, all_pg_polys):
        n_total  = X.shape[0]; n_models = len(self.current_models)
        n_dim    = self.n_vars - 1; n_ks = n_models - 1
        n_sys    = n_models * n_ks * self.n_comp

        k_map_cpu = np.zeros((n_models, n_ks), dtype=np.int32)
        for s in range(n_models):
            for loc, real in enumerate(k for k in range(n_models) if k != s):
                k_map_cpu[s, loc] = real

        s_arr  = np.repeat(np.arange(n_models), n_ks * self.n_comp)
        kl_arr = np.tile(np.repeat(np.arange(n_ks), self.n_comp), n_models)
        ic_arr = np.tile(np.arange(self.n_comp), n_models * n_ks)
        kr_arr = k_map_cpu[s_arr, kl_arr]

        # Correlation filter
        if self.corr_threshold < 1.0:
            _Xs = X[:min(self.chunk_size, n_total)]
            _phi, _, _, _ = self._precompute_all_models_fast(_Xs, self.current_models)
            _sg = cp.asarray(s_arr, dtype=cp.int32)
            _ig = cp.asarray(ic_arr, dtype=cp.int32)
            _kg = cp.asarray(kr_arr, dtype=cp.int32)
            valid = self._compute_correlation_mask(
                _phi, _sg, _kg, _ig, self.corr_threshold, self.jac_sys_chunk)
            del _phi, _sg, _ig, _kg; cp.get_default_memory_pool().free_all_blocks()
            nd = int(np.sum(~valid))
            if nd > 0:
                print(f'    Corr filter: dropped {nd}/{n_sys} pairs (|r|>={self.corr_threshold})')
            s_arr  = s_arr[valid];  kl_arr = kl_arr[valid]
            ic_arr = ic_arr[valid]; kr_arr = kr_arr[valid]
            n_sys  = int(valid.sum())

        s_gpu      = cp.asarray(s_arr,  dtype=cp.int32)
        ic_gpu     = cp.asarray(ic_arr, dtype=cp.int32)
        kr_gpu     = cp.asarray(kr_arr, dtype=cp.int32)
        ic_clamped = cp.minimum(ic_gpu, cp.int32(n_dim - 1))

        curr_sq = cp.zeros((n_models, self.n_comp), dtype=cp.float64)

        # Three-stage estimation order:
        #   1) alpha_0 (intercept)
        #   2) linear terms only: [s, k]
        #   3) nonlinear terms:   [sk, s^2, k^2]
        sum_b   = cp.zeros(n_sys,         dtype=cp.float64)  # Σ b
        bsq     = cp.zeros(n_sys,         dtype=cp.float64)  # ||b||^2
        XTX_lin = cp.zeros((n_sys, 2, 2), dtype=cp.float64)  # [s,k]^T [s,k]
        XTy_lin = cp.zeros((n_sys, 2),    dtype=cp.float64)  # [s,k]^T b
        sum_lin = cp.zeros((n_sys, 2),    dtype=cp.float64)  # Σ [s,k]
        XTX_non = cp.zeros((n_sys, 3, 3), dtype=cp.float64)  # [sk,ss,kk]^T [...]
        XTy_non = cp.zeros((n_sys, 3),    dtype=cp.float64)  # [sk,ss,kk]^T b
        sum_non = cp.zeros((n_sys, 3),    dtype=cp.float64)  # Σ [sk,ss,kk]
        XTX_non_lin = cp.zeros((n_sys, 3, 2), dtype=cp.float64)  # non^T lin

        for start in range(0, n_total, self.chunk_size):
            slc    = slice(start, min(start + self.chunk_size, n_total))
            X_c    = X[slc]; y_c = [yc[slc] for yc in y]; curr_n = X_c.shape[0]
            phi_all, dt_all, grad_all, lap_all = \
                self._precompute_all_models_fast(X_c, self.current_models)
            pg_all = cp.zeros((n_models, n_dim, curr_n), dtype=cp.float64)
            for m_idx, pg_model in enumerate(all_pg_polys):
                for d in range(n_dim):
                    if pg_model[d].exponents.shape[0] > 0:
                        pg_all[m_idx, d] = pg_model[d].evaluate(X_c)
            y_true = cp.stack(y_c, axis=0)
            diff = phi_all - y_true[None, :, :]   # [n_models, n_comp, curr_n]
            curr_sq += (diff * diff).sum(axis=2)   # [n_models, n_comp]
            del diff
            for sub_s in range(0, curr_n, self.qr_sub_size):
                sl = slice(sub_s, min(sub_s + self.qr_sub_size, curr_n))
                for ss in range(0, n_sys, self.jac_sys_chunk):
                    se = min(ss + self.jac_sys_chunk, n_sys)
                    J_sub, b_sub = self._compute_jacobian_rows_vectorized(
                        phi_all[:, :, sl], dt_all[:, :, sl],
                        grad_all[:, :, :, sl], lap_all[:, :, sl],
                        pg_all[:, :, sl], y_true[:, sl],
                        s_gpu[ss:se], kr_gpu[ss:se],
                        ic_gpu[ss:se], ic_clamped[ss:se], n_dim)
                    J_lin = J_sub[:, :, 1:3]  # [s, k]
                    J_non = J_sub[:, :, 3:6]  # [sk, ss, kk]
                    J_lin_T = J_lin.transpose(0, 2, 1)
                    J_non_T = J_non.transpose(0, 2, 1)
                    XTX_lin[ss:se] += J_lin_T @ J_lin
                    XTy_lin[ss:se] += (J_lin_T @ b_sub[:, :, None])[:, :, 0]
                    sum_lin[ss:se] += J_lin.sum(axis=1)
                    XTX_non[ss:se] += J_non_T @ J_non
                    XTy_non[ss:se] += (J_non_T @ b_sub[:, :, None])[:, :, 0]
                    sum_non[ss:se] += J_non.sum(axis=1)
                    XTX_non_lin[ss:se] += J_non_T @ J_lin
                    sum_b[ss:se] += b_sub.sum(axis=1)
                    bsq[ss:se] += (b_sub * b_sub).sum(axis=1)
                    del J_sub, b_sub
            del phi_all, dt_all, grad_all, lap_all, pg_all, y_true
            cp.get_default_memory_pool().free_all_blocks()

        # ---- Solve in required order: alpha_0, linear, nonlinear ----
        alpha_0 = sum_b / max(n_total, 1)

        XTX_lin = (XTX_lin + XTX_lin.transpose(0, 2, 1)) * cp.float64(0.5)
        XTy_lin_c = XTy_lin - alpha_0[:, None] * sum_lin
        U1, s1, Vh1 = cp.linalg.svd(XTX_lin, full_matrices=False)
        s1_max = s1.max(axis=-1, keepdims=True)
        s1_inv = cp.where(s1 > self.svd_rcond * s1_max,
                          cp.float64(1.0) / s1, cp.float64(0.0))
        alpha_lin = (Vh1.transpose(0, 2, 1) * s1_inv[:, None, :]) @ \
                    (U1.transpose(0, 2, 1) @ XTy_lin_c[:, :, None])
        alpha_lin = alpha_lin[:, :, 0]   # [n_sys, 2] -> [s, k]

        XTX_non = (XTX_non + XTX_non.transpose(0, 2, 1)) * cp.float64(0.5)
        XTy_non_c = XTy_non - alpha_0[:, None] * sum_non \
                    - (XTX_non_lin @ alpha_lin[:, :, None])[:, :, 0]
        U2, s2, Vh2 = cp.linalg.svd(XTX_non, full_matrices=False)
        s2_max = s2.max(axis=-1, keepdims=True)
        s2_inv = cp.where(s2 > self.svd_rcond * s2_max,
                          cp.float64(1.0) / s2, cp.float64(0.0))
        alpha_non = (Vh2.transpose(0, 2, 1) * s2_inv[:, None, :]) @ \
                    (U2.transpose(0, 2, 1) @ XTy_non_c[:, :, None])
        alpha_non = alpha_non[:, :, 0]   # [n_sys, 3] -> [sk, ss, kk]

        # Final alpha order: [a0, a_s, a_k, a_sk, a_ss, a_kk]
        alphas_gpu = cp.concatenate([alpha_0[:, None], alpha_lin, alpha_non], axis=1)

        # RMSE identity (no second pass), expanded for block solve.
        lin_XTy = (alpha_lin[:, None, :] @ XTy_lin[:, :, None])[:, 0, 0]
        non_XTy = (alpha_non[:, None, :] @ XTy_non[:, :, None])[:, 0, 0]
        lin_XTX_lin = (alpha_lin[:, None, :] @ XTX_lin @ alpha_lin[:, :, None])[:, 0, 0]
        non_XTX_non = (alpha_non[:, None, :] @ XTX_non @ alpha_non[:, :, None])[:, 0, 0]
        non_XTX_lin = (alpha_non[:, None, :] @ XTX_non_lin @ alpha_lin[:, :, None])[:, 0, 0]
        c_sum_lin = (alpha_lin * sum_lin).sum(axis=1)
        c_sum_non = (alpha_non * sum_non).sum(axis=1)
        residual_ss = cp.maximum(
            bsq
            + cp.float64(n_total) * alpha_0**2
            + lin_XTX_lin + non_XTX_non
            - np.float64(2.0) * alpha_0 * sum_b
            - np.float64(2.0) * lin_XTy
            - np.float64(2.0) * non_XTy
            + np.float64(2.0) * alpha_0 * (c_sum_lin + c_sum_non)
            + np.float64(2.0) * non_XTX_lin,
            cp.float64(0.0))
        rmse_all = cp.sqrt(residual_ss / max(n_total, 1))

        alphas_cpu = alphas_gpu.get()
        rmse_cpu   = rmse_all.get()
        current_rmse_cpu = cp.sqrt(curr_sq / max(n_total, 1)).get()  # [n_models, n_comp]

        candidates = []
        for sys_idx in range(n_sys):
            candidates.append({
                's':     int(s_arr[sys_idx]),
                'k':     int(kr_arr[sys_idx]),
                'i':     int(ic_arr[sys_idx]),
                'alpha': alphas_cpu[sys_idx],
                'err':   float(rmse_cpu[sys_idx]),
            })
        return candidates, current_rmse_cpu


    # ------------------------------------------------------------------
    def _compute_jacobian_rows_vectorized(self, phi_sub, dt_sub, grad_sub,
            lap_sub, pg_sub, y_sub, s_gpu, kr_gpu, ic_gpu, ic_clamped, n_dim):
        w_dat = np.float64(self.weights['data'])
        w_div = np.float64(self.weights['div'])
        w_mom = np.float64(self.weights['mom'])
        nu    = np.float64(self.viscosity)
        sp   = phi_sub[s_gpu, ic_gpu];  sdt  = dt_sub[s_gpu, ic_gpu]
        slap = lap_sub[s_gpu, ic_gpu];  kp   = phi_sub[kr_gpu, ic_gpu]
        kdt  = dt_sub[kr_gpu, ic_gpu];  klap = lap_sub[kr_gpu, ic_gpu]
        sg_all = grad_sub[s_gpu, ic_gpu]; kg_all = grad_sub[kr_gpu, ic_gpu]
        ub_all = phi_sub[s_gpu][:, :n_dim, :]
        pg_i = pg_sub[s_gpu, ic_clamped]
        if self.n_comp > n_dim:
            pg_i = cp.where((ic_gpu >= cp.int32(n_dim))[:, None], cp.float64(0.0), pg_i)
        y_t    = y_sub[ic_gpu]
        conv_s = (ub_all * sg_all).sum(axis=1)
        conv_k = (ub_all * kg_all).sum(axis=1)
        ar     = cp.arange(s_gpu.shape[0], dtype=cp.int32)
        gs_i   = sg_all[ar, ic_clamped]; gk_i = kg_all[ar, ic_clamped]
        res_w  = ((sp-y_t)*w_dat + gs_i*w_div + (sdt+conv_s-nu*slap+pg_i)*w_mom)
        b_train = -res_w
        sk = sp*kp; sk_dt = sdt*kp + sp*kdt
        sk_conv = (ub_all*(sg_all*kp[:,None,:] + sp[:,None,:]*kg_all)).sum(axis=1)
        dot_sk = (sg_all*kg_all).sum(axis=1)
        dot_ss = (sg_all*sg_all).sum(axis=1)
        dot_kk = (kg_all*kg_all).sum(axis=1)
        sk_lap = slap*kp + sp*klap + np.float64(2.0)*dot_sk
        s2_lap = np.float64(2.0)*dot_ss + np.float64(2.0)*sp*slap
        k2_lap = np.float64(2.0)*dot_kk + np.float64(2.0)*kp*klap
        # col 0 = zeros: constant basis function has zero physics derivative
        J = cp.stack([
            cp.zeros_like(sp),
            w_dat*sp + w_div*gs_i + w_mom*(sdt+conv_s-nu*slap),
            w_dat*kp + w_div*gk_i + w_mom*(kdt+conv_k-nu*klap),
            w_dat*sk + w_div*(gs_i*kp+sp*gk_i) + w_mom*(sk_dt+sk_conv-nu*sk_lap),
            w_dat*sp*sp + w_div*(np.float64(2.0)*sp*gs_i) +
                w_mom*(np.float64(2.0)*sp*sdt + np.float64(2.0)*sp*conv_s - nu*s2_lap),
            w_dat*kp*kp + w_div*(np.float64(2.0)*kp*gk_i) +
                w_mom*(np.float64(2.0)*kp*kdt + np.float64(2.0)*kp*conv_k - nu*k2_lap),
        ], axis=2)
        return J, b_train

    # ------------------------------------------------------------------
    def _pack_polys(self, polys):
        flat_exps, flat_coeffs, lengths = [], [], []
        for p in polys:
            p._ensure_cpu_cache()
            e = p._cpu_exps   if p._cpu_exps   is not None else np.zeros((0, self.n_vars), dtype=np.uint16)
            c = p._cpu_coeffs if p._cpu_coeffs is not None else np.zeros(0, dtype=np.float64)
            flat_exps.append(e); flat_coeffs.append(c); lengths.append(len(c))
        if any(len(e) > 0 for e in flat_exps):
            all_exps   = cp.asarray(np.concatenate(flat_exps,   axis=0).astype(np.uint16))
            all_coeffs = cp.asarray(np.concatenate(flat_coeffs, axis=0).astype(np.float64))
        else:
            all_exps   = cp.zeros((0, self.n_vars), dtype=cp.uint16)
            all_coeffs = cp.zeros(0, dtype=cp.float64)
        offsets    = cp.zeros(len(lengths) + 1, dtype=cp.int32)
        offsets[1:] = cp.cumsum(cp.array(lengths, dtype=cp.int32))
        return all_exps, all_coeffs, offsets

    # ------------------------------------------------------------------
    def _eval_poly_set(self, X_c, polys, n_polys):
        """
        Memory-efficient Vandermonde evaluation.

        Two OOM fixes vs the naive version:
        1. W_dt / W_grad / W_lap are dropped immediately after kernel launch
           (they are write-only from this path and never read here), reducing
           peak memory before the compaction step.
        2. POLY_CHUNK=32: hash table H_SIZE scales with terms per chunk,
           not total terms. Reduces W_phi ~30× at deep layers.
        3. Use explicit index gather (not boolean mask gather) to avoid a very
           large temporary allocation in CuPy mask indexing at deep layers.
        """
        n_samples  = X_c.shape[0]
        n_dim      = self.n_vars - 1
        POLY_CHUNK = 30
        result     = cp.zeros((n_samples, n_polys), dtype=cp.float64)

        for p_start in range(0, n_polys, POLY_CHUNK):
            p_end   = min(p_start + POLY_CHUNK, n_polys)
            p_count = p_end - p_start

            all_exps, all_coeffs, offsets = self._pack_polys(polys[p_start:p_end])
            total_terms = int(all_exps.shape[0])
            H_SIZE = max(1024, 1 << (max(total_terms * 2, 1) - 1).bit_length())
           
            # Allocate all four weight matrices at the correct chunk size.
            # W_dt / W_grad / W_lap are written by the kernel but never read
            # afterwards — they are deleted immediately after the call.
            # With POLY_CHUNK=32 and H_SIZE bounded to the chunk's unique
            # monomial count, these are small (≤ 67 MB each) and safe.
            # NOTE: stubs (1,1) caused cudaErrorIllegalAddress because the
            # kernel writes beyond index 0 — must use the real shape.
            W_phi  = cp.zeros((H_SIZE, p_count),         dtype=cp.float64)
            W_dt   = cp.zeros((H_SIZE, p_count),         dtype=cp.float64)
            W_grad = cp.zeros((H_SIZE, p_count, n_dim),  dtype=cp.float64)
            W_lap  = cp.zeros((H_SIZE, p_count),         dtype=cp.float64)
            u_exps = cp.zeros((H_SIZE, self.n_vars), dtype=cp.uint16)
            hkeys  = cp.full(H_SIZE, 0xFFFFFFFFFFFFFFFF, dtype=cp.uint64)

            # MAX_EXP must exceed every exponent value or the kernel's
            # key = key*MAX_EXP + e collides distinct monomials. Was 1000;
            # raised to EXP_BASE (65536) so it covers full uint16 range.
            build_universal_physics_weights_kernel[p_count, 256](
                all_exps, all_coeffs, offsets, u_exps,
                W_phi, W_dt, W_grad, W_lap,
                hkeys, p_count, self.n_vars, self.time_idx,
                n_dim, EXP_BASE, H_SIZE
            )

            # Release large write-only buffers before compaction. Keeping these
            # alive during gather can push deep layers over VRAM limits.
            del W_dt, W_grad, W_lap
            valid_idx = cp.where(hkeys != cp.uint64(0xFFFFFFFFFFFFFFFF))[0]
            n_u       = int(valid_idx.size)
            if n_u == 0:
                del W_phi, u_exps, hkeys, valid_idx
                continue
            v_exp   = u_exps[valid_idx]
            W_phi_m = W_phi[valid_idx]
            del W_phi, u_exps, hkeys, valid_idx

            VANDER_CHUNK = 2048
            bx = (n_samples + 255) // 256

            for h0 in range(0, n_u, VANDER_CHUNK):
                h1 = min(h0 + VANDER_CHUNK, n_u); chu = h1 - h0
                V_chunk = cp.zeros((chu, n_samples), dtype=cp.float64)
                fast_eval_poly_kernel[(bx, chu, 1), (256, 1, 1)](
                    X_c, v_exp[h0:h1],
                    cp.ones(chu, dtype=cp.float64),
                    cp.arange(chu + 1, dtype=cp.int32),
                    V_chunk)
                result[:, p_start:p_end] += V_chunk.T @ W_phi_m[h0:h1]
                del V_chunk

            del v_exp, W_phi_m

        return result.T  # [n_polys, n_samples]

    # ------------------------------------------------------------------
    def _precompute_all_models_fast(self, X_c, models):
        n_samples = X_c.shape[0]; n_models = len(models)
        n_polys   = n_models * self.n_comp; n_dim = self.n_vars - 1
        orig_polys = [models[m][i] for m in range(n_models) for i in range(self.n_comp)]
        phi_flat = self._eval_poly_set(X_c, orig_polys, n_polys)
        dt_flat  = self._eval_poly_set(X_c, [p.differentiate(self.time_idx) for p in orig_polys], n_polys)
        grad_parts = [self._eval_poly_set(X_c, [p.differentiate(d) for p in orig_polys], n_polys)
                      for d in range(n_dim)]
        lap_polys = []
        for p in orig_polys:
            lap_p = _poly_zero(self.n_vars)
            for d in range(n_dim):
                d2 = p.differentiate(d).differentiate(d)
                if d2.exponents.shape[0] > 0: lap_p = _poly_add(lap_p, d2)
            lap_polys.append(lap_p)
        lap_flat = self._eval_poly_set(X_c, lap_polys, n_polys)
        def rs(a): return a.reshape(n_models, self.n_comp, n_samples)
        grad_all = cp.stack(grad_parts, axis=1).reshape(n_models, self.n_comp, n_dim, n_samples)
        return rs(phi_flat), rs(dt_flat), grad_all, rs(lap_flat)

    # ------------------------------------------------------------------
    def _assemble(self, winners):
        # Each winner updates only one component (ic) from parent model s.
        # All other components are copied unchanged from current_models[s].
        new_models = []
        for w in winners:
            ic  = w['i']
            a   = w['alpha']
            s_p = self.current_models[w['s']][ic]
            k_p = self.current_models[w['k']][ic]
            new_ic = _poly_constant(a[0], self.n_vars)
            new_ic = _poly_add(new_ic, s_p)
            new_ic = _poly_add(new_ic, _poly_scale(s_p, a[1]))
            new_ic = _poly_add(new_ic, _poly_scale(k_p, a[2]))
            sk = _poly_mul(s_p, k_p); ss = _poly_mul(s_p, s_p); kk = _poly_mul(k_p, k_p)
            new_ic = _poly_add(new_ic, _poly_scale(sk, a[3]))
            new_ic = _poly_add(new_ic, _poly_scale(ss, a[4]))
            new_ic = _poly_add(new_ic, _poly_scale(kk, a[5]))
            new_ic = new_ic.prune(self.prune_thresh); new_ic.sync_to_cpu()
            # Copy parent model, replace only the one updated component
            new_model = list(self.current_models[w['s']])
            new_model[ic] = new_ic
            new_models.append(new_model)

        self.best_model = new_models[0]
        return new_models


def generate_taylor_green_data(total_points=10_000_000, nu=0.01):
    t_steps   = 10; n_spatial = int(np.sqrt(total_points / t_steps))
    x_r = np.linspace(-1, 1, n_spatial); y_r = np.linspace(-1, 1, n_spatial)
    t_r = np.linspace(0, 1.0, t_steps)
    X, Y, T = np.meshgrid(x_r, y_r, t_r, indexing='ij')
    xf, yf, tf = X.ravel(), Y.ravel(), T.ravel()
    decay = np.exp(-2 * nu * tf)
    u =  np.sin(xf) * np.cos(yf) * decay
    v = -np.cos(xf) * np.sin(yf) * decay
    return np.stack([xf, yf, tf], axis=1), [u.astype(np.float64), v.astype(np.float64)]


if __name__ == "__main__":
    X, y = generate_taylor_green_data(1000_000, nu=0.01)
    trainer = GMDHTrainerGPU(
        viscosity     = 0.01,
        chunk_size    = 10000,
        top_models    = 50,
        prune_thresh  = 1e-16,
        qr_sub_size   = 1000,
        jac_sys_chunk = 700,
        svd_rcond     = 1e-8,
        corr_threshold= 0.999,
    )
    trainer.fit(X, y, n_layers=21)