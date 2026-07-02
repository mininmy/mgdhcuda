"""
gpu_gmdh_newton_known_physics.py
==================================
GMDH variant where pressure gradient and one velocity component (v) are
supplied as exact analytical values (e.g. from Taylor-Green generation)
and are NOT estimated during training.

Differences from gpu_gmdh_newton_model.py
------------------------------------------
- n_comp=1: trainer fits only u (x-velocity component).
- Pressure gradient (∂p/∂x) is a fixed data array; _compute_pressure_grad_polys
  is removed entirely.
- Known v is injected into the convection term instead of the polynomial model's v.
- Divergence residual includes the exact ∂v/∂y from data.
- No ensemble pressure weighting (pressure source is exact, not estimated).
"""

import warnings
from numba import NumbaPerformanceWarning
import time
import cupy as cp
import cupyx
import numpy as np
from numba import cuda, float64

from cuda_kernels_PIGMDH import build_universal_physics_weights_kernel
from gpu_polynomial_module import PolynomialGPU

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

PRUNE_THRESH = 0

EXP_BASE = 65536  # base for exponent-tuple hashing; must exceed max exponent


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
    prod_exps   = (ea[:, np.newaxis, :].astype(np.int32) +
                   eb[np.newaxis, :, :].astype(np.int32)).reshape(na * nb, nv)
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
    """
    GMDH trainer that fits only u (component 0).  v and ∂p/∂x are supplied
    as exact data arrays and are injected directly into the physics residuals.
    """

    def __init__(self, viscosity, chunk_size=500_000, weights=None,
                 top_models=10, prune_thresh=PRUNE_THRESH,
                 qr_sub_size=2000, jac_sys_chunk=512,
                 svd_rcond=1e-10, corr_threshold=1.0,
                 ridge_lambda=0.0, var_threshold=1e-10, verbosity=1,
                 include_raw_inputs=True, diversity_gamma=1):
        self.viscosity      = viscosity
        self.chunk_size     = chunk_size
        self.weights        = weights or {"data": 1.0, "div": 1.0, "mom": 1.0}
        self.top_models     = top_models
        self.prune_thresh   = prune_thresh
        self.qr_sub_size    = qr_sub_size
        self.svd_rcond      = svd_rcond
        # ridge_lambda: Tikhonov coefficient added to diagonal of each XᵀX block
        # (scaled by n_total inside the solve).  Set alongside svd_rcond=0 to
        # replace hard truncation with smooth regularization.
        self.ridge_lambda   = ridge_lambda
        # var_threshold: skip nonlinear solve when max(diag(XᵀX_non)) <
        # var_threshold * max(diag(XᵀX_lin)).  Avoids wasting SVD on degenerate
        # sk/s²/k² terms that arise when s≈k.
        self.var_threshold  = var_threshold
        self.verbosity      = verbosity
        self.time_scale     = 1.0
        self.current_models = []
        self.best_model     = []
        self.jac_sys_chunk  = jac_sys_chunk
        self.corr_threshold = corr_threshold
        # When True, raw input variables (x, y, t) are always available as k
        # candidates in addition to models from the current layer.
        self.include_raw_inputs = include_raw_inputs
        # diversity_gamma: exponent on (1-rho) in diversity score
        # S_i = (1/RMSE_i) * (1 - rho_i)^gamma, rho_i = max corr to selected
        self.diversity_gamma    = diversity_gamma
        # Known physics arrays — set in fit()
        self.known_velocity_gpu = None   # v (y-velocity)       [N]
        self.known_dv_dy_gpu    = None   # ∂v/∂y                [N]
        self.known_pg_gpu       = None   # ∂p/∂x                [N]

    def _initialize_layer_zero(self):
        models = []
        for v_idx in range(self.n_vars):
            models.append([_poly_variable(v_idx, self.n_vars)
                           for _ in range(self.n_comp)])
        return models

    def _log_best_model(self, best, layer_idx):
        print(f"\n=== BEST MODEL @ Layer {layer_idx} ===")
        print(f"s={best['s']}  k={best['k']}  i={best['i']}  "
              f"combined_err={best['combined_err']:.6e}  fitted_err={best['err']:.6e}")
        print(f"  alpha={best['alpha']}")

    # ------------------------------------------------------------------
    def fit(self, X_cpu, y_target_cpu, n_layers=19,
            known_velocity_cpu=None, known_dv_dy_cpu=None, known_pg_cpu=None,
            layer_schedule=None):
        """
        Parameters
        ----------
        X_cpu            : ndarray [N, 3]  — (x, y, t) input points
        y_target_cpu     : list of 1 array [N]  — target u values
        known_velocity_cpu : ndarray [N]   — exact v at each point
        known_dv_dy_cpu    : ndarray [N]   — exact ∂v/∂y at each point
        known_pg_cpu       : ndarray [N]   — exact ∂p/∂x at each point
        """
        X_tr = cp.asarray(X_cpu, dtype=cp.float64)
        y_tr = [cp.asarray(yc, dtype=cp.float64) for yc in y_target_cpu]
        self.n_vars   = X_tr.shape[1]
        self.n_comp   = len(y_tr)          # must be 1
        self.time_idx = self.n_vars - 1

        if known_velocity_cpu is not None:
            self.known_velocity_gpu = cp.asarray(known_velocity_cpu, dtype=cp.float64)
            self.known_dv_dy_gpu    = cp.asarray(known_dv_dy_cpu,    dtype=cp.float64)
            self.known_pg_gpu       = cp.asarray(known_pg_cpu,       dtype=cp.float64)

        self.current_models = self._initialize_layer_zero()
        best_rmse = float('inf'); best_layer = -1
        prev_avg_rmse = float('inf')

        _sched = layer_schedule or []
        for layer in range(n_layers):
            cfg    = _sched[layer] if layer < len(_sched) else {}
            _top   = cfg.get('top_models',    self.top_models)
            _corr  = cfg.get('corr_threshold', self.corr_threshold)
            _rcond = cfg.get('svd_rcond',     self.svd_rcond)
            _ridge = cfg.get('ridge_lambda',  self.ridge_lambda)

            t_start = time.time()
            candidates, current_rmse, phi_sample = self._train_and_eval_layer(
                X_tr, y_tr,
                svd_rcond=_rcond, ridge_lambda=_ridge, corr_threshold=_corr)
            t_layer = time.time() - t_start

            # n_comp=1: combined_err equals the fitted component's error directly
            for c in candidates:
                c['combined_err'] = c['err']

            # Pre-sort and restrict pool before diversity selection
            candidates.sort(key=lambda x: x['combined_err'])
            pool    = candidates[:min(len(candidates), max(_top * 5, 100))]
            winners = self._diverse_select(pool, phi_sample, _top, self.diversity_gamma)

            if not winners:
                print("  → No candidates survived; stopping.")
                break

            layer_rmse = winners[0]['combined_err']
            avg_rmse   = sum(w['combined_err'] for w in winners) / max(len(winners), 1)
            print(f"Layer {layer} | Best RMSE: {layer_rmse:.6e} | "
                  f"Avg RMSE (top): {avg_rmse:.6e} | "
                  f"Time: {t_layer:.2f}s | n_models: {len(self.current_models)}")

            self._log_best_model(winners[0], layer)

            if (avg_rmse > prev_avg_rmse) and (layer_rmse > best_rmse * 0.99):
                print(f"  → Avg top RMSE increased "
                      f"({prev_avg_rmse:.6e} → {avg_rmse:.6e}). Stopping.")
                break

            best_rmse = layer_rmse; best_layer = layer
            prev_avg_rmse = avg_rmse
            self.current_models = self._assemble(winners)
            cp.get_default_memory_pool().free_all_blocks()

        print(f"Training complete. Best RMSE: {best_rmse:.6e} at layer {best_layer}.")

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
    @staticmethod
    def _diverse_select(candidates, phi_sample, n_top, gamma):
        """
        Greedy diversity-aware winner selection.

            S_i = (1 / RMSE_i) * (1 - rho_i)^gamma

        rho_i  — max |Pearson correlation| between candidate i's output and
                 any already-selected candidate's output, evaluated at
                 phi_sample points.
        phi_sample : [n_ext, n_pts]  — poly evaluations at sample points (CPU)
        """
        n_cand = len(candidates)
        if n_cand == 0:
            return []
        n_pts = phi_sample.shape[1]

        # Evaluate each candidate's assembled output: f = a0+a1*s+a2*k+a3*sk+a4*s²+a5*k²
        cand_vals = np.empty((n_cand, n_pts), dtype=np.float64)
        for idx, c in enumerate(candidates):
            a  = c['alpha']
            ps = phi_sample[c['s']]
            pk = phi_sample[c['k']]
            cand_vals[idx] = (a[0]
                              + a[1] * ps + a[2] * pk
                              + a[3] * ps * pk
                              + a[4] * ps * ps
                              + a[5] * pk * pk)

        # Unit-variance centered vectors for correlation
        mu        = cand_vals.mean(axis=1, keepdims=True)
        centered  = cand_vals - mu
        norms     = np.sqrt((centered ** 2).sum(axis=1))
        norms     = np.where(norms < 1e-12, 1.0, norms)
        normed    = centered / norms[:, np.newaxis]   # [n_cand, n_pts]

        rmses         = np.array([c['combined_err'] for c in candidates], dtype=np.float64)
        rmses         = np.maximum(rmses, 1e-30)
        selected_idx  = []
        selected_mask = np.zeros(n_cand, dtype=bool)
        max_corr      = np.zeros(n_cand, dtype=np.float64)

        for step in range(min(n_top, n_cand)):
            corr_penalty = np.clip(1.0 - max_corr, 0.0, 1.0)
            scores = (1.0 / rmses) * (corr_penalty ** gamma)
            scores[selected_mask] = -np.inf
            best_i = int(np.argmax(scores))

            selected_idx.append(best_i)
            selected_mask[best_i] = True

            # Update running max-correlation with the newly selected model
            new_corrs = np.abs(normed @ normed[best_i])   # [n_cand]
            max_corr  = np.maximum(max_corr, new_corrs)

        return [candidates[i] for i in selected_idx]

    # ------------------------------------------------------------------
    def _train_and_eval_layer(self, X, y,
                              svd_rcond=None, ridge_lambda=None,
                              corr_threshold=None):
        svd_rcond     = self.svd_rcond     if svd_rcond     is None else svd_rcond
        ridge_lambda  = self.ridge_lambda  if ridge_lambda  is None else ridge_lambda
        corr_threshold = self.corr_threshold if corr_threshold is None else corr_threshold
        n_total  = X.shape[0]; n_models = len(self.current_models)
        n_dim    = self.n_vars - 1
        # Optionally extend k-candidates with raw input variables (x, y, t)
        n_raw    = self.n_vars if self.include_raw_inputs else 0
        n_ext    = n_models + n_raw
        n_ks     = n_ext - 1
        n_sys    = n_models * n_ks * self.n_comp

        if self.include_raw_inputs:
            raw_polys  = [[_poly_variable(v, self.n_vars)] for v in range(self.n_vars)]
            ext_models = self.current_models + raw_polys
        else:
            ext_models = self.current_models

        k_map_cpu = np.zeros((n_models, n_ks), dtype=np.int32)
        for s in range(n_models):
            for loc, k in enumerate(k for k in range(n_ext) if k != s):
                k_map_cpu[s, loc] = k

        # Sample for diversity scoring: [n_ext, n_sample]
        n_sample      = min(self.qr_sub_size, n_total)
        _phi_samp, _, _, _ = self._precompute_all_models_fast(X[:n_sample], ext_models)
        phi_sample    = _phi_samp[:, 0, :].get()   # n_comp=1, so axis-1 is trivial
        del _phi_samp

        s_arr  = np.repeat(np.arange(n_models), n_ks * self.n_comp)
        kl_arr = np.tile(np.repeat(np.arange(n_ks), self.n_comp), n_models)
        ic_arr = np.tile(np.arange(self.n_comp), n_models * n_ks)
        kr_arr = k_map_cpu[s_arr, kl_arr]

        # Correlation filter
        if corr_threshold < 1.0:
            _Xs = X[:min(self.chunk_size, n_total)]
            _phi, _, _, _ = self._precompute_all_models_fast(_Xs, ext_models)
            _sg = cp.asarray(s_arr, dtype=cp.int32)
            _ig = cp.asarray(ic_arr, dtype=cp.int32)
            _kg = cp.asarray(kr_arr, dtype=cp.int32)
            valid = self._compute_correlation_mask(
                _phi, _sg, _kg, _ig, corr_threshold, self.jac_sys_chunk)
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

        w_dat = np.float64(self.weights['data'])
        w_div = np.float64(self.weights['div'])
        w_mom = np.float64(self.weights['mom'])

        sum_b   = cp.zeros(n_sys,      dtype=cp.float64)
        sum_lin = cp.zeros((n_sys, 2), dtype=cp.float64)
        sum_non = cp.zeros((n_sys, 3), dtype=cp.float64)

        _ptags = [t for t, w in (('dat', w_dat), ('div', w_div), ('mom', w_mom)) if w > 0.0]
        bsq_p     = {t: cp.zeros(n_sys,         dtype=cp.float64) for t in _ptags}
        XTX_lin_p = {t: cp.zeros((n_sys, 2, 2), dtype=cp.float64) for t in _ptags}
        XTy_lin_p = {t: cp.zeros((n_sys, 2),    dtype=cp.float64) for t in _ptags}
        XTX_non_p = {t: cp.zeros((n_sys, 3, 3), dtype=cp.float64) for t in _ptags}
        XTy_non_p = {t: cp.zeros((n_sys, 3),    dtype=cp.float64) for t in _ptags}
        XTX_nl_p  = {t: cp.zeros((n_sys, 3, 2), dtype=cp.float64) for t in _ptags}

        for start in range(0, n_total, self.chunk_size):
            slc    = slice(start, min(start + self.chunk_size, n_total))
            X_c    = X[slc]; y_c = [yc[slc] for yc in y]; curr_n = X_c.shape[0]

            # Slice known physics for this chunk
            v_known_c = self.known_velocity_gpu[slc]  # [curr_n]
            dv_dy_c   = self.known_dv_dy_gpu[slc]     # [curr_n]
            pg_x_c    = self.known_pg_gpu[slc]         # [curr_n]

            phi_all, dt_all, grad_all, lap_all = \
                self._precompute_all_models_fast(X_c, ext_models)
            y_true = cp.stack(y_c, axis=0)
            diff = phi_all[:n_models] - y_true[None, :, :]
            curr_sq += (diff * diff).sum(axis=2)
            del diff

            for sub_s in range(0, curr_n, self.qr_sub_size):
                sl = slice(sub_s, min(sub_s + self.qr_sub_size, curr_n))
                for ss in range(0, n_sys, self.jac_sys_chunk):
                    se = min(ss + self.jac_sys_chunk, n_sys)
                    J_dat, J_div, J_mom, b_dat, b_div, b_mom = \
                        self._compute_jacobian_rows_vectorized(
                            phi_all[:, :, sl], dt_all[:, :, sl],
                            grad_all[:, :, :, sl], lap_all[:, :, sl],
                            v_known_c[sl], dv_dy_c[sl], pg_x_c[sl],
                            y_true[:, sl],
                            s_gpu[ss:se], kr_gpu[ss:se],
                            ic_gpu[ss:se], ic_clamped[ss:se], n_dim)
                    sum_b[ss:se]   += b_dat.sum(axis=1)
                    sum_lin[ss:se] += J_dat[:, :, 1:3].sum(axis=1)
                    sum_non[ss:se] += J_dat[:, :, 3:6].sum(axis=1)
                    for _tag, J_t, b_t in (('dat', J_dat, b_dat),
                                           ('div', J_div, b_div),
                                           ('mom', J_mom, b_mom)):
                        if _tag not in bsq_p:
                            continue
                        J_lin_t = J_t[:, :, 1:3]; J_non_t = J_t[:, :, 3:6]
                        J_lin_T = J_lin_t.transpose(0, 2, 1)
                        J_non_T = J_non_t.transpose(0, 2, 1)
                        bsq_p[_tag][ss:se]     += (b_t * b_t).sum(axis=1)
                        XTX_lin_p[_tag][ss:se] += J_lin_T @ J_lin_t
                        XTy_lin_p[_tag][ss:se] += (J_lin_T @ b_t[:, :, None])[:, :, 0]
                        XTX_non_p[_tag][ss:se] += J_non_T @ J_non_t
                        XTy_non_p[_tag][ss:se] += (J_non_T @ b_t[:, :, None])[:, :, 0]
                        XTX_nl_p[_tag][ss:se]  += J_non_T @ J_lin_t
                    del J_dat, J_div, J_mom, b_dat, b_div, b_mom
            del phi_all, dt_all, grad_all, lap_all, y_true
            cp.get_default_memory_pool().free_all_blocks()

        # Build combined weighted normal equations for the solve
        _wtag_all = [('dat', w_dat), ('div', w_div), ('mom', w_mom)]
        XTX_lin     = cp.zeros((n_sys, 2, 2), dtype=cp.float64)
        XTy_lin     = cp.zeros((n_sys, 2),    dtype=cp.float64)
        XTX_non     = cp.zeros((n_sys, 3, 3), dtype=cp.float64)
        XTy_non     = cp.zeros((n_sys, 3),    dtype=cp.float64)
        XTX_non_lin = cp.zeros((n_sys, 3, 2), dtype=cp.float64)
        bsq         = cp.zeros(n_sys,          dtype=cp.float64)
        for _t, _w in _wtag_all:
            if _t not in bsq_p:
                continue
            XTX_lin     += _w * XTX_lin_p[_t]
            XTy_lin     += _w * XTy_lin_p[_t]
            XTX_non     += _w * XTX_non_p[_t]
            XTy_non     += _w * XTy_non_p[_t]
            XTX_non_lin += _w * XTX_nl_p[_t]
            bsq         += _w * bsq_p[_t]
        sum_lin_w = w_dat * sum_lin
        sum_non_w = w_dat * sum_non

        alpha_0 = sum_b / max(n_total, 1)

        XTX_lin = (XTX_lin + XTX_lin.transpose(0, 2, 1)) * cp.float64(0.5)
        XTX_non = (XTX_non + XTX_non.transpose(0, 2, 1)) * cp.float64(0.5)

        # --- Nonlinear-degenerate skip: if max(diag(XᵀX_non)) is negligible
        # relative to max(diag(XᵀX_lin)), the sk/s²/k² columns have near-zero
        # variance (typically when s≈k).  Solve is numerically useless; skip it.
        diag_lin     = cp.diagonal(XTX_lin, axis1=1, axis2=2)   # [n_sys, 2]
        diag_non_raw = cp.diagonal(XTX_non, axis1=1, axis2=2)   # [n_sys, 3]
        max_lin_diag = diag_lin.max(axis=1)                      # [n_sys]
        max_non_diag = diag_non_raw.max(axis=1)                  # [n_sys]
        skip_non = (max_non_diag < np.float64(self.var_threshold) * max_lin_diag)

        # --- Ridge regularization: add λ*I to stabilise near-singular systems.
        # Replaces hard SVD truncation with smooth shrinkage when ridge_lambda>0.
        if ridge_lambda > 0.0:
            lam = cp.float64(ridge_lambda * n_total)
            XTX_lin = XTX_lin + lam * cp.eye(2, dtype=cp.float64)[None]
            XTX_non = XTX_non + lam * cp.eye(3, dtype=cp.float64)[None]

        XTy_lin_c = XTy_lin - alpha_0[:, None] * sum_lin_w
        U1, s1, Vh1 = cp.linalg.svd(XTX_lin, full_matrices=False)
        s1_max = s1.max(axis=-1, keepdims=True)
        s1_inv = cp.where(s1 > np.float64(svd_rcond) * s1_max,
                          cp.float64(1.0) / s1, cp.float64(0.0))
        alpha_lin = (Vh1.transpose(0, 2, 1) * s1_inv[:, None, :]) @ \
                    (U1.transpose(0, 2, 1) @ XTy_lin_c[:, :, None])
        alpha_lin = alpha_lin[:, :, 0]

        XTy_non_c = XTy_non - alpha_0[:, None] * sum_non_w \
                    - (XTX_non_lin @ alpha_lin[:, :, None])[:, :, 0]
        U2, s2, Vh2 = cp.linalg.svd(XTX_non, full_matrices=False)
        s2_max = s2.max(axis=-1, keepdims=True)
        s2_inv = cp.where(s2 > np.float64(svd_rcond) * s2_max,
                          cp.float64(1.0) / s2, cp.float64(0.0))
        alpha_non = (Vh2.transpose(0, 2, 1) * s2_inv[:, None, :]) @ \
                    (U2.transpose(0, 2, 1) @ XTy_non_c[:, :, None])
        alpha_non = alpha_non[:, :, 0]

        # Zero out nonlinear coefficients for degenerate systems
        alpha_non = cp.where(skip_non[:, None], cp.float64(0.0), alpha_non)

        if self.verbosity >= 2:
            n_skip  = int(skip_non.sum())
            eff_lin = float((s1 > np.float64(svd_rcond) * s1_max).sum()) / max(n_sys * 2, 1)
            eff_non = float((s2 > np.float64(svd_rcond) * s2_max).sum()) / max(n_sys * 3, 1)
            print(f"    rank util: lin={eff_lin:.0%} non={eff_non:.0%}  "
                  f"skip_non={n_skip}/{n_sys}  "
                  f"ridge={ridge_lambda:.1e}  rcond={svd_rcond:.1e}")

        alphas_gpu = cp.concatenate([alpha_0[:, None], alpha_lin, alpha_non], axis=1)

        c_sum_lin_uw = (alpha_lin * sum_lin).sum(axis=1)
        c_sum_non_uw = (alpha_non * sum_non).sum(axis=1)
        rmse_phys = {}
        for _tag, _w in _wtag_all:
            if _tag not in bsq_p:
                rmse_phys[_tag] = cp.zeros(n_sys, dtype=cp.float64)
                continue
            _lXTy = (alpha_lin[:,None,:] @ XTy_lin_p[_tag][:,:,None])[:,0,0]
            _nXTy = (alpha_non[:,None,:] @ XTy_non_p[_tag][:,:,None])[:,0,0]
            _lXl  = (alpha_lin[:,None,:] @ XTX_lin_p[_tag] @ alpha_lin[:,:,None])[:,0,0]
            _nXn  = (alpha_non[:,None,:] @ XTX_non_p[_tag] @ alpha_non[:,:,None])[:,0,0]
            _nXl  = (alpha_non[:,None,:] @ XTX_nl_p[_tag]  @ alpha_lin[:,:,None])[:,0,0]
            _const = (cp.float64(n_total) * alpha_0**2
                      - np.float64(2.0) * alpha_0 * sum_b
                      + np.float64(2.0) * alpha_0 * (c_sum_lin_uw + c_sum_non_uw)
                      ) if _tag == 'dat' else cp.float64(0.0)
            _rss = cp.maximum(
                bsq_p[_tag] + _const + _lXl + _nXn
                - np.float64(2.0) * _lXTy - np.float64(2.0) * _nXTy
                + np.float64(2.0) * _nXl,
                cp.float64(0.0))
            rmse_phys[_tag] = cp.sqrt(_rss / max(n_total, 1))

        rmse_score = (w_dat * rmse_phys['dat'] + w_div * rmse_phys['div']
                      + w_mom * rmse_phys['mom'])

        alphas_cpu       = alphas_gpu.get()
        rmse_score_cpu   = rmse_score.get()
        current_rmse_cpu = cp.sqrt(curr_sq / max(n_total, 1)).get()

        candidates = []
        for sys_idx in range(n_sys):
            candidates.append({
                's':     int(s_arr[sys_idx]),
                'k':     int(kr_arr[sys_idx]),
                'i':     int(ic_arr[sys_idx]),
                'alpha': alphas_cpu[sys_idx],
                'err':   float(rmse_score_cpu[sys_idx]),
            })
        return candidates, current_rmse_cpu, phi_sample

    # ------------------------------------------------------------------
    def _compute_jacobian_rows_vectorized(self, phi_sub, dt_sub, grad_sub,
            lap_sub, v_known_sub, dv_dy_sub, pg_x_sub,
            y_sub, s_gpu, kr_gpu, ic_gpu, ic_clamped, n_dim):
        """
        v_known_sub : [n_pts]   — known y-velocity at current sub-slice points
        dv_dy_sub   : [n_pts]   — known ∂v/∂y at current sub-slice points
        pg_x_sub    : [n_pts]   — known ∂p/∂x at current sub-slice points
        """
        nu    = np.float64(self.viscosity)
        # phi_sub: [n_models, 1, n_pts]  (n_comp=1)
        sp   = phi_sub[s_gpu,  ic_gpu];  sdt  = dt_sub[s_gpu,  ic_gpu]
        slap = lap_sub[s_gpu,  ic_gpu];  kp   = phi_sub[kr_gpu, ic_gpu]
        kdt  = dt_sub[kr_gpu,  ic_gpu];  klap = lap_sub[kr_gpu, ic_gpu]
        sg_all = grad_sub[s_gpu,  ic_gpu]   # [n_sys, n_dim, n_pts]
        kg_all = grad_sub[kr_gpu, ic_gpu]   # [n_sys, n_dim, n_pts]

        # Effective 2D velocity for convection: [u_s, v_known]
        # sp shape: [n_sys, n_pts]; v_known_sub: [n_pts]
        v_bcast = cp.broadcast_to(v_known_sub[None, :], sp.shape)  # [n_sys, n_pts]
        ub_eff  = cp.stack([sp, v_bcast], axis=1)                   # [n_sys, 2, n_pts]

        y_t    = y_sub[ic_gpu]
        conv_s = (ub_eff * sg_all).sum(axis=1)
        conv_k = (ub_eff * kg_all).sum(axis=1)

        ar    = cp.arange(s_gpu.shape[0], dtype=cp.int32)
        gs_i  = sg_all[ar, ic_clamped]   # ∂s/∂x  (ic=0 always)
        gk_i  = kg_all[ar, ic_clamped]   # ∂k/∂x

        sk = sp*kp; sk_dt = sdt*kp + sp*kdt
        sk_conv = (ub_eff*(sg_all*kp[:,None,:] + sp[:,None,:]*kg_all)).sum(axis=1)
        dot_sk = (sg_all*kg_all).sum(axis=1)
        dot_ss = (sg_all*sg_all).sum(axis=1)
        dot_kk = (kg_all*kg_all).sum(axis=1)
        sk_lap = slap*kp + sp*klap + np.float64(2.0)*dot_sk
        s2_lap = np.float64(2.0)*dot_ss + np.float64(2.0)*sp*slap
        k2_lap = np.float64(2.0)*dot_kk + np.float64(2.0)*kp*klap

        J_dat = cp.stack([
            cp.ones_like(sp),
            sp, kp, sk, sp*sp, kp*kp,
        ], axis=2)
        b_dat = y_t - sp

        J_div = cp.stack([
            cp.zeros_like(sp),
            gs_i, gk_i,
            gs_i*kp + sp*gk_i,
            np.float64(2.0)*sp*gs_i,
            np.float64(2.0)*kp*gk_i,
        ], axis=2)
        # Incompressibility: ∂f_new/∂x + ∂v_known/∂y = 0
        # residual at parent s:  -(∂s/∂x + ∂v/∂y)
        b_div = -(gs_i + dv_dy_sub)

        J_mom = cp.stack([
            cp.zeros_like(sp),
            sdt + conv_s - nu*slap,
            kdt + conv_k - nu*klap,
            sk_dt + sk_conv - nu*sk_lap,
            np.float64(2.0)*sp*sdt + np.float64(2.0)*sp*conv_s - nu*s2_lap,
            np.float64(2.0)*kp*kdt + np.float64(2.0)*kp*conv_k - nu*k2_lap,
        ], axis=2)
        # Momentum residual at parent s with exact pressure gradient
        b_mom = -(sdt + conv_s - nu*slap + pg_x_sub)

        return J_dat, J_div, J_mom, b_dat, b_div, b_mom

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
        n_samples  = X_c.shape[0]
        n_dim      = self.n_vars - 1
        POLY_CHUNK = 20
        result     = cp.zeros((n_samples, n_polys), dtype=cp.float64)

        for p_start in range(0, n_polys, POLY_CHUNK):
            p_end   = min(p_start + POLY_CHUNK, n_polys)
            p_count = p_end - p_start

            all_exps, all_coeffs, offsets = self._pack_polys(polys[p_start:p_end])
            total_terms = int(all_exps.shape[0])
            H_SIZE = max(1024, 1 << (max(total_terms * 2, 1) - 1).bit_length())

            W_phi  = cp.zeros((H_SIZE, p_count),         dtype=cp.float64)
            W_dt   = cp.zeros((H_SIZE, p_count),         dtype=cp.float64)
            W_grad = cp.zeros((H_SIZE, p_count, n_dim),  dtype=cp.float64)
            W_lap  = cp.zeros((H_SIZE, p_count),         dtype=cp.float64)
            u_exps = cp.zeros((H_SIZE, self.n_vars), dtype=cp.uint16)
            hkeys  = cp.full(H_SIZE, 0xFFFFFFFFFFFFFFFF, dtype=cp.uint64)

            build_universal_physics_weights_kernel[p_count, 256](
                all_exps, all_coeffs, offsets, u_exps,
                W_phi, W_dt, W_grad, W_lap,
                hkeys, p_count, self.n_vars, self.time_idx,
                n_dim, EXP_BASE, H_SIZE
            )

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
        n_curr = len(self.current_models)
        new_models = []
        for w in winners:
            ic    = w['i']
            a     = w['alpha']
            s_p   = self.current_models[w['s']][ic]
            k_idx = w['k']
            if k_idx < n_curr:
                k_p = self.current_models[k_idx][ic]
            else:
                k_p = _poly_variable(k_idx - n_curr, self.n_vars)
            new_ic = _poly_constant(a[0], self.n_vars)
            new_ic = _poly_add(new_ic, s_p)
            new_ic = _poly_add(new_ic, _poly_scale(s_p, a[1]))
            new_ic = _poly_add(new_ic, _poly_scale(k_p, a[2]))
            sk = _poly_mul(s_p, k_p); ss = _poly_mul(s_p, s_p); kk = _poly_mul(k_p, k_p)
            new_ic = _poly_add(new_ic, _poly_scale(sk, a[3]))
            new_ic = _poly_add(new_ic, _poly_scale(ss, a[4]))
            new_ic = _poly_add(new_ic, _poly_scale(kk, a[5]))
            new_ic = new_ic.prune(self.prune_thresh); new_ic.sync_to_cpu()
            new_model = list(self.current_models[w['s']])
            new_model[ic] = new_ic
            new_models.append(new_model)
        self.best_model = new_models[0]
        return new_models


# ---------------------------------------------------------------------------
def generate_taylor_green_data_with_known(total_points=1000_000, nu=0.01):
    """
    Returns
    -------
    X          : [N, 3]  input coordinates (x, y, t)
    y          : list of 1 array [N]  — target u values
    v_known    : [N]  — exact v = -cos(x)sin(y)exp(-2νt)
    dv_dy      : [N]  — exact ∂v/∂y = -cos(x)cos(y)exp(-2νt)
    pg_x       : [N]  — exact ∂p/∂x = -sin(2x)/2 * exp(-4νt)
                        (derived from NS: ∂p/∂x = -(∂u/∂t + u·∇u - ν∇²u))
    """
    t_steps   = 10
    n_spatial = int(np.sqrt(total_points / t_steps))
    x_r = np.linspace(-1, 1, n_spatial)
    y_r = np.linspace(-1, 1, n_spatial)
    t_r = np.linspace(0, 1.0, t_steps)
    X, Y, T = np.meshgrid(x_r, y_r, t_r, indexing='ij')
    xf, yf, tf = X.ravel(), Y.ravel(), T.ravel()

    decay  = np.exp(-2.0 * nu * tf)
    decay4 = np.exp(-4.0 * nu * tf)

    u     =  np.sin(xf) * np.cos(yf) * decay
    v     = -np.cos(xf) * np.sin(yf) * decay
    dv_dy = -np.cos(xf) * np.cos(yf) * decay          # ∂v/∂y
    pg_x  = -0.5 * np.sin(2.0 * xf) * decay4          # ∂p/∂x

    return (np.stack([xf, yf, tf], axis=1),
            [u.astype(np.float64)],
            v.astype(np.float64),
            dv_dy.astype(np.float64),
            pg_x.astype(np.float64))


if __name__ == "__main__":
    X, y, v_known, dv_dy_known, pg_x_known = \
        generate_taylor_green_data_with_known(1000_000, nu=0.01)

    # Adaptive layer schedule:
    # - Early layers: maximum diversity (many models, loose correlation filter,
    #   moderate ridge to keep the s-k difference direction).
    # - Later layers: tight correlation filter prunes redundant pairs, smaller
    #   model pool focuses compute on the best candidates.
    N_LAYERS = 15
    sched = []
    for _l in range(N_LAYERS):
        frac = _l / max(N_LAYERS - 1, 1)          # 0.0 … 1.0
        if _l <=4:
            gamma = 1
            cr_filt = 0.95
        else:
            gamma = 5
            cr_filt = 0.95
        sched.append({
            'top_models':    120, #max(20, int(100 * (1.0 - 0.6 * frac))),   # 100→40
            'corr_threshold': cr_filt, # + 0.095 * frac,                     # 0.90→0.995
            'ridge_lambda':  1e-6 * (1.0 - frac) + 1e-8 * frac,        # 1e-6→1e-8
            'svd_rcond':     0.0,                                        # ridge handles it
            'diversity_gamma': gamma,
        })
    
    trainer = GMDHTrainerGPU(
        viscosity      = 0.01,
        chunk_size     = 5000,
        top_models     = 10,
        prune_thresh   = 1e-19,
        qr_sub_size    = 1000,
        jac_sys_chunk  = 700,
        svd_rcond      = 0.0,
        corr_threshold = 0.90,
        ridge_lambda   = 1e-6,
        var_threshold  = 1e-8,
        verbosity      = 2,
    )
    
    trainer.fit(X, y, n_layers=N_LAYERS,
                known_velocity_cpu = v_known,
                known_dv_dy_cpu    = dv_dy_known,
                known_pg_cpu       = pg_x_known,
                layer_schedule     = sched)
