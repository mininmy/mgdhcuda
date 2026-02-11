import rmm
from rmm import mr
from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import dask
from dask import delayed
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import delayed
import math
# --- RMM Memory Setup ---
initial_pool_size = 1*1024 * 1024 * 1024       # 512 MB
max_pool_size = 12 * 1024 * 1024 * 1024     # 10 GB max pool

upstream = mr.CudaMemoryResource()
pool = mr.PoolMemoryResource(
    upstream,
    initial_pool_size=initial_pool_size,
    maximum_pool_size=max_pool_size
)
tracked = mr.TrackingResourceAdaptor(pool)
rmm.mr.set_current_device_resource(tracked)
cp.cuda.set_allocator(rmm_cupy_allocator)
dask.config.set(scheduler='threads')  #
print(f"GPU free memory: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.2f} GB / "
      f"{cp.cuda.runtime.memGetInfo()[1] / 1024**3:.2f} GB")
print(f"RMM currently allocated: {tracked.get_allocated_bytes() / 1e6:.2f} MB")

from gpu_polynomial_module import PolynomialGPU
from cuda_poly_multiply import evaluate_model_pair
from cuda_least_squares import least_squares_gpu
from pressure_estimator import compute_pressure_from_polynomials
evaluate_model_pair_delayed = delayed(evaluate_model_pair)

import time
from contextlib import contextmanager
from collections import defaultdict

def _sync_gpu():
    try:
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        pass

class TimeAgg:
    def __init__(self):
        self.cpu = defaultdict(float)   # seconds
        self.gpu = defaultdict(float)   # milliseconds

    @contextmanager
    def cpu_block(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.cpu[key] += (time.perf_counter() - t0)

    @contextmanager
    def gpu_block(self, key):
        # measures GPU time using CUDA events
        start = cp.cuda.Event() if 'cp' in globals() else None
        end = cp.cuda.Event() if 'cp' in globals() else None
        if start is not None:
            _sync_gpu()
            start.record()
        try:
            yield
        finally:
            if start is not None:
                end.record()
                end.synchronize()
                self.gpu[key] += cp.cuda.get_elapsed_time(start, end)  # ms

    def report(self):
        # pretty print in descending order
        if self.cpu:
            print("\n=== CPU Timings (s) ===")
            for k, v in sorted(self.cpu.items(), key=lambda x: -x[1]):
                print(f"{k:30s} : {v:9.4f}")
        if self.gpu:
            print("\n=== GPU Timings (ms) ===")
            for k, v in sorted(self.gpu.items(), key=lambda x: -x[1]):
                print(f"{k:30s} : {v:9.2f}")
        print()
import time
from contextlib import contextmanager


# --- Helper functions ---
import time
import cupy as cp
import numpy as np


def gmdh_error_lightweight_eval(
    X_gpu,
    U_true_gpu,
    velocity_polys,
    pressure_poly=None,
    nu=0.01,
    g=None,
    return_components=True,
    weights=None,
    chunk_size=10000,
):
    """
    Physics-aware lightweight error estimator for GPU (CuPy).
    Works in evaluation mode (no heavy polynomial combining).
    """

    n_velocity = U_true_gpu.shape[1]
    nvars = X_gpu.shape[1]
    g = g if g is not None else cp.zeros(n_velocity, dtype=cp.float64)

    if weights is None:
        weights = {"data": 1.0, "incompressibility": 1.0, "navier_stokes": 1.0}

    timings = {}
    _sync_gpu()
    t0_total = time.perf_counter()

    # helper: chunked evaluation
    def eval_in_chunks(poly, X, chunk_size):
        n = X.shape[0]
        out = cp.zeros(n, dtype=cp.float64)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            out[start:end] = poly.evaluate(X[start:end])
        return out

    # --- 1. Velocity predictions ---
    t0 = time.perf_counter()
    U_pred_list = [eval_in_chunks(p, X_gpu, chunk_size) for p in velocity_polys]
    U_pred = cp.stack(U_pred_list, axis=1)
    _sync_gpu()
    timings['eval_vel'] = time.perf_counter() - t0

    # --- 2. Data error ---
    t0 = time.perf_counter()
    err_data = cp.mean((U_pred - U_true_gpu)**2)
    _sync_gpu()
    timings['data_err'] = time.perf_counter() - t0

    # --- 3. Incompressibility (∇·u) ---
    t0 = time.perf_counter()
    div_vals = cp.zeros(X_gpu.shape[0], dtype=cp.float64)
    for i, ui_poly in enumerate(velocity_polys):
        du_dx_vals = eval_in_chunks(ui_poly.differentiate(i), X_gpu, chunk_size)
        div_vals += du_dx_vals
    err_incomp = cp.mean(div_vals**2)
    _sync_gpu()
    timings['incompressibility'] = time.perf_counter() - t0

    # --- 4. Navier–Stokes residuals ---
    t0_ns = time.perf_counter()
    err_ns = 0.0
    for i in range(n_velocity):
        # ∂u_i/∂t
        du_dt_vals = eval_in_chunks(velocity_polys[i].differentiate(-1), X_gpu, chunk_size)

        # convection (u · ∇) u_i
        conv_vals = cp.zeros(X_gpu.shape[0], dtype=cp.float64)
        for j in range(n_velocity):
            du_dx_vals = eval_in_chunks(velocity_polys[i].differentiate(j), X_gpu, chunk_size)
            uj_vals = U_pred[:, j]  # already evaluated
            conv_vals += uj_vals * du_dx_vals

        # viscous term ν ∇²u_i
        visc_vals = cp.zeros(X_gpu.shape[0], dtype=cp.float64)
        for j in range(nvars - 1):  # spatial dims only
            d2_vals = eval_in_chunks(velocity_polys[i].differentiate(j).differentiate(j), X_gpu, chunk_size)
            visc_vals += d2_vals
        visc_vals *= nu

        # pressure gradient
        if pressure_poly:
            dp_vals = eval_in_chunks(pressure_poly.differentiate(i), X_gpu, chunk_size)
        else:
            dp_vals = 0.0

        # full residual
        ns_vals = du_dt_vals + conv_vals - visc_vals + dp_vals - g[i]
        err_ns += cp.mean(ns_vals**2)
    _sync_gpu()
    timings['navier_stokes'] = time.perf_counter() - t0_ns

    # --- 5. Weighted total error ---
    total_error = (
        weights["data"] * err_data +
        weights["incompressibility"] * err_incomp +
        weights["navier_stokes"] * err_ns
    )
    timings['total_elapsed'] = time.perf_counter() - t0_total

    if return_components:
        return {
            "total": float(total_error),
            "data": float(err_data),
            "incompressibility": float(err_incomp),
            "navier_stokes": float(err_ns),
            "timings": timings,
        }
    else:
        return float(total_error)


def generate_incompressibility_constraint_refactored(u_s, u_k, grad_s, grad_k):
    div_rest = cp.sum(grad_s[:-1], axis=0)

    phi0 = cp.zeros_like(u_s)
    phi1 = grad_s[-1]
    phi2 = grad_k[-1]
    phi3 = u_s * phi2 + u_k * phi1

    phi_row = [phi0, phi1, phi2, phi3]
    rhs = -div_rest
    return phi_row, rhs


def generate_momentum_constraint_refactored(
    X, u_s, u_k, grad_s, grad_k, u_components, u_grads, u_dt_s, u_dt_k,
    laplace_s, laplace_k, pressure_grads, i, viscosity
):
    n_coords = grad_s.shape[0]
    mask = [j for j in range(n_coords) if j != i]

    u_star = cp.stack([u_components[j] for j in mask])
    grads_star = cp.stack([u_grads[j][j] for j in mask])
    grads_dot = cp.sum(grads_star, axis=0)

    conv_s = cp.sum(u_star * grad_s[mask] - u_s * grads_star, axis=0)
    conv_k = cp.sum(u_star * grad_k[mask] - u_k * grads_star, axis=0)
    conv_sk = cp.sum(
        u_star * (u_k * grad_s[mask] + u_s * grad_k[mask]) - u_s * u_k * grads_star,
        axis=0
    )

    visc_s = cp.sum(laplace_s, axis=0)
    visc_k = cp.sum(laplace_k, axis=0)
    visc_sk = cp.sum(u_k * laplace_s + u_s * laplace_k + 2 * grad_s[mask] * grad_k[mask], axis=0)

    phi0 = -grads_dot
    phi1 = u_dt_s + conv_s - viscosity * visc_s
    phi2 = u_dt_k + conv_k - viscosity * visc_k
    phi3 = u_s * u_dt_k + u_k * u_dt_s + conv_sk - viscosity * visc_sk

    phi_row = [phi0, phi1, phi2, phi3]
    
    pressure_grad = (
    cp.asarray(pressure_grads[i]) if pressure_grads is not None else cp.zeros_like(phi0)
)

    return phi_row, pressure_grad

# ============================================================
# Adaptive chunk sizing utility
# ============================================================
def get_adaptive_chunk_size(n_cols, dtype=cp.float32, safety=0.5, 
                            min_chunk=10_000, max_chunk=200_000):
    """Estimate safe chunk size based on current free GPU memory."""
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    bytes_per_row = n_cols * cp.dtype(dtype).itemsize
    max_rows = int((free_mem * safety) // bytes_per_row)
    return max(min_chunk, min(max_rows, max_chunk))


# ============================================================
# Chunked evaluation (adaptive)
# ============================================================
def evaluate_models_chunked(models, X_cpu, chunk_size=None):
    """Evaluate a list of models on X_cpu in GPU-sized chunks."""
    n_samples = X_cpu.shape[0]
    n_models = len(models)
    results = np.empty((n_models, n_samples), dtype=np.float32)

    if chunk_size is None:
        chunk_size = get_adaptive_chunk_size(X_cpu.shape[1], dtype=np.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk_gpu = cp.asarray(X_cpu[start:end])

        for i, model in enumerate(models):
            res_gpu = model.evaluate(X_chunk_gpu)
            results[i, start:end] = cp.asnumpy(res_gpu)

        # cleanup
        del X_chunk_gpu, res_gpu
        _sync_gpu()

    return results


# ============================================================
# Chunked evaluation with differentiation (adaptive)
# ============================================================
def evaluate_models_chunked_diff(models, X_cpu, d, chunk_size=None):
    """Evaluate models differentiated wrt variable `d` on X_cpu in chunks."""
    n_samples = X_cpu.shape[0]
    n_models = len(models)
    results = np.empty((n_models, n_samples), dtype=np.float32)

    if chunk_size is None:
        chunk_size = get_adaptive_chunk_size(X_cpu.shape[1], dtype=np.float32)

    # precompute differentiated models once
    diff_models = [m.differentiate(d) for m in models]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk_gpu = cp.asarray(X_cpu[start:end])

        for i, diff_model in enumerate(diff_models):
            res_gpu = diff_model.evaluate(X_chunk_gpu)
            results[i, start:end] = cp.asnumpy(res_gpu)

        # cleanup
        del X_chunk_gpu, res_gpu
        _sync_gpu()

    return results

def compute_monomial_correlations_gpu(phi, y):
    """
    Compute Pearson correlation between each column in phi (N x m) and target y (N,).
    Returns abs(correlation) and raw correlations arrays (length m).
    All inputs are CuPy arrays.
    """
    N = phi.shape[0]
    # ensure float64 for stability
    phi = phi.astype(cp.float64)
    y = y.astype(cp.float64)
    y_mean = cp.mean(y)
    y_centered = y - y_mean
    y_var = cp.sum(y_centered * y_centered)

    # column means and centered columns
    col_mean = cp.mean(phi, axis=0)       # (m,)
    phi_centered = phi - col_mean[None, :]
    cov = phi_centered.T @ y_centered     # (m,)
    phi_var = cp.sum(phi_centered * phi_centered, axis=0)  # (m,)

    # avoid division by zero
    denom = cp.sqrt(phi_var * (y_var + 0.0))
    # denom may be zero; set those correlations to 0
    corr = cp.zeros_like(denom)
    nz = denom > 0
    corr[nz] = cov[nz] / denom[nz]
    return cp.abs(corr), corr  # abs and signed

def prune_polynomial_by_correlation_chunked(poly: PolynomialGPU, X_gpu, y_gpu,
                                            corr_threshold,
                                            chunk_size=None):
        """
        Prune low-correlation monomials.  Starts from lowest correlation
        and removes progressively until validation error increases.

        Returns:
            (pruned_poly, prune_report)
        """
        # Step 1: compute correlations on GPU
        corrs = poly.compute_correlations(X_gpu, y_gpu, chunk_size=chunk_size)
        abs_corr = cp.abs(corrs)

        # Step 2: sort monomials by correlation
        sorted_idx = cp.argsort(abs_corr)
        keep_mask = cp.ones_like(abs_corr, dtype=bool)

        base_pred = poly.evaluate(X_gpu)
        base_mse = float(cp.mean((base_pred - y_gpu) ** 2))

        # Step 3: progressive pruning
        last_mse = base_mse
        phi_means = cp.mean(poly._evaluate_monomials_gpu(X_gpu), axis=0)
        pruned_idx = []
        for idx in sorted_idx:
            idx = int(cp.asnumpy(idx))
            if abs_corr[idx] > corr_threshold:
                break
            # substitute term by mean
            tmp_poly = poly.substitute_term_by_mean(idx, phi_means[idx])
            keep_mask[idx] = False
            # evaluate new MSE
            pred = tmp_poly.evaluate(X_gpu)
            new_mse = float(cp.mean((pred - y_gpu) ** 2))
            if new_mse > last_mse:
                keep_mask[idx] = True  # revert
                break
            last_mse = new_mse
            pruned_idx.append(idx)

        pruned_poly = poly.remove_terms_mask(keep_mask)
        report = {
            'initial_terms': int(poly.coeffs.size),
            'pruned_terms': int(cp.sum(~keep_mask)),
            'final_terms': int(pruned_poly.coeffs.size),
            'base_mse': base_mse,
            'final_mse': last_mse,
            'threshold': float(corr_threshold),
            'pruned_idx': pruned_idx
        }
        return pruned_poly, report

# --- Main class ---

class PhysicsAwareGMDH:
    def __init__(self, n_features=3, max_layer=7, top_models=25, viscosity=0.01):
        self.n_features = n_features
        self.max_layer = max_layer
        self.top_models = top_models
        self.viscosity = viscosity
        self.models = [self._initialize_first_layer(n_components=1)]
        self.err_line = []
        self.y_pred = None
        self.current_layer = 0

        # Placeholders for training data and derivatives (CPU numpy)
        self.X_cpu = None
        self.y_cpu = None
        self.u_tensor_cpu = None
        self.u_grads_tensor_cpu = None
        self.u_dt_cpu = None
        self.u_laplace_cpu = None

        self.timers = TimeAgg()

    def _initialize_first_layer(self, n_components):
        return [[
            PolynomialGPU.from_dict({tuple(1 if j == i else 0 for j in range(self.n_features)): 1})
            for i in range(self.n_features)
        ] for _ in range(n_components)]

    def _build_phi_matrix(self, u_s, u_k):
        # u_s, u_k are CuPy arrays on GPU for chunked evaluation in delayed tasks
        return cp.stack([
            cp.ones_like(u_s),
            u_s,
            u_k,
            u_s * u_k
        ], axis=1)

    def _eval_velocity_tensor(self, previous_models, X_cpu):
        """
        Evaluate velocity components for each component in previous_models on X_cpu.
        Returns velocity components with shape (n_components, n_models, N)
        """
        velocity_components = []
        for comp_models in previous_models:
            velocity_components.append(evaluate_models_chunked(comp_models, X_cpu))
        return np.stack(velocity_components, axis=0)

    def _eval_grad_tensor(self, previous_models, X_cpu):
        """
        Evaluate gradient components for each component in previous_models on X_cpu.
        Returns array with shape:
        (n_components, n_models, n_vars, N)
        """
        grad_all = []
        n_vars = X_cpu.shape[1]

        for comp_models in previous_models:
            grads_comp = []
            for model in comp_models:
                grads = []
                for d in range(n_vars):
                    grad_vals = evaluate_models_chunked(
                        [model.differentiate(d)], X_cpu
                    )[0]
                    grads.append(grad_vals)
                grads_comp.append(np.stack(grads, axis=0))  # (n_vars, N)
            grad_all.append(np.stack(grads_comp, axis=0))  # (n_models, n_vars, N)

        return np.stack(grad_all, axis=0)  # (n_components, n_models, n_vars, N)


    def _eval_time_derivatives(self, previous_models, X_cpu):
        """
        Returns u_dt_cpu with shape:
        (n_components, n_models, N)
        """
        dt_all = []
        t_idx = X_cpu.shape[1] - 1
        for comp_models in previous_models:
            dt_comp = []
            for model in comp_models:
                dt_vals = evaluate_models_chunked(
                    [model.differentiate(t_idx)], X_cpu
                )[0]
                dt_comp.append(dt_vals)
            dt_all.append(np.stack(dt_comp, axis=0))
        return np.stack(dt_all, axis=0)
    
    def _eval_laplacian(self, previous_models, X_cpu):
        """
        Returns u_laplace_cpu with shape:
        (n_components, n_models, N)
        """
        lap_all = []
        n_spatial = X_cpu.shape[1] - 1

        for comp_models in previous_models:
            lap_comp = []
            for model in comp_models:
                terms = []
                for d in range(n_spatial):
                    d2 = model.differentiate(d).differentiate(d)
                    vals = evaluate_models_chunked([d2], X_cpu)[0]
                    terms.append(vals)
                lap_comp.append(np.sum(np.stack(terms, axis=0), axis=0))
            lap_all.append(np.stack(lap_comp, axis=0))
        return np.stack(lap_all, axis=0)

    def _to_gpu(self, **arrays):
        out = {}
        for k, v in arrays.items():
            out[k] = cp.asarray(v) if v is not None else None
        _sync_gpu()
        return out

    def _least_squares_step(self, pressure_poly, weights, group_weights=None):
        """
        Perform constrained least squares for all (component, s, k) pairs.
        LOGIC IDENTICAL to original version.
        """

        previous_models = self.models[-1]
        n_components = len(previous_models)
        X_cpu = self.X_cpu

        # --- weights ---
        w_data = weights.get("data", 1.0)
        w_incomp = weights.get("incomp", 1.0)
        w_mom = weights.get("mom", 1.0)

        # ======================================================
        # A. Evaluate parent model fields on CPU
        # ======================================================

        with self.timers.cpu_block("eval_u_tensor"):
            u_tensor_cpu = self._eval_velocity_tensor(previous_models, X_cpu)

        u_grads_tensor_cpu = None
        if w_incomp > 0:
            with self.timers.cpu_block("eval_u_grads"):
                u_grads_tensor_cpu = self._eval_grad_tensor(previous_models, X_cpu)

        u_dt_cpu = None
        u_laplace_cpu = None
        pressure_grads_cpu = None

        if w_mom > 0:
            with self.timers.cpu_block("eval_dt"):
                u_dt_cpu = self._eval_time_derivatives(previous_models, X_cpu)

            with self.timers.cpu_block("eval_laplace"):
                u_laplace_cpu = self._eval_laplacian(previous_models, X_cpu)

            # pressure gradients (unchanged logic)
            if self.pressures:
                p_grads = []
                for p in self.pressures:
                    grads = []
                    for d in range(X_cpu.shape[1] - 1):
                        vals = evaluate_models_chunked(
                            [p.differentiate(d)], X_cpu
                        )[0]
                        grads.append(vals)
                    p_grads.append(np.stack(grads, axis=0))
                pressure_grads_cpu = np.array(p_grads)

        # ======================================================
        # B. Move large tensors to GPU
        # ======================================================

        with self.timers.cpu_block("to_gpu"):
            gpu_state = self._to_gpu(
                X=X_cpu,
                y=self.y_cpu,
                u_tensor=u_tensor_cpu,
                u_grads=u_grads_tensor_cpu,
                u_dt=u_dt_cpu,
                u_laplace=u_laplace_cpu,
            )

        # ======================================================
        # C. Build model_state (IDENTICAL CONTENT)
        # ======================================================

        model_state = {
            "X_cpu": X_cpu,
            "X_gpu": gpu_state["X"],
            "y_cpu": self.y_cpu,
            "y_gpu": gpu_state["y"],
            "u_tensor_cpu": u_tensor_cpu,
            "u_tensor_gpu": gpu_state["u_tensor"],
            "u_grads_tensor_cpu": u_grads_tensor_cpu,
            "u_grads_tensor_gpu": gpu_state["u_grads"],
            "u_dt_gpu": gpu_state["u_dt"],
            "u_laplace_gpu": gpu_state["u_laplace"],
            "pressure_grads_cpu": pressure_grads_cpu,
            "viscosity": self.viscosity,
            "w_data": w_data,
            "w_incomp": w_incomp,
            "w_mom": w_mom,
            "parent_group_weights": group_weights,
            "build_phi": self._build_phi_matrix,
            "generate_incompressibility": generate_incompressibility_constraint_refactored,
            "generate_momentum": generate_momentum_constraint_refactored,
        }

        # ======================================================
        # D. Build Dask tasks (s,k) and (k,s)
        # ======================================================

        tasks = []
        with self.timers.cpu_block("build_tasks"):
            for comp_idx, comp_models in enumerate(previous_models):
                n_models = len(comp_models)
                for a in range(n_models):
                    for b in range(a + 1, n_models):
                        tasks.append(
                            evaluate_model_pair_delayed(comp_idx, a, b, model_state)
                        )
                        tasks.append(
                            evaluate_model_pair_delayed(comp_idx, b, a, model_state)
                        )

        # ======================================================
        # E. Execute tasks
        # ======================================================

        with self.timers.cpu_block("dask_compute"):
            results = dask.compute(*tasks, scheduler="threads")

        _sync_gpu()

        # ======================================================
        # F. Collect results
        # ======================================================

        candidates = []
        error_vectors = []
        y_preds = []

        for res in results:
            if res is None:
                continue
            comp_idx, s_idx, k_idx, coeffs, errors, y_pred = res
            candidates.append((comp_idx, s_idx, k_idx, coeffs))
            error_vectors.append(errors)
            y_preds.append(y_pred)

        print(f"Found {len(candidates)} candidates in least squares step.")
        return candidates, error_vectors, y_preds

    def fit(self, X, y):
        # -------------------------------------------------
        # 0. Preparation
        # -------------------------------------------------
        err_lines = [[], [], []]
        limit = min(1000, X.shape[0])

        self.X_cpu = np.asarray(X[:limit])
        self.y_cpu = np.asarray(y[:, :limit])  # (n_components, N)

        self.X_gpu_test = cp.asarray(X[:limit])
        self.y_gpu_test = cp.asarray(y[:, :limit]).T  # (N, n_components)

        n_components = y.shape[0]

        # -------------------------------------------------
        # 1. Base layer initialization
        # -------------------------------------------------
        if self.current_layer == 0:
            self.models = [self._initialize_first_layer(n_components=n_components)]
            self.pressures = []

            # compute pressure for base velocity models
            velocities_list = list(map(list, zip(*self.models[0])))
            for velocities in velocities_list:
                pressure = (
                    compute_pressure_from_polynomials(velocities, viscosity=self.viscosity)
                    if all(p.exponents.size > 0 for p in velocities)
                    else None
                )
                self.pressures.append(pressure)

            # attach default group weights
            for comp_models in self.models[-1]:
                for poly in comp_models:
                    poly.group_weights = {
                        "data": 1.0,
                        "incomp": 1.0,
                        "momentum": 1.0,
                    }

        selected_group_weights = None

        # -------------------------------------------------
        # 2. Layer-wise GMDH loop
        # -------------------------------------------------
        while self.current_layer < self.max_layer:
            print(f"\nLayer {self.current_layer + 1} processing...")

            free, total = cp.cuda.runtime.memGetInfo()
            print(f"GPU free memory: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
            print(f"RMM allocated: {tracked.get_allocated_bytes() / 1e6:.2f} MB")

            global_weights = {"data": 1.0, "incomp": 1.0, "mom": 1.0}

            # -------------------------------------------------
            # 2.1 Least-squares candidate generation
            # -------------------------------------------------
            candidates, errors_ls, preds = self._least_squares_step(
                pressure_poly=None,
                weights=global_weights,
                group_weights=selected_group_weights,
            )

            if not candidates:
                break

            # -------------------------------------------------
            # 2.2 Build full velocity models from candidates
            # -------------------------------------------------
            velocities_list = []
            coeffs_list = []
            sidxes_list = []
            kidxes_list = []
            compidxes_list = []

            for comp_idx, s, k, coef in candidates:
                coeffs_list.append(coef)
                sidxes_list.append(s)
                kidxes_list.append(k)
                compidxes_list.append(comp_idx)

                model = self.models[-1][comp_idx][s].combine_with_gpu(
                    self.models[-1][comp_idx][k], *coef
                )

                velocities = [
                    self.models[-1][j][s] if j != comp_idx else model
                    for j in range(n_components)
                ]
                velocities_list.append(velocities)

            # -------------------------------------------------
            # 2.3 Physics + data error evaluation
            # -------------------------------------------------
            physics_errors = []
            err_data = []
            err_incomp = []
            err_ns = []
            timings = []

            for velocities in velocities_list:
                physics_err = gmdh_error_lightweight_eval(
                    self.X_gpu_test,
                    self.y_gpu_test,
                    velocity_polys=velocities,
                    pressure_poly=self.pressures[0],
                    nu=self.viscosity,
                    g=None,
                    return_components=True,
                )

                timings.append(physics_err["timings"])
                physics_errors.append(physics_err["total"])
                err_data.append(physics_err["data"])
                err_incomp.append(physics_err["incompressibility"])
                err_ns.append(physics_err["navier_stokes"])

            # -------------------------------------------------
            # 2.4 Select best candidates
            # -------------------------------------------------
            physics_errors = np.asarray(physics_errors)
            best = np.argsort(physics_errors)[: self.top_models]

            velocities_list = [velocities_list[i] for i in best]

            self.err_line.append(float(physics_errors[best[0]]))
            err_lines[0].append(float(err_data[best[0]]))
            err_lines[1].append(float(err_incomp[best[0]]))
            err_lines[2].append(float(err_ns[best[0]]))

            # -------------------------------------------------
            # 2.5 Prune monomials (component-wise)
            # -------------------------------------------------
            pruned_velocities = []
            prune_reports = []

            for velocities in velocities_list:
                pruned = []
                for comp_idx, poly in enumerate(velocities):
                    y_ref = self.y_gpu_test[:, comp_idx]
                    poly_p, report = prune_polynomial_by_correlation_chunked(
                        poly,
                        self.X_gpu_test,
                        y_ref,
                        corr_threshold=1e-3,
                    )
                    pruned.append(poly_p)
                    prune_reports.append(report)
                pruned_velocities.append(pruned)

            velocities_list = pruned_velocities
            print("Prune reports:", prune_reports)

            # -------------------------------------------------
            # 2.6 Estimate group weights for selected models
            # -------------------------------------------------
            eps = 1e-12
            selected_group_weights = []

            for i in best:
                wd = 1.0 / (err_data[i] + eps)
                wi = 1.0 / (err_incomp[i] + eps)
                wm = 1.0 / (err_ns[i] + eps)
                s = wd + wi + wm + eps
                selected_group_weights.append({
                    "data": float(wd / s),
                    "incomp": float(wi / s),
                    "momentum": float(wm / s),
                })

            for cand_idx, gw in enumerate(selected_group_weights):
                for poly in velocities_list[cand_idx]:
                    poly.group_weights = gw

            # -------------------------------------------------
            # 2.7 Recompute pressure polynomials
            # -------------------------------------------------
            self.pressures = []
            for velocities in velocities_list:
                pressure = (
                    compute_pressure_from_polynomials(velocities, viscosity=self.viscosity)
                    if all(p.exponents.size > 0 for p in velocities)
                    else None
                )
                self.pressures.append(pressure)

            # -------------------------------------------------
            # 2.8 Store new layer
            # -------------------------------------------------
            new_models = list(map(list, zip(*velocities_list)))
            self.models.append(new_models)

            self.y_pred = [preds[int(best[0])]]
            self.current_layer += 1

        # -------------------------------------------------
        # 3. Reporting
        # -------------------------------------------------
        print("\nErrors (data / incomp / NS):")
        print(err_lines[0])
        print(err_lines[1])
        print(err_lines[2])

        return self

    

    def plot_profile(self, var_index=0, fixed_values=None, n_points=100, u_true_funcs=None):
        n_vars = self.X_cpu.shape[1]
        if fixed_values is None:
            fixed_values = np.mean(self.X_cpu, axis=0).tolist()

        x_range = np.linspace(np.min(self.X_cpu[:, var_index]), np.max(self.X_cpu[:, var_index]), n_points)
        X_test = np.tile(np.array(fixed_values), (n_points, 1))
        X_test[:, var_index] = x_range

        top_model = self.models[-1]
        preds = []
        for comp_models in top_model:
            best_model = comp_models[0]
            u_pred = best_model.evaluate(cp.asarray(X_test))
            preds.append(cp.asnumpy(u_pred))

        for i, (pred, true_label) in enumerate(zip(preds, ["$u_1$", "$u_2$"])):
            plt.figure(figsize=(15, 4))
            plt.plot(x_range, pred, label=f"Predicted {true_label}", linestyle="--")

            if u_true_funcs is not None:
                args = [X_test[:, j] for j in range(n_vars)]
                u_actual = u_true_funcs[i](*args)
                plt.plot(x_range, u_actual, label=f"Actual {true_label}", linestyle="-")

            plt.title(f"{true_label} vs $x_{{{var_index+1}}}$ (fixed others)")
            plt.xlabel(f"$x_{{{var_index+1}}}$")
            plt.ylabel(true_label)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()


# --- Usage example ---

if __name__ == "__main__":
    import time
    global_start = time.perf_counter()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Dask-CUDA cluster started:", client)

    # Define velocity functions
    def u1(x, y, t): return np.cos(x) * np.sin(y) * np.exp(-2 * 0.01 * t)
    def u2(x, y, t): return -np.sin(x) * np.cos(y) * np.exp(-2 * 0.01 * t)

    n = 2000
    x, y, t = np.random.rand(n), np.random.rand(n), np.random.rand(n)
    X = np.column_stack([x, y, t])

    u1_vals = u1(x, y, t).reshape(-1)
    u2_vals = u2(x, y, t).reshape(-1)
    
     # Fit the model
     # Note: For real use, increase n and max_layer for better results
     # Here we keep them small for demonstration purposes
     # Also, in a real scenario, use more data points for training
     # and possibly a validation set to monitor overfitting.
     # The current settings are for quick testing only.
     # In practice, n should be in the thousands or more.    
    model = PhysicsAwareGMDH(n_features=3, max_layer=3, top_models=25, viscosity=0.01)
    model.fit(X, y=np.stack([u1_vals, u2_vals]))
    global_end = time.perf_counter()
    print(f"\nTotal program execution time: {global_end - global_start:.2f} s")
    #plt.plot(model.err_line)
    model.plot_profile(var_index=2, u_true_funcs=[u1, u2])
