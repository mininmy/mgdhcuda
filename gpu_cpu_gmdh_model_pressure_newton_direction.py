# ============================================================
# A. CUDA / RMM / Dask setup
# ============================================================

import cupy as cp
import numpy as np
import rmm
from rmm import mr
from rmm.allocators.cupy import rmm_cupy_allocator

import dask
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

# --- RMM Memory Setup ---
INITIAL_POOL_SIZE = 1 * 1024**3
MAX_POOL_SIZE     = 12 * 1024**3

upstream = mr.CudaMemoryResource()
pool = mr.PoolMemoryResource(
    upstream,
    initial_pool_size=INITIAL_POOL_SIZE,
    maximum_pool_size=MAX_POOL_SIZE
)
tracked = mr.TrackingResourceAdaptor(pool)

rmm.mr.set_current_device_resource(tracked)
cp.cuda.set_allocator(rmm_cupy_allocator)
dask.config.set(scheduler="threads")

def print_gpu_mem():
    """Print current GPU free/total memory and RMM allocated bytes."""
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"GPU free memory: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
    print(f"RMM allocated: {tracked.get_allocated_bytes() / 1e6:.2f} MB")

print_gpu_mem()

# ============================================================
# B. Timing utilities
# ============================================================

import time
from contextlib import contextmanager
from collections import defaultdict
from phi import PhiData, build_phi_data, PhiCache, PhiDescriptor, generate_phi_descriptors

def _sync_gpu():
    try:
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        pass

class TimeAgg:
    def __init__(self):
        self.cpu = defaultdict(float)
        self.gpu = defaultdict(float)

    @contextmanager
    def cpu_block(self, key):
        t0 = time.perf_counter()
        yield
        self.cpu[key] += time.perf_counter() - t0

    @contextmanager
    def gpu_block(self, key):
        start = cp.cuda.Event()
        end   = cp.cuda.Event()
        _sync_gpu()
        start.record()
        yield
        end.record()
        end.synchronize()
        self.gpu[key] += cp.cuda.get_elapsed_time(start, end)

    def report(self):
        if self.cpu:
            print("\n=== CPU Timings (s) ===")
            for k, v in sorted(self.cpu.items(), key=lambda x: -x[1]):
                print(f"{k:30s} : {v:9.4f}")
        if self.gpu:
            print("\n=== GPU Timings (ms) ===")
            for k, v in sorted(self.gpu.items(), key=lambda x: -x[1]):
                print(f"{k:30s} : {v:9.2f}")

# ============================================================
# C. GPU chunked evaluation helpers
# ============================================================

def get_adaptive_chunk_size(n_cols, dtype=cp.float32,
                            safety=0.5,
                            min_chunk=10_000,
                            max_chunk=200_000):
    free_mem, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_row = n_cols * cp.dtype(dtype).itemsize
    max_rows = int((free_mem * safety) // bytes_per_row)
    return max(min_chunk, min(max_rows, max_chunk))


def evaluate_models_chunked(models, X_cpu, chunk_size=None):
    n_samples = X_cpu.shape[0]
    n_models  = len(models)
    out = np.empty((n_models, n_samples), dtype=np.float32)

    if chunk_size is None:
        chunk_size = get_adaptive_chunk_size(X_cpu.shape[1])

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        Xg = cp.asarray(X_cpu[start:end])

        for i, m in enumerate(models):
            out[i, start:end] = cp.asnumpy(m.evaluate(Xg))

        del Xg
        _sync_gpu()

    return out


def evaluate_models_chunked_diff(models, X_cpu, d, chunk_size=None):
    diff_models = [m.differentiate(d) for m in models]
    return evaluate_models_chunked(diff_models, X_cpu, chunk_size)

# ============================================================
# F. PhysicsAwareGMDH
# ============================================================

from gpu_polynomial_module import PolynomialGPU
from pressure_estimator import compute_pressure_from_polynomials

from dataclasses import dataclass

@dataclass
class ResidualBlocks:
    data: list          # list[cp.ndarray]
    incompressibility: cp.ndarray
    momentum: list      # list[cp.ndarray]

    def stack(self, weights=None):
        blocks = []

        if weights is None:
            blocks += self.data
            blocks.append(self.incompressibility)
            blocks += self.momentum
        else:
            blocks += [weights["data"] * r for r in self.data]
            blocks.append(weights["div"] * self.incompressibility)
            blocks += [weights["mom"] * r for r in self.momentum]

        return cp.concatenate(blocks)

class PhysicsAwareGMDH:
    def __init__(self, n_features, max_layers, top_models, viscosity):
        self.n_features = n_features
        
        self.viscosity = viscosity
        self.max_layers = max_layers
        self.top_models = top_models

        self.phi_cache = {}
        self.layers = []

        self.models = []
        self.current_layer = 0

        # CPU tensors (large, reused)
        self.X_cpu = None
        self.u_cpu = None
        self.grad_u_cpu = None
        self.dt_u_cpu = None
        self.lap_u_cpu = None
        self.timers = TimeAgg()

    @staticmethod
    def cpu_to_gpu_readonly(arr_cpu, name=None):
        """
        Transfer a CPU NumPy array to GPU as a read-only CuPy array.

        Args:
            arr_cpu: CPU array (NumPy-compatible).
            name: Optional label for debugging/errors.
        """
        try:
            arr_gpu = cp.asarray(arr_cpu)
        except Exception as e:
            label = f" '{name}'" if name else ""
            raise TypeError(f"Failed to transfer{label} to GPU") from e

        # Best-effort immutability (works for CuPy ndarrays)
        try:
            arr_gpu.flags.writeable = False
        except Exception:
            pass
        return arr_gpu
        
    def _initialize_first_layer(self, n_components):
        return [
            [
                PolynomialGPU.from_dict({
                    tuple(1 if j == i else 0 for j in range(self.n_features)): 1.0
                })
                for i in range(self.n_features)
            ]
            for _ in range(n_components)
        ]
    # ------------------------------------------------------------
    # CPU tensor evaluation (frozen fields)
    # ------------------------------------------------------------

    def _eval_velocity_tensor(self, models, X_cpu):
        return np.stack(
            [evaluate_models_chunked(comp, X_cpu) for comp in models],
            axis=0
        )

    def _eval_grad_tensor(self, models, X_cpu):
        n_vars = X_cpu.shape[1]
        grad_all = []

        for comp in models:
            comp_grads = []
            for m in comp:
                grads = [
                    evaluate_models_chunked([m.differentiate(d)], X_cpu)[0]
                    for d in range(n_vars)
                ]
                comp_grads.append(np.stack(grads, axis=0))
            grad_all.append(np.stack(comp_grads, axis=0))

        return np.stack(grad_all, axis=0)

    def _eval_time_derivative(self, models, X_cpu):
        t_idx = X_cpu.shape[1] - 1
        return np.stack(
            [
                np.stack(
                    [evaluate_models_chunked([m.differentiate(t_idx)], X_cpu)[0]
                     for m in comp],
                    axis=0
                )
                for comp in models
            ],
            axis=0
        )

    def _eval_laplacian(self, models, X_cpu):
        n_spatial = X_cpu.shape[1] - 1
        lap_all = []

        for comp in models:
            comp_lap = []
            for m in comp:
                lap = sum(
                    evaluate_models_chunked([m.differentiate(d).differentiate(d)], X_cpu)[0]
                    for d in range(n_spatial)
                )
                comp_lap.append(lap)
            lap_all.append(np.stack(comp_lap, axis=0))

        return np.stack(lap_all, axis=0)

    def freeze_fields(self, models, X_cpu):
        self.u_cpu = self._eval_velocity_tensor(models, X_cpu)
        self.grad_u_cpu = self._eval_grad_tensor(models, X_cpu)
        self.dt_u_cpu = self._eval_time_derivative(models, X_cpu)
        self.lap_u_cpu = self._eval_laplacian(models, X_cpu)

        # MOVE TO GPU ONCE
        self.u_gpu = cp.asarray(self.u_cpu)
        self.grad_u_gpu = cp.asarray(self.grad_u_cpu)
        self.dt_u_gpu = cp.asarray(self.dt_u_cpu)
        self.lap_u_gpu = cp.asarray(self.lap_u_cpu)

    def _op_convection(self, u, grad_v):
        # u · ∇v
        R = cp.zeros_like(grad_v[0])
        for j in range(len(u)):
            R += u[j] * grad_v[j]
        return R


    def _op_diffusion(self, nu, lap_v):
        return -nu * lap_v
    def evaluate_momentum_residual(
        self,
        *,
        u,
        du_dx,
        du_dt,
        laplace_u,
        p_grad,
        forcing,
        nu,
    ):
        d = len(u)
        mom_res = []

        for k in range(d):
            R = du_dt[k]
            R += self._op_convection(u, du_dx[k])
            R += self._op_diffusion(nu, laplace_u[k])
            R += p_grad[k]
            R -= forcing[k]
            mom_res.append(R)

        return mom_res
    def evaluate_momentum_linearized(
        self,
        *,
        u,
        du_dx,
        nu,
        phi_i,
        dphi_dx_i,
        dphi_dt_i,
        laplace_phi_i,
        component_i,
    ):
        d = len(u)
        mom_res = []

        for k in range(d):
            R = cp.zeros_like(phi_i)

            if k == component_i:
                R += dphi_dt_i
                R += self._op_convection(u, dphi_dx_i)
                R += phi_i * du_dx[k][k]
                R += self._op_diffusion(nu, laplace_phi_i)
            else:
                R += phi_i * du_dx[k][component_i]

            mom_res.append(R)

        return mom_res

    def _op_divergence(self, grad_field):
        R = cp.zeros_like(grad_field[0])
        for j in range(len(grad_field)):
            R += grad_field[j]
        return R
    
    def evaluate_data_linearized(self, *, phi_i, component_i, n_components):
        res = []
        for j in range(n_components):
            if j == component_i:
                res.append(phi_i)
            else:
                res.append(cp.zeros_like(phi_i))
        return res
    def evaluate_linearized_residual(
        self,
        *,
        state,
        basis,
        component_i,
    ):
        return ResidualBlocks(
            data=self.evaluate_data_linearized(
                phi_i=basis["phi"][component_i],
                component_i=component_i,
                n_components=len(state["u"]),
            ),
            incompressibility=self._op_divergence(
                basis["dphi_dx"][component_i]
            ),
            momentum=self.evaluate_momentum_linearized(
                u=state["u"],
                du_dx=state["du_dx"],
                nu=state["nu"],
                phi_i=basis["phi"][component_i],
                dphi_dx_i=basis["dphi_dx"][component_i],
                dphi_dt_i=basis["dphi_dt"][component_i],
                laplace_phi_i=basis["laplace_phi"][component_i],
                component_i=component_i,
            ),
        )
  
    def _frozen_operator_ls_step(self, residual_i, delta_R_list):
        """
        Solve:
            min || R_i + Σ α_k δR_i(φ_k) ||²
        """

        A = cp.stack(delta_R_list, axis=1)   # (N, n_basis)
        ATA = A.T @ A
        ATb = A.T @ residual_i

        alpha = cp.linalg.solve(ATA, -ATb)
        return alpha
  
    def _newton_correction_step(
        self,
        *,
        phi_descriptors,
        phi_cache,
        residual_blocks: ResidualBlocks,
        i_component,
        weights=None,
    ):
        """
        Frozen-operator Newton correction.

        Solves:
            min || R_i + Σ α_k δR_i(φ_k) ||²

        Parameters
        ----------
        phi_descriptors : list
            Candidate φ descriptors.
        phi_cache : PhiCache
            Cached φ evaluations.
        residual_blocks : ResidualBlocks
            Base residual (already computed).
        i_component : int
            Velocity component being corrected.
        weights : dict or None
            Optional block weights.
        """

        # ------------------------------------------------------------
        # 1. Stack base residual
        # ------------------------------------------------------------
        R_vec = residual_blocks.stack(weights=weights)

        # ------------------------------------------------------------
        # 2. Build linearized residual matrix A
        # ------------------------------------------------------------
        delta_R_cols = []

        state = {
            "u": self.u_gpu,
            "du_dx": self.grad_u_gpu,
            "nu": self.viscosity,
        }

        for desc in phi_descriptors:

            phi_data = phi_cache.get(desc)
            if phi_data is None:
                phi_data = build_phi_data(
                    desc,
                    self.u_cpu,
                    self.dt_u_cpu,
                    self.grad_u_cpu,
                    self.lap_u_cpu,
                )
                phi_cache.put(desc, phi_data)

            # ---- Linearized residual blocks
            dR_blocks = self.evaluate_linearized_residual(
                state=state,
                basis=phi_data,
                component_i=i_component,
            )

            # ---- Stack to vector
            delta_R_cols.append(dR_blocks.stack(weights=weights))

        # ------------------------------------------------------------
        # 3. Assemble LS matrix
        # ------------------------------------------------------------
        A = cp.stack(delta_R_cols, axis=1)  # (N_residual, n_basis)

        # ------------------------------------------------------------
        # 4. Solve normal equation
        # ------------------------------------------------------------
        ATA = A.T @ A
        ATb = A.T @ R_vec

        alpha = cp.linalg.solve(ATA, -ATb)

        return alpha


    def _prepare_data(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
    
    def _generate_phi_candidates(self, layer_idx):
        return generate_phi_descriptors(n_u=self.n_components)

    def _build_phi_cache(self, phi_descriptors):
        cache = {}

        for desc in phi_descriptors:
            if desc not in self.phi_cache:
                self.phi_cache[desc] = build_phi_data(desc, self.X)

            cache[desc] = self.phi_cache[desc]

        return cache

    # ------------------------------------------------------------
    # Fit (high-level skeleton)
    # ------------------------------------------------------------
    def _fit_single_candidate(self, phi_desc, phi_cache):
        phi = phi_cache[phi_desc]

        # 1. initial LS fit
        coeffs = self._initial_ls(phi)

        # 2. physics-aware Newton correction
        coeffs = self._newton_correction_step(
            phi_desc=phi_desc,
            phi_data=phi,
            coeffs=coeffs,
        )

        # 3. evaluate errors
        error_vec = self._evaluate_error_vector(
            phi_desc,
            phi,
            coeffs,
        )

        return CandidateModel(
            phi_desc=phi_desc,
            coeffs=coeffs,
            error_vec=error_vec,
        )
    def _select_best_models(self, models):
        scores = cp.array([
            cp.dot(self.error_weights, m.error_vec)
            for m in models
        ])


        idx = cp.argsort(scores)[: self.n_best]
        return [models[i] for i in idx]
    
    def _freeze_layer(self, selected_models):
        self.models.append(selected_models)
        

    def fit(self, X, y):
        """
        Phase 1: GMDH selection (unchanged logic)
        Phase 2: Frozen-operator Newton refinement
        """

        self.X_cpu = np.asarray(X)
        self.models = [self._initialize_first_layer(y.shape[0])]
        
        for layer_idx in range(self.max_layer):
            self.freeze_fields(self.models[-1], self.X_cpu)
            phi_descriptors = self._generate_phi_candidates(n_u=y.shape[0])

            self.build_phi_cache(phi_descriptors)

            layer_models = []

            for phi_desc in phi_descriptors:
                model = self._fit_single_candidate(
                    phi_desc,
                    self.phi_cache,
                )
                layer_models.append(model)

            selected = self._select_best_models(layer_models)

            self._freeze_layer(selected)

            if self._early_stop(selected):
                break
        return self

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
