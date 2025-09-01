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
# --- RMM Memory Setup ---
initial_pool_size = 512 * 1024 * 1024       # 512 MB
max_pool_size = 10 * 1024 * 1024 * 1024     # 10 GB max pool

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


# --- Helper functions ---

import cupy as cp

def gmdh_error_lightweight(X_gpu, U_true_gpu, velocity_polys, pressure_poly=None, nu=0.0, g=None, return_components=False):
    """
    Lightweight error estimator for Physics-Aware GMDH (GPU, CuPy).
    
    Parameters
    ----------
    X_gpu : cp.ndarray
        Input matrix (N x n_features), including spatial + time coords.
    U_true_gpu : cp.ndarray
        True velocity values (N x n_velocity).
    velocity_polys : list
        List of polynomial models for each velocity component.
    pressure_poly : Polynomial or None
        Pressure polynomial (optional).
    nu : float
        Viscosity.
    g : cp.ndarray or None
        External forcing vector (length n_velocity).
    return_components : bool
        If True, return dict with separate error components.
    
    Returns
    -------
    total_error : float
        Scalar total error if return_components=False
    components : dict
        Dictionary with error contributions if return_components=True
    """

    n_velocity = U_true_gpu.shape[1]
    g = g if g is not None else cp.zeros(n_velocity, dtype=cp.float64)

    # --- 1. Evaluate predicted velocities
    U_pred = cp.stack([p.evaluate(X_gpu) for p in velocity_polys], axis=1)

    # --- 2. Data error
    err_data = cp.mean((U_pred - U_true_gpu) ** 2)

    # --- 3. Incompressibility: div(u) = sum_i d(ui)/dxi
    div_terms = [ui_poly.differentiate(i).evaluate(X_gpu) for i, ui_poly in enumerate(velocity_polys)]
    div_u = cp.sum(cp.stack(div_terms, axis=0), axis=0)
    err_incomp = cp.mean(div_u ** 2)

    # --- 4. Navier–Stokes residual
    err_ns = 0.0
    for i in range(n_velocity):
        ui_poly = velocity_polys[i]

        # time derivative (last variable = t)
        dui_dt = ui_poly.differentiate(-1).evaluate(X_gpu)

        # convection term: (u · ∇)u_i
        conv_terms = [U_pred[:, j] * ui_poly.differentiate(j).evaluate(X_gpu) for j in range(n_velocity)]
        conv = cp.sum(cp.stack(conv_terms, axis=0), axis=0)

        # viscous term: ν ∇²u_i
        visc_terms = [ui_poly.differentiate(j).differentiate(j).evaluate(X_gpu)
                      for j in range(X_gpu.shape[1] - 1)]  # exclude time
        visc = nu * cp.sum(cp.stack(visc_terms, axis=0), axis=0)

        # pressure gradient
        dp_dx = pressure_poly.differentiate(i).evaluate(X_gpu) if pressure_poly else 0

        # NS residual
        ns_res = dui_dt + conv - visc + dp_dx - g[i]
        err_ns += cp.mean(ns_res ** 2)

    # --- Total error
    total_error = float(err_data + err_incomp + err_ns)

    if return_components:
        return {
            "total": total_error,
            "data": float(err_data),
            "incompressibility": float(err_incomp),
            "navier_stokes": float(err_ns),
        }
    else:
        return total_error


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
    laplace_s, laplace_k, pressure_poly, i, viscosity
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
        cp.asarray(pressure_poly.differentiate(i).evaluate(X))
        if pressure_poly else cp.zeros_like(phi0)
    )
    return phi_row, pressure_grad


# --- Chunked evaluation helper ---

def evaluate_models_chunked(models, X_cpu, chunk_size=100_000):
    """
    Evaluate a list of PolynomialGPU models on CPU data in chunks,
    transfer chunk to GPU, evaluate, and collect results on CPU.

    Returns: NumPy array with shape (len(models), len(X_cpu))
    """
    n_samples = X_cpu.shape[0]
    n_models = len(models)
    results = np.empty((n_models, n_samples), dtype=np.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk_gpu = cp.asarray(X_cpu[start:end])
        try:
            for i, model in enumerate(models):
                res_gpu = model.evaluate(X_chunk_gpu)
                results[i, start:end] = cp.asnumpy(res_gpu)
                del res_gpu
        finally:
            del X_chunk_gpu
            cp._default_memory_pool.free_all_blocks()


    return results



def evaluate_models_chunked_diff(models, X_cpu, diff_idx, chunk_size=100_000):
    """
    Evaluate derivatives of models (differentiate(diff_idx)) in chunks.
    """
    n_samples = X_cpu.shape[0]
    n_models = len(models)
    results = np.empty((n_models, n_samples), dtype=np.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk_cpu = X_cpu[start:end]
        X_chunk_gpu = cp.asarray(X_chunk_cpu)

        for i, model in enumerate(models):
            diff_model = model.differentiate(diff_idx)
            res_gpu = diff_model.evaluate(X_chunk_gpu)
            results[i, start:end] = cp.asnumpy(res_gpu)

        del X_chunk_gpu, res_gpu
        cp._default_memory_pool.free_all_blocks()

    return results


# --- Delayed task for least squares evaluation ---




# --- Main class ---

class PhysicsAwareGMDH:
    def __init__(self, n_features=3, max_layer=5, top_models=4, viscosity=0.01):
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

    def _append_constraint(self, phi, u_hat, phi_block, rhs):
        phi_block = cp.stack(phi_block, axis=1)
        return cp.vstack([phi, phi_block]), cp.append(u_hat, rhs)
    
    
    def _least_squares_step(self, pressure_poly, constraints):
        delayed_tasks = []
        previous_models = self.models[-1]
        total_components = len(previous_models)

        # Prepare CPU-side evaluation of all tensors chunk-wise
        # Evaluate u_tensor_cpu and u_grads_tensor_cpu

        # Convert self.X to CPU if needed
        X_cpu = self.X_cpu
        n_samples = X_cpu.shape[0]

        # Evaluate u_tensor_cpu: shape (components, n_models, samples)
        u_tensor_cpu_list = []
        u_grads_tensor_cpu_list = []

        for comp_models in previous_models:
            u_comp = evaluate_models_chunked(comp_models, X_cpu)
            u_tensor_cpu_list.append(u_comp)

            grads_comp = []
            for model in comp_models:
                grads = []
                for d in range(X_cpu.shape[1]):
                    grad_vals = evaluate_models_chunked([model.differentiate(d)], X_cpu)[0]
                    grads.append(grad_vals)
                grads_comp.append(np.stack(grads, axis=0))  # shape: (features, samples)
            u_grads_tensor_cpu_list.append(np.stack(grads_comp, axis=0))  # shape: (models, features, samples)

        u_tensor_cpu = np.stack(u_tensor_cpu_list, axis=0)          # (components, models, samples)
        u_grads_tensor_cpu = np.stack(u_grads_tensor_cpu_list, axis=0)  # (components, models, features, samples)

        u_dt_cpu_list = []
        for comp_models in previous_models:
            dt_comp = []
            for model in comp_models:
                # differentiate wrt the last variable (time)
                dt_vals = evaluate_models_chunked([model.differentiate(X_cpu.shape[1] - 1)], X_cpu)[0]
                dt_comp.append(dt_vals)
            u_dt_cpu_list.append(np.stack(dt_comp, axis=0))   # (models, samples)

        u_dt_cpu = np.stack(u_dt_cpu_list, axis=0)  # (components, models, samples)
        
        u_laplace_cpu_list = []
        for comp_models in previous_models:
            laplace_comp = []
            for model in comp_models:
                laplace_terms = []
                for d in range(X_cpu.shape[1] - 1):  # exclude time dimension
                    second_derivative = model.differentiate(d).differentiate(d)
                    vals = evaluate_models_chunked([second_derivative], X_cpu)[0]
                    laplace_terms.append(vals)
                laplace_comp.append(np.sum(np.stack(laplace_terms, axis=0), axis=0))  # sum over space
            u_laplace_cpu_list.append(np.stack(laplace_comp, axis=0))

        u_laplace_cpu = np.stack(u_laplace_cpu_list, axis=0)  # (components, models, samples)

        # Convert to CuPy arrays for the momentum constraint inputs later (or keep CPU if no GPU needed)
        X_gpu = cp.asarray(X_cpu)
        y_gpu = cp.asarray(self.y_cpu)

        model_state = {
            'X_cpu': X_cpu,
            'X_gpu': X_gpu,
            'y_cpu': self.y_cpu,
            'y_gpu': y_gpu,
            'u_tensor_cpu': u_tensor_cpu,
            'u_grads_tensor_cpu': u_grads_tensor_cpu,
            'u_tensor_gpu': cp.asarray(u_tensor_cpu),
            'u_grads_tensor_gpu': cp.asarray(u_grads_tensor_cpu),
            'u_dt_gpu': cp.asarray(u_dt_cpu),    
            'u_laplace_gpu': cp.asarray(u_laplace_cpu), 
            'pressure_poly': pressure_poly,
            'viscosity': self.viscosity,
            'constraints': constraints,
            'build_phi': self._build_phi_matrix,
            'append_constraint': self._append_constraint,
            'generate_incompressibility': generate_incompressibility_constraint_refactored,
            'generate_momentum': generate_momentum_constraint_refactored,
        }

        for comp_idx in range(total_components):
            models_in_component = previous_models[comp_idx]
            total_models = len(models_in_component)

            for idx_a in range(total_models):
                for idx_b in range(idx_a + 1, total_models):
                    for s_idx, k_idx in [(idx_a, idx_b), (idx_b, idx_a)]:
                        task = evaluate_model_pair_delayed(comp_idx, s_idx, k_idx, model_state)
                        delayed_tasks.append(task)

        results = dask.compute(*delayed_tasks, scheduler='threads')

        candidates, error_vectors, y_preds = [], [], []
        for res in results:
            if res is None:
                continue
            comp_idx, s_idx, k_idx, coef, error_vec, y_pred_i = res
            candidates.append((comp_idx, s_idx, k_idx, coef))
            error_vectors.append(error_vec)
            y_preds.append(y_pred_i)

        return candidates, error_vectors, y_preds


    

    def fit(self, X, y, pressure=None, constraints=None):
        self.X_cpu = np.asarray(X[:1000_000])  # limit to 1 million samples for memory
        self.y_cpu = np.asarray(y.T[:1000_000]).T
        X_gpu_test = cp.asarray(X[1000_000:])  # small test chunk on GPU
        y_gpu_test = cp.asarray(y.T[1000_000:])
        pressure_poly = None

        if self.current_layer == 0:
            self.models = [self._initialize_first_layer(n_components=y.shape[0])]

        while self.current_layer < self.max_layer:
            print(f"Layer {self.current_layer + 1} processing...")
            print(f"GPU free memory: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.2f} GB / "
                  f"{cp.cuda.runtime.memGetInfo()[1] / 1024**3:.2f} GB")
            print(f"RMM currently allocated: {tracked.get_allocated_bytes() / 1e6:.2f} MB")
            candidates, errors, preds = self._least_squares_step(pressure_poly, constraints or {})
            if not candidates:
                break
            
            new_models = [[] for _ in range(self.y_cpu.shape[0])]
            physics_errors = []
            pressures = []
            for idx in range(len(candidates)):
                comp_idx, s, k, coef = candidates[idx]
                model = self.models[-1][comp_idx][s].combine_with_gpu(
                    self.models[-1][comp_idx][k], *coef
                )

                velocities = [
                self.models[-1][j][s] if j != comp_idx else model
                for j in range(self.y_cpu.shape[0])
                ]

                pressure = compute_pressure_from_polynomials(
                    velocities, viscosity=self.viscosity
                ) if all(p.exponents.size > 0 for p in velocities) else None
                pressures.append(pressure)
                for j in range(y.shape[0]):
                    new_models[j].append(
                        self.models[-1][j][s] if j != comp_idx else model
                    )

                # --- physics-aware error ---
                physics_err = gmdh_error_lightweight(
                    X_gpu_test,
                    y_gpu_test,
                    velocity_polys=velocities,
                    pressure_poly=pressure,
                    nu=self.viscosity,
                    g=None
                    )
                physics_errors.append(physics_err)


                # optionally print debug (comment out if noisy)
                # print(f"Candidate {idx}: physics_err = {physics_err:.6e}")
    
            # choose based on physics-aware error instead of LS error
            physics_errors = np.array(physics_errors)
            best = np.argsort(physics_errors)[:self.top_models]

            # for plotting/debugging
            self.err_line.append(float(physics_errors[best[0]]))
            for i in range(len(new_models)):
                new_models[i] = [new_models[i][j] for j in best]
            print(f"Appendinging")
            self.models.append(new_models)
            pressure_poly = pressures[best[0]]   
            self.y_pred = [preds[int(best[0])]]
            self.current_layer += 1

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
            plt.figure(figsize=(8, 4))
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
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("Dask-CUDA cluster started:", client)

    # Define velocity functions
    def u1(x, y, t): return np.cos(x) * np.sin(y) * np.exp(-2 * 0.01 * t)
    def u2(x, y, t): return -np.sin(x) * np.cos(y) * np.exp(-2 * 0.01 * t)

    n = 2_000_000
    x, y, t = np.random.rand(n), np.random.rand(n), np.random.rand(n)
    X = np.column_stack([x, y, t])

    u1_vals = u1(x, y, t).reshape(-1)
    u2_vals = u2(x, y, t).reshape(-1)

    model = PhysicsAwareGMDH(n_features=3, max_layer=200, top_models=15, viscosity=0.01)
    model.fit(X, y=np.stack([u1_vals, u2_vals]), pressure=None,
              constraints={"incompressibility": True, "momentum": True})

    model.plot_profile(var_index=2, u_true_funcs=[u1, u2])
