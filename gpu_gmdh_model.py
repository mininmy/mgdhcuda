import rmm
from rmm import mr
from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp
import matplotlib.pyplot as plt

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

print(f"GPU free memory: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.2f} GB / "
      f"{cp.cuda.runtime.memGetInfo()[1] / 1024**3:.2f} GB")
print(f"RMM currently allocated: {tracked.get_allocated_bytes() / 1e6:.2f} MB")

from gpu_polynomial_module import PolynomialGPU
from cuda_least_squares import least_squares_gpu

# --- Helper functions ---

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


def solve_least_squares(phi, y):
    xtx = phi.T @ phi
    xty = phi.T @ y
    return cp.linalg.solve(xtx, xty)


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

        # Placeholders for training data and derivatives
        self.X = None
        self.y = None
        self.u_tensor = None
        self.u_grads_tensor = None
        self.u_dt = None
        self.u_laplace = None
        self.error_tensor = None

    def _initialize_first_layer(self, n_components):
        return [[
            PolynomialGPU.from_dict({tuple(1 if j == i else 0 for j in range(self.n_features)): 1})
            for i in range(self.n_features)
        ] for _ in range(n_components)]

    def _build_phi_matrix(self, u_s, u_k, batch_size=100_000):
        n_rows = u_s.shape[0]
        phi_parts = []

        for start_idx in range(0, n_rows, batch_size):
            end_idx = min(start_idx + batch_size, n_rows)
            u_s_batch = u_s[start_idx:end_idx]
            u_k_batch = u_k[start_idx:end_idx]

            phi_chunk = cp.stack([
                cp.ones_like(u_s_batch),
                u_s_batch,
                u_k_batch,
                u_s_batch * u_k_batch
            ], axis=1)

            phi_parts.append(phi_chunk)

        try:
            phi = cp.concatenate(phi_parts, axis=0)
        except cp.cuda.memory.OutOfMemoryError:
            cp._default_memory_pool.free_all_blocks()
            phi = cp.concatenate(phi_parts, axis=0)

        return phi

    def _append_constraint(self, phi, u_hat, phi_block, rhs):
        phi_block = cp.stack(phi_block, axis=1)
        return cp.vstack([phi, phi_block]), cp.append(u_hat, rhs)

    def _least_squares_step(self, pressure_poly, constraints):
        candidates, error_vectors, y_preds = [], [], []
        previous_models = self.models[-1]
        total_components = len(previous_models)

        for comp_idx in range(total_components):
            models_in_component = previous_models[comp_idx]
            total_models = len(models_in_component)

            for idx_a in range(total_models):
                for idx_b in range(idx_a + 1, total_models):
                    print("Layer= ", self.current_layer)
                    print(f"Processing component {comp_idx}, models {idx_a} and {idx_b}")
                    allocated = rmm.mr.get_current_device_resource().get_allocated_bytes()
                    print(f"RMM allocated: {allocated / 1e6:.2f} MB")

                    for s_idx, k_idx in [(idx_a, idx_b), (idx_b, idx_a)]:
                        u_s = self.u_tensor[comp_idx, s_idx]
                        u_k = self.u_tensor[comp_idx, k_idx]
                        phi = self._build_phi_matrix(u_s, u_k)
                        u_hat = self.y[comp_idx]

                        grad_s = self.u_grads_tensor[comp_idx, s_idx]
                        grad_k = self.u_grads_tensor[comp_idx, k_idx]

                        if constraints.get('incompressibility', False):
                            inc_row, inc_rhs = generate_incompressibility_constraint_refactored(
                                u_s, u_k, grad_s, grad_k
                            )
                            phi, u_hat = self._append_constraint(phi, u_hat, inc_row, inc_rhs)

                        if constraints.get('momentum', False):
                            mom_row, mom_rhs = generate_momentum_constraint_refactored(
                                self.X, u_s, u_k, grad_s, grad_k,
                                self.u_tensor[comp_idx], self.u_grads_tensor[comp_idx],
                                self.u_dt[comp_idx][s_idx], self.u_dt[comp_idx][k_idx],
                                self.u_laplace[comp_idx][s_idx], self.u_laplace[comp_idx][k_idx],
                                pressure_poly, comp_idx, self.viscosity
                            )
                            phi, u_hat = self._append_constraint(phi, u_hat, mom_row, mom_rhs)

                        try:
                            coef = cp.asarray(least_squares_gpu(phi, u_hat))
                        except cp.linalg.LinAlgError:
                            continue

                        y_pred_i = phi @ coef
                        error_vec = []
                        sample_size = self.X.shape[0]

                        # Main component MSE
                        main_diff = y_pred_i[:sample_size] - self.y[comp_idx]
                        mse_main = cp.sum(cp.nan_to_num(main_diff) ** 2) / main_diff.size
                        error_vec.append(mse_main)

                        # Other components MSE
                        for j in range(self.y.shape[0]):
                            if j == comp_idx:
                                continue
                            residual_diff = self.u_tensor[j, s_idx] - self.y[j]
                            mse_other = cp.sum(cp.nan_to_num(residual_diff) ** 2) / residual_diff.size
                            error_vec.append(mse_other)

                        # Constraint error
                        constraint_residual = y_pred_i[sample_size:] - u_hat[sample_size:]
                        mse_constraint = 0.5 * cp.sum(cp.nan_to_num(constraint_residual) ** 2) / constraint_residual.size
                        error_vec.append(mse_constraint)

                        error_vector = cp.array(error_vec)
                        error_vectors.append(error_vector)
                        candidates.append((comp_idx, s_idx, k_idx, coef))
                        y_preds.append(y_pred_i)

                        # Clean GPU memory immediately
                        del phi, coef, y_pred_i
                        import gc
                        gc.collect()
                        cp._default_memory_pool.free_all_blocks()
                        allocated = rmm.mr.get_current_device_resource().get_allocated_bytes()
                        print(f"RMM after cleanup allocated: {allocated / 1e6:.2f} MB")

        return candidates, error_vectors, y_preds

    def fit(self, X, y, pressure=None, constraints=None):
        self.X = cp.asarray(X)
        self.y = cp.asarray(y)
        pressure_poly = None

        if self.current_layer == 0:
            self.models = [self._initialize_first_layer(n_components=y.shape[0])]

        while self.current_layer < self.max_layer:
            self.u_tensor = cp.stack([
                cp.stack([cp.asarray(m.evaluate(self.X)) for m in comp_models])
                for comp_models in self.models[-1]
            ])

            self.u_grads_tensor = cp.stack([
                cp.stack([
                    cp.stack([cp.asarray(m.differentiate(m_idx).evaluate(self.X)) for m_idx in range(self.X.shape[1])])
                    for m in comp_models
                ]) for comp_models in self.models[-1]
            ])

            self.u_dt = [
                [cp.asarray(m.differentiate(self.X.shape[1] - 1).evaluate(self.X)) for m in comp_models]
                for comp_models in self.models[-1]
            ]

            self.u_laplace = [
                [[cp.asarray(m.differentiate(m_idx).differentiate(m_idx).evaluate(self.X)) for m_idx in range(self.X.shape[1] - 1)]
                 for m in comp_models]
                for comp_models in self.models[-1]
            ]

            candidates, errors, preds = self._least_squares_step(pressure_poly, constraints or {})
            if not candidates:
                break

            self.error_tensor = cp.stack(errors)
            best = cp.argsort(cp.nansum(self.error_tensor, axis=1))[:self.top_models]

            new_models = [[] for _ in range(self.y.shape[0])]
            for idx in best:
                comp_idx, s, k, coef = candidates[int(idx)]

                model = self.models[-1][comp_idx][s].combine_with_gpu(
                    self.models[-1][comp_idx][k], *cp.asnumpy(coef)
                )
                for j in range(self.y.shape[0]):
                    new_models[j].append(self.models[-1][j][s] if j != comp_idx else model)

            self.models.append(new_models)
            self.err_line.append(cp.min(cp.nansum(self.error_tensor, axis=1)))
            self.y_pred = [preds[int(best[0])]]
            self.current_layer += 1

        return self

    def plot_profile(self, var_index=0, fixed_values=None, n_points=100, u_true_funcs=None):
        n_vars = self.X.shape[1]
        if fixed_values is None:
            fixed_values = cp.mean(self.X, axis=0).tolist()

        x_range = cp.linspace(cp.min(self.X[:, var_index]), cp.max(self.X[:, var_index]), n_points)
        X_test = cp.tile(cp.array(fixed_values), (n_points, 1))
        X_test[:, var_index] = x_range

        top_model = self.models[-1]
        preds = []
        for comp_models in top_model:
            best_model = comp_models[0]
            u_pred = cp.asarray(best_model.evaluate(X_test))
            preds.append(u_pred)

        for i, (pred, true_label) in enumerate(zip(preds, ["$u_1$", "$u_2$"])):
            plt.figure(figsize=(8, 4))
            plt.plot(cp.asnumpy(x_range), cp.asnumpy(pred), label=f"Predicted {true_label}", linestyle="--")

            if u_true_funcs is not None:
                args = [X_test[:, j] for j in range(n_vars)]
                u_actual = u_true_funcs[i](*args)
                plt.plot(cp.asnumpy(x_range), cp.asnumpy(u_actual), label=f"Actual {true_label}", linestyle="-")

            plt.title(f"{true_label} vs $x_{{{var_index+1}}}$ (fixed others)")
            plt.xlabel(f"$x_{{{var_index+1}}}$")
            plt.ylabel(true_label)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()


# --- Usage example ---

if __name__ == "__main__":
    # Define velocity functions
    def u1(x, y, t): return cp.cos(x) * cp.sin(y) * cp.exp(-2 * 0.01 * t)
    def u2(x, y, t): return -cp.sin(x) * cp.cos(y) * cp.exp(-2 * 0.01 * t)

    n = 1_000_000
    x, y, t = cp.random.rand(n), cp.random.rand(n), cp.random.rand(n)
    X = cp.column_stack([x, y, t])

    u1_vals = cp.asarray(u1(x, y, t)).reshape(-1)
    u2_vals = cp.asarray(u2(x, y, t)).reshape(-1)
    assert u1_vals.shape == u2_vals.shape, f"Shape mismatch: {u1_vals.shape} vs {u2_vals.shape}"

    model = PhysicsAwareGMDH(n_features=3, max_layer=4, top_models=50)
    model.fit(X, y=cp.stack([u1_vals, u2_vals]), pressure=None, constraints={"incompressibility": False, "momentum": False})

    plt.plot(cp.asnumpy(cp.array(model.err_line)), marker='o')
    plt.title("Training Error per Layer")
    plt.xlabel("Layer")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()

    # Test plots
    model.plot_profile(var_index=0, u_true_funcs=[u1, u2])
    model.plot_profile(var_index=1, u_true_funcs=[u1, u2])
    model.plot_profile(var_index=2, u_true_funcs=[u1, u2])
