import cupy as cp
import numpy as np
from sklearn.metrics import mean_squared_error
from gpu_polynomial_module import PolynomialGPU, Polynomial, decode_keys
import matplotlib.pyplot as plt

class ModelGPU:
    def __init__(self, polynomials_gpu, viscosity=0.01):
        self.polynomials = polynomials_gpu  # List of PolynomialGPU
        self.viscosity = viscosity
        self.pressure = self.compute_pressure()

    def compute_pressure(self):
        pressure_terms = []

        for i, pol in enumerate(self.polynomials):
            time_derivative = pol.differentiate(-1)

            convection_term = PolynomialGPU.from_arrays(cp.zeros((1, pol.exponents.shape[1]), dtype=cp.uint8),
                                                        cp.zeros(1, dtype=cp.float64))
            for j, uj in enumerate(self.polynomials):
                convection_term = convection_term.combine_with_gpu(uj.differentiate(j))

            viscous_term = PolynomialGPU.from_arrays(cp.zeros((1, pol.exponents.shape[1]), dtype=cp.uint8),
                                                     cp.zeros(1, dtype=cp.float64))
            for j in range(len(self.polynomials)):
                viscous_term = viscous_term.combine_with_gpu(self.polynomials[i].differentiate(j).differentiate(j))

            total = time_derivative
            total = total.combine_with_gpu(convection_term).prune()
            total = total.combine_with_gpu(viscous_term).prune()

            integrated = total.integrate(i)
            pressure_terms.append(integrated)

        pressure_exp = pressure_terms[0]
        for pt in pressure_terms[1:]:
            pressure_exp = pressure_exp.combine_with_gpu(pt)
        return pressure_exp.prune()

class GMDHGPU:
    def __init__(self, n_features=3, max_layer=15, top_models=7):
        self.max_layer = max_layer
        self.top_models = top_models
        self.current_layer = 0
        self.models = [[PolynomialGPU.from_dict({tuple(1 if j == i else 0 for j in range(n_features)): 1})
                        for i in range(n_features)]]
        self.err_line = []

    def _train_models(self, X, y, X_test, y_test):
        best_models, errors, y_preds = [], [], []
        n_features = X.shape[1]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                X_poly = np.column_stack([
                    np.ones(X.shape[0]), X[:, i], X[:, j], X[:, i] * X[:, j]
                ])
                coef = np.linalg.lstsq(X_poly, y, rcond=None)[0]
                combined_gpu = self.models[-1][i].combine_with_gpu(self.models[-1][j], *coef)
                y_pred = cp.asnumpy(combined_gpu.evaluate(cp.asarray(X_test)))
                error = mean_squared_error(y_test, y_pred)
                best_models.append(combined_gpu)
                errors.append(error)
                y_preds.append(y_pred)

        return best_models, y_preds, errors

    def fit(self, X, y, X_test, y_test):
        X = X.get() if isinstance(X, cp.ndarray) else np.asarray(X)
        y = y.get() if isinstance(y, cp.ndarray) else np.asarray(y)
        X_test = X_test.get() if isinstance(X_test, cp.ndarray) else np.asarray(X_test)
        y_test = y_test.get() if isinstance(y_test, cp.ndarray) else np.asarray(y_test)

        while self.current_layer < self.max_layer:
            X_current = np.column_stack([cp.asnumpy(m.evaluate(cp.asarray(X))) for m in self.models[-1]])
            models, y_preds, errors = self._train_models(X_current, y, X_test, y_test)
            if not models:
                break
            sorted_indices = np.argsort(errors)[:self.top_models]
            self.models.append([models[i] for i in sorted_indices])
            self.err_line.append(errors[sorted_indices[0]])
            self.y_pred = y_preds[sorted_indices[0]]
            self.current_layer += 1

if __name__ == "__main__":
    # Synthetic test data
    def u(x, y, t): return cp.cos(x) * cp.sin(y) * cp.exp(-2 * 0.01 * t)
    def v(x, y, t): return -cp.cos(y) * cp.sin(x) * cp.exp(-2 * 0.01 * t)
    def p(x,y,t): return 0.25*(np.cos(2*x)+ np.cos(2*y)) * np.exp(-4*0.01*t)

    # Step 1: generate input points
    n_samples = 1000
    x = cp.random.rand(3*n_samples)
    y = cp.random.rand(3*n_samples)
    t = cp.random.rand(3*n_samples)

    X = cp.column_stack([x[:2*n_samples], y[:2*n_samples], t[:2*n_samples]])
    X_test = cp.column_stack([x[2*n_samples:], y[2*n_samples:], t[2*n_samples:]])
    # Step 2: compute velocity values (label)
    u_vals = u(x, y, t)
    v_vals = v(x, y, t)
    #keys = cp.array([123], dtype=cp.uint64)
    #exps = decode_keys(keys, nvars=3)
    #print(exps)

    model = GMDHGPU()
    model.fit(X, u_vals[:2*n_samples], X_test, u_vals[2*n_samples:])
    # Plot error line
    plt.figure(figsize=(8, 4))
    plt.plot(model.err_line, marker='o')
    plt.title("GMDH Training Error per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()