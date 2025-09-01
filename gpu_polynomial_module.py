import cupy as cp
import cudf
from collections import defaultdict
from config_constants import MAX_EXP, PRUNE_THRESHOLD
# --- Optional: adjust to match your encoding ---

def decode_keys(keys: cp.ndarray, nvars: int, max_exp=MAX_EXP) -> cp.ndarray:
    keys = keys.astype(cp.uint64).ravel()
    n_terms = keys.shape[0]
    exponents = cp.zeros((n_terms, nvars), dtype=cp.uint8)

    for i in range(nvars - 1, -1, -1):
        mod_res = (keys % max_exp).astype(cp.uint8)
        keys = keys // max_exp
        #  Explicit copy of values, avoids .reshape() issues
        exponents[:, i] = cp.asarray(mod_res)
    
    return exponents


class Polynomial:
    def __init__(self, expform):
        self.expform = expform

class PolynomialGPU:
    def __init__(self, exponents: cp.ndarray, coeffs: cp.ndarray):
        self.exponents = exponents  # shape (n_terms, n_vars)
        self.coeffs = coeffs        # shape (n_terms,)

    @classmethod
    def from_polynomial(cls, poly):
        exp_list, coeff_list = [], []
        for exp, coeff in poly.expform.items():
            exp_list.append(exp)
            coeff_list.append(coeff)
        exponents = cp.array(exp_list, dtype=cp.uint8)
        coeffs = cp.array(coeff_list, dtype=cp.float64)
        return cls(exponents, coeffs)

    @classmethod
    def from_dict(cls, expform):
        exp_list, coeff_list = [], []
        for exp, coeff in expform.items():
            exp_list.append(exp)
            coeff_list.append(coeff)
        exponents = cp.array(exp_list, dtype=cp.uint8)
        coeffs = cp.array(coeff_list, dtype=cp.float64)
        return cls(exponents, coeffs)

    @classmethod
    def from_arrays(cls, exponents_np, coeffs_np):
        return cls(cp.asarray(exponents_np, dtype=cp.uint8), cp.asarray(coeffs_np, dtype=cp.float64))

    def to_polynomial(self):
        exp_np = cp.asnumpy(self.exponents)
        coeff_np = cp.asnumpy(self.coeffs)
        expform = {tuple(exp): coeff for exp, coeff in zip(exp_np, coeff_np)}
        return Polynomial(expform)

    def differentiate(self, var_index):
        mask = self.exponents[:, var_index] > 0
        if not cp.any(mask):
            return PolynomialGPU(cp.zeros((0, self.exponents.shape[1]), dtype=cp.uint8),
                                 cp.zeros((0,), dtype=cp.float64))
        new_exponents = self.exponents[mask].copy()
        new_exponents[:, var_index] -= 1
        new_coeffs = self.coeffs[mask] * self.exponents[mask, var_index]
        return PolynomialGPU(new_exponents, new_coeffs)

    def integrate(self, var_index):
        new_exponents = self.exponents.copy()
        new_exponents[:, var_index] += 1
        new_coeffs = self.coeffs / new_exponents[:, var_index].astype(cp.float64)
        return PolynomialGPU(new_exponents, new_coeffs)

    def evaluate(self, points):
        n_terms = self.exponents.shape[0]
        result = cp.zeros(points.shape[0], dtype=cp.float64)
        for i in range(n_terms):
            powers = points ** self.exponents[i]
            result += cp.prod(powers, axis=1) * self.coeffs[i]
        return result

    def prune(self, threshold=1e-18):
        mask = cp.abs(self.coeffs) > threshold
        return PolynomialGPU(self.exponents[mask], self.coeffs[mask])

    def combine_with_gpu(self, other, c0=0.0, c1=1.0, c2=1.0, c3=1.0):
        from cuda_poly_multiply import launch_cuda_kernel
        import cudf
        if len(other.coeffs)==0: return PolynomialGPU(cp.zeros((0, self.exponents.shape[1]), dtype=cp.uint8),
                                 cp.zeros((0,), dtype=cp.float64))
        exponents_list = []
        coeffs_list = []

        # c1 * self
        if abs(c1) > 1e-22:
            exponents_list.append(self.exponents)
            coeffs_list.append(self.coeffs * c1)

        # c2 * other
        if abs(c2) > 1e-22:
            exponents_list.append(other.exponents)
            coeffs_list.append(other.coeffs * c2)

        # c3 * self * other (GPU multiply)
        if abs(c3) > 1e-22:
            result = launch_cuda_kernel(self.exponents, other.exponents,
                                        self.coeffs, other.coeffs)
            keys = result[:, 0].astype(cp.uint64)
            coeffs = c3 * result[:, 1]
            # Decode keys to exponents later after combining with others
            # We'll store keys separately for now
        else:
            keys = cp.zeros((0,), dtype=cp.uint64)
            coeffs = cp.zeros((0,), dtype=cp.float64)

        # c0 (constant term)
        if abs(c0) > 1e-22:
            zero_exp = cp.zeros((1, self.exponents.shape[1]), dtype=cp.uint8)
            zero_coeff = cp.array([c0], dtype=cp.float64)
            exponents_list.append(zero_exp)
            coeffs_list.append(zero_coeff)

        # Combine self, other, and constant terms first
        if exponents_list:
            exponents_all = cp.concatenate(exponents_list, axis=0)
            coeffs_all = cp.concatenate(coeffs_list, axis=0)
        else:
            exponents_all = cp.zeros((0, self.exponents.shape[1]), dtype=cp.uint8)
            coeffs_all = cp.zeros((0,), dtype=cp.float64)

        # Now combine with multiplication result
        if keys.size > 0:
            # Decode multiplication keys
            decoded_exp = decode_keys(keys, nvars=self.exponents.shape[1])
            # Concatenate with previous terms
            exponents_all = cp.concatenate([exponents_all, decoded_exp], axis=0)
            coeffs_all = cp.concatenate([cp.asarray(coeffs_all), cp.asarray(coeffs)], axis=0)

        if exponents_all.shape[0] == 0:
            return PolynomialGPU(exponents_all, coeffs_all)  # empty poly

        # Encode exponents into keys for grouping
        keys_combined = cp.zeros(exponents_all.shape[0], dtype=cp.uint64)
        base = 1
        for i in reversed(range(exponents_all.shape[1])):
            keys_combined += exponents_all[:, i].astype(cp.uint64) * base
            base *= MAX_EXP

        df = cudf.DataFrame({'key': keys_combined, 'coeff': coeffs_all})
        reduced = df.groupby('key').agg({'coeff': 'sum'}).reset_index()

        decoded_exponents = decode_keys(reduced['key'].to_cupy(), nvars=self.exponents.shape[1])
        return PolynomialGPU(decoded_exponents, reduced['coeff'].to_cupy()).prune()


if __name__ == "__main__":
    poly1 = Polynomial({(1, 0): 2.0, (0, 1): 3.0})
    poly2 = Polynomial({(0, 1): 4.0, (1, 0): 5.0, (1, 1): 1.0})
    P1 = PolynomialGPU.from_polynomial(poly1)
    P2 = PolynomialGPU.from_polynomial(poly2)

    # Combine on GPU
    P_combined = P1.combine_with_gpu(P2)

    # Differentiate on GPU
    dP = P_combined.differentiate(0)

    # Convert back to symbolic Polynomial if needed
    sym_poly = dP.to_polynomial()