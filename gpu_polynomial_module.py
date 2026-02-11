import cupy as cp
import cudf
from collections import defaultdict
from config_constants import MAX_EXP, PRUNE_THRESHOLD
import math
import numpy as np
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

def encode_keys(exponents: cp.ndarray, max_exp=MAX_EXP) -> cp.ndarray:
    """
    Encode exponent matrix into unique integer keys.

    Parameters
    ----------
    exponents : cp.ndarray
        Shape (n_terms, n_vars), dtype=uint8.
    max_exp : int
        Maximum exponent + 1 (the radix base).

    Returns
    -------
    cp.ndarray
        Shape (n_terms,), dtype=uint64 with encoded keys.
    """
    n_terms, nvars = exponents.shape
    keys = cp.zeros(n_terms, dtype=cp.uint64)

    base = 1
    for i in reversed(range(nvars)):
        keys += exponents[:, i].astype(cp.uint64) * base
        base *= max_exp

    return keys

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

        if len(self.coeffs) == 0 or len(other.coeffs) == 0:
            return PolynomialGPU(
            cp.zeros((0, self.exponents.shape[1]), dtype=cp.uint8),
            cp.zeros((0,), dtype=cp.float64)
        )

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
            result = launch_cuda_kernel(
                self.exponents, other.exponents,
                self.coeffs, other.coeffs
            )
            keys = cp.asarray(result[:, 0], dtype=cp.uint64)
            coeffs = cp.asarray(c3 * result[:, 1], dtype=cp.float64)
        else:
            keys = cp.zeros((0,), dtype=cp.uint64)
            coeffs = cp.zeros((0,), dtype=cp.float64)

        # c0 (constant term)
        if abs(c0) > 1e-22:
            zero_exp = cp.zeros((1, self.exponents.shape[1]), dtype=cp.uint8)
            zero_coeff = cp.array([c0], dtype=cp.float64)
            exponents_list.append(zero_exp)
            coeffs_list.append(zero_coeff)

        # Combine self, other, constant terms
        if exponents_list:
            exponents_all = cp.concatenate([cp.asarray(e) for e in exponents_list], axis=0)
            coeffs_all = cp.concatenate([cp.asarray(c) for c in coeffs_list], axis=0)
        else:
            exponents_all = cp.zeros((0, self.exponents.shape[1]), dtype=cp.uint8)
            coeffs_all = cp.zeros((0,), dtype=cp.float64)

        # Now merge with multiplication result
        if keys.size > 0:
            decoded_exp = decode_keys(keys, nvars=self.exponents.shape[1])
            exponents_all = cp.concatenate([cp.asarray(exponents_all), cp.asarray(decoded_exp)], axis=0)
            coeffs_all = cp.concatenate([cp.asarray(coeffs_all), cp.asarray(coeffs)], axis=0)

        if exponents_all.shape[0] == 0:
            return PolynomialGPU(exponents_all, coeffs_all)  # empty poly

        # Encode exponents into keys for grouping
        keys_combined = cp.zeros(exponents_all.shape[0], dtype=cp.uint64)
        base = 1
        for i in reversed(range(exponents_all.shape[1])):
            keys_combined += exponents_all[:, i].astype(cp.uint64) * base
            base *= MAX_EXP

        df = cudf.DataFrame({'key': cp.asarray(keys_combined), 'coeff': cp.asarray(coeffs_all)})
        reduced = df.groupby('key').agg({'coeff': 'sum'}).reset_index()

        decoded_exponents = decode_keys(reduced['key'].to_cupy(), nvars=self.exponents.shape[1])
        return PolynomialGPU(decoded_exponents, reduced['coeff'].to_cupy()).prune()

    def _evaluate_monomials_gpu(self, X):
        """
        Evaluate each monomial term on X (GPU) and return phi: shape (N, n_terms).
        - X: cp.ndarray shape (N, n_vars)
        - returns: phi (N, n_terms) cp.float64
        """
        # fast path: zero-term polynomial
        if self.coeffs.size == 0:
            return cp.zeros((X.shape[0], 0), dtype=cp.float64)

        n_terms = int(self.exponents.shape[0])
        N = X.shape[0]
        phi = cp.ones((N, n_terms), dtype=cp.float64)

        # multiply contributions variable-by-variable
        # vectorized over samples, loop over variables (n_vars typically small)
        for var_idx in range(self.exponents.shape[1]):
            exps = self.exponents[:, var_idx].astype(cp.int32)  # (n_terms,)
            # if all zero, skip
            if not int(cp.any(exps)):
                continue
            # compute X[:,var_idx] ** exps for all samples and terms in one go:
            # X[:,var_idx][:,None] ** exps[None,:]
            phi *= (X[:, var_idx:var_idx+1] ** exps[None, :])
        return phi

    def evaluate_monomials(self, X):
        """
        Evaluate each monomial term (without multiplying by coefficients).
        Returns CuPy array of shape (N, n_terms).
        """
        if isinstance(X, cp.ndarray):
            return self._evaluate_monomials_gpu(X)
        else:
            X_gpu = cp.asarray(X)
            phi = self._evaluate_monomials_gpu(X_gpu)
            return cp.asnumpy(phi)
    
    def compute_correlations(self, X, y, chunk_size=None):
        """
        Compute correlation between each monomial value and output y.
        Returns CuPy array of shape (n_terms,).
        """
        y_gpu = cp.asarray(y).ravel()
        N = y_gpu.size
        if chunk_size is None:
            chunk_size = min(32768, N)

        corrs = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            phi = self._evaluate_monomials_gpu(X[start:end])
            # center
            phi -= phi.mean(axis=0, keepdims=True)
            y_c = y_gpu[start:end] - y_gpu[start:end].mean()
            # correlation numerator
            num = cp.sum(phi * y_c[:, None], axis=0)
            # denominator
            denom = cp.sqrt(cp.sum(phi**2, axis=0) * cp.sum(y_c**2))
            corr_chunk = cp.nan_to_num(num / denom)
            corrs.append(corr_chunk)
        corrs = cp.mean(cp.stack(corrs), axis=0)
        return corrs

    def remove_terms_mask(self, keep_mask):
        """Return a new PolynomialGPU keeping only indices where keep_mask==True."""
        keep_mask = cp.asarray(keep_mask, dtype=bool)
        return PolynomialGPU(self.exponents[keep_mask], self.coeffs[keep_mask])

    def substitute_term_by_mean(self, idx, phi_j_mean):
        """
        Replace term idx by adding coeff[idx] * mean(phi_j) into constant term.
        Equivalent to removing oscillatory part of the term while keeping its mean effect.
        This modifies coefficients in place and returns a new PolynomialGPU.
        """
        coeffs = self.coeffs.copy()
        exps = self.exponents.copy()
        c = coeffs[idx]
        # find constant term (exponents all zero) or create one
        zero_mask = cp.all(exps == 0, axis=1)
        if cp.any(zero_mask):
            const_idx = int(cp.where(zero_mask)[0][0])
            coeffs[const_idx] = coeffs[const_idx] + c * float(phi_j_mean)
        else:
            # append constant term
            zero_exp = cp.zeros((1, exps.shape[1]), dtype=exps.dtype)
            exps = cp.concatenate([exps, zero_exp], axis=0)
            coeffs = cp.concatenate([coeffs, cp.array([c * float(phi_j_mean)], dtype=coeffs.dtype)])
        # zero the original coefficient
        coeffs = coeffs.copy()
        coeffs[idx] = 0.0
        return PolynomialGPU(exps, coeffs)

    


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