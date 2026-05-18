"""
gpu_polynomial_module.py  (rev 3)
==================================
Changes vs rev 2:
  _ensure_cpu_cache — renamed from private to public convention so
                    batch_cpu_eval.py can call it directly.
  evaluate_with_table — new method: evaluate using a precomputed X_pow
                    table supplied by the caller.  Zero np.power calls
                    inside the inner loop.
  evaluate_cpu    — unchanged API; internally calls evaluate_with_table
                    when given a pre-built table, or builds its own
                    single-polynomial table as a fallback.
"""

import cupy as cp
import numpy as np

PRUNE_THRESHOLD = 1e-12


class PolynomialGPU:
    def __init__(self, exponents, coeffs):
        self.exponents = cp.asarray(exponents, dtype=np.uint16)
        self.coeffs    = cp.asarray(coeffs,    dtype=np.float64)
        self.nvars     = int(self.exponents.shape[1]) if self.exponents.ndim > 1 else 0
        # CPU cache — populated by sync_to_cpu() or evaluate_cpu()
        self._cpu_exps   = None
        self._cpu_coeffs = None
        self._cpu_exp_id = None   # id of GPU exponents array when cache was built
        self._cpu_cof_id = None   # id of GPU coeffs array when cache was built

    @classmethod
    def from_dict(cls, term_dict, nvars=3):
        if not term_dict:
            return cls(
                cp.zeros((0, nvars), dtype=cp.uint16),
                cp.zeros((0,),       dtype=cp.float64)
            )
        exps   = list(term_dict.keys())
        coeffs = list(term_dict.values())
        return cls(
            cp.array(exps,   dtype=cp.uint16),
            cp.array(coeffs, dtype=cp.float64)
        )

    # ------------------------------------------------------------------
    # Symbolic operations
    # ------------------------------------------------------------------

    def differentiate(self, var_idx):
        # FIX: return a NEW zero polynomial, not self, to avoid aliasing
        if self.exponents.shape[0] == 0:
            return PolynomialGPU(
                cp.zeros((0, self.nvars), dtype=cp.uint16),
                cp.zeros((0,),           dtype=cp.float64)
            )
        powers = self.exponents[:, var_idx]
        mask   = powers > 0
        if not cp.any(mask):
            return PolynomialGPU(
                cp.zeros((0, self.nvars), dtype=cp.uint16),
                cp.zeros((0,),           dtype=cp.float64)
            )
        new_coeffs = self.coeffs[mask] * powers[mask].astype(cp.float64)
        new_exps   = self.exponents[mask].copy()
        new_exps[:, var_idx] -= 1
        valid = cp.abs(new_coeffs) > PRUNE_THRESHOLD
        return PolynomialGPU(new_exps[valid], new_coeffs[valid])

    def integrate(self, var_idx):
        """Analytical integration w.r.t. var_idx: x^n → x^(n+1)/(n+1)."""
        if self.exponents.shape[0] == 0:
            return PolynomialGPU(
                cp.zeros((0, self.nvars), dtype=cp.uint16),
                cp.zeros((0,),           dtype=cp.float64)
            )
        new_exps   = self.exponents.copy()
        new_exps[:, var_idx] += 1
        new_coeffs = self.coeffs / new_exps[:, var_idx].astype(cp.float64)
        return PolynomialGPU(new_exps, new_coeffs)

    def prune(self, threshold=PRUNE_THRESHOLD):
        if self.coeffs.shape[0] == 0:
            return self
        mask = cp.abs(self.coeffs) > threshold
        return PolynomialGPU(self.exponents[mask], self.coeffs[mask])

    # ------------------------------------------------------------------
    # Evaluation — GPU path
    # ------------------------------------------------------------------

    def evaluate(self, X):
        """
        Evaluate polynomial at all rows of X (GPU array, shape [N, nvars]).

        FIX vs rev 1: removed the inner chunk loop — one vectorised pass
        over all N samples is faster on GPU.  The term loop (over n_terms
        in blocks of term_chunk) is kept only to bound peak VRAM when
        n_terms is large (deep layers with many monomials).

        Returns cp.ndarray of shape [N].
        """
        n_samples = X.shape[0]
        n_terms   = self.exponents.shape[0]

        if n_terms == 0:
            return cp.zeros(n_samples, dtype=cp.float64)

        # For small n_terms do it in one shot
        term_chunk = 2048
        if n_terms <= term_chunk:
            return self._evaluate_block(X, 0, n_terms)

        # For large n_terms accumulate in term-blocks to bound VRAM
        result = cp.zeros(n_samples, dtype=cp.float64)
        for t_start in range(0, n_terms, term_chunk):
            t_end = min(t_start + term_chunk, n_terms)
            result += self._evaluate_block(X, t_start, t_end)
        return result

    def _evaluate_block(self, X, t_start, t_end):
        """Vandermonde block for terms t_start..t_end-1."""
        exps   = self.exponents[t_start:t_end]   # [block, nvars]
        coeffs = self.coeffs[t_start:t_end]       # [block]
        # term_matrix: [N, block]
        term_matrix = cp.ones((X.shape[0], t_end - t_start), dtype=cp.float64)
        for d in range(self.nvars):
            col = exps[:, d]                      # [block]
            if cp.any(col > 0):
                # X[:, d:d+1] broadcasts over block dimension
                term_matrix *= cp.power(X[:, d:d+1], col)
        return term_matrix @ coeffs

    # ------------------------------------------------------------------
    # Evaluation — CPU path (for pg_full pre-computation)
    # ------------------------------------------------------------------

    def sync_to_cpu(self):
        """
        Eagerly copy GPU arrays to RAM.
        Records the id of each GPU array so evaluate_cpu can detect
        if the GPU data was replaced since the last sync.
        """
        self._cpu_exps   = self.exponents.get()
        self._cpu_coeffs = self.coeffs.get()
        self._cpu_exp_id = id(self.exponents)
        self._cpu_cof_id = id(self.coeffs)

    def _ensure_cpu_cache(self):
        """Refresh CPU cache if GPU arrays have been replaced."""
        if (self._cpu_exps is None
                or id(self.exponents) != self._cpu_exp_id
                or id(self.coeffs)    != self._cpu_cof_id):
            self.sync_to_cpu()

    def evaluate_with_table(self, X_pow, term_chunk=256):
        """
        Evaluate this polynomial using a precomputed power table.

        Parameters
        ----------
        X_pow : np.ndarray float64, shape [n_vars, max_deg+1, N]
            X_pow[d, p] = X[:, d] ** p  for p in 0..max_deg.
            Built once per chunk by batch_cpu_eval.build_X_pow.
        term_chunk : int
            Max terms evaluated at once. Peak memory = N * term_chunk * 8 bytes.
            Default 256 keeps peak under ~256 MB for N=1M.

        Returns
        -------
        result : np.ndarray float64, shape [N]
        """
        self._ensure_cpu_cache()
        exps   = self._cpu_exps    # [n_terms, n_vars] uint16
        coeffs = self._cpu_coeffs  # [n_terms] float64
        n_terms = exps.shape[0]
        N       = X_pow.shape[2]

        if n_terms == 0:
            return np.zeros(N, dtype=np.float64)

        result = np.zeros(N, dtype=np.float64)
        for t0 in range(0, n_terms, term_chunk):
            t1 = min(t0 + term_chunk, n_terms)
            e = exps[t0:t1]              # [block, n_vars]
            c = coeffs[t0:t1]            # [block]
            tm = np.ones((N, t1 - t0), dtype=np.float64)
            for d in range(self.nvars):
                col = e[:, d]
                if np.any(col > 0):
                    tm *= X_pow[d][col].T
            result += tm @ c
        return result

    def evaluate_cpu(self, X_cpu, chunk_size=500_000, X_pow=None):
        """
        CPU evaluation.  Two calling modes:

        1. Standalone (X_pow=None):
           Builds its own single-polynomial power table per chunk.
           Use when evaluating one polynomial in isolation.

        2. Batch mode (X_pow supplied):
           Uses the caller-provided table — zero np.power calls here.
           Use via batch_cpu_eval.evaluate_pg_cpu for maximum efficiency.

        Returns np.ndarray of shape [N].
        """
        self._ensure_cpu_cache()

        if X_pow is not None:
            # Caller already built the table for this chunk
            return self.evaluate_with_table(X_pow)

        # Standalone path — build a minimal table for this polynomial only
        exps   = self._cpu_exps
        coeffs = self._cpu_coeffs
        n_terms   = exps.shape[0]
        n_samples = X_cpu.shape[0]

        if n_terms == 0:
            return np.zeros(n_samples, dtype=np.float64)

        max_deg = int(exps.max()) if n_terms > 0 else 1
        result  = np.zeros(n_samples, dtype=np.float64)

        for start in range(0, n_samples, chunk_size):
            end     = min(start + chunk_size, n_samples)
            # Import here to avoid circular import; module is lightweight
            from batch_cpu_eval import build_X_pow
            table           = build_X_pow(X_cpu[start:end], max_deg)
            result[start:end] = self.evaluate_with_table(table)

        return result