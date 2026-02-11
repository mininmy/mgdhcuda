import numpy as np
from numba import cuda, float64
import cupy as cp
from math import isnan   

@cuda.jit
def compute_weighted_XTX_XTy(X, y, weights, XTX, XTy):
    """
    CUDA kernel to compute weighted normal equation components:
        XTX = X^T W X
        XTy = X^T W y
    """
    row = cuda.grid(1)
    n_rows, n_cols = X.shape
    if row < n_rows:
        w = weights[row]
        if w == 0.0 or isnan(w):  # skip invalid rows
            return
        for i in range(n_cols):
            xi = X[row, i]
            for j in range(n_cols):
                cuda.atomic.add(XTX, (i, j), w * xi * X[row, j])
            cuda.atomic.add(XTy, i, w * xi * y[row])


def weighted_least_squares_with_errors_gpu(X, y, weights, group_bounds):
    """
    Solve weighted least squares on GPU and compute per-group weighted squared errors.

    Parameters
    ----------
    X : cp.ndarray, shape (n_rows, n_cols)
    y : cp.ndarray, shape (n_rows,)
    weights : cp.ndarray, shape (n_rows,)
        Row-wise scalar weights (typically from constraint scaling)
    group_bounds : dict[str, tuple[int, int]]
        Slice bounds per group, e.g.:
        {
            'data': (0, n_data),
            'incomp': (n_data, n_data + n_incomp),
            'momentum': (n_data + n_incomp, n_total)
        }

    Returns
    -------
    beta : cp.ndarray, (n_cols,)
        Regression coefficients.
    group_errors : dict[str, float]
        Weighted mean squared error per group.
    """

    n_rows, n_cols = X.shape
    
    # --- 1. Compute X^T W X and X^T W y ---
    XTX = cp.zeros((n_cols, n_cols), dtype=cp.float64)
    XTy = cp.zeros(n_cols, dtype=cp.float64)
    
    threads_per_block = 256
    blocks_per_grid = (n_rows + threads_per_block - 1) // threads_per_block

    compute_weighted_XTX_XTy[blocks_per_grid, threads_per_block](X, y, weights, XTX, XTy)
    cuda.synchronize()
    
    # --- 2. Solve for beta ---
    # Add small diagonal regularization for numerical stability
    reg = 1e-12
    beta = cp.linalg.solve(XTX + reg * cp.eye(n_cols), XTy)

    # --- 3. Compute residuals and per-group weighted MSEs ---
    residuals = X @ beta - y
    group_errors = {}
    
    for key, (start, end) in group_bounds.items():
        if end <= start:
            group_errors[key] = 0.0
            continue
        r = residuals[start:end]
        w = weights[start:end]

        # Weighted MSE (normalize by sum of weights to keep scale consistent)
        mse = cp.sum((w * r) ** 2) / cp.sum(w)
        group_errors[key] = float(mse)

    return beta, group_errors

def weighted_least_squares_gpu(blocks, weights, chunk_size=100000):
    """
    Compute weighted least squares solution:
        sum_i (w_i * A_i^T A_i) beta = sum_i (w_i * A_i^T b_i)
    Each A_i, b_i are GPU/CPU arrays.
    """
    # Assume all blocks share the same number of columns
    n_cols = blocks[0][0].shape[1]
    XTX_host = np.zeros((n_cols, n_cols), dtype=np.float64)
    XTy_host = np.zeros(n_cols, dtype=np.float64)

    threads_per_block = 256
    blocks_per_grid = (chunk_size + threads_per_block - 1) // threads_per_block

    for (A_host, b_host), w in zip(blocks, weights):
        n_rows = A_host.shape[0]
        for i in range(0, n_rows, chunk_size):
            end = min(i + chunk_size, n_rows)
            A_chunk = A_host[i:end]
            b_chunk = b_host[i:end]

            d_A = cuda.to_device(A_chunk)
            d_b = cuda.to_device(b_chunk)
            d_XTX = cuda.to_device(np.zeros((n_cols, n_cols), dtype=np.float64))
            d_XTy = cuda.to_device(np.zeros(n_cols, dtype=np.float64))

            compute_weighted_XTX_XTy[blocks_per_grid, threads_per_block](d_A, d_b, d_XTX, d_XTy, w)

            XTX_host += d_XTX.copy_to_host()
            XTy_host += d_XTy.copy_to_host()

    beta = np.linalg.solve(XTX_host, XTy_host)
    return beta


@cuda.jit
def compute_XTX_XTy(X, y, XTX, XTy):
    row = cuda.grid(1)
    n_rows, n_cols = X.shape
    if row < n_rows:
        for i in range(n_cols):
            xi = X[row, i]
            for j in range(n_cols):
                cuda.atomic.add(XTX, (i, j), xi * X[row, j])
            cuda.atomic.add(XTy, i, xi * y[row])

# keep your existing imports
import numpy as np
from numba import cuda, float64

# --- your existing kernel must remain unchanged and available ---
# compute_XTX_XTy(X, y, XTX, XTy)  # already in your code

# Helper: vertically concatenate & scale blocks (host-side)
def concat_and_scale_blocks(blocks, weights):
    """
    blocks: list of (A_block, b_block) where A_block: (n_rows, n_cols) numpy,
            b_block: (n_rows,) numpy
    weights: list of floats, same length as blocks

    Return: concatenated A (n_total_rows, n_cols), b (n_total_rows,)
    scaled by sqrt(weights[i]).
    """
    assert len(blocks) == len(weights)
    # if all weights zero -> return empty arrays
    if all((w == 0 or b is None) for (A, b), w in zip(blocks, weights)):
        n_cols = blocks[0][0].shape[1]  # assume at least one block exists
        return np.empty((0, n_cols), dtype=np.float64), np.empty((0,), dtype=np.float64)

    A_parts = []
    b_parts = []
    for (A, b), w in zip(blocks, weights):
        if A is None or b is None:
            continue
        if w == 0.0:
            continue
        s = np.sqrt(w)
        # Cast to float64 (kernel expects float64)
        A_scaled = (A.astype(np.float64, copy=False) * s)
        b_scaled = (b.astype(np.float64, copy=False) * s)
        A_parts.append(A_scaled)
        b_parts.append(b_scaled)
    if not A_parts:
        # no active blocks
        n_cols = blocks[0][0].shape[1]
        return np.empty((0, n_cols), dtype=np.float64), np.empty((0,), dtype=np.float64)

    A_concat = np.vstack(A_parts)
    b_concat = np.concatenate(b_parts)
    return A_concat, b_concat


def least_squares_weighted_gpu(blocks, weights, chunk_size=100_000, threads_per_block=256):
    """
    blocks: list of (A_block, b_block) host (numpy) arrays.
    weights: list of floats (same length)
    This function concatenates scaled blocks and uses your GPU kernel compute_XTX_XTy
    to compute XTX and XTy in chunks, then solves the normal equations.
    Returns: beta (1D numpy array) of length n_cols.
    """
    # Prepare scaled concatenated design and target
    X_all, y_all = concat_and_scale_blocks(blocks, weights)
    n_rows, n_cols = X_all.shape

    # Edge case: nothing to solve
    if n_rows == 0:
        return np.zeros((n_cols,), dtype=np.float64)

    # allocate accumulators
    XTX_host = np.zeros((n_cols, n_cols), dtype=np.float64)
    XTy_host = np.zeros((n_cols,), dtype=np.float64)

    threads = threads_per_block
    # blocks_per_grid computed by chunk_size (we reuse kernel-grid computation)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        X_chunk = X_all[start:end]
        y_chunk = y_all[start:end]

        # prepare device arrays
        d_X = cuda.to_device(X_chunk)
        d_y = cuda.to_device(y_chunk)
        d_XTX = cuda.to_device(np.zeros((n_cols, n_cols), dtype=np.float64))
        d_XTy = cuda.to_device(np.zeros((n_cols,), dtype=np.float64))

        # compute blocks per grid for current chunk length
        rows_cur = X_chunk.shape[0]
        blocks_per_grid = (rows_cur + threads - 1) // threads

        # run kernel
        compute_XTX_XTy[blocks_per_grid, threads](d_X, d_y, d_XTX, d_XTy)

        # accumulate
        XTX_host += d_XTX.copy_to_host()
        XTy_host += d_XTy.copy_to_host()

    # Solve XTX beta = XTy - with small regularization for numerical stability
    # Regularization lambda small (1e-12) to handle near-singular systems
    reg = 1e-12
    try:
        beta = np.linalg.solve(XTX_host + reg * np.eye(n_cols, dtype=np.float64), XTy_host)
    except np.linalg.LinAlgError:
        # fallback to least-squares solve (stable)
        beta, *_ = np.linalg.lstsq(XTX_host + reg * np.eye(n_cols, dtype=np.float64), XTy_host, rcond=None)
    return beta



def least_squares_gpu(X_host, y_host, chunk_size=100000):
    n_rows, n_cols = X_host.shape

    XTX_host = np.zeros((n_cols, n_cols), dtype=np.float64)
    XTy_host = np.zeros(n_cols, dtype=np.float64)

    threads_per_block = 256
    blocks_per_grid = (chunk_size + threads_per_block - 1) // threads_per_block

    for i in range(0, n_rows, chunk_size):
        end = min(i + chunk_size, n_rows)
        X_chunk = X_host[i:end]
        y_chunk = y_host[i:end]

        d_X = cuda.to_device(X_chunk)
        d_y = cuda.to_device(y_chunk)
        d_XTX = cuda.to_device(np.zeros((n_cols, n_cols), dtype=np.float64))
        d_XTy = cuda.to_device(np.zeros(n_cols, dtype=np.float64))

        compute_XTX_XTy[blocks_per_grid, threads_per_block](d_X, d_y, d_XTX, d_XTy)

        XTX_host += d_XTX.copy_to_host()
        XTy_host += d_XTy.copy_to_host()

    beta = np.linalg.solve(XTX_host, XTy_host)
    return beta

# Example usage:
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(10**6, 4)
    y = np.random.rand(10**6)
    beta = least_squares_gpu(X, y)
    print("Beta:", beta)
