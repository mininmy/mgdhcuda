import numpy as np
from numba import cuda, float64

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
