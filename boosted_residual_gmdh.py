"""
boosted_residual_gmdh.py
========================
Residual-boosting loop over GMDHTrainerGPU.

After each training run the predictions of the first assembled model
are subtracted from the current targets.  The residuals are divided
by their per-component standard deviation (so the next trainer sees a
unit-scale target) and the loop repeats.  Cumulative scale factors are
tracked so the combined multi-round prediction can be reconstructed:

    y ≈ pred_0
          + scale_0 * pred_1
          + scale_0 * scale_1 * pred_2
          + ...

where scale_k[ic] = std(residual_k[ic]).
"""

import numpy as np
import cupy as cp

from gpu_gmdh_newton_model import GMDHTrainerGPU, generate_taylor_green_data

# ------------------------------------------------------------------ #
# Hyperparameters                                                      #
# ------------------------------------------------------------------ #

TRAINER_KWARGS = dict(
    viscosity      = 0.01,
    chunk_size     = 10_000,
    top_models     = 50,
    prune_thresh   = 1e-16,
    qr_sub_size    = 1000,
    jac_sys_chunk  = 700,
    svd_rcond      = 1e-8,
    corr_threshold = 0.999,
)

N_LAYERS         = 21
MAX_BOOST_ROUNDS = 5
RESIDUAL_TOL     = 1e-8   # stop when every component's residual std < this
EVAL_CHUNK       = 50_000  # rows per GPU pass during prediction


# ------------------------------------------------------------------ #
# Utilities                                                            #
# ------------------------------------------------------------------ #

def predict(trainer, X_cpu):
    """
    Evaluate trainer.best_model (best per-component poly from the last layer).

    Splits X into EVAL_CHUNK-row batches so VRAM stays bounded even
    for the 1 M-sample dataset.

    Returns a list of n_comp NumPy float64 arrays, each of shape [N].
    """
    model = trainer.best_model
    n_comp = len(model)
    N = X_cpu.shape[0]
    preds = [np.zeros(N, dtype=np.float64) for _ in range(n_comp)]

    for start in range(0, N, EVAL_CHUNK):
        end   = min(start + EVAL_CHUNK, N)
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        for ic, poly in enumerate(model):
            if poly.exponents.shape[0] > 0:
                preds[ic][start:end] = poly.evaluate(X_gpu).get()
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()

    return preds


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    X, y = generate_taylor_green_data(1_000_000, nu=0.01)
    n_comp = len(y)

    # Combined prediction accumulated across rounds
    cum_pred  = [np.zeros_like(yc) for yc in y]
    # Multiplicative weight applied to round k's raw prediction
    # before adding to cum_pred.  Starts at 1; multiplied by std each round.
    cum_scale = [1.0] * n_comp

    current_targets = [yc.copy() for yc in y]
    round_log = []

    for round_idx in range(MAX_BOOST_ROUNDS):
        print(f"\n{'='*60}")
        print(f"  Boosting round {round_idx}")
        print(f"{'='*60}")

        trainer = GMDHTrainerGPU(**TRAINER_KWARGS)
        trainer.fit(X, current_targets, n_layers=N_LAYERS)

        preds = predict(trainer, X)

        # Accumulate into the combined prediction
        for ic in range(n_comp):
            cum_pred[ic] += cum_scale[ic] * preds[ic]

        combined_rmse = [_rmse(y[ic], cum_pred[ic]) for ic in range(n_comp)]
        print(f"\nRound {round_idx} combined RMSE : "
              f"{[f'{r:.4e}' for r in combined_rmse]}")

        residuals = [current_targets[ic] - preds[ic] for ic in range(n_comp)]
        res_std   = [float(np.std(r)) for r in residuals]
        print(f"Round {round_idx} residual std  : "
              f"{[f'{s:.4e}' for s in res_std]}")

        round_log.append(dict(
            round         = round_idx,
            combined_rmse = combined_rmse,
            residual_std  = res_std,
        ))

        if all(s < RESIDUAL_TOL for s in res_std):
            print("All residuals below tolerance — stopping.")
            break

        # Normalise residuals for the next round; track cumulative scale
        next_targets = []
        for ic in range(n_comp):
            s = res_std[ic]
            if s > RESIDUAL_TOL:
                next_targets.append(residuals[ic] / s)
                cum_scale[ic] *= s
            else:
                next_targets.append(np.zeros_like(residuals[ic]))
                cum_scale[ic] = 0.0

        current_targets = next_targets

    # ---- Final summary ------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Boosting complete — {len(round_log)} round(s).")
    for info in round_log:
        print(f"  Round {info['round']:2d}:"
              f"  RMSE {[f'{r:.3e}' for r in info['combined_rmse']]}"
              f"  res_std {[f'{s:.3e}' for s in info['residual_std']]}")
