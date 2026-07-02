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

After each round the cumulative polynomial (sum of all round polynomials
with their accumulated scale factors applied) is merged — equal monomials
are combined — and saved to poly_saves/round_R.json and round_R.txt.
"""
import json
import os

import numpy as np
import cupy as cp

from gpu_gmdh_newton_model import GMDHTrainerGPU, generate_taylor_green_data

# ------------------------------------------------------------------ #
# Hyperparameters                                                      #
# ------------------------------------------------------------------ #

TRAINER_KWARGS = dict(
    viscosity      = 0.01,
    chunk_size     = 10_000,
    top_models     = 30,
    prune_thresh   = 1e-19,
    qr_sub_size    = 1000,
    jac_sys_chunk  = 700,
    svd_rcond      = 1e-8,
    corr_threshold = 0.99,
)

N_LAYERS         = 4
MAX_BOOST_ROUNDS = 10
RESIDUAL_TOL     = 1e-8    # stop when every component's residual std < this
EVAL_CHUNK       = 50_000  # rows per GPU pass during prediction
POLY_SAVE_DIR    = "poly_saves"
VAR_NAMES        = ["x", "y", "t"]
COMP_NAMES       = ["u", "v"]
POLY_PRUNE       = 1e-15   # drop merged monomials with |coeff| below this


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


def _merge_polys(scale_poly_pairs, prune_thresh=1e-15):
    """
    Combine a list of (scale, PolynomialGPU) into a single polynomial by
    summing coefficients of identical monomials.

    scale_poly_pairs : list of (float, PolynomialGPU)
    Returns (exps, coeffs) as numpy arrays sorted by descending |coeff|.
    """
    merged = {}
    n_vars = None
    for scale, poly in scale_poly_pairs:
        poly._ensure_cpu_cache()
        exps   = poly._cpu_exps
        coeffs = poly._cpu_coeffs
        if exps is None or len(exps) == 0:
            continue
        if n_vars is None:
            n_vars = exps.shape[1]
        for i in range(len(coeffs)):
            key = tuple(int(e) for e in exps[i])
            merged[key] = merged.get(key, 0.0) + float(scale) * float(coeffs[i])

    if not merged or n_vars is None:
        nv = n_vars if n_vars is not None else len(VAR_NAMES)
        return np.zeros((0, nv), dtype=np.uint16), np.zeros(0, dtype=np.float64)

    items = [(k, v) for k, v in merged.items() if abs(v) > prune_thresh]
    if not items:
        return np.zeros((0, n_vars), dtype=np.uint16), np.zeros(0, dtype=np.float64)

    items.sort(key=lambda x: -abs(x[1]))
    out_exps   = np.array([k for k, _ in items], dtype=np.uint16)
    out_coeffs = np.array([v for _, v in items], dtype=np.float64)
    return out_exps, out_coeffs


def _monomial_str(exps_row, coeff, var_names):
    """Format one monomial as  +c.ddde±dd * x^a * y^b * ..."""
    parts = []
    for v, e in enumerate(exps_row):
        if e == 1:
            parts.append(var_names[v])
        elif e > 1:
            parts.append(f"{var_names[v]}^{int(e)}")
    body = " * ".join(parts) if parts else "1"
    return f"{coeff:+.10e} * {body}"


def save_polys_round(round_idx, cum_pairs, var_names, comp_names,
                     out_dir, prune_thresh=1e-15):
    """
    Merge the cumulative polynomial for each component and write:
      <out_dir>/round_<R>.json  — structured (variables, terms, exponents)
      <out_dir>/round_<R>.txt   — plain-text mathematical expression

    cum_pairs : list[list[(float, PolynomialGPU)]]
        cum_pairs[ic] = [(scale_r, poly_r), ...] for all rounds up to R
    """
    os.makedirs(out_dir, exist_ok=True)
    n_comp = len(cum_pairs)

    merged = []
    for ic in range(n_comp):
        exps, coeffs = _merge_polys(cum_pairs[ic], prune_thresh)
        merged.append((comp_names[ic] if ic < len(comp_names) else f"comp{ic}",
                       exps, coeffs))

    # ---- JSON --------------------------------------------------------
    json_path = os.path.join(out_dir, f"round_{round_idx}.json")
    doc = {"round": round_idx, "variables": var_names, "components": []}
    for comp_name, exps, coeffs in merged:
        terms = []
        for i in range(len(coeffs)):
            exp_dict = {var_names[v]: int(exps[i, v])
                        for v in range(len(var_names)) if exps[i, v] > 0}
            terms.append({"coeff": float(coeffs[i]), "exponents": exp_dict})
        doc["components"].append({
            "name":    comp_name,
            "n_terms": len(coeffs),
            "terms":   terms,
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    # ---- plain text --------------------------------------------------
    txt_path = os.path.join(out_dir, f"round_{round_idx}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Boosted GMDH — cumulative polynomial after round {round_idx}\n")
        f.write(f"# Variables : {', '.join(var_names)}\n\n")
        for comp_name, exps, coeffs in merged:
            f.write(f"{'='*60}\n")
            f.write(f"Component : {comp_name}  ({len(coeffs)} terms)\n")
            f.write(f"{'='*60}\n")
            if len(coeffs) == 0:
                f.write("  (zero polynomial)\n")
            else:
                for i in range(len(coeffs)):
                    f.write(f"  {_monomial_str(exps[i], coeffs[i], var_names)}\n")
            f.write("\n")

    print(f"  Saved polynomials → {json_path}  ({sum(len(c) for _, c, _ in merged if True)} terms total)")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    X, y = generate_taylor_green_data(1000_000, nu=0.01)
    n_comp = len(y)

    # Combined prediction accumulated across rounds
    cum_pred  = [np.zeros_like(yc) for yc in y]
    # Multiplicative weight applied to round k's raw prediction
    # before adding to cum_pred.  Starts at 1; multiplied by std each round.
    cum_scale = [1.0] * n_comp

    # (scale, PolynomialGPU) history per component for polynomial merging
    cum_pairs = [[] for _ in range(n_comp)]

    current_targets = [yc.copy() for yc in y]
    round_log = []

    for round_idx in range(MAX_BOOST_ROUNDS):
        print(f"\n{'='*60}")
        print(f"  Boosting round {round_idx}")
        print(f"{'='*60}")

        trainer = GMDHTrainerGPU(**TRAINER_KWARGS)
        trainer.fit(X, current_targets, n_layers=N_LAYERS)

        preds = predict(trainer, X)

        # Record (scale, poly) before cum_scale is updated for next round.
        # cum_scale[ic] at this point equals prod(res_std[0..round-1]),
        # which is exactly the factor this round's poly enters the sum with.
        polys = trainer.best_model
        for ic in range(n_comp):
            polys[ic]._ensure_cpu_cache()
            cum_pairs[ic].append((cum_scale[ic], polys[ic]))

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

        save_polys_round(round_idx, cum_pairs, VAR_NAMES, COMP_NAMES,
                         POLY_SAVE_DIR, POLY_PRUNE)

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
