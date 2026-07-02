"""
boosted_residual_gmdh.py
========================
Residual-boosting loop over GMDHTrainerGPU.

After each training run the prediction is added to a cumulative model.
The next target is always the common residual against the original target:

    residual_k = y - cumulative_prediction_k

That residual is divided by its per-component L2 norm, so each next trainer
sees a unit-norm residual target.  Round scale factors are tracked so the
combined multi-round prediction can be reconstructed:

    y ≈ norm_0 * pred_0
          + norm_1 * pred_1
          + norm_2 * pred_2
          + ...

where norm_k[ic] = ||residual_{k-1}[ic]||_2, with residual_-1 = y.

After each round the cumulative polynomial (sum of all round polynomials
with their accumulated scale factors applied) is merged — equal monomials
are combined — and saved to poly_saves/round_R.json and round_R.txt.
"""
import json
import os

import numpy as np
import cupy as cp

from gpu_gmdh_newton_known_physics import (
    GMDHTrainerGPU,
    generate_taylor_green_data_with_known,
)
from gpu_polynomial_module import PolynomialGPU

# ------------------------------------------------------------------ #
# Hyperparameters                                                      #
# ------------------------------------------------------------------ #

TRAINER_KWARGS = dict(
    viscosity      = 0.01,
    chunk_size     = 10_000,
    top_models     = 200,        # start value; schedule overrides per layer
    prune_thresh   = 0,         # no pruning: small coefficients matter for residuals
    qr_sub_size    = 1000,
    jac_sys_chunk  = 700,
    svd_rcond      = 0,       # ridge handles regularization
    corr_threshold = 1,      # start value; schedule tightens per layer
    ridge_lambda   = 1e-6,
    var_threshold  = 1e-8,
    verbosity      = 2,
)

N_LAYERS         = 5
MAX_BOOST_ROUNDS = 4


def _make_layer_schedule(n_layers):
    """Adaptive schedule: broad diversity early, tight refinement late."""
    sched = []
    for _l in range(n_layers):
        frac = _l / max(n_layers - 1, 1)
        sched.append({
            #'top_models':    120, #max(20, int(180 * (1.0 - 0.6 * frac))),
            #'corr_threshold': 0.8 + 0.095 * frac,
            'ridge_lambda':  1e-6 * (1.0 - frac) + 1e-8 * frac,
            #'svd_rcond':     0.0,
        })
    return sched
RESIDUAL_TOL     = 1e-8    # stop when every component's residual L2 norm < this
EVAL_CHUNK       = 50_000  # rows per GPU pass during prediction
POLY_SAVE_DIR    = "poly_saves"
VAR_NAMES        = ["x", "y", "t"]
COMP_NAMES       = ["u"]
POLY_PRUNE       = 0   # drop merged monomials with |coeff| below this
REFIT_RIDGE      = 1e-8
REFIT_TOP_MODELS = 5
MAX_REFIT_TERMS  = 10_000


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


def _residual_norm(residual):
    return float(np.linalg.norm(residual))


def _normalise_residuals(residuals):
    targets = []
    scales = []
    for residual in residuals:
        scale = _residual_norm(residual)
        scales.append(scale)
        if scale > RESIDUAL_TOL:
            targets.append(residual / scale)
        else:
            targets.append(np.zeros_like(residual))
    return targets, scales


def _evaluate_monomials_gpu(X_gpu, exps_cpu):
    n_rows = X_gpu.shape[0]
    n_terms = exps_cpu.shape[0]
    A = cp.ones((n_rows, n_terms), dtype=cp.float64)
    exps_gpu = cp.asarray(exps_cpu, dtype=cp.uint16)
    for v in range(exps_cpu.shape[1]):
        powers = exps_gpu[:, v]
        max_power = int(powers.max().get()) if n_terms else 0
        if max_power == 0:
            continue
        x_v = X_gpu[:, v]
        for p in range(1, max_power + 1):
            cols = powers == p
            if bool(cols.any()):
                A[:, cols] *= x_v[:, None] ** p
    return A


def _score_poly_terms(scores, poly, scale=1.0):
    poly._ensure_cpu_cache()
    exps = poly._cpu_exps
    coeffs = poly._cpu_coeffs
    if exps is None or coeffs is None:
        return
    for i in range(len(coeffs)):
        key = tuple(int(e) for e in exps[i])
        scores[key] = max(scores.get(key, 0.0), abs(float(scale) * float(coeffs[i])))


def _collect_refit_basis(prev_poly, round_model, component_idx, new_scale, n_vars,
                         max_terms=MAX_REFIT_TERMS):
    scores = {}
    if prev_poly is not None:
        _score_poly_terms(scores, prev_poly, 1.0)
    if component_idx < len(round_model):
        _score_poly_terms(scores, round_model[component_idx], new_scale)

    if not scores:
        return np.zeros((0, n_vars), dtype=np.uint16)

    items = sorted(scores.items(), key=lambda item: -item[1])
    if len(items) > max_terms:
        print(f"  Refit basis capped: {len(items)} -> {max_terms} monomials")
        items = items[:max_terms]

    return np.array([key for key, _ in items], dtype=np.uint16)


def _refit_polynomial(X_cpu, y_cpu, exps_cpu, ridge_lambda=REFIT_RIDGE):
    n_terms = exps_cpu.shape[0]
    n_vars = exps_cpu.shape[1] if exps_cpu.ndim == 2 else len(VAR_NAMES)
    if n_terms == 0:
        return PolynomialGPU(
            cp.zeros((0, n_vars), dtype=cp.uint16),
            cp.zeros(0, dtype=cp.float64),
        )

    gram = cp.zeros((n_terms, n_terms), dtype=cp.float64)
    rhs = cp.zeros(n_terms, dtype=cp.float64)

    for start in range(0, X_cpu.shape[0], EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        y_gpu = cp.asarray(y_cpu[start:end], dtype=cp.float64)
        A = _evaluate_monomials_gpu(X_gpu, exps_cpu)
        gram += A.T @ A
        rhs += A.T @ y_gpu
        del X_gpu, y_gpu, A
        cp.get_default_memory_pool().free_all_blocks()

    if ridge_lambda > 0:
        gram += cp.eye(n_terms, dtype=cp.float64) * np.float64(ridge_lambda)

    coeffs = cp.linalg.solve(gram, rhs)
    return PolynomialGPU(cp.asarray(exps_cpu, dtype=cp.uint16), coeffs)


def _predict_polys(polys, X_cpu):
    N = X_cpu.shape[0]
    preds = [np.zeros(N, dtype=np.float64) for _ in polys]
    for start in range(0, N, EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, N)
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        for ic, poly in enumerate(polys):
            if poly.exponents.shape[0] > 0:
                preds[ic][start:end] = poly.evaluate(X_gpu).get()
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()
    return preds


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
    X, y, v_known, dv_dy_known, pg_x_known = \
    generate_taylor_green_data_with_known(1000_000, nu=0.01)

    n_comp = len(y)   # now == 1

    # Combined prediction from the globally refit polynomial basis.
    cum_pred = [np.zeros_like(yc) for yc in y]
    refit_polys = [None] * n_comp

    # save_polys_round still accepts scale/poly pairs; after refit each
    # component is represented as one cumulative polynomial with scale 1.
    cum_pairs = [[] for _ in range(n_comp)]

    common_residuals = [yc - pred for yc, pred in zip(y, cum_pred)]
    current_targets, round_scale = _normalise_residuals(common_residuals)
    round_log = []

    for round_idx in range(MAX_BOOST_ROUNDS):
        print(f"\n{'='*60}")
        print(f"  Boosting round {round_idx}")
        print(f"{'='*60}")

        # Later boosting rounds already have the easy structure captured;
        # use more layers and slightly higher ridge for harder residual fitting.
        n_layers_r = N_LAYERS + round_idx          # 7, 8, 9, ...
        ridge_r    = TRAINER_KWARGS['ridge_lambda'] * (0.5 ** round_idx)  # halve each round

        round_kwargs = {**TRAINER_KWARGS, 'ridge_lambda': ridge_r}
        trainer = GMDHTrainerGPU(**round_kwargs)
        trainer.fit(
            X,
            current_targets,
            n_layers=n_layers_r,
            known_velocity_cpu=v_known,
            known_dv_dy_cpu=dv_dy_known,
            known_pg_cpu=pg_x_known,
            layer_schedule=_make_layer_schedule(n_layers_r),
        )

        round_models = trainer.current_models[:REFIT_TOP_MODELS]
        if not round_models:
            round_models = [trainer.best_model]

        for ic in range(n_comp):
            results = []
            print(f"\n  Component {ic} top-{len(round_models)} structure refit check:")
            print("    rank | before_refit_rmse | after_refit_rmse | n_terms")
            for rank, model in enumerate(round_models):
                if ic >= len(model):
                    continue
                model[ic]._ensure_cpu_cache()

                raw_pred = _predict_polys([model[ic]], X)[0]
                before_pred = cum_pred[ic] + round_scale[ic] * raw_pred
                before_rmse = _rmse(y[ic], before_pred)

                basis_exps = _collect_refit_basis(
                    refit_polys[ic],
                    model,
                    ic,
                    round_scale[ic],
                    X.shape[1],
                )
                candidate_poly = _refit_polynomial(
                    X,
                    y[ic],
                    basis_exps,
                    ridge_lambda=REFIT_RIDGE,
                )
                candidate_poly._ensure_cpu_cache()
                candidate_pred = _predict_polys([candidate_poly], X)[0]
                after_rmse = _rmse(y[ic], candidate_pred)

                results.append(dict(
                    rank=rank,
                    before_rmse=before_rmse,
                    after_rmse=after_rmse,
                    n_terms=basis_exps.shape[0],
                    poly=candidate_poly,
                    pred=candidate_pred,
                ))
                print(f"    {rank:4d} | {before_rmse:17.6e} | {after_rmse:16.6e} | {basis_exps.shape[0]:7d}")

            if not results:
                raise RuntimeError("No round model was available for refit.")

            best = min(results, key=lambda item: item['after_rmse'])
            refit_polys[ic] = best['poly']
            cum_pred[ic] = best['pred']
            cum_pairs[ic] = [(1.0, refit_polys[ic])]
            print(
                f"  Selected component {ic}: rank {best['rank']} after refit "
                f"RMSE={best['after_rmse']:.6e} ({best['n_terms']} terms)"
            )

        combined_rmse = [_rmse(y[ic], cum_pred[ic]) for ic in range(n_comp)]
        print(f"\nRound {round_idx} combined RMSE : "
              f"{[f'{r:.4e}' for r in combined_rmse]}")

        common_residuals = [y[ic] - cum_pred[ic] for ic in range(n_comp)]
        residual_norms = [_residual_norm(r) for r in common_residuals]
        print(f"Round {round_idx} residual norm : "
              f"{[f'{s:.4e}' for s in residual_norms]}")

        save_polys_round(round_idx, cum_pairs, VAR_NAMES, COMP_NAMES,
                         POLY_SAVE_DIR, POLY_PRUNE)

        round_log.append(dict(
            round         = round_idx,
            combined_rmse = combined_rmse,
            residual_norm = residual_norms,
        ))

        if all(s < RESIDUAL_TOL for s in residual_norms):
            print("All residuals below tolerance — stopping.")
            break

        # Recompute the common residual against the original target, normalize
        # it to unit L2 norm, and train the next round on that target.
        current_targets, round_scale = _normalise_residuals(common_residuals)

    # ---- Final summary ------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Boosting complete — {len(round_log)} round(s).")
    for info in round_log:
        print(f"  Round {info['round']:2d}:"
              f"  RMSE {[f'{r:.3e}' for r in info['combined_rmse']]}"
              f"  res_norm {[f'{s:.3e}' for s in info['residual_norm']]}")
