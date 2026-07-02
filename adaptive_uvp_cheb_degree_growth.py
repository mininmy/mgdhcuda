"""Adaptive growing-degree u/v/p fit using Chebyshev basis."""

import csv
import json
import math
import os
import time

import cupy as cp

import physics_only_uv_chebyshev as cheb


START_DEGREE = 2
MAX_DEGREE = 20
DEGREE_INCREMENT = 1
MIN_SWEEPS_PER_DEGREE = 30
MAX_SWEEPS_PER_DEGREE = 250
PLATEAU_WINDOW = 20
MIN_REL_IMPROVEMENT = 0.005
SCORE_EXPLOSION_FACTOR = 10.0

OUT_DIR = "poly_saves"
OUT_BASENAME = "adaptive_uvp_cheb_degree_growth"
METRICS_CSV = os.path.join(OUT_DIR, OUT_BASENAME + ".csv")
SUMMARY_JSON = os.path.join(OUT_DIR, OUT_BASENAME + "_summary.json")

CSV_FIELDS = [
    "global_sweep", "stage", "stage_sweep", "degree", "n_terms",
    "u_rmse", "v_rmse", "p_rmse", "px_rmse", "py_rmse", "score",
    "u_step", "p_after_u_step", "v_step", "p_after_v_step",
    "sweep_seconds", "elapsed_seconds", "is_best",
]


def plateau_reached(scores):
    if len(scores) < MIN_SWEEPS_PER_DEGREE or len(scores) < 2 * PLATEAU_WINDOW:
        return False
    previous_best = min(scores[:-PLATEAU_WINDOW])
    recent_best = min(scores[-PLATEAU_WINDOW:])
    rel_improvement = (previous_best - recent_best) / max(abs(previous_best), 1e-300)
    return rel_improvement < MIN_REL_IMPROVEMENT


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = cheb.generate_taylor_green(cheb.TOTAL_POINTS, cheb.NU)

    u_coeffs = v_coeffs = p_coeffs = None
    old_exps = None
    best = None
    summary_stages = []
    global_sweep = 0
    stage_idx = 0
    degree = START_DEGREE
    run_start = time.perf_counter()

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader(); csv_file.flush()

        while degree <= MAX_DEGREE:
            exps = cheb.make_total_degree_exponents(3, degree)
            n_terms = exps.shape[0]
            if u_coeffs is None:
                u_coeffs = cp.zeros(n_terms, dtype=cp.float64)
                v_coeffs = cp.zeros(n_terms, dtype=cp.float64)
                p_coeffs = cp.zeros(n_terms, dtype=cp.float64)
                copied = (0, 0, 0)
            else:
                u_coeffs, cu = cheb.expand_coefficients(old_exps, u_coeffs, exps)
                v_coeffs, cv = cheb.expand_coefficients(old_exps, v_coeffs, exps)
                p_coeffs, cp_ = cheb.expand_coefficients(old_exps, p_coeffs, exps)
                copied = (cu, cv, cp_)
            old_exps = exps
            print(f"\nCheb stage {stage_idx}: degree <= {degree}, terms={n_terms}, copied={copied}")

            stage_scores = []
            stage_best = None
            stop_reason = "max_sweeps"
            stage_start = global_sweep
            for stage_sweep in range(MAX_SWEEPS_PER_DEGREE):
                sweep_start = time.perf_counter()
                u_step, _ = cheb.solve_update(cheb.accumulate_u_update(X, u_true, exps, u_coeffs, v_coeffs, p_coeffs), u_coeffs)
                p_after_u_step, _ = cheb.solve_update(cheb.accumulate_pressure_update(X, exps, u_coeffs, v_coeffs, p_coeffs), p_coeffs)
                v_step, _ = cheb.solve_update(cheb.accumulate_v_update(X, v_true, exps, u_coeffs, v_coeffs, p_coeffs), v_coeffs)
                p_after_v_step, _ = cheb.solve_update(cheb.accumulate_pressure_update(X, exps, u_coeffs, v_coeffs, p_coeffs), p_coeffs)

                u_cpu, v_cpu, p_cpu = u_coeffs.get(), v_coeffs.get(), p_coeffs.get()
                diag = cheb.diagnostics(X, exps, u_cpu, v_cpu, p_cpu, u_true, v_true, p_true, pg_x_true, pg_y_true)
                score = cheb.score_diagnostics(diag)
                stage_scores.append(score)
                sweep_seconds = time.perf_counter() - sweep_start
                elapsed_seconds = time.perf_counter() - run_start

                if stage_best is None or score < stage_best["score"]:
                    stage_best = {"score": score, "stage_sweep": stage_sweep, "u": u_cpu.copy(), "v": v_cpu.copy(), "p": p_cpu.copy(), "diag": dict(diag)}
                is_best = best is None or score < best["score"]
                if is_best:
                    best = {"score": score, "global_sweep": global_sweep, "stage": stage_idx, "stage_sweep": stage_sweep,
                            "degree": degree, "diag": dict(diag), "exps": exps.copy(), "u": u_cpu.copy(), "v": v_cpu.copy(), "p": p_cpu.copy()}

                writer.writerow({"global_sweep": global_sweep, "stage": stage_idx, "stage_sweep": stage_sweep, "degree": degree,
                                 "n_terms": n_terms, "u_rmse": diag["u"], "v_rmse": diag["v"], "p_rmse": diag["p"],
                                 "px_rmse": diag["px"], "py_rmse": diag["py"], "score": score,
                                 "u_step": u_step, "p_after_u_step": p_after_u_step, "v_step": v_step,
                                 "p_after_v_step": p_after_v_step, "sweep_seconds": sweep_seconds,
                                 "elapsed_seconds": elapsed_seconds, "is_best": int(is_best)})
                csv_file.flush()
                print(f"Cheb sweep {global_sweep:03d} stage={stage_idx} local={stage_sweep:03d} degree={degree}: "
                      f"u={diag['u']:.6e} v={diag['v']:.6e} p={diag['p']:.6e} px={diag['px']:.6e} py={diag['py']:.6e} "
                      f"score={score:.6e} time={sweep_seconds:.1f}s" + (" best" if is_best else ""))
                global_sweep += 1

                if plateau_reached(stage_scores):
                    stop_reason = "plateau"; print(f"Cheb stage {stage_idx} plateau reached."); break
                if not math.isfinite(score) or score > stage_best["score"] * SCORE_EXPLOSION_FACTOR:
                    stop_reason = "exploded"; print(f"Cheb stage {stage_idx} stopped: score exploded/non-finite."); break

            if stage_best is not None:
                u_coeffs = cp.asarray(stage_best["u"], dtype=cp.float64)
                v_coeffs = cp.asarray(stage_best["v"], dtype=cp.float64)
                p_coeffs = cp.asarray(stage_best["p"], dtype=cp.float64)
                print(f"Cheb stage {stage_idx} pass-forward local={stage_best['stage_sweep']} score={stage_best['score']:.6e}")

            summary_stages.append({"stage": stage_idx, "degree": degree, "start_sweep": stage_start,
                                   "n_sweeps": len(stage_scores), "best_score": min(stage_scores),
                                   "last_score": stage_scores[-1], "best_stage_sweep": stage_best["stage_sweep"],
                                   "stop_reason": stop_reason, "elapsed_seconds": time.perf_counter() - run_start})
            degree += DEGREE_INCREMENT
            stage_idx += 1
            cp.get_default_memory_pool().free_all_blocks()

    best_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_best")
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    cheb.save_solution(best["exps"], best["u"], best["v"], best["p"], best_prefix)
    cheb.save_solution(exps, u_coeffs.get(), v_coeffs.get(), p_coeffs.get(), final_prefix)
    summary = {"mode": "chebyshev_adaptive_full_degree_growth", "metrics_csv": METRICS_CSV, "stages": summary_stages,
               "best": {k: v for k, v in best.items() if k not in {"exps", "u", "v", "p"}},
               "best_solution_prefix": best_prefix, "final_solution_prefix": final_prefix,
               "settings": {"start_degree": START_DEGREE, "max_degree": MAX_DEGREE, "degree_increment": DEGREE_INCREMENT,
                            "min_sweeps_per_degree": MIN_SWEEPS_PER_DEGREE, "max_sweeps_per_degree": MAX_SWEEPS_PER_DEGREE,
                            "plateau_window": PLATEAU_WINDOW, "min_rel_improvement": MIN_REL_IMPROVEMENT,
                            "score_explosion_factor": SCORE_EXPLOSION_FACTOR},
               "elapsed_seconds": time.perf_counter() - run_start}
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Cheb growth metrics to {METRICS_CSV}")


if __name__ == "__main__":
    main()
