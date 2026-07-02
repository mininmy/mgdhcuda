"""
adaptive_uvp_degree_growth.py
=============================
Adaptive full-basis degree continuation for unknown u, v, and pressure.

The active polynomial basis is all monomials with total degree <= current
degree.  After the score stagnates, the basis degree is increased by
DEGREE_INCREMENT and existing coefficients are copied into the larger basis.

This is the "increasing window" experiment:

    degree <= 5  ->  degree <= 6  ->  degree <= 7  -> ...

Per-sweep diagnostics are written immediately to CSV for plotting.
"""

import csv
import json
import math
import os
import time

import cupy as cp

import physics_only_uv_unknown_pressure as uvp


START_DEGREE = 5
MAX_DEGREE = 10
DEGREE_INCREMENT = 1

MIN_SWEEPS_PER_DEGREE = 30
MAX_SWEEPS_PER_DEGREE = 250
PLATEAU_WINDOW = 20
MIN_REL_IMPROVEMENT = 0.005
SCORE_EXPLOSION_FACTOR = 10.0

OUT_DIR = "poly_saves"
OUT_BASENAME = "adaptive_uvp_degree_growth"
METRICS_CSV = os.path.join(OUT_DIR, OUT_BASENAME + ".csv")
SUMMARY_JSON = os.path.join(OUT_DIR, OUT_BASENAME + "_summary.json")


CSV_FIELDS = [
    "global_sweep",
    "stage",
    "stage_sweep",
    "degree",
    "n_terms",
    "u_rmse",
    "v_rmse",
    "p_rmse",
    "px_rmse",
    "py_rmse",
    "score",
    "u_step",
    "p_after_u_step",
    "v_step",
    "p_after_v_step",
    "sweep_seconds",
    "elapsed_seconds",
    "is_best",
]


def plateau_reached(scores):
    if len(scores) < MIN_SWEEPS_PER_DEGREE or len(scores) < 2 * PLATEAU_WINDOW:
        return False
    previous_best = min(scores[:-PLATEAU_WINDOW])
    recent_best = min(scores[-PLATEAU_WINDOW:])
    rel_improvement = (previous_best - recent_best) / max(abs(previous_best), 1e-300)
    return rel_improvement < MIN_REL_IMPROVEMENT


def write_row(writer, csv_file, row):
    writer.writerow(row)
    csv_file.flush()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = uvp.generate_taylor_green(
        uvp.TOTAL_POINTS, uvp.NU
    )

    u_coeffs = None
    v_coeffs = None
    p_coeffs = None
    old_exps = None
    best = None
    summary_stages = []
    global_sweep = 0
    stage_idx = 0
    degree = START_DEGREE
    run_start_time = time.perf_counter()

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        while degree <= MAX_DEGREE:
            exps = uvp.make_total_degree_exponents(3, degree)
            deriv_cache = uvp.make_deriv_cache(exps)
            n_terms = exps.shape[0]

            if u_coeffs is None:
                u_coeffs = cp.zeros(n_terms, dtype=cp.float64)
                v_coeffs = cp.zeros(n_terms, dtype=cp.float64)
                p_coeffs = cp.zeros(n_terms, dtype=cp.float64)
                copied = (0, 0, 0)
            else:
                u_coeffs, copied_u = uvp.expand_coefficients(old_exps, u_coeffs, exps)
                v_coeffs, copied_v = uvp.expand_coefficients(old_exps, v_coeffs, exps)
                p_coeffs, copied_p = uvp.expand_coefficients(old_exps, p_coeffs, exps)
                copied = (copied_u, copied_v, copied_p)
            old_exps = exps

            print(
                f"\nStage {stage_idx}: degree <= {degree}, terms={n_terms}, "
                f"copied u/v/p={copied[0]}/{copied[1]}/{copied[2]}"
            )

            stage_scores = []
            stage_best = None
            stage_stop_reason = "max_sweeps"
            stage_start_sweep = global_sweep
            for stage_sweep in range(MAX_SWEEPS_PER_DEGREE):
                sweep_start_time = time.perf_counter()
                u_step, _u_phys = uvp.solve_update(
                    uvp.accumulate_u_update(X, u_true, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                    u_coeffs,
                )
                p_after_u_step, _p_after_u_phys = uvp.solve_update(
                    uvp.accumulate_pressure_update(X, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                    p_coeffs,
                )
                v_step, _v_phys = uvp.solve_update(
                    uvp.accumulate_v_update(X, v_true, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                    v_coeffs,
                )
                p_after_v_step, _p_after_v_phys = uvp.solve_update(
                    uvp.accumulate_pressure_update(X, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                    p_coeffs,
                )

                u_cpu = u_coeffs.get()
                v_cpu = v_coeffs.get()
                p_cpu = p_coeffs.get()
                diag = uvp.diagnostics(
                    X, exps, deriv_cache, u_cpu, v_cpu, p_cpu,
                    u_true, v_true, p_true, pg_x_true, pg_y_true,
                )
                score = uvp.score_diagnostics(diag)
                stage_scores.append(score)
                sweep_seconds = time.perf_counter() - sweep_start_time
                elapsed_seconds = time.perf_counter() - run_start_time

                if stage_best is None or score < stage_best["score"]:
                    stage_best = {
                        "score": score,
                        "global_sweep": global_sweep,
                        "stage_sweep": stage_sweep,
                        "diag": dict(diag),
                        "u": u_cpu.copy(),
                        "v": v_cpu.copy(),
                        "p": p_cpu.copy(),
                    }

                is_best = best is None or score < best["score"]
                if is_best:
                    best = {
                        "score": score,
                        "global_sweep": global_sweep,
                        "stage": stage_idx,
                        "stage_sweep": stage_sweep,
                        "degree": degree,
                        "diag": dict(diag),
                        "exps": exps.copy(),
                        "u": u_cpu.copy(),
                        "v": v_cpu.copy(),
                        "p": p_cpu.copy(),
                    }

                write_row(writer, csv_file, {
                    "global_sweep": global_sweep,
                    "stage": stage_idx,
                    "stage_sweep": stage_sweep,
                    "degree": degree,
                    "n_terms": n_terms,
                    "u_rmse": diag["u"],
                    "v_rmse": diag["v"],
                    "p_rmse": diag["p"],
                    "px_rmse": diag["px"],
                    "py_rmse": diag["py"],
                    "score": score,
                    "u_step": u_step,
                    "p_after_u_step": p_after_u_step,
                    "v_step": v_step,
                    "p_after_v_step": p_after_v_step,
                    "sweep_seconds": sweep_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "is_best": int(is_best),
                })

                best_note = " best" if is_best else ""
                print(
                    f"Sweep {global_sweep:03d} stage={stage_idx} local={stage_sweep:03d} "
                    f"degree={degree}: u={diag['u']:.6e} v={diag['v']:.6e} "
                    f"p={diag['p']:.6e} px={diag['px']:.6e} py={diag['py']:.6e} "
                    f"score={score:.6e} time={sweep_seconds:.1f}s{best_note}"
                )
                global_sweep += 1

                if plateau_reached(stage_scores):
                    print(
                        f"Stage {stage_idx} plateau: best recent improvement below "
                        f"{MIN_REL_IMPROVEMENT:g} over {PLATEAU_WINDOW} sweeps."
                    )
                    stage_stop_reason = "plateau"
                    break

                if (
                    not math.isfinite(score)
                    or score > stage_best["score"] * SCORE_EXPLOSION_FACTOR
                ):
                    print(
                        f"Stage {stage_idx} stopped: score exploded/non-finite "
                        f"(score={score:.6e}, stage_best={stage_best['score']:.6e})."
                    )
                    stage_stop_reason = "exploded"
                    break

            if stage_best is not None:
                u_coeffs = cp.asarray(stage_best["u"], dtype=cp.float64)
                v_coeffs = cp.asarray(stage_best["v"], dtype=cp.float64)
                p_coeffs = cp.asarray(stage_best["p"], dtype=cp.float64)
                print(
                    f"Stage {stage_idx} rollback/pass-forward checkpoint: "
                    f"local={stage_best['stage_sweep']} score={stage_best['score']:.6e}"
                )

            summary_stages.append({
                "stage": stage_idx,
                "degree": degree,
                "start_sweep": stage_start_sweep,
                "n_sweeps": len(stage_scores),
                "best_score": min(stage_scores),
                "last_score": stage_scores[-1],
                "best_stage_sweep": stage_best["stage_sweep"] if stage_best else None,
                "stop_reason": stage_stop_reason,
                "elapsed_seconds": time.perf_counter() - run_start_time,
            })

            degree += DEGREE_INCREMENT
            stage_idx += 1
            cp.get_default_memory_pool().free_all_blocks()

    best_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_best")
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    uvp.save_solution(best["exps"], best["u"], best["v"], best["p"], best_prefix)
    uvp.save_solution(exps, u_coeffs.get(), v_coeffs.get(), p_coeffs.get(), final_prefix)

    summary = {
        "mode": "adaptive_full_degree_growth",
        "metrics_csv": METRICS_CSV,
        "stages": summary_stages,
        "best": {k: v for k, v in best.items() if k not in {"exps", "u", "v", "p"}},
        "best_solution_prefix": best_prefix,
        "final_solution_prefix": final_prefix,
        "settings": {
            "start_degree": START_DEGREE,
            "max_degree": MAX_DEGREE,
            "degree_increment": DEGREE_INCREMENT,
            "min_sweeps_per_degree": MIN_SWEEPS_PER_DEGREE,
            "max_sweeps_per_degree": MAX_SWEEPS_PER_DEGREE,
            "plateau_window": PLATEAU_WINDOW,
            "min_rel_improvement": MIN_REL_IMPROVEMENT,
            "score_explosion_factor": SCORE_EXPLOSION_FACTOR,
        },
        "elapsed_seconds": time.perf_counter() - run_start_time,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved metrics to {METRICS_CSV}")
    print(f"Saved summary to {SUMMARY_JSON}")
    print(f"Saved best solution to {best_prefix}.json/.txt")
    print(f"Saved final solution to {final_prefix}.json/.txt")


if __name__ == "__main__":
    main()
