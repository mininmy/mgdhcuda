"""
adaptive_uvp_hybrid_local_global.py
===================================
Hybrid u/v/p fitting with moving degree windows and periodic full-basis cleanup.

Most stages train a moving correction window:

    0..5, 1..6, 2..7, ...

After every GLOBAL_REFIT_EVERY_N_WINDOWS windows, the cumulative polynomial is
converted to a full degree <= current_max basis and a short global update is
run.  This lets frozen low-degree coefficients rebalance without paying the
full global-update cost at every sweep.

Per-sweep metrics include wall-clock timing and are written immediately to CSV.
"""

import csv
import json
import math
import os
import time

import cupy as cp
import numpy as np

import physics_only_uv_unknown_pressure as uvp
import physics_only_uvp_window_boost as win


START_MIN_DEGREE = 0
START_MAX_DEGREE = 5
MAX_WINDOW_MAX_DEGREE = 10
WINDOW_SHIFT = 1

MIN_SWEEPS_PER_WINDOW = 30
MAX_SWEEPS_PER_WINDOW = 180
PLATEAU_WINDOW = 20
MIN_REL_IMPROVEMENT = 0.005

GLOBAL_REFIT_EVERY_N_WINDOWS = 2
GLOBAL_REFIT_SWEEPS = 35
GLOBAL_ACCEPT_WORSE_TOL = 0.01
GLOBAL_SCORE_EXPLOSION_FACTOR = 5.0

OUT_DIR = "poly_saves"
OUT_BASENAME = "adaptive_uvp_hybrid_local_global"
METRICS_CSV = os.path.join(OUT_DIR, OUT_BASENAME + ".csv")
SUMMARY_JSON = os.path.join(OUT_DIR, OUT_BASENAME + "_summary.json")


CSV_FIELDS = [
    "global_sweep",
    "phase",
    "phase_index",
    "phase_sweep",
    "degree_min",
    "degree_max",
    "active_terms",
    "cumulative_terms",
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
    if len(scores) < MIN_SWEEPS_PER_WINDOW or len(scores) < 2 * PLATEAU_WINDOW:
        return False
    previous_best = min(scores[:-PLATEAU_WINDOW])
    recent_best = min(scores[-PLATEAU_WINDOW:])
    rel_improvement = (previous_best - recent_best) / max(abs(previous_best), 1e-300)
    return rel_improvement < MIN_REL_IMPROVEMENT


def write_row(writer, csv_file, row):
    writer.writerow(row)
    csv_file.flush()


def copy_to_full_basis(src_exps, src_coeffs, degree):
    dst_exps = uvp.make_total_degree_exponents(3, degree)
    dst_coeffs, _copied = uvp.expand_coefficients(src_exps, src_coeffs, dst_exps)
    return dst_exps, dst_coeffs


def full_basis_to_numpy(exps, coeffs):
    return exps.copy(), coeffs.get()


def maybe_update_best(best, score, global_sweep, phase, phase_index, phase_sweep,
                      degree_window, diag, exps, u_coeffs, v_coeffs, p_coeffs):
    if best is not None and score >= best["score"]:
        return best, False
    return {
        "score": score,
        "global_sweep": global_sweep,
        "phase": phase,
        "phase_index": phase_index,
        "phase_sweep": phase_sweep,
        "degree_window": degree_window,
        "diag": dict(diag),
        "exps": exps.copy(),
        "u": u_coeffs.get() if isinstance(u_coeffs, cp.ndarray) else np.asarray(u_coeffs).copy(),
        "v": v_coeffs.get() if isinstance(v_coeffs, cp.ndarray) else np.asarray(v_coeffs).copy(),
        "p": p_coeffs.get() if isinstance(p_coeffs, cp.ndarray) else np.asarray(p_coeffs).copy(),
    }, True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = uvp.generate_taylor_green(
        uvp.TOTAL_POINTS, uvp.NU
    )

    cum_exps = np.zeros((0, 3), dtype=np.uint16)
    empty_cache = win.make_deriv_cache(cum_exps)
    u_cum = cp.zeros(0, dtype=cp.float64)
    v_cum = cp.zeros(0, dtype=cp.float64)
    p_cum = cp.zeros(0, dtype=cp.float64)

    best = None
    phases = []
    global_sweep = 0
    window_idx = 0
    d_min = START_MIN_DEGREE
    d_max = START_MAX_DEGREE
    run_start_time = time.perf_counter()

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        while d_max <= MAX_WINDOW_MAX_DEGREE:
            win_exps = win.make_degree_window_exponents(3, d_min, d_max)
            win_cache = win.make_deriv_cache(win_exps)
            u_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
            v_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
            p_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
            window_scores = []
            phase_start = global_sweep

            print(
                f"\nWindow {window_idx}: degree {d_min}..{d_max}, "
                f"update_terms={win_exps.shape[0]}, cumulative_terms={cum_exps.shape[0]}"
            )

            for local_sweep in range(MAX_SWEEPS_PER_WINDOW):
                sweep_start_time = time.perf_counter()
                cum_cache = win.make_deriv_cache(cum_exps) if cum_exps.shape[0] else empty_cache
                base_state = (
                    (cum_exps, cum_cache, u_cum),
                    (cum_exps, cum_cache, v_cum),
                    (cum_exps, cum_cache, p_cum),
                )

                u_step, _u_phys = win.solve_update(
                    win.accumulate_u_update(
                        X, u_true, base_state, win_exps, win_cache, u_win, v_win, p_win
                    ),
                    u_win,
                )
                p_after_u_step, _p_after_u_phys = win.solve_update(
                    win.accumulate_pressure_update(
                        X, base_state, win_exps, win_cache, u_win, v_win, p_win
                    ),
                    p_win,
                )
                v_step, _v_phys = win.solve_update(
                    win.accumulate_v_update(
                        X, v_true, base_state, win_exps, win_cache, u_win, v_win, p_win
                    ),
                    v_win,
                )
                p_after_v_step, _p_after_v_phys = win.solve_update(
                    win.accumulate_pressure_update(
                        X, base_state, win_exps, win_cache, u_win, v_win, p_win
                    ),
                    p_win,
                )

                tmp_exps, tmp_u = win.merge_coefficients(cum_exps, u_cum, win_exps, u_win)
                _tmp_exps, tmp_v = win.merge_coefficients(cum_exps, v_cum, win_exps, v_win)
                _tmp_exps, tmp_p = win.merge_coefficients(cum_exps, p_cum, win_exps, p_win)
                tmp_cache = win.make_deriv_cache(tmp_exps)
                u_cpu = tmp_u.get()
                v_cpu = tmp_v.get()
                p_cpu = tmp_p.get()
                diag = win.diagnostics(
                    X, tmp_exps, tmp_cache, u_cpu, v_cpu, p_cpu,
                    u_true, v_true, p_true, pg_x_true, pg_y_true,
                )
                score = win.score_diagnostics(diag)
                window_scores.append(score)
                sweep_seconds = time.perf_counter() - sweep_start_time
                elapsed_seconds = time.perf_counter() - run_start_time

                best, is_best = maybe_update_best(
                    best, score, global_sweep, "window", window_idx, local_sweep,
                    (d_min, d_max), diag, tmp_exps, tmp_u, tmp_v, tmp_p
                )

                write_row(writer, csv_file, {
                    "global_sweep": global_sweep,
                    "phase": "window",
                    "phase_index": window_idx,
                    "phase_sweep": local_sweep,
                    "degree_min": d_min,
                    "degree_max": d_max,
                    "active_terms": win_exps.shape[0],
                    "cumulative_terms": tmp_exps.shape[0],
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
                    f"Sweep {global_sweep:03d} window={window_idx} local={local_sweep:03d} "
                    f"degree={d_min}..{d_max}: u={diag['u']:.6e} v={diag['v']:.6e} "
                    f"p={diag['p']:.6e} px={diag['px']:.6e} py={diag['py']:.6e} "
                    f"score={score:.6e} time={sweep_seconds:.1f}s{best_note}"
                )
                global_sweep += 1

                if plateau_reached(window_scores):
                    print(f"Window {window_idx} plateau reached.")
                    break

            next_exps, next_u = win.merge_coefficients(cum_exps, u_cum, win_exps, u_win)
            _next_exps, next_v = win.merge_coefficients(cum_exps, v_cum, win_exps, v_win)
            _next_exps, next_p = win.merge_coefficients(cum_exps, p_cum, win_exps, p_win)
            cum_exps, u_cum, v_cum, p_cum = next_exps, next_u, next_v, next_p
            phases.append({
                "phase": "window",
                "index": window_idx,
                "degree_min": d_min,
                "degree_max": d_max,
                "start_sweep": phase_start,
                "n_sweeps": len(window_scores),
                "best_score": min(window_scores),
                "last_score": window_scores[-1],
                "elapsed_seconds": time.perf_counter() - run_start_time,
            })

            should_refit = (
                (window_idx + 1) % GLOBAL_REFIT_EVERY_N_WINDOWS == 0
                or d_max >= MAX_WINDOW_MAX_DEGREE
            )
            if should_refit:
                full_degree = d_max
                pre_global_exps = cum_exps.copy()
                pre_global_u = u_cum.get()
                pre_global_v = v_cum.get()
                pre_global_p = p_cum.get()
                pre_global_cache = uvp.make_deriv_cache(pre_global_exps)
                pre_global_diag = uvp.diagnostics(
                    X, pre_global_exps, pre_global_cache,
                    pre_global_u, pre_global_v, pre_global_p,
                    u_true, v_true, p_true, pg_x_true, pg_y_true,
                )
                pre_global_score = uvp.score_diagnostics(pre_global_diag)

                full_exps, u_full = copy_to_full_basis(cum_exps, u_cum, full_degree)
                _full_exps, v_full = copy_to_full_basis(cum_exps, v_cum, full_degree)
                _full_exps, p_full = copy_to_full_basis(cum_exps, p_cum, full_degree)
                full_cache = uvp.make_deriv_cache(full_exps)

                print(
                    f"\nGlobal cleanup after window {window_idx}: degree <= {full_degree}, "
                    f"terms={full_exps.shape[0]}, sweeps={GLOBAL_REFIT_SWEEPS}"
                )
                cleanup_scores = []
                cleanup_best = None
                cleanup_stop_reason = "max_sweeps"
                cleanup_start = global_sweep
                for cleanup_sweep in range(GLOBAL_REFIT_SWEEPS):
                    sweep_start_time = time.perf_counter()
                    u_step, _u_phys = uvp.solve_update(
                        uvp.accumulate_u_update(
                            X, u_true, full_exps, full_cache, u_full, v_full, p_full
                        ),
                        u_full,
                    )
                    p_after_u_step, _p_after_u_phys = uvp.solve_update(
                        uvp.accumulate_pressure_update(
                            X, full_exps, full_cache, u_full, v_full, p_full
                        ),
                        p_full,
                    )
                    v_step, _v_phys = uvp.solve_update(
                        uvp.accumulate_v_update(
                            X, v_true, full_exps, full_cache, u_full, v_full, p_full
                        ),
                        v_full,
                    )
                    p_after_v_step, _p_after_v_phys = uvp.solve_update(
                        uvp.accumulate_pressure_update(
                            X, full_exps, full_cache, u_full, v_full, p_full
                        ),
                        p_full,
                    )

                    u_cpu = u_full.get()
                    v_cpu = v_full.get()
                    p_cpu = p_full.get()
                    diag = uvp.diagnostics(
                        X, full_exps, full_cache, u_cpu, v_cpu, p_cpu,
                        u_true, v_true, p_true, pg_x_true, pg_y_true,
                    )
                    score = uvp.score_diagnostics(diag)
                    cleanup_scores.append(score)
                    sweep_seconds = time.perf_counter() - sweep_start_time
                    elapsed_seconds = time.perf_counter() - run_start_time

                    if cleanup_best is None or score < cleanup_best["score"]:
                        cleanup_best = {
                            "score": score,
                            "sweep": cleanup_sweep,
                            "diag": dict(diag),
                            "u": u_cpu.copy(),
                            "v": v_cpu.copy(),
                            "p": p_cpu.copy(),
                        }

                    best, is_best = maybe_update_best(
                        best, score, global_sweep, "global", window_idx, cleanup_sweep,
                        (0, full_degree), diag, full_exps, u_full, v_full, p_full
                    )

                    write_row(writer, csv_file, {
                        "global_sweep": global_sweep,
                        "phase": "global",
                        "phase_index": window_idx,
                        "phase_sweep": cleanup_sweep,
                        "degree_min": 0,
                        "degree_max": full_degree,
                        "active_terms": full_exps.shape[0],
                        "cumulative_terms": full_exps.shape[0],
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
                        f"Sweep {global_sweep:03d} global_after_window={window_idx} "
                        f"local={cleanup_sweep:03d} degree<= {full_degree}: "
                        f"u={diag['u']:.6e} v={diag['v']:.6e} p={diag['p']:.6e} "
                        f"px={diag['px']:.6e} py={diag['py']:.6e} "
                        f"score={score:.6e} time={sweep_seconds:.1f}s{best_note}"
                    )
                    global_sweep += 1

                    if not math.isfinite(score):
                        cleanup_stop_reason = "non_finite"
                        print(
                            f"Global cleanup after window {window_idx} stopped: "
                            "non-finite score."
                        )
                        break

                    if score > cleanup_best["score"] * GLOBAL_SCORE_EXPLOSION_FACTOR:
                        cleanup_stop_reason = "exploded_vs_cleanup_best"
                        print(
                            f"Global cleanup after window {window_idx} stopped: "
                            f"score exploded (score={score:.6e}, "
                            f"cleanup_best={cleanup_best['score']:.6e})."
                        )
                        break

                    if (
                        cleanup_sweep >= 1
                        and score > pre_global_score * (1.0 + GLOBAL_ACCEPT_WORSE_TOL)
                    ):
                        cleanup_stop_reason = "worse_than_pre_global"
                        print(
                            f"Global cleanup after window {window_idx} stopped: "
                            f"score worse than pre-cleanup by more than "
                            f"{GLOBAL_ACCEPT_WORSE_TOL:g}."
                        )
                        break

                if cleanup_best is not None and cleanup_best["score"] < pre_global_score:
                    cum_exps = full_exps.copy()
                    u_cum = cp.asarray(cleanup_best["u"], dtype=cp.float64)
                    v_cum = cp.asarray(cleanup_best["v"], dtype=cp.float64)
                    p_cum = cp.asarray(cleanup_best["p"], dtype=cp.float64)
                    cleanup_accepted = True
                    print(
                        f"Accepted global cleanup after window {window_idx}: "
                        f"local={cleanup_best['sweep']} "
                        f"score={cleanup_best['score']:.6e} "
                        f"pre_score={pre_global_score:.6e}"
                    )
                else:
                    cum_exps = pre_global_exps
                    u_cum = cp.asarray(pre_global_u, dtype=cp.float64)
                    v_cum = cp.asarray(pre_global_v, dtype=cp.float64)
                    p_cum = cp.asarray(pre_global_p, dtype=cp.float64)
                    cleanup_accepted = False
                    best_cleanup_score = cleanup_best["score"] if cleanup_best else float("nan")
                    print(
                        f"Rejected global cleanup after window {window_idx}: "
                        f"best_cleanup={best_cleanup_score:.6e} "
                        f"pre_score={pre_global_score:.6e}"
                    )

                phases.append({
                    "phase": "global",
                    "after_window": window_idx,
                    "degree_min": 0,
                    "degree_max": full_degree,
                    "start_sweep": cleanup_start,
                    "n_sweeps": len(cleanup_scores),
                    "best_score": min(cleanup_scores),
                    "last_score": cleanup_scores[-1],
                    "pre_global_score": pre_global_score,
                    "accepted": cleanup_accepted,
                    "best_cleanup_sweep": cleanup_best["sweep"] if cleanup_best else None,
                    "stop_reason": cleanup_stop_reason,
                    "elapsed_seconds": time.perf_counter() - run_start_time,
                })

            d_min += WINDOW_SHIFT
            d_max += WINDOW_SHIFT
            window_idx += 1
            cp.get_default_memory_pool().free_all_blocks()

    best_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_best")
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    win.save_solution(best["exps"], best["u"], best["v"], best["p"], best_prefix)
    win.save_solution(cum_exps, u_cum.get(), v_cum.get(), p_cum.get(), final_prefix)

    summary = {
        "mode": "hybrid_moving_window_periodic_global",
        "metrics_csv": METRICS_CSV,
        "phases": phases,
        "best": {k: v for k, v in best.items() if k not in {"exps", "u", "v", "p"}},
        "best_solution_prefix": best_prefix,
        "final_solution_prefix": final_prefix,
        "settings": {
            "start_min_degree": START_MIN_DEGREE,
            "start_max_degree": START_MAX_DEGREE,
            "max_window_max_degree": MAX_WINDOW_MAX_DEGREE,
            "window_shift": WINDOW_SHIFT,
            "min_sweeps_per_window": MIN_SWEEPS_PER_WINDOW,
            "max_sweeps_per_window": MAX_SWEEPS_PER_WINDOW,
            "plateau_window": PLATEAU_WINDOW,
            "min_rel_improvement": MIN_REL_IMPROVEMENT,
            "global_refit_every_n_windows": GLOBAL_REFIT_EVERY_N_WINDOWS,
            "global_refit_sweeps": GLOBAL_REFIT_SWEEPS,
            "global_accept_worse_tol": GLOBAL_ACCEPT_WORSE_TOL,
            "global_score_explosion_factor": GLOBAL_SCORE_EXPLOSION_FACTOR,
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
