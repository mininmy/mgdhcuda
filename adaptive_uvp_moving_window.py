"""
adaptive_uvp_moving_window.py
=============================
Adaptive moving-window polynomial fitting for unknown u, v, and pressure.

The cumulative model keeps everything already learned.  New updates are
restricted to a moving total-degree window:

    degrees 0..5  ->  1..6  ->  2..7  -> ...

After the score stagnates, the current correction is merged into the
cumulative model.  The next window increases by WINDOW_SHIFT and drops the
lowest-degree monomials from future updates.

Per-sweep diagnostics are written immediately to CSV for plotting.
"""

import csv
import json
import os

import cupy as cp
import numpy as np

import physics_only_uvp_window_boost as win


START_MIN_DEGREE = 0
START_MAX_DEGREE = 5
MAX_WINDOW_MAX_DEGREE = 10
WINDOW_SHIFT = 1

MIN_SWEEPS_PER_WINDOW = 30
MAX_SWEEPS_PER_WINDOW = 250
PLATEAU_WINDOW = 20
MIN_REL_IMPROVEMENT = 0.005

OUT_DIR = "poly_saves"
OUT_BASENAME = "adaptive_uvp_moving_window"
METRICS_CSV = os.path.join(OUT_DIR, OUT_BASENAME + ".csv")
SUMMARY_JSON = os.path.join(OUT_DIR, OUT_BASENAME + "_summary.json")


CSV_FIELDS = [
    "global_sweep",
    "window",
    "window_sweep",
    "degree_min",
    "degree_max",
    "window_terms",
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


def empty_base_state():
    exps = np.zeros((0, 3), dtype=np.uint16)
    cache = win.make_deriv_cache(exps)
    coeffs = cp.zeros(0, dtype=cp.float64)
    return exps, cache, coeffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = win.generate_taylor_green(
        win.TOTAL_POINTS, win.NU
    )

    cum_exps, empty_cache, u_cum = empty_base_state()
    v_cum = cp.zeros(0, dtype=cp.float64)
    p_cum = cp.zeros(0, dtype=cp.float64)

    best = None
    summary_windows = []
    global_sweep = 0
    window_idx = 0
    d_min = START_MIN_DEGREE
    d_max = START_MAX_DEGREE

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

            print(
                f"\nWindow {window_idx}: degree {d_min}..{d_max}, "
                f"terms={win_exps.shape[0]} per correction, "
                f"cumulative_terms={cum_exps.shape[0]}"
            )

            window_scores = []
            window_start_sweep = global_sweep
            for window_sweep in range(MAX_SWEEPS_PER_WINDOW):
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

                is_best = best is None or score < best["score"]
                if is_best:
                    best = {
                        "score": score,
                        "global_sweep": global_sweep,
                        "window": window_idx,
                        "window_sweep": window_sweep,
                        "degree_window": (d_min, d_max),
                        "diag": dict(diag),
                        "exps": tmp_exps.copy(),
                        "u": u_cpu.copy(),
                        "v": v_cpu.copy(),
                        "p": p_cpu.copy(),
                    }

                write_row(writer, csv_file, {
                    "global_sweep": global_sweep,
                    "window": window_idx,
                    "window_sweep": window_sweep,
                    "degree_min": d_min,
                    "degree_max": d_max,
                    "window_terms": win_exps.shape[0],
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
                    "is_best": int(is_best),
                })

                best_note = " best" if is_best else ""
                print(
                    f"Sweep {global_sweep:03d} window={window_idx} local={window_sweep:03d} "
                    f"degree={d_min}..{d_max}: u={diag['u']:.6e} v={diag['v']:.6e} "
                    f"p={diag['p']:.6e} px={diag['px']:.6e} py={diag['py']:.6e} "
                    f"score={score:.6e}{best_note}"
                )
                global_sweep += 1

                if plateau_reached(window_scores):
                    print(
                        f"Window {window_idx} plateau: best recent improvement below "
                        f"{MIN_REL_IMPROVEMENT:g} over {PLATEAU_WINDOW} sweeps."
                    )
                    break

            next_exps, next_u = win.merge_coefficients(cum_exps, u_cum, win_exps, u_win)
            _next_exps, next_v = win.merge_coefficients(cum_exps, v_cum, win_exps, v_win)
            _next_exps, next_p = win.merge_coefficients(cum_exps, p_cum, win_exps, p_win)
            cum_exps, u_cum, v_cum, p_cum = next_exps, next_u, next_v, next_p

            summary_windows.append({
                "window": window_idx,
                "degree_min": d_min,
                "degree_max": d_max,
                "start_sweep": window_start_sweep,
                "n_sweeps": len(window_scores),
                "best_score": min(window_scores),
                "last_score": window_scores[-1],
                "cumulative_terms": cum_exps.shape[0],
            })

            print(
                f"Window {window_idx} merged: cumulative_terms={cum_exps.shape[0]}, "
                f"next window shifts by {WINDOW_SHIFT}"
            )
            d_min += WINDOW_SHIFT
            d_max += WINDOW_SHIFT
            window_idx += 1
            cp.get_default_memory_pool().free_all_blocks()

    best_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_best")
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    win.save_solution(best["exps"], best["u"], best["v"], best["p"], best_prefix)
    win.save_solution(cum_exps, u_cum.get(), v_cum.get(), p_cum.get(), final_prefix)

    summary = {
        "mode": "adaptive_moving_degree_window",
        "metrics_csv": METRICS_CSV,
        "windows": summary_windows,
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
        },
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved metrics to {METRICS_CSV}")
    print(f"Saved summary to {SUMMARY_JSON}")
    print(f"Saved best solution to {best_prefix}.json/.txt")
    print(f"Saved final solution to {final_prefix}.json/.txt")


if __name__ == "__main__":
    main()
