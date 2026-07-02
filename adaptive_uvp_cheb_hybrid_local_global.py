"""Hybrid moving-window/global-cleanup u/v/p fit using Chebyshev basis."""

import csv
import json
import os
import time

import cupy as cp
import numpy as np

import physics_only_uv_chebyshev as cheb


START_MIN_DEGREE = 0
START_MAX_DEGREE = 5
MAX_WINDOW_MAX_DEGREE = 15
WINDOW_SHIFT = 1
MIN_SWEEPS_PER_WINDOW = 30
MAX_SWEEPS_PER_WINDOW = 180
PLATEAU_WINDOW = 20
MIN_REL_IMPROVEMENT = 0.005
GLOBAL_REFIT_EVERY_N_WINDOWS = 2
GLOBAL_REFIT_SWEEPS = 35

OUT_DIR = "poly_saves"
OUT_BASENAME = "adaptive_uvp_cheb_hybrid_local_global"
METRICS_CSV = os.path.join(OUT_DIR, OUT_BASENAME + ".csv")
SUMMARY_JSON = os.path.join(OUT_DIR, OUT_BASENAME + "_summary.json")

CSV_FIELDS = ["global_sweep", "phase", "phase_index", "phase_sweep", "degree_min", "degree_max", "active_terms", "cumulative_terms",
              "u_rmse", "v_rmse", "p_rmse", "px_rmse", "py_rmse", "score",
              "u_step", "p_after_u_step", "v_step", "p_after_v_step", "sweep_seconds", "elapsed_seconds", "is_best"]


def plateau_reached(scores):
    if len(scores) < MIN_SWEEPS_PER_WINDOW or len(scores) < 2 * PLATEAU_WINDOW:
        return False
    previous_best = min(scores[:-PLATEAU_WINDOW])
    recent_best = min(scores[-PLATEAU_WINDOW:])
    rel_improvement = (previous_best - recent_best) / max(abs(previous_best), 1e-300)
    return rel_improvement < MIN_REL_IMPROVEMENT


def update_best(best, score, global_sweep, phase, phase_index, phase_sweep, degree_window, diag, exps, u, v, p):
    if best is not None and score >= best["score"]:
        return best, False
    return {"score": score, "global_sweep": global_sweep, "phase": phase, "phase_index": phase_index,
            "phase_sweep": phase_sweep, "degree_window": degree_window, "diag": dict(diag), "exps": exps.copy(),
            "u": u.get() if isinstance(u, cp.ndarray) else np.asarray(u).copy(),
            "v": v.get() if isinstance(v, cp.ndarray) else np.asarray(v).copy(),
            "p": p.get() if isinstance(p, cp.ndarray) else np.asarray(p).copy()}, True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = cheb.generate_taylor_green(cheb.TOTAL_POINTS, cheb.NU)

    cum_exps = np.zeros((0, 3), dtype=np.uint16)
    u_cum = cp.zeros(0, dtype=cp.float64)
    v_cum = cp.zeros(0, dtype=cp.float64)
    p_cum = cp.zeros(0, dtype=cp.float64)
    best = None
    phases = []
    global_sweep = 0
    window_idx = 0
    d_min = START_MIN_DEGREE
    d_max = START_MAX_DEGREE
    run_start = time.perf_counter()

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader(); csv_file.flush()

        while d_max <= MAX_WINDOW_MAX_DEGREE:
            win_exps = cheb.make_degree_window_exponents(3, d_min, d_max)
            u_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
            v_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
            p_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
            scores = []
            phase_start = global_sweep
            print(f"\nCheb window {window_idx}: degree {d_min}..{d_max}, terms={win_exps.shape[0]}, cumulative={cum_exps.shape[0]}")

            for local in range(MAX_SWEEPS_PER_WINDOW):
                sweep_start = time.perf_counter()
                base_state = ((cum_exps, u_cum), (cum_exps, v_cum), (cum_exps, p_cum))
                u_step, _ = cheb.solve_update(cheb.accumulate_u_update_window(X, u_true, base_state, win_exps, u_win, v_win, p_win), u_win)
                p_after_u_step, _ = cheb.solve_update(cheb.accumulate_pressure_update_window(X, base_state, win_exps, u_win, v_win, p_win), p_win)
                v_step, _ = cheb.solve_update(cheb.accumulate_v_update_window(X, v_true, base_state, win_exps, u_win, v_win, p_win), v_win)
                p_after_v_step, _ = cheb.solve_update(cheb.accumulate_pressure_update_window(X, base_state, win_exps, u_win, v_win, p_win), p_win)

                tmp_exps, tmp_u = cheb.merge_coefficients(cum_exps, u_cum, win_exps, u_win)
                _tmp_exps, tmp_v = cheb.merge_coefficients(cum_exps, v_cum, win_exps, v_win)
                _tmp_exps, tmp_p = cheb.merge_coefficients(cum_exps, p_cum, win_exps, p_win)
                u_cpu, v_cpu, p_cpu = tmp_u.get(), tmp_v.get(), tmp_p.get()
                diag = cheb.diagnostics(X, tmp_exps, u_cpu, v_cpu, p_cpu, u_true, v_true, p_true, pg_x_true, pg_y_true)
                score = cheb.score_diagnostics(diag)
                scores.append(score)
                sweep_seconds = time.perf_counter() - sweep_start
                elapsed_seconds = time.perf_counter() - run_start
                best, is_best = update_best(best, score, global_sweep, "window", window_idx, local, (d_min, d_max), diag, tmp_exps, tmp_u, tmp_v, tmp_p)
                writer.writerow({"global_sweep": global_sweep, "phase": "window", "phase_index": window_idx, "phase_sweep": local,
                                 "degree_min": d_min, "degree_max": d_max, "active_terms": win_exps.shape[0], "cumulative_terms": tmp_exps.shape[0],
                                 "u_rmse": diag["u"], "v_rmse": diag["v"], "p_rmse": diag["p"], "px_rmse": diag["px"], "py_rmse": diag["py"], "score": score,
                                 "u_step": u_step, "p_after_u_step": p_after_u_step, "v_step": v_step, "p_after_v_step": p_after_v_step,
                                 "sweep_seconds": sweep_seconds, "elapsed_seconds": elapsed_seconds, "is_best": int(is_best)})
                csv_file.flush()
                print(f"Cheb sweep {global_sweep:03d} window={window_idx} local={local:03d} degree={d_min}..{d_max}: "
                      f"u={diag['u']:.6e} v={diag['v']:.6e} p={diag['p']:.6e} px={diag['px']:.6e} py={diag['py']:.6e} "
                      f"score={score:.6e} time={sweep_seconds:.1f}s" + (" best" if is_best else ""))
                global_sweep += 1
                if plateau_reached(scores):
                    print(f"Cheb window {window_idx} plateau reached."); break

            next_exps, next_u = cheb.merge_coefficients(cum_exps, u_cum, win_exps, u_win)
            _next_exps, next_v = cheb.merge_coefficients(cum_exps, v_cum, win_exps, v_win)
            _next_exps, next_p = cheb.merge_coefficients(cum_exps, p_cum, win_exps, p_win)
            cum_exps, u_cum, v_cum, p_cum = next_exps, next_u, next_v, next_p
            phases.append({"phase": "window", "index": window_idx, "degree_min": d_min, "degree_max": d_max,
                           "start_sweep": phase_start, "n_sweeps": len(scores), "best_score": min(scores), "last_score": scores[-1],
                           "elapsed_seconds": time.perf_counter() - run_start})

            if (window_idx + 1) % GLOBAL_REFIT_EVERY_N_WINDOWS == 0 or d_max >= MAX_WINDOW_MAX_DEGREE:
                full_degree = d_max
                full_exps = cheb.make_total_degree_exponents(3, full_degree)
                u_full, _ = cheb.expand_coefficients(cum_exps, u_cum, full_exps)
                v_full, _ = cheb.expand_coefficients(cum_exps, v_cum, full_exps)
                p_full, _ = cheb.expand_coefficients(cum_exps, p_cum, full_exps)
                cleanup_scores = []
                cleanup_start = global_sweep
                print(f"\nCheb global cleanup after window {window_idx}: degree <= {full_degree}, terms={full_exps.shape[0]}")
                for local in range(GLOBAL_REFIT_SWEEPS):
                    sweep_start = time.perf_counter()
                    u_step, _ = cheb.solve_update(cheb.accumulate_u_update(X, u_true, full_exps, u_full, v_full, p_full), u_full)
                    p_after_u_step, _ = cheb.solve_update(cheb.accumulate_pressure_update(X, full_exps, u_full, v_full, p_full), p_full)
                    v_step, _ = cheb.solve_update(cheb.accumulate_v_update(X, v_true, full_exps, u_full, v_full, p_full), v_full)
                    p_after_v_step, _ = cheb.solve_update(cheb.accumulate_pressure_update(X, full_exps, u_full, v_full, p_full), p_full)
                    u_cpu, v_cpu, p_cpu = u_full.get(), v_full.get(), p_full.get()
                    diag = cheb.diagnostics(X, full_exps, u_cpu, v_cpu, p_cpu, u_true, v_true, p_true, pg_x_true, pg_y_true)
                    score = cheb.score_diagnostics(diag)
                    cleanup_scores.append(score)
                    sweep_seconds = time.perf_counter() - sweep_start
                    elapsed_seconds = time.perf_counter() - run_start
                    best, is_best = update_best(best, score, global_sweep, "global", window_idx, local, (0, full_degree), diag, full_exps, u_full, v_full, p_full)
                    writer.writerow({"global_sweep": global_sweep, "phase": "global", "phase_index": window_idx, "phase_sweep": local,
                                     "degree_min": 0, "degree_max": full_degree, "active_terms": full_exps.shape[0], "cumulative_terms": full_exps.shape[0],
                                     "u_rmse": diag["u"], "v_rmse": diag["v"], "p_rmse": diag["p"], "px_rmse": diag["px"], "py_rmse": diag["py"], "score": score,
                                     "u_step": u_step, "p_after_u_step": p_after_u_step, "v_step": v_step, "p_after_v_step": p_after_v_step,
                                     "sweep_seconds": sweep_seconds, "elapsed_seconds": elapsed_seconds, "is_best": int(is_best)})
                    csv_file.flush()
                    print(f"Cheb sweep {global_sweep:03d} global_after={window_idx} local={local:03d} degree<={full_degree}: "
                          f"u={diag['u']:.6e} v={diag['v']:.6e} p={diag['p']:.6e} px={diag['px']:.6e} py={diag['py']:.6e} "
                          f"score={score:.6e} time={sweep_seconds:.1f}s" + (" best" if is_best else ""))
                    global_sweep += 1
                cum_exps = full_exps
                u_cum, v_cum, p_cum = u_full, v_full, p_full
                phases.append({"phase": "global", "after_window": window_idx, "degree_min": 0, "degree_max": full_degree,
                               "start_sweep": cleanup_start, "n_sweeps": len(cleanup_scores), "best_score": min(cleanup_scores),
                               "last_score": cleanup_scores[-1], "elapsed_seconds": time.perf_counter() - run_start})

            d_min += WINDOW_SHIFT; d_max += WINDOW_SHIFT; window_idx += 1
            cp.get_default_memory_pool().free_all_blocks()

    best_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_best")
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    cheb.save_solution(best["exps"], best["u"], best["v"], best["p"], best_prefix)
    cheb.save_solution(cum_exps, u_cum.get(), v_cum.get(), p_cum.get(), final_prefix)
    summary = {"mode": "chebyshev_hybrid_moving_window_periodic_global", "metrics_csv": METRICS_CSV, "phases": phases,
               "best": {k: v for k, v in best.items() if k not in {"exps", "u", "v", "p"}},
               "best_solution_prefix": best_prefix, "final_solution_prefix": final_prefix,
               "settings": {"start_min_degree": START_MIN_DEGREE, "start_max_degree": START_MAX_DEGREE,
                            "max_window_max_degree": MAX_WINDOW_MAX_DEGREE, "window_shift": WINDOW_SHIFT,
                            "global_refit_every_n_windows": GLOBAL_REFIT_EVERY_N_WINDOWS, "global_refit_sweeps": GLOBAL_REFIT_SWEEPS},
               "elapsed_seconds": time.perf_counter() - run_start}
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Cheb hybrid metrics to {METRICS_CSV}")


if __name__ == "__main__":
    main()
