"""
adaptive_uvp_agent_window.py
============================
Agent-like adaptive moving-window fitting for unknown u, v, and pressure.

The cumulative model keeps all learned coefficients.  Each training phase
updates only a selected degree window.  When a phase finishes, a small
deterministic "agent" tries several candidate next windows, including windows
that reach back down to lower-degree monomials.  A candidate is accepted if a
short trial improves the score, or if it does not worsen the score too much.

This tests the idea:

    "Can a moving window occasionally re-open lower powers when doing so is
     useful, without paying for a full global update all the time?"

Per-sweep and per-trial diagnostics are written immediately to CSV.
"""

import csv
import json
import math
import os
import time

import cupy as cp
import numpy as np

import physics_only_uvp_window_boost as win


START_MIN_DEGREE = 0
START_MAX_DEGREE = 5
MAX_WINDOW_MAX_DEGREE = 15
WINDOW_SHIFT = 1

MIN_SWEEPS_PER_WINDOW = 30
MAX_SWEEPS_PER_WINDOW = 160
PLATEAU_WINDOW = 20
MIN_REL_IMPROVEMENT = 0.005

TRIAL_SWEEPS = 5
ACCEPT_WORSE_TOL = 0.02
MAX_BACKTRACK = 3

OUT_DIR = "poly_saves"
OUT_BASENAME = "adaptive_uvp_agent_window"
METRICS_CSV = os.path.join(OUT_DIR, OUT_BASENAME + ".csv")
TRIALS_CSV = os.path.join(OUT_DIR, OUT_BASENAME + "_trials.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, OUT_BASENAME + "_summary.json")


METRIC_FIELDS = [
    "global_sweep",
    "phase",
    "phase_index",
    "phase_sweep",
    "action",
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


TRIAL_FIELDS = [
    "decision_index",
    "action",
    "degree_min",
    "degree_max",
    "trial_sweeps",
    "base_score",
    "trial_score",
    "score_ratio",
    "accepted",
    "selected",
    "reason",
    "elapsed_seconds",
]


def plateau_reached(scores):
    if len(scores) < MIN_SWEEPS_PER_WINDOW or len(scores) < 2 * PLATEAU_WINDOW:
        return False
    previous_best = min(scores[:-PLATEAU_WINDOW])
    recent_best = min(scores[-PLATEAU_WINDOW:])
    rel_improvement = (previous_best - recent_best) / max(abs(previous_best), 1e-300)
    return rel_improvement < MIN_REL_IMPROVEMENT


def base_state(cum_exps, u_cum, v_cum, p_cum):
    cache = win.make_deriv_cache(cum_exps)
    return (
        (cum_exps, cache, u_cum),
        (cum_exps, cache, v_cum),
        (cum_exps, cache, p_cum),
    )


def evaluate_combined(X, data, cum_exps, u_cum, v_cum, p_cum, win_exps, u_win, v_win, p_win):
    u_true, v_true, p_true, pg_x_true, pg_y_true = data
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
    return tmp_exps, tmp_u, tmp_v, tmp_p, diag, win.score_diagnostics(diag)


def evaluate_cumulative(X, data, cum_exps, u_cum, v_cum, p_cum):
    empty = np.zeros((0, 3), dtype=np.uint16)
    z = cp.zeros(0, dtype=cp.float64)
    return evaluate_combined(X, data, cum_exps, u_cum, v_cum, p_cum, empty, z, z, z)


def run_one_window_sweep(X, u_true, v_true, base, win_exps, win_cache, u_win, v_win, p_win):
    u_step, _u_phys = win.solve_update(
        win.accumulate_u_update(X, u_true, base, win_exps, win_cache, u_win, v_win, p_win),
        u_win,
    )
    p_after_u_step, _p_after_u_phys = win.solve_update(
        win.accumulate_pressure_update(X, base, win_exps, win_cache, u_win, v_win, p_win),
        p_win,
    )
    v_step, _v_phys = win.solve_update(
        win.accumulate_v_update(X, v_true, base, win_exps, win_cache, u_win, v_win, p_win),
        v_win,
    )
    p_after_v_step, _p_after_v_phys = win.solve_update(
        win.accumulate_pressure_update(X, base, win_exps, win_cache, u_win, v_win, p_win),
        p_win,
    )
    return u_step, p_after_u_step, v_step, p_after_v_step


def update_best(best, score, global_sweep, phase, phase_index, phase_sweep,
                action, degree_window, diag, exps, u, v, p):
    if best is not None and score >= best["score"]:
        return best, False
    return {
        "score": score,
        "global_sweep": global_sweep,
        "phase": phase,
        "phase_index": phase_index,
        "phase_sweep": phase_sweep,
        "action": action,
        "degree_window": degree_window,
        "diag": dict(diag),
        "exps": exps.copy(),
        "u": u.get() if isinstance(u, cp.ndarray) else np.asarray(u).copy(),
        "v": v.get() if isinstance(v, cp.ndarray) else np.asarray(v).copy(),
        "p": p.get() if isinstance(p, cp.ndarray) else np.asarray(p).copy(),
    }, True


def candidate_windows(current_min, current_max):
    next_max = current_max + WINDOW_SHIFT
    normal_min = current_min + WINDOW_SHIFT
    candidates = [("shift", normal_min, next_max)]
    for back in range(1, MAX_BACKTRACK + 1):
        candidates.append((f"backtrack_{back}", max(0, normal_min - back), next_max))
    candidates.append(("keep_low_edge", current_min, next_max))

    unique = []
    seen = set()
    for action, d_min, d_max in candidates:
        if d_max > MAX_WINDOW_MAX_DEGREE:
            continue
        key = (d_min, d_max)
        if key not in seen:
            seen.add(key)
            unique.append((action, d_min, d_max))
    return unique


def trial_candidate(X, data, cum_exps, u_cum, v_cum, p_cum, action, d_min, d_max):
    u_true, v_true, _p_true, _pg_x_true, _pg_y_true = data
    win_exps = win.make_degree_window_exponents(3, d_min, d_max)
    win_cache = win.make_deriv_cache(win_exps)
    u_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
    v_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
    p_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
    base = base_state(cum_exps, u_cum, v_cum, p_cum)

    steps = (0.0, 0.0, 0.0, 0.0)
    for _ in range(TRIAL_SWEEPS):
        steps = run_one_window_sweep(
            X, u_true, v_true, base, win_exps, win_cache, u_win, v_win, p_win
        )

    tmp_exps, tmp_u, tmp_v, tmp_p, diag, score = evaluate_combined(
        X, data, cum_exps, u_cum, v_cum, p_cum, win_exps, u_win, v_win, p_win
    )
    return {
        "action": action,
        "degree_min": d_min,
        "degree_max": d_max,
        "win_exps": win_exps,
        "win_cache": win_cache,
        "u_win": u_win,
        "v_win": v_win,
        "p_win": p_win,
        "tmp_exps": tmp_exps,
        "tmp_u": tmp_u,
        "tmp_v": tmp_v,
        "tmp_p": tmp_p,
        "diag": diag,
        "score": score,
        "steps": steps,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = win.generate_taylor_green(
        win.TOTAL_POINTS, win.NU
    )
    data = (u_true, v_true, p_true, pg_x_true, pg_y_true)

    cum_exps = np.zeros((0, 3), dtype=np.uint16)
    u_cum = cp.zeros(0, dtype=cp.float64)
    v_cum = cp.zeros(0, dtype=cp.float64)
    p_cum = cp.zeros(0, dtype=cp.float64)

    best = None
    phases = []
    decisions = []
    global_sweep = 0
    phase_idx = 0
    d_min = START_MIN_DEGREE
    d_max = START_MAX_DEGREE
    action = "initial"
    seed_trial = None
    run_start = time.perf_counter()

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as metrics_file, \
            open(TRIALS_CSV, "w", newline="", encoding="utf-8") as trials_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        trials_writer = csv.DictWriter(trials_file, fieldnames=TRIAL_FIELDS)
        metrics_writer.writeheader()
        trials_writer.writeheader()
        metrics_file.flush()
        trials_file.flush()

        while d_max <= MAX_WINDOW_MAX_DEGREE:
            if seed_trial is None:
                win_exps = win.make_degree_window_exponents(3, d_min, d_max)
                win_cache = win.make_deriv_cache(win_exps)
                u_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
                v_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
                p_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
                start_local = 0
            else:
                win_exps = seed_trial["win_exps"]
                win_cache = seed_trial["win_cache"]
                u_win = seed_trial["u_win"]
                v_win = seed_trial["v_win"]
                p_win = seed_trial["p_win"]
                start_local = TRIAL_SWEEPS
                seed_trial = None

            print(
                f"\nAgent phase {phase_idx}: action={action} degree {d_min}..{d_max}, "
                f"terms={win_exps.shape[0]}, cumulative={cum_exps.shape[0]}"
            )

            scores = []
            phase_start = global_sweep
            for local in range(start_local, MAX_SWEEPS_PER_WINDOW):
                sweep_start = time.perf_counter()
                base = base_state(cum_exps, u_cum, v_cum, p_cum)
                u_step, p_after_u_step, v_step, p_after_v_step = run_one_window_sweep(
                    X, u_true, v_true, base, win_exps, win_cache, u_win, v_win, p_win
                )
                tmp_exps, tmp_u, tmp_v, tmp_p, diag, score = evaluate_combined(
                    X, data, cum_exps, u_cum, v_cum, p_cum, win_exps, u_win, v_win, p_win
                )
                scores.append(score)
                sweep_seconds = time.perf_counter() - sweep_start
                elapsed_seconds = time.perf_counter() - run_start
                best, is_best = update_best(
                    best, score, global_sweep, "window", phase_idx, local,
                    action, (d_min, d_max), diag, tmp_exps, tmp_u, tmp_v, tmp_p
                )

                metrics_writer.writerow({
                    "global_sweep": global_sweep,
                    "phase": "window",
                    "phase_index": phase_idx,
                    "phase_sweep": local,
                    "action": action,
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
                metrics_file.flush()

                print(
                    f"Agent sweep {global_sweep:03d} phase={phase_idx} local={local:03d} "
                    f"action={action} degree={d_min}..{d_max}: "
                    f"u={diag['u']:.6e} v={diag['v']:.6e} p={diag['p']:.6e} "
                    f"px={diag['px']:.6e} py={diag['py']:.6e} score={score:.6e} "
                    f"time={sweep_seconds:.1f}s" + (" best" if is_best else "")
                )
                global_sweep += 1

                if plateau_reached(scores):
                    print(f"Agent phase {phase_idx} plateau reached.")
                    break
                if not math.isfinite(score):
                    print(f"Agent phase {phase_idx} stopped on non-finite score.")
                    break

            next_exps, next_u = win.merge_coefficients(cum_exps, u_cum, win_exps, u_win)
            _next_exps, next_v = win.merge_coefficients(cum_exps, v_cum, win_exps, v_win)
            _next_exps, next_p = win.merge_coefficients(cum_exps, p_cum, win_exps, p_win)
            cum_exps, u_cum, v_cum, p_cum = next_exps, next_u, next_v, next_p
            phases.append({
                "phase": phase_idx,
                "action": action,
                "degree_min": d_min,
                "degree_max": d_max,
                "start_sweep": phase_start,
                "n_sweeps": max(global_sweep - phase_start, 0),
                "best_score": min(scores) if scores else None,
                "last_score": scores[-1] if scores else None,
                "cumulative_terms": cum_exps.shape[0],
                "elapsed_seconds": time.perf_counter() - run_start,
            })

            _base_exps, _base_u, _base_v, _base_p, base_diag, base_score = evaluate_cumulative(
                X, data, cum_exps, u_cum, v_cum, p_cum
            )

            candidates = candidate_windows(d_min, d_max)
            if not candidates:
                break

            trial_results = []
            for cand_action, cand_min, cand_max in candidates:
                trial = trial_candidate(
                    X, data, cum_exps, u_cum, v_cum, p_cum,
                    cand_action, cand_min, cand_max,
                )
                ratio = trial["score"] / max(abs(base_score), 1e-300)
                accepted = ratio <= 1.0 + ACCEPT_WORSE_TOL
                reason = "accepted" if accepted else "too_much_worse"
                trial_results.append((trial, ratio, accepted, reason))

            accepted_trials = [item for item in trial_results if item[2]]
            if accepted_trials:
                selected, selected_ratio, _accepted, selected_reason = min(
                    accepted_trials, key=lambda item: item[0]["score"]
                )
            else:
                selected, selected_ratio, _accepted, selected_reason = min(
                    trial_results, key=lambda item: item[0]["score"]
                )
                selected_reason = "forced_best_rejected_set"

            decision_idx = len(decisions)
            for trial, ratio, accepted, reason in trial_results:
                selected_flag = trial is selected
                trials_writer.writerow({
                    "decision_index": decision_idx,
                    "action": trial["action"],
                    "degree_min": trial["degree_min"],
                    "degree_max": trial["degree_max"],
                    "trial_sweeps": TRIAL_SWEEPS,
                    "base_score": base_score,
                    "trial_score": trial["score"],
                    "score_ratio": ratio,
                    "accepted": int(accepted),
                    "selected": int(selected_flag),
                    "reason": selected_reason if selected_flag else reason,
                    "elapsed_seconds": time.perf_counter() - run_start,
                })
            trials_file.flush()

            decisions.append({
                "decision": decision_idx,
                "base_score": base_score,
                "selected_action": selected["action"],
                "selected_degree_min": selected["degree_min"],
                "selected_degree_max": selected["degree_max"],
                "selected_score": selected["score"],
                "selected_ratio": selected_ratio,
                "reason": selected_reason,
            })

            print(
                f"Agent decision {decision_idx}: base_score={base_score:.6e}, "
                f"selected={selected['action']} {selected['degree_min']}..{selected['degree_max']} "
                f"trial_score={selected['score']:.6e} ratio={selected_ratio:.4f} "
                f"reason={selected_reason}"
            )

            d_min = selected["degree_min"]
            d_max = selected["degree_max"]
            action = selected["action"]
            seed_trial = selected
            phase_idx += 1
            cp.get_default_memory_pool().free_all_blocks()

    best_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_best")
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    win.save_solution(best["exps"], best["u"], best["v"], best["p"], best_prefix)
    win.save_solution(cum_exps, u_cum.get(), v_cum.get(), p_cum.get(), final_prefix)

    summary = {
        "mode": "agent_adaptive_moving_window",
        "metrics_csv": METRICS_CSV,
        "trials_csv": TRIALS_CSV,
        "phases": phases,
        "decisions": decisions,
        "best": {k: v for k, v in best.items() if k not in {"exps", "u", "v", "p"}},
        "best_solution_prefix": best_prefix,
        "final_solution_prefix": final_prefix,
        "settings": {
            "start_min_degree": START_MIN_DEGREE,
            "start_max_degree": START_MAX_DEGREE,
            "max_window_max_degree": MAX_WINDOW_MAX_DEGREE,
            "window_shift": WINDOW_SHIFT,
            "max_backtrack": MAX_BACKTRACK,
            "trial_sweeps": TRIAL_SWEEPS,
            "accept_worse_tol": ACCEPT_WORSE_TOL,
            "min_sweeps_per_window": MIN_SWEEPS_PER_WINDOW,
            "max_sweeps_per_window": MAX_SWEEPS_PER_WINDOW,
            "plateau_window": PLATEAU_WINDOW,
            "min_rel_improvement": MIN_REL_IMPROVEMENT,
        },
        "elapsed_seconds": time.perf_counter() - run_start,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved metrics to {METRICS_CSV}")
    print(f"Saved trials to {TRIALS_CSV}")
    print(f"Saved summary to {SUMMARY_JSON}")
    print(f"Saved best solution to {best_prefix}.json/.txt")
    print(f"Saved final solution to {final_prefix}.json/.txt")


if __name__ == "__main__":
    main()
