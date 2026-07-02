"""
compare_uvp_stage_runs.py
=========================
Run staged u/v/p polynomial fits one after another and save per-sweep
diagnostics for plotting.

The compared schedules are:

    5_to_8 : degree 5, then degree 8
    8_to_9 : degree 8, then degree 9

Each row is written to CSV immediately after a sweep, so partial results are
still available if a long CUDA run is stopped.
"""

import csv
import json
import os

import cupy as cp

import physics_only_uv_unknown_pressure as uvp


RUN_CONFIGS = [
    ("5_to_8", [(5, 115), (8, 250)]),
    ("8_to_9", [(8, 115), (9, 250)]),
]

OUT_DIR = "poly_saves"
METRICS_CSV = os.path.join(OUT_DIR, "uvp_stage_compare_5_to_8_vs_8_to_9.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "uvp_stage_compare_5_to_8_vs_8_to_9_summary.json")


CSV_FIELDS = [
    "run",
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
    "u_phys_data",
    "u_phys_div",
    "u_phys_mx",
    "u_phys_my",
    "v_phys_data",
    "v_phys_div",
    "v_phys_mx",
    "v_phys_my",
    "p_phys_mx",
    "p_phys_my",
    "is_best",
]


def row_from_sweep(
    run_name,
    global_sweep,
    stage_idx,
    stage_sweep,
    degree,
    n_terms,
    diag,
    score,
    steps,
    u_phys,
    v_phys,
    p_phys,
    is_best,
):
    return {
        "run": run_name,
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
        "u_step": steps["u"],
        "p_after_u_step": steps["p_after_u"],
        "v_step": steps["v"],
        "p_after_v_step": steps["p_after_v"],
        "u_phys_data": u_phys["data"],
        "u_phys_div": u_phys["div"],
        "u_phys_mx": u_phys["mom_x"],
        "u_phys_my": u_phys["mom_y"],
        "v_phys_data": v_phys["data"],
        "v_phys_div": v_phys["div"],
        "v_phys_mx": v_phys["mom_x"],
        "v_phys_my": v_phys["mom_y"],
        "p_phys_mx": p_phys["mom_x"],
        "p_phys_my": p_phys["mom_y"],
        "is_best": int(is_best),
    }


def run_schedule(run_name, degree_stages, data, writer, csv_file):
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = data

    u_coeffs = None
    v_coeffs = None
    p_coeffs = None
    old_exps = None
    best = None
    rows = []
    global_sweep = 0

    print(f"\n{'=' * 72}")
    print(f"Run {run_name}: stages={degree_stages}")
    print(f"{'=' * 72}")

    for stage_idx, (degree, n_sweeps) in enumerate(degree_stages):
        exps = uvp.make_total_degree_exponents(3, degree)
        deriv_cache = uvp.make_deriv_cache(exps)
        n_terms = exps.shape[0]

        if u_coeffs is None:
            u_coeffs = cp.zeros(n_terms, dtype=cp.float64)
            v_coeffs = cp.zeros(n_terms, dtype=cp.float64)
            p_coeffs = cp.zeros(n_terms, dtype=cp.float64)
            print(f"\nStage {stage_idx}: degree <= {degree}, terms={n_terms} per field")
        else:
            u_coeffs, copied_u = uvp.expand_coefficients(old_exps, u_coeffs, exps)
            v_coeffs, copied_v = uvp.expand_coefficients(old_exps, v_coeffs, exps)
            p_coeffs, copied_p = uvp.expand_coefficients(old_exps, p_coeffs, exps)
            print(
                f"\nStage {stage_idx}: expanded degree <= {degree}, terms={n_terms} per field; "
                f"copied u/v/p terms={copied_u}/{copied_v}/{copied_p}"
            )
        old_exps = exps

        for stage_sweep in range(n_sweeps):
            u_step, u_phys = uvp.solve_update(
                uvp.accumulate_u_update(X, u_true, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                u_coeffs,
            )
            p_after_u_step, _p_after_u_phys = uvp.solve_update(
                uvp.accumulate_pressure_update(X, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                p_coeffs,
            )
            v_step, v_phys = uvp.solve_update(
                uvp.accumulate_v_update(X, v_true, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                v_coeffs,
            )
            p_after_v_step, p_after_v_phys = uvp.solve_update(
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
            is_best = best is None or score < best["score"]
            if is_best:
                best = {
                    "run": run_name,
                    "score": score,
                    "global_sweep": global_sweep,
                    "stage": stage_idx,
                    "stage_sweep": stage_sweep,
                    "degree": degree,
                    "n_terms": n_terms,
                    "diag": dict(diag),
                    "exps": exps.copy(),
                    "u": u_cpu.copy(),
                    "v": v_cpu.copy(),
                    "p": p_cpu.copy(),
                }

            row = row_from_sweep(
                run_name,
                global_sweep,
                stage_idx,
                stage_sweep,
                degree,
                n_terms,
                diag,
                score,
                {
                    "u": u_step,
                    "p_after_u": p_after_u_step,
                    "v": v_step,
                    "p_after_v": p_after_v_step,
                },
                u_phys,
                v_phys,
                p_after_v_phys,
                is_best,
            )
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

            best_note = " best" if is_best else ""
            print(
                f"{run_name} sweep {global_sweep:03d} stage={stage_idx} "
                f"local={stage_sweep:03d} degree={degree}: "
                f"u={diag['u']:.6e} v={diag['v']:.6e} p={diag['p']:.6e} "
                f"px={diag['px']:.6e} py={diag['py']:.6e} score={score:.6e}{best_note}"
            )
            global_sweep += 1

    out_prefix = os.path.join(OUT_DIR, f"uvp_stage_compare_{run_name}_best")
    uvp.save_solution(best["exps"], best["u"], best["v"], best["p"], out_prefix)
    final_prefix = os.path.join(OUT_DIR, f"uvp_stage_compare_{run_name}_final")
    uvp.save_solution(exps, u_coeffs.get(), v_coeffs.get(), p_coeffs.get(), final_prefix)

    print(
        f"Run {run_name} best: sweep={best['global_sweep']} degree={best['degree']} "
        f"score={best['score']:.6e} diag={best['diag']}"
    )
    return {
        "name": run_name,
        "degree_stages": degree_stages,
        "n_rows": len(rows),
        "best": {
            key: value
            for key, value in best.items()
            if key not in {"exps", "u", "v", "p"}
        },
        "best_solution_prefix": out_prefix,
        "final_solution_prefix": final_prefix,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = uvp.generate_taylor_green(uvp.TOTAL_POINTS, uvp.NU)

    summaries = []
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        for run_name, degree_stages in RUN_CONFIGS:
            summaries.append(run_schedule(run_name, degree_stages, data, writer, csv_file))
            cp.get_default_memory_pool().free_all_blocks()

    summary_doc = {
        "metrics_csv": METRICS_CSV,
        "runs": summaries,
        "weights": {
            "data": uvp.DATA_WEIGHT,
            "div": uvp.DIV_WEIGHT,
            "mom_x": uvp.MOM_X_WEIGHT,
            "mom_y": uvp.MOM_Y_WEIGHT,
            "score_pressure_grad": uvp.SCORE_PRESSURE_GRAD_WEIGHT,
        },
        "total_points": uvp.TOTAL_POINTS,
        "nu": uvp.NU,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2)

    print(f"\nSaved per-sweep metrics to {METRICS_CSV}")
    print(f"Saved comparison summary to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
