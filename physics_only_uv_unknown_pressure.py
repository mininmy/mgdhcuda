"""
physics_only_uv_unknown_pressure.py
===================================
Fit u, v, and pressure with fixed polynomial bases.

This experiment does not use GMDH layers or boosting.  Velocity data is used
as an anchor for u and v, while pressure is estimated only through the
Navier-Stokes momentum residuals.  All coefficients start at zero and the
updates are alternated:

    update u
    update pressure
    update v
    update pressure

The pressure constant is not identifiable from pressure gradients, so the
reported pressure RMSE removes the mean error before measuring it.
"""

import json
import os

import cupy as cp
import numpy as np


NU = 0.01
TOTAL_POINTS = 1_000_000
DEGREE_STAGES = [
    (8, 115),
    (8, 250),
]
CHUNK_SIZE = 20_000
RIDGE_LAMBDA = 1e-10
SCORE_PRESSURE_GRAD_WEIGHT = 0.1
DATA_WEIGHT = 1.0
DIV_WEIGHT = 1.0
MOM_X_WEIGHT = 1.0
MOM_Y_WEIGHT = 1.0
OUT_DIR = "poly_saves"
OUT_BASENAME = "physics_only_uvp_unknown_pressure_staged_degree_5_to_8"
VAR_NAMES = ["x", "y", "t"]


def generate_taylor_green(total_points=1_000_000, nu=0.01):
    t_steps = 10
    n_spatial = int(np.sqrt(total_points / t_steps))
    x_r = np.linspace(-1, 1, n_spatial)
    y_r = np.linspace(-1, 1, n_spatial)
    t_r = np.linspace(0, 1.0, t_steps)
    X, Y, T = np.meshgrid(x_r, y_r, t_r, indexing="ij")
    xf, yf, tf = X.ravel(), Y.ravel(), T.ravel()

    decay = np.exp(-2.0 * nu * tf)
    decay4 = np.exp(-4.0 * nu * tf)

    u = np.sin(xf) * np.cos(yf) * decay
    v = -np.cos(xf) * np.sin(yf) * decay
    p = 0.25 * (np.cos(2.0 * xf) + np.cos(2.0 * yf)) * decay4
    pg_x = -0.5 * np.sin(2.0 * xf) * decay4
    pg_y = -0.5 * np.sin(2.0 * yf) * decay4

    return (
        np.stack([xf, yf, tf], axis=1),
        u.astype(np.float64),
        v.astype(np.float64),
        p.astype(np.float64),
        pg_x.astype(np.float64),
        pg_y.astype(np.float64),
    )


def make_total_degree_exponents(n_vars, max_degree):
    exps = []

    def rec(prefix, remaining_vars, remaining_degree):
        if remaining_vars == 1:
            for e in range(remaining_degree + 1):
                exps.append(prefix + [e])
            return
        for e in range(remaining_degree + 1):
            rec(prefix + [e], remaining_vars - 1, remaining_degree - e)

    rec([], n_vars, max_degree)
    exps.sort(key=lambda row: (sum(row), row))
    return np.array(exps, dtype=np.uint16)


def expand_coefficients(old_exps, old_coeffs, new_exps):
    old_cpu = old_coeffs.get() if isinstance(old_coeffs, cp.ndarray) else old_coeffs
    new_cpu = np.zeros(new_exps.shape[0], dtype=np.float64)
    old_index = {tuple(int(e) for e in exp): i for i, exp in enumerate(old_exps)}
    copied = 0
    for i, exp in enumerate(new_exps):
        old_i = old_index.get(tuple(int(e) for e in exp))
        if old_i is not None:
            new_cpu[i] = old_cpu[old_i]
            copied += 1
    return cp.asarray(new_cpu, dtype=cp.float64), copied


def score_diagnostics(diag):
    return (diag["u"] + diag["v"]
            + SCORE_PRESSURE_GRAD_WEIGHT * (diag["px"] + diag["py"]))


def degree_from_exponents(exps_cpu):
    return int(np.max(np.sum(exps_cpu, axis=1))) if len(exps_cpu) else 0


def derivative_exponents(exps_cpu, var_idx, order=1):
    exps = exps_cpu.astype(np.int32)
    powers = exps[:, var_idx]
    mult = np.ones(len(exps), dtype=np.float64)
    for k in range(order):
        mult *= np.maximum(powers - k, 0)
    out = exps.copy()
    out[:, var_idx] = np.maximum(out[:, var_idx] - order, 0)
    mult[powers < order] = 0.0
    return mult, out.astype(np.uint16)


def make_deriv_cache(exps_cpu):
    cache = {}
    for var_idx in range(3):
        cache[("d1", var_idx)] = derivative_exponents(exps_cpu, var_idx, 1)
        cache[("d2", var_idx)] = derivative_exponents(exps_cpu, var_idx, 2)
    return cache


def eval_monomials(X_gpu, exps_cpu):
    n_rows = X_gpu.shape[0]
    n_terms = exps_cpu.shape[0]
    A = cp.ones((n_rows, n_terms), dtype=cp.float64)
    exps_gpu = cp.asarray(exps_cpu, dtype=cp.uint16)
    for v in range(exps_cpu.shape[1]):
        powers = exps_gpu[:, v]
        max_power = int(powers.max().get()) if n_terms else 0
        for power in range(1, max_power + 1):
            cols = powers == power
            if bool(cols.any()):
                A[:, cols] *= X_gpu[:, v, None] ** power
    return A


def basis_bundle(X_gpu, exps_cpu, deriv_cache):
    A = eval_monomials(X_gpu, exps_cpu)
    derivs = []
    for var_idx in range(3):
        mult, dexps = deriv_cache[("d1", var_idx)]
        derivs.append(eval_monomials(X_gpu, dexps) * cp.asarray(mult)[None, :])

    lap = cp.zeros_like(A)
    for var_idx in (0, 1):
        mult, dexps = deriv_cache[("d2", var_idx)]
        lap += eval_monomials(X_gpu, dexps) * cp.asarray(mult)[None, :]

    return A, derivs[0], derivs[1], derivs[2], lap


def eval_fields(A, Ax, Ay, At, Alap, coeffs):
    return (
        A @ coeffs,
        Ax @ coeffs,
        Ay @ coeffs,
        At @ coeffs,
        Alap @ coeffs,
    )


def add_residual(gram, rhs, rss, weight, residual, jacobian, name):
    if weight <= 0.0:
        return
    w = cp.float64(np.sqrt(weight))
    J = w * jacobian
    r = w * residual
    gram += J.T @ J
    rhs += J.T @ r
    rss[name] += cp.sum(residual * residual)


def accumulate_pressure_update(X_cpu, exps_cpu, deriv_cache, u_coeffs, v_coeffs, p_coeffs):
    n_terms = exps_cpu.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    n_total = 0

    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu, deriv_cache)

        u, ux, uy, ut, ulap = eval_fields(A, Ax, Ay, At, Alap, u_coeffs)
        v, vx, vy, vt, vlap = eval_fields(A, Ax, Ay, At, Alap, v_coeffs)
        _p, px, py, _pt, _plap = eval_fields(A, Ax, Ay, At, Alap, p_coeffs)

        r_mx = ut + u * ux + v * uy - cp.float64(NU) * ulap + px
        r_my = vt + u * vx + v * vy - cp.float64(NU) * vlap + py
        add_residual(gram, rhs, rss, MOM_X_WEIGHT, r_mx, Ax, "mom_x")
        add_residual(gram, rhs, rss, MOM_Y_WEIGHT, r_my, Ay, "mom_y")

        n_total += end - start
        del X_gpu, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap
        del _p, px, py, _pt, _plap, r_mx, r_my
        cp.get_default_memory_pool().free_all_blocks()

    return gram, rhs, rss, n_total


def accumulate_u_update(X_cpu, u_true_cpu, exps_cpu, deriv_cache, u_coeffs, v_coeffs, p_coeffs):
    n_terms = exps_cpu.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {
        "data": cp.float64(0.0),
        "div": cp.float64(0.0),
        "mom_x": cp.float64(0.0),
        "mom_y": cp.float64(0.0),
    }
    n_total = 0

    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        u_true = cp.asarray(u_true_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu, deriv_cache)

        u, ux, uy, ut, ulap = eval_fields(A, Ax, Ay, At, Alap, u_coeffs)
        v, vx, vy, vt, vlap = eval_fields(A, Ax, Ay, At, Alap, v_coeffs)
        _p, px, py, _pt, _plap = eval_fields(A, Ax, Ay, At, Alap, p_coeffs)

        r_data = u - u_true
        r_div = ux + vy
        r_mx = ut + u * ux + v * uy - cp.float64(NU) * ulap + px
        j_mx = At + ux[:, None] * A + u[:, None] * Ax + v[:, None] * Ay - cp.float64(NU) * Alap
        r_my = vt + u * vx + v * vy - cp.float64(NU) * vlap + py
        j_my = vx[:, None] * A

        add_residual(gram, rhs, rss, DATA_WEIGHT, r_data, A, "data")
        add_residual(gram, rhs, rss, DIV_WEIGHT, r_div, Ax, "div")
        add_residual(gram, rhs, rss, MOM_X_WEIGHT, r_mx, j_mx, "mom_x")
        add_residual(gram, rhs, rss, MOM_Y_WEIGHT, r_my, j_my, "mom_y")

        n_total += end - start
        del X_gpu, u_true, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap
        del _p, px, py, _pt, _plap, r_data, r_div, r_mx, j_mx, r_my, j_my
        cp.get_default_memory_pool().free_all_blocks()

    return gram, rhs, rss, n_total


def accumulate_v_update(X_cpu, v_true_cpu, exps_cpu, deriv_cache, u_coeffs, v_coeffs, p_coeffs):
    n_terms = exps_cpu.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {
        "data": cp.float64(0.0),
        "div": cp.float64(0.0),
        "mom_x": cp.float64(0.0),
        "mom_y": cp.float64(0.0),
    }
    n_total = 0

    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        v_true = cp.asarray(v_true_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu, deriv_cache)

        u, ux, uy, ut, ulap = eval_fields(A, Ax, Ay, At, Alap, u_coeffs)
        v, vx, vy, vt, vlap = eval_fields(A, Ax, Ay, At, Alap, v_coeffs)
        _p, px, py, _pt, _plap = eval_fields(A, Ax, Ay, At, Alap, p_coeffs)

        r_data = v - v_true
        r_div = ux + vy
        r_mx = ut + u * ux + v * uy - cp.float64(NU) * ulap + px
        j_mx = uy[:, None] * A
        r_my = vt + u * vx + v * vy - cp.float64(NU) * vlap + py
        j_my = At + u[:, None] * Ax + vy[:, None] * A + v[:, None] * Ay - cp.float64(NU) * Alap

        add_residual(gram, rhs, rss, DATA_WEIGHT, r_data, A, "data")
        add_residual(gram, rhs, rss, DIV_WEIGHT, r_div, Ay, "div")
        add_residual(gram, rhs, rss, MOM_X_WEIGHT, r_mx, j_mx, "mom_x")
        add_residual(gram, rhs, rss, MOM_Y_WEIGHT, r_my, j_my, "mom_y")

        n_total += end - start
        del X_gpu, v_true, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap
        del _p, px, py, _pt, _plap, r_data, r_div, r_mx, j_mx, r_my, j_my
        cp.get_default_memory_pool().free_all_blocks()

    return gram, rhs, rss, n_total


def solve_update(accum_result, coeffs):
    gram, rhs, rss, n_total = accum_result
    step = cp.linalg.solve(gram, -rhs)
    coeffs += step
    return float(cp.linalg.norm(step).get()), rmse_report(rss, n_total)


def rmse_report(rss, n_total):
    return {name: float(cp.sqrt(value / max(n_total, 1)).get()) for name, value in rss.items()}


def predict_value_and_grads(X_cpu, exps_cpu, deriv_cache, coeffs_cpu):
    value = np.zeros(X_cpu.shape[0], dtype=np.float64)
    gx = np.zeros(X_cpu.shape[0], dtype=np.float64)
    gy = np.zeros(X_cpu.shape[0], dtype=np.float64)
    coeffs_gpu = cp.asarray(coeffs_cpu, dtype=cp.float64)
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, _At, _Alap = basis_bundle(X_gpu, exps_cpu, deriv_cache)
        value[start:end] = (A @ coeffs_gpu).get()
        gx[start:end] = (Ax @ coeffs_gpu).get()
        gy[start:end] = (Ay @ coeffs_gpu).get()
        del X_gpu, A, Ax, Ay, _At, _Alap
        cp.get_default_memory_pool().free_all_blocks()
    return value, gx, gy


def diagnostics(X_cpu, exps_cpu, deriv_cache, u_coeffs, v_coeffs, p_coeffs,
                u_true, v_true, p_true, pg_x_true, pg_y_true):
    u_pred, _ux, _uy = predict_value_and_grads(X_cpu, exps_cpu, deriv_cache, u_coeffs)
    v_pred, _vx, _vy = predict_value_and_grads(X_cpu, exps_cpu, deriv_cache, v_coeffs)
    p_pred, px_pred, py_pred = predict_value_and_grads(X_cpu, exps_cpu, deriv_cache, p_coeffs)
    p_err = p_pred - p_true
    p_err -= p_err.mean()
    return {
        "u": float(np.sqrt(np.mean((u_pred - u_true) ** 2))),
        "v": float(np.sqrt(np.mean((v_pred - v_true) ** 2))),
        "p": float(np.sqrt(np.mean(p_err ** 2))),
        "px": float(np.sqrt(np.mean((px_pred - pg_x_true) ** 2))),
        "py": float(np.sqrt(np.mean((py_pred - pg_y_true) ** 2))),
    }


def save_solution(exps_cpu, u_coeffs_cpu, v_coeffs_cpu, p_coeffs_cpu, path_prefix):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    doc = {
        "variables": VAR_NAMES,
        "degree": degree_from_exponents(exps_cpu),
        "fit": "data_plus_physics_unknown_pressure",
        "components": [],
    }
    for name, coeffs in (("u", u_coeffs_cpu), ("v", v_coeffs_cpu), ("p", p_coeffs_cpu)):
        terms = []
        for exp, coeff in zip(exps_cpu, coeffs):
            terms.append({
                "coeff": float(coeff),
                "exponents": {
                    VAR_NAMES[i]: int(e)
                    for i, e in enumerate(exp)
                    if int(e) > 0
                },
            })
        doc["components"].append({"name": name, "n_terms": len(terms), "terms": terms})

    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    with open(path_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(f"# u, v, p fit with unknown pressure, degree <= {degree_from_exponents(exps_cpu)}\n")
        for comp in doc["components"]:
            f.write(f"\nComponent {comp['name']}\n")
            f.write("=" * 60 + "\n")
            for term in comp["terms"]:
                parts = []
                for vname in VAR_NAMES:
                    e = term["exponents"].get(vname, 0)
                    if e == 1:
                        parts.append(vname)
                    elif e > 1:
                        parts.append(f"{vname}^{e}")
                body = " * ".join(parts) if parts else "1"
                f.write(f"{term['coeff']:+.10e} * {body}\n")


def main():
    X, u_true, v_true, p_true, pg_x_true, pg_y_true = generate_taylor_green(TOTAL_POINTS, NU)

    u_coeffs = None
    v_coeffs = None
    p_coeffs = None
    old_exps = None
    best = None
    global_sweep = 0

    print("Starting from u=0, v=0, p=0. Fitting with staged basis growth.")
    print(
        f"Weights: data={DATA_WEIGHT:g}, div={DIV_WEIGHT:g}, "
        f"mom_x={MOM_X_WEIGHT:g}, mom_y={MOM_Y_WEIGHT:g}, "
        f"score_pressure_grad={SCORE_PRESSURE_GRAD_WEIGHT:g}"
    )

    for stage_idx, (degree, n_sweeps) in enumerate(DEGREE_STAGES):
        exps = make_total_degree_exponents(3, degree)
        deriv_cache = make_deriv_cache(exps)
        n_terms = exps.shape[0]

        if u_coeffs is None:
            u_coeffs = cp.zeros(n_terms, dtype=cp.float64)
            v_coeffs = cp.zeros(n_terms, dtype=cp.float64)
            p_coeffs = cp.zeros(n_terms, dtype=cp.float64)
            print(f"\nStage {stage_idx}: degree <= {degree}, terms={n_terms} per field")
        else:
            u_coeffs, copied_u = expand_coefficients(old_exps, u_coeffs, exps)
            v_coeffs, copied_v = expand_coefficients(old_exps, v_coeffs, exps)
            p_coeffs, copied_p = expand_coefficients(old_exps, p_coeffs, exps)
            print(
                f"\nStage {stage_idx}: expanded degree <= {degree}, terms={n_terms} per field; "
                f"copied u/v/p terms={copied_u}/{copied_v}/{copied_p}"
            )
        old_exps = exps

        for stage_sweep in range(n_sweeps):
            u_step, u_phys = solve_update(
                accumulate_u_update(X, u_true, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                u_coeffs,
            )
            p_after_u_step, p_after_u_phys = solve_update(
                accumulate_pressure_update(X, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                p_coeffs,
            )
            v_step, v_phys = solve_update(
                accumulate_v_update(X, v_true, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                v_coeffs,
            )
            p_after_v_step, p_after_v_phys = solve_update(
                accumulate_pressure_update(X, exps, deriv_cache, u_coeffs, v_coeffs, p_coeffs),
                p_coeffs,
            )

            u_cpu = u_coeffs.get()
            v_cpu = v_coeffs.get()
            p_cpu = p_coeffs.get()
            diag = diagnostics(
                X, exps, deriv_cache, u_cpu, v_cpu, p_cpu,
                u_true, v_true, p_true, pg_x_true, pg_y_true,
            )
            score = score_diagnostics(diag)
            if best is None or score < best["score"]:
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
                best_note = " best"
            else:
                best_note = ""

            print(
                f"Sweep {global_sweep:03d} stage={stage_idx} local={stage_sweep:03d} degree={degree}: "
                f"u_rmse={diag['u']:.6e} v_rmse={diag['v']:.6e} "
                f"p_rmse={diag['p']:.6e} px_rmse={diag['px']:.6e} py_rmse={diag['py']:.6e} "
                f"score={score:.6e}{best_note}"
            )
            print(
                "           "
                f"steps u={u_step:.3e} p_after_u={p_after_u_step:.3e} "
                f"v={v_step:.3e} p_after_v={p_after_v_step:.3e}"
            )
            print(
                "           "
                f"u data={u_phys['data']:.3e} div={u_phys['div']:.3e} "
                f"mx={u_phys['mom_x']:.3e} my={u_phys['mom_y']:.3e}"
            )
            print(
                "           "
                f"v data={v_phys['data']:.3e} div={v_phys['div']:.3e} "
                f"mx={v_phys['mom_x']:.3e} my={v_phys['mom_y']:.3e} "
                f"p_mx={p_after_v_phys['mom_x']:.3e} p_my={p_after_v_phys['mom_y']:.3e}"
            )
            global_sweep += 1

    out_prefix = os.path.join(OUT_DIR, OUT_BASENAME)
    save_solution(best["exps"], best["u"], best["v"], best["p"], out_prefix)
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    save_solution(exps, u_coeffs.get(), v_coeffs.get(), p_coeffs.get(), final_prefix)
    print(
        f"Best checkpoint: sweep={best['global_sweep']} stage={best['stage']} "
        f"local={best['stage_sweep']} degree={best['degree']} score={best['score']:.6e} "
        f"diag={best['diag']}"
    )
    print(f"Saved best solution to {out_prefix}.json and {out_prefix}.txt")
    print(f"Saved final solution to {final_prefix}.json and {final_prefix}.txt")


if __name__ == "__main__":
    main()
