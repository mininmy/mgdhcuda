"""
physics_only_uvp_window_boost.py
================================
Boost u, v, and pressure with overlapping polynomial degree windows.

Each window fits only a correction basis while previously learned coefficients
stay fixed.  After a window finishes, the correction is merged into the
cumulative u/v/p polynomials and the next shifted window is trained.

Default windows:

    degree 0..3
    degree 1..4
    degree 2..5
    degree 3..6
    degree 4..7
    degree 5..8

Velocity data anchors u and v.  Pressure is learned only through the momentum
residuals, as a single scalar pressure polynomial whose x/y derivatives enter
the equations.
"""

import json
import os

import cupy as cp
import numpy as np


NU = 0.01
TOTAL_POINTS = 1_000_000
DEGREE_WINDOWS = [
    (0, 3, 35),
    (1, 4, 35),
    (2, 5, 35),
    (3, 6, 35),
    (4, 7, 35),
    (5, 8, 35),
]
CHUNK_SIZE = 20_000
RIDGE_LAMBDA = 1e-10
DATA_WEIGHT = 1.0
DIV_WEIGHT = 1.0
MOM_X_WEIGHT = 1.0
MOM_Y_WEIGHT = 1.0
SCORE_PRESSURE_GRAD_WEIGHT = 0.1
OUT_DIR = "poly_saves"
OUT_BASENAME = "physics_only_uvp_window_boost"
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


def make_degree_window_exponents(n_vars, min_degree, max_degree):
    exps = []

    def rec(prefix, remaining_vars, remaining_degree):
        if remaining_vars == 1:
            exps.append(prefix + [remaining_degree])
            return
        for e in range(remaining_degree + 1):
            rec(prefix + [e], remaining_vars - 1, remaining_degree - e)

    for degree in range(min_degree, max_degree + 1):
        rec([], n_vars, degree)
    exps.sort(key=lambda row: (sum(row), row))
    return np.array(exps, dtype=np.uint16)


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
    if n_terms == 0:
        return cp.zeros((n_rows, 0), dtype=cp.float64)
    A = cp.ones((n_rows, n_terms), dtype=cp.float64)
    exps_gpu = cp.asarray(exps_cpu, dtype=cp.uint16)
    for v in range(exps_cpu.shape[1]):
        powers = exps_gpu[:, v]
        max_power = int(powers.max().get())
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


def zero_fields(n_rows):
    z = cp.zeros(n_rows, dtype=cp.float64)
    return z, z, z, z, z


def eval_fields(A, Ax, Ay, At, Alap, coeffs):
    if coeffs.shape[0] == 0:
        return zero_fields(A.shape[0])
    return (
        A @ coeffs,
        Ax @ coeffs,
        Ay @ coeffs,
        At @ coeffs,
        Alap @ coeffs,
    )


def add_field_tuple(a, b):
    return tuple(x + y for x, y in zip(a, b))


def add_residual(gram, rhs, rss, weight, residual, jacobian, name):
    if weight <= 0.0 or jacobian.shape[1] == 0:
        return
    w = cp.float64(np.sqrt(weight))
    J = w * jacobian
    r = w * residual
    gram += J.T @ J
    rhs += J.T @ r
    rss[name] += cp.sum(residual * residual)


def merge_coefficients(base_exps, base_coeffs, add_exps, add_coeffs):
    merged = {}
    if base_exps.shape[0] > 0:
        base_cpu = base_coeffs.get() if isinstance(base_coeffs, cp.ndarray) else base_coeffs
        for exp, coeff in zip(base_exps, base_cpu):
            merged[tuple(int(e) for e in exp)] = merged.get(tuple(int(e) for e in exp), 0.0) + float(coeff)
    add_cpu = add_coeffs.get() if isinstance(add_coeffs, cp.ndarray) else add_coeffs
    for exp, coeff in zip(add_exps, add_cpu):
        merged[tuple(int(e) for e in exp)] = merged.get(tuple(int(e) for e in exp), 0.0) + float(coeff)

    items = sorted(merged.items(), key=lambda item: (sum(item[0]), item[0]))
    exps = np.array([k for k, _ in items], dtype=np.uint16)
    coeffs = np.array([v for _, v in items], dtype=np.float64)
    return exps, cp.asarray(coeffs, dtype=cp.float64)


def score_diagnostics(diag):
    return (
        diag["u"] + diag["v"]
        + SCORE_PRESSURE_GRAD_WEIGHT * (diag["px"] + diag["py"])
    )


def degree_from_exponents(exps_cpu):
    return int(np.max(np.sum(exps_cpu, axis=1))) if len(exps_cpu) else 0


def total_fields(X_gpu, base, window):
    base_exps, base_cache, base_coeffs = base
    win_exps, win_cache, win_coeffs = window

    if base_exps.shape[0] > 0:
        base_bundle = basis_bundle(X_gpu, base_exps, base_cache)
        base_fields = eval_fields(*base_bundle, base_coeffs)
    else:
        base_fields = zero_fields(X_gpu.shape[0])

    win_bundle = basis_bundle(X_gpu, win_exps, win_cache)
    win_fields = eval_fields(*win_bundle, win_coeffs)
    return add_field_tuple(base_fields, win_fields), win_bundle


def accumulate_pressure_update(
        X_cpu, base_state, win_exps, win_cache,
        u_win, v_win, p_win):
    n_terms = win_exps.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    n_total = 0

    u_base, v_base, p_base = base_state
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)

        u_fields, _u_bundle = total_fields(X_gpu, u_base, (win_exps, win_cache, u_win))
        v_fields, _v_bundle = total_fields(X_gpu, v_base, (win_exps, win_cache, v_win))
        p_fields, p_bundle = total_fields(X_gpu, p_base, (win_exps, win_cache, p_win))

        u, ux, uy, ut, ulap = u_fields
        v, vx, vy, vt, vlap = v_fields
        _p, px, py, _pt, _plap = p_fields
        _A, p_x_basis, p_y_basis, _At, _Alap = p_bundle

        r_mx = ut + u * ux + v * uy - cp.float64(NU) * ulap + px
        r_my = vt + u * vx + v * vy - cp.float64(NU) * vlap + py
        add_residual(gram, rhs, rss, MOM_X_WEIGHT, r_mx, p_x_basis, "mom_x")
        add_residual(gram, rhs, rss, MOM_Y_WEIGHT, r_my, p_y_basis, "mom_y")

        n_total += end - start
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()

    return gram, rhs, rss, n_total


def accumulate_u_update(
        X_cpu, u_true_cpu, base_state, win_exps, win_cache,
        u_win, v_win, p_win):
    n_terms = win_exps.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {
        "data": cp.float64(0.0),
        "div": cp.float64(0.0),
        "mom_x": cp.float64(0.0),
        "mom_y": cp.float64(0.0),
    }
    n_total = 0

    u_base, v_base, p_base = base_state
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        u_true = cp.asarray(u_true_cpu[start:end], dtype=cp.float64)

        u_fields, u_bundle = total_fields(X_gpu, u_base, (win_exps, win_cache, u_win))
        v_fields, _v_bundle = total_fields(X_gpu, v_base, (win_exps, win_cache, v_win))
        p_fields, _p_bundle = total_fields(X_gpu, p_base, (win_exps, win_cache, p_win))

        u, ux, uy, ut, ulap = u_fields
        v, vx, vy, vt, vlap = v_fields
        _p, px, py, _pt, _plap = p_fields
        A, Ax, Ay, At, Alap = u_bundle

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
        del X_gpu, u_true
        cp.get_default_memory_pool().free_all_blocks()

    return gram, rhs, rss, n_total


def accumulate_v_update(
        X_cpu, v_true_cpu, base_state, win_exps, win_cache,
        u_win, v_win, p_win):
    n_terms = win_exps.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {
        "data": cp.float64(0.0),
        "div": cp.float64(0.0),
        "mom_x": cp.float64(0.0),
        "mom_y": cp.float64(0.0),
    }
    n_total = 0

    u_base, v_base, p_base = base_state
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        v_true = cp.asarray(v_true_cpu[start:end], dtype=cp.float64)

        u_fields, _u_bundle = total_fields(X_gpu, u_base, (win_exps, win_cache, u_win))
        v_fields, v_bundle = total_fields(X_gpu, v_base, (win_exps, win_cache, v_win))
        p_fields, _p_bundle = total_fields(X_gpu, p_base, (win_exps, win_cache, p_win))

        u, ux, uy, ut, ulap = u_fields
        v, vx, vy, vt, vlap = v_fields
        _p, px, py, _pt, _plap = p_fields
        A, Ax, Ay, At, Alap = v_bundle

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
        del X_gpu, v_true
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
        "fit": "window_boost_unknown_pressure",
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
        f.write(f"# window-boost u, v, p fit, degree <= {degree_from_exponents(exps_cpu)}\n")
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

    cum_exps = np.zeros((0, 3), dtype=np.uint16)
    empty_cache = make_deriv_cache(cum_exps)
    u_cum = cp.zeros(0, dtype=cp.float64)
    v_cum = cp.zeros(0, dtype=cp.float64)
    p_cum = cp.zeros(0, dtype=cp.float64)
    best = None
    global_sweep = 0

    print("Starting window-boosted u/v/p fit with unknown pressure.")
    print(
        f"Weights: data={DATA_WEIGHT:g}, div={DIV_WEIGHT:g}, "
        f"mom_x={MOM_X_WEIGHT:g}, mom_y={MOM_Y_WEIGHT:g}, "
        f"score_pressure_grad={SCORE_PRESSURE_GRAD_WEIGHT:g}"
    )

    for window_idx, (d_min, d_max, n_sweeps) in enumerate(DEGREE_WINDOWS):
        win_exps = make_degree_window_exponents(3, d_min, d_max)
        win_cache = make_deriv_cache(win_exps)
        u_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
        v_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)
        p_win = cp.zeros(win_exps.shape[0], dtype=cp.float64)

        print(
            f"\nWindow {window_idx}: degree {d_min}..{d_max}, "
            f"terms={win_exps.shape[0]} per correction"
        )

        for local_sweep in range(n_sweeps):
            cum_cache = make_deriv_cache(cum_exps) if cum_exps.shape[0] else empty_cache
            base_state = (
                (cum_exps, cum_cache, u_cum),
                (cum_exps, cum_cache, v_cum),
                (cum_exps, cum_cache, p_cum),
            )

            u_step, u_phys = solve_update(
                accumulate_u_update(X, u_true, base_state, win_exps, win_cache, u_win, v_win, p_win),
                u_win,
            )
            p_after_u_step, p_after_u_phys = solve_update(
                accumulate_pressure_update(X, base_state, win_exps, win_cache, u_win, v_win, p_win),
                p_win,
            )
            v_step, v_phys = solve_update(
                accumulate_v_update(X, v_true, base_state, win_exps, win_cache, u_win, v_win, p_win),
                v_win,
            )
            p_after_v_step, p_after_v_phys = solve_update(
                accumulate_pressure_update(X, base_state, win_exps, win_cache, u_win, v_win, p_win),
                p_win,
            )

            tmp_exps, tmp_u = merge_coefficients(cum_exps, u_cum, win_exps, u_win)
            _tmp_exps, tmp_v = merge_coefficients(cum_exps, v_cum, win_exps, v_win)
            _tmp_exps, tmp_p = merge_coefficients(cum_exps, p_cum, win_exps, p_win)
            tmp_cache = make_deriv_cache(tmp_exps)
            u_cpu = tmp_u.get()
            v_cpu = tmp_v.get()
            p_cpu = tmp_p.get()
            diag = diagnostics(
                X, tmp_exps, tmp_cache, u_cpu, v_cpu, p_cpu,
                u_true, v_true, p_true, pg_x_true, pg_y_true,
            )
            score = score_diagnostics(diag)
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "global_sweep": global_sweep,
                    "window": window_idx,
                    "local_sweep": local_sweep,
                    "degree_window": (d_min, d_max),
                    "diag": dict(diag),
                    "exps": tmp_exps.copy(),
                    "u": u_cpu.copy(),
                    "v": v_cpu.copy(),
                    "p": p_cpu.copy(),
                }
                best_note = " best"
            else:
                best_note = ""

            print(
                f"Sweep {global_sweep:03d} window={window_idx} local={local_sweep:03d} "
                f"degree={d_min}..{d_max}: "
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

        next_exps, next_u = merge_coefficients(cum_exps, u_cum, win_exps, u_win)
        _next_exps, next_v = merge_coefficients(cum_exps, v_cum, win_exps, v_win)
        _next_exps, next_p = merge_coefficients(cum_exps, p_cum, win_exps, p_win)
        cum_exps, u_cum, v_cum, p_cum = next_exps, next_u, next_v, next_p
        print(
            f"Window {window_idx} merged: cumulative terms={cum_exps.shape[0]} "
            f"degree <= {degree_from_exponents(cum_exps)}"
        )

    out_prefix = os.path.join(OUT_DIR, OUT_BASENAME)
    save_solution(best["exps"], best["u"], best["v"], best["p"], out_prefix)
    final_prefix = os.path.join(OUT_DIR, OUT_BASENAME + "_final")
    save_solution(cum_exps, u_cum.get(), v_cum.get(), p_cum.get(), final_prefix)
    print(
        f"Best checkpoint: sweep={best['global_sweep']} window={best['window']} "
        f"local={best['local_sweep']} degree_window={best['degree_window']} "
        f"score={best['score']:.6e} diag={best['diag']}"
    )
    print(f"Saved best solution to {out_prefix}.json and {out_prefix}.txt")
    print(f"Saved final solution to {final_prefix}.json and {final_prefix}.txt")


if __name__ == "__main__":
    main()
