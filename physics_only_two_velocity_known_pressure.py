"""
physics_only_two_velocity_known_pressure.py
===========================================
Fit both velocity components with fixed polynomial bases and known pressure.

This experiment does not use GMDH layers or boosting.  Both velocities start
at zero:

    u(x, y, t) = 0
    v(x, y, t) = 0

Then the script alternates physics-only Gauss-Newton updates:

    update u while v is fixed
    update v while u is fixed

The pressure gradients are known analytically for Taylor-Green:

    dp/dx = -0.5 sin(2x) exp(-4 nu t)
    dp/dy = -0.5 sin(2y) exp(-4 nu t)

The real velocity data is also included as a coefficient-estimation residual,
with DATA_WEIGHT controlling its strength relative to the physics residuals.
"""

import json
import os

import cupy as cp
import numpy as np


NU = 0.01
TOTAL_POINTS = 1_000_000
MAX_DEGREE = 5
CHUNK_SIZE = 20_000
N_SWEEPS = 100
INNER_GN_STEPS = 1
RIDGE_LAMBDA = 1e-10
DATA_WEIGHT = 1.0
DIV_WEIGHT = 1.0
MOM_X_WEIGHT = 1.0
MOM_Y_WEIGHT = 1.0
OUT_DIR = "poly_saves"
OUT_BASENAME = "physics_only_uv_known_pressure_degree_8"
VAR_NAMES = ["x", "y", "t"]


def generate_taylor_green_with_pressure(total_points=1_000_000, nu=0.01):
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
    pg_x = -0.5 * np.sin(2.0 * xf) * decay4
    pg_y = -0.5 * np.sin(2.0 * yf) * decay4

    return (
        np.stack([xf, yf, tf], axis=1),
        u.astype(np.float64),
        v.astype(np.float64),
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
        for p in range(1, max_power + 1):
            cols = powers == p
            if bool(cols.any()):
                A[:, cols] *= X_gpu[:, v, None] ** p
    return A


def basis_bundle(X_gpu, exps_cpu, deriv_cache):
    A = eval_monomials(X_gpu, exps_cpu)
    d = []
    for var_idx in range(3):
        mult, dexps = deriv_cache[("d1", var_idx)]
        d.append(eval_monomials(X_gpu, dexps) * cp.asarray(mult)[None, :])

    lap = cp.zeros_like(A)
    for var_idx in (0, 1):
        mult, dexps = deriv_cache[("d2", var_idx)]
        lap += eval_monomials(X_gpu, dexps) * cp.asarray(mult)[None, :]

    return A, d[0], d[1], d[2], lap


def eval_fields(A, Ax, Ay, At, Alap, coeffs):
    return (
        A @ coeffs,
        Ax @ coeffs,
        Ay @ coeffs,
        At @ coeffs,
        Alap @ coeffs,
    )


def accumulate_u_update(
        X_cpu, u_true_cpu, pg_x_cpu, pg_y_cpu,
        exps_cpu, deriv_cache, u_coeffs, v_coeffs):
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
        pg_x = cp.asarray(pg_x_cpu[start:end], dtype=cp.float64)
        pg_y = cp.asarray(pg_y_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu, deriv_cache)

        u, ux, uy, ut, ulap = eval_fields(A, Ax, Ay, At, Alap, u_coeffs)
        v, vx, vy, vt, vlap = eval_fields(A, Ax, Ay, At, Alap, v_coeffs)

        residuals = [
            (DATA_WEIGHT, u - u_true, A, "data"),
            (DIV_WEIGHT, ux + vy, Ax, "div"),
            (
                MOM_X_WEIGHT,
                ut + u * ux + v * uy - cp.float64(NU) * ulap + pg_x,
                At + ux[:, None] * A + u[:, None] * Ax + v[:, None] * Ay - cp.float64(NU) * Alap,
                "mom_x",
            ),
            (
                MOM_Y_WEIGHT,
                vt + u * vx + v * vy - cp.float64(NU) * vlap + pg_y,
                vx[:, None] * A,
                "mom_y",
            ),
        ]
        for weight, r_raw, j_raw, name in residuals:
            if weight <= 0.0:
                continue
            w = cp.float64(np.sqrt(weight))
            J = w * j_raw
            r = w * r_raw
            gram += J.T @ J
            rhs += J.T @ r
            rss[name] += cp.sum(r_raw * r_raw)

        n_total += end - start
        del X_gpu, u_true, pg_x, pg_y, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap
        cp.get_default_memory_pool().free_all_blocks()

    return gram, rhs, rss, n_total


def accumulate_v_update(
        X_cpu, v_true_cpu, pg_x_cpu, pg_y_cpu,
        exps_cpu, deriv_cache, u_coeffs, v_coeffs):
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
        pg_x = cp.asarray(pg_x_cpu[start:end], dtype=cp.float64)
        pg_y = cp.asarray(pg_y_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu, deriv_cache)

        u, ux, uy, ut, ulap = eval_fields(A, Ax, Ay, At, Alap, u_coeffs)
        v, vx, vy, vt, vlap = eval_fields(A, Ax, Ay, At, Alap, v_coeffs)

        residuals = [
            (DATA_WEIGHT, v - v_true, A, "data"),
            (DIV_WEIGHT, ux + vy, Ay, "div"),
            (
                MOM_X_WEIGHT,
                ut + u * ux + v * uy - cp.float64(NU) * ulap + pg_x,
                uy[:, None] * A,
                "mom_x",
            ),
            (
                MOM_Y_WEIGHT,
                vt + u * vx + v * vy - cp.float64(NU) * vlap + pg_y,
                At + u[:, None] * Ax + vy[:, None] * A + v[:, None] * Ay - cp.float64(NU) * Alap,
                "mom_y",
            ),
        ]
        for weight, r_raw, j_raw, name in residuals:
            if weight <= 0.0:
                continue
            w = cp.float64(np.sqrt(weight))
            J = w * j_raw
            r = w * r_raw
            gram += J.T @ J
            rhs += J.T @ r
            rss[name] += cp.sum(r_raw * r_raw)

        n_total += end - start
        del X_gpu, v_true, pg_x, pg_y, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap
        cp.get_default_memory_pool().free_all_blocks()

    return gram, rhs, rss, n_total


def rmse_report(rss, n_total):
    return {name: float(cp.sqrt(value / max(n_total, 1)).get()) for name, value in rss.items()}


def predict_polynomial(X_cpu, exps_cpu, coeffs_cpu):
    pred = np.zeros(X_cpu.shape[0], dtype=np.float64)
    coeffs_gpu = cp.asarray(coeffs_cpu, dtype=cp.float64)
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        A = eval_monomials(X_gpu, exps_cpu)
        pred[start:end] = (A @ coeffs_gpu).get()
        del X_gpu, A
        cp.get_default_memory_pool().free_all_blocks()
    return pred


def data_rmse(X_cpu, exps_cpu, u_coeffs_cpu, v_coeffs_cpu, u_true, v_true):
    u_pred = predict_polynomial(X_cpu, exps_cpu, u_coeffs_cpu)
    v_pred = predict_polynomial(X_cpu, exps_cpu, v_coeffs_cpu)
    u_rmse = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
    v_rmse = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
    return u_rmse, v_rmse


def save_solution(exps_cpu, u_coeffs_cpu, v_coeffs_cpu, path_prefix):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    doc = {
        "variables": VAR_NAMES,
        "degree": MAX_DEGREE,
        "fit": "physics_only_unknown_uv_known_pressure",
        "components": [],
    }
    for name, coeffs in (("u", u_coeffs_cpu), ("v", v_coeffs_cpu)):
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
        f.write(f"# Physics-only u,v fit with known pressure, degree <= {MAX_DEGREE}\n")
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
    X, u_true, v_true, pg_x, pg_y = generate_taylor_green_with_pressure(TOTAL_POINTS, NU)
    exps = make_total_degree_exponents(3, MAX_DEGREE)
    deriv_cache = make_deriv_cache(exps)
    n_terms = exps.shape[0]
    u_coeffs = cp.zeros(n_terms, dtype=cp.float64)
    v_coeffs = cp.zeros(n_terms, dtype=cp.float64)

    print(f"Basis: degree <= {MAX_DEGREE}, terms={n_terms} per velocity")
    print("Starting from u=0, v=0. Fitting from data + physics residuals.")
    print(
        f"Weights: data={DATA_WEIGHT:g}, div={DIV_WEIGHT:g}, "
        f"mom_x={MOM_X_WEIGHT:g}, mom_y={MOM_Y_WEIGHT:g}"
    )

    for sweep in range(N_SWEEPS):
        for _ in range(INNER_GN_STEPS):
            gram, rhs, rss, n_total = accumulate_u_update(
                X, u_true, pg_x, pg_y, exps, deriv_cache, u_coeffs, v_coeffs
            )
            step = cp.linalg.solve(gram, -rhs)
            u_coeffs += step
            u_step = float(cp.linalg.norm(step).get())
            u_phys = rmse_report(rss, n_total)

        for _ in range(INNER_GN_STEPS):
            gram, rhs, rss, n_total = accumulate_v_update(
                X, v_true, pg_x, pg_y, exps, deriv_cache, u_coeffs, v_coeffs
            )
            step = cp.linalg.solve(gram, -rhs)
            v_coeffs += step
            v_step = float(cp.linalg.norm(step).get())
            v_phys = rmse_report(rss, n_total)

        u_cpu = u_coeffs.get()
        v_cpu = v_coeffs.get()
        u_rmse, v_rmse = data_rmse(X, exps, u_cpu, v_cpu, u_true, v_true)
        print(
            f"Sweep {sweep:02d}: "
            f"u_step={u_step:.3e} v_step={v_step:.3e} | "
            f"u_data_rmse={u_rmse:.6e} v_data_rmse={v_rmse:.6e}"
        )
        print(
            "           "
            f"after u-update data={u_phys['data']:.3e} div={u_phys['div']:.3e} "
            f"mx={u_phys['mom_x']:.3e} my={u_phys['mom_y']:.3e}"
        )
        print(
            "           "
            f"after v-update data={v_phys['data']:.3e} div={v_phys['div']:.3e} "
            f"mx={v_phys['mom_x']:.3e} my={v_phys['mom_y']:.3e}"
        )

    u_cpu = u_coeffs.get()
    v_cpu = v_coeffs.get()
    out_prefix = os.path.join(OUT_DIR, OUT_BASENAME)
    save_solution(exps, u_cpu, v_cpu, out_prefix)
    print(f"Saved solution to {out_prefix}.json and {out_prefix}.txt")


if __name__ == "__main__":
    main()
