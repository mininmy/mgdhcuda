"""
physics_only_known_fit.py
=========================
Fit one velocity component with a fixed polynomial basis and known physics.

This experiment does not use GMDH layers or boosting.  The x-velocity u is
represented as a polynomial in (x, y, t) using all monomials with total degree
up to MAX_DEGREE.  Coefficients are estimated from physics residuals only:

    div:      du/dx + dv_known/dy = 0
    momentum: du/dt + u du/dx + v_known du/dy - nu lap(u) + dp/dx_known = 0

The true Taylor-Green u data is used only after fitting, to report RMSE.
"""

import json
import os

import cupy as cp
import numpy as np

from gpu_gmdh_newton_known_physics import generate_taylor_green_data_with_known


NU = 0.01
TOTAL_POINTS = 1_000_000
MAX_DEGREE = 8
CHUNK_SIZE = 20_000
GN_ITERS = 12
RIDGE_LAMBDA = 1e-10
DIV_WEIGHT = 1.0
MOM_WEIGHT = 1.0
OUT_DIR = "poly_saves"
OUT_BASENAME = "physics_only_degree_8"
VAR_NAMES = ["x", "y", "t"]


def make_total_degree_exponents(n_vars, max_degree):
    """Return exponents for all monomials with total degree <= max_degree."""
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
    """Return derivative multipliers and exponents for d^order/dvar^order."""
    exps = exps_cpu.astype(np.int32)
    powers = exps[:, var_idx]
    mult = np.ones(len(exps), dtype=np.float64)
    for k in range(order):
        mult *= np.maximum(powers - k, 0)
    out = exps.copy()
    out[:, var_idx] = np.maximum(out[:, var_idx] - order, 0)
    mult[powers < order] = 0.0
    return mult, out.astype(np.uint16)


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


def make_deriv_cache(exps_cpu):
    cache = {}
    for var_idx in range(3):
        cache[("d1", var_idx)] = derivative_exponents(exps_cpu, var_idx, 1)
        cache[("d2", var_idx)] = derivative_exponents(exps_cpu, var_idx, 2)
    return cache


def physics_residual_and_jacobian(A, Ax, Ay, At, Alap, coeffs, v_known, dv_dy, pg_x):
    u = A @ coeffs
    ux = Ax @ coeffs
    uy = Ay @ coeffs
    ut = At @ coeffs
    lap = Alap @ coeffs

    r_div = ux + dv_dy
    j_div = Ax

    r_mom = ut + u * ux + v_known * uy - cp.float64(NU) * lap + pg_x
    j_mom = At + ux[:, None] * A + u[:, None] * Ax + v_known[:, None] * Ay - cp.float64(NU) * Alap

    return r_div, j_div, r_mom, j_mom


def fit_physics_only(X_cpu, v_cpu, dv_dy_cpu, pg_x_cpu, exps_cpu):
    deriv_cache = make_deriv_cache(exps_cpu)
    n_terms = exps_cpu.shape[0]
    coeffs = cp.zeros(n_terms, dtype=cp.float64)

    for it in range(GN_ITERS):
        gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
        rhs = cp.zeros(n_terms, dtype=cp.float64)
        rss_div = cp.float64(0.0)
        rss_mom = cp.float64(0.0)
        n_total = 0

        for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, X_cpu.shape[0])
            X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
            v_gpu = cp.asarray(v_cpu[start:end], dtype=cp.float64)
            dv_dy_gpu = cp.asarray(dv_dy_cpu[start:end], dtype=cp.float64)
            pg_x_gpu = cp.asarray(pg_x_cpu[start:end], dtype=cp.float64)

            A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu, deriv_cache)
            r_div, j_div, r_mom, j_mom = physics_residual_and_jacobian(
                A, Ax, Ay, At, Alap, coeffs, v_gpu, dv_dy_gpu, pg_x_gpu
            )

            if DIV_WEIGHT > 0.0:
                wd = cp.float64(np.sqrt(DIV_WEIGHT))
                J = wd * j_div
                r = wd * r_div
                gram += J.T @ J
                rhs += J.T @ r
                rss_div += cp.sum(r_div * r_div)

            if MOM_WEIGHT > 0.0:
                wm = cp.float64(np.sqrt(MOM_WEIGHT))
                J = wm * j_mom
                r = wm * r_mom
                gram += J.T @ J
                rhs += J.T @ r
                rss_mom += cp.sum(r_mom * r_mom)

            n_total += end - start
            del X_gpu, v_gpu, dv_dy_gpu, pg_x_gpu, A, Ax, Ay, At, Alap
            del r_div, j_div, r_mom, j_mom, J, r
            cp.get_default_memory_pool().free_all_blocks()

        step = cp.linalg.solve(gram, -rhs)
        coeffs += step

        step_norm = float(cp.linalg.norm(step).get())
        div_rmse = float(cp.sqrt(rss_div / max(n_total, 1)).get())
        mom_rmse = float(cp.sqrt(rss_mom / max(n_total, 1)).get())
        print(
            f"GN {it:02d}: div_rmse={div_rmse:.6e} "
            f"mom_rmse={mom_rmse:.6e} step_norm={step_norm:.6e}"
        )
        if step_norm < 1e-12:
            break

    return coeffs.get()


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


def save_polynomial(exps_cpu, coeffs_cpu, path_prefix):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    terms = []
    for exp, coeff in zip(exps_cpu, coeffs_cpu):
        terms.append({
            "coeff": float(coeff),
            "exponents": {
                VAR_NAMES[i]: int(e)
                for i, e in enumerate(exp)
                if int(e) > 0
            },
        })

    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({
            "variables": VAR_NAMES,
            "degree": MAX_DEGREE,
            "fit": "physics_only_known_v_pg",
            "terms": terms,
        }, f, indent=2)

    with open(path_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(f"# Physics-only polynomial fit, degree <= {MAX_DEGREE}\n")
        f.write("# Variables: x, y, t\n\n")
        for exp, coeff in zip(exps_cpu, coeffs_cpu):
            parts = []
            for v, e in enumerate(exp):
                if e == 1:
                    parts.append(VAR_NAMES[v])
                elif e > 1:
                    parts.append(f"{VAR_NAMES[v]}^{int(e)}")
            body = " * ".join(parts) if parts else "1"
            f.write(f"{coeff:+.10e} * {body}\n")


def main():
    X, y, v_known, dv_dy_known, pg_x_known = generate_taylor_green_data_with_known(
        TOTAL_POINTS, nu=NU
    )
    y_true = y[0]
    exps = make_total_degree_exponents(3, MAX_DEGREE)
    print(f"Basis: degree <= {MAX_DEGREE}, terms={len(exps)}")
    print("Fitting coefficients from physics residuals only...")

    coeffs = fit_physics_only(X, v_known, dv_dy_known, pg_x_known, exps)

    pred = predict_polynomial(X, exps, coeffs)
    rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(pred - y_true)))
    max_abs = float(np.max(np.abs(pred - y_true)))
    print("\nData check after physics-only fit:")
    print(f"  RMSE    : {rmse:.6e}")
    print(f"  MAE     : {mae:.6e}")
    print(f"  max_abs : {max_abs:.6e}")

    out_prefix = os.path.join(OUT_DIR, OUT_BASENAME)
    save_polynomial(exps, coeffs, out_prefix)
    print(f"Saved polynomial to {out_prefix}.json and {out_prefix}.txt")


if __name__ == "__main__":
    main()
