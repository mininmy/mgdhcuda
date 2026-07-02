"""
physics_only_uv_chebyshev.py
============================
Shared Chebyshev-basis helpers for fitting u, v, and pressure.

Basis terms are tensor products

    T_a(x) * T_b(y) * T_c(tau),  tau = 2t - 1

with either total degree <= D or a moving exact-degree window.  The fitting
logic mirrors the monomial u/v/p experiments, but evaluates Chebyshev values
and exact derivatives by recurrence.
"""

import json
import os

import cupy as cp
import numpy as np


NU = 0.01
TOTAL_POINTS = 1_000_000
CHUNK_SIZE = 20_000
RIDGE_LAMBDA = 1e-10
DATA_WEIGHT = 1.0
DIV_WEIGHT = 1.0
MOM_X_WEIGHT = 1.0
MOM_Y_WEIGHT = 1.0
SCORE_PRESSURE_GRAD_WEIGHT = 0.1
OUT_DIR = "poly_saves"
VAR_NAMES = ["x", "y", "tau"]


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


def merge_coefficients(base_exps, base_coeffs, add_exps, add_coeffs):
    merged = {}
    if base_exps.shape[0] > 0:
        base_cpu = base_coeffs.get() if isinstance(base_coeffs, cp.ndarray) else base_coeffs
        for exp, coeff in zip(base_exps, base_cpu):
            key = tuple(int(e) for e in exp)
            merged[key] = merged.get(key, 0.0) + float(coeff)
    add_cpu = add_coeffs.get() if isinstance(add_coeffs, cp.ndarray) else add_coeffs
    for exp, coeff in zip(add_exps, add_cpu):
        key = tuple(int(e) for e in exp)
        merged[key] = merged.get(key, 0.0) + float(coeff)
    items = sorted(merged.items(), key=lambda item: (sum(item[0]), item[0]))
    if not items:
        return np.zeros((0, 3), dtype=np.uint16), cp.zeros(0, dtype=cp.float64)
    exps = np.array([k for k, _ in items], dtype=np.uint16)
    coeffs = np.array([v for _, v in items], dtype=np.float64)
    return exps, cp.asarray(coeffs, dtype=cp.float64)


def degree_from_exponents(exps_cpu):
    return int(np.max(np.sum(exps_cpu, axis=1))) if len(exps_cpu) else 0


def score_diagnostics(diag):
    return diag["u"] + diag["v"] + SCORE_PRESSURE_GRAD_WEIGHT * (diag["px"] + diag["py"])


def cheb_values_1d(z, max_degree, first_scale=1.0, second_scale=1.0):
    n = z.shape[0]
    T = cp.empty((n, max_degree + 1), dtype=cp.float64)
    dT = cp.empty_like(T)
    ddT = cp.empty_like(T)
    T[:, 0] = 1.0
    dT[:, 0] = 0.0
    ddT[:, 0] = 0.0
    if max_degree >= 1:
        T[:, 1] = z
        dT[:, 1] = 1.0
        ddT[:, 1] = 0.0
    for k in range(2, max_degree + 1):
        T[:, k] = 2.0 * z * T[:, k - 1] - T[:, k - 2]
        dT[:, k] = 2.0 * T[:, k - 1] + 2.0 * z * dT[:, k - 1] - dT[:, k - 2]
        ddT[:, k] = 4.0 * dT[:, k - 1] + 2.0 * z * ddT[:, k - 1] - ddT[:, k - 2]
    if first_scale != 1.0:
        dT *= first_scale
    if second_scale != 1.0:
        ddT *= second_scale
    return T, dT, ddT


def basis_bundle(X_gpu, exps_cpu):
    n_rows = X_gpu.shape[0]
    n_terms = exps_cpu.shape[0]
    if n_terms == 0:
        z = cp.zeros((n_rows, 0), dtype=cp.float64)
        return z, z, z, z, z

    max_degree = int(exps_cpu.max()) if n_terms else 0
    Tx, dTx, ddTx = cheb_values_1d(X_gpu[:, 0], max_degree)
    Ty, dTy, ddTy = cheb_values_1d(X_gpu[:, 1], max_degree)
    tau = 2.0 * X_gpu[:, 2] - 1.0
    Tt, dTt, _ddTt = cheb_values_1d(tau, max_degree, first_scale=2.0, second_scale=4.0)

    a = cp.asarray(exps_cpu[:, 0], dtype=cp.int32)
    b = cp.asarray(exps_cpu[:, 1], dtype=cp.int32)
    c = cp.asarray(exps_cpu[:, 2], dtype=cp.int32)

    Xv = Tx[:, a]
    Yv = Ty[:, b]
    Tv = Tt[:, c]
    A = Xv * Yv * Tv
    Ax = dTx[:, a] * Yv * Tv
    Ay = Xv * dTy[:, b] * Tv
    At = Xv * Yv * dTt[:, c]
    Alap = ddTx[:, a] * Yv * Tv + Xv * ddTy[:, b] * Tv
    return A, Ax, Ay, At, Alap


def zero_fields(n_rows):
    z = cp.zeros(n_rows, dtype=cp.float64)
    return z, z, z, z, z


def eval_fields(A, Ax, Ay, At, Alap, coeffs):
    if coeffs.shape[0] == 0:
        return zero_fields(A.shape[0])
    return A @ coeffs, Ax @ coeffs, Ay @ coeffs, At @ coeffs, Alap @ coeffs


def add_field_tuple(a, b):
    return tuple(x + y for x, y in zip(a, b))


def total_fields(X_gpu, base, window):
    base_exps, base_coeffs = base
    win_exps, win_coeffs = window
    if base_exps.shape[0] > 0:
        base_fields = eval_fields(*basis_bundle(X_gpu, base_exps), base_coeffs)
    else:
        base_fields = zero_fields(X_gpu.shape[0])
    win_bundle = basis_bundle(X_gpu, win_exps)
    win_fields = eval_fields(*win_bundle, win_coeffs)
    return add_field_tuple(base_fields, win_fields), win_bundle


def add_residual(gram, rhs, rss, weight, residual, jacobian, name):
    if weight <= 0.0 or jacobian.shape[1] == 0:
        return
    w = cp.float64(np.sqrt(weight))
    J = w * jacobian
    r = w * residual
    gram += J.T @ J
    rhs += J.T @ r
    rss[name] += cp.sum(residual * residual)


def rmse_report(rss, n_total):
    return {name: float(cp.sqrt(value / max(n_total, 1)).get()) for name, value in rss.items()}


def solve_update(accum_result, coeffs):
    gram, rhs, rss, n_total = accum_result
    step = cp.linalg.solve(gram, -rhs)
    coeffs += step
    return float(cp.linalg.norm(step).get()), rmse_report(rss, n_total)


def accumulate_pressure_update(X_cpu, exps_cpu, u_coeffs, v_coeffs, p_coeffs):
    n_terms = exps_cpu.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    n_total = 0
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu)
        u, ux, uy, ut, ulap = eval_fields(A, Ax, Ay, At, Alap, u_coeffs)
        v, vx, vy, vt, vlap = eval_fields(A, Ax, Ay, At, Alap, v_coeffs)
        _p, px, py, _pt, _plap = eval_fields(A, Ax, Ay, At, Alap, p_coeffs)
        r_mx = ut + u * ux + v * uy - cp.float64(NU) * ulap + px
        r_my = vt + u * vx + v * vy - cp.float64(NU) * vlap + py
        add_residual(gram, rhs, rss, MOM_X_WEIGHT, r_mx, Ax, "mom_x")
        add_residual(gram, rhs, rss, MOM_Y_WEIGHT, r_my, Ay, "mom_y")
        n_total += end - start
        del X_gpu, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap, _p, px, py
        cp.get_default_memory_pool().free_all_blocks()
    return gram, rhs, rss, n_total


def accumulate_u_update(X_cpu, u_true_cpu, exps_cpu, u_coeffs, v_coeffs, p_coeffs):
    n_terms = exps_cpu.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"data": cp.float64(0.0), "div": cp.float64(0.0), "mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    n_total = 0
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        u_true = cp.asarray(u_true_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu)
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
        del X_gpu, u_true, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap, _p, px, py
        cp.get_default_memory_pool().free_all_blocks()
    return gram, rhs, rss, n_total


def accumulate_v_update(X_cpu, v_true_cpu, exps_cpu, u_coeffs, v_coeffs, p_coeffs):
    n_terms = exps_cpu.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"data": cp.float64(0.0), "div": cp.float64(0.0), "mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    n_total = 0
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        v_true = cp.asarray(v_true_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, At, Alap = basis_bundle(X_gpu, exps_cpu)
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
        del X_gpu, v_true, A, Ax, Ay, At, Alap, u, ux, uy, ut, ulap, v, vx, vy, vt, vlap, _p, px, py
        cp.get_default_memory_pool().free_all_blocks()
    return gram, rhs, rss, n_total


def accumulate_pressure_update_window(X_cpu, base_state, win_exps, u_win, v_win, p_win):
    n_terms = win_exps.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    u_base, v_base, p_base = base_state
    n_total = 0
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        u_fields, _ = total_fields(X_gpu, u_base, (win_exps, u_win))
        v_fields, _ = total_fields(X_gpu, v_base, (win_exps, v_win))
        p_fields, p_bundle = total_fields(X_gpu, p_base, (win_exps, p_win))
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


def accumulate_u_update_window(X_cpu, u_true_cpu, base_state, win_exps, u_win, v_win, p_win):
    n_terms = win_exps.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"data": cp.float64(0.0), "div": cp.float64(0.0), "mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    u_base, v_base, p_base = base_state
    n_total = 0
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        u_true = cp.asarray(u_true_cpu[start:end], dtype=cp.float64)
        u_fields, u_bundle = total_fields(X_gpu, u_base, (win_exps, u_win))
        v_fields, _ = total_fields(X_gpu, v_base, (win_exps, v_win))
        p_fields, _ = total_fields(X_gpu, p_base, (win_exps, p_win))
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


def accumulate_v_update_window(X_cpu, v_true_cpu, base_state, win_exps, u_win, v_win, p_win):
    n_terms = win_exps.shape[0]
    gram = cp.eye(n_terms, dtype=cp.float64) * cp.float64(RIDGE_LAMBDA)
    rhs = cp.zeros(n_terms, dtype=cp.float64)
    rss = {"data": cp.float64(0.0), "div": cp.float64(0.0), "mom_x": cp.float64(0.0), "mom_y": cp.float64(0.0)}
    u_base, v_base, p_base = base_state
    n_total = 0
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        v_true = cp.asarray(v_true_cpu[start:end], dtype=cp.float64)
        u_fields, _ = total_fields(X_gpu, u_base, (win_exps, u_win))
        v_fields, v_bundle = total_fields(X_gpu, v_base, (win_exps, v_win))
        p_fields, _ = total_fields(X_gpu, p_base, (win_exps, p_win))
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


def predict_value_and_grads(X_cpu, exps_cpu, coeffs_cpu):
    value = np.zeros(X_cpu.shape[0], dtype=np.float64)
    gx = np.zeros(X_cpu.shape[0], dtype=np.float64)
    gy = np.zeros(X_cpu.shape[0], dtype=np.float64)
    coeffs_gpu = cp.asarray(coeffs_cpu, dtype=cp.float64)
    for start in range(0, X_cpu.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, X_cpu.shape[0])
        X_gpu = cp.asarray(X_cpu[start:end], dtype=cp.float64)
        A, Ax, Ay, _At, _Alap = basis_bundle(X_gpu, exps_cpu)
        value[start:end] = (A @ coeffs_gpu).get()
        gx[start:end] = (Ax @ coeffs_gpu).get()
        gy[start:end] = (Ay @ coeffs_gpu).get()
        del X_gpu, A, Ax, Ay, _At, _Alap
        cp.get_default_memory_pool().free_all_blocks()
    return value, gx, gy


def diagnostics(X_cpu, exps_cpu, u_coeffs, v_coeffs, p_coeffs, u_true, v_true, p_true, pg_x_true, pg_y_true):
    u_pred, _ux, _uy = predict_value_and_grads(X_cpu, exps_cpu, u_coeffs)
    v_pred, _vx, _vy = predict_value_and_grads(X_cpu, exps_cpu, v_coeffs)
    p_pred, px_pred, py_pred = predict_value_and_grads(X_cpu, exps_cpu, p_coeffs)
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
    doc = {"variables": VAR_NAMES, "basis": "chebyshev", "degree": degree_from_exponents(exps_cpu), "components": []}
    for name, coeffs in (("u", u_coeffs_cpu), ("v", v_coeffs_cpu), ("p", p_coeffs_cpu)):
        terms = []
        for exp, coeff in zip(exps_cpu, coeffs):
            terms.append({"coeff": float(coeff), "cheb_degrees": {VAR_NAMES[i]: int(e) for i, e in enumerate(exp) if int(e) > 0}})
        doc["components"].append({"name": name, "n_terms": len(terms), "terms": terms})
    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    with open(path_prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(f"# Chebyshev u, v, p fit, degree <= {degree_from_exponents(exps_cpu)}\n")
        f.write("# Variables: x, y, tau=2t-1\n")
        for comp in doc["components"]:
            f.write(f"\nComponent {comp['name']}\n")
            f.write("=" * 60 + "\n")
            for term in comp["terms"]:
                parts = []
                for vname in VAR_NAMES:
                    e = term["cheb_degrees"].get(vname, 0)
                    if e > 0:
                        parts.append(f"T{e}({vname})")
                body = " * ".join(parts) if parts else "1"
                f.write(f"{term['coeff']:+.10e} * {body}\n")
