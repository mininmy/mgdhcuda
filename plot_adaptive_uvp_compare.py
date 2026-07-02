"""
plot_adaptive_uvp_compare.py
============================
Plot and compare adaptive degree-growth and moving-window experiments.

Inputs:
    poly_saves/adaptive_uvp_degree_growth.csv
    poly_saves/adaptive_uvp_moving_window.csv

Outputs:
    poly_saves/adaptive_uvp_compare_rmse.png
    poly_saves/adaptive_uvp_compare_pressure_grad.png
    poly_saves/adaptive_uvp_compare_score.png
    poly_saves/adaptive_uvp_compare_basis.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


DEFAULT_GROWTH_CSV = "poly_saves/adaptive_uvp_degree_growth.csv"
DEFAULT_WINDOW_CSV = "poly_saves/adaptive_uvp_moving_window.csv"
DEFAULT_OUT_DIR = "poly_saves"


def _float(row, key):
    return float(row[key])


def _int(row, key):
    return int(row[key])


def load_growth(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "mode": "degree_growth",
                "global_sweep": _int(row, "global_sweep"),
                "degree": _int(row, "degree"),
                "degree_min": 0,
                "degree_max": _int(row, "degree"),
                "terms": _int(row, "n_terms"),
                "u_rmse": _float(row, "u_rmse"),
                "v_rmse": _float(row, "v_rmse"),
                "p_rmse": _float(row, "p_rmse"),
                "px_rmse": _float(row, "px_rmse"),
                "py_rmse": _float(row, "py_rmse"),
                "score": _float(row, "score"),
            })
    return sorted(rows, key=lambda r: r["global_sweep"])


def load_window(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "mode": "moving_window",
                "global_sweep": _int(row, "global_sweep"),
                "degree": _int(row, "degree_max"),
                "degree_min": _int(row, "degree_min"),
                "degree_max": _int(row, "degree_max"),
                "terms": _int(row, "window_terms"),
                "cumulative_terms": _int(row, "cumulative_terms"),
                "u_rmse": _float(row, "u_rmse"),
                "v_rmse": _float(row, "v_rmse"),
                "p_rmse": _float(row, "p_rmse"),
                "px_rmse": _float(row, "px_rmse"),
                "py_rmse": _float(row, "py_rmse"),
                "score": _float(row, "score"),
            })
    return sorted(rows, key=lambda r: r["global_sweep"])


def require_rows(name, rows, path):
    if not rows:
        raise SystemExit(f"No rows found for {name}: {path}")


def plot_metrics(series, metrics, title, ylabel, out_path, logy=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, rows in series.items():
        x = [r["global_sweep"] for r in rows]
        for metric in metrics:
            y = [r[metric] for r in rows]
            ax.plot(x, y, label=f"{label} {metric}")

    ax.set_title(title)
    ax.set_xlabel("Global sweep")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_basis(growth_rows, window_rows, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    gx = [r["global_sweep"] for r in growth_rows]
    wx = [r["global_sweep"] for r in window_rows]

    axes[0].step(gx, [r["degree_max"] for r in growth_rows],
                 where="post", label="degree_growth max degree")
    axes[0].step(wx, [r["degree_min"] for r in window_rows],
                 where="post", label="moving_window min degree")
    axes[0].step(wx, [r["degree_max"] for r in window_rows],
                 where="post", label="moving_window max degree")
    axes[0].set_ylabel("Degree")
    axes[0].set_title("Basis Degree Schedule")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].step(gx, [r["terms"] for r in growth_rows],
                 where="post", label="degree_growth active terms")
    axes[1].step(wx, [r["terms"] for r in window_rows],
                 where="post", label="moving_window update terms")
    axes[1].step(wx, [r["cumulative_terms"] for r in window_rows],
                 where="post", label="moving_window cumulative terms")
    axes[1].set_xlabel("Global sweep")
    axes[1].set_ylabel("Terms per field")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_best(label, rows):
    best = min(rows, key=lambda r: r["score"])
    print(
        f"{label} best: sweep={best['global_sweep']} "
        f"score={best['score']:.6e} "
        f"u={best['u_rmse']:.6e} v={best['v_rmse']:.6e} "
        f"p={best['p_rmse']:.6e} px={best['px_rmse']:.6e} py={best['py_rmse']:.6e}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--growth-csv", default=DEFAULT_GROWTH_CSV)
    parser.add_argument("--window-csv", default=DEFAULT_WINDOW_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    growth_rows = load_growth(args.growth_csv)
    window_rows = load_window(args.window_csv)
    require_rows("degree_growth", growth_rows, args.growth_csv)
    require_rows("moving_window", window_rows, args.window_csv)

    os.makedirs(args.out_dir, exist_ok=True)
    series = {
        "degree_growth": growth_rows,
        "moving_window": window_rows,
    }

    rmse_path = os.path.join(args.out_dir, "adaptive_uvp_compare_rmse.png")
    grad_path = os.path.join(args.out_dir, "adaptive_uvp_compare_pressure_grad.png")
    score_path = os.path.join(args.out_dir, "adaptive_uvp_compare_score.png")
    basis_path = os.path.join(args.out_dir, "adaptive_uvp_compare_basis.png")

    plot_metrics(series, ["u_rmse", "v_rmse", "p_rmse"],
                 "Adaptive u/v/p RMSE Comparison", "RMSE", rmse_path)
    plot_metrics(series, ["px_rmse", "py_rmse"],
                 "Adaptive Pressure Gradient RMSE Comparison", "Gradient RMSE", grad_path)
    plot_metrics(series, ["score"],
                 "Adaptive Score Comparison", "Score", score_path)
    plot_basis(growth_rows, window_rows, basis_path)

    print_best("degree_growth", growth_rows)
    print_best("moving_window", window_rows)
    print(f"Saved {rmse_path}")
    print(f"Saved {grad_path}")
    print(f"Saved {score_path}")
    print(f"Saved {basis_path}")


if __name__ == "__main__":
    main()
