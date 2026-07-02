"""
plot_uvp_stage_compare.py
=========================
Plot per-sweep diagnostics saved by compare_uvp_stage_runs.py.

Input:
    poly_saves/uvp_stage_compare_5_to_8_vs_8_to_9.csv

Outputs:
    poly_saves/uvp_stage_compare_rmse.png
    poly_saves/uvp_stage_compare_pressure_grad.png
    poly_saves/uvp_stage_compare_score.png
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


DEFAULT_CSV = "poly_saves/uvp_stage_compare_5_to_8_vs_8_to_9.csv"
DEFAULT_OUT_DIR = "poly_saves"


def load_rows(csv_path):
    runs = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"]
            parsed = {
                "global_sweep": int(row["global_sweep"]),
                "stage": int(row["stage"]),
                "stage_sweep": int(row["stage_sweep"]),
                "degree": int(row["degree"]),
                "u_rmse": float(row["u_rmse"]),
                "v_rmse": float(row["v_rmse"]),
                "p_rmse": float(row["p_rmse"]),
                "px_rmse": float(row["px_rmse"]),
                "py_rmse": float(row["py_rmse"]),
                "score": float(row["score"]),
            }
            runs[run].append(parsed)

    for rows in runs.values():
        rows.sort(key=lambda r: r["global_sweep"])
    return dict(runs)


def add_stage_lines(ax, rows):
    seen = set()
    for row in rows:
        key = (row["stage"], row["global_sweep"])
        if row["stage_sweep"] == 0 and row["global_sweep"] != 0 and key not in seen:
            ax.axvline(row["global_sweep"], color="0.75", linewidth=1, linestyle="--")
            seen.add(key)


def plot_group(runs, metrics, title, ylabel, out_path, logy=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    for run, rows in runs.items():
        x = [r["global_sweep"] for r in rows]
        for metric in metrics:
            y = [r[metric] for r in rows]
            ax.plot(x, y, label=f"{run} {metric}")

    if runs:
        add_stage_lines(ax, next(iter(runs.values())))

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


def plot_degrees(runs, out_path):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for run, rows in runs.items():
        x = [r["global_sweep"] for r in rows]
        y = [r["degree"] for r in rows]
        ax.step(x, y, where="post", label=run)

    ax.set_title("Degree Schedule")
    ax.set_xlabel("Global sweep")
    ax.set_ylabel("Degree")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Comparison CSV path")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for PNG outputs")
    args = parser.parse_args()

    runs = load_rows(args.csv)
    if not runs:
        raise SystemExit(f"No rows found in {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    rmse_path = os.path.join(args.out_dir, "uvp_stage_compare_rmse.png")
    grad_path = os.path.join(args.out_dir, "uvp_stage_compare_pressure_grad.png")
    score_path = os.path.join(args.out_dir, "uvp_stage_compare_score.png")
    degree_path = os.path.join(args.out_dir, "uvp_stage_compare_degree.png")

    plot_group(
        runs,
        ["u_rmse", "v_rmse", "p_rmse"],
        "u/v/p RMSE by Sweep",
        "RMSE",
        rmse_path,
    )
    plot_group(
        runs,
        ["px_rmse", "py_rmse"],
        "Pressure Gradient RMSE by Sweep",
        "Gradient RMSE",
        grad_path,
    )
    plot_group(
        runs,
        ["score"],
        "Comparison Score by Sweep",
        "Score",
        score_path,
    )
    plot_degrees(runs, degree_path)

    print(f"Saved {rmse_path}")
    print(f"Saved {grad_path}")
    print(f"Saved {score_path}")
    print(f"Saved {degree_path}")


if __name__ == "__main__":
    main()
