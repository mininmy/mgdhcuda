"""
run_cheb_scheduled_growth.py
============================
Run the scheduled Chebyshev growth experiment and plot its metrics.

Default scheduled solver:
  degree <= 8 -> 10 -> 12 -> 14 -> 15

Useful options:
  --plot-only      only regenerate plots from the existing CSV

Outputs are written under poly_saves/.
"""

import argparse
import csv
import math
import os

import matplotlib.pyplot as plt

import adaptive_uvp_cheb_scheduled_growth as scheduled


OUT_DIR = "poly_saves"
BASENAME = scheduled.OUT_BASENAME
METRICS_CSV = scheduled.METRICS_CSV

PLOT_RMSE_TIME = os.path.join(OUT_DIR, BASENAME + "_rmse_time.png")
PLOT_SCORE_TIME = os.path.join(OUT_DIR, BASENAME + "_score_time.png")
PLOT_RMSE_SWEEP = os.path.join(OUT_DIR, BASENAME + "_rmse_sweep.png")
PLOT_SCORE_SWEEP = os.path.join(OUT_DIR, BASENAME + "_score_sweep.png")
PLOT_BASIS_TIME = os.path.join(OUT_DIR, BASENAME + "_basis_time.png")
PLOT_EXPLOSION_FACTOR = 20.0


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "global_sweep": int(row["global_sweep"]),
                "stage": int(row["stage"]),
                "stage_sweep": int(row["stage_sweep"]),
                "degree": int(row["degree"]),
                "n_terms": int(row["n_terms"]),
                "u_rmse": float(row["u_rmse"]),
                "v_rmse": float(row["v_rmse"]),
                "p_rmse": float(row["p_rmse"]),
                "px_rmse": float(row["px_rmse"]),
                "py_rmse": float(row["py_rmse"]),
                "score": float(row["score"]),
                "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
            })
    rows.sort(key=lambda r: r["global_sweep"])
    return rows


def stable_prefix(rows, explosion_factor=PLOT_EXPLOSION_FACTOR):
    stable = []
    best_score = None
    for row in rows:
        score = row["score"]
        if not math.isfinite(score) or score <= 0.0:
            break
        if best_score is not None and score > best_score * explosion_factor:
            break
        stable.append(row)
        if best_score is None or score < best_score:
            best_score = score
    return stable


def plot_metrics(rows, x_key, metrics, title, ylabel, out_path, logy=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = [r[x_key] for r in rows]
    for metric in metrics:
        ax.plot(x, [r[metric] for r in rows], label=metric)
    ax.set_title(title)
    ax.set_xlabel("Elapsed seconds" if x_key == "elapsed_seconds" else "Global sweep")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_basis(rows, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = [r["elapsed_seconds"] for r in rows]
    axes[0].step(x, [r["degree"] for r in rows], where="post")
    axes[0].set_title("Scheduled Chebyshev Growth Basis")
    axes[0].set_ylabel("Degree")
    axes[0].grid(True, alpha=0.25)
    axes[1].step(x, [r["n_terms"] for r in rows], where="post")
    axes[1].set_xlabel("Elapsed seconds")
    axes[1].set_ylabel("Terms per field")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_best(rows):
    best = min(rows, key=lambda r: r["score"])
    print(
        f"Cheb scheduled best: sweep={best['global_sweep']} "
        f"time={best['elapsed_seconds']:.1f}s stage={best['stage']} "
        f"degree={best['degree']} score={best['score']:.6e} "
        f"u={best['u_rmse']:.6e} v={best['v_rmse']:.6e} "
        f"p={best['p_rmse']:.6e} px={best['px_rmse']:.6e} "
        f"py={best['py_rmse']:.6e}"
    )


def make_plots():
    if not os.path.exists(METRICS_CSV):
        raise SystemExit(f"Missing {METRICS_CSV}. Run without --plot-only first.")
    rows = read_csv(METRICS_CSV)
    if not rows:
        raise SystemExit(f"No rows in {METRICS_CSV}")

    print_best(rows)
    plot_rows = stable_prefix(rows)
    if len(plot_rows) != len(rows):
        print(
            f"Cheb scheduled: plotting first {len(plot_rows)} of {len(rows)} rows; "
            "dropped exploded/non-finite tail"
        )

    plot_metrics(
        plot_rows,
        "elapsed_seconds",
        ["u_rmse", "v_rmse", "p_rmse"],
        "Scheduled Chebyshev Growth RMSE by Time",
        "RMSE",
        PLOT_RMSE_TIME,
    )
    plot_metrics(
        plot_rows,
        "elapsed_seconds",
        ["score"],
        "Scheduled Chebyshev Growth Score by Time",
        "Score",
        PLOT_SCORE_TIME,
    )
    plot_metrics(
        plot_rows,
        "global_sweep",
        ["u_rmse", "v_rmse", "p_rmse"],
        "Scheduled Chebyshev Growth RMSE by Sweep",
        "RMSE",
        PLOT_RMSE_SWEEP,
    )
    plot_metrics(
        plot_rows,
        "global_sweep",
        ["score"],
        "Scheduled Chebyshev Growth Score by Sweep",
        "Score",
        PLOT_SCORE_SWEEP,
    )
    plot_basis(plot_rows, PLOT_BASIS_TIME)

    print(f"Saved {PLOT_RMSE_TIME}")
    print(f"Saved {PLOT_SCORE_TIME}")
    print(f"Saved {PLOT_RMSE_SWEEP}")
    print(f"Saved {PLOT_SCORE_SWEEP}")
    print(f"Saved {PLOT_BASIS_TIME}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.plot_only:
        print("\nRunning scheduled Chebyshev growth experiment")
        scheduled.main()

    make_plots()


if __name__ == "__main__":
    main()
