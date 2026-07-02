"""
run_cheb_growth_to15.py
=======================
Run only the Chebyshev growing-degree approach up to degree 15 and plot it.

This is the growth-only version of run_compare_cheb_growth_hybrid_to15.py.

Useful options:
  --plot-only      only regenerate plots from the existing CSV

Outputs are written under poly_saves/.
"""

import argparse
import csv
import math
import os

import matplotlib.pyplot as plt

import adaptive_uvp_cheb_degree_growth as growth


OUT_DIR = "poly_saves"
GROWTH_BASENAME = "adaptive_uvp_cheb_degree_growth_to20"
GROWTH_CSV = os.path.join(OUT_DIR, GROWTH_BASENAME + ".csv")

PLOT_RMSE_TIME = os.path.join(OUT_DIR, "cheb_growth_to20_rmse_time.png")
PLOT_SCORE_TIME = os.path.join(OUT_DIR, "cheb_growth_to20_score_time.png")
PLOT_RMSE_SWEEP = os.path.join(OUT_DIR, "cheb_growth_to20_rmse_sweep.png")
PLOT_SCORE_SWEEP = os.path.join(OUT_DIR, "cheb_growth_to20_score_sweep.png")
PLOT_BASIS_TIME = os.path.join(OUT_DIR, "cheb_growth_to20_basis_time.png")
PLOT_EXPLOSION_FACTOR = 20.0


def configure_growth():
    growth.MAX_DEGREE = 20
    growth.OUT_BASENAME = GROWTH_BASENAME
    growth.METRICS_CSV = GROWTH_CSV
    growth.SUMMARY_JSON = os.path.join(OUT_DIR, GROWTH_BASENAME + "_summary.json")


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "global_sweep": int(row["global_sweep"]),
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
    axes[0].set_title("Chebyshev Growth Degree Schedule")
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
        f"Cheb growth best: sweep={best['global_sweep']} "
        f"time={best['elapsed_seconds']:.1f}s degree={best['degree']} "
        f"score={best['score']:.6e} u={best['u_rmse']:.6e} "
        f"v={best['v_rmse']:.6e} p={best['p_rmse']:.6e} "
        f"px={best['px_rmse']:.6e} py={best['py_rmse']:.6e}"
    )


def make_plots():
    if not os.path.exists(GROWTH_CSV):
        raise SystemExit(f"Missing {GROWTH_CSV}. Run without --plot-only first.")
    rows = read_csv(GROWTH_CSV)
    if not rows:
        raise SystemExit(f"No rows in {GROWTH_CSV}")
    print_best(rows)
    plot_rows = stable_prefix(rows)
    if len(plot_rows) != len(rows):
        print(
            f"Cheb growth: plotting first {len(plot_rows)} of {len(rows)} rows; "
            "dropped exploded/non-finite tail"
        )

    plot_metrics(
        plot_rows,
        "elapsed_seconds",
        ["u_rmse", "v_rmse", "p_rmse"],
        "Chebyshev Growth RMSE by Time",
        "RMSE",
        PLOT_RMSE_TIME,
    )
    plot_metrics(
        plot_rows,
        "elapsed_seconds",
        ["score"],
        "Chebyshev Growth Score by Time",
        "Score",
        PLOT_SCORE_TIME,
    )
    plot_metrics(
        plot_rows,
        "global_sweep",
        ["u_rmse", "v_rmse", "p_rmse"],
        "Chebyshev Growth RMSE by Sweep",
        "RMSE",
        PLOT_RMSE_SWEEP,
    )
    plot_metrics(
        plot_rows,
        "global_sweep",
        ["score"],
        "Chebyshev Growth Score by Sweep",
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
    configure_growth()

    if not args.plot_only:
        print("\nRunning Chebyshev growing-degree experiment to degree 20")
        growth.main()

    make_plots()


if __name__ == "__main__":
    main()
