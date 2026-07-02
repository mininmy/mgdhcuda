"""
run_compare_growth_hybrid_to15.py
=================================
Run and compare Chebyshev growing-degree and hybrid local/global approaches
up to degree 15.

The script:
  1. Runs adaptive_uvp_degree_growth.py with MAX_DEGREE = 15.
  2. Runs adaptive_uvp_hybrid_local_global.py with MAX_WINDOW_MAX_DEGREE = 15.
  3. Plots RMSE/score against wall-clock time and sweep count.

Useful options:
  --plot-only      only regenerate plots from existing CSV files
  --skip-growth    do not run the growing-degree experiment
  --skip-hybrid    do not run the hybrid experiment

Outputs are written under poly_saves/.
"""

import argparse
import csv
import math
import os

import matplotlib.pyplot as plt

import adaptive_uvp_cheb_degree_growth as growth
import adaptive_uvp_cheb_hybrid_local_global as hybrid


OUT_DIR = "poly_saves"

GROWTH_BASENAME = "adaptive_uvp_cheb_degree_growth_to15"
HYBRID_BASENAME = "adaptive_uvp_cheb_hybrid_local_global_to15"

GROWTH_CSV = os.path.join(OUT_DIR, GROWTH_BASENAME + ".csv")
HYBRID_CSV = os.path.join(OUT_DIR, HYBRID_BASENAME + ".csv")

PLOT_RMSE_TIME = os.path.join(OUT_DIR, "cheb_growth_vs_hybrid_to15_rmse_time.png")
PLOT_SCORE_TIME = os.path.join(OUT_DIR, "cheb_growth_vs_hybrid_to15_score_time.png")
PLOT_RMSE_SWEEP = os.path.join(OUT_DIR, "cheb_growth_vs_hybrid_to15_rmse_sweep.png")
PLOT_SCORE_SWEEP = os.path.join(OUT_DIR, "cheb_growth_vs_hybrid_to15_score_sweep.png")
PLOT_BASIS_TIME = os.path.join(OUT_DIR, "cheb_growth_vs_hybrid_to15_basis_time.png")
PLOT_EXPLOSION_FACTOR = 20.0


def configure_growth():
    growth.MAX_DEGREE = 15
    growth.OUT_BASENAME = GROWTH_BASENAME
    growth.METRICS_CSV = GROWTH_CSV
    growth.SUMMARY_JSON = os.path.join(OUT_DIR, GROWTH_BASENAME + "_summary.json")


def configure_hybrid():
    hybrid.MAX_WINDOW_MAX_DEGREE = 15
    hybrid.OUT_BASENAME = HYBRID_BASENAME
    hybrid.METRICS_CSV = HYBRID_CSV
    hybrid.SUMMARY_JSON = os.path.join(OUT_DIR, HYBRID_BASENAME + "_summary.json")


def read_csv(path, label):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "label": label,
                "global_sweep": int(row["global_sweep"]),
                "u_rmse": float(row["u_rmse"]),
                "v_rmse": float(row["v_rmse"]),
                "p_rmse": float(row["p_rmse"]),
                "px_rmse": float(row["px_rmse"]),
                "py_rmse": float(row["py_rmse"]),
                "score": float(row["score"]),
                "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
            }
            if "degree" in row:
                parsed["degree_min"] = 0
                parsed["degree_max"] = int(row["degree"])
                parsed["active_terms"] = int(row["n_terms"])
            else:
                parsed["degree_min"] = int(row["degree_min"])
                parsed["degree_max"] = int(row["degree_max"])
                parsed["active_terms"] = int(row["active_terms"])
            rows.append(parsed)
    rows.sort(key=lambda r: r["global_sweep"])
    return rows


def require_file(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run experiments first or remove --plot-only.")


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


def plot_metrics(series, x_key, metrics, title, ylabel, out_path, logy=True):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, rows in series.items():
        x = [r[x_key] for r in rows]
        for metric in metrics:
            y = [r[metric] for r in rows]
            ax.plot(x, y, label=f"{label} {metric}")
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


def plot_basis_time(series, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for label, rows in series.items():
        x = [r["elapsed_seconds"] for r in rows]
        axes[0].step(x, [r["degree_max"] for r in rows], where="post", label=f"{label} max degree")
        if label == "hybrid":
            axes[0].step(x, [r["degree_min"] for r in rows], where="post", label="hybrid min degree")
        axes[1].step(x, [r["active_terms"] for r in rows], where="post", label=f"{label} active terms")

    axes[0].set_title("Degree Schedule vs Time")
    axes[0].set_ylabel("Degree")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("Elapsed seconds")
    axes[1].set_ylabel("Active terms per field")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_best(label, rows):
    best = min(rows, key=lambda r: r["score"])
    print(
        f"{label} best: sweep={best['global_sweep']} time={best['elapsed_seconds']:.1f}s "
        f"degree={best['degree_min']}..{best['degree_max']} score={best['score']:.6e} "
        f"u={best['u_rmse']:.6e} v={best['v_rmse']:.6e} "
        f"p={best['p_rmse']:.6e} px={best['px_rmse']:.6e} py={best['py_rmse']:.6e}"
    )


def make_plots(include_hybrid=True):
    require_file(GROWTH_CSV)
    if include_hybrid:
        require_file(HYBRID_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)

    series = {"growth": read_csv(GROWTH_CSV, "growth")}
    if include_hybrid:
        series["hybrid"] = read_csv(HYBRID_CSV, "hybrid")
    for label, rows in series.items():
        if not rows:
            raise SystemExit(f"No rows in {label} CSV")
        print_best(label, rows)

    plot_series = {}
    for label, rows in series.items():
        filtered = stable_prefix(rows)
        if not filtered:
            raise SystemExit(f"No stable rows left after filtering {label}")
        plot_series[label] = filtered
        if len(filtered) != len(rows):
            print(
                f"{label}: plotting first {len(filtered)} of {len(rows)} rows; "
                "dropped exploded/non-finite tail"
            )

    plot_metrics(
        plot_series,
        "elapsed_seconds",
        ["u_rmse", "v_rmse", "p_rmse"],
        "Chebyshev Growth vs Hybrid RMSE by Time",
        "RMSE",
        PLOT_RMSE_TIME,
    )
    plot_metrics(
        plot_series,
        "elapsed_seconds",
        ["score"],
        "Chebyshev Growth vs Hybrid Score by Time",
        "Score",
        PLOT_SCORE_TIME,
    )
    plot_metrics(
        plot_series,
        "global_sweep",
        ["u_rmse", "v_rmse", "p_rmse"],
        "Chebyshev Growth vs Hybrid RMSE by Sweep",
        "RMSE",
        PLOT_RMSE_SWEEP,
    )
    plot_metrics(
        plot_series,
        "global_sweep",
        ["score"],
        "Chebyshev Growth vs Hybrid Score by Sweep",
        "Score",
        PLOT_SCORE_SWEEP,
    )
    plot_basis_time(plot_series, PLOT_BASIS_TIME)

    print(f"Saved {PLOT_RMSE_TIME}")
    print(f"Saved {PLOT_SCORE_TIME}")
    print(f"Saved {PLOT_RMSE_SWEEP}")
    print(f"Saved {PLOT_SCORE_SWEEP}")
    print(f"Saved {PLOT_BASIS_TIME}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-growth", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    configure_growth()
    configure_hybrid()

    if not args.plot_only:
        if not args.skip_growth:
            print("\nRunning Chebyshev growing-degree experiment to degree 15")
            growth.main()
        if not args.skip_hybrid:
            print("\nRunning Chebyshev hybrid local/global experiment to degree 15")
            hybrid.main()

    make_plots(include_hybrid=not args.skip_hybrid)


if __name__ == "__main__":
    main()
