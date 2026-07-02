"""
round_plotter.py — Load and plot boosted-GMDH polynomial rounds vs Taylor-Green.

Reads round_*.json files produced by boosted_residual_gmdh.py (transfer them
first with round_transfer.py) and generates comparison plots of the learned
velocity fields against the analytical Taylor-Green solution.

Usage
-----
  # Show all plot types interactively:
  python round_plotter.py --local-path ./poly_saves

  # Save to disk, only rounds 2 and 3, custom time slices:
  python round_plotter.py --local-path ./poly_saves \
      --rounds 2 3 --t-values 0 0.5 1 --save-dir ./figures

  # Single plot type:
  python round_plotter.py --local-path ./poly_saves --plot rmse

Available --plot values:  heatmap  error  progression  time  rmse  all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── domain matches the training data generator exactly ───────────────────────
NU: float = 0.01
X_LIM = (-1.0, 1.0)
Y_LIM = (-1.0, 1.0)
T_LIM = (0.0, 1.0)

COMPONENTS = ("u", "v")

# ─────────────────────────────────────────────────────────────────────────────
# JSON LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_round(json_path: str | Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def load_all_rounds(local_path: str | Path, rounds: Optional[list[int]] = None) -> dict[int, dict]:
    """Return {round_index: round_dict} for all JSON files found in local_path."""
    local = Path(local_path)
    result: dict[int, dict] = {}
    for f in sorted(local.glob("round_*.json")):
        idx = int(f.stem.split("_")[1])
        if rounds is None or idx in rounds:
            result[idx] = load_round(f)
    if not result:
        raise FileNotFoundError(f"No round_*.json files found in {local}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CPU POLYNOMIAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _eval_polynomial(terms: list[dict], variables: list[str], X: np.ndarray) -> np.ndarray:
    """
    Vectorised CPU evaluation of one polynomial component.

    terms     : list of {"coeff": float, "exponents": {var: int}}
    variables : ordered variable names, e.g. ["x", "y", "t"]
    X         : shape (N, n_vars) in the same variable order

    Returns shape (N,).
    """
    n = len(terms)
    if n == 0:
        return np.zeros(X.shape[0])

    n_vars = len(variables)
    var_idx = {v: i for i, v in enumerate(variables)}

    coeffs = np.empty(n, dtype=np.float64)
    exps = np.zeros((n, n_vars), dtype=np.int32)
    for k, term in enumerate(terms):
        coeffs[k] = term["coeff"]
        for var, e in term["exponents"].items():
            if var in var_idx:
                exps[k, var_idx[var]] = e

    N = X.shape[0]
    monos = np.ones((N, n), dtype=np.float64)
    for j in range(n_vars):
        nz = exps[:, j] > 0
        if not nz.any():
            continue
        monos[:, nz] *= X[:, j : j + 1] ** exps[nz, j]

    return monos @ coeffs


def eval_round(round_data: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Evaluate all velocity components of one round on points X.

    X : shape (N, 3) with columns [x, y, t]

    Returns {component_name: array(N,)}.
    """
    variables = round_data["variables"]
    return {
        comp["name"]: _eval_polynomial(comp["terms"], variables, X)
        for comp in round_data["components"]
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAYLOR-GREEN ANALYTICAL SOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def taylor_green(X: np.ndarray, nu: float = NU) -> dict[str, np.ndarray]:
    """
    Analytical Taylor-Green vortex matching the training data generator.

    X : shape (N, 3) with columns [x, y, t]; x, y ∈ [-1, 1], t ∈ [0, 1]
    """
    x, y, t = X[:, 0], X[:, 1], X[:, 2]
    decay = np.exp(-2.0 * nu * t)
    return {
        "u":  np.sin(x) * np.cos(y) * decay,
        "v": -np.cos(x) * np.sin(y) * decay,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION GRIDS
# ─────────────────────────────────────────────────────────────────────────────

def make_spatial_grid(
    nx: int = 64,
    ny: int = 64,
    t_values: Optional[list[float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Build a flat evaluation grid over the spatial domain at given time slices.

    Returns (X_flat, x1d, y1d, t_vals) where X_flat has shape (N_total, 3).
    """
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    x1d = np.linspace(*X_LIM, nx)
    y1d = np.linspace(*Y_LIM, ny)
    Xg, Yg = np.meshgrid(x1d, y1d, indexing="ij")
    slices = []
    for tv in t_values:
        Tg = np.full_like(Xg, tv)
        slices.append(np.stack([Xg.ravel(), Yg.ravel(), Tg.ravel()], axis=1))
    return np.concatenate(slices, axis=0), x1d, y1d, list(t_values)


def make_time_grid(x0: float, y0: float, nt: int = 200) -> np.ndarray:
    """Single spatial probe point swept over t ∈ [0, 1]. Returns shape (nt, 3)."""
    t1d = np.linspace(*T_LIM, nt)
    return np.column_stack([np.full(nt, x0), np.full(nt, y0), t1d])


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_VEL = "RdBu_r"
_CMAP_ERR = "bwr"


def _vel_norm(arr: np.ndarray) -> TwoSlopeNorm:
    absmax = max(abs(arr.min()), abs(arr.max()), 1e-12)
    return TwoSlopeNorm(vcenter=0.0, vmin=-absmax, vmax=absmax)


def plot_heatmaps(
    rounds_data: dict[int, dict],
    t_values: Optional[list[float]] = None,
    nx: int = 64,
    ny: int = 64,
    nu: float = NU,
    save_dir: Optional[str] = None,
) -> None:
    """
    Per round, per component: [Truth | Prediction | Signed error] at each time slice.
    """
    if t_values is None:
        t_values = [0.0, 0.5, 1.0]

    X_flat, _, _, t_vals = make_spatial_grid(nx, ny, t_values)
    n_pts = nx * ny
    tg = taylor_green(X_flat, nu)
    ext = [*Y_LIM, *X_LIM]

    for r_idx, rd in sorted(rounds_data.items()):
        pred = eval_round(rd, X_flat)
        print(f"  Round {r_idx}: evaluated {X_flat.shape[0]:,} points")

        for comp in COMPONENTS:
            truth_arr = tg[comp]
            pred_arr = pred.get(comp, np.zeros(X_flat.shape[0]))
            n_t = len(t_vals)

            fig, axes = plt.subplots(n_t, 3, figsize=(12, 3.5 * n_t), squeeze=False)
            fig.suptitle(
                f"Round {r_idx} — {comp}  (RMSE = {_rmse(truth_arr, pred_arr):.4e})",
                fontsize=13, fontweight="bold",
            )

            for ti, tv in enumerate(t_vals):
                sl = slice(ti * n_pts, (ti + 1) * n_pts)
                tr_2d = truth_arr[sl].reshape(nx, ny)
                pr_2d = pred_arr[sl].reshape(nx, ny)
                er_2d = pr_2d - tr_2d

                norm_v = _vel_norm(tr_2d)
                emax = np.abs(er_2d).max()
                norm_e = TwoSlopeNorm(vcenter=0, vmin=-emax, vmax=emax)

                im0 = axes[ti, 0].imshow(tr_2d, origin="lower", extent=ext,
                                          cmap=_CMAP_VEL, norm=norm_v, aspect="auto")
                axes[ti, 0].set_title(f"Taylor-Green  t={tv:.2f}")
                fig.colorbar(im0, ax=axes[ti, 0], shrink=0.85)

                im1 = axes[ti, 1].imshow(pr_2d, origin="lower", extent=ext,
                                          cmap=_CMAP_VEL, norm=norm_v, aspect="auto")
                axes[ti, 1].set_title(f"GMDH round {r_idx}  t={tv:.2f}")
                fig.colorbar(im1, ax=axes[ti, 1], shrink=0.85)

                im2 = axes[ti, 2].imshow(er_2d, origin="lower", extent=ext,
                                          cmap=_CMAP_ERR, norm=norm_e, aspect="auto")
                axes[ti, 2].set_title(f"Error (pred − truth)  t={tv:.2f}")
                fig.colorbar(im2, ax=axes[ti, 2], shrink=0.85)

                for ax in axes[ti]:
                    ax.set_xlabel("y")
                    ax.set_ylabel("x")

            fig.tight_layout()
            _save_or_show(fig, save_dir, f"heatmap_round{r_idx}_{comp}.png")


def plot_error_maps(
    rounds_data: dict[int, dict],
    t_values: Optional[list[float]] = None,
    nx: int = 64,
    ny: int = 64,
    nu: float = NU,
    save_dir: Optional[str] = None,
) -> None:
    """
    Absolute-error heatmaps for all rounds on one figure per (component, time slice),
    sharing a common colour scale so spatial improvement across rounds is visible.
    """
    if t_values is None:
        t_values = [0.0, 0.5, 1.0]

    X_flat, _, _, t_vals = make_spatial_grid(nx, ny, t_values)
    n_pts = nx * ny
    tg = taylor_green(X_flat, nu)
    sorted_rounds = sorted(rounds_data)
    round_preds = {r: eval_round(rounds_data[r], X_flat) for r in sorted_rounds}

    for comp in COMPONENTS:
        for ti, tv in enumerate(t_vals):
            sl = slice(ti * n_pts, (ti + 1) * n_pts)
            truth_2d = tg[comp][sl].reshape(nx, ny)

            all_err = np.concatenate([
                np.abs(round_preds[r].get(comp, np.zeros(n_pts))[sl] - tg[comp][sl])
                for r in sorted_rounds
            ])
            vmax = np.percentile(all_err, 98)

            n_r = len(sorted_rounds)
            fig, axes = plt.subplots(1, n_r, figsize=(4.5 * n_r, 4), squeeze=False)
            fig.suptitle(
                f"|pred − truth|  —  {comp}  t={tv:.2f}",
                fontsize=12, fontweight="bold",
            )

            for ci, r in enumerate(sorted_rounds):
                pr_2d = round_preds[r].get(comp, np.zeros(n_pts))[sl].reshape(nx, ny)
                im = axes[0, ci].imshow(
                    np.abs(pr_2d - truth_2d), origin="lower",
                    extent=[*Y_LIM, *X_LIM],
                    cmap="hot_r", vmin=0, vmax=vmax, aspect="auto",
                )
                axes[0, ci].set_title(f"Round {r}\nRMSE={_rmse(truth_2d.ravel(), pr_2d.ravel()):.3e}")
                axes[0, ci].set_xlabel("y")
                axes[0, ci].set_ylabel("x")
                fig.colorbar(im, ax=axes[0, ci], shrink=0.85)

            fig.tight_layout()
            _save_or_show(fig, save_dir, f"error_map_{comp}_t{tv:.2f}.png")


def plot_round_progression(
    rounds_data: dict[int, dict],
    t_fixed: float = 0.5,
    nx: int = 64,
    ny: int = 64,
    nu: float = NU,
    save_dir: Optional[str] = None,
) -> None:
    """
    Truth + all rounds side-by-side at a fixed time slice, one figure per component.
    """
    X_flat, _, _, _ = make_spatial_grid(nx, ny, [t_fixed])
    tg = taylor_green(X_flat, nu)
    sorted_rounds = sorted(rounds_data)
    ext = [*Y_LIM, *X_LIM]

    for comp in COMPONENTS:
        truth_2d = tg[comp].reshape(nx, ny)
        norm_v = _vel_norm(truth_2d)
        n_cols = 1 + len(sorted_rounds)

        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), squeeze=False)
        fig.suptitle(f"Round progression — {comp}   t = {t_fixed:.2f}",
                     fontsize=12, fontweight="bold")

        im = axes[0, 0].imshow(truth_2d, origin="lower", extent=ext,
                                cmap=_CMAP_VEL, norm=norm_v, aspect="auto")
        axes[0, 0].set_title("Taylor-Green\n(truth)")
        axes[0, 0].set_xlabel("y")
        axes[0, 0].set_ylabel("x")
        fig.colorbar(im, ax=axes[0, 0], shrink=0.85)

        for ci, r in enumerate(sorted_rounds):
            pred = eval_round(rounds_data[r], X_flat)
            pr_2d = pred.get(comp, np.zeros(nx * ny)).reshape(nx, ny)
            im = axes[0, ci + 1].imshow(pr_2d, origin="lower", extent=ext,
                                         cmap=_CMAP_VEL, norm=norm_v, aspect="auto")
            axes[0, ci + 1].set_title(f"Round {r}\nRMSE={_rmse(truth_2d.ravel(), pr_2d.ravel()):.3e}")
            axes[0, ci + 1].set_xlabel("y")
            axes[0, ci + 1].set_ylabel("x")
            fig.colorbar(im, ax=axes[0, ci + 1], shrink=0.85)

        fig.tight_layout()
        _save_or_show(fig, save_dir, f"progression_{comp}_t{t_fixed:.2f}.png")


def plot_time_evolution(
    rounds_data: dict[int, dict],
    spatial_points: Optional[list[tuple[float, float]]] = None,
    nt: int = 200,
    nu: float = NU,
    save_dir: Optional[str] = None,
) -> None:
    """
    u(t) and v(t) at fixed (x, y) probe points, all rounds overlaid on the truth.
    """
    if spatial_points is None:
        spatial_points = [(0.5, 0.5), (-0.5, 0.5), (0.0, 0.0)]

    t1d = np.linspace(*T_LIM, nt)
    sorted_rounds = sorted(rounds_data)
    cmap_r = plt.get_cmap("tab10")

    for x0, y0 in spatial_points:
        X_line = make_time_grid(x0, y0, nt)
        tg = taylor_green(X_line, nu)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Time evolution at (x={x0:.2f}, y={y0:.2f})", fontsize=12)

        for ai, comp in enumerate(COMPONENTS):
            ax = axes[ai]
            ax.plot(t1d, tg[comp], "k-", lw=2, label="Taylor-Green", zorder=10)
            for ci, r in enumerate(sorted_rounds):
                pred = eval_round(rounds_data[r], X_line)
                ax.plot(t1d, pred.get(comp, np.zeros(nt)),
                        color=cmap_r(ci), lw=1.4, linestyle="--",
                        label=f"Round {r}", alpha=0.85)
            ax.set_xlabel("t")
            ax.set_ylabel(comp)
            ax.set_title(f"Component {comp}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        _save_or_show(fig, save_dir, f"time_evol_x{x0:.2f}_y{y0:.2f}.png")


def plot_rmse_summary(
    rounds_data: dict[int, dict],
    nx: int = 80,
    ny: int = 80,
    t_values: Optional[list[float]] = None,
    nu: float = NU,
    save_dir: Optional[str] = None,
) -> None:
    """
    Grouped bar chart of RMSE per round/time-slice and a mean-RMSE line across rounds.
    """
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    X_flat, _, _, t_vals = make_spatial_grid(nx, ny, t_values)
    n_pts = nx * ny
    tg = taylor_green(X_flat, nu)
    sorted_rounds = sorted(rounds_data)

    rmse_table: dict[str, dict[int, list[float]]] = {c: {} for c in COMPONENTS}
    for r in sorted_rounds:
        pred = eval_round(rounds_data[r], X_flat)
        for comp in COMPONENTS:
            rmse_table[comp][r] = [
                _rmse(tg[comp][ti * n_pts:(ti + 1) * n_pts],
                      pred.get(comp, np.zeros(n_pts))[ti * n_pts:(ti + 1) * n_pts])
                for ti in range(len(t_vals))
            ]

    cmap_t = plt.get_cmap("plasma")
    n_r = len(sorted_rounds)
    n_t = len(t_vals)

    for comp in COMPONENTS:
        fig, ax = plt.subplots(figsize=(max(7, 2 * n_r), 4.5))
        x_pos = np.arange(n_r)
        width = 0.8 / n_t
        for ti, tv in enumerate(t_vals):
            ax.bar(x_pos + ti * width,
                   [rmse_table[comp][r][ti] for r in sorted_rounds],
                   width, label=f"t={tv:.2f}",
                   color=cmap_t(ti / max(n_t - 1, 1)), alpha=0.85)
        ax.set_xticks(x_pos + width * (n_t - 1) / 2)
        ax.set_xticklabels([f"Round {r}" for r in sorted_rounds])
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE per round — {comp}")
        ax.legend(title="time slice", fontsize=8)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save_or_show(fig, save_dir, f"rmse_bars_{comp}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for comp in COMPONENTS:
        mean_rmse = [np.mean(rmse_table[comp][r]) for r in sorted_rounds]
        ax.plot(sorted_rounds, mean_rmse, "o-", lw=2, label=comp)
        for r, v in zip(sorted_rounds, mean_rmse):
            ax.annotate(f"{v:.2e}", (r, v), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Round index")
    ax.set_ylabel("Mean RMSE (over time slices)")
    ax.set_title("RMSE improvement across boosting rounds")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_dir, "rmse_summary.png")


def plot_all(
    rounds_data: dict[int, dict],
    nu: float = NU,
    save_dir: Optional[str] = None,
) -> None:
    """Run every plot type in sequence."""
    print("→ Heatmaps…")
    plot_heatmaps(rounds_data, nu=nu, save_dir=save_dir)
    print("→ Error maps…")
    plot_error_maps(rounds_data, nu=nu, save_dir=save_dir)
    print("→ Round progression…")
    plot_round_progression(rounds_data, nu=nu, save_dir=save_dir)
    print("→ Time evolution…")
    plot_time_evolution(rounds_data, nu=nu, save_dir=save_dir)
    print("→ RMSE summary…")
    plot_rmse_summary(rounds_data, nu=nu, save_dir=save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _save_or_show(fig: plt.Figure, save_dir: Optional[str], fname: str) -> None:
    if save_dir:
        out = Path(save_dir) / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  saved → {out}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

PLOT_CHOICES = ("heatmap", "error", "progression", "time", "rmse", "all")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot boosted-GMDH polynomial rounds vs Taylor-Green.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--local-path", metavar="DIR", default="./poly_saves",
                   help="Directory containing round_*.json (default: ./poly_saves)")
    p.add_argument("--rounds", metavar="N", nargs="+", type=int,
                   help="Round indices to include (default: all found)")
    p.add_argument("--nu", type=float, default=NU,
                   help=f"Kinematic viscosity (default: {NU})")
    p.add_argument("--plot", choices=PLOT_CHOICES, default="all",
                   help="Plot type (default: all)")
    p.add_argument("--save-dir", metavar="DIR",
                   help="Save figures here instead of displaying them")
    p.add_argument("--t-values", metavar="T", nargs="+", type=float,
                   help="Time slices for spatial plots (default: 0 0.25 0.5 0.75 1)")
    p.add_argument("--nx", type=int, default=64, help="Grid points in x (default: 64)")
    p.add_argument("--ny", type=int, default=64, help="Grid points in y (default: 64)")
    p.add_argument("--probe-points", metavar="X,Y", nargs="+",
                   help='Probe points for time-evolution, e.g. "0.5,0.5" "-0.5,0.5"')
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Loading rounds from {args.local_path} …")
    rounds_data = load_all_rounds(args.local_path, args.rounds)
    print(f"  Found rounds: {sorted(rounds_data)}")
    for r, rd in sorted(rounds_data.items()):
        for comp in rd["components"]:
            print(f"  round {r}  {comp['name']}: {comp['n_terms']} terms")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    probe_pts = None
    if args.probe_points:
        probe_pts = [tuple(float(v) for v in pt.split(",")) for pt in args.probe_points]

    kwargs = dict(nu=args.nu, save_dir=args.save_dir)

    if args.plot == "all":
        plot_all(rounds_data, **kwargs)
    elif args.plot == "heatmap":
        plot_heatmaps(rounds_data, t_values=args.t_values, nx=args.nx, ny=args.ny, **kwargs)
    elif args.plot == "error":
        plot_error_maps(rounds_data, t_values=args.t_values, nx=args.nx, ny=args.ny, **kwargs)
    elif args.plot == "progression":
        for tv in (args.t_values or [0.5]):
            plot_round_progression(rounds_data, t_fixed=tv, nx=args.nx, ny=args.ny, **kwargs)
    elif args.plot == "time":
        plot_time_evolution(rounds_data, spatial_points=probe_pts, **kwargs)
    elif args.plot == "rmse":
        plot_rmse_summary(rounds_data, nx=args.nx, ny=args.ny, t_values=args.t_values, **kwargs)


if __name__ == "__main__":
    plt.figure()
    plt.plot([1,2,3],[1,4,9])
    plt.show()
    #main()
