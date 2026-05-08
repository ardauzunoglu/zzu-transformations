"""
Cost analysis for the ZZU benchmark.

For every (dataset, method, seed) combination measured by run_comparison.py,
this script also records:
  - wall-clock fit time
  - number of optimizer iterations
  - number of model_fn evaluations (captures Jacobian work, since numerical
    Jacobian calls model_fn 2p times per Jacobian)

It additionally fits "BFGS_warmstart": BFGS started from the
transformation-derived theta_init that ZZU's screening would produce.
Comparing that against the regular "BFGS" row (heuristic init) tells us
whether ZZU's screening step earns back its cost in faster convergence.

Outputs (./comparison_results/):
  - cost_results.csv      long-form: 1 row per (dataset, method, seed)
  - cost_summary.csv      mean fit_time / n_iter / n_model_evals per method
  - cost_pareto.png       log RMSE vs log fit time, one panel per dataset
  - warm_vs_cold.png      BFGS cold vs warm-start: iterations, calls, time

Usage: python cost_analysis.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transformation_algorithms as ta

from run_comparison import (
    DATASET_SPECS,
    FAMILY_COLOR,
    LINEAR_FAMILY,
    NONLINEAR_FAMILY,
    OUTPUT_DIR,
    ZZU_FAMILY,
    make_transformed_suite,
    make_zzu,
)
# Pull seeds and split size from the central registry so all benchmarks
# stay synchronized when the project bumps N_SEEDS or TEST_FRACTION.
from reproducibility import N_SEEDS, TEST_FRACTION


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

class CountingModelFn:
    """Wrap a model_fn so that each call increments a counter."""
    __slots__ = ("fn", "n_calls")

    def __init__(self, fn: Callable):
        self.fn = fn
        self.n_calls = 0

    def __call__(self, X, theta):
        self.n_calls += 1
        return self.fn(X, theta)


def _time(fn: Callable) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _safe_predict(predict_fn: Callable, X_test: np.ndarray) -> Optional[np.ndarray]:
    try:
        return np.asarray(predict_fn(X_test), dtype=float).ravel()
    except Exception:
        return None


def _rmse(y_true: np.ndarray, pred: Optional[np.ndarray]) -> float:
    if pred is None or not np.all(np.isfinite(pred)):
        return float("nan")
    return float(ta.regression_metrics(y_true, pred)["rmse"])


def cost_row(
    dataset: str,
    method: str,
    family: str,
    seed: int,
    *,
    fit_time: float,
    n_iter: Optional[int],
    n_calls: Optional[int],
    rmse: float,
    converged: Optional[bool],
    error: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "dataset": dataset,
        "method": method,
        "family": family,
        "seed": seed,
        "fit_time_sec": fit_time,
        "n_iter": n_iter,
        "n_model_evals": n_calls,
        "rmse": rmse,
        "converged": converged,
        "error": error,
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def _make_nonlinear(name: str, model_fn: Callable):
    if name == "GD":
        return ta.GradientDescentRegressor(
            model_fn=model_fn, learning_rate=1e-4, decay=0.9999, max_iter=5000
        )
    if name == "GN":
        return ta.GaussNewtonRegressor(model_fn=model_fn, max_iter=200)
    if name == "BFGS":
        return ta.BFGSRegressor(model_fn=model_fn, max_iter=500)
    raise ValueError(name)


def evaluate_costs_for_split(
    dataset: str,
    spec: Dict[str, Any],
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # --- Linearized OLS suite ------------------------------------------------
    # Pure linear algebra; no model_fn, no iteration count.
    for name, model in make_transformed_suite().items():
        try:
            t = _time(lambda m=model: m.fit(X_tr, y_tr))
            pred = model.predict(X_te)
            rows.append(cost_row(
                dataset, name, LINEAR_FAMILY, seed,
                fit_time=t, n_iter=None, n_calls=None,
                rmse=_rmse(y_te, pred), converged=None,
            ))
        except Exception as exc:
            rows.append(cost_row(
                dataset, name, LINEAR_FAMILY, seed,
                fit_time=float("nan"), n_iter=None, n_calls=None,
                rmse=float("nan"), converged=None, error=str(exc),
            ))

    # --- Nonlinear regressors with heuristic init (cold start) ---------------
    theta_init = spec["theta_init_fn"](X_tr, y_tr)
    for name in ("GD", "GN", "BFGS"):
        counter = CountingModelFn(spec["model_fn"])
        reg = _make_nonlinear(name, counter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t = _time(lambda r=reg: r.fit(X_tr, y_tr, theta_init))
            pred = _safe_predict(reg.predict, X_te)
        rows.append(cost_row(
            dataset, name, NONLINEAR_FAMILY, seed,
            fit_time=t, n_iter=reg.n_iter_, n_calls=counter.n_calls,
            rmse=_rmse(y_te, pred), converged=reg.converged_,
            error=reg.fit_error_ or "",
            extra={"theta_init_source": "heuristic"},
        ))

    # --- ZZU hybrid (full pipeline) ------------------------------------------
    counter = CountingModelFn(spec["model_fn"])
    spec_for_zzu = dict(spec)
    spec_for_zzu["model_fn"] = counter
    zzu = make_zzu(spec_for_zzu, X_tr, y_tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t = _time(lambda z=zzu: z.fit(X_tr, y_tr))
        pred = _safe_predict(zzu.predict, X_te)
    nonlin = zzu.nonlinear_regressor_
    rows.append(cost_row(
        dataset, "ZZU", ZZU_FAMILY, seed,
        fit_time=t,
        n_iter=getattr(nonlin, "n_iter_", None),
        n_calls=counter.n_calls,
        rmse=_rmse(y_te, pred),
        converged=getattr(nonlin, "converged_", None),
        error=zzu.fit_error_ or "",
        extra={"zzu_best_transform": zzu.best_transform_name_},
    ))

    # --- Warm-start BFGS (only when coeff_to_init exists) --------------------
    # Same target as cold BFGS, but seeded from the transformation-derived
    # init that ZZU's screening produces.  Tells us whether the screening
    # overhead pays for itself in faster nonlinear convergence.
    if spec["zzu_coeff_to_init"] is not None and spec["zzu_transformations"]:
        warm_theta = None
        for transformation_name, transformation_model in spec["zzu_transformations"].items():
            try:
                transformation_model.fit(X_tr, y_tr)
                warm_theta = np.asarray(
                    spec["zzu_coeff_to_init"](transformation_model), dtype=float
                ).ravel()
                break
            except Exception:
                continue
        if warm_theta is not None and np.all(np.isfinite(warm_theta)):
            counter = CountingModelFn(spec["model_fn"])
            reg = ta.BFGSRegressor(model_fn=counter, max_iter=500)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                t = _time(lambda r=reg: r.fit(X_tr, y_tr, warm_theta))
                pred = _safe_predict(reg.predict, X_te)
            rows.append(cost_row(
                dataset, "BFGS_warmstart", NONLINEAR_FAMILY, seed,
                fit_time=t, n_iter=reg.n_iter_, n_calls=counter.n_calls,
                rmse=_rmse(y_te, pred), converged=reg.converged_,
                error=reg.fit_error_ or "",
                extra={"theta_init_source": "zzu_screen"},
            ))

    return rows


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------

def summarize_costs(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(["dataset", "method", "family"], dropna=False)
    out = g.agg(
        n=("rmse", "size"),
        mean_rmse=("rmse", "mean"),
        mean_fit_time_sec=("fit_time_sec", "mean"),
        std_fit_time_sec=("fit_time_sec", "std"),
        mean_n_iter=("n_iter", "mean"),
        mean_n_model_evals=("n_model_evals", "mean"),
    ).reset_index()
    return out.sort_values(
        ["dataset", "mean_rmse"], na_position="last"
    ).reset_index(drop=True)


def _draw_pareto_panel(ax, sub: pd.DataFrame, *, dataset: str,
                       show_legend: bool) -> None:
    """Draw a single dataset's Pareto plot on ax, with adjustText
    auto-placement of method labels to avoid overlap.  Used by both the
    combined and per-dataset Pareto plotters."""
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None  # graceful fallback; see below

    texts = []
    for family, color in FAMILY_COLOR.items():
        s = sub[sub["family"] == family]
        if s.empty:
            continue
        ax.scatter(
            s["mean_fit_time_sec"], s["mean_rmse"],
            s=70, c=color, edgecolor="black", label=family, zorder=3,
        )
        for _, r in s.iterrows():
            texts.append(ax.text(
                r["mean_fit_time_sec"], r["mean_rmse"], r["method"],
                fontsize=9, color="black", alpha=0.95, zorder=4,
            ))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fit time (s)")
    ax.set_ylabel("test RMSE")
    ax.set_title(dataset, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # Add a bit of headroom on both axes for label placement.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 * 0.85, y1 * 1.6)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 * 0.7, x1 * 1.4)

    if adjust_text is not None and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray",
                            lw=0.6, alpha=0.7),
            expand=(1.2, 1.6),
            force_text=(0.6, 0.8),
            force_points=(0.4, 0.6),
        )
    else:
        # Fallback: rotated labels with offset (less reliable).
        for txt, (xp, yp) in zip(
            texts,
            [(ax.lines[0].get_xydata()[0])] if False else
            [(t.get_position()) for t in texts],
        ):
            txt.set_rotation(32)
            txt.set_horizontalalignment("left")
            txt.set_verticalalignment("bottom")

    if show_legend:
        ax.legend(loc="upper right", fontsize=9)


def plot_pareto(summary: pd.DataFrame, out_path: Path) -> None:
    """Combined view: 1×5 grid of Pareto panels."""
    datasets = list(DATASET_SPECS.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.0),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = summary[(summary["dataset"] == dataset)
                      & summary["mean_rmse"].notna()]
        _draw_pareto_panel(ax, sub, dataset=dataset, show_legend=False)

    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle(f"Cost-vs-accuracy Pareto (mean over {N_SEEDS} splits)",
                 fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pareto_per_dataset(summary: pd.DataFrame, out_dir: Path) -> list:
    """Per-dataset view: write one PNG per dataset at full size so labels
    have enough room to be readable.  Returns the list of paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset in DATASET_SPECS.keys():
        sub = summary[(summary["dataset"] == dataset)
                      & summary["mean_rmse"].notna()]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
        _draw_pareto_panel(ax, sub, dataset=dataset, show_legend=True)
        fig.suptitle(f"Cost vs. accuracy — {dataset}  "
                     f"(mean over {N_SEEDS} splits)", fontsize=13)
        out = out_dir / f"cost_pareto_{dataset}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_warm_vs_cold(raw: pd.DataFrame, out_path: Path) -> None:
    sub = raw[raw["method"].isin(["BFGS", "BFGS_warmstart"])]
    if sub["method"].nunique() < 2:
        return  # no warm-startable datasets in this run

    pivot = sub.groupby(["dataset", "method"]).agg(
        mean_iter=("n_iter", "mean"),
        mean_calls=("n_model_evals", "mean"),
        mean_time=("fit_time_sec", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metrics = [
        ("mean_iter", "iterations"),
        ("mean_calls", "model_fn calls"),
        ("mean_time", "fit time (s)"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        wide = pivot.pivot(index="dataset", columns="method", values=col)
        wide = wide.dropna(how="any")
        if wide.empty:
            ax.set_visible(False)
            continue
        # Order columns for stable colors: cold first, warm second.
        cols_in_order = [c for c in ("BFGS", "BFGS_warmstart") if c in wide.columns]
        wide = wide[cols_in_order]
        wide.plot.bar(
            ax=ax, edgecolor="black",
            color=[FAMILY_COLOR[NONLINEAR_FAMILY], FAMILY_COLOR[ZZU_FAMILY]],
        )
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.set_title(f"BFGS cold vs warm-start: {label}")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cold init = data-driven heuristic;  warm = ZZU's transformation-derived theta",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for dataset, spec in DATASET_SPECS.items():
        bundle = spec["generator"]()
        X_full = bundle.X.values
        y_full = bundle.y.values
        print(f"[{dataset}] n={len(y_full)}, p={X_full.shape[1]}")
        for seed in range(N_SEEDS):
            X_tr, X_te, y_tr, y_te = ta.train_test_split_arrays(
                X_full, y_full, test_size=TEST_FRACTION, seed=seed
            )
            rows.extend(evaluate_costs_for_split(
                dataset, spec, X_tr, X_te, y_tr, y_te, seed
            ))

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "cost_results.csv", index=False)
    print(f"Wrote {len(raw)} rows to {OUTPUT_DIR / 'cost_results.csv'}")

    summary = summarize_costs(raw)
    summary.to_csv(OUTPUT_DIR / "cost_summary.csv", index=False)
    print(f"Wrote summary to {OUTPUT_DIR / 'cost_summary.csv'}")

    plot_pareto(summary, OUTPUT_DIR / "cost_pareto.png")
    per_dataset_paths = plot_pareto_per_dataset(summary, OUTPUT_DIR)
    plot_warm_vs_cold(raw, OUTPUT_DIR / "warm_vs_cold.png")
    print(f"Wrote plots to {OUTPUT_DIR}/")
    for p in per_dataset_paths:
        print(f"  - {p.name}")

    # --- Console summary -----------------------------------------------------
    print("\nMean cost per method (cheapest 5 per dataset):")
    for dataset in DATASET_SPECS:
        sub = summary[summary["dataset"] == dataset].nsmallest(5, "mean_fit_time_sec")
        print(f"\n[{dataset}]")
        cols = ["method", "family", "mean_rmse",
                "mean_fit_time_sec", "mean_n_iter", "mean_n_model_evals"]
        print(sub[cols].to_string(index=False))

    sub = raw[raw["method"].isin(["BFGS", "BFGS_warmstart"])]
    if sub["method"].nunique() == 2:
        print("\nWarm-start vs cold (BFGS only, datasets where ZZU has a real coeff_to_init):")
        cmp = sub.groupby(["dataset", "method"]).agg(
            mean_iter=("n_iter", "mean"),
            mean_calls=("n_model_evals", "mean"),
            mean_time=("fit_time_sec", "mean"),
            mean_rmse=("rmse", "mean"),
        ).reset_index()
        print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
