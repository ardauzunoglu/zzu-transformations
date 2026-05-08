"""
Compare ZZU's choice of *inner* nonlinear optimizer across the synthetic
suite, with pure (cold-start) versions side by side as baselines.

Six configurations per dataset:
  - GD             (pure, heuristic init)
  - ZZU + GD       (screen → warm-start GD → bias correct)
  - GN             (pure, heuristic init)
  - ZZU + GN       (screen → warm-start GN → bias correct)
  - BFGS           (pure, heuristic init)            <-- nonlinear default
  - ZZU + BFGS     (screen → warm-start BFGS → bias correct)  <-- ZZU default

The plot pairs each pure optimizer with its ZZU-augmented counterpart so
you can read both axes off a single bar group:
    "what does the optimizer do alone?" vs "what does ZZU add?"

Outputs (./comparison_results/):
  - zzu_inner_method_results.csv     long-form: 1 row per (dataset, method, variant, seed)
  - zzu_inner_method_summary.csv     mean / std of RMSE, fit time, n_iter
  - zzu_inner_method.png             grouped-bar comparison plot

Usage:  python scripts/zzu_inner_method_comparison.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.algorithms as ta
from run_comparison import (
    DATASET_SPECS,
    _pretty_dataset_name,
    make_zzu,
    resolve_datasets,
)
from reproducibility import N_SEEDS, TEST_FRACTION, reproduce_dir

OUTPUT_DIR = reproduce_dir("comparison_results", PROJECT_ROOT)

# Set this to a list of dataset names to run a subset, for example:
# ["exponential_additive", "logistic_growth"]
# Leave as None to compare across every dataset in DATASET_SPECS.
DATASETS_TO_COMPARE = ["exponential_multiplicative", "logistic_growth", "multivariable_nonlinear"]

# Method-specific kwargs are tuned to mirror what the standalone nonlinear
# regressors use elsewhere in the project, so the only thing that changes
# is the post-screening optimizer choice.
METHODS: Dict[str, Dict[str, Any]] = {
    "gradient_descent": {"learning_rate": 1e-4, "decay": 0.9999, "max_iter": 5000},
    "gauss_newton":     {"max_iter": 200},
    "bfgs":             {"max_iter": 500},
}

SHORT_LABEL = {
    "gradient_descent": "GD",
    "gauss_newton":     "GN",
    "bfgs":             "BFGS",
}

# Color = optimizer family.  Hatching distinguishes pure vs. ZZU-augmented.
COLOR = {
    "gradient_descent": "#9C9C9C",
    "gauss_newton":     "#1F77B4",
    "bfgs":             "#2CA02C",
}
HATCH_BY_VARIANT = {"pure": "//", "zzu": ""}
ALPHA_BY_VARIANT = {"pure": 0.55, "zzu": 1.0}


# ---------------------------------------------------------------------------
# Per-(dataset, method, seed) evaluation
# ---------------------------------------------------------------------------

_PURE_REGRESSORS = {
    "gradient_descent": ta.GradientDescentRegressor,
    "gauss_newton":     ta.GaussNewtonRegressor,
    "bfgs":             ta.BFGSRegressor,
}


def _common_split(spec, seed):
    bundle = spec["generator"]()
    return ta.train_test_split_arrays(
        bundle.X.values, bundle.y.values,
        test_size=TEST_FRACTION, seed=seed,
    )


def _time_fit_and_rmse(reg, X_tr, y_tr, X_te, y_te, theta_init=None):
    """Fit `reg`, predict on (X_te, y_te), and return (fit_time, rmse).

    Wraps the timer + RuntimeWarning filter + safe-predict pattern shared
    by both the pure and ZZU evaluators.  ``theta_init`` is forwarded to
    standalone regressors (whose ``fit`` takes it) and omitted for ZZU."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if theta_init is None:
            reg.fit(X_tr, y_tr)
        else:
            reg.fit(X_tr, y_tr, theta_init)
        try:
            pred = reg.predict(X_te)
            rmse = float(ta.regression_metrics(y_te, pred)["rmse"])
        except Exception:
            rmse = float("nan")
    return time.perf_counter() - t0, rmse


def _row(dataset, method, variant, seed, *, fit_time, rmse, inner, fit_error):
    return {
        "dataset": dataset,
        "method": method,
        "variant": variant,
        "seed": seed,
        "fit_time_sec": fit_time,
        "n_iter": getattr(inner, "n_iter_", None),
        "converged": getattr(inner, "converged_", None),
        "rmse": rmse,
        "fit_error": fit_error or "",
    }


def evaluate_zzu(dataset, spec, method, kwargs, seed):
    """ZZU + this inner optimizer.  Times the entire screen+warm-start fit."""
    X_tr, X_te, y_tr, y_te = _common_split(spec, seed)
    zzu = make_zzu(spec, X_tr, y_tr,
                   nonlinear_method=method, nonlinear_kwargs=kwargs)
    fit_time, rmse = _time_fit_and_rmse(zzu, X_tr, y_tr, X_te, y_te)
    return _row(dataset, method, "zzu", seed,
                fit_time=fit_time, rmse=rmse,
                inner=zzu.nonlinear_regressor_, fit_error=zzu.fit_error_)


def evaluate_pure(dataset, spec, method, kwargs, seed):
    """Standalone optimizer with the cold heuristic init."""
    X_tr, X_te, y_tr, y_te = _common_split(spec, seed)
    theta_init = spec["theta_init_fn"](X_tr, y_tr)
    reg = _PURE_REGRESSORS[method](model_fn=spec["model_fn"], **kwargs)
    fit_time, rmse = _time_fit_and_rmse(reg, X_tr, y_tr, X_te, y_te,
                                        theta_init=theta_init)
    return _row(dataset, method, "pure", seed,
                fit_time=fit_time, rmse=rmse,
                inner=reg, fit_error=reg.fit_error_)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _bar_offsets(n_methods: int, n_variants: int,
                 bar_w: float, pair_gap: float) -> list[float]:
    """Center-aligned bar offsets for n_methods × n_variants bars per group."""
    base = -(2 * bar_w + pair_gap) - 0.5 * bar_w
    return [
        base + mi * (2 * bar_w + pair_gap) + vi * bar_w + 0.5 * bar_w
        for mi in range(n_methods)
        for vi in range(n_variants)
    ]


def _draw_paired_bars(
    ax,
    summary: pd.DataFrame,
    datasets: list,
    methods: list,
    variants: list,
    offsets: list,
    bar_w: float,
    *,
    value_col: str,
    std_col: Optional[str],
    ylabel: str,
) -> None:
    """Draw one panel of paired (pure vs ZZU) bars per dataset.

    Shared by the RMSE and fit-time panels — only the value column,
    optional error bars, and y-axis label differ.
    """
    def lookup(d, m, v, col):
        row = summary[(summary["dataset"] == d)
                      & (summary["method"] == m)
                      & (summary["variant"] == v)]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    x = np.arange(len(datasets))
    k = 0
    for method in methods:
        for variant in variants:
            means = [lookup(d, method, variant, value_col) for d in datasets]
            stds = ([lookup(d, method, variant, std_col) for d in datasets]
                    if std_col is not None else None)
            label = (f"Pure {SHORT_LABEL[method]}" if variant == "pure"
                     else f"ZZU + {SHORT_LABEL[method]}")
            ax.bar(
                x + offsets[k], means, bar_w,
                yerr=stds, capsize=2 if stds is not None else 0,
                color=COLOR[method], alpha=ALPHA_BY_VARIANT[variant],
                hatch=HATCH_BY_VARIANT[variant],
                edgecolor="black", linewidth=0.7,
                label=label,
            )
            k += 1
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_pretty_dataset_name(d) for d in datasets],
        rotation=0, ha="center", fontsize=10,
    )
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(ncol=3, fontsize=9, loc="upper left")


def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    methods = list(METHODS.keys())
    variants = ["pure", "zzu"]

    bar_w = 0.13
    pair_gap = 0.04
    offsets = _bar_offsets(len(methods), len(variants), bar_w, pair_gap)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8.6), constrained_layout=True)
    _draw_paired_bars(
        axes[0], summary, datasets, methods, variants, offsets, bar_w,
        value_col="mean_rmse", std_col="std_rmse", ylabel="RMSE",
    )
    _draw_paired_bars(
        axes[1], summary, datasets, methods, variants, offsets, bar_w,
        value_col="mean_fit_time", std_col=None, ylabel="Fit Time (s)",
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_datasets = resolve_datasets(DATASETS_TO_COMPARE)

    rows = []
    for dataset in selected_datasets:
        spec = DATASET_SPECS[dataset]
        print(f"[{dataset}]")
        for method, kwargs in METHODS.items():
            for seed in range(N_SEEDS):
                rows.append(evaluate_pure(dataset, spec, method, kwargs, seed))
                rows.append(evaluate_zzu(dataset, spec, method, kwargs, seed))

    raw = pd.DataFrame(rows)
    raw_path = OUTPUT_DIR / "zzu_inner_method_results.csv"
    raw.to_csv(raw_path, index=False)
    print(f"\nWrote {len(raw)} rows to {raw_path.name}")

    summary = raw.groupby(["dataset", "method", "variant"]).agg(
        n=("rmse", "size"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_fit_time=("fit_time_sec", "mean"),
        std_fit_time=("fit_time_sec", "std"),
        mean_n_iter=("n_iter", lambda s: float(np.nanmean(s.dropna().astype(float)))
                     if s.notna().any() else float("nan")),
        frac_converged=("converged", lambda s: float(np.nanmean(s.dropna().astype(float)))
                        if s.notna().any() else float("nan")),
    ).reset_index()
    summary_path = OUTPUT_DIR / "zzu_inner_method_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path.name}")

    plot_comparison(summary, OUTPUT_DIR / "zzu_inner_method.png")
    print(f"Wrote plot:  zzu_inner_method.png")

    # Console digest.
    print("\nMean RMSE / fit time / n_iter / convergence per (dataset, optimizer, variant):")
    digest = summary[["dataset", "method", "variant", "mean_rmse",
                      "mean_fit_time", "mean_n_iter", "frac_converged"]]
    print(digest.to_string(index=False))


if __name__ == "__main__":
    main()
