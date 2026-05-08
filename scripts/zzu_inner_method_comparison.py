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
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import transformation_algorithms as ta
from run_comparison import DATASET_SPECS
from reproducibility import N_SEEDS, TEST_FRACTION

OUTPUT_DIR = PROJECT_ROOT / "comparison_results"

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


def _pretty_dataset_name(name: str) -> str:
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Per-(dataset, method, seed) evaluation
# ---------------------------------------------------------------------------

def _build_zzu(spec, X_tr, y_tr, method, kwargs):
    heuristic_init = spec["theta_init_fn"](X_tr, y_tr)
    coeff_to_init = spec["zzu_coeff_to_init"]
    if coeff_to_init is None:
        coeff_to_init = lambda _m, _h=heuristic_init: _h
    return ta.ZZUTransformRegressor(
        model_fn=spec["model_fn"],
        coeff_to_init=coeff_to_init,
        nonlinear_method=method,
        nonlinear_kwargs=dict(kwargs),
        transformations=spec["zzu_transformations"],
        fallback_theta_init=heuristic_init,
    )


def _build_pure(spec, method, kwargs):
    """Standalone nonlinear regressor with the dataset's heuristic init."""
    if method == "gradient_descent":
        return ta.GradientDescentRegressor(model_fn=spec["model_fn"], **kwargs)
    if method == "gauss_newton":
        return ta.GaussNewtonRegressor(model_fn=spec["model_fn"], **kwargs)
    if method == "bfgs":
        return ta.BFGSRegressor(model_fn=spec["model_fn"], **kwargs)
    raise ValueError(method)


def _common_split(spec, seed):
    bundle = spec["generator"]()
    X_full = bundle.X.values
    y_full = bundle.y.values
    return ta.train_test_split_arrays(
        X_full, y_full, test_size=TEST_FRACTION, seed=seed
    )


def evaluate_zzu(dataset, spec, method, kwargs, seed):
    """ZZU + this inner optimizer.  Times the entire screen+warm-start fit."""
    X_train, X_test, y_train, y_test = _common_split(spec, seed)
    zzu = _build_zzu(spec, X_train, y_train, method, kwargs)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zzu.fit(X_train, y_train)
        try:
            pred = zzu.predict(X_test)
            rmse = float(ta.regression_metrics(y_test, pred)["rmse"])
        except Exception:
            rmse = float("nan")
    fit_time = time.perf_counter() - t0

    nonlin = zzu.nonlinear_regressor_
    return {
        "dataset": dataset,
        "method": method,
        "variant": "zzu",
        "seed": seed,
        "fit_time_sec": fit_time,
        "n_iter": getattr(nonlin, "n_iter_", None),
        "converged": getattr(nonlin, "converged_", None),
        "rmse": rmse,
        "fit_error": zzu.fit_error_ or "",
    }


def evaluate_pure(dataset, spec, method, kwargs, seed):
    """Standalone optimizer with the cold heuristic init."""
    X_tr, X_te, y_tr, y_te = _common_split(spec, seed)
    theta_init = spec["theta_init_fn"](X_tr, y_tr)
    reg = _build_pure(spec, method, kwargs)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reg.fit(X_tr, y_tr, theta_init)
        try:
            pred = reg.predict(X_te)
            rmse = float(ta.regression_metrics(y_te, pred)["rmse"])
        except Exception:
            rmse = float("nan")
    fit_time = time.perf_counter() - t0

    return {
        "dataset": dataset,
        "method": method,
        "variant": "pure",
        "seed": seed,
        "fit_time_sec": fit_time,
        "n_iter": getattr(reg, "n_iter_", None),
        "converged": getattr(reg, "converged_", None),
        "rmse": rmse,
        "fit_error": reg.fit_error_ or "",
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    dataset_labels = [_pretty_dataset_name(dataset) for dataset in datasets]
    methods = list(METHODS.keys())
    variants = ["pure", "zzu"]

    # Layout: per dataset, 3 method-pairs (GD, GN, BFGS) × 2 bars (pure, zzu)
    # = 6 bars per group, with a small gap between method-pairs.
    bar_w = 0.13
    pair_gap = 0.04
    # Offsets (signed) for each (method_idx, variant_idx).
    offsets = []
    base = -(2 * bar_w + pair_gap) - 0.5 * bar_w  # leftmost edge of group
    for mi in range(len(methods)):
        for vi in range(len(variants)):
            offsets.append(
                base + mi * (2 * bar_w + pair_gap) + vi * bar_w + 0.5 * bar_w
            )
    x = np.arange(len(datasets))

    def lookup(d, m, v, col):
        row = summary[(summary["dataset"] == d)
                      & (summary["method"] == m)
                      & (summary["variant"] == v)]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8.6),
                             constrained_layout=True)

    # --- RMSE ---
    ax = axes[0]
    k = 0
    for method in methods:
        for variant in variants:
            means = [lookup(d, method, variant, "mean_rmse") for d in datasets]
            stds  = [lookup(d, method, variant, "std_rmse") for d in datasets]
            label = (f"Pure {SHORT_LABEL[method]}" if variant == "pure"
                     else f"ZZU + {SHORT_LABEL[method]}")
            ax.bar(x + offsets[k], means, bar_w,
                   yerr=stds, capsize=2,
                   color=COLOR[method], alpha=ALPHA_BY_VARIANT[variant],
                   hatch=HATCH_BY_VARIANT[variant],
                   edgecolor="black", linewidth=0.7,
                   label=label)
            k += 1
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(ncol=3, fontsize=9, loc="upper left")

    # --- Fit time ---
    ax = axes[1]
    k = 0
    for method in methods:
        for variant in variants:
            means = [lookup(d, method, variant, "mean_fit_time") for d in datasets]
            label = (f"Pure {SHORT_LABEL[method]}" if variant == "pure"
                     else f"ZZU + {SHORT_LABEL[method]}")
            ax.bar(x + offsets[k], means, bar_w,
                   color=COLOR[method], alpha=ALPHA_BY_VARIANT[variant],
                   hatch=HATCH_BY_VARIANT[variant],
                   edgecolor="black", linewidth=0.7,
                   label=label)
            k += 1
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("Fit Time (s)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(ncol=3, fontsize=9, loc="upper left")

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_datasets = (
        list(DATASET_SPECS.keys()) if DATASETS_TO_COMPARE is None else DATASETS_TO_COMPARE
    )
    invalid = [name for name in selected_datasets if name not in DATASET_SPECS]
    if invalid:
        raise ValueError(f"Unknown dataset(s) in DATASETS_TO_COMPARE: {invalid}")

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
