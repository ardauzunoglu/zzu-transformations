"""
Full comparison of transformation-based OLS, direct nonlinear regression, and
the ZZU hybrid workflow across the five synthetic datasets in toy_data.py.

For each dataset:
  - Build the default 7-model TransformedOLS screening suite.
  - Build three nonlinear regressors (GD, GN, BFGS) sharing the same
    dataset-specific model_fn and a heuristic theta_init derived from the data.
  - Build a ZZUTransformRegressor with a dataset-appropriate screening dict
    and coeff_to_init (or a heuristic fallback).
  - Run K repeated train/test splits and evaluate every model on the test set.

Outputs (written to ./comparison_results/):
  - raw_results.csv            long-form: one row per (dataset, method, seed)
  - summary_by_method.csv      mean / std RMSE, R2, convergence rate
  - rmse_by_method.png         per-dataset RMSE bar chart with error bars
  - fit_overlay.png            best-fit overlays for the four 1D datasets

Usage: python run_comparison.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scripts.algorithms as ta
import toy_data as td
# All benchmark seeds and split sizes live in reproducibility.py — a single
# source of truth that the writeup can cite without grepping for constants.
from reproducibility import N_SEEDS, TEST_FRACTION, reproduce_dir


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = reproduce_dir("comparison_results", PROJECT_ROOT)

# Set this to a list of dataset names to run a subset, for example:
# ["exponential_additive", "logistic_growth"]
# Leave as None to compare across every dataset in DATASET_SPECS.
DATASETS_TO_COMPARE = ["exponential_multiplicative", "logistic_growth", "multivariable_nonlinear"]

# Method-family tagging for plotting.
LINEAR_FAMILY = "linearized_ols"
NONLINEAR_FAMILY = "nonlinear"
ZZU_FAMILY = "zzu_hybrid"


def _pretty_dataset_name(name: str) -> str:
    return name.replace("_", " ").title()


def _pretty_method_name(name: str) -> str:
    replacements = {
        "yeojohnson": "Y-J",
        "boxcox": "B-C",
        "reciprocal": "Reciprocal",
        "identity": "Identity",
        "log": "Log",
        "power": "Power",
        "smear": "smear",
    }
    parts = name.split("_")
    pretty_parts = [
        replacements.get(part, part)
        for part in parts
        if part not in {"no", "smear"}
    ]
    if name.endswith("_smear") and not name.endswith("_no_smear"):
        pretty_parts.append("+ Smear")
    pretty = " ".join(pretty_parts)
    return pretty.replace(" + Smear", "\n+ Smear")


def resolve_datasets(
    datasets_to_compare: Optional[List[str]] = None,
) -> List[str]:
    """Validate a `DATASETS_TO_COMPARE`-style list against `DATASET_SPECS`.

    Used by every benchmark script so the validation rule lives in one
    place: ``None`` means "all datasets"; otherwise the given names must
    all exist in ``DATASET_SPECS``.

    Setting ``REPRODUCE_DATASETS=all`` in the environment forces the
    full 5-dataset sweep regardless of the script's ``DATASETS_TO_COMPARE``
    constant — used for diff-based reproduction against the canonical
    outputs, which were generated on all five datasets.
    """
    if os.environ.get("REPRODUCE_DATASETS") == "all":
        return list(DATASET_SPECS.keys())
    if datasets_to_compare is None:
        datasets_to_compare = DATASETS_TO_COMPARE
    selected = list(DATASET_SPECS.keys()) if datasets_to_compare is None else list(datasets_to_compare)
    invalid = [name for name in selected if name not in DATASET_SPECS]
    if invalid:
        raise ValueError(f"Unknown dataset(s) requested: {invalid}")
    return selected


# Back-compat: callers that were importing the old name continue to work.
_selected_dataset_names = resolve_datasets


# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------
#
# Each spec describes how to fit a direct nonlinear model on that dataset:
#   - model_fn(X, theta) -> y_hat  (pure-numpy, vectorized)
#   - theta_init_fn(X, y) -> theta_init  (data-driven heuristic init)
#   - p: number of parameters (used only for sanity checks)
#   - zzu_transformations: dict passed as ZZUTransformRegressor.transformations.
#       None means use the default 7-model suite.
#   - zzu_coeff_to_init: function mapping the best fitted TransformedOLS to a
#       nonlinear theta_init.  Must be consistent with zzu_transformations.
#       If None, ZZU falls back to theta_init_fn output.
#
# The key design rule (see README): coeff_to_init must be invertible from the
# specific transforms in the screening suite.  Where no clean inversion exists
# (MM, logistic, multivariable), we restrict the screen and use a heuristic
# fallback init computed from training data.
# ---------------------------------------------------------------------------

def _exp_model(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    return t[0] * np.exp(t[1] * X[:, 0])


def _exp_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # a near min(y), b small and positive: a safe, monotonic-growth start.
    a0 = max(float(np.min(np.maximum(y, 1e-3))), 1e-3)
    return np.array([a0, 0.1])


def _exp_coeff_to_init(m: ta.TransformedOLS) -> np.ndarray:
    # log(y) = beta0 + beta1 * x  =>  a = exp(beta0), b = beta1
    beta = m.beta_
    return np.array([float(np.exp(beta[0])), float(beta[1])])


def _mm_model(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    # y = Vmax * x / (Km + x)
    return t[0] * X[:, 0] / (t[1] + X[:, 0])


def _mm_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Vmax ~ slightly above max observed y; Km ~ x at half-max.
    vmax0 = float(np.max(y)) * 1.1
    km0 = float(np.median(X[:, 0]))
    return np.array([max(vmax0, 1e-3), max(km0, 1e-3)])


def _logistic_model(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    # y = L / (1 + exp(-k * (x - x0))).  Clip exponent to avoid overflow
    # warnings during BFGS line searches.
    z = -t[1] * (X[:, 0] - t[2])
    z = np.clip(z, -50.0, 50.0)
    return t[0] / (1.0 + np.exp(z))


def _logistic_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    L0 = float(np.max(y)) * 1.05
    x0_0 = float(np.median(X[:, 0]))
    return np.array([max(L0, 1e-3), 1.0, x0_0])


def _multivar_model(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    # y = c1 * exp(r * x1) + c2 * x2^p + c3 / (1 + x3)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    z = np.clip(t[1] * x1, -50.0, 50.0)
    # Guard against negative bases for the power term during line searches.
    safe_x2 = np.maximum(x2, 1e-12)
    return t[0] * np.exp(z) + t[2] * np.power(safe_x2, t[3]) + t[4] / (1.0 + x3)


def _multivar_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Conservative positive init; rate kept small to avoid early overflow.
    return np.array([1.0, 0.1, 1.0, 1.0, 1.0])


DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "exponential_multiplicative": {
        "generator": td.make_exponential_multiplicative,
        "model_fn": _exp_model,
        "theta_init_fn": _exp_init,
        "p": 2,
        "zzu_transformations": {
            "log_smear": ta.TransformedOLS(transform="log", use_smearing=True),
        },
        "zzu_coeff_to_init": _exp_coeff_to_init,
    },
    "exponential_additive": {
        "generator": td.make_exponential_additive,
        "model_fn": _exp_model,
        "theta_init_fn": _exp_init,
        "p": 2,
        "zzu_transformations": {
            "log_smear": ta.TransformedOLS(transform="log", use_smearing=True),
        },
        "zzu_coeff_to_init": _exp_coeff_to_init,
    },
    "michaelis_menten": {
        "generator": td.make_michaelis_menten,
        "model_fn": _mm_model,
        "theta_init_fn": _mm_init,
        "p": 2,
        # No clean univariate transform of y linearizes MM.  Restrict screening
        # to identity + power(0.5) just to record diagnostic value, and rely on
        # the heuristic init for the warm start.
        "zzu_transformations": {
            "identity": ta.TransformedOLS(transform="identity", use_smearing=False),
            "power_0.5_smear": ta.TransformedOLS(
                transform="power", param=0.5, use_smearing=True
            ),
        },
        "zzu_coeff_to_init": None,  # use heuristic fallback
    },
    "logistic_growth": {
        "generator": td.make_logistic_growth,
        "model_fn": _logistic_model,
        "theta_init_fn": _logistic_init,
        "p": 3,
        "zzu_transformations": {
            "identity": ta.TransformedOLS(transform="identity", use_smearing=False),
            "yeojohnson_smear": ta.TransformedOLS(
                transform="yeojohnson", use_smearing=True
            ),
        },
        "zzu_coeff_to_init": None,
    },
    "multivariable_nonlinear": {
        "generator": td.make_multivariable_nonlinear,
        "model_fn": _multivar_model,
        "theta_init_fn": _multivar_init,
        "p": 5,
        # Multivariable target has no global linearization.  Use the default
        # screen for diagnostic value, with heuristic fallback init.
        "zzu_transformations": None,
        "zzu_coeff_to_init": None,
    },
}


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_transformed_suite() -> Dict[str, ta.TransformedOLS]:
    """Eleven TransformedOLS specs (default 7 + 4 useful extras)."""
    return {
        "identity": ta.TransformedOLS(transform="identity", use_smearing=False),
        "log_no_smear": ta.TransformedOLS(transform="log", use_smearing=False),
        "log_smear": ta.TransformedOLS(transform="log", use_smearing=True),
        "reciprocal_no_smear": ta.TransformedOLS(transform="reciprocal", use_smearing=False),
        "reciprocal_smear": ta.TransformedOLS(transform="reciprocal", use_smearing=True),
        "power_0.5_smear": ta.TransformedOLS(transform="power", param=0.5, use_smearing=True),
        "power_2_smear": ta.TransformedOLS(transform="power", param=2.0, use_smearing=True),
        "boxcox_no_smear": ta.TransformedOLS(transform="boxcox", use_smearing=False),
        "boxcox_smear": ta.TransformedOLS(transform="boxcox", use_smearing=True),
        "yeojohnson_no_smear": ta.TransformedOLS(transform="yeojohnson", use_smearing=False),
        "yeojohnson_smear": ta.TransformedOLS(transform="yeojohnson", use_smearing=True),
    }


def make_nonlinear_regressors(model_fn: Callable) -> Dict[str, Any]:
    """Three nonlinear regressors sharing one model_fn."""
    return {
        "GD": ta.GradientDescentRegressor(
            model_fn=model_fn,
            learning_rate=1e-4,
            decay=0.9999,
            max_iter=5000,
        ),
        "GN": ta.GaussNewtonRegressor(model_fn=model_fn, max_iter=200),
        "BFGS": ta.BFGSRegressor(model_fn=model_fn, max_iter=500),
    }


def make_zzu(
    spec: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    nonlinear_method: str = "bfgs",
    nonlinear_kwargs: Optional[Dict[str, Any]] = None,
) -> ta.ZZUTransformRegressor:
    """Build a ZZU regressor from a dataset spec.  The heuristic init is
    baked into ``fallback_theta_init`` so the screen never crashes the
    pipeline when ``coeff_to_init`` cannot be applied.

    ``nonlinear_method`` and ``nonlinear_kwargs`` are forwarded to
    ``ZZUTransformRegressor`` so callers can sweep over the inner
    optimizer (used by ``zzu_inner_method_comparison.py``)."""
    heuristic_init = spec["theta_init_fn"](X_train, y_train)
    coeff_to_init = spec["zzu_coeff_to_init"]
    if coeff_to_init is None:
        # No clean inversion: ignore the screened model and return the heuristic.
        coeff_to_init = lambda _m, _h=heuristic_init: _h
    return ta.ZZUTransformRegressor(
        model_fn=spec["model_fn"],
        coeff_to_init=coeff_to_init,
        nonlinear_method=nonlinear_method,
        nonlinear_kwargs=dict(nonlinear_kwargs) if nonlinear_kwargs else None,
        transformations=spec["zzu_transformations"],
        fallback_theta_init=heuristic_init,
    )


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def _safe_predict(predict_fn: Callable, X_test: np.ndarray) -> Optional[np.ndarray]:
    """Predict, returning None on any exception."""
    try:
        return np.asarray(predict_fn(X_test), dtype=float).ravel()
    except Exception:
        return None


def _row_from_pred(
    dataset: str,
    method: str,
    family: str,
    seed: int,
    y_test: np.ndarray,
    pred: Optional[np.ndarray],
    *,
    converged: Optional[bool] = None,
    n_iter: Optional[int] = None,
    error: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if pred is None or not np.all(np.isfinite(pred)):
        m = {"n_valid": 0, "rmse": np.nan, "mae": np.nan, "mse": np.nan, "r2": np.nan}
    else:
        m = ta.regression_metrics(y_test, pred)
    row = {
        "dataset": dataset,
        "method": method,
        "family": family,
        "seed": seed,
        "converged": converged,
        "n_iter": n_iter,
        "error": error,
        **m,
    }
    if extra:
        row.update(extra)
    return row


def evaluate_split(
    dataset: str,
    spec: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # --- Linearized OLS suite ---
    for name, model in make_transformed_suite().items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append(_row_from_pred(dataset, name, LINEAR_FAMILY, seed, y_test, pred))
        except Exception as exc:
            rows.append(_row_from_pred(
                dataset, name, LINEAR_FAMILY, seed, y_test, None, error=str(exc)
            ))

    # --- Nonlinear regressors with shared heuristic init ---
    theta_init = spec["theta_init_fn"](X_train, y_train)
    for name, reg in make_nonlinear_regressors(spec["model_fn"]).items():
        # Suppress benign overflow warnings from line search exploration.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reg.fit(X_train, y_train, theta_init)
            pred = _safe_predict(reg.predict, X_test)
        rows.append(_row_from_pred(
            dataset, name, NONLINEAR_FAMILY, seed, y_test, pred,
            converged=getattr(reg, "converged_", None),
            n_iter=getattr(reg, "n_iter_", None),
            error=getattr(reg, "fit_error_", "") or "",
        ))

    # --- ZZU hybrid ---
    zzu = make_zzu(spec, X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zzu.fit(X_train, y_train)
        pred = _safe_predict(zzu.predict, X_test)
    rows.append(_row_from_pred(
        dataset, "ZZU", ZZU_FAMILY, seed, y_test, pred,
        converged=getattr(getattr(zzu, "nonlinear_regressor_", None), "converged_", None),
        n_iter=getattr(getattr(zzu, "nonlinear_regressor_", None), "n_iter_", None),
        error=zzu.fit_error_ or "",
        extra={"zzu_best_transform": zzu.best_transform_name_},
    ))

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FAMILY_COLOR = {
    LINEAR_FAMILY: "#4C78A8",
    NONLINEAR_FAMILY: "#F58518",
    ZZU_FAMILY: "#54A24B",
}

FAMILY_LABEL = {
    LINEAR_FAMILY: "Linearized OLS",
    NONLINEAR_FAMILY: "Nonlinear",
    ZZU_FAMILY: "ZZU Hybrid",
}


def plot_rmse_by_method(summary: pd.DataFrame, out_path: Path) -> None:
    """One subplot per dataset, bar chart of mean RMSE per method."""
    datasets = list(summary["dataset"].drop_duplicates())
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.2 * n), constrained_layout=False)
    if n == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("mean_rmse")
        x = np.arange(len(sub))
        colors = [FAMILY_COLOR.get(f, "gray") for f in sub["family"]]
        ax.bar(
            x, sub["mean_rmse"].values,
            yerr=sub["std_rmse"].fillna(0.0).values,
            color=colors, edgecolor="black", linewidth=0.5, capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_pretty_method_name(method) for method in sub["method"].values],
            rotation=0,
            ha="center",
            fontsize=7,
        )
        ax.set_ylabel("RMSE")
        ax.set_title(_pretty_dataset_name(dataset))
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3, which="both")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=c, ec="black") for c in FAMILY_COLOR.values()
    ]
    fig.legend(
        handles, [FAMILY_LABEL[family] for family in FAMILY_COLOR],
        loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=10,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_fit_overlay(out_path: Path, datasets: List[str], seed: int = 0) -> None:
    """Best-fit overlays for the four 1D datasets at a single representative seed."""
    one_d_all = [
        "exponential_multiplicative",
        "exponential_additive",
        "michaelis_menten",
        "logistic_growth",
    ]
    one_d = [dataset for dataset in one_d_all if dataset in datasets]
    if not one_d:
        return

    ncols = 2
    nrows = int(np.ceil(len(one_d) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13, 4.5 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, dataset in zip(axes, one_d):
        spec = DATASET_SPECS[dataset]
        bundle = spec["generator"]()
        X_full = bundle.X.values
        y_full = bundle.y.values
        X_tr, X_te, y_tr, y_te = ta.train_test_split_arrays(
            X_full, y_full, test_size=TEST_FRACTION, seed=seed
        )

        # Best linear model from the full suite, picked by training MSE.
        best_lin_name, best_lin_pred_full = None, None
        best_lin_rmse = np.inf
        for name, m in make_transformed_suite().items():
            try:
                m.fit(X_tr, y_tr)
                pred_te = m.predict(X_te)
                rmse = ta.regression_metrics(y_te, pred_te)["rmse"]
                if np.isfinite(rmse) and rmse < best_lin_rmse:
                    best_lin_rmse = rmse
                    best_lin_name = name
                    best_lin_pred_full = m.predict(X_full)
            except Exception:
                continue

        # BFGS direct fit.
        theta_init = spec["theta_init_fn"](X_tr, y_tr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bf = ta.BFGSRegressor(model_fn=spec["model_fn"], max_iter=500)
            bf.fit(X_tr, y_tr, theta_init)
            bfgs_pred_full = _safe_predict(bf.predict, X_full)

        # ZZU.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zzu = make_zzu(spec, X_tr, y_tr)
            zzu.fit(X_tr, y_tr)
            zzu_pred_full = _safe_predict(zzu.predict, X_full)

        x = X_full[:, 0]
        order = np.argsort(x)
        ax.scatter(x, y_full, s=15, alpha=0.4, color="gray", label="observed y")
        ax.plot(x[order], bundle.y_true.values[order], "k--", lw=1.5, label="true signal")
        if best_lin_pred_full is not None:
            ax.plot(
                x[order], best_lin_pred_full[order],
                lw=1.8, color=FAMILY_COLOR[LINEAR_FAMILY],
                label=f"best linear ({best_lin_name})",
            )
        if bfgs_pred_full is not None:
            ax.plot(
                x[order], bfgs_pred_full[order],
                lw=1.8, color=FAMILY_COLOR[NONLINEAR_FAMILY], label="BFGS",
            )
        if zzu_pred_full is not None:
            ax.plot(
                x[order], zzu_pred_full[order],
                lw=1.8, color=FAMILY_COLOR[ZZU_FAMILY], label="ZZU",
            )
        ax.set_title(_pretty_dataset_name(dataset))
        ax.set_xlabel(bundle.X.columns[0])
        ax.set_ylabel("y")
        ax.legend(fontsize=8)

    for ax in axes[len(one_d):]:
        ax.set_visible(False)

    fig.suptitle(f"Best-fit overlays (seed={seed})", fontsize=12)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean / std of RMSE, R2; convergence rate across seeds."""
    grouped = raw.groupby(["dataset", "method", "family"], dropna=False)
    summary = grouped.agg(
        n=("rmse", "size"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        median_rmse=("rmse", "median"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
        n_failures=("rmse", lambda s: int(s.isna().sum())),
        frac_converged=("converged", lambda s: float(np.nanmean(s.dropna().astype(float)))
                        if s.notna().any() else np.nan),
    ).reset_index()
    return summary.sort_values(["dataset", "mean_rmse"], na_position="last").reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_datasets = _selected_dataset_names()

    all_rows: List[Dict[str, Any]] = []
    for dataset in selected_datasets:
        spec = DATASET_SPECS[dataset]
        bundle = spec["generator"]()
        X_full = bundle.X.values
        y_full = bundle.y.values
        print(f"[{dataset}] n={len(y_full)}, p={X_full.shape[1]}")

        for seed in range(N_SEEDS):
            X_tr, X_te, y_tr, y_te = ta.train_test_split_arrays(
                X_full, y_full, test_size=TEST_FRACTION, seed=seed
            )
            rows = evaluate_split(dataset, spec, X_tr, X_te, y_tr, y_te, seed)
            all_rows.extend(rows)

    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)
    print(f"Wrote {len(raw)} rows to {OUTPUT_DIR / 'raw_results.csv'}")

    summary = summarize(raw)
    summary.to_csv(OUTPUT_DIR / "summary_by_method.csv", index=False)
    print(f"Wrote summary to {OUTPUT_DIR / 'summary_by_method.csv'}")

    plot_rmse_by_method(summary, OUTPUT_DIR / "rmse_by_method.png")
    plot_fit_overlay(OUTPUT_DIR / "fit_overlay.png", selected_datasets, seed=0)
    print(f"Wrote plots to {OUTPUT_DIR}/")

    # Print the top-3 methods per dataset for a quick console scan.
    print("\nTop 3 methods (by mean test RMSE) per dataset:")
    for dataset in selected_datasets:
        sub = summary[summary["dataset"] == dataset].head(3)
        print(f"\n[{dataset}]")
        print(sub[["method", "family", "mean_rmse", "std_rmse",
                   "mean_r2", "frac_converged"]].to_string(index=False))


if __name__ == "__main__":
    main()
