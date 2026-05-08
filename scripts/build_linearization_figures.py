"""
Three figures showing how different response transformations interact with
different data types.  Pedagogical companion to the ZZU presentation.

Outputs (./comparison_results/):
  - lin_fits_original.png       3x4 grid: each transform's fit on the
                                original scale, per dataset.  Tells you
                                "which method's curve hugs the data?"
  - lin_transformed_scale.png   3x3 grid: T(y) vs x with OLS line.
                                Tells you "does this transform actually
                                linearize the relationship?"
  - lin_residuals.png           3x3 grid: residuals on the transformed
                                scale vs x.  Tells you "is the OLS
                                assumption (homoscedasticity, no pattern)
                                satisfied after the transform?"

Usage:  python scripts/build_linearization_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

import toy_data as td
import scripts.algorithms as ta
from reproducibility import reproduce_dir

OUT_DIR = reproduce_dir("comparison_results", PROJECT_ROOT)
NAVY_HEX = "#1E2761"
CYAN_HEX = "#4FC3F7"
RED_HEX = "#E05252"
GRAY = "#888888"
TRANSFORM_COLORS: Dict[str, str] = {
    "identity":   "#9C9C9C",
    "log":        "#1F77B4",
    "reciprocal": "#D62728",
    "boxcox":     "#2CA02C",
}


# ---------------------------------------------------------------------------
# Datasets and transforms
# ---------------------------------------------------------------------------

def get_datasets() -> List[Tuple[str, "td.DatasetBundle"]]:
    return [
        ("exponential_multiplicative", td.make_exponential_multiplicative()),
        ("exponential_additive", td.make_exponential_additive()),
        ("michaelis_menten", td.make_michaelis_menten()),
    ]


def fresh_transforms() -> Dict[str, ta.TransformedOLS]:
    """Build a fresh dict of TransformedOLS specs (must be re-built per fit)."""
    return {
        "identity":   ta.TransformedOLS(transform="identity", use_smearing=False),
        "log":        ta.TransformedOLS(transform="log", use_smearing=True),
        "reciprocal": ta.TransformedOLS(transform="reciprocal", use_smearing=True),
        "boxcox":     ta.TransformedOLS(transform="boxcox", use_smearing=True),
    }


# ---------------------------------------------------------------------------
# Figure 1 — fits on the original response scale
# ---------------------------------------------------------------------------

def plot_original_fits(out_path: Path) -> None:
    datasets = get_datasets()
    transforms_keys = ["identity", "log", "reciprocal", "boxcox"]

    fig, axes = plt.subplots(
        len(datasets), len(transforms_keys),
        figsize=(15, 9), constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Linearization fits on the original scale — which curve hugs the data?",
        fontsize=14, color=NAVY_HEX, fontweight="bold",
    )

    for r, (dname, bundle) in enumerate(datasets):
        x = bundle.X.iloc[:, 0].values
        y = bundle.y.values
        order = np.argsort(x)
        suite = fresh_transforms()

        for c, transformation_name in enumerate(transforms_keys):
            ax = axes[r, c]
            ax.scatter(x, y, s=14, alpha=0.45, color=GRAY, label="observed")
            ax.plot(x[order], bundle.y_true.values[order],
                    "--", color="black", lw=1.4, label="true signal")

            model = suite[transformation_name]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x.reshape(-1, 1), y)
                    yhat = model.predict(x.reshape(-1, 1))
                rmse = ta.regression_metrics(y, yhat)["rmse"]
                ax.plot(x[order], yhat[order],
                        color=TRANSFORM_COLORS[transformation_name], lw=2.2,
                        label=f"{transformation_name} fit")
                # Annotate RMSE and selected λ if Box-Cox.
                lam = model.selected_param_
                if transformation_name == "boxcox" and lam is not None and np.isfinite(lam):
                    note = f"RMSE = {rmse:.3g}\nλ = {lam:.2f}"
                else:
                    note = f"RMSE = {rmse:.3g}"
                ax.text(0.97, 0.03, note, transform=ax.transAxes,
                        fontsize=9, color=NAVY_HEX, ha="right", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="lightgray", pad=2))
            except Exception as exc:
                ax.text(0.5, 0.5, f"fit failed:\n{exc!s}",
                        transform=ax.transAxes, fontsize=9, ha="center",
                        va="center", color=RED_HEX)

            if r == 0:
                ax.set_title(transformation_name, fontsize=12, color=NAVY_HEX, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{dname}\n\ny",
                              fontsize=10, color=NAVY_HEX)
            ax.set_xlabel("x")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — transformed-scale linearity check
# ---------------------------------------------------------------------------

def _forward(model: ta.TransformedOLS, y: np.ndarray) -> np.ndarray:
    """Apply the transform's forward map (drives _forward via fitted state)."""
    return model._forward(y)


def plot_transformed_scale(out_path: Path) -> None:
    """For each transform that actually transforms y (skip identity), plot
    T(y) vs x with the OLS line and the R^2 in the transformed space."""
    datasets = get_datasets()
    transforms_keys = ["log", "reciprocal", "boxcox"]
    labels = {
        "log":        r"$T(y) = \log y$",
        "reciprocal": r"$T(y) = 1/y$",
        "boxcox":     r"$T(y) = (y^{\lambda}-1)/\lambda$",
    }

    fig, axes = plt.subplots(
        len(datasets), len(transforms_keys),
        figsize=(13, 9), constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Transformed-scale linearity — does $T(y)$ vs $x$ become straight?",
        fontsize=14, color=NAVY_HEX, fontweight="bold",
    )

    for r, (dataset_name, bundle) in enumerate(datasets):
        x = bundle.X.iloc[:, 0].values
        y = bundle.y.values
        order = np.argsort(x)
        suite = fresh_transforms()

        for c, transformation_name in enumerate(transforms_keys):
            ax = axes[r, c]
            model = suite[transformation_name]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x.reshape(-1, 1), y)
                z = _forward(model, y)
                z_hat = model.predict_transformed(x.reshape(-1, 1))
                # R^2 on transformed scale.
                ss_res = float(np.sum((z - z_hat) ** 2))
                ss_tot = float(np.sum((z - np.mean(z)) ** 2))
                r2 = 1 - ss_res / max(ss_tot, 1e-12)

                ax.scatter(x, z, s=14, alpha=0.5, color=GRAY)
                ax.plot(x[order], z_hat[order],
                        color=TRANSFORM_COLORS[transformation_name], lw=2.2)

                lam = model.selected_param_
                if transformation_name == "boxcox" and lam is not None and np.isfinite(lam):
                    note = f"$R^2$ = {r2:.3f}\nλ = {lam:.2f}"
                else:
                    note = f"$R^2$ = {r2:.3f}"
                ax.text(0.97, 0.03, note, transform=ax.transAxes,
                        fontsize=9, color=NAVY_HEX, ha="right", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="lightgray", pad=2))
            except Exception as exc:
                ax.text(0.5, 0.5, f"fails:\n{exc!s}",
                        transform=ax.transAxes, fontsize=9, ha="center",
                        va="center", color=RED_HEX)

            if r == 0:
                ax.set_title(labels[transformation_name], fontsize=12, color=NAVY_HEX)
            if c == 0:
                ax.set_ylabel(f"{dataset_name}\n\nT(y)",
                              fontsize=10, color=NAVY_HEX)
            ax.set_xlabel("x")
            ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — transformed-scale residual diagnostics
# ---------------------------------------------------------------------------

def plot_residuals(out_path: Path) -> None:
    """Residuals on the transformed scale vs x.  Reveals heteroscedasticity
    and systematic patterns that violate the OLS assumption after T(y)."""
    datasets = get_datasets()
    transforms_keys = ["log", "reciprocal", "boxcox"]
    labels = {
        "log":        r"residuals after $\log y$",
        "reciprocal": r"residuals after $1/y$",
        "boxcox":     r"residuals after Box-Cox",
    }

    fig, axes = plt.subplots(
        len(datasets), len(transforms_keys),
        figsize=(13, 9), constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Transformed-scale residuals — does the OLS assumption hold after T(y)?",
        fontsize=14, color=NAVY_HEX, fontweight="bold",
    )

    for r, (dname, bundle) in enumerate(datasets):
        x = bundle.X.iloc[:, 0].values
        y = bundle.y.values
        suite = fresh_transforms()

        for c, transformation_name in enumerate(transforms_keys):
            ax = axes[r, c]
            model = suite[transformation_name]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x.reshape(-1, 1), y)
                z = _forward(model, y)
                z_hat = model.predict_transformed(x.reshape(-1, 1))
                resid = z - z_hat

                ax.axhline(0, color="black", lw=0.8, alpha=0.6)
                ax.scatter(x, resid, s=14, alpha=0.6,
                           color=TRANSFORM_COLORS[transformation_name])

                # Quick diagnostic: correlation of |resid| with x
                # — large |corr| signals heteroscedasticity.
                corr = float(np.corrcoef(np.abs(resid), x)[0, 1])
                std = float(np.std(resid, ddof=1))
                note = f"std = {std:.3g}\ncorr(|r|, x) = {corr:+.2f}"
                ax.text(0.97, 0.03, note, transform=ax.transAxes,
                        fontsize=9, color=NAVY_HEX, ha="right", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="lightgray", pad=2))
            except Exception as exc:
                ax.text(0.5, 0.5, f"fails:\n{exc!s}",
                        transform=ax.transAxes, fontsize=9, ha="center",
                        va="center", color=RED_HEX)

            if r == 0:
                ax.set_title(labels[transformation_name], fontsize=12, color=NAVY_HEX)
            if c == 0:
                ax.set_ylabel(f"{dname}\n\nresidual",
                              fontsize=10, color=NAVY_HEX)
            ax.set_xlabel("x")
            ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_original_fits(OUT_DIR / "lin_fits_original.png")
    plot_transformed_scale(OUT_DIR / "lin_transformed_scale.png")
    plot_residuals(OUT_DIR / "lin_residuals.png")
    print("\nThree figures written to comparison_results/.")


if __name__ == "__main__":
    main()
