from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
    category=UserWarning,
)

import toy_data as td
from reproducibility import reproduce_dir


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = reproduce_dir("synthetic_visualizations", REPO_ROOT)
ONE_DIMENSIONAL_DATASETS = [
    "exponential_multiplicative",
    "exponential_additive",
    "michaelis_menten",
    "logistic_growth",
]


def save_overview_grid(suite: dict[str, td.DatasetBundle], output_dir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, name in zip(axes.ravel(), ONE_DIMENSIONAL_DATASETS):
        td.plot_one_dimensional_dataset(
            suite[name],
            ax=ax,
            title=name.replace("_", " ").title(),
        )
    fig.suptitle("Synthetic 1D Dataset Overview", fontsize=16)
    path = output_dir / "synthetic_data_overview.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_individual_one_dimensional_plots(
    suite: dict[str, td.DatasetBundle],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    written: list[Path] = []
    for name in ONE_DIMENSIONAL_DATASETS:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        td.plot_one_dimensional_dataset(
            suite[name],
            ax=ax,
            title=name.replace("_", " ").title(),
        )
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def save_multivariable_plot(suite: dict[str, td.DatasetBundle], output_dir: Path, dpi: int) -> Path:
    fig = td.plot_multivariable_marginals(suite["multivariable_nonlinear"], figsize=(15, 4.5))
    fig.suptitle("Multivariable Nonlinear Dataset Marginals", fontsize=16)
    path = output_dir / "multivariable_nonlinear_marginals.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset visualizations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG files will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Resolution for saved PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = td.generate_default_suite()
    written = [save_overview_grid(suite, output_dir, args.dpi)]
    written.extend(save_individual_one_dimensional_plots(suite, output_dir, args.dpi))
    written.append(save_multivariable_plot(suite, output_dir, args.dpi))

    print("Wrote visualizations:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
