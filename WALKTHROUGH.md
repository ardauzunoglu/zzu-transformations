# ZZU Transformations — Simple Walkthrough

This is a quick map of the repo, where we show what each main file or folder is for, and what it produces.

## Main Code

- [scripts/algorithms.py](/Users/ardauzunoglu/zzu-transformations/scripts/algorithms.py): Core modeling library.
  Result: implements transformed OLS, nonlinear optimizers, and the ZZU hybrid regressor.

- [scripts/toy_data.py](/Users/ardauzunoglu/zzu-transformations/scripts/toy_data.py): Synthetic dataset generators.
  Result: creates the benchmark datasets used throughout the project.

- [reproducibility.py](/Users/ardauzunoglu/zzu-transformations/reproducibility.py): Shared seeds and split settings.
  Result: keeps benchmark runs reproducible.

## Benchmark Scripts

- [run_comparison.py](/Users/ardauzunoglu/zzu-transformations/run_comparison.py): Main benchmark across synthetic datasets.
  Result: writes summary CSVs and plots into [comparison_results](/Users/ardauzunoglu/zzu-transformations/comparison_results). Run to reproduce Figure 2 from the report.

- [scripts/cost_analysis.py](/Users/ardauzunoglu/zzu-transformations/scripts/cost_analysis.py): Cost/efficiency analysis.
  Result: compares RMSE against fit time, iterations, and model evaluations. Run to reproduce Figure 1 from the report.

- [scripts/zzu_inner_method_comparison.py](/Users/ardauzunoglu/zzu-transformations/scripts/zzu_inner_method_comparison.py): Inner-optimizer ablation.
  Result: compares pure optimizers against ZZU plus the same optimizer. Run to reproduce Figure 3 from the report.

## Visualization Scripts

- [scripts/visualize_synthetic_data.py](/Users/ardauzunoglu/zzu-transformations/scripts/visualize_synthetic_data.py): Synthetic-data overview plots.
  Result: writes dataset visualizations into [synthetic_visualizations](/Users/ardauzunoglu/zzu-transformations/synthetic_visualizations).

- [scripts/build_linearization_figures.py](/Users/ardauzunoglu/zzu-transformations/scripts/build_linearization_figures.py): Educational transform figures.
  Result: illustrates when linearization helps or fails.

- [scripts/build_optimizer_trajectories.py](/Users/ardauzunoglu/zzu-transformations/scripts/build_optimizer_trajectories.py): Optimizer trajectory figures.
  Result: shows how GD, Gauss-Newton, and BFGS move across the loss surface.

## Notebooks

- [concrete_analysis.ipynb](/Users/ardauzunoglu/zzu-transformations/concrete_analysis.ipynb): Real-world concrete dataset analysis. Run to reproduce Figure 4 from the report.

- [bike_analysis.ipynb](/Users/ardauzunoglu/zzu-transformations/bike_analysis.ipynb): Real-world bike-sharing dataset analysis.

## Output Folders

- [comparison_results](/Users/ardauzunoglu/zzu-transformations/comparison_results): Benchmark outputs.
  Result: CSV summaries, RMSE plots, Pareto plots, overlays, and ablation figures.

- [synthetic_visualizations](/Users/ardauzunoglu/zzu-transformations/synthetic_visualizations): Synthetic dataset figures.

- [datasets](/Users/ardauzunoglu/zzu-transformations/datasets): Input data.
  Result: synthetic CSVs plus real-world datasets used by the notebooks.

## Tests

- [tests](/Users/ardauzunoglu/zzu-transformations/tests): Test suite.
  Result: checks reproducibility, transforms, optimizers, and helper utilities.

## Short Mental Model

- `algorithms.py` is the engine.
- `toy_data.py` creates the synthetic problems.
- `run_comparison.py`, `cost_analysis.py`, and `zzu_inner_method_comparison.py` are the main experiments.
- `comparison_results/` and `synthetic_visualizations/` hold the outputs.
- The notebooks show the workflow on real datasets.