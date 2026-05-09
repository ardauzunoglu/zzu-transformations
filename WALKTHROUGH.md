# ZZU Transformations — Simple Walkthrough

This is a quick map of the repo, where we show what each main file or folder is for, and what it produces.

## Main Code

- [scripts/algorithms.py](scripts/algorithms.py): Core modeling library.
  Result: implements transformed OLS, nonlinear optimizers, and the ZZU hybrid regressor.

- [scripts/toy_data.py](scripts/toy_data.py): Synthetic dataset generators.
  Result: creates the benchmark datasets used throughout the project.

- [reproducibility.py](scripts/reproducibility.py): Shared seeds and split settings.
  Result: keeps benchmark runs reproducible.

## Benchmark Scripts

- [run_comparison.py](scripts/run_comparison.py): Main benchmark across synthetic datasets.
  Result: writes summary CSVs and plots into [comparison_results](comparison_results). Run to reproduce Figure 2 from the report.

- [scripts/cost_analysis.py](scripts/cost_analysis.py): Cost/efficiency analysis.
  Result: compares RMSE against fit time, iterations, and model evaluations. Run to reproduce Figure 1 from the report.

- [scripts/zzu_inner_method_comparison.py](scripts/zzu_inner_method_comparison.py): Inner-optimizer ablation.
  Result: compares pure optimizers against ZZU plus the same optimizer. Run to reproduce Figure 3 from the report.

## Visualization Scripts

- [scripts/visualize_synthetic_data.py](scripts/visualize_synthetic_data.py): Synthetic-data overview plots.
  Result: writes dataset visualizations into [synthetic_visualizations](synthetic_visualizations).

- [scripts/build_linearization_figures.py](scripts/build_linearization_figures.py): Educational transform figures.
  Result: illustrates when linearization helps or fails.

- [scripts/build_optimizer_trajectories.py](scripts/build_optimizer_trajectories.py): Optimizer trajectory figures.
  Result: shows how GD, Gauss-Newton, and BFGS move across the loss surface.

## Notebooks

- [pipeline_walkthrough.ipynb](pipeline_walkthrough.ipynb): Head-to-tail synthetic pipeline. Generates the five datasets, renders the pedagogical figures, runs the main accuracy benchmark, the cost analysis, the ZZU inner-method ablation, and four stress-test studies (noise sensitivity, convergence rate, init robustness, sample-size sensitivity). Outputs land in `notebook_outputs/` so the canonical folders are untouched.

- [concrete_analysis.ipynb](concrete_analysis.ipynb): Real-world concrete dataset analysis. Run to reproduce Figure 4 from the report.

- [bike_analysis.ipynb](bike_analysis.ipynb): Real-world bike-sharing dataset analysis.

## Output Folders

- [comparison_results](comparison_results): Benchmark outputs.
  Result: CSV summaries, RMSE plots, Pareto plots, overlays, and ablation figures.

- [synthetic_visualizations](synthetic_visualizations): Synthetic dataset figures.

- [datasets](datasets): Input data.
  Result: synthetic CSVs plus real-world datasets used by the notebooks.

## Tests

- [tests](tests): Test suite.
  Result: checks reproducibility, transforms, optimizers, and helper utilities.

## Short Mental Model

- `algorithms.py` is the engine.
- `toy_data.py` creates the synthetic problems.
- `run_comparison.py`, `cost_analysis.py`, and `zzu_inner_method_comparison.py` are the main experiments.
- `comparison_results/` and `synthetic_visualizations/` hold the outputs.
- `pipeline_walkthrough.ipynb` runs the whole synthetic pipeline end-to-end in one place.
- The other notebooks show the workflow on real datasets.
