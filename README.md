# ZZU Transformations

**Zhang-Zhao-Uzunoglu (ZZU) Transformations** — a comparative study of response-transformation OLS versus direct nonlinear regression, plus a diagnostic-guided hybrid workflow that combines both.

**Authors:** Alvin Zhang · Wynn Zhao · Arda Uzunoglu  
**Course project, April–May 2026**

---

## Overview

When a relationship between predictors and response is nonlinear, two broad strategies exist:

1. **Linearization** — apply a transformation T(y) so that the transformed response is linear in the predictors, then fit OLS. Simple and interpretable, but may distort the error structure, introduce heteroscedasticity, and bias back-transformed predictions.
2. **Direct nonlinear optimization** — fit y = f(x, θ) directly by minimizing SSE. Avoids distortion but is sensitive to initialization.

The **ZZU workflow** bridges both: screen candidate transformations on a validation split, use the best linearized fit to warm-start a nonlinear optimizer, and apply a bias correction on the original scale.

---

## Repository Structure

```
zzu-transformations/
├── transformation_algorithms.py   # Core algorithm library (Sections 1–15)
├── toy_data.py                    # Synthetic dataset generators
├── ZZU_main_with_py_imports.ipynb # Main experiment notebook
├── transformation_algorithms.ipynb# Algorithm documentation notebook
├── toy_data.ipynb                 # Dataset documentation notebook
├── test.py                        # End-to-end validation script
├── run_comparison.py              # Full benchmark across all 5 datasets
├── comparison_results/            # Outputs of run_comparison.py
│   ├── raw_results.csv            #   long-form: 1 row per (dataset, method, seed)
│   ├── summary_by_method.csv      #   mean ± std RMSE / R² / convergence
│   ├── rmse_by_method.png         #   per-dataset bar chart, methods sorted
│   └── fit_overlay.png            #   best-fit overlays for the 4 1D datasets
├── generated_datasets/            # CSV exports of all five synthetic datasets
│   ├── exponential_multiplicative.csv
│   ├── exponential_additive.csv
│   ├── michaelis_menten.csv
│   ├── logistic_growth.csv
│   └── multivariable_nonlinear.csv
└── main (2).tex                   # Project write-up (LaTeX)
```

---

## Implemented Components

### `transformation_algorithms.py`

The library is organized into 15 sections. Sections 1–9 cover transformation-based OLS (completed first); Sections 10–15 add the nonlinear baselines and ZZU hybrid.

#### Transformation-based OLS (Sections 1–9)

| Section | What it does |
|---------|-------------|
| 1 | Setup and imports |
| 2 | Linear algebra helpers: `as_2d`, `add_intercept`, `ols_fit`, `ols_predict` |
| 3 | Basic transformation families: identity, log, reciprocal, power-scale (each with forward, inverse, log-Jacobian) |
| 4 | Box-Cox transformation with profile-likelihood λ selection |
| 5 | Yeo-Johnson transformation (handles zero and negative responses) |
| 6 | `choose_lambda_by_profile_likelihood` — grid search over λ ∈ [−2, 2] |
| 7 | Duan-style smearing correction for retransformation bias |
| 8 | `TransformedOLS` — unified dataclass: fit/predict/summary for all transform families |
| 9 | `regression_metrics`, `residual_diagnostics`, `evaluate_transformed_models` |

#### Nonlinear baselines + ZZU hybrid (Sections 10–15)

| Section | Symbol | Description |
|---------|--------|-------------|
| 10 | `numerical_jacobian` | Central finite-difference Jacobian with adaptive per-parameter step |
| 11 | `GradientDescentRegressor` | Vanilla gradient descent on SSE/n with optional learning-rate decay |
| 12 | `GaussNewtonRegressor` | Gauss-Newton with self-activating Levenberg-Marquardt damping |
| 13 | `BFGSRegressor` | Pure-numpy BFGS with inverse Hessian update and backtracking Armijo line search |
| 14 | `ZZUTransformRegressor` | Three-step ZZU hybrid workflow (screen → warm-start → bias correction) |
| 15 | `evaluate_nonlinear_models` | Batch-fit/evaluate dict of nonlinear regressors, returning a sorted DataFrame |

All three nonlinear regressors share a common interface:
```python
reg.fit(X, y, theta_init)  # -> self
reg.predict(X)             # -> ndarray (n,)
reg.summary()              # -> dict
```
Failures are caught and stored in `reg.fit_error_` rather than crashing the caller.

---

### `toy_data.py`

Five synthetic datasets, each returned as a `DatasetBundle` (with `X`, `y`, `y_true`, `params`, `description`):

| Name | Function | Purpose |
|------|----------|---------|
| `exponential_multiplicative` | y = a·exp(b·x)·η, log(η)~N | Best case for log-linearization |
| `exponential_additive` | y = a·exp(b·x) + ε | Failure mode for naïve log-linearization |
| `michaelis_menten` | y = Vmax·x/(Km+x) + ε | Saturating curve; tests reciprocal linearization |
| `logistic_growth` | y = L/(1+exp(−k(x−x₀))) + ε | S-curve; tests nonlinear optimizer sensitivity |
| `multivariable_nonlinear` | y = 2·exp(0.4·x₁) + 3·x₂^1.5 + 10/(1+x₃) + ε | No single global linearization; designed for ZZU |

All five are exported to `generated_datasets/` as CSV files.

---

## ZZU Workflow

```python
import numpy as np
import transformation_algorithms as ta

# 1. Define the nonlinear model f(X, theta)
model_fn = lambda X, t: t[0] * np.exp(t[1] * X[:, 0])

# 2. Define how to convert the best linearized model's coefficients
#    into a nonlinear warm-start.  This is model-specific.
#    Here: log(y) = beta0 + beta1*x  =>  a = exp(beta0), b = beta1
def coeff_to_init(best_tols_model):
    beta = best_tols_model.beta_
    return np.array([np.exp(beta[0]), beta[1]])

# 3. Fit ZZU
zzu = ta.ZZUTransformRegressor(
    model_fn=model_fn,
    coeff_to_init=coeff_to_init,
    nonlinear_method='bfgs',           # or 'gauss_newton' / 'gradient_descent'
    transformations={                  # restrict screening to valid transforms
        'log_smear': ta.TransformedOLS(transform='log', use_smearing=True),
    },
).fit(X, y)

print(zzu.summary())
# {'best_transform': 'log_smear', 'converged': True, 'final_theta': [2.167, 0.679], ...}
```

**Key design rule:** `coeff_to_init` must be consistent with the transforms in the screening suite. Restrict `transformations` to those you know how to invert back to nonlinear parameters.

---

## Quick Start

```bash
# Sanity check: all three optimizers + ZZU on one dataset
python test.py

# Full benchmark: 11 transformed-OLS variants + GD/GN/BFGS + ZZU
# across all 5 datasets, 10 train/test splits each
python run_comparison.py
```

`test.py` prints, for the exponential multiplicative dataset (n=120):
```
GD:   converged=False, n_iter=5000, RMSE≈7.21, theta≈[2.13, 0.68]
GN:   converged=True,  n_iter=7,    RMSE≈7.21, theta≈[2.17, 0.68]
BFGS: converged=True,  n_iter=24,   RMSE≈7.21, theta≈[2.17, 0.68]
ZZU:  best_transform=log_smear, converged=True, RMSE≈7.21
```

`run_comparison.py` writes raw and summarized CSVs plus two plots to
`comparison_results/`, and prints the top-3 methods per dataset.

Dependencies: `numpy`, `pandas`, `matplotlib` (no scipy).

---

## Benchmark Results

Top method per dataset, mean test RMSE over 10 random 80/20 splits:

| Dataset | Winner | Family | Mean RMSE | R² | Notes |
|---------|--------|--------|-----------|----|-------|
| `exponential_multiplicative` | `log_smear` | linearized OLS | 7.58 | 0.81 | Log linearization is exact when noise is multiplicative lognormal |
| `exponential_additive` | GD / GN / BFGS | nonlinear | 5.14 | 0.90 | Log distorts additive noise; nonlinear sweeps the top 3 |
| `michaelis_menten` | BFGS / GN / **ZZU** (tie) | nonlinear / hybrid | 0.297 | 0.97 | ZZU matches direct nonlinear with no clean linearization available |
| `logistic_growth` | **ZZU** | hybrid | 3.065 | 0.99 | Edges BFGS/GN by ~0.002 RMSE |
| `multivariable_nonlinear` | **ZZU** | hybrid | 4.65 | 0.97 | Designed to defeat any single transform; ZZU beats BFGS/GN by ≈0.10 RMSE |

ZZU wins or ties three of five datasets, including the multivariable benchmark
the dataset was designed to expose. Log-linearization wins outright only on
the multiplicative-noise case where its assumptions are exactly satisfied.

See `comparison_results/summary_by_method.csv` for the full per-method table
and `rmse_by_method.png` for a visual ranking.

---

## Progress

| Task | Status |
|------|--------|
| Finalize scope, choose dataset, references | Done |
| Transformation baselines (Box-Cox, YJ, log, reciprocal, power, smearing) | Done |
| 2D toy examples (exponential multiplicative & additive) | Done |
| Synthetic multivariable dataset (`multivariable_nonlinear`) | Done |
| Nonlinear baselines: gradient descent, Gauss-Newton, BFGS | Done |
| ZZU hybrid workflow | Done |
| Full comparison on synthetic suite (`run_comparison.py`) | Done |
| Notebook integration (`ZZU_main_with_py_imports.ipynb`) | In progress |
| Real-world dataset application | Upcoming |
| Final report / presentation | Upcoming |

---

## Contribution Summary

- **Alvin Zhang** — nonlinear baselines (GD, Gauss-Newton, BFGS), ZZU hybrid implementation, report writing
- **Wynn Zhao** — real-world dataset selection and preprocessing, evaluation and visualization, ZZU development
- **Arda Uzunoglu** — literature review, toy examples, introduction and related work, synthetic experiments, ZZU development
