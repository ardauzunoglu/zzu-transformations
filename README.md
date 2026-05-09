# ZZU Transformations

**Zhang-Zhao-Uzunoglu (ZZU) Transformations** — a comparative study of response-transformation OLS versus direct nonlinear regression, plus a diagnostic-guided hybrid workflow that combines both.

**Authors:** Alvin Zhang · Wynn Zhao · Arda Uzunoglu  
**Course project, April–May 2026**

Please refer to [WALKTHROUGH.md](WALKTHROUGH.md) for reproducing our results presented in the report.
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
├── scripts/
│   ├── algorithms.py                   # Core algorithm library (Sections 1–15)
│   ├── toy_data.py                     # Synthetic dataset generators
│   ├── reproducibility.py              # Single source of truth for seeds / split sizes
│   ├── run_comparison.py               # Full benchmark across all 5 synthetic datasets
│   ├── cost_analysis.py                # Time / iterations / model_fn calls per fit
│   ├── zzu_inner_method_comparison.py  # Ablation: pure vs ZZU+{GD,GN,BFGS}
│   ├── visualize_synthetic_data.py     # Per-dataset PNGs + 2x2 overview
│   ├── build_linearization_figures.py  # 3 pedagogical figures on transforms
│   └── build_optimizer_trajectories.py # SSE-surface contour + GD/GN/BFGS paths
├── tests/                          # 158 pytest cases across 7 modules
├── pipeline_walkthrough.ipynb      # Head-to-tail synthetic pipeline (recommended entry point)
├── concrete_analysis.ipynb         # Real-world: UCI Concrete Compressive Strength
├── bike_analysis.ipynb             # Real-world: UCI Bike Sharing
├── comparison_results/             # Outputs of run_comparison.py + cost_analysis.py +
│   │                               # zzu_inner_method_comparison.py + figure scripts
│   ├── raw_results.csv             #   accuracy: 1 row per (dataset, method, seed)
│   ├── summary_by_method.csv       #   mean ± std RMSE / R² / convergence
│   ├── rmse_by_method.png          #   per-dataset bar chart, methods sorted
│   ├── fit_overlay.png             #   best-fit overlays for the 4 1D datasets
│   ├── cost_results.csv            #   cost: fit_time_sec, n_iter, n_model_evals
│   ├── cost_summary.csv            #   mean cost metrics per (dataset, method)
│   ├── cost_pareto.png             #   RMSE vs fit-time Pareto (log-log)
│   └── warm_vs_cold.png            #   BFGS warm-start vs cold-start comparison
├── datasets/                       # Input data
│   ├── synthetic_datasets/         #   CSV exports of the five synthetic datasets
│   ├── concrete.csv                #   UCI Concrete Compressive Strength
│   └── bike_sharing_dataset/       #   UCI Bike Sharing
└── synthetic_visualizations/       # Per-dataset PNGs (output of visualize_synthetic_data.py)
```

---

## Implemented Components

### `algorithms.py`

The library is organized into 15 sections. Sections 1–9 cover transformation-based OLS, and Sections 10–15 add the nonlinear baselines and ZZU hybrid.

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

## How to Use

All three method families share the same scikit-learn-style interface
(`fit` returns `self`; fitted attributes end with `_`). The sections
below give the minimal call pattern for each.

### Inputs and outputs (uniform across all classes)

| Argument | Shape | Notes |
|----------|-------|-------|
| `X` (fit, predict) | `(n, p)` or `(n,)` | 1D arrays are auto-promoted to 2D via `as_2d` at fit time |
| `y` (fit) | `(n,)` | Original response scale |
| Return of `predict(X_new)` | `(m,)` | **Always on the original response scale** — back-transformation and bias correction are handled internally |

Common fitted attributes: `.beta_` / `.theta_` (coefficients),
`.converged_`, `.n_iter_`, `.fit_error_` (an error string if the fit
caught an exception, or `None` otherwise — fits never raise).

### Linear methods — `TransformedOLS`

Fits OLS in a transformed response space, then inverts back to the
original scale (with optional Duan-style smearing correction):

```python
import algorithms as ta

m = ta.TransformedOLS(transform="boxcox", use_smearing=True).fit(X, y)
y_hat = m.predict(X_new)         # original scale, smearing applied
m.summary()                      # {transform, selected_param_or_lambda, coefficients, ...}
```

Six transform families are available via the `transform=` argument:
`identity`, `log`, `reciprocal`, `power` (requires `param=p`), `boxcox`,
`yeojohnson`. For `boxcox` and `yeojohnson`, λ is auto-selected by
profile likelihood unless `lambda_=` is given. Set `use_smearing=False`
to disable retransformation bias correction.

### Nonlinear methods — `GradientDescentRegressor`, `GaussNewtonRegressor`, `BFGSRegressor`

Direct nonlinear least squares on the original scale. Supply your model
function `f(X, theta) -> (n,)` and an initial parameter vector:

```python
import numpy as np

model_fn = lambda X, t: t[0] * np.exp(t[1] * X[:, 0])    # y = a · exp(b·x)
theta_init = np.array([1.0, 0.1])

reg = ta.BFGSRegressor(model_fn=model_fn).fit(X, y, theta_init)
y_hat = reg.predict(X_new)
reg.theta_, reg.converged_, reg.n_iter_, reg.fit_error_
```

Picking the optimizer:
- **`BFGSRegressor`** — robust default; pure-NumPy inverse Hessian update with backtracking Armijo line search.
- **`GaussNewtonRegressor`** — fastest when the problem is well-posed; self-activating Levenberg-Marquardt damping handles ill-conditioned steps.
- **`GradientDescentRegressor`** — pedagogical baseline; usually 100× slower than BFGS/GN for the same accuracy.

If you don't pass `jacobian_fn=`, the optimizer uses central
finite-difference numerical Jacobians (2p model evaluations per
Jacobian).

### Hybrid — `ZZUTransformRegressor`

Combines the two families: screen linearizations on a validation split,
warm-start a nonlinear optimizer from the winning linearization, and
apply an additive bias correction at predict time. Inputs and outputs
follow the same conventions as above (`X: (n, p)`, `y: (n,)`,
`predict → (m,)` on the original scale).

```python
import numpy as np
import algorithms as ta

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

**Key design rule:** `coeff_to_init` must be consistent with the
transforms in the screening suite. Restrict `transformations` to those
you know how to invert back to nonlinear parameters; supply
`fallback_theta_init=` for safety.

---

## Quick Start

The fastest path to reproducing the synthetic-side results is the [pipeline_walkthrough.ipynb](pipeline_walkthrough.ipynb) notebook — it generates the five datasets, renders all pedagogical figures, runs the main accuracy benchmark, the cost analysis, the ZZU inner-method ablation, and four stress-test studies in one place (~3–6 minutes end-to-end). Outputs land in `notebook_outputs/` so the canonical folders stay untouched. The script entry points below remain available for running individual stages from the command line.

```bash
# Sanity check: all three optimizers + ZZU on one dataset
python tests/test.py

# Full benchmark: 11 transformed-OLS variants + GD/GN/BFGS + ZZU
# across all 5 datasets, 10 train/test splits each
python scripts/run_comparison.py

# Generate synthetic dataset figures in ./synthetic_visualizations
python scripts/visualize_synthetic_data.py
```

`test.py` prints, for the exponential multiplicative dataset (n=120):
```
GD:   converged=False, n_iter=5000, RMSE≈7.21, theta≈[2.13, 0.68]
GN:   converged=True,  n_iter=7,    RMSE≈7.21, theta≈[2.17, 0.68]
BFGS: converged=True,  n_iter=24,   RMSE≈7.21, theta≈[2.17, 0.68]
ZZU:  best_transform=log_smear, converged=True, RMSE≈7.21
```

`scripts/run_comparison.py` writes raw and summarized CSVs plus two
plots to `comparison_results/`, and prints the top-3 methods per
dataset.

Dependencies: `numpy`, `pandas`, `matplotlib` (no scipy).

---

## Reproducibility

All randomness in the codebase flows through `np.random.default_rng(seed)`
— a *local* generator that does not touch global state. Every default
seed is registered in [`scripts/reproducibility.py`](scripts/reproducibility.py)
so "what random state produced figure X" is answerable in one place
rather than three:

| Constant | Value | Used by |
|----------|------:|---------|
| `N_SEEDS` | 10 | `run_comparison.py`, `cost_analysis.py`, `zzu_inner_method_comparison.py` (loop seeds 0..9) |
| `TEST_FRACTION` | 0.2 | every benchmark train/test split |
| `DEFAULT_SEED` | 123 | `train_test_split_arrays` default |
| `ZZU_VALIDATION_SEED` | 0 | `ZZUTransformRegressor` internal screen split |
| `DATASET_SEEDS` | 101–105 | `make_exponential_multiplicative`, …, `make_multivariable_nonlinear` |

Two consequences of this design:

- **Bit-identical reproduction.** Running any benchmark script twice
  with the same seed produces byte-equal CSVs. Verified by
  `tests/test_reproducibility.py`, which exercises every layer of the
  pipeline (dataset generation → split → TransformedOLS / Box-Cox λ
  selection → ZZU end-to-end).
- **No hidden global state.** Project code never calls
  `np.random.seed(...)` or the stdlib `random` module. The
  `seed_everything()` helper in `reproducibility.py` exists *only* as a
  defensive escape hatch for notebooks or third-party libraries that
  may rely on global PRNGs.

To pin down a specific run: pass `seed=` explicitly anywhere it's
accepted, or import the constants:

```python
# From the project root, with scripts/ on PYTHONPATH:
from reproducibility import N_SEEDS, TEST_FRACTION, make_rng

rng = make_rng(seed=42)              # fresh local generator
```

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

## Cost Analysis

Run `python scripts/cost_analysis.py` to instrument every fit with wall-clock time,
optimizer iterations, and model-function evaluations (the latter captures
Jacobian work, since numerical Jacobian calls `model_fn` 2p times per
Jacobian). Outputs land in `comparison_results/`.

### Cost-vs-accuracy by family

Order-of-magnitude per-fit cost on these datasets (n = 120–500):

| Family | Time per fit | Iterations | Notes |
|--------|-------------:|-----------:|-------|
| Linearized OLS | ~0.1 ms | n/a | Pure linear algebra; one Box-Cox/Yeo-Johnson grid search dominates |
| BFGS / Gauss-Newton | ~1–10 ms | 5–50 | ~10× slower than linear; numerical Jacobian is the bottleneck |
| ZZU hybrid | ~3–30 ms | 5–50 | Adds the screening phase on top of one nonlinear fit |
| Gradient descent | ~100 ms – 1 s | 5000 (capped) | Dominated everywhere — slow and rarely converges |

`cost_pareto.png` shows the full RMSE-vs-time Pareto per dataset. Linearized
OLS is on the frontier only on `exponential_multiplicative`. On every other
dataset, GN or BFGS (or ZZU) dominates: the ~10× extra cost buys a 5–100×
RMSE reduction. ZZU appears on the frontier on `multivariable_nonlinear`,
where its ~3× overhead over BFGS purchases a real RMSE improvement
(4.65 vs 4.75).

### Does the screening step pay for itself? Warm-start vs cold-start BFGS

The two strategies are run with **identical** model, dataset, optimizer,
and convergence tolerance. The only difference is the initial parameter
vector handed to BFGS.

**Cold start (baseline).** A *data-driven heuristic* derived directly
from the observed `(X, y)` without fitting any model. For the exponential
datasets `y = a · exp(b · x)`:
```
theta_cold = [a0, b0]   where   a0 = max(min(y), 1e-3),   b0 = 0.1
```
This is intentionally cheap: it uses only summary statistics and a fixed
small rate. It's what someone would code if they didn't know any better.

**Warm start (ZZU's contribution).** Run Step 1 of ZZU first — fit the
log-linearized OLS `log(y) = β₀ + β₁·x` on the training data — then
*invert* the linear coefficients back into nonlinear parameters via the
user-supplied `coeff_to_init`:
```
theta_warm = [exp(β0), β1]    # the analytic inverse of log-linearization
```
This costs one extra OLS solve (~0.1 ms) but produces an init that — when
the linearization assumption is correct — already lies near the SSE
optimum.

**Concretely** on `exponential_multiplicative` (true params `a = 2`, `b = 0.7`):
| Init | a₀ | b₀ | Distance to truth (L²) |
|------|----:|----:|------------------------:|
| cold | ~2.0 | 0.10 | ~0.60 (driven by b₀) |
| warm | ~2.0 | ~0.69 | ~0.01 |

Mean over 10 splits, with the same BFGS settings either way:

| Dataset | Init | Iterations | model_fn calls | Time (ms) |
|---------|------|-----------:|---------------:|----------:|
| `exponential_multiplicative` | cold | 13 | 205 | 1.6 |
| `exponential_multiplicative` | warm | **9** | **150** | **1.2** |
| `exponential_additive` | cold | **20** | **302** | **2.4** |
| `exponential_additive` | warm | 26 | 395 | 3.1 |

This is the cleanest theory-vs-empirics match in the project:

- On `exponential_multiplicative`, the noise is multiplicative lognormal, so
  the log transform is *exact* and the screened init lands in the basin of
  the SSE optimum. Warm start cuts iterations and model evals by ~25–30%.
- On `exponential_additive`, the noise is additive Gaussian, so log
  *distorts* the residuals. The screened init looks plausible but is biased
  away from the original-scale SSE optimum. Warm start runs ~30% slower
  than cold.

The takeaway for ZZU: the screening phase is worth its cost when the
chosen transform is consistent with the noise structure. When it isn't,
diagnostics on the screening table should flag it before the warm start
is used. This is exactly the diagnostic-guided spirit of the workflow.

### Practical recommendations

- For `exp_mult`-style data (multiplicative noise), use linearized OLS
  directly — it's 10× faster than any nonlinear method at the same RMSE.
- For everything else, prefer **Gauss-Newton or BFGS** as the workhorse;
  drop GD entirely (it's 100× slower without accuracy benefit).
- Use ZZU when (a) you have no good heuristic init for the nonlinear
  optimizer, or (b) the screening step is genuinely informative about the
  noise structure (in which case it pays back via faster warm-started
  convergence).

---

## Real-World Dataset Evaluation

`concrete_analysis.ipynb` applies the full ZZU workflow to the **UCI Concrete Compressive Strength** dataset (UCI ML Repository #165): 1 030 laboratory mix-design records, 8 predictors, target in MPa.

### Dataset

| Property | Value |
|----------|-------|
| Observations | 1030 |
| Predictors | cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age |
| Target | Compressive strength (MPa) |
| Range | ~2.3 – 82.6 MPa (~36× ratio) |
| Zero-inflated predictors | slag, fly ash, superplasticizer (cannot appear inside log/power transforms) |

The response is right-skewed; a log transform substantially reduces skewness. Age exhibits near-log-linear growth in strength, motivating a power-law time term. The dominant strength driver is the water/cement ratio (Abrams' Law, 1919).

### Feature Engineering

Two domain-motivated features are added before fitting:

- `wc_ratio = water / cement` — highest single-feature correlation with strength
- `binder = cement + slag + fly_ash` — total cementitious content

The nonlinear model uses log-transformed inputs to align the screening step with the nonlinear parameterisation:

```
X_zzu = [ log(cement/water),  log(age),  slag,  fly_ash,  superplasticizer ]
```

### Model

Functional form:

```
y = exp( θ₀ + θ₁·log(cement/water) + θ₂·log(age)
             + θ₃·slag + θ₄·fly_ash + θ₅·SP )
```

Because the model is log-linear in the transformed inputs, the ZZU screening step fits `log(y) ~ X_zzu` via OLS and the warm-start inversion is exact: `θ_init = beta_` directly.

### Methods Compared

| Method | Family | Description |
|--------|--------|-------------|
| `identity_ols` | Linearized OLS | Plain OLS on log-transformed X features, no y-transform |
| `log_smear` | Linearized OLS | ZZU Step 1 alone; log(y) ~ X with Duan smearing |
| `boxcox_smear` | Linearized OLS | Box-Cox λ selected by profile likelihood |
| `bfgs_cold` | Nonlinear cold | BFGS with heuristic Abrams'-Law init, no screening |
| `gn_cold` | Nonlinear cold | Gauss-Newton with the same cold init |
| `zzu_bfgs` | **ZZU hybrid** | Screening → warm start → BFGS refinement |

Evaluation: 10 random 80/20 seeds, test RMSE and R² reported as mean ± std.

### Key Findings

- **ZZU warm-start vs cold-start**: because the model is exactly log-linear, the screened init lands near the SSE optimum; `zzu_bfgs` consistently converges in fewer iterations than `bfgs_cold` and achieves lower or equal test RMSE.
- **Linearized OLS alone** (`log_smear`) is a strong baseline given the near-log-linear structure, but the additive supplement terms for zero-inflated features are not captured by the linear model, leaving residual error that the nonlinear optimizer corrects.
- **Box-Cox** (`boxcox_smear`) offers no advantage over `log_smear` here: profile-likelihood selects λ ≈ 0 (the log), confirming the log transform is appropriate.
- **Gauss-Newton cold** struggles more than BFGS cold because the Jacobian is ill-conditioned early in the search when starting far from the optimum.

The concrete experiment illustrates the core ZZU thesis on real data: when a good linearization exists, screening identifies it automatically and the warm start cuts optimizer cost; when zero-inflated or otherwise unlinearizable terms are present, the nonlinear refinement step recovers what the screened OLS cannot represent.

---

## Contribution Summary

- **Alvin Zhang** — nonlinear baselines (GD, Gauss-Newton, BFGS), ZZU hybrid implementation, cost analysis, report writing
- **Wynn Zhao** — real-world dataset selection and preprocessing, evaluation and visualization, ZZU development
- **Arda Uzunoglu** — literature review, toy examples, introduction and related work, synthetic experiments, ZZU development
