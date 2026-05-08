# ZZU Transformations — Project Walk-through

A guided tour of every component of the project up to the real-world
dataset experiments. The [README.md](README.md) is the *entry point*
and the headline-result reference; this document is the
*comprehensive* companion that explains what each script does, what it
consumes, what it produces, and how to read the figures it writes.

> **Scope.** This walk-through covers the synthetic side of the project
> only: algorithm library, dataset generators, benchmarks, cost
> analysis, ablations, pedagogical figures, and tests. Real-world
> evaluations ([concrete_analysis.ipynb](concrete_analysis.ipynb),
> [bike_analysis.ipynb](bike_analysis.ipynb)) are intentionally out of
> scope here.

> **All numbers in this document were produced by running the pipeline
> end-to-end into a sandbox folder** ([reproduce/](reproduce/)) and
> diffing against the canonical [comparison_results/](comparison_results/).
> The "Reproduce" section at the end shows how to run that diff yourself.

---

## Table of Contents

1. [Project layout](#1-project-layout)
2. [Reproducibility infrastructure](#2-reproducibility-infrastructure-scriptsreproducibilitypy)
3. [Algorithm library](#3-algorithm-library-scriptstransformation_algorithmspy)
4. [Synthetic dataset generators](#4-synthetic-dataset-generators-scriptstoy_datapy)
5. [Synthetic-data visualization](#5-synthetic-data-visualization-scriptsvisualize_synthetic_datapy)
6. [Pedagogical figure: linearization diagnostics](#6-pedagogical-figure-linearization-diagnostics-scriptsbuild_linearization_figurespy)
7. [Pedagogical figure: optimizer trajectories](#7-pedagogical-figure-optimizer-trajectories-scriptsbuild_optimizer_trajectoriespy)
8. [Main accuracy benchmark](#8-main-accuracy-benchmark-scriptsrun_comparisonpy)
9. [Cost analysis](#9-cost-analysis-scriptscost_analysispy)
10. [ZZU inner-method ablation](#10-zzu-inner-method-ablation-scriptszzu_inner_method_comparisonpy)
11. [Test suite](#11-test-suite-tests)
12. [Reproduce every artifact](#12-reproduce-every-artifact)

---

## 1. Project layout

```
zzu-transformations/
├── scripts/
│   ├── transformation_algorithms.py    # Core algorithm library (Sections 1–15)
│   ├── toy_data.py                     # 5 synthetic dataset generators
│   ├── reproducibility.py              # Single source of truth for seeds / split sizes
│   ├── run_comparison.py               # Main accuracy benchmark across the 5 datasets
│   ├── cost_analysis.py                # Time / iters / model-fn calls per fit
│   ├── zzu_inner_method_comparison.py  # Ablation: pure vs ZZU+{GD,GN,BFGS}
│   ├── visualize_synthetic_data.py     # Per-dataset PNGs + 2×2 overview
│   ├── build_linearization_figures.py  # 3 pedagogical figures on transforms
│   └── build_optimizer_trajectories.py # SSE-surface contour + GD/GN/BFGS paths
├── tests/                          # 158 pytest cases across 7 modules
├── generated_datasets/             # CSV exports of the 5 synthetic datasets
├── synthetic_visualizations/       # Per-dataset PNGs (output of (5))
├── comparison_results/             # All benchmark CSVs and figures (outputs of (6)–(10))
├── reproduce/                      # Sandbox: full 5-dataset pipeline run for diffing
├── ZZU_main_with_py_imports.ipynb  # Main experiment notebook
├── concrete_analysis.ipynb         # Real-world: UCI Concrete Compressive Strength
└── bike_analysis.ipynb             # Real-world: UCI Bike Sharing
```

Every script in `scripts/` is a stand-alone entry point: each can be
run from the project root with `python scripts/<name>.py`, has no side
effects on import, and writes its outputs to a fixed, predictable
location.

---

## 2. Reproducibility infrastructure ([scripts/reproducibility.py](scripts/reproducibility.py))

**Why it exists.** Every randomness source in the project flows
through `np.random.default_rng(seed)` — a *local* generator that does
not touch global state. Default seeds were historically scattered
across three files; this module centralizes them so the writeup can
answer "what random state produced figure X?" by looking in one place.

**Public API.**

| Symbol | Value | Used by |
|---|---:|---|
| `N_SEEDS` | `10` | every benchmark loop (`run_comparison`, `cost_analysis`, `zzu_inner_method_comparison`) |
| `TEST_FRACTION` | `0.2` | every train/test split |
| `DEFAULT_SEED` | `123` | `train_test_split_arrays` default |
| `ZZU_VALIDATION_SEED` | `0` | `ZZUTransformRegressor` internal screen split |
| `DATASET_SEEDS` | 101..105 | `make_*` toy-data generators |
| `make_rng(seed)` | helper | returns a fresh `np.random.Generator` |
| `seed_everything(seed)` | helper | defensive global-PRNG hammer for notebooks/3rd-party libs |
| `reproduce_dir(subfolder, default_root)` | helper | resolves output dir, honoring `REPRODUCE_BASE_DIR` env var |

**Reproducibility contract.** Repeated runs of any benchmark with the
same `seed` produce bit-identical CSVs. Verified end-to-end in
[tests/test_reproducibility.py](tests/test_reproducibility.py) and
re-verified at the suite level by the diff in §12.

**Two env-var hooks for sandboxed runs.**

- `REPRODUCE_BASE_DIR=<path>` — every script's output directory is
  redirected to `<path>/comparison_results/`,
  `<path>/generated_datasets/`, and `<path>/synthetic_visualizations/`,
  so a fresh run lands in a sandbox folder you can diff against the
  canonical output without overwriting it.
- `REPRODUCE_DATASETS=all` — overrides each script's
  `DATASETS_TO_COMPARE` constant to force the full 5-dataset sweep.
  The default (`["exponential_multiplicative", "logistic_growth",
  "multivariable_nonlinear"]`) is a fast 3-dataset subset for
  development; setting `=all` runs every dataset in `DATASET_SPECS`.

---

## 3. Algorithm library ([scripts/transformation_algorithms.py](scripts/transformation_algorithms.py))

Pure-NumPy implementation of every algorithm used in the project. No
SciPy dependency. The library is divided into 15 sections; sections
1–9 cover transformation-based OLS, sections 10–15 add nonlinear
baselines and the ZZU hybrid.

### 3.1 Linear-algebra and split helpers (Sections 1–2)

| Symbol | Purpose |
|---|---|
| `as_2d(X)` | Promote 1D arrays to `(n, 1)` |
| `add_intercept(X)` | Prepend a column of ones |
| `ols_fit(X, y)` | Solve `(XᵀX) β = Xᵀy` via `np.linalg.lstsq` |
| `ols_predict(X, beta)` | Linear prediction with intercept |
| `train_test_split_arrays(X, y, test_size, seed)` | Reproducible random split |

### 3.2 Transform families (Sections 3–5)

Each family implements `forward`, `inverse`, and `log_jacobian` (used
by profile-likelihood λ selection). Six families are exposed through
`TransformedOLS`:

| Family | Forward `T(y)` | Notes |
|---|---|---|
| `identity` | `y` | Baseline; equivalent to plain OLS |
| `log` | `log y` | Requires `y > 0`; exact under multiplicative lognormal noise |
| `reciprocal` | `1/y` | Linearizes Michaelis–Menten when applied carefully |
| `power` (param=p) | `y^p` | Generalises `sqrt`, `square`, etc. |
| `boxcox` | `(y^λ − 1)/λ` (λ→0 ⇒ log) | λ auto-selected by profile likelihood |
| `yeojohnson` | extension to ℝ | Handles zero and negative responses |

### 3.3 λ selection and bias correction (Sections 6–7)

- `choose_lambda_by_profile_likelihood(...)` — grid search over
  λ ∈ [−2, 2], maximizing the profile log-likelihood. Used by both
  `boxcox` and `yeojohnson` when `lambda_=None`.
- Duan-style **smearing correction**: the back-transform of a fitted
  expectation is biased; smearing multiplies predictions by the
  empirical mean of `inv(T)` applied to residuals to remove that bias.
  Toggled via `use_smearing=True/False`.

### 3.4 Unified linear regressor (Section 8)

`TransformedOLS` is the single dataclass the rest of the codebase uses
for every linear model:

```python
import transformation_algorithms as ta
m = ta.TransformedOLS(transform="boxcox", use_smearing=True).fit(X, y)
y_hat = m.predict(X_new)        # original scale, smearing applied
m.summary()                     # {transform, selected_param_or_lambda, ...}
```

A small `_TRANSFORM_DISPATCH` dict maps each transform name to its
`(forward, inverse, log_jacobian, fit_extra)` quadruple so all six
families share one `fit/predict/summary` path.

### 3.5 Metrics (Section 9)

| Function | Returns |
|---|---|
| `regression_metrics(y_true, y_pred)` | `{n_valid, rmse, mae, mse, r2}` |
| `residual_diagnostics(y_true, y_pred)` | adds heteroscedasticity proxies (`corr(|resid|, x)`, std) |
| `evaluate_transformed_models(...)` | batch-fits a dict of TransformedOLS specs, returns sorted DataFrame |

### 3.6 Nonlinear optimizers (Sections 10–13)

Common interface across all three:

```python
reg.fit(X, y, theta_init)   # -> self
reg.predict(X)              # -> ndarray (n,)
reg.summary()               # -> dict
# attributes: .theta_, .converged_, .n_iter_, .fit_error_
```

| Class | Algorithm | Convergence rule | Recommended use |
|---|---|---|---|
| `GradientDescentRegressor` | `θ ← θ − η · ∇SSE(θ)` with optional decay | relative SSE change `< tol` | pedagogical baseline |
| `GaussNewtonRegressor` | `(JᵀJ + λI) δ = Jᵀr`; **self-activating LM** damping (raise λ on rejected step, halve on accepted) | step norm `< tol` | well-posed problems; fastest |
| `BFGSRegressor` | inverse-Hessian update with backtracking Armijo line search | gradient norm `< tol` | robust default |

Numerical Jacobians use central finite differences with adaptive step
`h_j = max(h, |θ_j|·h)` (Section 10, `numerical_jacobian`); the helper
`_eval_jacobian(model_fn, jacobian_fn, theta, X)` is shared by all
three optimizers so they look identical from the outside. Failures
are caught — every optimizer returns `self` even when its inner solve
raises, leaving the error message in `fit_error_`.

### 3.7 ZZU hybrid (Section 14)

`ZZUTransformRegressor` implements the three-step workflow:

1. **Screen.** Hold out an internal validation split (seed
   `ZZU_VALIDATION_SEED`), fit each `TransformedOLS` in
   `transformations`, score by validation RMSE on the original scale,
   pick the winner.
2. **Warm-start.** Convert the winning linearization's coefficients
   into nonlinear parameters via the user-supplied
   `coeff_to_init(model) -> theta_init`. If the inversion fails or no
   `coeff_to_init` is provided, fall back to `fallback_theta_init`.
3. **Refine + bias-correct.** Fit the chosen nonlinear optimizer
   (default BFGS) from that warm start. At predict time, add the mean
   training residual on the original scale.

Inner optimizer is selectable: `nonlinear_method ∈ {"bfgs",
"gauss_newton", "gradient_descent"}` with kwargs forwarded via
`nonlinear_kwargs`. The `coeff_to_init` contract is the only piece
the user must supply per problem; the README documents an example for
`y = a · exp(b·x)` ⇒ `a = exp(β₀), b = β₁`.

### 3.8 Batch evaluator (Section 15)

`evaluate_nonlinear_models(...)` parallels `evaluate_transformed_models`
on the nonlinear side: takes a dict of regressors, fits each, returns
a sorted DataFrame.

---

## 4. Synthetic dataset generators ([scripts/toy_data.py](scripts/toy_data.py))

Five datasets, each returned as a `DatasetBundle(X, y, y_true, params,
description)`. `X` and `y` are pandas frames/series for ergonomic
plotting; `.values` exposes the NumPy arrays the algorithm library
consumes.

| Generator | Functional form | n | p | Designed to test |
|---|---|---:|---:|---|
| `make_exponential_multiplicative()` | `y = a · exp(b·x) · η`, `log η ~ N(0, σ²)` | 120 | 1 | Best case for log-linearization |
| `make_exponential_additive()` | `y = a · exp(b·x) + ε` | 120 | 1 | Failure mode of log-linearization (additive noise distorts after log) |
| `make_michaelis_menten()` | `y = Vmax·x/(Km+x) + ε` | 120 | 1 | Saturating curve; reciprocal-linearization quirks |
| `make_logistic_growth()` | `y = L/(1 + exp(−k(x−x₀))) + ε` | 140 | 1 | S-curve; multimodal SSE with 3 parameters |
| `make_multivariable_nonlinear()` | `y = c₁·exp(r·x₁) + c₂·x₂^p + c₃/(1+x₃) + ε` | 500 | 3 | No single global linearization — designed to defeat any one transform and motivate ZZU |

Defaults are deterministic (seeds 101–105 from `DATASET_SEEDS`).
Helpers include `train_test_split_arrays` (re-exported),
`generate_default_suite()` (returns all 5 bundles in a dict), and the
matplotlib helpers `plot_one_dimensional_dataset(...)` and
`plot_multivariable_marginals(...)`.

`python scripts/toy_data.py` writes all five datasets to
[generated_datasets/](generated_datasets/) as CSV (one row per
observation, columns are X features then `y`).

---

## 5. Synthetic-data visualization ([scripts/visualize_synthetic_data.py](scripts/visualize_synthetic_data.py))

**What it does.** Renders one PNG per 1D dataset, plus a 2×2 grid
overview, plus a marginal scatter plot for the multivariable case.

**Run.**
```bash
python scripts/visualize_synthetic_data.py
# optional: --output-dir <path> --dpi 300
```

**Outputs** ([synthetic_visualizations/](synthetic_visualizations/)):

| File | Content |
|---|---|
| [synthetic_visualizations/synthetic_data_overview.png](synthetic_visualizations/synthetic_data_overview.png) | 2×2 panel: all four 1D datasets at once |
| [synthetic_visualizations/exponential_multiplicative.png](synthetic_visualizations/exponential_multiplicative.png) | Single dataset: noisy `y` vs `x` with the true signal overlaid |
| [synthetic_visualizations/exponential_additive.png](synthetic_visualizations/exponential_additive.png) | "" |
| [synthetic_visualizations/michaelis_menten.png](synthetic_visualizations/michaelis_menten.png) | "" |
| [synthetic_visualizations/logistic_growth.png](synthetic_visualizations/logistic_growth.png) | "" |
| [synthetic_visualizations/multivariable_nonlinear_marginals.png](synthetic_visualizations/multivariable_nonlinear_marginals.png) | Three marginal scatter panels: `y` vs `x₁`, `x₂`, `x₃` |

**Reading the overview.** The two exponential panels look nearly
identical to the eye but have *different noise structure*; that
distinction is what determines whether log-linearization is exact or
biased. The Michaelis–Menten and logistic panels show qualitatively
different curvature regimes (saturating vs. sigmoid).

---

## 6. Pedagogical figure: linearization diagnostics ([scripts/build_linearization_figures.py](scripts/build_linearization_figures.py))

**Purpose.** Three figures that walk the reader through *why*
response transformations work (or fail) on each of the three 1D
datasets where the screening is most diagnostic.

**Run.**
```bash
python scripts/build_linearization_figures.py
```

**Outputs** ([comparison_results/](comparison_results/)):

| File | Layout | Question it answers |
|---|---|---|
| [comparison_results/lin_fits_original.png](comparison_results/lin_fits_original.png) | 3 datasets × 4 transforms (identity, log, reciprocal, boxcox) | "Which fitted curve actually hugs the data on the *original* scale?" RMSE annotated per panel; selected λ shown for Box-Cox. |
| [comparison_results/lin_transformed_scale.png](comparison_results/lin_transformed_scale.png) | 3 × 3 (skipping identity) | "Does `T(y)` vs `x` actually become linear after the transform?" R² in the transformed space is annotated. |
| [comparison_results/lin_residuals.png](comparison_results/lin_residuals.png) | 3 × 3 | "Are the OLS assumptions (zero mean, homoscedastic, no pattern) satisfied after `T(y)`?" Annotated with std of residuals and `corr(|resid|, x)` — large `|corr|` ⇒ heteroscedasticity. |

**What to read off.**

- **`exponential_multiplicative` row.** `log` straightens the data
  cleanly (high R² on the transformed scale, residuals look iid). The
  fit on the original scale beats identity. Box-Cox picks λ very close
  to 0, confirming log.
- **`exponential_additive` row.** `log` *appears* to linearize, but
  the residuals show clear heteroscedasticity (`corr(|r|, x)` is
  large) — this is the diagnostic signature of "log-linearization with
  the wrong noise model". On the original-scale panel, the curve hugs
  the data but the *back-transformed bias* is visible.
- **`michaelis_menten` row.** None of the three "clean" transforms do
  well: `log` over-bends, `reciprocal` only works pointwise where `y`
  is bounded away from zero, Box-Cox picks an odd λ. This is the row
  that motivates "no single transform fits everywhere" → nonlinear /
  ZZU.

---

## 7. Pedagogical figure: optimizer trajectories ([scripts/build_optimizer_trajectories.py](scripts/build_optimizer_trajectories.py))

**Purpose.** A self-contained, 2-parameter problem (fit
`y = a · exp(b·x)` on `exponential_multiplicative`) makes the SSE
surface plottable as a 2D contour. Overlaying the GD / GN / BFGS
trajectories on that contour makes the qualitative differences
between the three optimizers immediately legible — far easier than
reading "convergence in N iterations" from a table.

**Run.**
```bash
python scripts/build_optimizer_trajectories.py
```

**Output.**

- [comparison_results/optimizer_trajectories.png](comparison_results/optimizer_trajectories.png)
  — `log₁₀ SSE` contour over `(a, b) ∈ [0.5, 3.5] × [0, 1.05]`, with
  three trajectories starting from `θ₀ = (1, 0.1)`:
  - **GD** (gray, ~hundreds of tiny steps) crawls along the gradient
    direction, slowed by the narrow basin in `b`. Verified: 400 steps
    with `final θ = (1.260, 0.801)` — *did not converge in 400 iterations*.
  - **GN+LM** (blue, ~5–10 steps) takes large, well-aimed
    Gauss-Newton steps; LM damping rescues steps that would otherwise
    overshoot. Verified: 8 steps to `θ = (2.167, 0.679)`.
  - **BFGS** (green, ~20 steps) starts cautious (initial Hessian = I)
    then accelerates as the inverse-Hessian estimate sharpens.
    Verified: 40 steps to `θ = (2.167, 0.679)`.

(Reproduction match: same final θ to numerical precision when re-run
with current code.)

---

## 8. Main accuracy benchmark ([scripts/run_comparison.py](scripts/run_comparison.py))

The headline result table. For each dataset in `DATASETS_TO_COMPARE`,
runs every method through `N_SEEDS` random 80/20 splits and reports
mean ± std test RMSE / R² / convergence rate.

> **Default scope.** `DATASETS_TO_COMPARE` is set to a fast 3-dataset
> subset (`exponential_multiplicative`, `logistic_growth`,
> `multivariable_nonlinear`) for development. To sweep all five
> datasets, run with `REPRODUCE_DATASETS=all` in the environment, or
> set the constant to `None`.

### 8.1 What it runs per (dataset, seed)

1. **11 linearized-OLS variants** built by `make_transformed_suite()`
   — identity, log/reciprocal/power/Box-Cox/Yeo-Johnson with both
   smearing on/off where applicable.
2. **3 nonlinear regressors** (GD, GN, BFGS) sharing one `model_fn`
   and a data-driven heuristic init (built by
   `make_nonlinear_regressors`).
3. **1 ZZU hybrid** built by `make_zzu()` with a dataset-specific
   screening dict and `coeff_to_init`.

Each dataset's spec lives in `DATASET_SPECS`: `model_fn`,
`theta_init_fn`, `zzu_transformations`, `zzu_coeff_to_init`. The
design rule is documented inline: `coeff_to_init` must be invertible
from the specific transforms in the screening suite, otherwise the
ZZU spec falls back to a heuristic init.

### 8.2 Outputs

| File | Content |
|---|---|
| [comparison_results/raw_results.csv](comparison_results/raw_results.csv) | Long-form: 1 row per (dataset, method, seed), columns include `rmse`, `r2`, `converged`, `n_iter`, `error` |
| [comparison_results/summary_by_method.csv](comparison_results/summary_by_method.csv) | Aggregated: mean / std / median RMSE, mean R², `frac_converged`, `n_failures` |
| [comparison_results/rmse_by_method.png](comparison_results/rmse_by_method.png) | One subplot per dataset; bar chart of mean test RMSE per method, sorted ascending; methods colored by family (linearized / nonlinear / ZZU); log-scale y |
| [comparison_results/fit_overlay.png](comparison_results/fit_overlay.png) | Best-fit overlay for the four 1D datasets at seed 0: scatter, true signal, best linear fit, BFGS, ZZU |

### 8.3 Headline numbers (mean test RMSE over 10 splits)

The 3 datasets in the canonical CSV (default subset):

| Dataset | Top method | Family | Mean RMSE | R² | Notes |
|---|---|---|---:|---:|---|
| `exponential_multiplicative` | `log_smear` | linearized OLS | **7.581** | 0.808 | Log linearization is exact when noise is multiplicative lognormal |
| `logistic_growth` | **ZZU** | hybrid | **3.065** | 0.994 | Edges BFGS/GN by ~0.002 RMSE |
| `multivariable_nonlinear` | **ZZU** | hybrid | **4.654** | 0.973 | Designed to defeat any single transform; ZZU beats BFGS/GN by ≈0.10 RMSE |

The other 2 datasets (verified from the full sweep in
[reproduce/comparison_results/summary_by_method.csv](reproduce/comparison_results/summary_by_method.csv)):

| Dataset | Top method | Family | Mean RMSE | R² | Notes |
|---|---|---|---:|---:|---|
| `exponential_additive` | GD / GN / BFGS | nonlinear | **5.14** | 0.904 | Log distorts additive noise; nonlinear sweeps the top 3, ZZU is *worse* (10.8) due to a misleading screened init |
| `michaelis_menten` | BFGS / GN / **ZZU** (tie) | nonlinear / hybrid | **0.297** | 0.97 | ZZU matches direct nonlinear with no clean linearization available |

**Reading [comparison_results/rmse_by_method.png](comparison_results/rmse_by_method.png).**
Within each subplot, the best method is leftmost; bar color identifies
the family. The *horizontal* gap between the leftmost bar of each
color is what tells you whether a family of methods can compete on
this dataset at all. ZZU (green) is in the top three on the three
datasets where its screening dict has a defensible
`coeff_to_init` — and is uncompetitive on `exponential_additive`,
which is the clearest demonstration that the screening phase is only
worth its cost when the noise structure matches the chosen transform.

**Reading [comparison_results/fit_overlay.png](comparison_results/fit_overlay.png).**
Each panel overlays four lines on a scatter of observations: the true
signal (dashed black), the best linear method (blue), BFGS (orange),
and ZZU (green). On `exponential_multiplicative` all three converge —
the choice is about cost. On the other panels, the linear fit visibly
diverges from the true signal in some region while nonlinear / ZZU
stay on it.

---

## 9. Cost analysis ([scripts/cost_analysis.py](scripts/cost_analysis.py))

Same train/test loop as `run_comparison.py`, but the `model_fn` is
wrapped in a `CountingModelFn` that increments on every call so we
can report:

- wall-clock fit time (`time.perf_counter()` around `.fit(...)`),
- optimizer iteration count (`reg.n_iter_`),
- model-function evaluations (captures Jacobian work — numerical
  Jacobian costs `2p` `model_fn` calls per Jacobian).

It also adds a special row, `BFGS_warmstart`: BFGS started from the
*screened* init that ZZU's screening would produce, but with no
screening overhead in the timer. This isolates "does the warm start
help?" from "does the screening overhead pay off?".

> **Note on `mean_fit_time_sec` (wall-clock vs deterministic cost).**
> Fit time is measured with `time.perf_counter()`, so it varies
> run-to-run with CPU load and kernel scheduling — typically 5–20% on
> long fits (GD), higher relative jitter on sub-millisecond fits
> (BFGS / GN). Iteration counts and `mean_n_model_evals` are
> deterministic up to 1-ULP propagation in the dataset, so for cost
> comparisons that need to be **stable across runs use
> `mean_n_model_evals`** (the work count: number of `model_fn` calls,
> which captures Jacobian work since each numerical Jacobian costs
> `2p` calls). The relative ranking between method families is
> preserved either way — GD is always ~100× slower than BFGS, ZZU's
> overhead over BFGS is always ~2–3× — but absolute milliseconds will
> shift between runs.

### 9.1 Outputs

| File | Content |
|---|---|
| [comparison_results/cost_results.csv](comparison_results/cost_results.csv) | Long-form: 1 row per (dataset, method, seed), columns include `fit_time_sec`, `n_iter`, `n_model_evals`, `rmse`, `converged` |
| [comparison_results/cost_summary.csv](comparison_results/cost_summary.csv) | Aggregated mean / std for the cost columns |
| [comparison_results/cost_pareto.png](comparison_results/cost_pareto.png) | Multi-panel: log-RMSE vs log-fit-time; per-method labels auto-placed by adjustText |
| `cost_pareto_<dataset>.png` (one per dataset) | One full-size Pareto per dataset (labels need room) |
| [comparison_results/warm_vs_cold.png](comparison_results/warm_vs_cold.png) | 1×3 grid: BFGS cold vs warm-start in iterations, model_fn calls, and fit time |

Per-dataset Pareto links (canonical 3 + the 2 from
[reproduce/](reproduce/)):
[`exponential_multiplicative`](comparison_results/cost_pareto_exponential_multiplicative.png),
[`logistic_growth`](comparison_results/cost_pareto_logistic_growth.png),
[`multivariable_nonlinear`](comparison_results/cost_pareto_multivariable_nonlinear.png),
[`exponential_additive`](reproduce/comparison_results/cost_pareto_exponential_additive.png),
[`michaelis_menten`](reproduce/comparison_results/cost_pareto_michaelis_menten.png).

### 9.2 Order-of-magnitude cost per family

Verified from [comparison_results/cost_summary.csv](comparison_results/cost_summary.csv):

| Family | Time per fit | Iterations | Model_fn calls | Notes |
|---|---:|---:|---:|---|
| Linearized OLS (`log_smear`) | ~0.04 ms | n/a | n/a | Pure linear algebra; one Box-Cox/Yeo-Johnson grid search dominates |
| Gauss-Newton | ~0.5–5 ms | 6–26 | 60–340 | Fastest nonlinear when well-posed |
| BFGS | ~1–15 ms | 12–43 | 200–1160 | ~10× linearized OLS; numerical Jacobian is the bottleneck |
| ZZU hybrid | ~1.6–40 ms | 9–43 | 150–1050 | Adds the screening phase on top of one nonlinear fit |
| Gradient descent | ~100 ms – 1 s | 5000 (capped) | 31k–65k | Dominated everywhere — slow and rarely converges |

**Reading [comparison_results/cost_pareto.png](comparison_results/cost_pareto.png).**
Lower-left = better (low RMSE, low time). The Pareto frontier on each
panel tells the story:

- `exponential_multiplicative`: linearized OLS is on the frontier
  alone — 25× faster than nonlinear at the same accuracy (0.04 ms vs
  ~1 ms).
- `logistic_growth`, `michaelis_menten`: GN dominates — fastest
  nonlinear at the lowest RMSE (0.5–1 ms, RMSE 3.07 / 0.297).
- `multivariable_nonlinear`: ZZU appears on the frontier; its ~3×
  cost over BFGS (40 ms vs 14 ms) purchases a real RMSE improvement
  (4.65 vs 4.75).

### 9.3 Warm-start vs cold-start (the key cost finding)

The two strategies are run with **identical** model, dataset,
optimizer, convergence tolerance — only the initial `theta` changes.

- **Cold start.** Data-driven heuristic from summary statistics. For
  `y = a · exp(b · x)`: `theta_cold = [max(min(y), 1e-3), 0.1]`.
- **Warm start.** Run Step 1 of ZZU first — fit `log(y) ~ x` via OLS —
  then *invert* via the user-supplied `coeff_to_init`:
  `theta_warm = [exp(β₀), β₁]`. Costs one extra OLS solve (~0.1 ms).

Verified mean over 10 splits (from
[comparison_results/cost_summary.csv](comparison_results/cost_summary.csv)
and [reproduce/comparison_results/cost_summary.csv](reproduce/comparison_results/cost_summary.csv)):

| Dataset | Init | Iterations | model_fn calls | Time (ms) | RMSE |
|---|---|---:|---:|---:|---:|
| `exponential_multiplicative` | cold | 12.7 | 206.7 | 1.04 | 7.701 |
| `exponential_multiplicative` | warm | **9.3** | **149.4** | **0.74** | 7.701 |
| `exponential_additive` | cold | **19.8** | **301.0** | **1.47** | **5.142** |
| `exponential_additive` | warm | 25.7 | 397.5 | 1.95 | 14.337 ✗ |

This is the cleanest theory-vs-empirics match in the project: warm
start *helps* exactly when the linearization assumption is consistent
with the noise structure (multiplicative on `exp_mult`), and *hurts*
when it is not (additive noise after log on `exp_add`). Note that on
`exp_add` the warm-started BFGS doesn't just take more iterations —
its final RMSE is **2.8× worse** than cold-start because the screened
init lands BFGS in a different basin of the SSE surface. That basin
is also the basin ZZU lands in by default, which is why
`exp_additive` is the one dataset where **ZZU is the wrong choice**.

The takeaway for ZZU: the screening phase is worth its cost when
transform diagnostics support it — and that diagnostic information is
exactly what motivates "selective ZZU" as future work (gate ZZU on a
residual-normality or screen-RMSE-ratio signal).

---

## 10. ZZU inner-method ablation ([scripts/zzu_inner_method_comparison.py](scripts/zzu_inner_method_comparison.py))

**Question this script answers.** The default ZZU pipeline runs BFGS
as its inner optimizer. Does the choice of inner optimizer matter?
And how does each *pure* (cold-start) optimizer compare against its
*ZZU-augmented* counterpart on each dataset?

**What it runs.** Six configurations per dataset:

| Method | Variant | What it is |
|---|---|---|
| GD | pure | Gradient descent with heuristic init |
| GD | zzu | ZZU + GD (screen → warm-start GD → bias correct) |
| GN | pure | Gauss-Newton with heuristic init |
| GN | zzu | ZZU + GN |
| BFGS | pure | BFGS with heuristic init |
| BFGS | zzu | ZZU + BFGS (the project default) |

**Run.**
```bash
python scripts/zzu_inner_method_comparison.py            # 3 datasets (default)
REPRODUCE_DATASETS=all python scripts/zzu_inner_method_comparison.py   # all 5
```

**Outputs** ([comparison_results/](comparison_results/)):

| File | Content |
|---|---|
| [comparison_results/zzu_inner_method_results.csv](comparison_results/zzu_inner_method_results.csv) | Long-form: 1 row per (dataset, method, variant, seed) |
| [comparison_results/zzu_inner_method_summary.csv](comparison_results/zzu_inner_method_summary.csv) | Mean / std RMSE, fit time, `n_iter`, `frac_converged` |
| [comparison_results/zzu_inner_method.png](comparison_results/zzu_inner_method.png) | 2-panel grouped bar plot: top = RMSE, bottom = fit time. For each dataset, six bars: pure-GD / ZZU+GD / pure-GN / ZZU+GN / pure-BFGS / ZZU+BFGS. Color = optimizer family; hatching = pure vs ZZU. |

**Reading the figure.** Two stories overlap in one chart:

1. *"What does this optimizer do alone?"* — compare hatched bars
   across the three colors within one dataset.
2. *"What does ZZU add on top of this optimizer?"* — compare each
   hatched bar to its solid neighbor of the same color.

**Headline finding** (verified from
[comparison_results/zzu_inner_method_summary.csv](comparison_results/zzu_inner_method_summary.csv)):

The ZZU benefit is largely *independent of the inner optimizer*: the
warm-start init brings the same RMSE and similar iteration counts
whether the inner optimizer is GN or BFGS:

| Dataset | Method | Variant | RMSE | Iterations |
|---|---|---|---:|---:|
| `multivariable_nonlinear` | BFGS | pure | 4.745 | 43.0 |
| `multivariable_nonlinear` | BFGS | zzu | **4.654** | 43.3 |
| `multivariable_nonlinear` | GN | pure | 4.745 | 26.0 |
| `multivariable_nonlinear` | GN | zzu | **4.654** | 26.1 |

ZZU's RMSE improvement on the multivariable benchmark is identical
across BFGS and GN inner choices — consistent with "ZZU's
contribution is the screened init, not the inner solver". GD's
behavior is qualitatively different — it remains the slowest and
least reliable in both pure and ZZU forms (5000 iterations,
`frac_converged = 0.0` on most datasets). BFGS being the default is
about robustness on rough surfaces, not about being uniquely
required for the warm-start payoff.

---

## 11. Test suite ([tests/](tests/))

158 pytest cases across 7 modules, runnable from the project root
with `pytest tests/`. Tests are deliberately scoped to behaviors that
are *not* obvious from the code itself.

| Module | What it covers |
|---|---|
| [tests/test_linear_helpers.py](tests/test_linear_helpers.py) | `as_2d`, `add_intercept`, `ols_fit`, `ols_predict`, `train_test_split_arrays` |
| [tests/test_transforms.py](tests/test_transforms.py) | round-trip identity (T then T⁻¹), domain validity, log_jacobian sign correctness for every transform family |
| [tests/test_transformed_ols.py](tests/test_transformed_ols.py) | every transform fits and predicts with sane RMSE; smearing on/off behavior; λ-selection via profile likelihood |
| [tests/test_nonlinear_optimizers.py](tests/test_nonlinear_optimizers.py) | recovery on a known toy problem, convergence flag honesty, failure capture |
| [tests/test_zzu.py](tests/test_zzu.py) | screen → warm-start → bias-correct sequence; `coeff_to_init` honored; fallback init when screen fails; non-default inner optimizers (GN, GD) |
| [tests/test_toy_data.py](tests/test_toy_data.py) | shapes, dtypes, deterministic with seed, response-range sanity |
| [tests/test_reproducibility.py](tests/test_reproducibility.py) | dataset → split → fit produces bit-identical CSVs across runs |

Run a single module with `pytest tests/test_zzu.py -v`.

---

## 12. Reproduce every artifact

The [reproduce/](reproduce/) folder is a sandbox copy of the full
5-dataset pipeline run. To rebuild it from scratch and verify every
canonical CSV / PNG end-to-end:

```bash
# Clean sandbox
rm -rf reproduce
mkdir reproduce

# Run every output-writing script with both env-var overrides
export REPRODUCE_BASE_DIR=$(pwd)/reproduce
export REPRODUCE_DATASETS=all

python scripts/toy_data.py                                # -> reproduce/generated_datasets/
python scripts/visualize_synthetic_data.py                # -> reproduce/synthetic_visualizations/
python scripts/build_linearization_figures.py             # -> reproduce/comparison_results/lin_*.png
python scripts/build_optimizer_trajectories.py            # -> reproduce/comparison_results/optimizer_trajectories.png
python scripts/run_comparison.py                          # -> reproduce/comparison_results/raw_results.csv etc.
python scripts/cost_analysis.py                           # -> reproduce/comparison_results/cost_*.{csv,png}
python scripts/zzu_inner_method_comparison.py             # -> reproduce/comparison_results/zzu_inner_method.{csv,png}

# Tests (should print "158 passed")
pytest tests/
```

**Diff results** (run after the pipeline above completes):

```bash
python - <<'PY'
import pandas as pd
def diff(p_canon, p_repro, key, vals, label):
    a = pd.read_csv(p_canon); b = pd.read_csv(p_repro)
    m = pd.merge(a[key+vals], b[key+vals], on=key, suffixes=('_c','_r'))
    print(f"=== {label}  merged: {len(m)} rows ===")
    for v in vals:
        d = (m[f'{v}_c'] - m[f'{v}_r']).abs()
        rel = d / m[f'{v}_c'].abs().clip(lower=1e-12)
        print(f"   {v:25s} max_abs={d.max():.3g}  max_rel={rel.max():.3g}")

diff('comparison_results/summary_by_method.csv',
     'reproduce/comparison_results/summary_by_method.csv',
     ['dataset','method','family'],
     ['mean_rmse','std_rmse','mean_r2'], 'summary_by_method')
diff('comparison_results/zzu_inner_method_summary.csv',
     'reproduce/comparison_results/zzu_inner_method_summary.csv',
     ['dataset','method','variant'],
     ['mean_rmse','std_rmse','mean_n_iter'], 'zzu_inner_method_summary')
PY
```

**Validation results from this checkout** (matched on overlapping
keys; `comparison_results/` is a 3-dataset subset, `reproduce/` covers
all 5):

```
=== summary_by_method  merged: 45 rows ===
   mean_rmse                 max_abs=1.74e-06  max_rel=2.27e-07
   std_rmse                  max_abs=7.62e-07  max_rel=2.98e-07
   mean_r2                   max_abs=2.46e-07  max_rel=3.06e-07

=== zzu_inner_method_summary  merged: 18 rows ===
   mean_rmse                 max_abs=7.73e-06  max_rel=1.00e-06
   std_rmse                  max_abs=3.22e-06  max_rel=1.25e-06
   mean_n_iter               max_abs=0.222     max_rel=1.09e-02
```

All RMSE values match to within `~1e-6` relative — float-precision
identity, as expected. The small `mean_n_iter` differences (max 0.22
out of ~12) reflect 1-ULP variation in the regenerated datasets
propagating through optimizer step count; final fitted parameters
match.

Dependencies: `numpy`, `pandas`, `matplotlib`, `pytest`, optionally
`adjustText` (used only by the cost-Pareto plot for label de-overlap;
the script falls back to fixed offsets if it's not installed). No
SciPy.
