"""Tests for the synthetic dataset generators in toy_data.py.

Why these tests matter
----------------------
The five toy datasets are the *foundation* of every benchmark in the
project.  If a generator silently changes its shape, range, or
reproducibility behavior, every RMSE / cost number cited in the writeup
becomes incomparable — and there is no way to spot the regression by
eyeballing the result CSVs.  These tests guard the invariants that the
rest of the codebase assumes:

  - the (n, p) shapes match what the docstrings advertise;
  - y > 0 holds where claimed (downstream log-linearization breaks otherwise);
  - the params dict carries the metadata needed to reproduce a run;
  - identical seeds yield bit-identical samples (paper claims about
    "10 random splits" depend on this).
"""

from __future__ import annotations

import numpy as np
import pytest

import toy_data as td


# ---------------------------------------------------------------------------
# Generic contracts — apply to every generator in the suite
# ---------------------------------------------------------------------------

GENERATORS = [
    ("exp_mult", td.make_exponential_multiplicative, 1),
    ("exp_add",  td.make_exponential_additive, 1),
    ("mm",       td.make_michaelis_menten, 1),
    ("logistic", td.make_logistic_growth, 1),
    ("multivar", td.make_multivariable_nonlinear, 3),
]


@pytest.mark.parametrize("name, gen, expected_p", GENERATORS)
class TestGenericContracts:
    """Each test runs once per generator via parametrize."""

    def test_returns_dataset_bundle(self, name, gen, expected_p):
        # Every downstream consumer expects a DatasetBundle (not a tuple
        # or dict).  A type drift would break run_comparison.py et al.
        assert isinstance(gen(), td.DatasetBundle)

    def test_X_dimensionality_matches_docstring(self, name, gen, expected_p):
        # Catches accidental adding/dropping of predictor columns
        # (e.g. someone refactoring multivariable to also return x4).
        bundle = gen()
        assert bundle.X.shape[1] == expected_p

    def test_y_and_y_true_aligned_to_X(self, name, gen, expected_p):
        # n must be consistent across X / y / y_true; otherwise downstream
        # train_test_split_arrays would index into mismatched arrays and
        # produce silently wrong benchmarks.
        bundle = gen()
        assert len(bundle.y) == bundle.X.shape[0]
        assert len(bundle.y_true) == bundle.X.shape[0]

    def test_params_records_seed(self, name, gen, expected_p):
        # The seed is the reproducibility metadata that lets a reader of
        # the writeup regenerate the data we cite.  A missing or None
        # seed would make claims like "Table 3 used seed 101" unverifiable.
        bundle = gen()
        assert "seed" in bundle.params
        assert bundle.params["seed"] is not None

    def test_same_seed_gives_bit_identical_output(self, name, gen, expected_p):
        # Bit-identical reproducibility under fixed seed is what makes
        # the "mean over 10 splits, seeds 0..9" methodology verifiable.
        # This guards against accidental use of np.random.* (which uses
        # global state) instead of np.random.default_rng (which doesn't).
        b1 = gen()
        b2 = gen()
        assert np.array_equal(b1.X.values, b2.X.values)
        assert np.array_equal(b1.y.values, b2.y.values)

    def test_different_seed_changes_output(self, name, gen, expected_p):
        # Sanity: the seed parameter actually controls randomness.  If
        # different seeds produced identical data, the benchmark "10
        # splits" would secretly be one split repeated 10×.
        b1 = gen(seed=1)
        b2 = gen(seed=2)
        assert not np.array_equal(b1.y.values, b2.y.values)


# ---------------------------------------------------------------------------
# Per-generator invariants documented in the toy_data.py docstrings
# ---------------------------------------------------------------------------

class TestExpMultInvariants:
    """y = a·exp(b·x)·η,  log η ~ N — multiplicative noise can't flip sign."""

    def test_y_strictly_positive(self):
        # The downstream log_smear screening relies on y > 0; this is
        # the *only* generator where the invariant holds without clipping.
        # A regression here (e.g. someone allowing negative log_mean to
        # break things) would crash the log path silently.
        assert (td.make_exponential_multiplicative().y.values > 0).all()

    def test_noise_type_recorded(self):
        # The noise_type label is consumed by analysis scripts to decide
        # whether multiplicative or additive correction is appropriate.
        bundle = td.make_exponential_multiplicative()
        assert bundle.params["noise_type"] == "multiplicative_lognormal"


class TestExpAddInvariants:
    """y = a·exp(b·x) + ε with clipping to min_y."""

    def test_y_at_or_above_min_y(self):
        # Additive Gaussian noise CAN drive y negative (this is the whole
        # point of this dataset's "failure mode" framing).  The generator
        # clips to min_y so log-based methods can still run.  Without
        # this guard, log_smear would crash on this dataset and the
        # "exp_add favors nonlinear" comparison couldn't be made.
        bundle = td.make_exponential_additive()
        assert (bundle.y.values >= bundle.params["min_y"]).all()

    def test_n_clipped_recorded(self):
        # The clipping count is part of reproducibility metadata — if
        # you change sigma, the number of clipped samples changes, and
        # that should be visible without rerunning the generator.
        assert "n_clipped" in td.make_exponential_additive().params


class TestMichaelisMentenInvariants:
    """Saturating curve: y_true monotone increasing, bounded by Vmax."""

    def test_y_true_monotone_in_x(self):
        # MM is documented as a saturating monotone curve.  A swapped
        # vmax/km assignment would break monotonicity silently.
        bundle = td.make_michaelis_menten()
        order = np.argsort(bundle.X.iloc[:, 0].values)
        y_t = bundle.y_true.values[order]
        assert (np.diff(y_t) >= 0).all()

    def test_y_true_bounded_by_vmax(self):
        # MM has a horizontal asymptote at Vmax (y_true → Vmax as x → ∞);
        # for finite x, y_true < Vmax always.  Catches sign or formula bugs.
        bundle = td.make_michaelis_menten()
        assert bundle.y_true.values.max() < bundle.params["vmax"]


class TestLogisticGrowthInvariants:
    """S-curve: y_true(x = x0) = L / 2 by construction."""

    def test_midpoint_value_is_L_over_two(self):
        # The inflection point of the logistic is where y = L/2.  This
        # encodes the parameterization (k, x0) → (steepness, midpoint)
        # explicitly; a swapped k/x0 would fail this test.  Tolerance is
        # 5% to absorb the discretization offset at the sample nearest x0
        # (linspace step ≈ 0.07 with k=1.2 shifts the value by ~2%).
        bundle = td.make_logistic_growth()
        x0 = bundle.params["x0"]
        L = bundle.params["L"]
        idx = int(np.argmin(np.abs(bundle.X.iloc[:, 0].values - x0)))
        assert bundle.y_true.values[idx] == pytest.approx(L / 2, rel=0.05)


class TestMultivariableInvariants:
    """3-predictor benchmark designed to defeat any single linearization."""

    def test_predictor_columns_named_x1_x2_x3(self):
        # run_comparison.py and friends index columns by name.  A silent
        # rename to X1/Y1/Z1 would break the multivariable benchmark.
        bundle = td.make_multivariable_nonlinear()
        assert list(bundle.X.columns) == ["x1", "x2", "x3"]

    def test_predictor_ranges_respected(self):
        # The generator advertises explicit ranges for each predictor.
        # If sampling drifted outside those ranges, downstream heuristic
        # theta_init values (which assume e.g. x2 ∈ [1, 10]) would
        # silently produce bad warm starts.
        bundle = td.make_multivariable_nonlinear()
        for col, key in [("x1", "x1_range"),
                         ("x2", "x2_range"),
                         ("x3", "x3_range")]:
            lo, hi = bundle.params[key]
            assert bundle.X[col].min() >= lo
            assert bundle.X[col].max() <= hi


# ---------------------------------------------------------------------------
# DatasetBundle helper methods (used by exporters and notebooks)
# ---------------------------------------------------------------------------

class TestDatasetBundleHelpers:
    def test_to_frame_combines_columns(self):
        # to_frame is the path used by export_suite_to_csv.  If it dropped
        # y or y_true, every CSV in generated_datasets/ would be useless.
        df = td.make_exponential_multiplicative().to_frame(include_true=True)
        assert "y" in df.columns and "y_true" in df.columns

    def test_summary_returns_documented_keys(self):
        # The summary fields are the "first table" in toy_data.ipynb;
        # they are part of the public contract for that notebook.
        s = td.make_exponential_multiplicative().summary()
        for k in ("name", "n", "p", "y_mean", "y_std",
                  "y_min", "y_max", "noise_type", "seed"):
            assert k in s
