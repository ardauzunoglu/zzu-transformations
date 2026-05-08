"""End-to-end reproducibility regression tests.

Why these tests matter
----------------------
The project's writeup makes claims like "mean test RMSE over 10 splits,
seeds 0-9, was 4.65 on multivariable_nonlinear."  Anyone who clones the
repo and runs the benchmark must get the same number.  These tests pin
that contract: the same seed must produce a bit-identical result, end
to end, through every layer of the pipeline.

If a future refactor accidentally introduces a non-deterministic
dependency (a `np.random.normal()` call without a passed-in RNG, a
`set()` iteration order leaking into a sort, etc.), one of these tests
will fail and pinpoint where.

These tests run twice through the same random-state-touching paths and
assert bit-equality.  They are fast (no full benchmark loops) — they
exercise just enough of the pipeline to catch a regression.
"""

from __future__ import annotations

import numpy as np
import pytest

import reproducibility as repro
import toy_data as td
import transformation_algorithms as ta


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestReproducibilityConstants:
    """The registry must agree with the per-module defaults — otherwise
    consumers get conflicting "official" values for N_SEEDS etc."""

    def test_dataset_seeds_match_toy_data_defaults(self):
        # The DATASET_SEEDS dict claims to be synchronized with the
        # default `seed=` values inside each generator.  Drift here
        # would silently mean the benchmark CSVs disagree with the
        # numbers cited in the writeup.
        for name, expected in repro.DATASET_SEEDS.items():
            gen = td.DATASET_GENERATORS[name]
            # Build the bundle and pull the seed it actually used.
            bundle = gen()
            assert bundle.params["seed"] == expected, (
                f"{name}: registry says seed={expected} "
                f"but generator default is {bundle.params['seed']}"
            )

    def test_n_seeds_is_positive(self):
        assert repro.N_SEEDS >= 1

    def test_test_fraction_is_in_unit_interval(self):
        assert 0 < repro.TEST_FRACTION < 1


# ---------------------------------------------------------------------------
# Bit-identical reproduction through the whole stack
# ---------------------------------------------------------------------------

class TestEndToEndReproducibility:
    """For each component that consumes a seed, verify that running it
    twice with the same seed yields bit-identical output.  Together
    these assertions cover the full pipeline used by run_comparison.py."""

    def test_dataset_generation_is_bit_identical(self):
        # Step 1 of the pipeline: dataset generation.  toy_data
        # generators must produce identical X, y, y_true under the
        # same seed.
        b1 = td.make_multivariable_nonlinear(seed=42)
        b2 = td.make_multivariable_nonlinear(seed=42)
        assert np.array_equal(b1.X.values, b2.X.values)
        assert np.array_equal(b1.y.values, b2.y.values)
        assert np.array_equal(b1.y_true.values, b2.y_true.values)

    def test_train_test_split_is_bit_identical(self):
        # Step 2: ta.train_test_split_arrays — the workhorse used by
        # every benchmark loop.  Same input + same seed must yield
        # the same four arrays.
        n, p = 50, 3
        X = np.arange(n * p, dtype=float).reshape(n, p)
        y = np.arange(n, dtype=float)
        a = ta.train_test_split_arrays(X, y, test_size=0.2, seed=7)
        b = ta.train_test_split_arrays(X, y, test_size=0.2, seed=7)
        for arr_a, arr_b in zip(a, b):
            assert np.array_equal(arr_a, arr_b)

    def test_transformed_ols_fit_is_deterministic(self):
        # Step 3a: TransformedOLS contains no randomness, but Box-Cox's
        # profile-likelihood grid search could theoretically return
        # different λ values if some implementation detail leaked
        # randomness.  Pin that down.
        bundle = td.make_exponential_multiplicative(seed=42)
        m1 = ta.TransformedOLS(transform="boxcox").fit(bundle.X.values, bundle.y.values)
        m2 = ta.TransformedOLS(transform="boxcox").fit(bundle.X.values, bundle.y.values)
        assert np.array_equal(m1.beta_, m2.beta_)
        assert m1.selected_param_ == m2.selected_param_

    def test_zzu_with_fixed_seed_is_bit_identical(self):
        # Step 3b: ZZU's screening uses an internal validation split
        # controlled by self.seed.  Two ZZU runs with the same seed
        # must select the same transform and reach the same final theta.
        bundle = td.make_exponential_multiplicative(seed=42)
        X, y = bundle.X.values, bundle.y.values

        model_fn = lambda X, t: t[0] * np.exp(t[1] * X[:, 0])
        coeff_to_init = lambda m: np.array([float(np.exp(m.beta_[0])), float(m.beta_[1])])

        def fit_one():
            return ta.ZZUTransformRegressor(
                model_fn=model_fn,
                coeff_to_init=coeff_to_init,
                nonlinear_method="bfgs",
                transformations={"log_smear": ta.TransformedOLS(transform="log",
                                                                 use_smearing=True)},
                seed=repro.ZZU_VALIDATION_SEED,
            ).fit(X, y)

        z1 = fit_one()
        z2 = fit_one()
        assert z1.best_transform_name_ == z2.best_transform_name_
        assert np.array_equal(z1.theta_init_used_, z2.theta_init_used_)
        assert np.array_equal(z1.nonlinear_regressor_.theta_,
                              z2.nonlinear_regressor_.theta_)


# ---------------------------------------------------------------------------
# seed_everything helper
# ---------------------------------------------------------------------------

class TestSeedEverything:
    """seed_everything is a defensive helper for code paths that may
    reach into ``np.random.*`` global state.  Tests below verify that
    its side effects are observable and that it returns a usable RNG."""

    def test_returns_numpy_generator(self):
        rng = repro.seed_everything(0)
        assert isinstance(rng, np.random.Generator)

    def test_same_seed_returns_rng_with_identical_first_draw(self):
        # Two calls with the same seed should yield Generators that
        # produce the same sequence — a basic determinism check.
        a = repro.seed_everything(42).integers(0, 1_000_000, size=5)
        b = repro.seed_everything(42).integers(0, 1_000_000, size=5)
        assert np.array_equal(a, b)

    def test_seeds_global_numpy_state(self):
        # The defensive contract: np.random.* (legacy global state)
        # should also be deterministic after seed_everything.  This is
        # the failure mode the helper exists to defend against.
        repro.seed_everything(7)
        a = np.random.rand(4)
        repro.seed_everything(7)
        b = np.random.rand(4)
        assert np.array_equal(a, b)
