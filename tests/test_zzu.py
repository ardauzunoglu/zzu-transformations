"""Tests for the ZZUTransformRegressor end-to-end workflow
(Section 14 of transformation_algorithms.py).

Covers:
  - 3-step pipeline runs to a low RMSE on log-linear-friendly data
  - screening_table_ is populated and ranked by val RMSE
  - coeff_to_init failure → falls back to fallback_theta_init
  - graceful failure: bad model_fn lands in fit_error_, doesn't raise
  - predict on original scale (with and without smearing)
  - summary returns the documented keys
"""

from __future__ import annotations

import numpy as np
import pytest

import transformation_algorithms as ta


GOOD_INIT = np.array([1.0, 0.1])
BAD_FN = lambda X, t: (_ for _ in ()).throw(RuntimeError("boom"))


def _log_coeff_to_init(m):
    """Convert log-OLS coefficients into nonlinear (a, b) for y = a·exp(b·x)."""
    beta = m.beta_
    return np.array([float(np.exp(beta[0])), float(beta[1])])


@pytest.fixture
def restricted_log_suite():
    """Single-entry suite — keeps the test independent of the default
    suite contents and ensures the screen always picks log."""
    return {"log_smear": ta.TransformedOLS(transform="log", use_smearing=True)}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestZZUEndToEnd:
    def test_low_rmse_on_log_linear_data(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        X, y, true_ab = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
        ).fit(X, y)
        assert zzu.fit_error_ in (None, "")

        pred = zzu.predict(X)
        rmse = ta.regression_metrics(y, pred)["rmse"]
        # Should be on the order of the noise scale, not the response scale.
        assert rmse < 0.5 * float(np.std(y))

        # Final theta should be close to the truth.
        assert zzu.nonlinear_regressor_.theta_[0] == pytest.approx(true_ab[0], abs=0.4)
        assert zzu.nonlinear_regressor_.theta_[1] == pytest.approx(true_ab[1], abs=0.05)

    def test_screening_table_populated(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
        ).fit(X, y)
        table = zzu.screening_table_
        assert table is not None
        assert "name" in table.columns
        assert "val_rmse" in table.columns
        # The screen used 1 candidate, and that candidate should have a
        # finite RMSE.
        assert len(table) == 1
        assert np.isfinite(table["val_rmse"].iloc[0])

    def test_default_suite_runs_without_user_transforms(
            self, exp_mult_data, exp_model_fn):
        """When transformations=None, ZZU uses the 7-model default suite."""
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            # On the default suite, the screen may pick anything; provide
            # a fallback_theta_init in case coeff_to_init's log inverse
            # is wrong for whatever wins.
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            fallback_theta_init=np.array([1.0, 0.5]),
        ).fit(X, y)
        # Either it succeeded outright, or coeff_to_init mismatch was
        # caught and fallback was used; either way fit completes and a
        # nonlinear regressor exists.
        assert zzu.nonlinear_regressor_ is not None


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------

class TestZZUFallbacks:
    def test_coeff_to_init_failure_uses_fallback(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        """If coeff_to_init raises, ZZU should use fallback_theta_init."""
        X, y, _ = exp_mult_data

        def raising_init(_m):
            raise RuntimeError("intentional")

        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=raising_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
            fallback_theta_init=np.array([1.5, 0.2]),
        ).fit(X, y)

        # The fallback should have been used.
        assert zzu.theta_init_used_ == pytest.approx([1.5, 0.2])
        assert zzu.fit_error_ and "coeff_to_init" in zzu.fit_error_
        # The fit should still have proceeded.
        assert zzu.nonlinear_regressor_ is not None

    def test_coeff_to_init_failure_default_fallback(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        """If neither coeff_to_init nor fallback is supplied, ZZU uses
        np.ones with the inferred dimension."""
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=lambda _m: (_ for _ in ()).throw(RuntimeError("boom")),
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
            # fallback_theta_init left as None
        ).fit(X, y)
        assert zzu.theta_init_used_ is not None
        assert np.all(zzu.theta_init_used_ == 1.0)


# ---------------------------------------------------------------------------
# Error containment
# ---------------------------------------------------------------------------

class TestZZUErrorContainment:
    def test_bad_model_fn_does_not_raise(
            self, exp_mult_data, restricted_log_suite):
        """A model_fn that raises in the nonlinear step should be captured
        in the inner regressor's fit_error_; ZZU.fit() must not propagate."""
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=BAD_FN,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
        ).fit(X, y)  # must not raise
        # The screen completed, so screening_table_ is set.
        assert zzu.screening_table_ is not None
        # The inner regressor's fit_error_ should be set.
        assert zzu.nonlinear_regressor_ is not None
        assert zzu.nonlinear_regressor_.fit_error_

    def test_predict_before_fit_raises(self, exp_model_fn, restricted_log_suite):
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
        )
        with pytest.raises(RuntimeError, match="fit"):
            zzu.predict(np.array([[1.0]]))

    def test_summary_before_fit_raises(self, exp_model_fn, restricted_log_suite):
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
        )
        with pytest.raises(RuntimeError, match="fit"):
            zzu.summary()

    def test_unknown_nonlinear_method_raises_or_captures(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        """ZZU's fit() wraps everything in try/except and stores the error
        rather than propagating, so an unknown method should be reflected
        in fit_error_ but not raise."""
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="mystery",
            transformations=restricted_log_suite,
        ).fit(X, y)
        assert zzu.fit_error_  # captured, not raised


# ---------------------------------------------------------------------------
# Predict and summary
# ---------------------------------------------------------------------------

class TestZZUPredictAndSummary:
    def test_predict_returns_original_scale(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
        ).fit(X, y)
        pred = zzu.predict(X)
        assert pred.shape == (len(y),)
        # Original-scale predictions: positive on this dataset.
        assert np.all(pred > 0)

    def test_smearing_flag_changes_predictions(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
            use_smearing=True,
        ).fit(X, y)
        pred_with = zzu.predict(X, use_smearing=True)
        pred_without = zzu.predict(X, use_smearing=False)
        # Smearing adds the mean training residual, which shouldn't be 0.
        assert not np.allclose(pred_with, pred_without)

    def test_summary_returns_expected_keys(
            self, exp_mult_data, exp_model_fn, restricted_log_suite):
        X, y, _ = exp_mult_data
        zzu = ta.ZZUTransformRegressor(
            model_fn=exp_model_fn,
            coeff_to_init=_log_coeff_to_init,
            nonlinear_method="bfgs",
            transformations=restricted_log_suite,
        ).fit(X, y)
        s = zzu.summary()
        for k in ("best_transform", "selected_lambda", "nonlinear_method",
                  "final_theta", "converged", "theta_init_used",
                  "train_metrics"):
            assert k in s
