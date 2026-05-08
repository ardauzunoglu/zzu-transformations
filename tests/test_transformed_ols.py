"""Tests for the TransformedOLS class and the original-scale evaluation
helpers (Sections 7-9 of algorithms.py).

Covers:
  - end-to-end fit/predict accuracy for each transform family
  - smearing-correction reduces back-transformation bias
  - regression_metrics + residual_diagnostics return the expected fields
  - graceful failures (predict-before-fit, invalid transform)
"""

from __future__ import annotations

import numpy as np
import pytest

import algorithms as ta


# ---------------------------------------------------------------------------
# TransformedOLS — fit / predict
# ---------------------------------------------------------------------------

class TestTransformedOLSIdentity:
    def test_recovers_linear_truth(self, linear_data):
        X, y = linear_data
        m = ta.TransformedOLS(transform="identity").fit(X, y)
        pred = m.predict(X)
        assert pred == pytest.approx(y, abs=1e-9)

    def test_predict_returns_1d_array_of_length_m(self, linear_data):
        X, y = linear_data
        m = ta.TransformedOLS(transform="identity").fit(X, y)
        X_new = np.array([[5.0], [7.5]])
        pred = m.predict(X_new)
        assert pred.shape == (2,)


class TestTransformedOLSLog:
    def test_low_rmse_on_log_linear_data(self, exp_mult_data):
        X, y, _ = exp_mult_data
        m = ta.TransformedOLS(transform="log", use_smearing=True).fit(X, y)
        pred = m.predict(X)
        rmse = ta.regression_metrics(y, pred)["rmse"]
        # Best-case data; RMSE should be small relative to y scale.
        assert rmse < 0.5 * float(np.std(y))

    def test_negative_y_raises_on_fit(self, rng):
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = rng.normal(0, 1, 20)  # contains negatives
        with pytest.raises(ValueError):
            ta.TransformedOLS(transform="log").fit(X, y)


class TestTransformedOLSBoxCox:
    def test_lambda_selected_and_reported(self, exp_mult_data):
        X, y, _ = exp_mult_data
        m = ta.TransformedOLS(transform="boxcox").fit(X, y)
        # Box-Cox should pick a finite λ and store it.
        assert np.isfinite(m.selected_param_)
        # The lambda table should be populated by the profile-likelihood search.
        assert m.lambda_table_ is not None
        assert len(m.lambda_table_) > 5


class TestTransformedOLSContractsAndErrors:
    def test_predict_before_fit_raises(self):
        m = ta.TransformedOLS(transform="identity")
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(np.array([[1.0]]))

    def test_summary_before_fit_raises(self):
        m = ta.TransformedOLS(transform="identity")
        with pytest.raises(RuntimeError, match="fit"):
            m.summary()

    def test_unknown_transform_raises_on_fit(self, linear_data):
        X, y = linear_data
        m = ta.TransformedOLS(transform="bogus")
        with pytest.raises(ValueError):
            m.fit(X, y)

    def test_summary_returns_expected_keys(self, linear_data):
        X, y = linear_data
        m = ta.TransformedOLS(transform="identity").fit(X, y)
        s = m.summary()
        for k in ("transform", "selected_param_or_lambda", "use_smearing",
                  "coefficients", "transformed_residual_std"):
            assert k in s


# ---------------------------------------------------------------------------
# Smearing should reduce retransformation bias
# ---------------------------------------------------------------------------

class TestSmearing:
    def test_smearing_changes_predictions_on_log_model(self, exp_mult_data):
        """For a log model, naïve back-transformation underestimates the
        mean.  With Duan smearing the predicted mean should shift upward."""
        X, y, _ = exp_mult_data
        m_naive = ta.TransformedOLS(transform="log", use_smearing=False).fit(X, y)
        m_smear = ta.TransformedOLS(transform="log", use_smearing=True).fit(X, y)
        pred_naive = m_naive.predict(X)
        pred_smear = m_smear.predict(X)
        # Smearing multiplies by mean(exp(residuals)) ≥ 1, so smeared
        # predictions should be at least as large element-wise.
        assert np.mean(pred_smear) >= np.mean(pred_naive)
        # And not identical — they should differ by a meaningful amount.
        assert not np.allclose(pred_naive, pred_smear, rtol=1e-6)


# ---------------------------------------------------------------------------
# regression_metrics + residual_diagnostics
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_perfect_prediction_zero_error(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = ta.regression_metrics(y, y.copy())
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["mse"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_constant_prediction_r2_is_zero(self):
        """When predicting y_mean for everything, R² = 0 by definition."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.full_like(y, y.mean())
        m = ta.regression_metrics(y, pred)
        assert m["r2"] == pytest.approx(0.0, abs=1e-12)

    def test_returns_required_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        m = ta.regression_metrics(y, y.copy())
        for k in ("rmse", "mae", "mse", "r2", "n_valid"):
            assert k in m

    def test_drops_non_finite_values(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, np.nan, 3.0])
        m = ta.regression_metrics(y, pred)
        assert m["n_valid"] == 2


class TestResidualDiagnostics:
    def test_returns_required_keys(self, linear_data_noisy):
        X, y = linear_data_noisy
        m = ta.TransformedOLS(transform="identity").fit(X, y)
        diag = ta.residual_diagnostics(y, m.predict(X))
        for k in ("residual_mean", "residual_std",
                  "residual_skew", "residual_excess_kurtosis"):
            assert k in diag


# ---------------------------------------------------------------------------
# evaluate_transformed_models
# ---------------------------------------------------------------------------

class TestEvaluateTransformedModels:
    def test_returns_dataframe_sorted_by_rmse(self, exp_mult_data):
        X, y, _ = exp_mult_data
        X_tr, X_te, y_tr, y_te = ta.train_test_split_arrays(
            X, y, test_size=0.2, seed=0,
        )
        models = {
            "identity": ta.TransformedOLS(transform="identity"),
            "log_smear": ta.TransformedOLS(transform="log", use_smearing=True),
            "boxcox_smear": ta.TransformedOLS(transform="boxcox", use_smearing=True),
        }
        df = ta.evaluate_transformed_models(models, X_tr, y_tr, X_te, y_te)
        assert len(df) == 3
        # Sorted ascending by RMSE.
        rmses = df["rmse"].dropna().values
        assert all(rmses[i] <= rmses[i + 1] for i in range(len(rmses) - 1))
        for col in ("model", "transform", "selected_param_or_lambda",
                    "rmse", "mae", "r2", "error"):
            assert col in df.columns

    def test_failure_recorded_not_raised(self, rng):
        """Passing a model that doesn't fit the data (log on negatives)
        should record the error string, not crash the screen."""
        n = 30
        X = np.linspace(0, 1, n).reshape(-1, 1)
        y = rng.normal(0, 1, n)  # has negatives → log fails
        models = {
            "log": ta.TransformedOLS(transform="log"),
            "identity": ta.TransformedOLS(transform="identity"),
        }
        df = ta.evaluate_transformed_models(models, X, y, X, y)
        assert len(df) == 2
        log_row = df[df["model"] == "log"].iloc[0]
        assert log_row["error"]  # non-empty error string
        # Identity should have completed.
        id_row = df[df["model"] == "identity"].iloc[0]
        assert id_row["error"] == ""
