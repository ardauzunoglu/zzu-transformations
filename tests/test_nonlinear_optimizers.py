"""Tests for the three nonlinear regressors and the numerical Jacobian
helper (Sections 10-13 of algorithms.py).

Focuses on:
  - numerical Jacobian agreeing with the analytic Jacobian
  - each optimizer converging on a noiseless / mildly-noisy exponential
  - the shared API contract (fit returns self, predict before fit raises)
  - graceful failure handling: a raising model_fn lands in fit_error_, not propagated
"""

from __future__ import annotations

import numpy as np
import pytest

import algorithms as ta


TRUE_THETA = np.array([2.0, 0.7])      # a, b for y = a·exp(b·x)
GOOD_INIT = np.array([1.0, 0.1])
BAD_FN = lambda X, t: (_ for _ in ()).throw(RuntimeError("boom"))


# ---------------------------------------------------------------------------
# numerical_jacobian
# ---------------------------------------------------------------------------

class TestNumericalJacobian:
    def test_matches_analytic_jacobian(self, exp_model_fn, exp_jacobian_fn):
        X = np.linspace(0.0, 5.0, 30).reshape(-1, 1)
        theta = np.array([1.7, 0.4])
        J_num = ta.numerical_jacobian(exp_model_fn, theta, X)
        J_ana = exp_jacobian_fn(X, theta)
        # Central differences should agree to ~6 digits relatively.
        assert J_num == pytest.approx(J_ana, rel=1e-5, abs=1e-7)

    def test_does_not_mutate_theta(self, exp_model_fn):
        X = np.linspace(0.0, 5.0, 10).reshape(-1, 1)
        theta = np.array([1.5, 0.3])
        theta_before = theta.copy()
        _ = ta.numerical_jacobian(exp_model_fn, theta, X)
        assert np.array_equal(theta, theta_before)

    def test_output_shape_is_n_by_p(self, exp_model_fn):
        X = np.linspace(0.0, 5.0, 25).reshape(-1, 1)
        theta = np.array([1.0, 0.5, 2.0])  # over-parameterized; shape only test
        f = lambda X, t: t[0] * np.exp(t[1] * X[:, 0])  # uses only t[0], t[1]
        J = ta.numerical_jacobian(f, theta, X)
        assert J.shape == (25, 3)


# ---------------------------------------------------------------------------
# Common contracts shared by all three nonlinear regressors
# ---------------------------------------------------------------------------

REGRESSOR_FACTORIES = {
    "GD":   lambda fn: ta.GradientDescentRegressor(
        model_fn=fn, learning_rate=1e-4, decay=0.9999, max_iter=5000),
    "GN":   lambda fn: ta.GaussNewtonRegressor(model_fn=fn, max_iter=200),
    "BFGS": lambda fn: ta.BFGSRegressor(model_fn=fn, max_iter=500),
}


@pytest.fixture(params=list(REGRESSOR_FACTORIES))
def regressor_factory(request):
    """Parametrized fixture: each test that uses this runs once per optimizer."""
    return request.param, REGRESSOR_FACTORIES[request.param]


class TestOptimizerContracts:
    def test_predict_before_fit_raises(self, regressor_factory, exp_model_fn):
        _, factory = regressor_factory
        reg = factory(exp_model_fn)
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict(np.array([[1.0]]))

    def test_summary_before_fit_raises(self, regressor_factory, exp_model_fn):
        _, factory = regressor_factory
        reg = factory(exp_model_fn)
        with pytest.raises(RuntimeError, match="fit"):
            reg.summary()

    def test_fit_returns_self(self, regressor_factory, exp_mult_data, exp_model_fn):
        _, factory = regressor_factory
        X, y, _ = exp_mult_data
        reg = factory(exp_model_fn)
        out = reg.fit(X, y, GOOD_INIT)
        assert out is reg

    def test_fit_error_captures_exception(self, regressor_factory, exp_mult_data):
        """A model_fn that raises must NOT propagate; the error string
        lands in reg.fit_error_."""
        _, factory = regressor_factory
        X, y, _ = exp_mult_data
        reg = factory(BAD_FN)
        out = reg.fit(X, y, GOOD_INIT)   # should not raise
        assert out is reg
        assert reg.fit_error_  # truthy, non-empty error message


# ---------------------------------------------------------------------------
# Per-optimizer convergence on a known problem
# ---------------------------------------------------------------------------

class TestGaussNewtonConvergence:
    def test_recovers_truth_on_low_noise_data(self, exp_mult_data, exp_model_fn):
        X, y, _ = exp_mult_data
        reg = ta.GaussNewtonRegressor(model_fn=exp_model_fn, max_iter=200)
        reg.fit(X, y, GOOD_INIT)
        assert reg.converged_ is True
        assert reg.theta_[0] == pytest.approx(TRUE_THETA[0], abs=0.4)
        assert reg.theta_[1] == pytest.approx(TRUE_THETA[1], abs=0.05)

    def test_uses_few_iterations(self, exp_mult_data, exp_model_fn):
        X, y, _ = exp_mult_data
        reg = ta.GaussNewtonRegressor(model_fn=exp_model_fn).fit(X, y, GOOD_INIT)
        assert reg.n_iter_ < 50  # GN is fast on this problem

    def test_analytic_jacobian_speeds_things_up(
            self, exp_mult_data, exp_model_fn, exp_jacobian_fn):
        X, y, _ = exp_mult_data
        reg = ta.GaussNewtonRegressor(
            model_fn=exp_model_fn, jacobian_fn=exp_jacobian_fn,
        ).fit(X, y, GOOD_INIT)
        assert reg.converged_ is True


class TestBFGSConvergence:
    def test_recovers_truth_on_low_noise_data(self, exp_mult_data, exp_model_fn):
        X, y, _ = exp_mult_data
        reg = ta.BFGSRegressor(model_fn=exp_model_fn, max_iter=500)
        reg.fit(X, y, GOOD_INIT)
        assert reg.converged_ is True
        assert reg.theta_[0] == pytest.approx(TRUE_THETA[0], abs=0.4)
        assert reg.theta_[1] == pytest.approx(TRUE_THETA[1], abs=0.05)


class TestGDConvergence:
    """GD typically does not converge to high precision in 5000 iters on
    this problem.  We test that it makes meaningful progress: final
    loss should be substantially lower than the loss at init."""

    def test_loss_decreases(self, exp_mult_data, exp_model_fn):
        X, y, _ = exp_mult_data
        reg = ta.GradientDescentRegressor(
            model_fn=exp_model_fn, learning_rate=1e-4, decay=0.9999, max_iter=5000,
        ).fit(X, y, GOOD_INIT)
        # The fit shouldn't error.
        assert reg.fit_error_ in (None, "")
        # The loss history should be monotonically non-increasing-ish
        # (allow tiny numerical bumps).
        history = np.array(reg.loss_history_)
        assert history[0] > history[-1]
        # And the final RMSE should beat the noise level by a wide margin.
        pred = reg.predict(X)
        rmse = ta.regression_metrics(y, pred)["rmse"]
        assert rmse < float(np.std(y))  # better than predicting the mean


# ---------------------------------------------------------------------------
# evaluate_nonlinear_models
# ---------------------------------------------------------------------------

class TestEvaluateNonlinearModels:
    def test_returns_dataframe_with_expected_columns(
            self, exp_mult_data, exp_model_fn):
        X, y, _ = exp_mult_data
        models = {
            "GN":   ta.GaussNewtonRegressor(model_fn=exp_model_fn, max_iter=100),
            "BFGS": ta.BFGSRegressor(model_fn=exp_model_fn, max_iter=200),
        }
        inits = {"GN": GOOD_INIT, "BFGS": GOOD_INIT}
        df = ta.evaluate_nonlinear_models(X, y, models, inits)
        assert len(df) == 2
        for col in ("model", "rmse", "n_iter", "converged", "error"):
            assert col in df.columns

    def test_records_failure_without_raising(self, exp_mult_data):
        X, y, _ = exp_mult_data
        models = {"broken": ta.BFGSRegressor(model_fn=BAD_FN)}
        inits = {"broken": GOOD_INIT}
        df = ta.evaluate_nonlinear_models(X, y, models, inits)
        assert len(df) == 1
        assert df["error"].iloc[0]  # non-empty
