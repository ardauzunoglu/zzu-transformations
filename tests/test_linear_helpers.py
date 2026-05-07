"""Tests for the low-level linear-algebra helpers (Section 2 of
transformation_algorithms.py): as_2d, add_intercept, ols_fit, ols_predict.
"""

from __future__ import annotations

import numpy as np
import pytest

import transformation_algorithms as ta


class TestAs2D:
    def test_1d_array_promoted_to_column(self):
        out = ta.as_2d(np.array([1.0, 2.0, 3.0]))
        assert out.shape == (3, 1)

    def test_2d_array_unchanged(self):
        x = np.zeros((5, 3))
        out = ta.as_2d(x)
        assert out.shape == (5, 3)
        assert np.shares_memory(out, x) or np.array_equal(out, x)

    def test_python_list_accepted(self):
        out = ta.as_2d([[1, 2], [3, 4]])
        assert out.shape == (2, 2)

    def test_3d_array_raises(self):
        with pytest.raises(ValueError):
            ta.as_2d(np.zeros((2, 3, 4)))

    def test_returns_float_dtype(self):
        out = ta.as_2d(np.array([1, 2, 3], dtype=int))
        assert np.issubdtype(out.dtype, np.floating)


class TestAddIntercept:
    def test_appends_ones_column_at_left(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = ta.add_intercept(X)
        assert out.shape == (2, 3)
        assert np.all(out[:, 0] == 1.0)
        assert np.array_equal(out[:, 1:], X)


class TestOLSFit:
    def test_recovers_true_coefficients_no_noise(self, linear_data):
        X, y = linear_data
        beta, fitted, residuals = ta.ols_fit(X, y)
        # beta = [intercept, slope]  for the column-stack [1, x].
        assert beta == pytest.approx([2.0, 3.0], abs=1e-9)
        assert np.allclose(fitted, y, atol=1e-9)
        assert np.allclose(residuals, 0.0, atol=1e-9)

    def test_residuals_orthogonal_to_design_matrix(self, linear_data_noisy):
        X, y = linear_data_noisy
        _, _, residuals = ta.ols_fit(X, y)
        # Normal equations imply [1, X]ᵀ · r = 0.
        X_int = ta.add_intercept(X)
        ortho = X_int.T @ residuals
        assert np.allclose(ortho, 0.0, atol=1e-8)

    def test_returns_three_arrays_with_expected_shapes(self, linear_data_noisy):
        X, y = linear_data_noisy
        beta, fitted, residuals = ta.ols_fit(X, y)
        assert beta.shape == (X.shape[1] + 1,)     # +1 for intercept
        assert fitted.shape == y.shape
        assert residuals.shape == y.shape


class TestOLSPredict:
    def test_matches_fitted_on_training_data(self, linear_data_noisy):
        X, y = linear_data_noisy
        beta, fitted, _ = ta.ols_fit(X, y)
        pred = ta.ols_predict(X, beta)
        assert pred == pytest.approx(fitted, abs=1e-12)

    def test_extrapolates_linearly(self, linear_data):
        X, _ = linear_data
        beta, _, _ = ta.ols_fit(X, 2.0 + 3.0 * X[:, 0])
        # Predict at a new point not in training.
        x_new = np.array([[100.0]])
        pred = ta.ols_predict(x_new, beta)
        assert pred[0] == pytest.approx(2.0 + 3.0 * 100.0, abs=1e-9)
