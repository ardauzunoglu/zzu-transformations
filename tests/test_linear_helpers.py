"""Tests for the low-level linear-algebra helpers (Section 2 of
algorithms.py): as_2d, add_intercept, ols_fit, ols_predict.
"""

from __future__ import annotations

import numpy as np
import pytest

import algorithms as ta


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


# ---------------------------------------------------------------------------
# train_test_split_arrays — used by every benchmark in the project.
# ---------------------------------------------------------------------------
#
# Why this section matters
# ------------------------
# Every RMSE / cost number in run_comparison.py and cost_analysis.py comes
# from this split.  A silent partition bug here (data leakage, dropped
# rows, non-deterministic splits) would invalidate every figure in the
# writeup, and the bug would not be visible by inspecting the CSVs —
# you'd just see plausible numbers that no longer correspond to the
# methodology the paper claims.

class TestTrainTestSplitArrays:
    def test_partition_covers_all_indices_no_overlap(self):
        # The fundamental contract: every row appears in exactly one of
        # (train, test).  Anything else is data leakage or dropped rows.
        n = 50
        X = np.arange(n, dtype=float).reshape(-1, 1)
        y = np.arange(n, dtype=float)
        X_tr, X_te, _, _ = ta.train_test_split_arrays(
            X, y, test_size=0.2, seed=42,
        )
        train_set = set(X_tr.ravel().astype(int).tolist())
        test_set = set(X_te.ravel().astype(int).tolist())
        assert train_set & test_set == set()              # no overlap
        assert train_set | test_set == set(range(n))      # full coverage

    def test_split_proportions_match_test_size(self):
        # Documented behavior: n_test = ceil(test_size * n).  Catches a
        # silent change to floor() or rounding that would shift the
        # train/test ratio across all benchmarks.
        n = 100
        X = np.arange(n, dtype=float).reshape(-1, 1)
        y = np.zeros(n)
        X_tr, X_te, _, _ = ta.train_test_split_arrays(
            X, y, test_size=0.25, seed=0,
        )
        assert len(X_te) == 25
        assert len(X_tr) == 75

    def test_same_seed_is_bit_identical(self):
        # Reproducibility: paper figures over "seeds 0..9" need
        # deterministic splits to be regenerable from scratch.
        n = 30
        X = np.arange(n, dtype=float).reshape(-1, 1)
        y = np.zeros(n)
        a = ta.train_test_split_arrays(X, y, test_size=0.3, seed=7)
        b = ta.train_test_split_arrays(X, y, test_size=0.3, seed=7)
        for arr_a, arr_b in zip(a, b):
            assert np.array_equal(arr_a, arr_b)

    def test_different_seed_changes_split(self):
        # Sanity: the seed parameter actually controls randomness.  If
        # different seeds produced identical splits, "10 splits" would
        # secretly be one split repeated 10×.
        n = 30
        X = np.arange(n, dtype=float).reshape(-1, 1)
        y = np.zeros(n)
        _, X_te_a, _, _ = ta.train_test_split_arrays(X, y, test_size=0.3, seed=1)
        _, X_te_b, _, _ = ta.train_test_split_arrays(X, y, test_size=0.3, seed=2)
        assert not np.array_equal(X_te_a, X_te_b)

    def test_1d_X_promoted_to_2d(self):
        # API contract — must mirror as_2d behavior so users can pass
        # raw 1D arrays without manual reshaping.
        n = 20
        X = np.arange(n, dtype=float)   # 1D
        y = np.zeros(n)
        X_tr, X_te, _, _ = ta.train_test_split_arrays(X, y, test_size=0.5, seed=0)
        assert X_tr.ndim == 2 and X_te.ndim == 2

    def test_2d_X_preserves_feature_columns(self):
        # When X has multiple features, all p columns must survive the
        # split unchanged — otherwise multivariable_nonlinear would
        # silently lose predictors during evaluation.
        n, p = 40, 3
        X = np.arange(n * p, dtype=float).reshape(n, p)
        y = np.zeros(n)
        X_tr, X_te, _, _ = ta.train_test_split_arrays(X, y, test_size=0.25, seed=0)
        assert X_tr.shape[1] == p and X_te.shape[1] == p

    def test_X_and_y_remain_aligned_after_split(self):
        # If permutation indices for X and y diverge, every prediction
        # would be evaluated against a mis-aligned target — yielding
        # plausible-looking but meaningless RMSE values.  This test
        # encodes y_i = 10 * X_i and checks that the relationship
        # survives the split.
        n = 40
        X = np.arange(n, dtype=float).reshape(-1, 1)
        y = 10.0 * X.ravel()
        X_tr, X_te, y_tr, y_te = ta.train_test_split_arrays(
            X, y, test_size=0.25, seed=11,
        )
        assert np.allclose(y_tr, 10.0 * X_tr.ravel())
        assert np.allclose(y_te, 10.0 * X_te.ravel())
