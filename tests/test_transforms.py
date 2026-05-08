"""Tests for the response-transformation families and λ selection
(Sections 3-6 of algorithms.py).
"""

from __future__ import annotations

import numpy as np
import pytest

import algorithms as ta


# ---------------------------------------------------------------------------
# Per-family round-trip tests: T⁻¹(T(y)) ≈ y on valid input
# ---------------------------------------------------------------------------

POSITIVE_Y = np.array([0.5, 1.0, 2.5, 10.0, 100.0])
MIXED_Y = np.array([-3.0, -1.0, -0.1, 0.5, 2.5])  # for Yeo-Johnson


class TestIdentity:
    def test_round_trip(self):
        out = ta.identity_inverse(ta.identity_forward(POSITIVE_Y))
        assert out == pytest.approx(POSITIVE_Y)


class TestLog:
    def test_round_trip(self):
        out = ta.log_inverse(ta.log_forward(POSITIVE_Y))
        assert out == pytest.approx(POSITIVE_Y, rel=1e-12)

    def test_log_of_one_is_zero(self):
        z = ta.log_forward(np.array([1.0]))
        assert z[0] == pytest.approx(0.0)

    def test_negative_input_raises(self):
        with pytest.raises(ValueError, match="y > 0"):
            ta.log_forward(np.array([-1.0, 2.0]))

    def test_zero_input_raises(self):
        with pytest.raises(ValueError):
            ta.log_forward(np.array([0.0, 1.0]))


class TestReciprocal:
    def test_round_trip(self):
        out = ta.reciprocal_inverse(ta.reciprocal_forward(POSITIVE_Y))
        assert out == pytest.approx(POSITIVE_Y, rel=1e-12)

    def test_near_zero_input_raises(self):
        with pytest.raises(ValueError, match="zero"):
            ta.reciprocal_forward(np.array([1e-15, 1.0]))


class TestPower:
    @pytest.mark.parametrize("p", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_round_trip_positive_power(self, p):
        z = ta.power_forward(POSITIVE_Y, p)
        out = ta.power_inverse(z, p)
        assert out == pytest.approx(POSITIVE_Y, rel=1e-12)

    def test_power_zero_falls_back_to_log(self):
        """power=0 is documented as the log limit of the family."""
        out = ta.power_forward(POSITIVE_Y, 0.0)
        # The library treats |power| < 1e-10 specially; in that branch the
        # output should be finite and monotone with y.
        assert np.all(np.isfinite(out))
        assert np.all(np.diff(out) > 0)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="y > 0"):
            ta.power_forward(np.array([-1.0, 2.0]), 0.5)


class TestBoxCox:
    @pytest.mark.parametrize("lam", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_round_trip(self, lam):
        z = ta.boxcox_forward(POSITIVE_Y, lam)
        out = ta.boxcox_inverse(z, lam)
        assert out == pytest.approx(POSITIVE_Y, rel=1e-9)

    def test_lambda_one_is_y_minus_one(self):
        """Box-Cox with λ=1 is the identity shifted by 1: T(y) = y − 1."""
        out = ta.boxcox_forward(POSITIVE_Y, 1.0)
        assert out == pytest.approx(POSITIVE_Y - 1.0)

    def test_lambda_zero_is_log(self):
        """Box-Cox with λ=0 is exactly log y."""
        out = ta.boxcox_forward(POSITIVE_Y, 0.0)
        assert out == pytest.approx(np.log(POSITIVE_Y), rel=1e-12)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="y > 0"):
            ta.boxcox_forward(np.array([-1.0, 1.0]), 0.5)


class TestYeoJohnson:
    @pytest.mark.parametrize("lam", [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_round_trip_includes_negatives(self, lam):
        """Yeo-Johnson is the only family that handles y ≤ 0."""
        z = ta.yeojohnson_forward(MIXED_Y, lam)
        out = ta.yeojohnson_inverse(z, lam)
        assert out == pytest.approx(MIXED_Y, rel=1e-8, abs=1e-10)

    def test_lambda_one_positive_branch(self):
        """For y ≥ 0 with λ=1, Yeo-Johnson is exactly (y+1)−1 = y."""
        y_pos = np.array([0.0, 1.0, 2.5])
        out = ta.yeojohnson_forward(y_pos, 1.0)
        assert out == pytest.approx(y_pos)


# ---------------------------------------------------------------------------
# Profile-likelihood λ selection
# ---------------------------------------------------------------------------

class TestLambdaSelection:
    def test_boxcox_selects_lambda_near_zero_on_log_linear_data(self, rng):
        """When the noiseless model is y = exp(a + b·x), Box-Cox should
        choose λ ≈ 0 (the log)."""
        n = 200
        x = np.linspace(0.0, 3.0, n)
        # log y = 1 + 2x + N(0, 0.05²) → multiplicative-lognormal y.
        log_y = 1.0 + 2.0 * x + rng.normal(0.0, 0.05, size=n)
        y = np.exp(log_y)
        lam, table = ta.choose_lambda_by_profile_likelihood(
            x.reshape(-1, 1), y, family="boxcox",
        )
        # The optimum λ should be close to 0 for clean log-linear data.
        assert abs(lam) < 0.25, f"expected λ≈0, got {lam}"
        # The returned table should cover the search grid.
        assert "lambda" in table.columns
        assert "profile_loglik" in table.columns
        assert table["profile_loglik"].notna().sum() > 5

    def test_boxcox_selects_lambda_near_one_on_linear_data(self, linear_data_noisy):
        """When the relationship is already linear, λ ≈ 1 (identity-ish)."""
        X, y = linear_data_noisy
        # ols_fit is fine on linear data; log-likelihood of Box-Cox should
        # peak near λ = 1.
        lam, _ = ta.choose_lambda_by_profile_likelihood(
            X, y, family="boxcox",
        )
        assert 0.5 < lam <= 2.0, f"expected λ≈1, got {lam}"

    def test_yeojohnson_handles_negative_y(self, rng):
        """Yeo-Johnson should run successfully on data with negative y."""
        n = 100
        x = np.linspace(-1.0, 1.0, n)
        y = x + rng.normal(0.0, 0.1, size=n)   # contains negatives
        lam, _ = ta.choose_lambda_by_profile_likelihood(
            x.reshape(-1, 1), y, family="yeojohnson",
        )
        assert np.isfinite(lam)

    def test_unknown_family_in_profile_loglik_raises(self, linear_data):
        """The low-level profile-loglik function rejects unknown families."""
        X, y = linear_data
        with pytest.raises(ValueError, match="boxcox"):
            ta.transformation_profile_loglik(X, y, family="bogus", lam=0.5)

    def test_choose_lambda_swallows_inner_errors_per_grid_point(self, linear_data):
        """choose_lambda_by_profile_likelihood catches per-grid exceptions
        (e.g., bad family, log on negatives) and records -inf scores rather
        than propagating.  All scores being -inf means no λ is preferred."""
        X, y = linear_data
        _, table = ta.choose_lambda_by_profile_likelihood(
            X, y, family="bogus",
        )
        # Every grid point should have failed and recorded -inf.
        assert (table["profile_loglik"] == -np.inf).all()
