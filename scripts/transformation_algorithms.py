"""Transformation algorithms for the ZZU regression experiments.

This module implements:

  - Identity, log, reciprocal, and power response transformations.
  - Box-Cox (positive responses) and Yeo-Johnson (any real response).
  - Profile-likelihood λ selection for both Box-Cox and Yeo-Johnson.
  - OLS on the transformed scale (`TransformedOLS`) with optional Duan-style
    smearing correction for retransformation bias.
  - Three pure-NumPy nonlinear regressors: gradient descent,
    Gauss-Newton (with self-activating LM damping), and BFGS.
  - The ZZU hybrid workflow (`ZZUTransformRegressor`): screen → warm-start →
    bias-correct.
  - Original-scale metric and residual diagnostics, plus batch evaluators
    that record errors instead of crashing on a bad model.

Every transformation exposes a forward map, an inverse map, and (where
needed) a log-Jacobian term used by the likelihood-based λ selection.

Importing this module is side-effect-free: no global numpy or matplotlib
state is mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Numerical tolerance used to guard divisions, near-zero comparisons, and
# fall-back branches throughout this module.
EPS = 1e-12


# ---------------------------------------------------------------------------
# Linear algebra helpers
#
# OLS for the transformed regression model  z = beta_0 + X @ beta + epsilon.
# ---------------------------------------------------------------------------


def as_2d(X: np.ndarray) -> np.ndarray:
    """
    Convert a one-dimensional predictor array into a two-dimensional design matrix.

    Parameters
    ----------
    X:
        Predictor array of shape (n,) or (n, p).

    Returns
    -------
    np.ndarray
        Array of shape (n, p).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be one- or two-dimensional.")
    return X


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Add an intercept column to a predictor matrix.
    """
    X = as_2d(X)
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_fit(X: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit ordinary least squares z = beta_0 + X beta + residual.

    Parameters
    ----------
    X:
        Predictor matrix of shape (n, p), without the intercept column.
    z:
        Transformed response of shape (n,).

    Returns
    -------
    beta:
        Estimated coefficients, including the intercept.
    fitted:
        Fitted values on the transformed scale.
    residuals:
        Residuals on the transformed scale.
    """
    X_design = add_intercept(X)
    z = np.asarray(z, dtype=float).ravel()
    beta = np.linalg.pinv(X_design) @ z
    fitted = X_design @ beta
    residuals = z - fitted
    return beta, fitted, residuals


def ols_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Predict transformed-scale fitted values from an OLS coefficient vector.
    """
    return add_intercept(X) @ np.asarray(beta, dtype=float)


# ---------------------------------------------------------------------------
# Basic transformation families
#
# Each family provides forward and inverse maps:
#   identity   :  T(y) = y,    T^-1(z) = z
#   log        :  T(y) = ln y, T^-1(z) = e^z         (requires y > 0)
#   reciprocal :  T(y) = 1/y,  T^-1(z) = 1/z         (unsafe near y = 0)
#   power      :  T_p(y) = y^p (or ln y if p = 0)    (requires y > 0)
# ---------------------------------------------------------------------------


def identity_forward(y: np.ndarray, param: Optional[float] = None) -> np.ndarray:
    """Identity transformation."""
    return np.asarray(y, dtype=float)


def identity_inverse(z: np.ndarray, param: Optional[float] = None) -> np.ndarray:
    """Inverse identity transformation."""
    return np.asarray(z, dtype=float)


def log_forward(y: np.ndarray, param: Optional[float] = None) -> np.ndarray:
    """Logarithmic transformation for strictly positive y."""
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("log transform requires y > 0.")
    return np.log(y)


def log_inverse(z: np.ndarray, param: Optional[float] = None) -> np.ndarray:
    """Inverse logarithmic transformation."""
    return np.exp(np.asarray(z, dtype=float))


def reciprocal_forward(y: np.ndarray, param: Optional[float] = None) -> np.ndarray:
    """Reciprocal transformation T(y) = 1 / y."""
    y = np.asarray(y, dtype=float)
    if np.any(np.abs(y) < EPS):
        raise ValueError("reciprocal transform is unsafe when y is close to zero.")
    return 1.0 / y


def reciprocal_inverse(z: np.ndarray, param: Optional[float] = None) -> np.ndarray:
    """Inverse reciprocal transformation."""
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, np.nan, dtype=float)
    mask = np.abs(z) >= EPS
    out[mask] = 1.0 / z[mask]
    return out


def power_forward(y: np.ndarray, power: float) -> np.ndarray:
    """
    Positive-response power-scale transformation.

    power = 0 corresponds to the log transform.
    """
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("power transform requires y > 0 in this implementation.")
    if abs(power) < 1e-10:
        return np.log(y)
    return np.power(y, power)


def power_inverse(z: np.ndarray, power: float) -> np.ndarray:
    """
    Inverse of the positive-response power-scale transformation.

    For p != 0, only positive transformed values are safely invertible.
    """
    z = np.asarray(z, dtype=float)
    if abs(power) < 1e-10:
        return np.exp(z)
    out = np.full_like(z, np.nan, dtype=float)
    mask = z > 0
    out[mask] = np.power(z[mask], 1.0 / power)
    return out


# ---------------------------------------------------------------------------
# Box-Cox transformation  (requires y > 0)
#
#   T_lam(y) = (y^lam - 1) / lam     if lam != 0,    else  log(y)
#   T_lam^-1(z) = (lam z + 1)^(1/lam) if lam != 0,   else  exp(z)
#
# λ is selected by maximizing the profile log-likelihood
#   l(lam) = -n/2 * log(SSE_lam / n) + (lam - 1) * sum(log y_i),
# where SSE_lam is the residual sum of squares of OLS on T_lam(y).
# ---------------------------------------------------------------------------


def boxcox_forward(y: np.ndarray, lam: float) -> np.ndarray:
    """
    Box-Cox transformation for strictly positive y.
    """
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("Box--Cox requires y > 0.")
    if abs(lam) < 1e-10:
        return np.log(y)
    return (np.power(y, lam) - 1.0) / lam


def boxcox_inverse(z: np.ndarray, lam: float) -> np.ndarray:
    """
    Inverse Box-Cox transformation.

    Values outside the valid inverse domain are returned as NaN instead of
    raising an exception, which is useful during smearing correction.
    """
    z = np.asarray(z, dtype=float)
    if abs(lam) < 1e-10:
        return np.exp(z)

    base = lam * z + 1.0
    out = np.full_like(base, np.nan, dtype=float)
    mask = base > 0
    out[mask] = np.power(base[mask], 1.0 / lam)
    return out


def boxcox_log_jacobian(y: np.ndarray, lam: float) -> float:
    """
    Log-Jacobian term for Box-Cox transformation.
    """
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("Box-Cox log-Jacobian requires y > 0.")
    return float((lam - 1.0) * np.sum(np.log(y)))


# ---------------------------------------------------------------------------
# Yeo-Johnson transformation  (handles any real y, including y <= 0)
#
# Piecewise in y and lam:
#   y >= 0, lam != 0  :  ((y+1)^lam - 1) / lam
#   y >= 0, lam == 0  :  log(y + 1)
#   y <  0, lam != 2  :  -((1-y)^(2-lam) - 1) / (2 - lam)
#   y <  0, lam == 2  :  -log(1 - y)
#
# Log-Jacobian contribution per observation:
#   y_i >= 0  :  (lam - 1) * log(1 + y_i)
#   y_i <  0  :  (1 - lam) * log(1 - y_i)
# ---------------------------------------------------------------------------


def yeojohnson_forward(y: np.ndarray, lam: float) -> np.ndarray:
    """
    Yeo-Johnson transformation for any real-valued y.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)

    nonnegative = y >= 0
    negative = ~nonnegative

    if abs(lam) < 1e-10:
        out[nonnegative] = np.log1p(y[nonnegative])
    else:
        out[nonnegative] = (np.power(y[nonnegative] + 1.0, lam) - 1.0) / lam

    if abs(lam - 2.0) < 1e-10:
        out[negative] = -np.log1p(-y[negative])
    else:
        out[negative] = -(np.power(1.0 - y[negative], 2.0 - lam) - 1.0) / (2.0 - lam)

    return out


def yeojohnson_inverse(z: np.ndarray, lam: float) -> np.ndarray:
    """
    Inverse Yeo-Johnson transformation.

    The inverse branches are determined by transformed-scale sign:
    z >= 0 corresponds to y >= 0, and z < 0 corresponds to y < 0.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    nonnegative = z >= 0
    negative = ~nonnegative

    if abs(lam) < 1e-10:
        out[nonnegative] = np.expm1(z[nonnegative])
    else:
        base_pos = lam * z[nonnegative] + 1.0
        out[nonnegative] = np.where(
            base_pos > 0,
            np.power(base_pos, 1.0 / lam) - 1.0,
            np.nan,
        )

    if abs(lam - 2.0) < 1e-10:
        out[negative] = 1.0 - np.exp(-z[negative])
    else:
        base_neg = 1.0 - (2.0 - lam) * z[negative]
        out[negative] = np.where(
            base_neg > 0,
            1.0 - np.power(base_neg, 1.0 / (2.0 - lam)),
            np.nan,
        )

    return out


def yeojohnson_log_jacobian(y: np.ndarray, lam: float) -> float:
    """
    Log-Jacobian term for Yeo-Johnson transformation.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)
    nonnegative = y >= 0
    negative = ~nonnegative

    out[nonnegative] = (lam - 1.0) * np.log1p(y[nonnegative])
    out[negative] = (1.0 - lam) * np.log1p(-y[negative])
    return float(np.sum(out))


# ---------------------------------------------------------------------------
# Profile-likelihood λ selection for Box-Cox / Yeo-Johnson
#
# A grid search is used (slower than continuous optimization but transparent,
# robust, and easy to audit).  Per-grid-point exceptions are caught and
# recorded as -inf scores so a single bad λ doesn't abort the whole search.
# ---------------------------------------------------------------------------


def transformation_profile_loglik(
    X: np.ndarray,
    y: np.ndarray,
    family: str,
    lam: float,
) -> float:
    """
    Compute profile log-likelihood for a lambda-based transformation family.

    Parameters
    ----------
    X:
        Predictor matrix.
    y:
        Original-scale response.
    family:
        Either "boxcox" or "yeojohnson".
    lam:
        Candidate transformation parameter.

    Returns
    -------
    float
        Profile log-likelihood value.
    """
    y = np.asarray(y, dtype=float).ravel()

    if family == "boxcox":
        z = boxcox_forward(y, lam)
        log_jac = boxcox_log_jacobian(y, lam)
    elif family == "yeojohnson":
        z = yeojohnson_forward(y, lam)
        log_jac = yeojohnson_log_jacobian(y, lam)
    else:
        raise ValueError("family must be 'boxcox' or 'yeojohnson'.")

    _, _, residuals = ols_fit(X, z)
    n = len(y)
    sse = float(np.sum(residuals ** 2))
    sse = max(sse, EPS)

    return -0.5 * n * np.log(sse / n) + log_jac


def choose_lambda_by_profile_likelihood(
    X: np.ndarray,
    y: np.ndarray,
    family: str,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Select lambda by maximizing profile log-likelihood on a grid.

    Returns the selected lambda and a table of all candidate scores.
    """
    if grid is None:
        grid = np.linspace(-2.0, 2.0, 161)

    rows = []
    for lam in grid:
        try:
            score = transformation_profile_loglik(X, y, family=family, lam=float(lam))
        except Exception:
            score = -np.inf
        rows.append({"lambda": float(lam), "profile_loglik": score})

    table = pd.DataFrame(rows)
    best_idx = table["profile_loglik"].idxmax()
    best_lambda = float(table.loc[best_idx, "lambda"])

    return best_lambda, table


# ---------------------------------------------------------------------------
# Duan-style smearing correction
#
# A nonlinear inverse transform does not generally commute with expectation:
#   T^-1(E[T(Y) | X=x])  !=  E[Y | X=x].
#
# The generalized smearing estimator averages the inverse-transformed
# prediction over the empirical residual distribution:
#
#   y_smear(x) = (1/n) * sum_i  T^-1( zhat(x) + residual_i ).
#
# For the log transform this reduces to the familiar multiplicative
# correction  exp(zhat) * mean( exp(residuals) ).
# ---------------------------------------------------------------------------


def generalized_smearing_predict(
    zhat: np.ndarray,
    residuals: np.ndarray,
    inverse_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Generalized Duan-style smearing correction.

    For each transformed-scale prediction zhat_j, average
    inverse_fn(zhat_j + residual_i) over training residuals.

    Invalid inverse values are ignored via nanmean. If all smeared values
    are invalid for a row, the function falls back to the naive inverse.
    """
    zhat = np.asarray(zhat, dtype=float).ravel()
    residuals = np.asarray(residuals, dtype=float).ravel()

    candidate_z = zhat[:, None] + residuals[None, :]
    candidate_y = inverse_fn(candidate_z)

    with np.errstate(invalid="ignore", divide="ignore"):
        smeared = np.nanmean(candidate_y, axis=1)

    fallback = inverse_fn(zhat)
    bad = ~np.isfinite(smeared)
    smeared[bad] = fallback[bad]
    return smeared


# ---------------------------------------------------------------------------
# TransformedOLS — unified fit/predict interface for every transform family
#
# Usage:
#   m = TransformedOLS(transform="boxcox", use_smearing=True).fit(X, y)
#   y_hat = m.predict(X_new)        # original scale, smearing applied
#
# `transform` is one of: "identity", "log", "reciprocal", "power"
# (requires param=p), "boxcox", "yeojohnson".  For Box-Cox and Yeo-Johnson
# λ is auto-selected by profile likelihood unless `lambda_=` is supplied.
# ---------------------------------------------------------------------------

# Single source of truth for the (forward, inverse) pair belonging to each
# transform family.  All wrappers accept `(arg, param)` so we can dispatch
# uniformly — `param` is ignored by identity / log / reciprocal.
_TRANSFORM_DISPATCH: Dict[str, Tuple[Callable, Callable]] = {
    "identity":   (identity_forward,   identity_inverse),
    "log":        (log_forward,        log_inverse),
    "reciprocal": (reciprocal_forward, reciprocal_inverse),
    "power":      (power_forward,      power_inverse),
    "boxcox":     (boxcox_forward,     boxcox_inverse),
    "yeojohnson": (yeojohnson_forward, yeojohnson_inverse),
}


@dataclass
class TransformedOLS:
    """
    Ordinary least squares model fit after transforming the response.

    Parameters
    ----------
    transform:
        Name of response transformation.
    param:
        Fixed parameter for transforms that need one.
        For "power", this is the power p.
    lambda_:
        Fixed lambda for "boxcox" or "yeojohnson". If None, lambda is selected
        by profile likelihood.
    use_smearing:
        Whether to apply generalized Duan-style smearing correction during
        original-scale prediction.
    lambda_grid:
        Optional grid for Box--Cox or Yeo--Johnson lambda selection.
    """
    transform: str = "identity"
    param: Optional[float] = None
    lambda_: Optional[float] = None
    use_smearing: bool = True
    lambda_grid: Optional[np.ndarray] = None

    beta_: Optional[np.ndarray] = field(default=None, init=False)
    residuals_: Optional[np.ndarray] = field(default=None, init=False)
    fitted_transformed_: Optional[np.ndarray] = field(default=None, init=False)
    selected_param_: Optional[float] = field(default=None, init=False)
    lambda_table_: Optional[pd.DataFrame] = field(default=None, init=False)
    n_features_: Optional[int] = field(default=None, init=False)

    def _resolve_param(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Determine the transformation parameter to use.
        """
        if self.transform == "power":
            if self.param is None:
                raise ValueError("Power transform requires param=<power>.")
            return float(self.param)

        if self.transform in {"boxcox", "yeojohnson"}:
            if self.lambda_ is not None:
                return float(self.lambda_)
            best_lambda, table = choose_lambda_by_profile_likelihood(
                X, y, family=self.transform, grid=self.lambda_grid
            )
            self.lambda_table_ = table
            return best_lambda

        return np.nan

    def _forward(self, y: np.ndarray) -> np.ndarray:
        """Apply the model's response transformation."""
        try:
            fwd, _ = _TRANSFORM_DISPATCH[self.transform]
        except KeyError:
            raise ValueError(f"Unknown transform: {self.transform!r}")
        return fwd(y, self.selected_param_)

    def _inverse(self, z: np.ndarray) -> np.ndarray:
        """Apply the inverse of the model's response transformation."""
        try:
            _, inv = _TRANSFORM_DISPATCH[self.transform]
        except KeyError:
            raise ValueError(f"Unknown transform: {self.transform!r}")
        return inv(z, self.selected_param_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformedOLS":
        """
        Fit transformed OLS model.
        """
        X = as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(y) != X.shape[0]:
            raise ValueError("X and y have incompatible lengths.")

        self.n_features_ = X.shape[1]
        self.selected_param_ = self._resolve_param(X, y)

        z = self._forward(y)
        beta, fitted, residuals = ols_fit(X, z)

        self.beta_ = beta
        self.fitted_transformed_ = fitted
        self.residuals_ = residuals
        return self

    def predict_transformed(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on the transformed response scale.
        """
        if self.beta_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        return ols_predict(X, self.beta_)

    def predict(self, X: np.ndarray, use_smearing: Optional[bool] = None) -> np.ndarray:
        """
        Predict on the original response scale.
        """
        if self.beta_ is None:
            raise RuntimeError("Model must be fit before prediction.")

        if use_smearing is None:
            use_smearing = self.use_smearing

        zhat = self.predict_transformed(X)
        if use_smearing:
            return generalized_smearing_predict(
                zhat=zhat,
                residuals=self.residuals_,
                inverse_fn=self._inverse,
            )
        return self._inverse(zhat)

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact model summary.
        """
        if self.beta_ is None:
            raise RuntimeError("Model must be fit before summary.")
        return {
            "transform": self.transform,
            "selected_param_or_lambda": self.selected_param_,
            "use_smearing": self.use_smearing,
            "coefficients": self.beta_,
            "transformed_residual_std": float(np.std(self.residuals_, ddof=1)),
        }


# ---------------------------------------------------------------------------
# Metrics, residual diagnostics, train/test split, and batch evaluator
#
# All metrics are computed on the original response scale so transforms
# with different inverse maps can be compared directly.  The diagnostics
# include skew, excess kurtosis, and a |residual|-vs-fitted correlation
# (a lightweight proxy for heteroscedasticity).
# ---------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute original-scale regression metrics.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return {"n_valid": 0, "rmse": np.nan, "mae": np.nan, "mse": np.nan, "r2": np.nan}

    yt = y_true[mask]
    yp = y_pred[mask]
    err = yt - yp
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > EPS else np.nan

    return {"n_valid": int(mask.sum()), "rmse": rmse, "mae": mae, "mse": mse, "r2": r2}


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute simple original-scale residual diagnostics.

    The absolute-residual/fitted correlation is a lightweight proxy for
    heteroscedasticity. Large magnitude suggests residual spread changes
    with fitted value.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    residuals = y_true[mask] - y_pred[mask]
    fitted = y_pred[mask]

    if len(residuals) < 3:
        return {
            "residual_mean": np.nan,
            "residual_std": np.nan,
            "residual_skew": np.nan,
            "residual_excess_kurtosis": np.nan,
            "corr_abs_resid_fitted": np.nan,
        }

    centered = residuals - residuals.mean()
    sd = residuals.std(ddof=1)
    if sd < EPS:
        skew = 0.0
        kurtosis = -3.0
    else:
        standardized = centered / sd
        skew = float(np.mean(standardized ** 3))
        kurtosis = float(np.mean(standardized ** 4) - 3.0)

    if np.std(np.abs(residuals)) < EPS or np.std(fitted) < EPS:
        corr_abs = np.nan
    else:
        corr_abs = float(np.corrcoef(np.abs(residuals), fitted)[0, 1])

    return {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(sd),
        "residual_skew": skew,
        "residual_excess_kurtosis": kurtosis,
        "corr_abs_resid_fitted": corr_abs,
    }


def train_test_split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal train/test split utility.
    """
    X = as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(np.ceil(test_size * len(y)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def evaluate_transformed_models(
    model_specs: Dict[str, TransformedOLS],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Fit and evaluate a dictionary of TransformedOLS models.

    Models that are not valid for a dataset, such as log on nonpositive y,
    are recorded with an error message instead of stopping the whole screen.
    """
    rows = []

    for name, model in model_specs.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = regression_metrics(y_test, pred)
            diags = residual_diagnostics(y_test, pred)
            row = {
                "model": name,
                "transform": model.transform,
                "selected_param_or_lambda": model.selected_param_,
                "error": "",
                **metrics,
                **diags,
            }
        except Exception as exc:
            # Empty-array calls return the canonical NaN dicts; reuse them
            # so the error-row schema cannot drift from the success path.
            row = {
                "model": name,
                "transform": model.transform,
                "selected_param_or_lambda": np.nan,
                "error": str(exc),
                **regression_metrics(np.array([]), np.array([])),
                **residual_diagnostics(np.array([]), np.array([])),
            }
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["rmse", "model"], na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Numerical Jacobian helper used by the nonlinear regressors
#
# Direct nonlinear least squares minimizes  SSE(theta) = sum (y_i - f_i)^2
# over theta.  All three optimizers below need either an analytic Jacobian
# or this central-finite-difference fallback.
# ---------------------------------------------------------------------------


def numerical_jacobian(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta: np.ndarray,
    X: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """
    Approximate the Jacobian of f(X, theta) with respect to theta using
    central finite differences.

    Parameters
    ----------
    f:
        Function with signature f(X, theta) -> ndarray of shape (n,).
    theta:
        Current parameter vector of shape (p,).
    X:
        Predictor matrix of shape (n, q).
    h:
        Base finite-difference step size.  An adaptive per-parameter step
        h_j = max(h, |theta_j| * h) is used so that relative precision is
        maintained near zero.

    Returns
    -------
    J : ndarray of shape (n, p)
        J[i, j] = df_i / d theta_j at the given theta.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    p = len(theta)
    f0 = np.asarray(f(X, theta), dtype=float).ravel()
    n = len(f0)
    J = np.empty((n, p), dtype=float)

    for j in range(p):
        h_j = max(h, abs(theta[j]) * h)
        e_j = np.zeros(p, dtype=float)
        e_j[j] = h_j
        f_plus = np.asarray(f(X, theta + e_j), dtype=float).ravel()
        f_minus = np.asarray(f(X, theta - e_j), dtype=float).ravel()
        J[:, j] = (f_plus - f_minus) / (2.0 * h_j)

    return J


def _eval_jacobian(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    jacobian_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
    theta: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Return df/dtheta — the analytic Jacobian if supplied, else central
    finite differences.  Centralizes the prefer-analytic-then-numerical
    fallback used by all three nonlinear regressors below."""
    if jacobian_fn is not None:
        return jacobian_fn(X, theta)
    return numerical_jacobian(model_fn, theta, X)


# ---------------------------------------------------------------------------
# Gradient-descent nonlinear regressor
#
# Vanilla GD on SSE/n with an optional multiplicative learning-rate decay.
# Slow but simple — pedagogical baseline; convergence by relative SSE change.
# ---------------------------------------------------------------------------


@dataclass
class GradientDescentRegressor:
    """
    Nonlinear least squares via gradient descent.

    Parameters
    ----------
    model_fn:
        Callable f(X, theta) -> ndarray of shape (n,) returning predictions.
    jacobian_fn:
        Optional analytic Jacobian df/dtheta of shape (n, p).  If None,
        central finite differences are used.
    learning_rate:
        Initial step size.
    max_iter:
        Maximum number of gradient steps.
    tol:
        Relative change in SSE used as convergence criterion.
    decay:
        Multiplicative per-step learning-rate decay.  1.0 means no decay.
    """

    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    jacobian_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    learning_rate: float = 0.01
    max_iter: int = 2000
    tol: float = 1e-8
    decay: float = 1.0

    theta_: Optional[np.ndarray] = field(default=None, init=False)
    loss_history_: Optional[np.ndarray] = field(default=None, init=False)
    converged_: Optional[bool] = field(default=None, init=False)
    n_iter_: Optional[int] = field(default=None, init=False)
    fit_error_: Optional[str] = field(default=None, init=False)

    def fit(
        self, X: np.ndarray, y: np.ndarray, theta_init: np.ndarray
    ) -> "GradientDescentRegressor":
        """
        Fit the model by gradient descent on SSE/n.

        Parameters
        ----------
        X:
            Predictor matrix of shape (n, q) or (n,).
        y:
            Response vector of shape (n,).
        theta_init:
            Initial parameter vector of shape (p,).

        Returns
        -------
        self
        """
        try:
            X = as_2d(X)
            y = np.asarray(y, dtype=float).ravel()
            theta = np.asarray(theta_init, dtype=float).ravel().copy()
            n = len(y)

            losses: list = []
            converged = False
            sse_prev = np.inf

            for t in range(self.max_iter):
                r = y - self.model_fn(X, theta)
                J = _eval_jacobian(self.model_fn, self.jacobian_fn, theta, X)

                grad = -2.0 * (J.T @ r) / n
                lr_t = self.learning_rate * (self.decay ** t)
                theta_new = theta - lr_t * grad

                sse_new = float(np.sum((y - self.model_fn(X, theta_new)) ** 2))
                losses.append(sse_new / n)

                if abs(sse_prev - sse_new) / max(abs(sse_prev), EPS) < self.tol:
                    theta = theta_new
                    converged = True
                    self.n_iter_ = t + 1
                    break

                theta = theta_new
                sse_prev = sse_new
            else:
                self.n_iter_ = self.max_iter

            self.theta_ = theta
            self.loss_history_ = np.array(losses)
            self.converged_ = converged

        except Exception as exc:
            self.theta_ = np.asarray(theta_init, dtype=float).ravel().copy()
            self.loss_history_ = np.array([])
            self.converged_ = False
            self.n_iter_ = 0
            self.fit_error_ = str(exc)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on the original response scale.
        """
        if self.theta_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        return np.asarray(self.model_fn(as_2d(X), self.theta_), dtype=float).ravel()

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact model summary.
        """
        if self.theta_ is None:
            raise RuntimeError("Model must be fit before summary.")
        final_loss = float(self.loss_history_[-1]) if len(self.loss_history_) > 0 else np.nan
        return {
            "method": "gradient_descent",
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "final_loss": final_loss,
            "theta": self.theta_,
            "learning_rate": self.learning_rate,
            "decay": self.decay,
        }


# ---------------------------------------------------------------------------
# Gauss-Newton with self-activating Levenberg-Marquardt damping
#
# Solves  (J^T J + lam * I) * delta = J^T r  each iteration.  lam = 0 gives
# pure Gauss-Newton; rejected steps escalate lam toward gradient-descent
# behavior; accepted steps relax lam back toward 0.  Self-activating, so
# the user doesn't need to tune lam manually.
# ---------------------------------------------------------------------------


@dataclass
class GaussNewtonRegressor:
    """
    Nonlinear least squares via Gauss-Newton with Levenberg-Marquardt damping.

    Parameters
    ----------
    model_fn:
        Callable f(X, theta) -> ndarray of shape (n,).
    jacobian_fn:
        Optional analytic Jacobian.  If None, central finite differences are
        used.
    max_iter:
        Maximum number of iterations.
    tol:
        Relative step-size convergence criterion.
    damping:
        Initial LM damping coefficient lambda.  Set to 0 for pure
        Gauss-Newton; the solver self-activates damping if needed.
    max_damping:
        Upper bound on the damping coefficient.  Reaching this value
        triggers early termination.
    damping_factor:
        Multiplicative factor by which lambda is increased (rejected steps)
        or decreased (accepted steps).
    """

    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    jacobian_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    max_iter: int = 100
    tol: float = 1e-8
    damping: float = 0.0
    max_damping: float = 1e8
    damping_factor: float = 10.0

    theta_: Optional[np.ndarray] = field(default=None, init=False)
    converged_: Optional[bool] = field(default=None, init=False)
    n_iter_: Optional[int] = field(default=None, init=False)
    fit_error_: Optional[str] = field(default=None, init=False)

    def fit(
        self, X: np.ndarray, y: np.ndarray, theta_init: np.ndarray
    ) -> "GaussNewtonRegressor":
        """
        Fit the model by Gauss-Newton with optional LM damping.

        Parameters
        ----------
        X:
            Predictor matrix of shape (n, q) or (n,).
        y:
            Response vector of shape (n,).
        theta_init:
            Initial parameter vector of shape (p,).

        Returns
        -------
        self
        """
        try:
            X = as_2d(X)
            y = np.asarray(y, dtype=float).ravel()
            theta = np.asarray(theta_init, dtype=float).ravel().copy()
            p = len(theta)
            lam = float(self.damping)
            converged = False

            for t in range(self.max_iter):
                r = y - self.model_fn(X, theta)
                J = _eval_jacobian(self.model_fn, self.jacobian_fn, theta, X)

                A = J.T @ J           # (p, p) approximate Hessian
                g = J.T @ r           # (p,)   gradient of -0.5 * SSE

                # Solve (A + lam*I) delta = g; fall back to least-squares if
                # the matrix is singular even after damping.
                try:
                    delta = np.linalg.solve(A + lam * np.eye(p), g)
                except np.linalg.LinAlgError:
                    delta, *_ = np.linalg.lstsq(A + lam * np.eye(p), g, rcond=None)

                theta_new = theta + delta
                sse_new = float(np.sum((y - self.model_fn(X, theta_new)) ** 2))
                sse_old = float(np.sum(r ** 2))

                if sse_new <= sse_old:
                    # Accept step; reduce damping toward Gauss-Newton regime.
                    theta = theta_new
                    lam = max(lam / self.damping_factor, 0.0)
                else:
                    # Reject step; increase damping toward gradient-descent regime.
                    lam = lam * self.damping_factor if lam > 0.0 else 1e-4
                    lam = min(lam, self.max_damping)
                    if lam >= self.max_damping:
                        self.n_iter_ = t + 1
                        break

                if np.linalg.norm(delta) / (np.linalg.norm(theta) + EPS) < self.tol:
                    converged = True
                    self.n_iter_ = t + 1
                    break
            else:
                self.n_iter_ = self.max_iter

            self.theta_ = theta
            self.converged_ = converged

        except Exception as exc:
            self.theta_ = np.asarray(theta_init, dtype=float).ravel().copy()
            self.converged_ = False
            self.n_iter_ = 0
            self.fit_error_ = str(exc)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on the original response scale.
        """
        if self.theta_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        return np.asarray(self.model_fn(as_2d(X), self.theta_), dtype=float).ravel()

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact model summary.
        """
        if self.theta_ is None:
            raise RuntimeError("Model must be fit before summary.")
        return {
            "method": "gauss_newton",
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "theta": self.theta_,
            "damping": self.damping,
        }


# ---------------------------------------------------------------------------
# BFGS quasi-Newton nonlinear regressor
#
# Maintains an inverse-Hessian approximation H of SSE/n; search direction
# is d = -H g, step size alpha by backtracking Armijo line search.
# Inverse-Hessian update (when sy > eps):
#   H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T,
#   with s = alpha d,  y = g_{k+1} - g_k,  rho = 1 / (s @ y).
# ---------------------------------------------------------------------------


@dataclass
class BFGSRegressor:
    """
    Nonlinear least squares via BFGS quasi-Newton optimization.

    This is a pure-numpy implementation; no scipy is required.

    Parameters
    ----------
    model_fn:
        Callable f(X, theta) -> ndarray of shape (n,).
    jacobian_fn:
        Optional analytic Jacobian.  If None, central finite differences are
        used.
    max_iter:
        Maximum number of BFGS iterations.
    tol:
        Gradient-norm convergence criterion.
    c1:
        Armijo sufficient-decrease constant for backtracking line search.
    """

    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    jacobian_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    max_iter: int = 500
    tol: float = 1e-8
    c1: float = 1e-4

    theta_: Optional[np.ndarray] = field(default=None, init=False)
    converged_: Optional[bool] = field(default=None, init=False)
    n_iter_: Optional[int] = field(default=None, init=False)
    fit_error_: Optional[str] = field(default=None, init=False)

    def _objective(
        self, theta: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """SSE / n at theta."""
        r = y - np.asarray(self.model_fn(X, theta), dtype=float).ravel()
        return float(np.sum(r ** 2)) / len(y)

    def _gradient(
        self, theta: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Gradient of SSE/n with respect to theta: -2 J^T r / n."""
        r = y - np.asarray(self.model_fn(X, theta), dtype=float).ravel()
        J = _eval_jacobian(self.model_fn, self.jacobian_fn, theta, X)
        return -2.0 * (J.T @ r) / len(y)

    def fit(
        self, X: np.ndarray, y: np.ndarray, theta_init: np.ndarray
    ) -> "BFGSRegressor":
        """
        Fit the model using BFGS with backtracking Armijo line search.

        Parameters
        ----------
        X:
            Predictor matrix of shape (n, q) or (n,).
        y:
            Response vector of shape (n,).
        theta_init:
            Initial parameter vector of shape (p,).

        Returns
        -------
        self
        """
        try:
            X = as_2d(X)
            y = np.asarray(y, dtype=float).ravel()
            theta = np.asarray(theta_init, dtype=float).ravel().copy()
            p = len(theta)

            H = np.eye(p)          # inverse Hessian approximation
            I_p = np.eye(p)
            converged = False

            for k in range(self.max_iter):
                g = self._gradient(theta, X, y)

                # Convergence check on gradient norm.
                if np.linalg.norm(g) < self.tol:
                    converged = True
                    self.n_iter_ = k
                    break

                # Search direction.
                d = -(H @ g)

                # Backtracking Armijo line search.
                f_k = self._objective(theta, X, y)
                gd = float(g @ d)
                alpha = 1.0
                for _ in range(60):
                    if self._objective(theta + alpha * d, X, y) <= f_k + self.c1 * alpha * gd:
                        break
                    alpha *= 0.5
                else:
                    # Line search failed; keep best alpha found rather than
                    # stopping, which may still make progress.
                    pass

                s = alpha * d
                theta_new = theta + s
                g_new = self._gradient(theta_new, X, y)
                yk = g_new - g
                sy = float(s @ yk)

                # BFGS inverse Hessian update (only when curvature condition holds).
                if sy > EPS:
                    rho = 1.0 / sy
                    A = I_p - rho * np.outer(s, yk)
                    B = I_p - rho * np.outer(yk, s)
                    H = A @ H @ B + rho * np.outer(s, s)
                    # Enforce symmetry to counteract floating-point drift.
                    H = 0.5 * (H + H.T)
                    # Reset H if it degrades numerically.
                    if np.linalg.norm(H) > 1e12:
                        H = np.eye(p)

                # Relative step-size convergence check.
                if np.linalg.norm(s) / (np.linalg.norm(theta) + EPS) < self.tol:
                    theta = theta_new
                    converged = True
                    self.n_iter_ = k + 1
                    break

                theta = theta_new
            else:
                self.n_iter_ = self.max_iter

            self.theta_ = theta
            self.converged_ = converged

        except Exception as exc:
            self.theta_ = np.asarray(theta_init, dtype=float).ravel().copy()
            self.converged_ = False
            self.n_iter_ = 0
            self.fit_error_ = str(exc)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on the original response scale.
        """
        if self.theta_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        return np.asarray(self.model_fn(as_2d(X), self.theta_), dtype=float).ravel()

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact model summary.
        """
        if self.theta_ is None:
            raise RuntimeError("Model must be fit before summary.")
        return {
            "method": "bfgs",
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "theta": self.theta_,
            "c1": self.c1,
        }


# ---------------------------------------------------------------------------
# ZZU hybrid transformation → nonlinear workflow
#
# Three steps:
#   1. Screen      — fit a suite of TransformedOLS models on a validation
#                    split, rank by held-out RMSE on the original scale.
#   2. Warm start  — invert the best linearized model's coefficients to a
#                    nonlinear theta_init via a user-supplied callable, then
#                    refine with direct nonlinear optimization.
#   3. Bias correct — record training residuals; optionally add their mean
#                    to predictions to remove systematic offset.
# ---------------------------------------------------------------------------


def _default_transformation_suite() -> Dict[str, TransformedOLS]:
    """
    Return the default set of TransformedOLS models used for ZZU screening.
    """
    return {
        "identity": TransformedOLS(transform="identity", use_smearing=False),
        "log_smear": TransformedOLS(transform="log", use_smearing=True),
        "reciprocal_smear": TransformedOLS(transform="reciprocal", use_smearing=True),
        "power_0.5_smear": TransformedOLS(transform="power", param=0.5, use_smearing=True),
        "power_2_smear": TransformedOLS(transform="power", param=2.0, use_smearing=True),
        "boxcox_smear": TransformedOLS(transform="boxcox", use_smearing=True),
        "yeojohnson_smear": TransformedOLS(transform="yeojohnson", use_smearing=True),
    }


@dataclass
class ZZUTransformRegressor:
    """
    ZZU hybrid workflow: screen transformations, warm-start nonlinear fitting.

    Parameters
    ----------
    model_fn:
        Callable f(X, theta) -> ndarray of shape (n,) for the nonlinear model.
    coeff_to_init:
        Callable that receives the best-fitting TransformedOLS model after it
        has been re-fit on the full training data and returns an initial
        parameter vector theta_init of shape (p,) for the nonlinear optimizer.
    jacobian_fn:
        Optional analytic Jacobian passed to the nonlinear optimizer.
    nonlinear_method:
        Which optimizer to use after the warm start.  One of
        "gradient_descent", "gauss_newton", or "bfgs".
    transformations:
        Custom dict of TransformedOLS models for Step 1 screening.  If None,
        the default seven-model suite is used.
    val_fraction:
        Fraction of training data held out for screening.  If fewer than ten
        observations are available, screening uses training RMSE instead.
    use_smearing:
        Whether to apply additive bias correction (mean training residual) to
        original-scale predictions.
    nonlinear_kwargs:
        Extra keyword arguments forwarded to the nonlinear regressor's
        constructor (e.g., max_iter, learning_rate).
    seed:
        Seed for the reproducible validation split.
    fallback_theta_init:
        Parameter vector used if coeff_to_init raises an exception.  If None,
        a vector of ones with length inferred from beta_ is used.
    """

    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    coeff_to_init: Callable[["TransformedOLS"], np.ndarray]
    jacobian_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    nonlinear_method: str = "bfgs"
    transformations: Optional[Dict[str, TransformedOLS]] = None
    val_fraction: float = 0.2
    use_smearing: bool = True
    nonlinear_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 0
    fallback_theta_init: Optional[np.ndarray] = None

    best_transform_name_: Optional[str] = field(default=None, init=False)
    best_transform_model_: Optional[TransformedOLS] = field(default=None, init=False)
    nonlinear_regressor_: Optional[object] = field(default=None, init=False)
    screening_table_: Optional[pd.DataFrame] = field(default=None, init=False)
    train_residuals_: Optional[np.ndarray] = field(default=None, init=False)
    theta_init_used_: Optional[np.ndarray] = field(default=None, init=False)
    X_train_: Optional[np.ndarray] = field(default=None, init=False)
    y_train_: Optional[np.ndarray] = field(default=None, init=False)
    fit_error_: Optional[str] = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ZZUTransformRegressor":
        """
        Run the three-step ZZU workflow on training data.

        Parameters
        ----------
        X:
            Predictor matrix of shape (n, q) or (n,).
        y:
            Response vector of shape (n,).

        Returns
        -------
        self
        """
        try:
            X = as_2d(X)
            y = np.asarray(y, dtype=float).ravel()
            self.X_train_ = X.copy()
            self.y_train_ = y.copy()
            n = len(y)

            suite = (
                self.transformations
                if self.transformations is not None
                else _default_transformation_suite()
            )
            kwargs = self.nonlinear_kwargs or {}

            # ------------------------------------------------------------------
            # Step 1: Screen transformations.
            # ------------------------------------------------------------------
            if n >= 10:
                n_val = max(2, int(self.val_fraction * n))
                rng = np.random.default_rng(self.seed)
                idx = rng.permutation(n)
                val_idx = idx[:n_val]
                train_idx = idx[n_val:]
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                use_val = True
            else:
                # Too few observations; skip validation split.
                X_tr, y_tr, X_val, y_val = X, y, X, y
                use_val = False

            rows = []
            for name, tols_model in suite.items():
                try:
                    tols_model.fit(X_tr, y_tr)
                    val_pred = tols_model.predict(X_val)
                    val_rmse = regression_metrics(y_val, val_pred)["rmse"]
                    rows.append({"name": name, "model": tols_model,
                                 "val_rmse": val_rmse, "error": ""})
                except Exception as exc:
                    rows.append({"name": name, "model": tols_model,
                                 "val_rmse": np.inf, "error": str(exc)})

            screening_df = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
            self.screening_table_ = screening_df

            # Re-fit the best model on the full training data before extracting
            # its coefficients, so the warm start reflects all available data.
            best_row = screening_df.iloc[0]
            best_model: TransformedOLS = best_row["model"]
            best_model.fit(X, y)
            self.best_transform_name_ = best_row["name"]
            self.best_transform_model_ = best_model

            # ------------------------------------------------------------------
            # Step 2: Warm start — convert linearized coefficients to theta_init
            # and run direct nonlinear optimization.
            # ------------------------------------------------------------------
            try:
                theta_init = np.asarray(
                    self.coeff_to_init(best_model), dtype=float
                ).ravel()
            except Exception as exc:
                # Fall back gracefully when the user-supplied converter fails.
                self.fit_error_ = f"coeff_to_init failed: {exc}"
                if self.fallback_theta_init is not None:
                    theta_init = np.asarray(self.fallback_theta_init, dtype=float).ravel()
                else:
                    theta_init = np.ones(len(best_model.beta_) - 1, dtype=float)

            self.theta_init_used_ = theta_init.copy()

            _method = self.nonlinear_method
            if _method == "gradient_descent":
                reg = GradientDescentRegressor(
                    model_fn=self.model_fn, jacobian_fn=self.jacobian_fn, **kwargs
                )
            elif _method == "gauss_newton":
                reg = GaussNewtonRegressor(
                    model_fn=self.model_fn, jacobian_fn=self.jacobian_fn, **kwargs
                )
            elif _method == "bfgs":
                reg = BFGSRegressor(
                    model_fn=self.model_fn, jacobian_fn=self.jacobian_fn, **kwargs
                )
            else:
                raise ValueError(
                    f"Unknown nonlinear_method: {_method!r}. "
                    "Choose 'gradient_descent', 'gauss_newton', or 'bfgs'."
                )

            reg.fit(X, y, theta_init)
            self.nonlinear_regressor_ = reg

            # ------------------------------------------------------------------
            # Step 3: Store training residuals for optional bias correction.
            # ------------------------------------------------------------------
            self.train_residuals_ = y - reg.predict(X)

        except Exception as exc:
            self.fit_error_ = str(exc)

        return self

    def predict(
        self, X: np.ndarray, use_smearing: Optional[bool] = None
    ) -> np.ndarray:
        """
        Predict on the original response scale.

        Parameters
        ----------
        X:
            Predictor matrix of shape (m, q) or (m,).
        use_smearing:
            If True, add the mean training residual as an additive bias
            correction.  If None, the instance default is used.

        Returns
        -------
        y_pred : ndarray of shape (m,)
        """
        if self.nonlinear_regressor_ is None:
            raise RuntimeError("Model must be fit before prediction.")

        if use_smearing is None:
            use_smearing = self.use_smearing

        y_hat = self.nonlinear_regressor_.predict(X)

        if use_smearing and self.train_residuals_ is not None:
            bias = float(np.mean(self.train_residuals_))
            y_hat = y_hat + bias

        return y_hat.ravel()

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact model summary.
        """
        if self.nonlinear_regressor_ is None:
            raise RuntimeError("Model must be fit before summary.")

        train_metrics = regression_metrics(
            self.y_train_, self.nonlinear_regressor_.predict(self.X_train_)
        )
        selected_lambda = (
            self.best_transform_model_.selected_param_
            if self.best_transform_model_ is not None
            else np.nan
        )

        return {
            "best_transform": self.best_transform_name_,
            "selected_lambda": selected_lambda,
            "nonlinear_method": self.nonlinear_method,
            "final_theta": self.nonlinear_regressor_.theta_,
            "converged": self.nonlinear_regressor_.converged_,
            "theta_init_used": self.theta_init_used_,
            "train_metrics": train_metrics,
        }


# ---------------------------------------------------------------------------
# Batch evaluator for nonlinear regressors
#
# Mirrors `evaluate_transformed_models`: fits every regressor in a dict,
# captures metrics + diagnostics + convergence flags, and records errors
# instead of crashing on a misbehaving model.
# ---------------------------------------------------------------------------


def evaluate_nonlinear_models(
    X: np.ndarray,
    y: np.ndarray,
    models_dict: Dict[str, Any],
    theta_inits: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Fit and evaluate a dictionary of nonlinear regressors.

    Parameters
    ----------
    X:
        Predictor matrix of shape (n, q) or (n,).
    y:
        Response vector of shape (n,).
    models_dict:
        Dict mapping name to an instance of GradientDescentRegressor,
        GaussNewtonRegressor, or BFGSRegressor.
    theta_inits:
        Dict mapping the same names to initial parameter vectors.

    Returns
    -------
    DataFrame sorted by RMSE (ascending), with columns: model, method,
    converged, n_iter, error, and all regression_metrics /
    residual_diagnostics columns.
    """
    rows = []

    for name, model in models_dict.items():
        try:
            theta_init = theta_inits[name]
            model.fit(X, y, theta_init)
            pred = model.predict(X)
            metrics = regression_metrics(y, pred)
            diags = residual_diagnostics(y, pred)
            row = {
                "model": name,
                "method": type(model).__name__,
                "converged": model.converged_,
                "n_iter": model.n_iter_,
                "error": "",
                **metrics,
                **diags,
            }
        except Exception as exc:
            # Reuse the NaN dicts from the metric helpers (empty-array
            # branches) so the error-row schema cannot drift from the
            # success path.
            row = {
                "model": name,
                "method": type(model).__name__,
                "converged": False,
                "n_iter": 0,
                "error": str(exc),
                **regression_metrics(np.array([]), np.array([])),
                **residual_diagnostics(np.array([]), np.array([])),
            }
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["rmse", "model"], na_position="last").reset_index(drop=True)

