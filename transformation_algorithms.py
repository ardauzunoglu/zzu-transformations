#!/usr/bin/env python
# coding: utf-8

# # Transformation Algorithms for ZZU Regression Experiments
# 
# This notebook implements the transformation-based regression algorithms discussed in the related work:
# 
# - Classical relationship-specific transforms: identity, logarithmic, reciprocal, and power-scale transforms;
# - Box-Cox transformations for strictly positive responses;
# - Yeo-Johnson transformations for responses that may include zero or negative values;
# - Maximum-likelihood selection of transformation parameter $\lambda$;
# - Ordinary least squares on the transformed scale;
# - Inverse prediction on the original scale;
# - Duan-style smearing correction for retransformation bias;
# - Residual and prediction diagnostics for transformation screening.

# ## 1. Setup
# 
# The implementation below intentionally uses a small dependency set: `numpy`, `pandas`, and `matplotlib`.
# 
# The central design choice is that every transformation has a **forward map**, an **inverse map**, and, where needed, a **log-Jacobian correction** used for likelihood-based parameter selection.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Tuple, Any

np.set_printoptions(precision=4, suppress=True)

# Global numerical tolerance used throughout the notebook.
EPS = 1e-12


# ## 2. Linear algebra helpers
# 
# For a response transformation $z = T(y)$, the transformed regression model is
# 
# $$
# z_i = \beta_0 + x_i^\top \beta + \varepsilon_i.
# $$
# 
# We fit this transformed model using ordinary least squares.

# In[2]:


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


# ## 3. Basic transformation families
# 
# ### Identity
# 
# $$
# T(y) = y, \qquad T^{-1}(z) = z.
# $$
# 
# ### Logarithmic
# 
# $$
# T(y) = \log y, \qquad T^{-1}(z) = e^z.
# $$
# 
# This is appropriate only for positive responses.
# 
# ### Reciprocal
# 
# $$
# T(y) = \frac{1}{y}, \qquad T^{-1}(z) = \frac{1}{z}.
# $$
# 
# This is fragile near zero, so the implementation checks for unsafe values.
# 
# ### Power-scale
# 
# For a fixed power $p$,
# 
# $$
# T_p(y) =
# \begin{cases}
# \log y, & p = 0, \\
# y^p, & p \ne 0,
# \end{cases}
# \qquad
# T_p^{-1}(z) =
# \begin{cases}
# e^z, & p = 0, \\
# z^{1/p}, & p \ne 0.
# \end{cases}
# $$
# 
# This implementation uses the positive-response version of the power transform.

# In[3]:


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


# ## 4. Box-Cox transformation
# 
# The Box-Cox family is defined for strictly positive responses:
# 
# $$
# T_\lambda(y) =
# \begin{cases}
# \dfrac{y^\lambda - 1}{\lambda}, & \lambda \ne 0, \\
# \log y, & \lambda = 0.
# \end{cases}
# $$
# 
# The inverse transformation is
# 
# $$
# T_\lambda^{-1}(z) =
# \begin{cases}
# (\lambda z + 1)^{1/\lambda}, & \lambda \ne 0, \\
# e^z, & \lambda = 0.
# \end{cases}
# $$
# 
# For transformed OLS, a common profile log-likelihood objective is
# 
# $$
# \ell(\lambda)
# =
# -\frac{n}{2}\log\left(\frac{\operatorname{SSE}_\lambda}{n}\right)
# +
# (\lambda - 1)\sum_{i=1}^n \log y_i,
# $$
# 
# where $\operatorname{SSE}_\lambda$ is the residual sum of squares after fitting OLS to $T_\lambda(y)$.

# In[4]:


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


# ## 5. Yeo-Johnson transformation
# 
# The Yeo-Johnson family extends power transformations to all real responses:
# 
# $$
# T_\lambda(y)=
# \begin{cases}
# \dfrac{(y+1)^\lambda - 1}{\lambda}, & y \ge 0,\ \lambda \ne 0,\\
# \log(y+1), & y \ge 0,\ \lambda = 0,\\
# -\dfrac{(1-y)^{2-\lambda} - 1}{2-\lambda}, & y < 0,\ \lambda \ne 2,\\
# -\log(1-y), & y < 0,\ \lambda = 2.
# \end{cases}
# $$
# 
# Its log-Jacobian contribution is
# 
# $$
# \log |J_\lambda(y_i)| =
# \begin{cases}
# (\lambda - 1)\log(1+y_i), & y_i \ge 0,\\
# (1-\lambda)\log(1-y_i), & y_i < 0.
# \end{cases}
# $$

# In[5]:


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


# ## 6. Profile likelihood for selecting $\lambda$
# 
# For Box-Cox and Yeo-Johnson, this notebook chooses $\lambda$ by maximizing the transformed-regression profile log-likelihood over a grid.
# 
# This grid-search strategy is slower than continuous optimization but transparent, robust, and easy to audit.

# In[7]:


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


# ## 7. Duan-style smearing correction
# 
# A transformed-scale model predicts
# 
# $$
# \widehat{z}(x) = \widehat{\beta}_0 + x^\top \widehat{\beta}.
# $$
# 
# A naive inverse prediction is
# 
# $$
# \widehat{y}_{\text{naive}}(x) = T^{-1}(\widehat{z}(x)).
# $$
# 
# However, because nonlinear inverse transformations do not generally commute with expectation,
# 
# $$
# T^{-1}\left(E[T(Y)\mid X=x]\right)
# \ne
# E[Y\mid X=x].
# $$
# 
# A generalized smearing correction estimates the original-scale conditional mean by averaging over transformed-scale residuals:
# 
# $$
# \widehat{y}_{\text{smear}}(x)
# =
# \frac{1}{n}
# \sum_{i=1}^n
# T^{-1}\left(\widehat{z}(x) + \widehat{\varepsilon}_i\right).
# $$
# 
# For the log transform, this reduces to the familiar multiplicative correction
# 
# $$
# \widehat{y}_{\text{smear}}(x)
# =
# e^{\widehat{z}(x)}
# \cdot
# \frac{1}{n}\sum_{i=1}^n e^{\widehat{\varepsilon}_i}.
# $$

# In[8]:


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


# ## 8. Unified transformed OLS model
# 
# The `TransformedOLS` class provides a consistent interface:
# 
# ```python
# model = TransformedOLS(transform="boxcox", use_smearing=True)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# ```
# 
# Supported transformations:
# 
# - `"identity"`
# - `"log"`
# - `"reciprocal"`
# - `"power"` with `param=<power>`
# - `"boxcox"` with fixed or automatically selected `lambda_`
# - `"yeojohnson"` with fixed or automatically selected `lambda_`

# In[9]:


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
        """
        Apply the model's response transformation.
        """
        t = self.transform
        p = self.selected_param_

        if t == "identity":
            return identity_forward(y)
        if t == "log":
            return log_forward(y)
        if t == "reciprocal":
            return reciprocal_forward(y)
        if t == "power":
            return power_forward(y, p)
        if t == "boxcox":
            return boxcox_forward(y, p)
        if t == "yeojohnson":
            return yeojohnson_forward(y, p)

        raise ValueError(f"Unknown transform: {t!r}")

    def _inverse(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the inverse of the model's response transformation.
        """
        t = self.transform
        p = self.selected_param_

        if t == "identity":
            return identity_inverse(z)
        if t == "log":
            return log_inverse(z)
        if t == "reciprocal":
            return reciprocal_inverse(z)
        if t == "power":
            return power_inverse(z, p)
        if t == "boxcox":
            return boxcox_inverse(z, p)
        if t == "yeojohnson":
            return yeojohnson_inverse(z, p)

        raise ValueError(f"Unknown transform: {t!r}")

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


# ## 9. Metrics and Diagnostics
# 
# Transformation screening should not rely only on visual straightening. The helper functions below report original-scale predictive accuracy and simple residual diagnostics.
# 
# The main metrics are
# 
# $$
# \operatorname{RMSE}
# =
# \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i-\widehat{y}_i)^2},
# $$
# 
# $$
# \operatorname{MAE}
# =
# \frac{1}{n}\sum_{i=1}^n |y_i-\widehat{y}_i|,
# $$
# 
# and
# 
# $$
# R^2
# =
# 1-\frac{\sum_i (y_i-\widehat{y}_i)^2}
# {\sum_i (y_i-\bar{y})^2}.
# $$

# In[10]:


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
            row = {
                "model": name,
                "transform": model.transform,
                "selected_param_or_lambda": np.nan,
                "error": str(exc),
                "n_valid": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "r2": np.nan,
                "residual_mean": np.nan,
                "residual_std": np.nan,
                "residual_skew": np.nan,
                "residual_excess_kurtosis": np.nan,
                "corr_abs_resid_fitted": np.nan,
            }
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["rmse", "model"], na_position="last").reset_index(drop=True)


# ## 10. Nonlinear regression utilities
#
# Direct nonlinear least squares minimizes
#
#   SSE(theta) = sum_i (y_i - f(x_i, theta))^2
#
# over the parameter vector theta.  The gradient and Jacobian are central to
# all three optimizers below.  When an analytic Jacobian is unavailable,
# central finite differences provide a reliable numerical approximation.


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


# ## 11. Gradient descent nonlinear regressor
#
# Vanilla gradient descent updates theta by stepping opposite the gradient of
# SSE/n.  An optional multiplicative learning-rate decay reduces the step
# size over iterations to aid convergence near the minimum.


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
                if self.jacobian_fn is not None:
                    J = self.jacobian_fn(X, theta)
                else:
                    J = numerical_jacobian(self.model_fn, theta, X)

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


# ## 12. Gauss-Newton / Levenberg-Marquardt nonlinear regressor
#
# Gauss-Newton linearizes the residuals around the current theta and solves
# the resulting linear system for the update delta:
#
#   (J^T J + lambda * I) delta = J^T r
#
# When lambda = 0 this is pure Gauss-Newton.  If a step increases SSE, the
# damping coefficient lambda is automatically activated and increased,
# producing the Levenberg-Marquardt (LM) behavior.  Accepted steps reduce
# lambda back toward zero, recovering the faster Gauss-Newton convergence.


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
                if self.jacobian_fn is not None:
                    J = self.jacobian_fn(X, theta)
                else:
                    J = numerical_jacobian(self.model_fn, theta, X)

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


# ## 13. BFGS quasi-Newton nonlinear regressor
#
# BFGS maintains an approximation H to the inverse Hessian of SSE/n and uses
# it to compute a search direction d = -H g.  After each accepted step the
# inverse Hessian is updated via the rank-two formula:
#
#   H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
#
# where s = alpha*d, y = g_{k+1} - g_k, and rho = 1 / (s^T y).
#
# The step size alpha is found by backtracking Armijo line search.


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
        if self.jacobian_fn is not None:
            J = self.jacobian_fn(X, theta)
        else:
            J = numerical_jacobian(self.model_fn, theta, X)
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


# ## 14. ZZU hybrid transformation-to-nonlinear workflow
#
# The ZZU approach combines the two paradigms in three steps:
#
#   1. Screen — fit a suite of TransformedOLS models on a validation split and
#      rank them by RMSE on held-out data.
#   2. Warm start — convert the best linearized model's coefficients into
#      initial nonlinear parameters via a user-supplied callable, then refine
#      with direct nonlinear optimization.
#   3. Bias correction — store training residuals; optionally subtract their
#      mean from predictions to remove systematic offset.


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


# ## 15. Nonlinear model evaluation helper
#
# Mirrors evaluate_transformed_models: fit every regressor in a dict, record
# metrics and diagnostics, and return a sorted DataFrame.  Models that raise
# are recorded with an error message rather than stopping the evaluation.


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
            row = {
                "model": name,
                "method": type(model).__name__,
                "converged": False,
                "n_iter": 0,
                "error": str(exc),
                "n_valid": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "r2": np.nan,
                "residual_mean": np.nan,
                "residual_std": np.nan,
                "residual_skew": np.nan,
                "residual_excess_kurtosis": np.nan,
                "corr_abs_resid_fitted": np.nan,
            }
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["rmse", "model"], na_position="last").reset_index(drop=True)

