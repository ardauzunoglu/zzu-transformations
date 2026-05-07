"""Shared pytest fixtures for the ZZU test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable so `import transformation_algorithms` works
# regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def rng():
    """Reproducible RNG.  Use this in any test that needs randomness."""
    return np.random.default_rng(20260507)


@pytest.fixture
def linear_data():
    """Exactly linear: y = 2 + 3·x with no noise (50 points)."""
    x = np.linspace(0.0, 10.0, 50)
    y = 2.0 + 3.0 * x
    return x.reshape(-1, 1), y


@pytest.fixture
def linear_data_noisy(rng):
    """y = 2 + 3·x + N(0, 0.5²), 100 points."""
    x = np.linspace(0.0, 10.0, 100)
    y = 2.0 + 3.0 * x + rng.normal(0.0, 0.5, size=x.size)
    return x.reshape(-1, 1), y


@pytest.fixture
def exp_mult_data(rng):
    """y = a·exp(b·x)·η with log η ~ N(0, 0.1²); 100 points; a=2, b=0.7.

    Best-case for log-linearization: log y is linear in x with Gaussian noise.
    """
    n = 100
    x = np.linspace(0.0, 5.0, n)
    eta = rng.lognormal(mean=0.0, sigma=0.1, size=n)
    y = 2.0 * np.exp(0.7 * x) * eta
    return x.reshape(-1, 1), y, (2.0, 0.7)


@pytest.fixture
def exp_model_fn():
    """f(X, θ) = θ₀ · exp(θ₁ · X[:, 0]).  Vectorized."""
    return lambda X, t: t[0] * np.exp(t[1] * X[:, 0])


@pytest.fixture
def exp_jacobian_fn():
    """Analytic Jacobian for y = a·exp(b·x):
       ∂f/∂a = exp(b·x);  ∂f/∂b = a·x·exp(b·x).
    """
    def jac(X, t):
        a, b = t
        e = np.exp(b * X[:, 0])
        return np.column_stack([e, a * X[:, 0] * e])
    return jac
