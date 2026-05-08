"""Single source of truth for random seeds and reproducibility settings.

Why this module exists
----------------------
Every randomness source in the project flows through
``np.random.default_rng(seed)`` — a *local* generator that cannot leak
through ``np.random.*`` global state.  Default seeds were historically
scattered across three files (`toy_data.py` generators, `run_comparison.py`
benchmark loop, `transformation_algorithms.py` ZZU validation split).

This module centralizes the defaults so a reader of the writeup can audit
"what random state was used to generate figure X" by looking in one
place.  Importing modules pull `N_SEEDS`, `TEST_FRACTION`, etc. from
here rather than redefining them locally.

Reproducibility contract
------------------------
- All randomness uses ``np.random.default_rng(seed)`` — no
  ``np.random.seed(...)`` (global), no ``random.random()``.
- Repeated runs of any benchmark with the same `seed` produce
  bit-identical results (verified by `tests/test_reproducibility.py`).
- The `seed_everything()` helper below is purely defensive — it sets
  global state for any third-party library that might rely on it.
  Project code does not require it.
"""

from __future__ import annotations

import os
import random as _stdlib_random

import numpy as np


#: Number of repeated train/test splits used by every benchmark in
#: run_comparison.py, cost_analysis.py, and zzu_inner_method_comparison.py.
#: Seeds 0..N_SEEDS-1 are passed to ``train_test_split_arrays``.
N_SEEDS: int = 10

#: Fraction of each dataset held out for testing in repeated-split benchmarks.
TEST_FRACTION: float = 0.2

#: Default seed for ad-hoc utilities (e.g. one-off splits in scripts).
DEFAULT_SEED: int = 123

#: Seed used by ``ZZUTransformRegressor`` for its internal validation split
#: during the screening step.  Kept distinct from DEFAULT_SEED so the screen
#: doesn't collide with an outer split that happens to use the same seed.
ZZU_VALIDATION_SEED: int = 0

#: Per-dataset seeds, synchronized with the defaults in toy_data.py.
DATASET_SEEDS: dict[str, int] = {
    "exponential_multiplicative": 101,
    "exponential_additive":       102,
    "michaelis_menten":           103,
    "logistic_growth":            104,
    "multivariable_nonlinear":    105,
}


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a fresh ``np.random.Generator`` seeded with `seed`.

    Use this in any new code that needs randomness, never do
    ``np.random.normal(...)`` etc., which read from the global state.
    """
    return np.random.default_rng(seed)


def seed_everything(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Defensively seed every PRNG that any library might touch.

    The project itself does not need this — every random draw uses a
    local Generator.  Call this from notebooks, ad-hoc scripts, or any
    code path that may pull in third-party libraries that rely on
    ``np.random.*`` or the stdlib ``random`` module.

    Returns a fresh ``np.random.Generator`` so the caller has an
    isolated handle for explicit-RNG code.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    _stdlib_random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
