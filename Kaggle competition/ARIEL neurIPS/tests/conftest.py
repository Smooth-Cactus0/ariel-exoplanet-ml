"""Shared pytest fixtures for the Ariel project test suite."""
import numpy as np
import pytest


@pytest.fixture
def synthetic_planet():
    """Returns (airs, fgs1) arrays for a synthetic 500-timestep transit observation."""
    rng = np.random.default_rng(42)
    airs = rng.normal(1000.0, 5.0, size=(500, 356)).astype(np.float32)
    fgs1 = rng.normal(1000.0, 3.0, size=(500,)).astype(np.float32)
    # Inject a synthetic transit (20% to 80% of time)
    n = 500
    ingress_idx = int(0.2 * n)
    egress_idx = int(0.8 * n)
    airs[ingress_idx:egress_idx] *= 0.99   # 1% transit depth
    fgs1[ingress_idx:egress_idx] *= 0.99
    return airs, fgs1
