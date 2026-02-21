"""Tests for src/preprocessing.py — all functions, 8 test cases."""

import numpy as np
import pytest

from src.preprocessing import (
    bin_time,
    baseline_normalize,
    common_mode_correction,
    extract_transit_depth,
    out_of_transit_mask,
    preprocess_planet,
)


# ---------------------------------------------------------------------------
# out_of_transit_mask
# ---------------------------------------------------------------------------

def test_oot_mask_shape():
    mask = out_of_transit_mask(100)
    assert mask.shape == (100,)
    assert mask.dtype == bool


def test_oot_mask_fractions():
    ingress, egress = 0.2, 0.8
    n = 100
    mask = out_of_transit_mask(n, ingress=ingress, egress=egress)
    t = np.linspace(0.0, 1.0, n)
    # Verify against the exact time values, not integer-index assumptions
    expected = (t < ingress) | (t > egress)
    np.testing.assert_array_equal(mask, expected, err_msg="mask must match (t<ingress)|(t>egress)")


# ---------------------------------------------------------------------------
# baseline_normalize
# ---------------------------------------------------------------------------

def test_baseline_normalize_oot_median_is_one():
    rng = np.random.default_rng(42)
    flux = rng.normal(1000.0, 5.0, size=(100, 50))
    mask = out_of_transit_mask(100)
    norm = baseline_normalize(flux, mask)
    oot_medians = np.median(norm[mask], axis=0)
    np.testing.assert_allclose(oot_medians, 1.0, atol=1e-6)


def test_baseline_normalize_1d_input():
    """baseline_normalize must handle 1-D (time,) input without crashing."""
    rng = np.random.default_rng(0)
    flux1d = rng.normal(500.0, 2.0, size=(100,))
    mask = out_of_transit_mask(100)
    result = baseline_normalize(flux1d, mask)
    assert result.shape == (100,), "1-D output must stay 1-D"
    np.testing.assert_allclose(np.median(result[mask]), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# bin_time
# ---------------------------------------------------------------------------

def test_bin_time_output_shape():
    flux = np.ones((100, 50))
    binned = bin_time(flux, bin_size=5)
    assert binned.shape == (20, 50)


def test_bin_time_correct_average():
    # [0,1,2,...,9] binned by 2 → [0.5, 2.5, 4.5, 6.5, 8.5]
    flux = np.arange(10, dtype=float).reshape(-1, 1)
    binned = bin_time(flux, bin_size=2)
    expected = np.array([0.5, 2.5, 4.5, 6.5, 8.5]).reshape(-1, 1)
    np.testing.assert_allclose(binned, expected)


# ---------------------------------------------------------------------------
# extract_transit_depth
# ---------------------------------------------------------------------------

def test_extract_depth_flat_curve_is_zero():
    """Flat light curve (no transit) must yield depth ≈ 0."""
    flux = np.ones((100, 10))
    mask = out_of_transit_mask(100)
    depth, _ = extract_transit_depth(flux, mask)
    np.testing.assert_allclose(depth, 0.0, atol=1e-10)


def test_extract_depth_known_signal():
    """In-transit flux of 0.99 must produce depth ≈ 0.01."""
    flux = np.ones((100, 1))
    mask = out_of_transit_mask(100)
    flux[~mask] = 0.99
    depth, _ = extract_transit_depth(flux, mask)
    np.testing.assert_allclose(depth, 0.01, atol=1e-6)


# ---------------------------------------------------------------------------
# preprocess_planet (integration)
# ---------------------------------------------------------------------------

def test_preprocess_planet_output_contract(synthetic_planet):
    """preprocess_planet must return the right keys and shapes."""
    airs, fgs1 = synthetic_planet
    result = preprocess_planet(airs, fgs1, bin_size=5)

    expected_keys = {"airs_norm", "fgs1_norm", "transit_depth", "transit_depth_err", "mask_oot"}
    assert set(result.keys()) == expected_keys

    n_time, n_wl = airs.shape
    n_bins = n_time // 5
    assert result["airs_norm"].shape == (n_bins, n_wl), "airs_norm shape mismatch"
    assert result["fgs1_norm"].shape == (n_bins,),      "fgs1_norm shape mismatch"
    assert result["transit_depth"].shape == (n_wl,),    "transit_depth shape mismatch"
    assert result["transit_depth_err"].shape == (n_wl,),"transit_depth_err shape mismatch"
    assert result["mask_oot"].shape == (n_bins,),       "mask_oot shape mismatch"
