"""
preprocessing.py — Ariel transit light curve preprocessing pipeline.

Pipeline (applied in order):
  1. out_of_transit_mask        — identify in/out-of-transit time steps
  2. baseline_normalize         — divide each channel by its OOT median
  3. common_mode_correction     — remove correlated systematics across all channels
  4. bin_time                   — temporal binning to boost SNR
  5. extract_transit_depth      — weighted mean flux deficit in transit

All functions are pure (no side effects) and operate on numpy arrays.
Designed to be readable by astronomers unfamiliar with ML pipelines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def out_of_transit_mask(
    n_time: int,
    ingress: float = 0.2,
    egress: float = 0.8,
) -> np.ndarray:
    """
    Boolean mask identifying out-of-transit (OOT) time steps.

    Returns True where the planet is NOT in front of the star.
    Assumes a normalised time axis [0, 1] where the transit window
    spans [ingress, egress].  In practice, ingress/egress fractions
    are derived from the orbital period and transit duration.

    Parameters
    ----------
    n_time  : number of time steps in the observation
    ingress : fraction of total time at which transit begins  (default 0.2)
    egress  : fraction of total time at which transit ends    (default 0.8)

    Returns
    -------
    mask : (n_time,) bool — True = out of transit
    """
    t = np.linspace(0.0, 1.0, n_time)
    return (t < ingress) | (t > egress)


def baseline_normalize(
    flux: np.ndarray,
    mask_oot: np.ndarray,
) -> np.ndarray:
    """
    Normalise each spectral channel by its out-of-transit median.

    After normalisation the out-of-transit flux is dimensionless and
    centred near 1.0 for every wavelength channel independently.
    A transit signal then appears as a dip below 1.0 during ingress–egress.

    Parameters
    ----------
    flux     : (time, wavelength) — raw detector counts
    mask_oot : (time,) bool — True = out of transit

    Returns
    -------
    normalised : (time, wavelength) — dimensionless flux, OOT median ≈ 1.0
    """
    if flux.ndim == 1:
        flux = flux[:, None]
        squeeze = True
    else:
        squeeze = False

    baseline = np.median(flux[mask_oot], axis=0, keepdims=True)  # (1, wavelength)
    baseline = np.where(baseline == 0.0, 1.0, baseline)          # guard /0
    result = flux / baseline
    return result[:, 0] if squeeze else result


def common_mode_correction(
    flux_norm: np.ndarray,
    mask_oot: np.ndarray,
) -> np.ndarray:
    """
    Remove correlated systematics shared across all wavelength channels.

    The "common mode" is the mean light curve collapsed over the wavelength
    axis.  Instrument-level or stellar-variability trends that affect every
    channel equally (e.g. pointing jitter, telescope breathing) are captured
    here and divided out, leaving wavelength-specific signals intact.

    Parameters
    ----------
    flux_norm : (time, wavelength) — baseline-normalised flux
    mask_oot  : (time,) bool — out-of-transit mask

    Returns
    -------
    corrected : (time, wavelength)
    """
    common = flux_norm.mean(axis=1, keepdims=True)  # (time, 1)
    oot_level = common[mask_oot].mean()
    common = common / oot_level                      # normalise common mode to 1.0
    return flux_norm / common


def bin_time(
    flux: np.ndarray,
    bin_size: int,
) -> np.ndarray:
    """
    Average every `bin_size` consecutive time steps (temporal binning).

    Binning reduces photon-noise by sqrt(bin_size) at the cost of time
    resolution.  Trailing frames that do not fill a complete bin are dropped.

    Parameters
    ----------
    flux     : (time, ...) — any shape whose first axis is time
    bin_size : number of frames to co-add per output bin

    Returns
    -------
    binned : (time // bin_size, ...) — same shape except first axis
    """
    if bin_size <= 1:
        return flux
    n_time = flux.shape[0]
    n_bins = n_time // bin_size
    trimmed = flux[: n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size, *flux.shape[1:]).mean(axis=1)


def extract_transit_depth(
    flux_norm: np.ndarray,
    mask_oot: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the transit depth per wavelength channel.

    Transit depth is defined as:
        depth[λ] = 1 − <flux_norm[in-transit, λ]>

    This is equivalent to (Rp/Rs)² in the flat-spectrum approximation.
    The uncertainty is the standard error of the in-transit flux mean.

    Parameters
    ----------
    flux_norm : (time, wavelength) — baseline-normalised, corrected flux
    mask_oot  : (time,) bool — True = out of transit
    weights   : (time,) optional per-timestep inverse-variance weights

    Returns
    -------
    depth     : (wavelength,) — transit depth, fractional units
    depth_err : (wavelength,) — 1-sigma uncertainty on depth
    """
    mask_it = ~mask_oot
    if not mask_it.any():
        n_wl = flux_norm.shape[1] if flux_norm.ndim > 1 else 1
        return np.zeros(n_wl), np.full(n_wl, np.inf)

    in_transit = flux_norm[mask_it]  # (n_in, wavelength)

    if weights is not None:
        w = weights[mask_it]
        w = w / w.sum()              # normalise
        depth = 1.0 - np.average(in_transit, axis=0, weights=w)
    else:
        depth = 1.0 - in_transit.mean(axis=0)

    n = mask_it.sum()
    depth_err = (1.0 - in_transit).std(axis=0) / np.sqrt(n)

    return depth, depth_err


def preprocess_planet(
    airs: np.ndarray,
    fgs1: np.ndarray,
    ingress: float = 0.2,
    egress: float = 0.8,
    bin_size: int = 5,
) -> dict:
    """
    Full preprocessing pipeline for a single planet observation.

    Applies baseline normalisation → common-mode correction → temporal
    binning to both AIRS-CH0 (multi-channel IR spectrometer) and FGS1
    (single-channel visible photometer), then extracts the transit depth
    spectrum from the binned AIRS data.

    Parameters
    ----------
    airs     : (time, n_airs_channels) — raw AIRS-CH0 detector counts
    fgs1     : (time,) — raw FGS1 detector counts
    ingress  : transit ingress as fraction of total observation time
    egress   : transit egress as fraction of total observation time
    bin_size : number of time frames to average per output bin

    Returns
    -------
    dict with keys:
        airs_norm        : (time//bin_size, n_airs_channels) normalised, binned AIRS
        fgs1_norm        : (time//bin_size,) normalised, binned FGS1
        transit_depth    : (n_airs_channels,) transit depth per channel
        transit_depth_err: (n_airs_channels,) 1-sigma uncertainty
        mask_oot         : (time//bin_size,) out-of-transit boolean mask
    """
    n_time = airs.shape[0]
    mask_oot = out_of_transit_mask(n_time, ingress, egress)

    # --- AIRS pipeline ---
    airs_norm = baseline_normalize(airs, mask_oot)
    airs_norm = common_mode_correction(airs_norm, mask_oot)
    airs_binned = bin_time(airs_norm, bin_size)
    mask_binned = bin_time(mask_oot.astype(np.float32), bin_size) > 0.5

    # --- FGS1 pipeline ---
    fgs1_norm = baseline_normalize(fgs1, mask_oot)   # handles 1-D via squeeze logic
    fgs1_binned = bin_time(fgs1_norm, bin_size)

    # --- Transit depth extraction ---
    depth, depth_err = extract_transit_depth(airs_binned, mask_binned)

    return {
        "airs_norm": airs_binned,
        "fgs1_norm": fgs1_binned,
        "transit_depth": depth,
        "transit_depth_err": depth_err,
        "mask_oot": mask_binned,
    }
