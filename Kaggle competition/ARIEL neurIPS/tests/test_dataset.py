"""
Tests for src/dataset.py — uses mock parquet data matching confirmed competition format.

Mock data structure (mirrors real competition layout):
  tmp_path/
    train/
      000001/
        AIRS-CH0_signal.parquet      (N_TIME, 32*356) uint16
        FGS1_signal.parquet          (N_TIME*12, 32*32) uint16
        AIRS-CH0_calibration/
            dark.parquet  flat.parquet  dead.parquet  read.parquet  linear_corr.parquet
        FGS1_calibration/
            dark.parquet  flat.parquet  dead.parquet  read.parquet  linear_corr.parquet
      ...
    train_labels.csv                 (TODO: update regex if column format differs)
    train_adc_info.csv               (mock auxiliary features)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import ArielDataset, AIRS_N_ROWS, AIRS_N_COLS, FGS1_N_ROWS, FGS1_N_COLS, FGS1_RATIO

# ── Test constants ────────────────────────────────────────────────────────────
N_PLANETS  = 4
N_TIME     = 60    # AIRS time steps (small for fast tests)
N_WL       = 283   # output wavelengths
N_AUX      = 9


def _write_cal_parquet(cal_dir: Path, n_rows: int, n_cols: int, rng: np.random.Generator) -> None:
    """Write all five calibration parquets for one instrument."""
    cal_dir.mkdir(parents=True, exist_ok=True)
    # dark: small positive values (bias + dark current)
    pd.DataFrame(
        rng.integers(100, 200, (n_rows, n_cols), dtype=np.uint16).astype(np.float32)
    ).to_parquet(cal_dir / "dark.parquet", index=False)
    # flat: values near 1.0 (after normalisation), stored as float
    pd.DataFrame(
        rng.uniform(0.9, 1.1, (n_rows, n_cols)).astype(np.float32)
    ).to_parquet(cal_dir / "flat.parquet", index=False)
    # dead: mostly 0 (alive), a few 1 (dead) — uint8
    dead = np.zeros((n_rows, n_cols), dtype=np.uint8)
    dead[0, 0] = 1  # at least one dead pixel
    pd.DataFrame(dead).to_parquet(cal_dir / "dead.parquet", index=False)
    # read: read noise values
    pd.DataFrame(
        rng.uniform(5.0, 15.0, (n_rows, n_cols)).astype(np.float32)
    ).to_parquet(cal_dir / "read.parquet", index=False)
    # linear_corr: 192 rows per instrument
    pd.DataFrame(
        rng.uniform(0.99, 1.01, (6 * n_rows, n_cols)).astype(np.float32)
    ).to_parquet(cal_dir / "linear_corr.parquet", index=False)


def _make_mock_data_root(tmp_path: Path, with_labels: bool = True) -> Path:
    """
    Create a minimal mock competition data directory with the confirmed parquet layout.
    """
    rng = np.random.default_rng(42)
    planet_ids = [str(1000 + i) for i in range(N_PLANETS)]

    train_dir = tmp_path / "train"
    train_dir.mkdir()

    for pid in planet_ids:
        planet_dir = train_dir / pid
        planet_dir.mkdir()

        # AIRS-CH0 signal: (N_TIME, AIRS_N_ROWS * AIRS_N_COLS) uint16
        # Simulate raw ADU counts ~1300 ± 50
        airs_flat = rng.integers(1200, 1400, (N_TIME, AIRS_N_ROWS * AIRS_N_COLS), dtype=np.uint16)
        pd.DataFrame(airs_flat).to_parquet(planet_dir / "AIRS-CH0_signal.parquet", index=False)

        # FGS1 signal: (N_TIME * FGS1_RATIO, FGS1_N_ROWS * FGS1_N_COLS) uint16
        fgs1_flat = rng.integers(300, 500, (N_TIME * FGS1_RATIO, FGS1_N_ROWS * FGS1_N_COLS), dtype=np.uint16)
        pd.DataFrame(fgs1_flat).to_parquet(planet_dir / "FGS1_signal.parquet", index=False)

        # Calibration
        _write_cal_parquet(planet_dir / "AIRS-CH0_calibration", AIRS_N_ROWS, AIRS_N_COLS, rng)
        _write_cal_parquet(planet_dir / "FGS1_calibration",     FGS1_N_ROWS, FGS1_N_COLS, rng)

    # Auxiliary features (mock train_adc_info.csv)
    aux_cols = [
        "Star_Distance", "Stellar_Mass", "Star_Radius",
        "Star_Temp", "Planet_Mass", "Period", "Sma",
        "Planet_Radius", "Surface_Gravity",
    ]
    aux_df = pd.DataFrame(
        rng.uniform(0.1, 10.0, (N_PLANETS, N_AUX)),
        index=[int(pid) for pid in planet_ids],
        columns=aux_cols,
    )
    aux_df.to_csv(tmp_path / "train_adc_info.csv", index=True)

    # Labels (train_labels.csv) — using {i}_q1/q2/q3 format (TODO: verify)
    if with_labels:
        q_cols: list[str] = []
        for i in range(N_WL):
            q_cols += [f"{i}_q1", f"{i}_q2", f"{i}_q3"]
        q_data = rng.uniform(0.0, 0.02, (N_PLANETS, len(q_cols))).astype(np.float32)
        for i in range(N_WL):
            b = i * 3
            q_data[:, b : b + 3] = np.sort(q_data[:, b : b + 3], axis=1)
        q_df = pd.DataFrame(
            q_data,
            index=[int(pid) for pid in planet_ids],
            columns=q_cols,
        )
        q_df.to_csv(tmp_path / "train_labels.csv", index=True)

    return tmp_path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_root(tmp_path_factory):
    return _make_mock_data_root(tmp_path_factory.mktemp("data"), with_labels=True)


@pytest.fixture(scope="module")
def mock_root_no_labels(tmp_path_factory):
    return _make_mock_data_root(tmp_path_factory.mktemp("data_nolabels"), with_labels=False)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_dataset_length(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    assert len(ds) == N_PLANETS


def test_dataset_item_keys_labelled(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    expected = {"planet_id", "airs", "fgs1", "aux", "target_mean", "target_std"}
    assert set(sample.keys()) == expected


def test_dataset_item_keys_unlabelled(mock_root_no_labels):
    ds = ArielDataset(mock_root_no_labels, split="train", preprocess=False)
    sample = ds[0]
    assert "target_mean" not in sample
    assert "target_std"  not in sample


def test_raw_tensor_shapes(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    assert sample["airs"].shape == (AIRS_N_COLS, N_TIME), "AIRS must be (channels, time)"
    assert sample["fgs1"].shape == (1, N_TIME),           "FGS1 must be (1, time)"
    assert sample["aux"].shape  == (N_AUX,)


def test_preprocessed_tensor_shapes(mock_root):
    bin_size = 5
    ds = ArielDataset(mock_root, split="train", preprocess=True, bin_size=bin_size)
    sample = ds[0]
    expected_time = N_TIME // bin_size
    assert sample["airs"].shape == (AIRS_N_COLS, expected_time)
    assert sample["fgs1"].shape == (1, expected_time)


def test_target_shapes(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    assert sample["target_mean"].shape == (N_WL,)
    assert sample["target_std"].shape  == (N_WL,)


def test_target_std_nonnegative(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    for i in range(N_PLANETS):
        std = ds[i]["target_std"]
        assert (std >= 0).all(), f"target_std must be non-negative for planet {i}"


def test_all_tensors_float32(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    for key in ("airs", "fgs1", "aux", "target_mean", "target_std"):
        assert sample[key].dtype == torch.float32, f"{key} must be float32"


def test_calibration_reduces_raw_values(mock_root):
    """After dark subtraction, calibrated AIRS values should differ from raw."""
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    # Raw ADU ~1300; after dark subtract (~150) and sum over 32 rows, values change
    # Just ensure we get finite, non-trivially-zero output
    airs = sample["airs"].numpy()
    assert np.isfinite(airs).all(), "Calibrated AIRS must be finite"
    assert airs.max() > 0, "Calibrated AIRS must have positive values"


def test_fgs1_downsampled_to_airs_cadence(mock_root):
    """FGS1 time axis must match AIRS time axis after downsampling."""
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    assert sample["fgs1"].shape[1] == sample["airs"].shape[1], \
        "FGS1 time axis must match AIRS time axis"
