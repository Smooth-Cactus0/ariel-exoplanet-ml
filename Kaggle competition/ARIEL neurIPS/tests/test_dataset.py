"""Tests for src/dataset.py — uses mock HDF5 and CSV data."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset import ArielDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_PLANETS = 5
N_TIME = 500
N_AIRS = 356
N_AUX = 9
N_WL = 283  # output wavelengths


def _make_mock_data_root(tmp_path: Path, with_labels: bool = True) -> Path:
    """
    Create a minimal mock competition data directory:
      - train.hdf5  with N_PLANETS planets, each having AIRS-CH0 and FGS1
      - AuxillaryTable.csv  with N_PLANETS rows and N_AUX features
      - QuartilesTable.csv  (optional) with N_WL * 3 columns
    """
    rng = np.random.default_rng(0)

    # HDF5
    h5_path = tmp_path / "train.hdf5"
    planet_ids = [str(i) for i in range(N_PLANETS)]
    with h5py.File(h5_path, "w") as f:
        for pid in planet_ids:
            grp = f.create_group(pid)
            airs = rng.normal(1000.0, 5.0, (N_TIME, N_AIRS)).astype(np.float32)
            fgs1 = rng.normal(1000.0, 3.0, (N_TIME,)).astype(np.float32)
            grp.create_dataset("AIRS-CH0", data=airs)
            grp.create_dataset("FGS1", data=fgs1)

    # AuxillaryTable.csv
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
    aux_df.to_csv(tmp_path / "AuxillaryTable.csv", index=True)

    # QuartilesTable.csv (label columns: 0_q1, 0_q2, 0_q3, 1_q1, ...)
    if with_labels:
        q_cols = []
        for i in range(N_WL):
            q_cols += [f"{i}_q1", f"{i}_q2", f"{i}_q3"]
        q_data = rng.uniform(0.0, 0.02, (N_PLANETS, len(q_cols))).astype(np.float32)
        # Ensure q1 < q2 < q3 per wavelength
        for i in range(N_WL):
            base = i * 3
            vals = np.sort(q_data[:, base : base + 3], axis=1)
            q_data[:, base : base + 3] = vals
        q_df = pd.DataFrame(
            q_data,
            index=[int(pid) for pid in planet_ids],
            columns=q_cols,
        )
        q_df.to_csv(tmp_path / "QuartilesTable.csv", index=True)

    return tmp_path


@pytest.fixture
def mock_root(tmp_path):
    return _make_mock_data_root(tmp_path, with_labels=True)


@pytest.fixture
def mock_root_no_labels(tmp_path):
    return _make_mock_data_root(tmp_path, with_labels=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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
    assert "target_std" not in sample


def test_raw_tensor_shapes(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    assert sample["airs"].shape == (N_AIRS, N_TIME), "AIRS must be channel-first"
    assert sample["fgs1"].shape == (1, N_TIME), "FGS1 must have leading channel dim"
    assert sample["aux"].shape == (N_AUX,)


def test_preprocessed_tensor_shapes(mock_root):
    bin_size = 5
    ds = ArielDataset(mock_root, split="train", preprocess=True, bin_size=bin_size)
    sample = ds[0]
    expected_time = N_TIME // bin_size
    assert sample["airs"].shape == (N_AIRS, expected_time)
    assert sample["fgs1"].shape == (1, expected_time)


def test_target_shapes(mock_root):
    ds = ArielDataset(mock_root, split="train", preprocess=False)
    sample = ds[0]
    assert sample["target_mean"].shape == (N_WL,)
    assert sample["target_std"].shape == (N_WL,)


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
