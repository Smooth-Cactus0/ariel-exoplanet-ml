"""
dataset.py — PyTorch Dataset wrapper for Ariel competition data.

Handles two access patterns:
  - Train: returns airs, fgs1, aux, and (optionally) target_mean, target_std
  - Test/inference: returns airs, fgs1, aux only

HDF5 key names confirmed as 'AIRS-CH0' and 'FGS1' from competition docs.
If your local data uses different keys, set ArielDataset._airs_key / _fgs_key.

NOTE: This class requires the competition HDF5 files to be present at data_root.
      Tests use a mock HDF5 to avoid needing real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import preprocess_planet


class ArielDataset(Dataset):
    """
    PyTorch Dataset for Ariel transit photometry.

    Parameters
    ----------
    data_root  : path to competition data root (contains train.hdf5, AuxillaryTable.csv, etc.)
    split      : 'train' or 'test'
    ingress    : transit ingress as fraction of total observation time  (default 0.2)
    egress     : transit egress as fraction of total observation time   (default 0.8)
    bin_size   : temporal binning factor applied during preprocessing   (default 5)
    preprocess : if True, run full preprocessing pipeline; if False, return raw arrays

    Returns (per __getitem__)
    -------------------------
    dict with keys:
        planet_id    : str
        airs         : (n_airs_channels, time) float32 tensor  — channel-first for Conv1d
        fgs1         : (1, time) float32 tensor
        aux          : (9,) float32 tensor  — raw stellar/planetary parameters
        target_mean  : (283,) float32 tensor  — present only for labelled train planets
        target_std   : (283,) float32 tensor  — present only for labelled train planets
    """

    # Competition HDF5 key names — update if data exploration reveals different names
    _airs_key: str = "AIRS-CH0"
    _fgs_key: str = "FGS1"

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        ingress: float = 0.2,
        egress: float = 0.8,
        bin_size: int = 5,
        preprocess: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.ingress = ingress
        self.egress = egress
        self.bin_size = bin_size
        self.preprocess = preprocess

        # Auxiliary table — indexed by planet_id (integer)
        aux_path = self.data_root / "AuxillaryTable.csv"
        self.aux = pd.read_csv(aux_path, index_col=0)
        self.planet_ids: list[str] = list(self.aux.index.astype(str))

        # Labels (train only, ~24 % of planets have ground truth)
        self.labels: Optional[pd.DataFrame] = None
        if split == "train":
            label_path = self.data_root / "QuartilesTable.csv"
            if label_path.exists():
                self.labels = pd.read_csv(label_path, index_col=0)
                # Build a set for O(1) lookup
                self._labelled_ids: set[str] = set(self.labels.index.astype(str))
            else:
                self._labelled_ids = set()
        else:
            self._labelled_ids = set()

        # HDF5 handle — opened lazily to support multiprocessing DataLoader workers
        self._h5_path = self.data_root / f"{split}.hdf5"
        self._h5: Optional[h5py.File] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_h5(self) -> h5py.File:
        """Open the HDF5 file on first access (lazy, worker-safe)."""
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")
        return self._h5

    def _load_raw(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """Load raw AIRS and FGS1 arrays for planet `pid`."""
        h5 = self._open_h5()
        airs = h5[pid][self._airs_key][()].astype(np.float32)  # (time, n_channels)
        fgs1 = h5[pid][self._fgs_key][()].astype(np.float32)   # (time,)
        return airs, fgs1

    def _parse_labels(self, pid: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse QuartilesTable row into (target_mean, target_std).

        The table has columns named like '0_q1', '0_q2', '0_q3', '1_q1', ...
        where the number is the wavelength index and q1/q2/q3 are the 16th,
        50th, and 84th percentiles of the atmospheric posterior.

        Target mean  = q2 (median)
        Target std   = (q3 - q1) / 2  (half-interquartile range ≈ 1-sigma)
        """
        row = self.labels.loc[int(pid)]
        q1 = row.filter(like="_q1").values.astype(np.float32)
        q2 = row.filter(like="_q2").values.astype(np.float32)
        q3 = row.filter(like="_q3").values.astype(np.float32)
        return q2, (q3 - q1) / 2.0

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.planet_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.planet_ids[idx]
        airs_raw, fgs1_raw = self._load_raw(pid)
        aux_row = self.aux.loc[int(pid)].values.astype(np.float32)  # (9,)

        if self.preprocess:
            result = preprocess_planet(
                airs_raw, fgs1_raw, self.ingress, self.egress, self.bin_size
            )
            airs_out = result["airs_norm"].T.astype(np.float32)  # (channels, time)
            fgs1_out = result["fgs1_norm"][None].astype(np.float32)  # (1, time)
        else:
            airs_out = airs_raw.T  # (channels, time)
            fgs1_out = fgs1_raw[None]  # (1, time)

        sample = {
            "planet_id": pid,
            "airs": torch.from_numpy(airs_out),
            "fgs1": torch.from_numpy(fgs1_out),
            "aux": torch.from_numpy(aux_row),
        }

        if pid in self._labelled_ids:
            target_mean, target_std = self._parse_labels(pid)
            sample["target_mean"] = torch.from_numpy(target_mean)
            sample["target_std"] = torch.from_numpy(target_std)

        return sample

    def __del__(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
