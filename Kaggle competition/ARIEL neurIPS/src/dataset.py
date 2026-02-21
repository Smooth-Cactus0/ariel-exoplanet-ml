"""
dataset.py — PyTorch Dataset wrapper for Ariel competition data.

Data format (confirmed from Kaggle exploration 2026-02-21):
  {data_root}/{split}/{planet_id}/
      AIRS-CH0_signal.parquet      (n_time, 32*356) uint16 — raw detector frames
      FGS1_signal.parquet          (n_time_fgs, 32*32) uint16 — raw detector frames (12× AIRS rate)
      AIRS-CH0_calibration/
          dark.parquet             (32, 356) — dark current per pixel
          flat.parquet             (32, 356) — flat field (pixel sensitivity)
          dead.parquet             (32, 356) — dead/hot pixel mask
          read.parquet             (32, 356) — read noise
          linear_corr.parquet      (192, 356) — linearity correction coefficients
      FGS1_calibration/
          dark.parquet             (32, 32)
          flat.parquet             (32, 32)
          dead.parquet             (32, 32)
          read.parquet             (32, 32)
          linear_corr.parquet      (192, 32)

Root-level files:
  train_labels.csv    — ground-truth transmission spectra: planet_id | wl_1 ... wl_283
                        ONLY mean values (no quartile/sigma columns). ~24% of planets labelled.
  train_adc_info.csv  — per-planet ADC calibration: planet_id | FGS1_adc_offset |
                        FGS1_adc_gain | AIRS-CH0_adc_offset | AIRS-CH0_adc_gain | star
                        These 5 values are used as auxiliary features for the model.
  wavelengths.csv     — (1, 283) output wavelength values in microns:
                        wl_1 = 0.705 µm (FGS1 channel), wl_2–wl_283 = 282 AIRS bins
  axis_info.parquet   — (135000, 4) time/wavelength axis metadata

Detector constants (confirmed):
  AIRS_N_ROWS   = 32   (spatial rows)
  AIRS_N_COLS   = 356  (spectral channels)
  FGS1_N_ROWS   = 32
  FGS1_N_COLS   = 32
  FGS1_RATIO    = 12   (FGS1 frames per AIRS frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import preprocess_planet

# ── Detector geometry (confirmed from parquet shapes) ──────────────────────
AIRS_N_ROWS: int = 32    # spatial rows on AIRS-CH0 detector
AIRS_N_COLS: int = 356   # spectral channels
FGS1_N_ROWS: int = 32
FGS1_N_COLS: int = 32
FGS1_RATIO:  int = 12    # FGS1 frames per AIRS frame (135000 / 11250)


def _calibrate(
    raw: np.ndarray,          # (n_time, n_rows, n_cols)  float32
    dark: np.ndarray,         # (n_rows, n_cols)
    flat: np.ndarray,         # (n_rows, n_cols)
    dead: np.ndarray,         # (n_rows, n_cols) — non-zero means dead
) -> np.ndarray:
    """
    Apply basic detector calibration in-place:
        1. Dark subtraction: raw - dark
        2. Flat-field division: / flat  (guard against zero flat)
        3. Dead pixel zeroing

    Returns calibrated array of same shape as raw.
    """
    cal = raw - dark[None]                                    # dark subtract
    flat_safe = np.where(flat == 0, 1.0, flat)
    cal /= flat_safe[None]                                    # flat field
    dead_mask = dead.astype(bool)
    cal[:, dead_mask] = 0.0                                   # zero dead pixels
    return cal


class ArielDataset(Dataset):
    """
    PyTorch Dataset for Ariel transit photometry.

    Calibration pipeline applied per planet (unless preprocess=False):
        raw uint16 pixels
          → dark subtract, flat field, dead pixel mask
          → sum over spatial rows  →  (n_time, 356) for AIRS
          → sum all FGS1 pixels    →  (n_time_fgs,) → downsample to AIRS cadence
          → preprocess_planet()    →  baseline-norm, common-mode, bin_time

    Parameters
    ----------
    data_root : path to competition data root
    split     : 'train' or 'test'
    ingress   : transit ingress fraction (default 0.2)
    egress    : transit egress fraction  (default 0.8)
    bin_size  : temporal binning factor  (default 5)
    preprocess: if True, run full preprocessing; if False, return calibrated but un-preprocessed

    Returns (per __getitem__)
    -------------------------
    dict with keys:
        planet_id    : str
        airs         : (356, time) float32 tensor  — channel-first for Conv1d
        fgs1         : (1, time)  float32 tensor
        aux          : (5,)       float32 tensor   — ADC gain/offset + star type
        target_mean  : (283,)     float32 tensor   — only for labelled train planets
                                                     (sigma is not supervised; learned via GLL loss)
    """

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

        self.split_dir = self.data_root / split

        # Planet IDs = sorted subdirectory names (e.g. "100468857")
        self.planet_ids: list[str] = sorted(
            d.name for d in self.split_dir.iterdir() if d.is_dir()
        )

        # ── Auxiliary features (confirmed: train_adc_info.csv) ───────────
        # 5 columns: FGS1_adc_offset, FGS1_adc_gain, AIRS-CH0_adc_offset,
        #            AIRS-CH0_adc_gain, star  (indexed by planet_id int)
        self._aux_cols = [
            "FGS1_adc_offset", "FGS1_adc_gain",
            "AIRS-CH0_adc_offset", "AIRS-CH0_adc_gain",
            "star",
        ]
        adc_path = self.data_root / f"{split}_adc_info.csv"
        self.aux: Optional[pd.DataFrame] = None
        if adc_path.exists():
            self.aux = pd.read_csv(adc_path).set_index("planet_id")

        # ── Labels (train split only, confirmed format) ──────────────────
        # train_labels.csv: planet_id | wl_1 | wl_2 | ... | wl_283
        # Provides mean transmission spectrum only — NO sigma ground truth.
        # Sigma is learned purely via GLL loss during training. ~24% labelled.
        self.labels: Optional[pd.DataFrame] = None
        self._labelled_ids: set[str] = set()
        if split == "train":
            label_path = self.data_root / "train_labels.csv"
            if label_path.exists():
                self.labels = pd.read_csv(label_path).set_index("planet_id")
                self._labelled_ids = set(self.labels.index.astype(str))

    # ── Internal helpers ─────────────────────────────────────────────────

    def _load_airs(self, planet_dir: Path) -> np.ndarray:
        """
        Load, calibrate, and collapse AIRS-CH0 signal.

        Returns
        -------
        airs : (n_time, 356) float32
        """
        raw = (
            pd.read_parquet(planet_dir / "AIRS-CH0_signal.parquet")
            .values.astype(np.float32)
        )                                                         # (n_time, 32*356)
        n_time = raw.shape[0]
        raw = raw.reshape(n_time, AIRS_N_ROWS, AIRS_N_COLS)      # (n_time, 32, 356)

        cal_dir = planet_dir / "AIRS-CH0_calibration"
        dark = pd.read_parquet(cal_dir / "dark.parquet").values.astype(np.float32)
        flat = pd.read_parquet(cal_dir / "flat.parquet").values.astype(np.float32)
        dead = pd.read_parquet(cal_dir / "dead.parquet").values

        cal = _calibrate(raw, dark, flat, dead)                   # (n_time, 32, 356)
        return cal.sum(axis=1)                                    # (n_time, 356)

    def _load_fgs1(self, planet_dir: Path, n_time_airs: int) -> np.ndarray:
        """
        Load, calibrate, sum, and downsample FGS1 signal to AIRS cadence.

        Returns
        -------
        fgs1 : (n_time_airs,) float32
        """
        raw = (
            pd.read_parquet(planet_dir / "FGS1_signal.parquet")
            .values.astype(np.float32)
        )                                                         # (n_time_fgs, 32*32)
        n_time_fgs = raw.shape[0]
        raw = raw.reshape(n_time_fgs, FGS1_N_ROWS, FGS1_N_COLS) # (n_time_fgs, 32, 32)

        cal_dir = planet_dir / "FGS1_calibration"
        dark = pd.read_parquet(cal_dir / "dark.parquet").values.astype(np.float32)
        flat = pd.read_parquet(cal_dir / "flat.parquet").values.astype(np.float32)
        dead = pd.read_parquet(cal_dir / "dead.parquet").values

        cal = _calibrate(raw, dark, flat, dead)                   # (n_time_fgs, 32, 32)
        fgs1_full = cal.sum(axis=(1, 2))                          # (n_time_fgs,)

        # Downsample to AIRS cadence
        ratio = n_time_fgs // n_time_airs
        if ratio > 1:
            trimmed = fgs1_full[: n_time_airs * ratio]
            return trimmed.reshape(n_time_airs, ratio).mean(axis=1)  # (n_time_airs,)
        return fgs1_full[:n_time_airs]

    def _parse_labels(self, pid: str) -> np.ndarray:
        """
        Parse train_labels.csv row into target_mean.

        Confirmed format (2026-02-21): planet_id | wl_1 | wl_2 | ... | wl_283
        Only mean transmission values are provided — no sigma/quartile columns.
        Sigma is purely model-predicted and trained via GLL loss.
        """
        row = self.labels.loc[int(pid)]
        wl_cols = [f"wl_{i}" for i in range(1, 284)]
        return row[wl_cols].values.astype(np.float32)

    def _get_aux(self, pid: str) -> np.ndarray:
        """
        Return the 5 ADC auxiliary features for planet `pid`.

        Confirmed columns: FGS1_adc_offset, FGS1_adc_gain, AIRS-CH0_adc_offset,
        AIRS-CH0_adc_gain, star  (from train_adc_info.csv).
        """
        if self.aux is not None:
            try:
                return self.aux.loc[int(pid)][self._aux_cols].values.astype(np.float32)
            except KeyError:
                pass
        return np.zeros(5, dtype=np.float32)

    # ── Dataset protocol ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.planet_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.planet_ids[idx]
        planet_dir = self.split_dir / pid

        airs = self._load_airs(planet_dir)                 # (n_time, 356)
        fgs1 = self._load_fgs1(planet_dir, airs.shape[0]) # (n_time,)

        if self.preprocess:
            result = preprocess_planet(
                airs, fgs1, self.ingress, self.egress, self.bin_size
            )
            airs_out = result["airs_norm"].T.astype(np.float32)   # (356, time_bin)
            fgs1_out = result["fgs1_norm"][None].astype(np.float32)  # (1, time_bin)
        else:
            airs_out = airs.T.astype(np.float32)   # (356, time)
            fgs1_out = fgs1[None].astype(np.float32)  # (1, time)

        aux = self._get_aux(pid)

        sample: dict = {
            "planet_id": pid,
            "airs": torch.from_numpy(airs_out),
            "fgs1": torch.from_numpy(fgs1_out),
            "aux":  torch.from_numpy(aux),
        }

        if pid in self._labelled_ids:
            target_mean = self._parse_labels(pid)
            sample["target_mean"] = torch.from_numpy(target_mean)

        return sample
