# Ariel Exoplanet ML — Full Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete, portfolio-quality ML pipeline for the NeurIPS Ariel Data Challenge 2024 — from raw transit photometry to calibrated atmospheric spectra posteriors — and publish the preprocessed dataset on HuggingFace Hub.

**Architecture:** Two-stage pipeline: (1) signal preprocessing (baseline normalization, temporal binning, common-mode systematics removal) on the raw 2D flux images (time × wavelength); (2) deep learning retrieval model (1D/2D CNN or Temporal Transformer) trained with Gaussian NLL loss to predict transmission spectrum mean + uncertainty per wavelength bin.

**Tech Stack:** Python 3.10+, PyTorch, h5py, pandas, numpy, scikit-learn, plotly, matplotlib, huggingface `datasets` + `huggingface_hub`, Kaggle notebooks for training (180GB dataset).

**Workflow:** Code developed locally → pushed to GitHub → Kaggle notebook clones repo and runs against competition data.

---

## Data Format Reference

From the arxiv paper (2505.08940):
- **AIRS-CH0**: 2D image per planet — (time_steps × 356 wavelength channels) — IR spectrometer 1.95–3.90 µm
- **FGS1**: 1D light curve per planet — (time_steps,) — visible photometer 0.60–0.80 µm
- **AuxiliaryTable.csv**: 9 features (Star Distance, Stellar Mass, Stellar Radius, Stellar Temperature, Planet Mass, Orbital Period, Semi-Major Axis, Planet Radius, Surface Gravity)
- **Output**: 283 wavelength transmission spectrum values + uncertainty per planet
- **Scoring metric**: Gaussian Log-Likelihood: `GLL(y, μ, σ) = -0.5 * (log(2π) + log(σ²) + (y−μ)²/σ²)`, normalized against reference/ideal baselines
- **File format**: Likely HDF5 (SpectralData.hdf5 nested by planet_id) or parquet — **confirm during Task 2**

---

## Task 1: Repo Skeleton (local)

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `CLAUDE.md`
- Create: `data/download.sh`
- Create: `src/__init__.py`
- Create: `src/preprocessing.py` (stub)
- Create: `src/dataset.py` (stub)
- Create: `src/model.py` (stub)
- Create: `src/train.py` (stub)
- Create: `src/evaluate.py` (stub)
- Create: `notebooks/` (empty dir with .gitkeep)
- Create: `hf_dataset/ariel_dataset.py` (stub)

**Step 1: Create directory structure**

```bash
cd "c:/Users/alexy/Documents/Claude_projects/Kaggle competition/ARIEL neurIPS"
mkdir -p src notebooks data hf_dataset docs/plans
touch src/__init__.py notebooks/.gitkeep
```

**Step 2: Create .gitignore**

```
# Data (too large, lives on Kaggle)
data/raw/
data/processed/
*.hdf5
*.parquet
*.h5
*.npy
*.npz

# Kaggle credentials
.kaggle/

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
.eggs/
dist/
build/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Models / artifacts
models/
checkpoints/
*.ckpt
*.pt
*.pth

# HuggingFace local cache
hf_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

**Step 3: Create requirements.txt**

```
# Core ML
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Data I/O
h5py>=3.9.0
pyarrow>=13.0.0
fastparquet>=2023.7.0

# Signal processing
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
plotly>=5.17.0
seaborn>=0.12.0

# HuggingFace
datasets>=2.14.0
huggingface_hub>=0.17.0

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0

# Optional: Gaussian Processes for detrending
# george>=0.4.0
```

**Step 4: Create stub src files** (each with module docstring + TODO markers only — filled in later tasks)

**Step 5: Create data/download.sh**

```bash
#!/usr/bin/env bash
# Download competition data via Kaggle API (requires ~/.kaggle/kaggle.json)
# NOTE: ~180GB — run on Kaggle directly, or use a machine with sufficient disk

pip install kaggle --quiet
kaggle competitions download -c ariel-data-challenge-2024
unzip ariel-data-challenge-2024.zip -d data/raw/
```

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: initialize ariel-exoplanet-ml repo skeleton"
```

---

## Task 2: Data Exploration Script (runs on Kaggle)

This is a lightweight script run ONCE on Kaggle to map the exact file structure — output gets saved as a markdown snippet that drives all subsequent notebook design.

**Files:**
- Create: `scripts/explore_data.py`

**Step 1: Write the exploration script**

```python
"""
Run once on Kaggle to map exact data structure.
Usage: python scripts/explore_data.py --data-root /kaggle/input/ariel-data-challenge-2024
"""

import argparse
import os
import h5py
import pandas as pd
import numpy as np

def explore_hdf5(path: str, depth: int = 0, max_planets: int = 3):
    """Recursively print HDF5 structure and shapes."""
    with h5py.File(path, 'r') as f:
        planet_ids = list(f.keys())
        print(f"  Total planets: {len(planet_ids)}")
        for pid in planet_ids[:max_planets]:
            print(f"\n  Planet {pid}:")
            for key in f[pid].keys():
                arr = f[pid][key][()]
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}, "
                      f"min={arr.min():.4f}, max={arr.max():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/kaggle/input/ariel-data-challenge-2024')
    args = parser.parse_args()
    root = args.data_root

    # List all files
    print("=== File tree ===")
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(dirpath)}/")
        for f in filenames:
            fpath = os.path.join(dirpath, f)
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"{indent}  {f}  ({size_mb:.1f} MB)")

    # Inspect HDF5 files
    for fname in ['train', 'test']:
        for ext in ['.hdf5', '.h5']:
            candidate = os.path.join(root, f'{fname}{ext}')
            if os.path.exists(candidate):
                print(f"\n=== {fname}{ext} ===")
                explore_hdf5(candidate)

    # Inspect parquet files
    for fname in os.listdir(root):
        if fname.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(root, fname), engine='pyarrow')
            print(f"\n=== {fname} ===")
            print(f"  Shape: {df.shape}")
            print(f"  Columns[:10]: {list(df.columns[:10])}")
            print(f"  Dtypes:\n{df.dtypes.value_counts()}")

    # Inspect CSV files
    for fname in os.listdir(root):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(root, fname), nrows=5)
            print(f"\n=== {fname} ===")
            print(df.head())
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

if __name__ == '__main__':
    main()
```

**Step 2: Push and run on Kaggle**

```bash
git add scripts/explore_data.py
git commit -m "feat: add data exploration script"
git push
```

On Kaggle: Internet ON, pull repo, run the script, copy the terminal output into `docs/data_format.md`.

**Step 3: Document findings in `docs/data_format.md`**

Fill in from Kaggle output:
- Exact file names and paths
- Shapes of AIRS-CH0 and FGS1 arrays per planet
- Column names of any CSV/parquet
- Number of train vs test planets

---

## Task 3: Notebook 01 — Exploratory Data Analysis

**Files:**
- Create: `notebooks/01_eda.ipynb`

This notebook runs on Kaggle. It should be self-contained (clones repo from GitHub, installs requirements). Target: ~20 rich cells with real visualizations.

**Step 1: Notebook header cell (Kaggle setup)**

```python
# Install and clone (run on Kaggle)
import subprocess
subprocess.run(['pip', 'install', '-q', 'plotly', 'h5py'], check=True)
# !git clone https://github.com/YOUR_USERNAME/ariel-exoplanet-ml.git
# import sys; sys.path.insert(0, 'ariel-exoplanet-ml')

DATA_ROOT = '/kaggle/input/ariel-data-challenge-2024'
```

**Step 2: Data loading cell**

```python
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load auxiliary table
aux = pd.read_csv(f'{DATA_ROOT}/AuxillaryTable.csv', index_col=0)
print(f"Train planets: {len(aux)}")
print(aux.describe())
```

**Step 3: Light curve visualization cell** — pick 3 planets spanning SNR range

```python
def plot_transit(planet_id: str, hdf5_path: str) -> go.Figure:
    """
    Plot FGS1 and AIRS-CH0 light curves for a single planet.
    Shows: (1) FGS1 photometry, (2) AIRS-CH0 flux image, (3) mean AIRS spectrum
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Replace keys with actual names discovered in Task 2
        fgs1 = f[planet_id]['FGS1'][()]       # (time_steps,) or similar
        airs = f[planet_id]['AIRS-CH0'][()]    # (time_steps, 356)

    fig = make_subplots(rows=3, cols=1,
        subplot_titles=['FGS1 Light Curve', 'AIRS-CH0 Flux Image (time × wavelength)', 'Mean AIRS Spectrum'])

    # FGS1 time series
    fig.add_trace(go.Scatter(y=fgs1, mode='lines', name='FGS1'), row=1, col=1)

    # AIRS 2D image
    fig.add_trace(go.Heatmap(z=airs.T, colorscale='Viridis', name='AIRS-CH0'), row=2, col=1)

    # Mean spectrum (collapse time axis)
    fig.add_trace(go.Scatter(y=airs.mean(axis=0), mode='lines', name='Mean flux'), row=3, col=1)

    fig.update_layout(height=800, title=f'Planet {planet_id}')
    return fig
```

**Step 4: SNR analysis cell**

```python
def estimate_transit_snr(airs: np.ndarray, transit_mask: np.ndarray) -> float:
    """
    SNR = transit_depth / out_of_transit_std
    transit_mask: bool array of shape (time_steps,), True = in-transit
    """
    in_flux = airs[transit_mask].mean(axis=0)
    out_flux = airs[~transit_mask].mean(axis=0)
    depth = 1.0 - (in_flux / out_flux)  # ppm after ×1e6
    noise = airs[~transit_mask].std(axis=0)
    return (depth / noise).mean()
```

**Step 5: Stellar parameter distributions**

```python
# Scatter matrix: stellar T, radius, planet radius → expected SNR
# Color by SNR
import plotly.express as px
fig = px.scatter_matrix(aux, dimensions=['Star_Radius', 'Star_Temp', 'Planet_Radius', 'Stellar_Mass'],
                        color=aux['Planet_Radius'],  # proxy for signal strength
                        title='Stellar parameter space coverage')
fig.show()
```

**Step 6: Label distribution cell** (train split only — 24% of planets have ground truth)

```python
# Load Tracedata.hdf5 or QuartilesTable.csv
# Plot distribution of transit depths per wavelength
# Plot posterior width (uncertainty) distribution
```

**Step 7: Wavelength–parameter correlation cell**

```python
# For labelled planets: does transit depth at long wavelengths correlate with log_H2O?
# Scatter: depth[λ=2.5µm] vs log_H2O (from training labels)
```

**Step 8: Commit**

```bash
git add notebooks/01_eda.ipynb docs/
git commit -m "feat: add EDA notebook with transit visualizations and SNR analysis"
```

---

## Task 4: src/preprocessing.py — Signal Processing Pipeline

**Files:**
- Create: `src/preprocessing.py`

This is the core data science contribution. All functions must be independently testable, pure (no side effects), and well-documented for the astronomer audience.

**Step 1: Write the module**

```python
"""
preprocessing.py — Ariel transit light curve preprocessing

Pipeline (in order):
  1. out_of_transit_mask        — identify in/out transit time steps
  2. baseline_normalize         — divide by out-of-transit median
  3. bin_time                   — temporal binning to reduce noise
  4. common_mode_correction      — remove correlated systematics across wavelengths
  5. extract_transit_depth       — weighted mean of normalized in-transit flux deficit
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Tuple


def out_of_transit_mask(
    n_time: int,
    ingress: float = 0.2,
    egress: float = 0.8,
) -> np.ndarray:
    """
    Returns boolean mask (shape: n_time) — True = out of transit.

    Assumes normalized time axis [0, 1] where transit runs from ingress to egress.
    For real data these fractions come from the orbital parameters.

    Parameters
    ----------
    n_time : int
    ingress : float  fraction of total time where transit starts
    egress  : float  fraction of total time where transit ends
    """
    t = np.linspace(0, 1, n_time)
    return (t < ingress) | (t > egress)


def baseline_normalize(
    flux: np.ndarray,
    mask_oot: np.ndarray,
) -> np.ndarray:
    """
    Normalize each spectral channel by its out-of-transit (OOT) median.

    Parameters
    ----------
    flux    : (time, wavelength) — raw detector counts
    mask_oot: (time,) bool — True = out of transit

    Returns
    -------
    normalized : (time, wavelength) — dimensionless, OOT ≈ 1.0
    """
    baseline = np.median(flux[mask_oot], axis=0, keepdims=True)  # (1, wavelength)
    baseline = np.where(baseline == 0, 1.0, baseline)             # guard division by zero
    return flux / baseline


def bin_time(
    flux: np.ndarray,
    bin_size: int,
) -> np.ndarray:
    """
    Temporal binning: average every `bin_size` consecutive time steps.

    Parameters
    ----------
    flux     : (time, wavelength) or (time,)
    bin_size : int — number of frames to average

    Returns
    -------
    binned : (time // bin_size, wavelength) or (time // bin_size,)
    """
    n_time = flux.shape[0]
    n_bins = n_time // bin_size
    trimmed = flux[:n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size, *flux.shape[1:]).mean(axis=1)


def common_mode_correction(
    flux_norm: np.ndarray,
    mask_oot: np.ndarray,
) -> np.ndarray:
    """
    Remove common-mode systematics shared across wavelength channels.

    Strategy: compute the mean OOT light curve across all channels (the
    "common mode"), then divide each channel by this trend. Correlated
    instrument/stellar variability that affects all channels equally is removed.

    Parameters
    ----------
    flux_norm : (time, wavelength) — baseline-normalized flux
    mask_oot  : (time,) bool — out-of-transit mask

    Returns
    -------
    corrected : (time, wavelength)
    """
    # Common mode: mean across wavelengths, computed from OOT only, then interpolated
    common = flux_norm.mean(axis=1, keepdims=True)       # (time, 1)
    oot_mean = common[mask_oot].mean()
    common = common / oot_mean                            # normalize to 1.0
    corrected = flux_norm / common
    return corrected


def extract_transit_depth(
    flux_norm: np.ndarray,
    mask_oot: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transit depth (Rp/Rs)² per wavelength channel as:
        depth[λ] = 1 - mean(flux_norm[in_transit, λ])

    Parameters
    ----------
    flux_norm : (time, wavelength)
    mask_oot  : (time,) bool — out-of-transit mask (True = OOT)
    weights   : (time,) optional per-timestep weights (e.g. 1/variance)

    Returns
    -------
    depth     : (wavelength,) — transit depth in fractional units
    depth_err : (wavelength,) — propagated uncertainty (std of in-transit mean)
    """
    mask_it = ~mask_oot
    in_transit = flux_norm[mask_it]                       # (n_intransit, wavelength)

    if weights is not None:
        w = weights[mask_it, None]
        depth = 1.0 - np.average(in_transit, axis=0, weights=w.ravel())
    else:
        depth = 1.0 - in_transit.mean(axis=0)

    # Uncertainty: std of individual-timestep depth estimates / sqrt(N)
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
    Full preprocessing pipeline for a single planet.

    Parameters
    ----------
    airs     : (time, 356) raw AIRS-CH0 flux
    fgs1     : (time,) raw FGS1 flux
    ingress  : transit ingress fraction
    egress   : transit egress fraction
    bin_size : temporal binning factor

    Returns
    -------
    dict with keys:
        airs_norm       : (time//bin_size, 356) normalized, binned AIRS flux
        fgs1_norm       : (time//bin_size,) normalized, binned FGS1 flux
        transit_depth   : (356,) estimated transit depth per AIRS channel
        transit_depth_err: (356,) uncertainty
        mask_oot        : (time//bin_size,) out-of-transit boolean mask
    """
    n_time = airs.shape[0]
    mask_oot = out_of_transit_mask(n_time, ingress, egress)

    # AIRS pipeline
    airs_norm = baseline_normalize(airs, mask_oot)
    airs_norm = common_mode_correction(airs_norm, mask_oot)
    airs_binned = bin_time(airs_norm, bin_size)
    mask_binned = bin_time(mask_oot.astype(float), bin_size) > 0.5

    # FGS1 pipeline
    fgs1_2d = fgs1[:, None]                   # make 2D for reuse of functions
    fgs1_norm = baseline_normalize(fgs1_2d, mask_oot)
    fgs1_binned = bin_time(fgs1_norm, bin_size)[:, 0]

    # Extract transit depth from AIRS
    depth, depth_err = extract_transit_depth(airs_binned, mask_binned)

    return {
        'airs_norm': airs_binned,
        'fgs1_norm': fgs1_binned,
        'transit_depth': depth,
        'transit_depth_err': depth_err,
        'mask_oot': mask_binned,
    }
```

**Step 2: Write tests**

```python
# tests/test_preprocessing.py

import numpy as np
import pytest
from src.preprocessing import (
    out_of_transit_mask, baseline_normalize, bin_time,
    common_mode_correction, extract_transit_depth, preprocess_planet
)

def test_out_of_transit_mask_shape():
    mask = out_of_transit_mask(100)
    assert mask.shape == (100,)
    assert mask.dtype == bool

def test_out_of_transit_mask_fractions():
    mask = out_of_transit_mask(100, ingress=0.2, egress=0.8)
    # First 20% and last 20% should be OOT
    assert mask[:20].all()
    assert mask[80:].all()
    assert not mask[20:80].any()

def test_baseline_normalize_oot_is_one():
    rng = np.random.default_rng(42)
    flux = rng.normal(1000.0, 5.0, size=(100, 50))
    mask_oot = out_of_transit_mask(100)
    norm = baseline_normalize(flux, mask_oot)
    oot_median = np.median(norm[mask_oot], axis=0)
    np.testing.assert_allclose(oot_median, 1.0, atol=1e-6)

def test_bin_time_shape():
    flux = np.ones((100, 50))
    binned = bin_time(flux, bin_size=5)
    assert binned.shape == (20, 50)

def test_bin_time_averages_correctly():
    flux = np.arange(10, dtype=float).reshape(-1, 1)  # [0,1,2,...,9]
    binned = bin_time(flux, bin_size=2)
    expected = np.array([0.5, 2.5, 4.5, 6.5, 8.5]).reshape(-1, 1)
    np.testing.assert_allclose(binned, expected)

def test_extract_transit_depth_zero_for_flat():
    # Flat light curve → depth ≈ 0
    flux = np.ones((100, 10))
    mask_oot = out_of_transit_mask(100)
    depth, err = extract_transit_depth(flux, mask_oot)
    np.testing.assert_allclose(depth, 0.0, atol=1e-10)

def test_extract_transit_depth_known_signal():
    # Transit makes in-transit flux 0.99 → depth ≈ 0.01
    flux = np.ones((100, 1))
    mask_oot = out_of_transit_mask(100)
    flux[~mask_oot] = 0.99
    depth, _ = extract_transit_depth(flux, mask_oot)
    np.testing.assert_allclose(depth, 0.01, atol=1e-6)

def test_preprocess_planet_output_keys():
    rng = np.random.default_rng(0)
    airs = rng.normal(1.0, 0.01, size=(500, 356))
    fgs1 = rng.normal(1.0, 0.01, size=(500,))
    result = preprocess_planet(airs, fgs1, bin_size=5)
    assert set(result.keys()) == {'airs_norm', 'fgs1_norm', 'transit_depth',
                                  'transit_depth_err', 'mask_oot'}
    assert result['airs_norm'].shape == (100, 356)
    assert result['fgs1_norm'].shape == (100,)
    assert result['transit_depth'].shape == (356,)
```

**Step 3: Run tests**

```bash
cd "c:/Users/alexy/Documents/Claude_projects/Kaggle competition/ARIEL neurIPS"
python -m pytest tests/test_preprocessing.py -v
```

Expected: All 8 tests PASS

**Step 4: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: implement signal preprocessing pipeline with full test suite"
```

---

## Task 5: src/dataset.py — PyTorch Dataset

**Files:**
- Create: `src/dataset.py`

**Step 1: Write the Dataset class**

```python
"""
dataset.py — PyTorch Dataset wrapper for Ariel competition data

Handles two access patterns:
  - Train: returns (airs, fgs1, aux_features, target_mean, target_std)
  - Test/inference: returns (airs, fgs1, aux_features)

Also provides an HFDatasetConverter for pushing preprocessed data to Hub.
"""

from __future__ import annotations
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class ArielDataset(Dataset):
    """
    PyTorch Dataset for Ariel transit photometry.

    Parameters
    ----------
    data_root  : path to competition data root
    split      : 'train' or 'test'
    ingress    : transit ingress fraction
    egress     : transit egress fraction
    bin_size   : temporal binning factor
    preprocess : if True, run preprocessing pipeline; else return raw flux
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        ingress: float = 0.2,
        egress: float = 0.8,
        bin_size: int = 5,
        preprocess: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.ingress = ingress
        self.egress = egress
        self.bin_size = bin_size
        self.preprocess = preprocess

        # Load auxiliary table
        self.aux = pd.read_csv(self.data_root / 'AuxillaryTable.csv', index_col=0)

        # NOTE: actual HDF5 key names must be confirmed from Task 2 data exploration
        # Adjust self._airs_key and self._fgs_key if different
        self._airs_key = 'AIRS-CH0'
        self._fgs_key = 'FGS1'

        # Load labels for train split
        self.labels = None
        if split == 'train':
            label_path = self.data_root / 'QuartilesTable.csv'
            if label_path.exists():
                self.labels = pd.read_csv(label_path, index_col=0)

        # Open HDF5 file (kept open for fast random access)
        hdf5_name = f'{split}.hdf5'
        self._h5_path = self.data_root / hdf5_name
        self._h5_file = None  # lazy open for multiprocessing safety

        self.planet_ids = list(self.aux.index.astype(str))

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self._h5_path, 'r')
        return self._h5_file

    def __len__(self):
        return len(self.planet_ids)

    def __getitem__(self, idx: int) -> dict:
        from src.preprocessing import preprocess_planet, out_of_transit_mask

        pid = self.planet_ids[idx]
        h5 = self._get_h5()

        airs_raw = h5[pid][self._airs_key][()].astype(np.float32)  # (time, 356)
        fgs1_raw = h5[pid][self._fgs_key][()].astype(np.float32)   # (time,)

        # Auxiliary features: normalize to zero mean, unit variance
        aux_row = self.aux.loc[int(pid)].values.astype(np.float32)

        if self.preprocess:
            from src.preprocessing import preprocess_planet
            result = preprocess_planet(airs_raw, fgs1_raw, self.ingress, self.egress, self.bin_size)
            airs_out = result['airs_norm'].T        # (wavelength, time) — channel-first for conv1d
            fgs1_out = result['fgs1_norm'][None]   # (1, time)
        else:
            airs_out = airs_raw.T
            fgs1_out = fgs1_raw[None]

        sample = {
            'planet_id': pid,
            'airs': torch.from_numpy(airs_out),       # (356, time)
            'fgs1': torch.from_numpy(fgs1_out),       # (1, time)
            'aux': torch.from_numpy(aux_row),          # (9,)
        }

        if self.labels is not None and pid in self.labels.index.astype(str):
            # Quartiles: columns like "wl_0_q1", "wl_0_q2", "wl_0_q3", ...
            # Target = q2 (median), uncertainty = (q3 - q1) / 2
            label_row = self.labels.loc[int(pid)]
            # Parse quartile columns — adjust column naming to match actual data
            q1 = label_row.filter(like='_q1').values.astype(np.float32)
            q2 = label_row.filter(like='_q2').values.astype(np.float32)
            q3 = label_row.filter(like='_q3').values.astype(np.float32)
            sample['target_mean'] = torch.from_numpy(q2)           # (283,)
            sample['target_std']  = torch.from_numpy((q3 - q1) / 2)  # (283,)

        return sample

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()
```

**Step 2: Write tests**

```python
# tests/test_dataset.py

import numpy as np
import pytest
import torch

# We mock h5py.File and CSV loading to test without real data

def test_dataset_imports():
    from src.dataset import ArielDataset
    # Just ensure module loads
```

**Step 3: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add PyTorch Dataset class for Ariel competition"
```

---

## Task 6: Notebook 02 — Preprocessing Demo

**Files:**
- Create: `notebooks/02_preprocessing.ipynb`

This notebook runs on Kaggle. Shows before/after preprocessing plots for 3 example planets.

Key cells:
1. Load raw AIRS-CH0 for one planet
2. Apply each preprocessing step, visualize change
3. Compare baseline-subtracted flux to ground truth transit depth
4. Demonstrate SNR improvement from temporal binning (show noise vs bin_size curve)
5. Demonstrate common-mode correction (show residual correlations before/after)

```python
# Cell: SNR vs bin size
bin_sizes = [1, 2, 5, 10, 20, 50]
snr_values = []
for bs in bin_sizes:
    from src.preprocessing import bin_time, extract_transit_depth
    binned = bin_time(airs_norm, bs)
    mask_b = bin_time(mask_oot.astype(float), bs) > 0.5
    depth, err = extract_transit_depth(binned, mask_b)
    snr_values.append((depth / err).mean())

plt.figure(figsize=(8, 4))
plt.plot(bin_sizes, snr_values, 'o-')
plt.xlabel('Bin size (time steps)')
plt.ylabel('Mean SNR across wavelengths')
plt.title('SNR improvement from temporal binning')
plt.grid(True); plt.show()
```

**Commit:**

```bash
git add notebooks/02_preprocessing.ipynb
git commit -m "feat: add preprocessing demo notebook"
```

---

## Task 7: src/model.py — Transit Spectrum Retrieval Model

**Files:**
- Create: `src/model.py`

**Step 1: Write the model**

```python
"""
model.py — Neural network for exoplanet atmospheric retrieval

Architecture: TransitCNN
  - Separate 1D CNN encoders for AIRS-CH0 (along time axis, per wavelength) and FGS1
  - Shared temporal attention to weight time steps
  - MLP fusion with stellar/planetary auxiliary features
  - Output head: 283-dim mean + 283-dim log_variance

Loss: Gaussian NLL = 0.5 * (log_var + (y - mean)^2 / exp(log_var))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalEncoder(nn.Module):
    """
    1D CNN encoder along the time axis.
    Input:  (batch, channels, time)
    Output: (batch, embed_dim)
    """
    def __init__(self, in_channels: int, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),          nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),   nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)          # (batch, embed_dim, time)
        x = self.pool(x)          # (batch, embed_dim, 1)
        return x.squeeze(-1)      # (batch, embed_dim)


class TransitCNN(nn.Module):
    """
    Full retrieval model.

    Parameters
    ----------
    n_airs_channels : int — number of AIRS wavelength channels (356)
    n_aux           : int — number of auxiliary stellar/planet features (9)
    n_output_wl     : int — number of output wavelengths (283)
    embed_dim       : int — internal embedding size
    """

    def __init__(
        self,
        n_airs_channels: int = 356,
        n_aux: int = 9,
        n_output_wl: int = 283,
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_output_wl = n_output_wl

        self.airs_encoder = TemporalEncoder(n_airs_channels, embed_dim, dropout)
        self.fgs1_encoder = TemporalEncoder(1, embed_dim // 4, dropout)

        fusion_dim = embed_dim + embed_dim // 4 + n_aux
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 512),        nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.GELU(),
        )
        self.head_mean    = nn.Linear(256, n_output_wl)
        self.head_log_var = nn.Linear(256, n_output_wl)

    def forward(
        self,
        airs: torch.Tensor,   # (batch, 356, time)
        fgs1: torch.Tensor,   # (batch, 1, time)
        aux: torch.Tensor,    # (batch, 9)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_airs = self.airs_encoder(airs)   # (batch, embed_dim)
        z_fgs1 = self.fgs1_encoder(fgs1)  # (batch, embed_dim//4)
        z = torch.cat([z_airs, z_fgs1, aux], dim=-1)
        z = self.fusion(z)
        mean    = self.head_mean(z)
        log_var = self.head_log_var(z)
        return mean, log_var


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL: 0.5 * (log_var + (target - mean)^2 / var)
    Averaged over all elements.
    """
    var = torch.exp(log_var)
    return 0.5 * (log_var + (target - mean).pow(2) / var).mean()
```

**Step 2: Write tests**

```python
# tests/test_model.py

import torch
import pytest
from src.model import TransitCNN, gaussian_nll_loss

def make_batch(batch_size=4, time=100):
    return {
        'airs': torch.randn(batch_size, 356, time),
        'fgs1': torch.randn(batch_size, 1, time),
        'aux':  torch.randn(batch_size, 9),
    }

def test_model_output_shape():
    model = TransitCNN()
    batch = make_batch()
    mean, log_var = model(batch['airs'], batch['fgs1'], batch['aux'])
    assert mean.shape == (4, 283)
    assert log_var.shape == (4, 283)

def test_model_forward_no_nan():
    model = TransitCNN()
    batch = make_batch()
    mean, log_var = model(batch['airs'], batch['fgs1'], batch['aux'])
    assert not torch.isnan(mean).any()
    assert not torch.isnan(log_var).any()

def test_gaussian_nll_positive():
    mean = torch.zeros(4, 283)
    log_var = torch.zeros(4, 283)
    target = torch.randn(4, 283)
    loss = gaussian_nll_loss(mean, log_var, target)
    assert loss > 0

def test_gaussian_nll_perfect_prediction():
    # When mean == target, loss ≈ 0.5 * (log_var) at minimum
    mean = torch.ones(4, 283)
    target = torch.ones(4, 283)
    log_var = torch.zeros(4, 283)  # var=1
    loss = gaussian_nll_loss(mean, log_var, target)
    # Expected: 0.5 * (0 + 0) = 0
    assert abs(loss.item()) < 1e-5
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_model.py -v
```

Expected: All 4 tests PASS

**Step 4: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: implement TransitCNN retrieval model with Gaussian NLL loss"
```

---

## Task 8: src/train.py — Training Loop

**Files:**
- Create: `src/train.py`

```python
"""
train.py — Training loop for TransitCNN

Usage (on Kaggle):
    python src/train.py \
        --data-root /kaggle/input/ariel-data-challenge-2024 \
        --epochs 50 \
        --batch-size 32 \
        --lr 3e-4 \
        --output-dir /kaggle/working/checkpoints
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import ArielDataset
from src.model import TransitCNN, gaussian_nll_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--val-fraction', type=float, default=0.15)
    p.add_argument('--output-dir', default='checkpoints')
    p.add_argument('--bin-size', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset — only labelled planets (have target_mean/target_std)
    dataset = ArielDataset(args.data_root, split='train', bin_size=args.bin_size)
    labelled = [i for i in range(len(dataset))
                if dataset.labels is not None and dataset.planet_ids[i] in
                dataset.labels.index.astype(str)]

    from torch.utils.data import Subset
    labelled_ds = Subset(dataset, labelled)

    n_val = int(len(labelled_ds) * args.val_fraction)
    n_train = len(labelled_ds) - n_val
    train_ds, val_ds = random_split(labelled_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)

    model = TransitCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            airs   = batch['airs'].to(device)
            fgs1   = batch['fgs1'].to(device)
            aux    = batch['aux'].to(device)
            target = batch['target_mean'].to(device)

            optimizer.zero_grad()
            mean, log_var = model(airs, fgs1, aux)
            loss = gaussian_nll_loss(mean, log_var, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                airs   = batch['airs'].to(device)
                fgs1   = batch['fgs1'].to(device)
                aux    = batch['aux'].to(device)
                target = batch['target_mean'].to(device)
                mean, log_var = model(airs, fgs1, aux)
                val_losses.append(gaussian_nll_loss(mean, log_var, target).item())

        scheduler.step()
        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val={val_loss:.4f})")

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train(parse_args())
```

**Commit:**

```bash
git add src/train.py
git commit -m "feat: add training loop with cosine LR schedule and early stopping"
```

---

## Task 9: src/evaluate.py — Inference + GLL Metric

**Files:**
- Create: `src/evaluate.py`

```python
"""
evaluate.py — Compute competition GLL metric and generate submission

Usage:
    python src/evaluate.py \
        --data-root /kaggle/input/ariel-data-challenge-2024 \
        --checkpoint checkpoints/best_model.pt \
        --output submission.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import ArielDataset
from src.model import TransitCNN


def gaussian_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Scalar GLL averaged over all planets and wavelengths."""
    return -0.5 * np.mean(np.log(2 * np.pi * sigma**2) + ((y - mu) / sigma)**2)


def predict(model, loader, device) -> dict:
    model.eval()
    planet_ids, means, stds = [], [], []
    with torch.no_grad():
        for batch in loader:
            airs = batch['airs'].to(device)
            fgs1 = batch['fgs1'].to(device)
            aux  = batch['aux'].to(device)
            mean, log_var = model(airs, fgs1, aux)
            std = torch.exp(0.5 * log_var)
            planet_ids.extend(batch['planet_id'])
            means.append(mean.cpu().numpy())
            stds.append(std.cpu().numpy())
    return {
        'planet_id': planet_ids,
        'mean': np.concatenate(means, axis=0),
        'std':  np.concatenate(stds,  axis=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',   required=True)
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--output',      default='submission.csv')
    parser.add_argument('--split',       default='test')
    parser.add_argument('--bin-size',    type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransitCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    dataset = ArielDataset(args.data_root, split=args.split, bin_size=args.bin_size)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    preds   = predict(model, loader, device)

    # Format submission CSV (adapt column names to match competition requirements)
    n_wl = preds['mean'].shape[1]
    wl_cols_mean = [f'wl_{i}_mean' for i in range(n_wl)]
    wl_cols_std  = [f'wl_{i}_std'  for i in range(n_wl)]
    df = pd.DataFrame(
        np.concatenate([preds['mean'], preds['std']], axis=1),
        columns=wl_cols_mean + wl_cols_std,
    )
    df.insert(0, 'planet_id', preds['planet_id'])
    df.to_csv(args.output, index=False)
    print(f"Submission saved: {args.output} ({len(df)} planets, {n_wl} wavelengths)")


if __name__ == '__main__':
    main()
```

**Commit:**

```bash
git add src/evaluate.py
git commit -m "feat: add inference and GLL metric evaluation script"
```

---

## Task 10: Notebooks 03 and 04

**Files:**
- Create: `notebooks/03_baseline.ipynb`
- Create: `notebooks/04_deep_learning.ipynb`

**Notebook 03 — Baseline:**
1. Compute transit depth per AIRS channel using `extract_transit_depth()`
2. Set uncertainty = pixel-level noise / sqrt(n_intransit)
3. Resample from 356 AIRS channels to 283 output wavelengths (linear interpolation)
4. Compute GLL on held-out labelled planets
5. Visualize predicted vs true spectrum for 5 example planets

**Notebook 04 — Deep Learning (runs on Kaggle GPU):**
1. Setup cell: clone repo, install requirements, detect GPU
2. Initialize `ArielDataset` and `DataLoader`
3. Train `TransitCNN` using `src/train.py` logic (inline or via `!python src/train.py ...`)
4. Plot training/validation loss curves
5. Show predicted spectrum + uncertainty bands vs ground truth for validation planets
6. Generate final submission

**Commit:**

```bash
git add notebooks/03_baseline.ipynb notebooks/04_deep_learning.ipynb
git commit -m "feat: add baseline and deep learning training notebooks"
```

---

## Task 11: HuggingFace Upload

**Files:**
- Create: `notebooks/05_huggingface_upload.ipynb`
- Create: `hf_dataset/ariel_dataset.py`

**Step 1: Write HF loading script**

```python
# hf_dataset/ariel_dataset.py

"""
HuggingFace Datasets loading script for the Ariel Exoplanet Dataset.

Usage:
    from datasets import load_dataset
    ds = load_dataset("alexy-louis/ariel-exoplanet-2024", split="train")
"""

import datasets
import h5py
import pandas as pd
import numpy as np

_CITATION = """
@misc{ariel2024,
  title  = {NeurIPS Ariel Data Challenge 2024},
  author = {Ariel Data Challenge Team},
  year   = {2024},
  url    = {https://www.kaggle.com/competitions/ariel-data-challenge-2024}
}
"""

_DESCRIPTION = """
Preprocessed transit photometry dataset from the ESA Ariel Data Challenge 2024.
Contains normalized, temporally-binned light curves from AIRS-CH0 (IR spectrometer,
356 channels, 1.95–3.90 µm) and FGS1 (visible photometer), along with auxiliary
stellar/planetary parameters and ground-truth transmission spectra for labelled planets.
"""

class ArielExoplanetDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features({
                'planet_id': datasets.Value('string'),
                'airs_norm': datasets.Array2D(shape=(356, None), dtype='float32'),
                'fgs1_norm': datasets.Sequence(datasets.Value('float32')),
                'aux':       datasets.Sequence(datasets.Value('float32'), length=9),
                'transit_depth':     datasets.Sequence(datasets.Value('float32'), length=356),
                'transit_depth_err': datasets.Sequence(datasets.Value('float32'), length=356),
            })
        )

    def _split_generators(self, dl_manager):
        # Preprocessed data should be uploaded to Hub separately
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={'split': 'train'}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={'split': 'test'}),
        ]

    def _generate_examples(self, split):
        # Load from locally preprocessed numpy files
        # (generated by running preprocess_planet on all planets)
        import glob, os
        data_dir = f'data/preprocessed/{split}'
        for fpath in sorted(glob.glob(f'{data_dir}/*.npz')):
            pid = os.path.basename(fpath).replace('.npz', '')
            arr = np.load(fpath)
            yield pid, {
                'planet_id': pid,
                'airs_norm': arr['airs_norm'],
                'fgs1_norm': arr['fgs1_norm'].tolist(),
                'aux':       arr['aux'].tolist(),
                'transit_depth':     arr['transit_depth'].tolist(),
                'transit_depth_err': arr['transit_depth_err'].tolist(),
            }
```

**Step 2: Notebook 05 — upload to Hub**

```python
# Key cells:

# 1. Preprocess all planets and save to .npz
from src.dataset import ArielDataset
from src.preprocessing import preprocess_planet
import numpy as np

dataset = ArielDataset('/kaggle/input/ariel-data-challenge-2024', preprocess=False)
for i in range(len(dataset)):
    sample = dataset[i]
    result = preprocess_planet(sample['airs_raw'], sample['fgs1_raw'], bin_size=5)
    np.savez_compressed(f'data/preprocessed/train/{sample["planet_id"]}.npz', **result)

# 2. Push to Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='data/preprocessed',
    repo_id='alexy-louis/ariel-exoplanet-2024',
    repo_type='dataset',
)
```

**Commit:**

```bash
git add hf_dataset/ notebooks/05_huggingface_upload.ipynb
git commit -m "feat: add HuggingFace dataset loading script and upload notebook"
```

---

## Task 12: README and CLAUDE.md

**Files:**
- Create: `README.md` (two-audience format)
- Create: `CLAUDE.md`

**README.md structure:**
1. Title + badges (license, HF dataset link, Kaggle competition link)
2. **For ML practitioners**: quick start, data format, model architecture diagram, reproduction steps
3. **For astronomers**: science context (what Ariel does, why atmospheric retrieval matters), how to apply this to your own data
4. Results table (baseline GLL vs model GLL)
5. Citation

**CLAUDE.md** — project guidance for future Claude sessions (commands, architecture notes, workflow)

**Final commit:**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add dual-audience README and CLAUDE.md project guide"
git push
```

---

## Execution Order

1. Task 1 — local, ~10 min
2. Task 2 — push locally, run on Kaggle, document findings in `docs/data_format.md`
3. Task 4 — local + tests (most important: preprocessing)
4. Task 5 — local + tests
5. Task 7 — local + tests
6. Task 3 — Kaggle notebook (uses Task 4 preprocessing)
7. Task 6 — Kaggle notebook
8. Task 8 — local only (runs on Kaggle)
9. Task 9 — local only (runs on Kaggle)
10. Task 10 — Kaggle notebooks
11. Task 11 — Kaggle + Hub
12. Task 12 — local, last

**Critical path:** Task 2 (data format discovery) must complete before finalizing Tasks 3, 4, 5 — column/key names need to be confirmed.
