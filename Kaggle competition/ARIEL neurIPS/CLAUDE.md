# CLAUDE.md — Project Guidance for Claude Code

This file provides guidance for Claude Code sessions working on the **Ariel Exoplanet Atmospheric Retrieval** project.

---

## Project Overview

End-to-end ML pipeline for the NeurIPS 2024 Ariel Data Challenge. Extracts exoplanet atmospheric transmission spectra from simulated ESA Ariel telescope photometry using a CNN retrieval model trained with Gaussian NLL loss.

**Author**: Alexy Louis
**Competition**: https://www.kaggle.com/competitions/ariel-data-challenge-2024
**HF Dataset**: https://huggingface.co/datasets/alexy-louis/ariel-exoplanet-2024

---

## Environment Notes

- **Python version**: Use `C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe` (Python 3.11.9).
  Python 3.14 is installed but has broken numpy — do NOT use it.
- **Platform**: Windows 10, bash shell.
- **Data**: ~180 GB, lives on Kaggle only. No data files are present locally.

---

## Common Commands

```bash
# Run all tests (25 tests, no data required — uses synthetic fixtures)
C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe -m pytest tests/ -v

# Run a specific test file
C:/Users/alexy/AppData/Local/Programs/Python/Python311/python.exe -m pytest tests/test_preprocessing.py -v

# On Kaggle: train the model
python src/train.py \
    --data-root /kaggle/input/ariel-data-challenge-2024 \
    --epochs 50 \
    --batch-size 32 \
    --lr 3e-4 \
    --output-dir /kaggle/working/checkpoints

# On Kaggle: generate submission
python src/evaluate.py \
    --data-root /kaggle/input/ariel-data-challenge-2024 \
    --checkpoint /kaggle/working/checkpoints/best_model.pt \
    --split test \
    --output submission.csv

# On Kaggle: explore data structure
python scripts/explore_data.py --data-root /kaggle/input/ariel-data-challenge-2024
```

---

## Architecture

### Source Modules

| File | Purpose | Key exports |
|------|---------|-------------|
| `src/preprocessing.py` | Signal processing pipeline | `preprocess_planet`, `out_of_transit_mask`, `baseline_normalize`, `common_mode_correction`, `bin_time`, `extract_transit_depth` |
| `src/dataset.py` | PyTorch Dataset | `ArielDataset(root, split, bin_size, preprocess)` |
| `src/model.py` | Neural network | `TransitCNN(embed_dim, dropout)`, `gaussian_nll_loss` |
| `src/train.py` | Training loop | `train(args)`, `train_epoch`, `eval_epoch`, `make_labelled_subset` |
| `src/evaluate.py` | Inference + submission | `run_inference`, `build_submission`, `gaussian_log_likelihood` |

### Data Format (best-guess — verify with explore_data.py on Kaggle)

- HDF5 structure: `f[planet_id]["AIRS-CH0"]` → `(n_time, 356)`, `f[planet_id]["FGS1"]` → `(n_time,)`
- Labels: `QuartilesTable.csv` columns `{i}_q1`, `{i}_q2`, `{i}_q3` for i in 0..282
- Auxiliary: `AuxillaryTable.csv` — 9 stellar/planetary features
- Only ~24% of train planets have labels

### Critical: HDF5 Key Names Are Unconfirmed

All code uses `"AIRS-CH0"` and `"FGS1"` as best-guess HDF5 key names, marked with `# TODO: verify key names`. Before submitting to Kaggle, run:

```python
import h5py
with h5py.File("/kaggle/input/ariel-data-challenge-2024/train.hdf5", "r") as f:
    pid = list(f.keys())[0]
    print(list(f[pid].keys()))  # confirm actual key names
```

Also verify `build_submission` column format against `sample_submission.csv` (see warning in `src/evaluate.py`).

---

## Design Decisions

### Common-Mode Correction (important physics)

In `src/preprocessing.py`, `common_mode_correction()` **clamps in-transit frames to the OOT mean** before dividing. This is critical — if you naively divide all frames (including in-transit) by the common mode, you erase the transit signal because the common mode dips with the planet. See `tests/test_preprocessing.py::test_cmc_preserves_transit_depth`.

### Gaussian NLL Loss

The model outputs `(mean, log_var)`. Loss = `0.5 * (log_var.clamp(-20, 20) + (target - mean)² / exp(log_var))`. The clamp prevents float32 overflow for extreme `log_var` values. This trains calibrated uncertainty alongside the mean, which is essential because the competition metric rewards well-calibrated sigma.

### ArielDataset Lazy HDF5

`ArielDataset` keeps the HDF5 file open for fast random access. The `__del__` method uses `getattr(self, "_h5", None)` to guard against AttributeError if `__init__` fails partway. For multi-worker DataLoader, the file handle is opened per-process on first `__getitem__` call.

### Label Parsing (regex, not filter(like=...))

`dataset.py` uses `row.filter(regex=r"^\d+_q1$")` (not `filter(like="_q1")`) because `like` is a substring match that would also catch `_q10`, `_q100`, etc.

---

## Testing

Tests in `tests/` use synthetic fixtures (no real data needed):
- `tests/conftest.py` — `synthetic_planet` fixture: 500-timestep AIRS (500, 356) + FGS1 with 1% transit injected at 20–80%
- `tests/test_preprocessing.py` — 9 tests for all 6 preprocessing functions
- `tests/test_dataset.py` — 8 tests using mock HDF5 via `tmp_path`
- `tests/test_model.py` — 8 tests including extreme `log_var` NaN check

---

## Notebooks (all Kaggle-ready)

| Notebook | Purpose | Prerequisites |
|----------|---------|---------------|
| `01_eda.ipynb` | Explore data structure, visualize light curves | Competition dataset attached |
| `02_preprocessing.ipynb` | Step-by-step preprocessing walkthrough | Competition dataset attached |
| `03_baseline.ipynb` | Statistical baselines (constant, Ridge regression) | Labels CSV only |
| `04_deep_learning.ipynb` | Full TransitCNN training + submission | Competition dataset + GPU |
| `05_huggingface_upload.ipynb` | Preprocess all planets + push to HF Hub | Competition dataset + HF token |

All notebooks have a synthetic-data fallback for running without the competition dataset.

---

## Workflow: Local → Kaggle

1. Write/edit code locally.
2. `git push` to GitHub.
3. In Kaggle notebook: `git clone https://github.com/alexy-louis/ariel-exoplanet-ml.git` then `sys.path.insert(0, "ariel-exoplanet-ml")`.
4. Run notebook against the attached competition dataset.

---

## Known Issues / TODOs

- **HDF5 key names unconfirmed** — run `scripts/explore_data.py` on Kaggle and update `src/dataset.py` if `"AIRS-CH0"` or `"FGS1"` are wrong.
- **`build_submission` column format unconfirmed** — verify against `sample_submission.csv` before submitting. See warning in `src/evaluate.py::build_submission`.
- **Results table in README.md** — populate after running the baseline and DL notebooks on Kaggle.
- **HuggingFace token** — set `os.environ["HF_TOKEN"]` in `notebooks/05_huggingface_upload.ipynb` before running.
- **GitHub repo URL** — replace `YOUR_USERNAME` placeholder in all notebook setup cells with actual username.
