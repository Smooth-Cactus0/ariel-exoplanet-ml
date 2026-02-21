"""
evaluate.py — Inference, GLL metric computation, and submission generation.

Usage:
    python src/evaluate.py \
        --data-root  /kaggle/input/ariel-data-challenge-2024 \
        --checkpoint /kaggle/working/checkpoints/best_model.pt \
        --split      test \
        --output     submission.csv

The output CSV format matches what the competition expects:
one row per planet, with columns for mean and std per output wavelength.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import ArielDataset
from src.model import TransitCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a trained TransitCNN.")
    p.add_argument("--data-root",   required=True)
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pt")
    p.add_argument("--split",       default="test", choices=["train", "test"])
    p.add_argument("--output",      default="submission.csv")
    p.add_argument("--bin-size",    type=int,   default=5)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--embed-dim",   type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.0,
                   help="Set to 0 for deterministic inference (default)")
    return p.parse_args()


def gaussian_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Compute the Gaussian log-likelihood score used by the competition.

    GLL(y, mu, sigma) = -0.5 * mean(log(2π σ²) + ((y - mu) / σ)²)

    Higher is better. A perfect prediction gives 0; random noise predictions
    are strongly negative.

    Parameters
    ----------
    y     : (n_planets, n_wl) — ground truth
    mu    : (n_planets, n_wl) — predicted mean
    sigma : (n_planets, n_wl) — predicted std (must be positive)
    """
    sigma = np.clip(sigma, 1e-9, None)  # guard against zero std
    return float(-0.5 * np.mean(np.log(2 * np.pi * sigma**2) + ((y - mu) / sigma)**2))


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Run inference over all batches and collect planet IDs, means, and stds.

    Returns
    -------
    planet_ids : list of str
    means      : (n_planets, n_output_wl) float32 numpy array
    stds       : (n_planets, n_output_wl) float32 numpy array
    """
    model.eval()
    planet_ids: list[str] = []
    means_list: list[np.ndarray] = []
    stds_list:  list[np.ndarray] = []

    for batch in loader:
        airs = batch["airs"].to(device)
        fgs1 = batch["fgs1"].to(device)
        aux  = batch["aux"].to(device)

        mean, log_var = model(airs, fgs1, aux)
        std = torch.exp(0.5 * log_var.clamp(-20.0, 20.0))

        planet_ids.extend(batch["planet_id"])
        means_list.append(mean.cpu().numpy())
        stds_list.append(std.cpu().numpy())

    return planet_ids, np.concatenate(means_list), np.concatenate(stds_list)


def build_submission(
    planet_ids: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> pd.DataFrame:
    """
    Build a submission DataFrame matching the competition format (verified).

    Format (567 columns total):
        planet_id | wl_1 | wl_2 | ... | wl_283 | sigma_1 | sigma_2 | ... | sigma_283

    Columns are 1-indexed. All 283 means come first (blocked), then all 283 sigmas.
    This was confirmed from sample_submission.csv:
        Total columns: 567
        First 6: ['planet_id', 'wl_1', 'wl_2', 'wl_3', 'wl_4', 'wl_5']
        Last  6: ['sigma_278', 'sigma_279', 'sigma_280', 'sigma_281', 'sigma_282', 'sigma_283']
    """
    n_wl = means.shape[1]
    wl_cols    = [f"wl_{i+1}"    for i in range(n_wl)]
    sigma_cols = [f"sigma_{i+1}" for i in range(n_wl)]

    df = pd.DataFrame(means, columns=wl_cols)
    for col, vals in zip(sigma_cols, stds.T):
        df[col] = vals
    df.insert(0, "planet_id", planet_ids)
    return df


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    model = TransitCNN(embed_dim=args.embed_dim, dropout=args.dropout).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Dataset + loader
    dataset = ArielDataset(
        args.data_root, split=args.split,
        bin_size=args.bin_size, preprocess=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # Inference
    planet_ids, means, stds = run_inference(model, loader, device)
    log.info(f"Predicted {len(planet_ids)} planets  ({means.shape[1]} wavelengths)")

    # Optional: compute GLL on labelled split
    if args.split == "train" and dataset.labels is not None:
        labelled_mask = [pid in dataset._labelled_ids for pid in planet_ids]
        if any(labelled_mask):
            idxs = [i for i, m in enumerate(labelled_mask) if m]
            labelled_pids = [planet_ids[i] for i in idxs]
            mu_lab  = means[idxs]
            std_lab = stds[idxs]
            # Fetch ground truth medians
            gt = np.stack([
                dataset.labels.loc[int(pid)].filter(regex=r"^\d+_q2$").values
                for pid in labelled_pids
            ])
            gll = gaussian_log_likelihood(gt, mu_lab, std_lab)
            log.info(f"GLL on labelled train subset ({len(idxs)} planets): {gll:.4f}")

    # Write submission
    df = build_submission(planet_ids, means, stds)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    log.info(f"Submission saved: {output_path}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
