"""
train.py — Training loop for TransitCNN on Ariel competition data.

Designed to run on a Kaggle GPU notebook:
    python src/train.py \
        --data-root /kaggle/input/ariel-data-challenge-2024 \
        --epochs 50 \
        --batch-size 32 \
        --lr 3e-4 \
        --output-dir /kaggle/working/checkpoints

Only labelled planets (~24% of training set) are used for supervised training.
Unlabelled planets could be used for self-supervised pre-training (future work).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from src.dataset import ArielDataset
from src.model import TransitCNN, gaussian_nll_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransitCNN on Ariel data.")
    p.add_argument("--data-root",     required=True,           help="Path to competition data root")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--val-fraction",  type=float, default=0.15)
    p.add_argument("--bin-size",      type=int,   default=5)
    p.add_argument("--embed-dim",     type=int,   default=128)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--num-workers",   type=int,   default=2)
    p.add_argument("--output-dir",    default="checkpoints")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def make_labelled_subset(dataset: ArielDataset) -> Subset:
    """Return a Subset containing only planets with ground-truth labels."""
    labelled_idx = [
        i for i, pid in enumerate(dataset.planet_ids)
        if pid in dataset._labelled_ids
    ]
    if not labelled_idx:
        raise RuntimeError(
            "No labelled planets found. Check that train_labels.csv exists "
            f"in {dataset.data_root}."
        )
    log.info(f"Labelled planets: {len(labelled_idx)} / {len(dataset)}")
    return Subset(dataset, labelled_idx)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        airs   = batch["airs"].to(device)
        fgs1   = batch["fgs1"].to(device)
        aux    = batch["aux"].to(device)
        target = batch["target_mean"].to(device)

        optimizer.zero_grad()
        mean, log_var = model(airs, fgs1, aux)
        loss = gaussian_nll_loss(mean, log_var, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        airs   = batch["airs"].to(device)
        fgs1   = batch["fgs1"].to(device)
        aux    = batch["aux"].to(device)
        target = batch["target_mean"].to(device)
        mean, log_var = model(airs, fgs1, aux)
        losses.append(gaussian_nll_loss(mean, log_var, target).item())
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ---- Dataset ----
    dataset = ArielDataset(
        args.data_root, split="train", bin_size=args.bin_size, preprocess=True
    )
    labelled = make_labelled_subset(dataset)

    n_val   = max(1, int(len(labelled) * args.val_fraction))
    n_train = len(labelled) - n_val
    train_ds, val_ds = random_split(
        labelled, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info(f"Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    # ---- Model ----
    model = TransitCNN(embed_dim=args.embed_dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- Training loop ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model, val_loader, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "lr": lr_now})
        log.info(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            log.info(f"  -> Saved best model  (val={val_loss:.4f})")

    # Save final checkpoint and history
    torch.save(model.state_dict(), output_dir / "last_model.pt")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    train(parse_args())
