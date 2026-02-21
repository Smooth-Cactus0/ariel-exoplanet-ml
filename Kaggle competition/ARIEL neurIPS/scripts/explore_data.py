"""
explore_data.py — One-shot data structure discovery for the Ariel competition.

Run on Kaggle after attaching the competition dataset:
    python scripts/explore_data.py --data-root /kaggle/input/ariel-data-challenge-2024

Copy the terminal output into docs/data_format.md to document the confirmed
file structure for all subsequent notebook and pipeline development.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _sep(title: str, width: int = 72) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _size_mb(path: str | Path) -> float:
    return os.path.getsize(path) / 1e6


def print_file_tree(root: Path) -> None:
    _sep("FILE TREE")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        level = Path(dirpath).relative_to(root).parts
        indent = "  " * len(level)
        print(f"{indent}{Path(dirpath).name}/")
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            print(f"{indent}  {fname}  ({_size_mb(fpath):.1f} MB)")


def inspect_hdf5(path: Path, max_planets: int = 3) -> None:
    try:
        import h5py
    except ImportError:
        print("  [SKIP] h5py not installed")
        return

    _sep(f"HDF5: {path.name}")
    try:
        with h5py.File(path, "r") as f:
            planet_ids = sorted(f.keys())
            print(f"  Top-level keys (planets): {len(planet_ids)}")
            print(f"  First {min(max_planets, len(planet_ids))} planets:\n")

            for pid in planet_ids[:max_planets]:
                print(f"  [{pid}]")
                for key in f[pid].keys():
                    try:
                        arr = f[pid][key][()]
                        print(
                            f"    {key}: shape={arr.shape}  dtype={arr.dtype}"
                            f"  min={float(np.nanmin(arr)):.4g}"
                            f"  max={float(np.nanmax(arr)):.4g}"
                        )
                    except Exception as e:
                        print(f"    {key}: ERROR reading — {e}")
                print()
    except Exception as e:
        print(f"  ERROR opening file: {e}")


def inspect_parquet(path: Path) -> None:
    _sep(f"PARQUET: {path.name}")
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        print(f"  Shape: {df.shape}")
        print(f"  Columns (first 10): {list(df.columns[:10])}")
        if len(df.columns) > 10:
            print(f"  ... and {len(df.columns) - 10} more columns")
        print(f"  Dtypes:\n{df.dtypes.value_counts().to_string()}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(f"\n  Head (2 rows, first 5 cols):")
        print(df.iloc[:2, :5].to_string())
    except Exception as e:
        print(f"  ERROR reading parquet: {e}")


def inspect_csv(path: Path) -> None:
    _sep(f"CSV: {path.name}")
    try:
        df = pd.read_csv(path, nrows=5)
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes.to_string()}")
        print(f"\n  Head (5 rows):")
        print(df.to_string())

        # Full row count
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_rows = sum(1 for _ in open(path)) - 1  # subtract header
        print(f"\n  Total rows (approx): {n_rows}")
    except Exception as e:
        print(f"  ERROR reading CSV: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ariel competition data structure explorer."
    )
    parser.add_argument(
        "--data-root",
        default="/kaggle/input/ariel-data-challenge-2024",
        help="Path to the competition data directory",
    )
    parser.add_argument(
        "--max-planets",
        type=int,
        default=3,
        help="Number of example planets to print from HDF5 files",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: data root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Ariel Data Explorer — root: {root}")
    print(f"Python {sys.version}")

    # 1. File tree
    print_file_tree(root)

    # 2. HDF5 files
    for fpath in sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5")):
        inspect_hdf5(fpath, max_planets=args.max_planets)

    # 3. Parquet files
    for fpath in sorted(root.rglob("*.parquet")):
        inspect_parquet(fpath)

    # 4. CSV files
    for fpath in sorted(root.rglob("*.csv")):
        inspect_csv(fpath)

    _sep("DONE — copy this output into docs/data_format.md")


if __name__ == "__main__":
    main()
