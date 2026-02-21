"""
ariel_dataset.py — HuggingFace Datasets loading script for the Ariel Exoplanet Dataset.

Usage (after uploading preprocessed data to Hub):
    from datasets import load_dataset
    ds = load_dataset("alexy-louis/ariel-exoplanet-2024", split="train")
    sample = ds[0]
    # sample keys: planet_id, airs_norm, fgs1_norm, aux, transit_depth, transit_depth_err
    # Labelled train planets also have: target_mean, target_std (the q2 spectrum + uncertainty)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

import datasets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset metadata constants
# ---------------------------------------------------------------------------

_CITATION = """\
@misc{ariel-neurips-2024,
  title        = {NeurIPS 2024 Ariel Data Challenge},
  author       = {Changeat, Q. and others},
  year         = {2024},
  howpublished = {\\url{https://www.ariel-datachallenge.space}},
  note         = {Kaggle competition: ariel-data-challenge-2024},
}
"""

_DESCRIPTION = """\
Ariel Exoplanet Atmospheric Spectra Dataset (NeurIPS 2024).

Each example contains preprocessed photometric time series from the Ariel
space telescope simulator:
  - airs_norm        : baseline-normalised, common-mode-corrected, temporally binned
                       AIRS-CH0 spectral photometry  (time_binned × 356 channels)
  - fgs1_norm        : corresponding FGS1 broadband channel  (time_binned,)
  - aux              : 9 stellar/planetary auxiliary parameters
  - transit_depth    : extracted transit depth per AIRS channel  (356,)
  - transit_depth_err: 1-sigma uncertainty on the transit depth  (356,)

For labelled training planets only:
  - target_mean      : q2 (median) atmospheric spectrum from QuartilesTable  (283,)
  - target_std       : (q3 - q1) / 2 — half inter-quartile range ≈ 1-sigma  (283,)

Preprocessing applied (in order):
  1. out-of-transit mask (ingress=0.20, egress=0.80)
  2. per-channel baseline normalisation (OOT median)
  3. common-mode systematic correction
  4. temporal binning (bin_size=5)
  5. weighted transit-depth extraction
"""

_HOMEPAGE = "https://huggingface.co/datasets/alexy-louis/ariel-exoplanet-2024"

# ---------------------------------------------------------------------------
# Required keys present in every valid .npz file
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "airs_norm",
    "fgs1_norm",
    "aux",
    "transit_depth",
    "transit_depth_err",
}

_OPTIONAL_LABEL_KEYS = {"target_mean", "target_std"}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class ArielExoplanetDataset(datasets.GeneratorBasedBuilder):
    """HuggingFace Datasets builder for the Ariel Exoplanet 2024 challenge."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default config — loads train and test splits.",
        )
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                # Planet identifier (integer string, e.g. "123")
                "planet_id": datasets.Value("string"),
                # AIRS-CH0 spectrometer: (time_binned, 356) normalised flux
                # NOTE: shape=(None, 356) means variable time dimension across planets
                "airs_norm": datasets.Array2D(shape=(None, 356), dtype="float32"),
                # FGS1 broadband channel: (time_binned,)
                "fgs1_norm": datasets.Sequence(datasets.Value("float32")),
                # Auxiliary stellar / planetary parameters (9 values)
                "aux": datasets.Sequence(datasets.Value("float32"), length=9),
                # Transit depth spectrum extracted from AIRS (356 channels)
                "transit_depth": datasets.Sequence(
                    datasets.Value("float32"), length=356
                ),
                # 1-sigma uncertainty on transit depth (356 channels)
                "transit_depth_err": datasets.Sequence(
                    datasets.Value("float32"), length=356
                ),
                # Optional — present only for labelled train planets
                # Atmospheric spectrum median (283 output wavelength channels)
                "target_mean": datasets.Sequence(
                    datasets.Value("float32"), length=283
                ),
                # Atmospheric spectrum uncertainty ≈ (q3 - q1) / 2
                "target_std": datasets.Sequence(
                    datasets.Value("float32"), length=283
                ),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        """Return train and test SplitGenerators pointing at local .npz directories."""

        # The preprocessed .npz files are stored in:
        #   data/preprocessed/train/   (within the dataset repository)
        #   data/preprocessed/test/
        #
        # dl_manager.manual_dir is the local path to the downloaded dataset repo
        # (or the Hub cache directory).  Fall back to the script's parent dir.
        if dl_manager.manual_dir:
            base_dir = Path(dl_manager.manual_dir)
        else:
            base_dir = Path(self.base_path)

        train_dir = base_dir / "data" / "preprocessed" / "train"
        test_dir = base_dir / "data" / "preprocessed" / "test"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"npz_dir": str(train_dir), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"npz_dir": str(test_dir), "split": "test"},
            ),
        ]

    def _generate_examples(
        self, npz_dir: str, split: str
    ) -> Iterator[Tuple[int, dict]]:
        """
        Yield (key, example) pairs from sorted .npz files in `npz_dir`.

        Each .npz file is named `{planet_id}.npz` and must contain at minimum:
            airs_norm, fgs1_norm, aux, transit_depth, transit_depth_err

        Files that are missing required keys are skipped with a warning.
        `target_mean` and `target_std` are optional — emitted as empty lists
        when not present so that the feature schema is satisfied.
        """
        npz_path = Path(npz_dir)

        if not npz_path.exists():
            logger.warning(
                "Preprocessed directory does not exist: %s  "
                "(did you run the upload notebook first?)",
                npz_dir,
            )
            return

        # Sort for reproducibility — ensures the same global index every run
        npz_files = sorted(npz_path.glob("*.npz"))

        if not npz_files:
            logger.warning(
                "No .npz files found in %s for split=%s", npz_dir, split
            )
            return

        for idx, npz_file in enumerate(npz_files):
            planet_id = npz_file.stem  # filename without extension

            try:
                data = np.load(npz_file, allow_pickle=False)
            except Exception as exc:
                logger.warning(
                    "Failed to load %s: %s — skipping.", npz_file.name, exc
                )
                continue

            # Validate required keys
            missing = _REQUIRED_KEYS - set(data.files)
            if missing:
                logger.warning(
                    "Planet %s (%s) is missing required keys %s — skipping.",
                    planet_id,
                    npz_file.name,
                    missing,
                )
                continue

            # Build the example dict
            example = {
                "planet_id": planet_id,
                # airs_norm is (time_binned, 356) — keep as-is; HF Array2D accepts it
                "airs_norm": data["airs_norm"].astype(np.float32),
                "fgs1_norm": data["fgs1_norm"].astype(np.float32).tolist(),
                "aux": data["aux"].astype(np.float32).tolist(),
                "transit_depth": data["transit_depth"].astype(np.float32).tolist(),
                "transit_depth_err": data["transit_depth_err"]
                .astype(np.float32)
                .tolist(),
            }

            # Optional label keys — emit empty lists when absent (schema compatibility)
            if "target_mean" in data.files and "target_std" in data.files:
                example["target_mean"] = data["target_mean"].astype(np.float32).tolist()
                example["target_std"] = data["target_std"].astype(np.float32).tolist()
            else:
                example["target_mean"] = []
                example["target_std"] = []

            yield idx, example
