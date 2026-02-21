#!/usr/bin/env bash
# Download competition data via Kaggle API (requires ~/.kaggle/kaggle.json)
# NOTE: ~180GB — run on Kaggle directly or on a machine with sufficient disk.

set -e
pip install kaggle --quiet
kaggle competitions download -c ariel-data-challenge-2024
unzip ariel-data-challenge-2024.zip -d data/raw/ && rm ariel-data-challenge-2024.zip
echo "Download complete. Raw data in data/raw/"
