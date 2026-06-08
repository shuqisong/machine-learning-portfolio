"""
Download the MovieLens dataset from Kaggle into data/raw/.

The MovieLens "small latest" dataset (~100,000 ratings from 600 users on
9,000 movies) is the classic teaching dataset for recommendation systems:
small enough to run on a laptop, rich enough to learn every core technique.

Prerequisites
-------------
1. A Kaggle account.
2. A Kaggle API token saved at ~/.kaggle/kaggle.json  (see the project README).
3. The `kaggle` package installed (it's in requirements.txt).

Usage
-----
    python src/download_data.py
"""

from pathlib import Path
import subprocess
import sys

# Kaggle dataset slug ("owner/dataset-name"). This one ships the official
# MovieLens ml-latest-small files (movies.csv, ratings.csv, tags.csv, links.csv).
KAGGLE_DATASET = "shubhammehta21/movie-lens-small-latest-dataset"

# Resolve data/raw/ relative to this file, so the script works from anywhere.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Use the `kaggle` CLI that lives next to the current Python interpreter.
# This works whether or not the venv is "activated", as long as you run the
# script with the venv's python (e.g. .venv/bin/python src/download_data.py).
KAGGLE_BIN = Path(sys.executable).parent / "kaggle"
KAGGLE_CMD = str(KAGGLE_BIN) if KAGGLE_BIN.exists() else "kaggle"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{KAGGLE_DATASET}' into {RAW_DIR} ...")
    try:
        subprocess.run(
            [
                KAGGLE_CMD, "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(RAW_DIR),
                "--unzip",
            ],
            check=True,
        )
    except FileNotFoundError:
        sys.exit(
            "ERROR: the `kaggle` command was not found.\n"
            "Activate the venv first:  source .venv/bin/activate"
        )
    except subprocess.CalledProcessError:
        sys.exit(
            "ERROR: Kaggle download failed.\n"
            "Most likely your API token is missing or not yet accepted.\n"
            "See the 'Get the data' section of the project README."
        )

    print("\nDone. Files now in data/raw/:")
    for f in sorted(RAW_DIR.glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
