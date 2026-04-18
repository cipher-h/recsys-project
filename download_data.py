"""
download_data.py — Dataset download script
============================================
Downloads required datasets into data/raw/.

Usage:
    python download_data.py --dataset movielens-1m
    python download_data.py --dataset lastfm
    python download_data.py --dataset all
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

DATASETS = {
    "movielens-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "subdir": "movielens-1m",
        "check_file": "ml-1m/ratings.dat",
    },
    "lastfm": {
        "url": "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
        "subdir": "lastfm",
        "check_file": "user_artists.dat",
    },
}


def download_and_extract(name: str):
    info = DATASETS[name]
    target_dir = os.path.join(DATA_DIR, info["subdir"])
    os.makedirs(target_dir, exist_ok=True)

    check_path = os.path.join(target_dir, info["check_file"])
    if os.path.exists(check_path):
        logger.info(f"[{name}] Already exists, skipping: {check_path}")
        return

    zip_path = os.path.join(target_dir, "download.zip")
    logger.info(f"[{name}] Downloading from: {info['url']}")

    def reporthook(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        sys.stdout.write(f"\r  Progress: {pct:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(info["url"], zip_path, reporthook=reporthook)
    print()

    logger.info(f"[{name}] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    os.remove(zip_path)
    logger.info(f"[{name}] Done: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download recommender system datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to download",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            download_and_extract(name)
    else:
        download_and_extract(args.dataset)

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
