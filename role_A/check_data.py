"""
check_data.py — Data integrity validation script
==================================================
Verifies all processed data files. Run from project root:

    python check_data.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []
warnings = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name} -- {detail}")
        errors.append(f"{name}: {detail}")


def warn(name, detail):
    print(f"  {WARN} {name} -- {detail}")
    warnings.append(f"{name}: {detail}")


def check_raw_files():
    """Check raw data files"""
    print("\n" + "=" * 70)
    print("1. Checking raw data files")
    print("=" * 70)

    # MovieLens 1M
    ml_file = "data/raw/movielens-1m/ml-1m/ratings.dat"
    if os.path.exists(ml_file):
        print(f"\n  [MovieLens 1M] {ml_file}")
        with open(ml_file, "r", encoding="latin-1") as f:
            lines = [f.readline() for _ in range(3)]
        print(f"    First 3 lines:")
        for line in lines:
            print(f"      {line.strip()}")

        df = pd.read_csv(ml_file, sep="::",
                         names=["user_id", "item_id", "rating", "timestamp"],
                         engine="python", encoding="latin-1")
        print(f"    shape: {df.shape}")
        print(f"    columns: {list(df.columns)}")
        print(f"    dtypes:\n{df.dtypes.to_string()}")
        print(f"    user_id range: {df['user_id'].min()} ~ {df['user_id'].max()}")
        print(f"    item_id range: {df['item_id'].min()} ~ {df['item_id'].max()}")
        print(f"    rating range: {df['rating'].min()} ~ {df['rating'].max()}")
        print(f"    rating distribution:\n{df['rating'].value_counts().sort_index().to_string()}")
        check("MovieLens row count ~1,000,209", 900000 < len(df) < 1100000, f"actual {len(df)}")
        check("rating in [1, 5]", df["rating"].between(1, 5).all(),
              f"min={df['rating'].min()}, max={df['rating'].max()}")
        check("no missing values", df.isnull().sum().sum() == 0,
              f"nulls: {df.isnull().sum().to_dict()}")
    else:
        print(f"  {FAIL} MovieLens 1M raw file missing: {ml_file}")
        errors.append("MovieLens raw file missing")

    # Last.fm
    print()
    lfm_files = [
        "data/raw/lastfm/user_taggedartists-timestamps.dat",
        "data/raw/lastfm/user_artists.dat",
    ]
    for fpath in lfm_files:
        if os.path.exists(fpath):
            print(f"  [Last.fm] {fpath}")
            df = pd.read_csv(fpath, sep="\t", nrows=5)
            print(f"    columns: {list(df.columns)}")
            print(f"    first 5 rows:")
            print(f"{df.to_string()}")
            print()
        else:
            print(f"  File not found: {fpath}")


def check_splits(dataset_name, feedback_type):
    """Check splits for a dataset"""
    print(f"\n{'=' * 70}")
    print(f"Checking {dataset_name} splits ({feedback_type})")
    print(f"{'=' * 70}")

    splits_dir = f"data/splits/{dataset_name}"
    if not os.path.exists(splits_dir):
        print(f"  {FAIL} Splits directory missing: {splits_dir}")
        errors.append(f"{dataset_name} splits dir missing")
        return

    required_files = ["train.csv", "val.csv", "test.csv", "interactions.csv",
                      "user2idx.json", "item2idx.json", "metadata.json"]
    for fname in required_files:
        fpath = os.path.join(splits_dir, fname)
        check(f"File exists: {fname}", os.path.exists(fpath), f"missing: {fpath}")

    # Load
    train = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    interactions = pd.read_csv(os.path.join(splits_dir, "interactions.csv"))

    with open(os.path.join(splits_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(splits_dir, "user2idx.json"), "r") as f:
        user2idx = json.load(f)
    with open(os.path.join(splits_dir, "item2idx.json"), "r") as f:
        item2idx = json.load(f)

    print(f"\n  --- Basic statistics ---")
    print(f"    interactions: {len(interactions):,} rows")
    print(f"    train: {len(train):,} ({len(train)/len(interactions)*100:.1f}%)")
    print(f"    val: {len(val):,} ({len(val)/len(interactions)*100:.1f}%)")
    print(f"    test: {len(test):,} ({len(test)/len(interactions)*100:.1f}%)")
    print(f"    users: {meta['n_users']:,}")
    print(f"    items: {meta['n_items']:,}")
    print(f"    sparsity: {meta['sparsity']:.6f}")

    # Column check
    print(f"\n  --- Column check ---")
    expected_cols = {"user_id", "item_id", "rating", "timestamp", "label"}
    for name, df in [("train", train), ("val", val), ("test", test), ("interactions", interactions)]:
        actual_cols = set(df.columns)
        check(f"{name}.csv columns correct", actual_cols == expected_cols,
              f"expected {expected_cols}, got {actual_cols}")

    # dtypes
    print(f"\n  --- Data types ---")
    print(f"    train dtypes:\n{train.dtypes.to_string()}")

    # ID continuity
    print(f"\n  --- ID continuity ---")
    all_users = interactions["user_id"].unique()
    all_items = interactions["item_id"].unique()
    check("user_id starts from 0", all_users.min() == 0, f"min={all_users.min()}")
    check("user_id contiguous", len(all_users) == all_users.max() + 1,
          f"n_unique={len(all_users)}, max={all_users.max()}")
    check("item_id starts from 0", all_items.min() == 0, f"min={all_items.min()}")
    check("item_id contiguous", len(all_items) == all_items.max() + 1,
          f"n_unique={len(all_items)}, max={all_items.max()}")

    # Mapping consistency
    print(f"\n  --- ID mapping consistency ---")
    check("user2idx count == n_users", len(user2idx) == meta["n_users"],
          f"user2idx={len(user2idx)}, meta={meta['n_users']}")
    check("item2idx count == n_items", len(item2idx) == meta["n_items"],
          f"item2idx={len(item2idx)}, meta={meta['n_items']}")

    # Split ratio
    print(f"\n  --- Split ratio check ---")
    total = len(train) + len(val) + len(test)
    check("train+val+test == interactions", total == len(interactions),
          f"sum={total}, interactions={len(interactions)}")
    train_pct = len(train) / total * 100
    val_pct = len(val) / total * 100
    test_pct = len(test) / total * 100
    print(f"    Actual ratios: train={train_pct:.1f}% / val={val_pct:.1f}% / test={test_pct:.1f}%")
    check("train ratio ~80%", 70 < train_pct < 90, f"actual {train_pct:.1f}%")
    check("val ratio ~10%", 3 < val_pct < 20, f"actual {val_pct:.1f}%")
    check("test ratio ~10%", 3 < test_pct < 20, f"actual {test_pct:.1f}%")

    # No duplicates
    print(f"\n  --- No duplicates check ---")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        n_dup = df.duplicated(subset=["user_id", "item_id"]).sum()
        check(f"{name} no (user,item) duplicates", n_dup == 0, f"{n_dup} duplicates")

    train_pairs = set(zip(train["user_id"], train["item_id"]))
    val_pairs = set(zip(val["user_id"], val["item_id"]))
    test_pairs = set(zip(test["user_id"], test["item_id"]))
    check("train ∩ val empty", len(train_pairs & val_pairs) == 0,
          f"{len(train_pairs & val_pairs)} overlaps")
    check("train ∩ test empty", len(train_pairs & test_pairs) == 0,
          f"{len(train_pairs & test_pairs)} overlaps")
    check("val ∩ test empty", len(val_pairs & test_pairs) == 0,
          f"{len(val_pairs & test_pairs)} overlaps")

    # Critical: temporal leakage check
    print(f"\n  --- [CRITICAL] Temporal split correctness ---")
    leakage_count = 0
    checked_users = 0
    sample_users = np.random.choice(train["user_id"].unique(),
                                     size=min(500, len(train["user_id"].unique())),
                                     replace=False)

    for uid in sample_users:
        t_train = train[train["user_id"] == uid]["timestamp"]
        t_val = val[val["user_id"] == uid]["timestamp"]
        t_test = test[test["user_id"] == uid]["timestamp"]
        checked_users += 1

        if len(t_train) > 0 and len(t_val) > 0:
            if t_val.min() < t_train.max():
                leakage_count += 1
        if len(t_train) > 0 and len(t_test) > 0:
            if t_test.min() < t_train.max():
                leakage_count += 1
        if len(t_val) > 0 and len(t_test) > 0:
            if t_test.min() < t_val.max():
                leakage_count += 1

    check(f"No temporal leakage (sampled {checked_users} users)",
          leakage_count == 0, f"{leakage_count} leaks found")

    print(f"\n    Example user timelines:")
    for uid in sample_users[:3]:
        t_tr = train[train["user_id"] == uid]["timestamp"]
        t_va = val[val["user_id"] == uid]["timestamp"]
        t_te = test[test["user_id"] == uid]["timestamp"]
        if len(t_va) > 0 and len(t_te) > 0:
            print(f"      user {uid}: train=[{t_tr.min():.0f}~{t_tr.max():.0f}] "
                  f"val=[{t_va.min():.0f}~{t_va.max():.0f}] "
                  f"test=[{t_te.min():.0f}~{t_te.max():.0f}]")
        else:
            print(f"      user {uid}: train={len(t_tr)}, val={len(t_va)}, test={len(t_te)}")

    # Label
    print(f"\n  --- Label check ---")
    if feedback_type == "explicit":
        print(f"    Binarization threshold: rating >= {meta.get('relevance_threshold', 4)}")
        print(f"    Rating distribution:\n{interactions['rating'].value_counts().sort_index().to_string()}")
        threshold = meta.get("relevance_threshold", 4)
        expected_label = (interactions["rating"] >= threshold).astype(int)
        check("label consistent with rating", (interactions["label"] == expected_label).all(),
              "label != (rating >= threshold)")
        n_pos = (interactions["label"] == 1).sum()
        n_neg = (interactions["label"] == 0).sum()
        print(f"    label=1 (relevant):     {n_pos:,} ({n_pos/len(interactions)*100:.1f}%)")
        print(f"    label=0 (not relevant): {n_neg:,} ({n_neg/len(interactions)*100:.1f}%)")
    else:
        check("implicit label all 1", (interactions["label"] == 1).all(),
              f"label distribution: {interactions['label'].value_counts().to_dict()}")
        check("implicit rating all 1.0", (interactions["rating"] == 1.0).all(),
              f"rating distribution: {interactions['rating'].value_counts().to_dict()}")

    # k-core
    print(f"\n  --- k-core filtering check ---")
    user_counts = interactions.groupby("user_id").size()
    item_counts = interactions.groupby("item_id").size()
    min_user = user_counts.min()
    min_item = item_counts.min()
    check("each user has >= 5 interactions", min_user >= 5, f"min {min_user}")
    check("each item has >= 5 interactions", min_item >= 5, f"min {min_item}")

    # Cold-start
    print(f"\n  --- Cold-start user check ---")
    train_user_counts = train.groupby("user_id").size()
    cold_users = train_user_counts[train_user_counts < 5]
    all_data_users = set(interactions["user_id"].unique())
    train_users = set(train["user_id"].unique())
    unseen_users = all_data_users - train_users
    print(f"    Users with < 5 train interactions: {len(cold_users)}")
    print(f"    Users missing from train set:      {len(unseen_users)}")
    total_cold = len(cold_users) + len(unseen_users)
    print(f"    Total cold-start users:            {total_cold}")
    print(f"    Regular users:                     {len(train_users) - len(cold_users)}")

    # metadata
    print(f"\n  --- metadata.json ---")
    print(f"    Content:")
    for k, v in meta.items():
        print(f"      {k}: {v}")
    check("metadata.n_users consistent", meta["n_users"] == len(all_users),
          f"meta={meta['n_users']}, actual={len(all_users)}")
    check("metadata.n_items consistent", meta["n_items"] == len(all_items),
          f"meta={meta['n_items']}, actual={len(all_items)}")
    check("metadata.train_size consistent", meta["train_size"] == len(train),
          f"meta={meta['train_size']}, actual={len(train)}")

    # Samples
    print(f"\n  --- Sample rows ---")
    print(f"    train head:\n{train.head().to_string()}")
    print(f"    val head:\n{val.head().to_string()}")
    print(f"    test head:\n{test.head().to_string()}")


def check_eda_plots():
    """Check EDA plots"""
    print(f"\n{'=' * 70}")
    print("Checking EDA plots")
    print("=" * 70)
    eda_dir = "outputs/eda"
    expected = ["eda_movielens-1m.png", "cold_start_movielens-1m.png",
                "eda_lastfm.png", "cold_start_lastfm.png"]
    for fname in expected:
        fpath = os.path.join(eda_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            check(f"{fname} exists ({size/1024:.0f} KB)", size > 1000,
                  f"file too small: {size} bytes")
        else:
            warn(fname, "file missing (OK if Last.fm not processed)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("=" * 70)
    print("Recommender System Data Integrity Check")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 70)

    np.random.seed(42)

    check_raw_files()

    if os.path.exists("data/splits/movielens-1m/train.csv"):
        check_splits("movielens-1m", "explicit")
    else:
        print(f"\n  {FAIL} MovieLens 1M splits missing")
        errors.append("MovieLens splits missing")

    if os.path.exists("data/splits/lastfm/train.csv"):
        check_splits("lastfm", "implicit")
    else:
        print(f"\n  {WARN} Last.fm splits missing (fix and re-run run_data_pipeline.py)")
        warnings.append("Last.fm splits missing")

    check_eda_plots()

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    if errors:
        print(f"\n{FAIL} Found {len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print(f"\n{WARN} Found {len(warnings)} warnings:")
        for w in warnings:
            print(f"  - {w}")
    if not errors and not warnings:
        print(f"\n{PASS} All checks passed.")
    elif not errors:
        print(f"\n{PASS} No errors. Warnings can be addressed later.")
    print()


if __name__ == "__main__":
    main()
