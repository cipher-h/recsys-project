"""
run_data_pipeline.py — Full data pipeline runner
==================================================
Downloads, processes, and splits the datasets; generates EDA plots.

Usage:
    python run_data_pipeline.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import RecommenderDataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_eda(loader: RecommenderDataLoader, output_dir: str):
    """Generate EDA visualizations for a dataset"""
    os.makedirs(output_dir, exist_ok=True)
    train, val, test = loader.get_splits()
    dataset_name = loader.dataset

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exploratory Data Analysis: {dataset_name}", fontsize=16, fontweight="bold")

    # 1. User interaction count distribution
    ax = axes[0, 0]
    user_counts = train.groupby("user_id").size()
    ax.hist(user_counts, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Number of Interactions")
    ax.set_ylabel("Number of Users")
    ax.set_title("User Interaction Count Distribution (Train)")
    ax.axvline(user_counts.median(), color="red", linestyle="--",
               label=f"Median={user_counts.median():.0f}")
    ax.legend()

    # 2. Item popularity distribution
    ax = axes[0, 1]
    item_counts = train.groupby("item_id").size()
    sorted_counts = item_counts.sort_values(ascending=False).values
    ax.plot(range(len(sorted_counts)), sorted_counts, color="#DD8452", linewidth=1.5)
    ax.set_xlabel("Item Rank (by popularity)")
    ax.set_ylabel("Number of Interactions")
    ax.set_title("Item Popularity Distribution (Long Tail)")
    ax.set_yscale("log")

    # 3. Rating distribution (explicit) or cumulative distribution (implicit)
    ax = axes[1, 0]
    if loader.feedback_type == "explicit":
        ratings = loader.interactions["rating"]
        ax.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                color="#55A868", edgecolor="white", alpha=0.8, rwidth=0.8)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title("Rating Distribution")
        ax.axvline(loader.relevance_threshold, color="red", linestyle="--",
                   label=f"Relevance threshold >= {loader.relevance_threshold}")
        ax.legend()
    else:
        ax.hist(user_counts, bins=50, color="#55A868", edgecolor="white", alpha=0.8,
                cumulative=True, density=True)
        ax.set_xlabel("Number of Interactions")
        ax.set_ylabel("Cumulative Proportion")
        ax.set_title("Cumulative User Interaction Distribution")

    # 4. Split sizes
    ax = axes[1, 1]
    split_sizes = [len(train), len(val), len(test)]
    split_labels = ["Train", "Validation", "Test"]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    bars = ax.bar(split_labels, split_sizes, color=colors, edgecolor="white", alpha=0.8)
    for bar, size in zip(bars, split_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{size:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of Interactions")
    ax.set_title("Temporal Split Sizes")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"eda_{dataset_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  EDA plot saved: {fig_path}")

    # 5. Cold-start users
    fig, ax = plt.subplots(figsize=(8, 5))
    cold_info = loader.get_cold_start_users()
    ax.bar(
        ["Cold-start Users", "Regular Users"],
        [cold_info["cold_start_count"], cold_info["regular_count"]],
        color=["#C44E52", "#4C72B0"], edgecolor="white", alpha=0.8,
    )
    ax.set_ylabel("Number of Users")
    ax.set_title(f"Cold-start Analysis (threshold < {cold_info['threshold']} interactions)")
    for i, v in enumerate([cold_info["cold_start_count"], cold_info["regular_count"]]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=12)
    fig_path = os.path.join(output_dir, f"cold_start_{dataset_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Cold-start plot saved: {fig_path}")


def demo_negative_sampling(loader: RecommenderDataLoader):
    """Demonstrate negative sampling"""
    if loader.feedback_type != "implicit":
        return

    print("\n  === Negative sampling demo ===")
    train, _, _ = loader.get_splits()

    for num_neg in [1, 4, 10]:
        sampled = loader.get_negative_samples(train, num_neg=num_neg, strategy="uniform")
        pos = (sampled["label"] == 1).sum()
        neg = (sampled["label"] == 0).sum()
        print(f"  num_neg={num_neg}: positives={pos:,}, negatives={neg:,}, ratio=1:{neg/pos:.1f}")

    sampled_pop = loader.get_negative_samples(train, num_neg=4, strategy="popularity")
    print(f"  popularity strategy (num_neg=4): total={len(sampled_pop):,}")


def demo_pytorch_dataset(loader: RecommenderDataLoader):
    """Demonstrate PyTorch Dataset interface"""
    try:
        import torch  # noqa
        dataset = loader.get_torch_dataset(split="train")
        print(f"\n  === PyTorch Dataset ===")
        print(f"  Train set size: {len(dataset)}")
        sample = dataset[0]
        print(f"  Sample: user_id={sample['user_id'].item()}, "
              f"item_id={sample['item_id'].item()}, "
              f"label={sample['label'].item()}, "
              f"rating={sample['rating'].item()}")
    except ImportError:
        print("\n  PyTorch not installed, skipping Dataset demo")


def main():
    print("=" * 70)
    print("Recommender System Data Pipeline")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    eda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "eda")

    # =============================================
    # MovieLens 1M (explicit feedback)
    # =============================================
    print("\n" + "=" * 70)
    print("[1/2] Processing MovieLens 1M (explicit feedback)")
    print("=" * 70)

    try:
        ml_loader = RecommenderDataLoader(
            dataset="movielens-1m",
            data_dir=data_dir,
            relevance_threshold=4,
        )
        ml_loader.describe()
        plot_eda(ml_loader, eda_dir)
        demo_pytorch_dataset(ml_loader)

        gt = ml_loader.get_all_test_ground_truth()
        print(f"\n  Users with ground-truth items in test set: {len(gt):,}")

    except Exception as e:
        print(f"  MovieLens 1M processing failed: {e}")
        print("  Run first: python download_data.py --dataset movielens-1m")

    # =============================================
    # Last.fm (implicit feedback)
    # =============================================
    print("\n" + "=" * 70)
    print("[2/2] Processing Last.fm (implicit feedback)")
    print("=" * 70)

    try:
        lfm_loader = RecommenderDataLoader(
            dataset="lastfm",
            data_dir=data_dir,
        )
        lfm_loader.describe()
        plot_eda(lfm_loader, eda_dir)
        demo_negative_sampling(lfm_loader)
        demo_pytorch_dataset(lfm_loader)

    except Exception as e:
        print(f"  Last.fm processing failed: {e}")
        print("  Run first: python download_data.py --dataset lastfm")

    print("\n" + "=" * 70)
    print("Data pipeline complete.")
    print(f"Splits directory: {os.path.join(data_dir, 'splits')}")
    print(f"EDA plots directory: {eda_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
