"""
data_loader.py — Unified Data Loader for Recommender System Benchmarks
========================================================================
Handles dataset loading, cleaning, temporal splitting, negative sampling,
and cold-start user identification for MovieLens and Last.fm datasets.

Usage:
    from src.data_loader import RecommenderDataLoader

    # Explicit feedback - MovieLens 1M
    ml_loader = RecommenderDataLoader(dataset="movielens-1m", data_dir="data")
    train, val, test = ml_loader.get_splits()

    # Implicit feedback - Last.fm
    lfm_loader = RecommenderDataLoader(dataset="lastfm", data_dir="data")
    train, val, test = lfm_loader.get_splits()

    # Get negative samples for training (implicit feedback)
    neg_samples = lfm_loader.get_negative_samples(train, num_neg=4)

    # Get cold-start users
    cold_users = ml_loader.get_cold_start_users(threshold=5)

    # PyTorch Dataset
    torch_dataset = ml_loader.get_torch_dataset(split="train")
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Binarization threshold: rating >= RELEVANCE_THRESHOLD is considered "relevant"
# (assignment requirement: rating >= 4)
RELEVANCE_THRESHOLD = 4

# Temporal split ratios: 80% train / 10% val / 10% test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Cold-start user definition: users with < COLD_START_THRESHOLD interactions in train set
COLD_START_THRESHOLD = 5

# Random seed
RANDOM_SEED = 42


class RecommenderDataLoader:
    """
    Unified data loader for recommender systems.

    Supported datasets:
      - "movielens-1m":  Explicit feedback (ratings 1-5)
      - "lastfm":        Implicit feedback (play counts)
      - "amazon-digital-music": Implicit feedback (purchases/reviews)

    Core features:
      1. Dataset download and cleaning
      2. User/item ID remapping to contiguous integers
      3. Temporal train/val/test splitting (per-user timeline)
      4. Negative sampling (for implicit feedback)
      5. Cold-start user identification
      6. PyTorch Dataset interface
    """

    SUPPORTED_DATASETS = ["movielens-1m", "lastfm", "amazon-digital-music"]

    def __init__(
        self,
        dataset: str,
        data_dir: str = "data",
        relevance_threshold: int = RELEVANCE_THRESHOLD,
        cold_start_threshold: int = COLD_START_THRESHOLD,
        random_seed: int = RANDOM_SEED,
        force_reprocess: bool = False,
    ):
        """
        Args:
            dataset: Dataset name, see SUPPORTED_DATASETS
            data_dir: Root data directory
            relevance_threshold: Binarization threshold for explicit feedback
                                 (ratings >= this value are relevant)
            cold_start_threshold: Cold-start user threshold
                                  (users with < this many train interactions)
            random_seed: Random seed for reproducibility
            force_reprocess: Force re-processing (ignore cache)
        """
        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}. Choose from {self.SUPPORTED_DATASETS}")

        self.dataset = dataset
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw", dataset)
        self.processed_dir = os.path.join(data_dir, "processed", dataset)
        self.splits_dir = os.path.join(data_dir, "splits", dataset)
        self.relevance_threshold = relevance_threshold
        self.cold_start_threshold = cold_start_threshold
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Create directories
        for d in [self.raw_dir, self.processed_dir, self.splits_dir]:
            os.makedirs(d, exist_ok=True)

        # Data attributes (populated after loading)
        self._interactions: Optional[pd.DataFrame] = None
        self._user2idx: Optional[Dict[Any, int]] = None
        self._idx2user: Optional[Dict[int, Any]] = None
        self._item2idx: Optional[Dict[Any, int]] = None
        self._idx2item: Optional[Dict[int, Any]] = None
        self._train: Optional[pd.DataFrame] = None
        self._val: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None
        self._metadata: Optional[Dict] = None

        # Load or process data
        self._load_or_process(force_reprocess)

    # ============================================================
    # Public interface
    # ============================================================

    @property
    def feedback_type(self) -> str:
        """Return feedback type: 'explicit' or 'implicit'"""
        if self.dataset == "movielens-1m":
            return "explicit"
        return "implicit"

    @property
    def n_users(self) -> int:
        return len(self._user2idx)

    @property
    def n_items(self) -> int:
        return len(self._item2idx)

    @property
    def interactions(self) -> pd.DataFrame:
        """Return full interaction data (cleaned, ID remapped)"""
        return self._interactions.copy()

    @property
    def metadata(self) -> Dict:
        """Return dataset metadata"""
        return self._metadata.copy()

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return temporally split (train, val, test) DataFrames.

        Each DataFrame contains columns:
          - user_id (int): remapped user ID
          - item_id (int): remapped item ID
          - rating (float): original rating (explicit) or 1.0 (implicit)
          - timestamp (int): Unix timestamp
          - label (int): binary label (1=relevant, 0=not relevant)
        """
        return self._train.copy(), self._val.copy(), self._test.copy()

    def get_user_mapping(self) -> Tuple[Dict[Any, int], Dict[int, Any]]:
        """Return (original_id -> new_id, new_id -> original_id) mappings"""
        return self._user2idx.copy(), self._idx2user.copy()

    def get_item_mapping(self) -> Tuple[Dict[Any, int], Dict[int, Any]]:
        """Return (original_id -> new_id, new_id -> original_id) mappings"""
        return self._item2idx.copy(), self._idx2item.copy()

    def get_cold_start_users(self, threshold: Optional[int] = None) -> Dict[str, List[int]]:
        """
        Get cold-start users (those with < threshold interactions in training set).

        Returns:
            dict: {
                "cold_start_user_ids": [list of user IDs],
                "regular_user_ids": [list of regular user IDs],
                "cold_start_count": int,
                "regular_count": int,
                "threshold": int
            }
        """
        if threshold is None:
            threshold = self.cold_start_threshold

        train_user_counts = self._train.groupby("user_id").size()

        # Cold-start users: < threshold train interactions
        cold_users = train_user_counts[train_user_counts < threshold].index.tolist()

        # Also include users appearing only in val/test but not in train
        all_users_in_data = set(self._interactions["user_id"].unique())
        train_users = set(self._train["user_id"].unique())
        unseen_users = list(all_users_in_data - train_users)

        cold_users = list(set(cold_users + unseen_users))
        regular_users = [u for u in train_user_counts.index if u not in set(cold_users)]

        return {
            "cold_start_user_ids": sorted(cold_users),
            "regular_user_ids": sorted(regular_users),
            "cold_start_count": len(cold_users),
            "regular_count": len(regular_users),
            "threshold": threshold,
        }

    def get_negative_samples(
        self,
        data: Optional[pd.DataFrame] = None,
        num_neg: int = 4,
        strategy: str = "uniform",
    ) -> pd.DataFrame:
        """
        Generate negative samples for implicit feedback data.

        Args:
            data: Input positive samples DataFrame (default: training set)
            num_neg: Number of negatives per positive sample
            strategy: "uniform" (random) or "popularity" (popularity-weighted)

        Returns:
            DataFrame with positives and negatives, label column marking 1/0
        """
        if data is None:
            data = self._train

        all_items = set(range(self.n_items))

        # Build per-user set of interacted items
        user_pos_items = data.groupby("user_id")["item_id"].apply(set).to_dict()

        if strategy == "popularity":
            # Sample by item popularity (frequency)
            item_counts = data["item_id"].value_counts()
            item_probs = item_counts / item_counts.sum()
            pop_items = item_probs.index.values
            pop_probs = item_probs.values

        neg_records = []
        for _, row in data.iterrows():
            uid = row["user_id"]
            pos_items = user_pos_items.get(uid, set())
            candidate_items = list(all_items - pos_items)

            if len(candidate_items) == 0:
                continue

            actual_neg = min(num_neg, len(candidate_items))

            if strategy == "uniform":
                sampled = self.rng.choice(candidate_items, size=actual_neg, replace=False)
            elif strategy == "popularity":
                cand_set = set(candidate_items)
                mask = np.array([item in cand_set for item in pop_items])
                filtered_items = pop_items[mask]
                filtered_probs = pop_probs[mask]
                if len(filtered_items) == 0:
                    continue
                filtered_probs = filtered_probs / filtered_probs.sum()
                actual_neg = min(actual_neg, len(filtered_items))
                sampled = self.rng.choice(
                    filtered_items, size=actual_neg, replace=False, p=filtered_probs
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            for neg_item in sampled:
                neg_records.append({
                    "user_id": uid,
                    "item_id": neg_item,
                    "rating": 0.0,
                    "timestamp": row["timestamp"],
                    "label": 0,
                })

        neg_df = pd.DataFrame(neg_records)

        # Combine positives and negatives
        pos_df = data.copy()
        pos_df["label"] = 1

        combined = pd.concat([pos_df, neg_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        logger.info(
            f"Negative sampling done: {len(pos_df)} positives + {len(neg_df)} negatives "
            f"= {len(combined)} total (strategy={strategy}, num_neg={num_neg})"
        )
        return combined

    def get_user_item_matrix(self, split: str = "train", binary: bool = False):
        """
        Build user-item interaction matrix (sparse).

        Args:
            split: "train", "val", "test", or "all"
            binary: whether to binarize (True: interacted=1, else 0)

        Returns:
            scipy.sparse.csr_matrix of shape (n_users, n_items)
        """
        import scipy.sparse as sp

        if split == "train":
            df = self._train
        elif split == "val":
            df = self._val
        elif split == "test":
            df = self._test
        elif split == "all":
            df = self._interactions
        else:
            raise ValueError(f"Unknown split: {split}")

        if binary:
            values = np.ones(len(df))
        else:
            values = df["rating"].values

        matrix = sp.csr_matrix(
            (values, (df["user_id"].values, df["item_id"].values)),
            shape=(self.n_users, self.n_items),
        )
        return matrix

    def get_torch_dataset(self, split: str = "train", num_neg: int = 0):
        """
        Return a PyTorch Dataset.

        Args:
            split: "train", "val", "test"
            num_neg: number of negatives (0 = no negative sampling)
        """
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        if split == "train":
            df = self._train
        elif split == "val":
            df = self._val
        elif split == "test":
            df = self._test
        else:
            raise ValueError(f"Unknown split: {split}")

        if num_neg > 0 and self.feedback_type == "implicit":
            df = self.get_negative_samples(df, num_neg=num_neg)

        class RecDataset(Dataset):
            def __init__(self, dataframe):
                self.users = torch.LongTensor(dataframe["user_id"].values)
                self.items = torch.LongTensor(dataframe["item_id"].values)
                self.labels = torch.FloatTensor(dataframe["label"].values)
                self.ratings = torch.FloatTensor(dataframe["rating"].values)

            def __len__(self):
                return len(self.users)

            def __getitem__(self, idx):
                return {
                    "user_id": self.users[idx],
                    "item_id": self.items[idx],
                    "label": self.labels[idx],
                    "rating": self.ratings[idx],
                }

        return RecDataset(df)

    def get_test_candidates(self, user_id: int) -> List[int]:
        """Get the ground-truth relevant items for a user in the test set"""
        test_items = self._test[self._test["user_id"] == user_id]
        if self.feedback_type == "explicit":
            return test_items[test_items["label"] == 1]["item_id"].tolist()
        else:
            return test_items["item_id"].tolist()

    def get_all_test_ground_truth(self) -> Dict[int, List[int]]:
        """
        Get ground-truth relevant items for all test users (for evaluation).

        Returns:
            {user_id: [list of relevant item_ids]}
        """
        if self.feedback_type == "explicit":
            relevant = self._test[self._test["label"] == 1]
        else:
            relevant = self._test

        return relevant.groupby("user_id")["item_id"].apply(list).to_dict()

    def get_user_history(self, user_id: int, split: str = "train") -> pd.DataFrame:
        """Get a user's interaction history (sorted by time)"""
        if split == "train":
            df = self._train
        elif split == "all":
            df = self._interactions
        else:
            raise ValueError(f"Unknown split: {split}")

        return df[df["user_id"] == user_id].sort_values("timestamp")

    def describe(self) -> str:
        """Print dataset summary statistics"""
        info = []
        info.append(f"{'='*60}")
        info.append(f"Dataset: {self.dataset}")
        info.append(f"Feedback type: {self.feedback_type}")
        info.append(f"{'='*60}")
        info.append(f"Users:        {self.n_users:,}")
        info.append(f"Items:        {self.n_items:,}")
        info.append(f"Interactions: {len(self._interactions):,}")
        info.append(f"Sparsity:     {1 - len(self._interactions) / (self.n_users * self.n_items):.6f}")
        info.append(f"")
        info.append(f"Split statistics:")
        info.append(f"  Train: {len(self._train):,} ({len(self._train)/len(self._interactions)*100:.1f}%)")
        info.append(f"  Val:   {len(self._val):,} ({len(self._val)/len(self._interactions)*100:.1f}%)")
        info.append(f"  Test:  {len(self._test):,} ({len(self._test)/len(self._interactions)*100:.1f}%)")
        info.append(f"")

        if self.feedback_type == "explicit":
            info.append(f"Rating statistics:")
            info.append(f"  Range:    {self._interactions['rating'].min():.0f} - {self._interactions['rating'].max():.0f}")
            info.append(f"  Mean:     {self._interactions['rating'].mean():.2f}")
            info.append(f"  Relevance threshold: >= {self.relevance_threshold}")
            n_relevant = (self._interactions["label"] == 1).sum()
            info.append(f"  Relevant fraction: {n_relevant/len(self._interactions)*100:.1f}%")
        else:
            info.append(f"Implicit feedback (all interactions are positive)")

        info.append(f"")
        cold_info = self.get_cold_start_users()
        info.append(f"Cold-start analysis (threshold < {cold_info['threshold']} interactions):")
        info.append(f"  Cold-start users: {cold_info['cold_start_count']:,}")
        info.append(f"  Regular users:    {cold_info['regular_count']:,}")
        info.append(f"")

        train_user_counts = self._train.groupby("user_id").size()
        info.append(f"User interaction distribution (train):")
        info.append(f"  Min:    {train_user_counts.min()}")
        info.append(f"  Median: {train_user_counts.median():.0f}")
        info.append(f"  Mean:   {train_user_counts.mean():.1f}")
        info.append(f"  Max:    {train_user_counts.max()}")
        info.append(f"{'='*60}")

        result = "\n".join(info)
        print(result)
        return result

    # ============================================================
    # Internal methods
    # ============================================================

    def _load_or_process(self, force_reprocess: bool):
        """Load from cache or process from scratch"""
        cache_file = os.path.join(self.processed_dir, "processed_data.pkl")

        if not force_reprocess and os.path.exists(cache_file):
            logger.info(f"Loading from cache: {cache_file}")
            self._load_cache(cache_file)
            return

        logger.info(f"Processing dataset: {self.dataset}")

        # Step 1: Load raw data
        raw_df = self._load_raw_data()
        logger.info(f"Raw data loaded: {len(raw_df):,} interactions")

        # Step 2: Clean data
        clean_df = self._clean_data(raw_df)
        logger.info(f"Data cleaning done: {len(clean_df):,} interactions")

        # Step 3: ID remapping
        clean_df, user2idx, idx2user, item2idx, idx2item = self._remap_ids(clean_df)
        logger.info(f"ID remapping done: {len(user2idx)} users, {len(item2idx)} items")

        # Step 4: Add label column
        clean_df = self._add_labels(clean_df)

        # Step 5: Temporal split
        train, val, test = self._temporal_split(clean_df)
        logger.info(
            f"Temporal split done: train={len(train):,}, val={len(val):,}, test={len(test):,}"
        )

        # Store
        self._interactions = clean_df
        self._user2idx = user2idx
        self._idx2user = idx2user
        self._item2idx = item2idx
        self._idx2item = idx2item
        self._train = train
        self._val = val
        self._test = test
        self._metadata = self._build_metadata()

        # Save cache
        self._save_cache(cache_file)

        # Save splits as CSV (so B and C can use directly)
        self._save_splits_csv()

        logger.info("Data processing complete!")

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw data"""
        if self.dataset == "movielens-1m":
            return self._load_movielens_1m()
        elif self.dataset == "lastfm":
            return self._load_lastfm()
        elif self.dataset == "amazon-digital-music":
            return self._load_amazon_digital_music()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def _load_movielens_1m(self) -> pd.DataFrame:
        """
        Load MovieLens 1M dataset.

        File format: UserID::MovieID::Rating::Timestamp
        Expected path: data/raw/movielens-1m/ml-1m/ratings.dat
        """
        ratings_file = os.path.join(self.raw_dir, "ml-1m", "ratings.dat")

        if not os.path.exists(ratings_file):
            self._download_movielens_1m()

        logger.info(f"Reading MovieLens 1M: {ratings_file}")
        df = pd.read_csv(
            ratings_file,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )
        return df

    def _download_movielens_1m(self):
        """Download MovieLens 1M dataset"""
        import urllib.request
        import zipfile

        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(self.raw_dir, "ml-1m.zip")

        logger.info(f"Downloading MovieLens 1M from {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MovieLens 1M. Please download manually:\n"
                f"  1. Go to https://grouplens.org/datasets/movielens/1m/\n"
                f"  2. Download ml-1m.zip\n"
                f"  3. Extract to {self.raw_dir}/ml-1m/\n"
                f"  Expected file: {self.raw_dir}/ml-1m/ratings.dat\n"
                f"Error: {e}"
            )

        logger.info("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.raw_dir)
        os.remove(zip_path)
        logger.info("MovieLens 1M downloaded and extracted")

    def _load_lastfm(self) -> pd.DataFrame:
        """
        Load Last.fm HetRec 2011 dataset.

        Expected files in data/raw/lastfm/:
          - user_taggedartists-timestamps.dat (with timestamps, preferred)
          - user_artists.dat (with play counts, fallback)
        """
        timestamped_file = os.path.join(self.raw_dir, "user_taggedartists-timestamps.dat")
        artists_file = os.path.join(self.raw_dir, "user_artists.dat")

        if not os.path.exists(artists_file) and not os.path.exists(timestamped_file):
            self._download_lastfm()

        # Preferred: file with real timestamps
        if os.path.exists(timestamped_file):
            logger.info(f"Reading Last.fm (with timestamps): {timestamped_file}")
            df_tagged = pd.read_csv(timestamped_file, sep="\t")
            logger.info(f"  Columns: {list(df_tagged.columns)}")

            # Actual columns: userID, artistID, tagID, timestamp (milliseconds)
            if "timestamp" in df_tagged.columns:
                df_tagged["ts"] = df_tagged["timestamp"] // 1000
            elif "year" in df_tagged.columns:
                df_tagged["ts"] = pd.to_datetime(
                    df_tagged[["year", "month", "day"]]
                ).astype(int) // 10**9
            else:
                raise ValueError(f"Cannot find timestamp column: {list(df_tagged.columns)}")

            # For each (user, artist) pair, keep the earliest timestamp as interaction time
            df = df_tagged.groupby(["userID", "artistID"]).agg(
                timestamp=("ts", "min")
            ).reset_index()
            df.columns = ["user_id", "item_id", "timestamp"]
            df["rating"] = 1.0  # Implicit feedback

        # Fallback: play counts without timestamps
        elif os.path.exists(artists_file):
            logger.info(f"Reading Last.fm (play counts): {artists_file}")
            df = pd.read_csv(artists_file, sep="\t")
            df.columns = ["user_id", "item_id", "weight"]
            df["rating"] = 1.0
            logger.warning(
                "user_artists.dat has no timestamps; using pseudo-timestamps. "
                "Prefer user_taggedartists-timestamps.dat for real temporal ordering."
            )
            df["timestamp"] = 0
            df = df.sort_values(["user_id", "weight"], ascending=[True, True])
            df["timestamp"] = df.groupby("user_id").cumcount()

        return df[["user_id", "item_id", "rating", "timestamp"]]

    def _download_lastfm(self):
        """Download Last.fm HetRec 2011 dataset"""
        import urllib.request
        import zipfile

        url = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
        zip_path = os.path.join(self.raw_dir, "lastfm.zip")

        logger.info(f"Downloading Last.fm from {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Last.fm. Please download manually:\n"
                f"  1. Go to https://grouplens.org/datasets/hetrec2011/\n"
                f"  2. Download hetrec2011-lastfm-2k.zip\n"
                f"  3. Extract to {self.raw_dir}/\n"
                f"  Expected file: {self.raw_dir}/user_artists.dat\n"
                f"Error: {e}"
            )

        logger.info("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.raw_dir)
        os.remove(zip_path)
        logger.info("Last.fm downloaded and extracted")

    def _load_amazon_digital_music(self) -> pd.DataFrame:
        """
        Load Amazon Digital Music dataset (2023 version).

        Expected path: data/raw/amazon-digital-music/Digital_Music.jsonl(.gz)
        """
        jsonl_gz = os.path.join(self.raw_dir, "Digital_Music.jsonl.gz")
        jsonl = os.path.join(self.raw_dir, "Digital_Music.jsonl")

        if os.path.exists(jsonl_gz):
            import gzip
            logger.info(f"Reading Amazon Digital Music (gzipped): {jsonl_gz}")
            records = []
            with gzip.open(jsonl_gz, "rt", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    records.append({
                        "user_id": obj.get("user_id", obj.get("reviewerID")),
                        "item_id": obj.get("parent_asin", obj.get("asin")),
                        "rating": float(obj.get("rating", 1.0)),
                        "timestamp": int(obj.get("timestamp", 0)),
                    })
            df = pd.DataFrame(records)
        elif os.path.exists(jsonl):
            logger.info(f"Reading Amazon Digital Music: {jsonl}")
            records = []
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    records.append({
                        "user_id": obj.get("user_id", obj.get("reviewerID")),
                        "item_id": obj.get("parent_asin", obj.get("asin")),
                        "rating": float(obj.get("rating", 1.0)),
                        "timestamp": int(obj.get("timestamp", 0)),
                    })
            df = pd.DataFrame(records)
        else:
            raise FileNotFoundError(
                f"Amazon Digital Music data file not found. Please download manually:\n"
                f"  1. Go to https://amazon-reviews-2023.github.io/\n"
                f"  2. Download Digital_Music review data\n"
                f"  3. Place in {self.raw_dir}/\n"
            )

        df["rating"] = 1.0
        return df[["user_id", "item_id", "rating", "timestamp"]]

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data:
        1. Remove duplicate interactions
        2. Remove missing values
        3. k-core filtering (users and items with >= 5 interactions)
        """
        original_len = len(df)

        # Deduplicate, keep latest
        df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
        logger.info(f"Dedup: {original_len:,} -> {len(df):,} ({original_len - len(df):,} removed)")

        # Drop missing values
        df = df.dropna()

        # k-core filtering: iterate until stable
        min_interactions = 5
        prev_len = 0
        iteration = 0
        while len(df) != prev_len:
            prev_len = len(df)
            iteration += 1

            user_counts = df["user_id"].value_counts()
            valid_users = user_counts[user_counts >= min_interactions].index
            df = df[df["user_id"].isin(valid_users)]

            item_counts = df["item_id"].value_counts()
            valid_items = item_counts[item_counts >= min_interactions].index
            df = df[df["item_id"].isin(valid_items)]

            logger.info(f"  k-core iter {iteration}: {len(df):,} interactions")

        logger.info(
            f"k-core filtering done ({min_interactions}-core): "
            f"{len(df):,} interactions, "
            f"{df['user_id'].nunique():,} users, "
            f"{df['item_id'].nunique():,} items"
        )

        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        return df

    def _remap_ids(self, df: pd.DataFrame):
        """Remap raw IDs to contiguous integers [0, N)"""
        unique_users = sorted(df["user_id"].unique())
        user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        idx2user = {idx: uid for uid, idx in user2idx.items()}

        unique_items = sorted(df["item_id"].unique())
        item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        idx2item = {idx: iid for iid, idx in item2idx.items()}

        df = df.copy()
        df["user_id"] = df["user_id"].map(user2idx)
        df["item_id"] = df["item_id"].map(item2idx)

        return df, user2idx, idx2user, item2idx, idx2item

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary label column"""
        df = df.copy()
        if self.feedback_type == "explicit":
            df["label"] = (df["rating"] >= self.relevance_threshold).astype(int)
        else:
            df["label"] = 1
        return df

    def _temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Temporally split data per user.

        For each user, sort interactions by time and split into
        80% train / 10% val / 10% test. Ensures no data leakage
        (val/test timestamps are always after train).
        """
        train_records = []
        val_records = []
        test_records = []

        for user_id, group in df.groupby("user_id"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            n = len(group)

            if n < 3:
                # Too few interactions: all to train
                train_records.append(group)
                continue

            n_train = max(1, int(n * TRAIN_RATIO))
            n_val = max(1, int(n * VAL_RATIO))
            n_test = n - n_train - n_val
            if n_test <= 0:
                n_val = max(1, n - n_train)
                n_test = 0

            train_records.append(group.iloc[:n_train])
            if n_val > 0:
                val_records.append(group.iloc[n_train:n_train + n_val])
            if n_test > 0:
                test_records.append(group.iloc[n_train + n_val:])

        train = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
        val = pd.concat(val_records, ignore_index=True) if val_records else pd.DataFrame()
        test = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()

        self._validate_temporal_split(train, val, test)
        return train, val, test

    def _validate_temporal_split(self, train, val, test):
        """Verify no leakage in temporal split"""
        issues = 0
        for uid in train["user_id"].unique():
            t_max_train = train[train["user_id"] == uid]["timestamp"].max()

            uid_val = val[val["user_id"] == uid]
            if len(uid_val) > 0:
                t_min_val = uid_val["timestamp"].min()
                if t_min_val < t_max_train:
                    issues += 1

            uid_test = test[test["user_id"] == uid]
            if len(uid_test) > 0:
                t_min_test = uid_test["timestamp"].min()
                if t_min_test < t_max_train:
                    issues += 1

        if issues > 0:
            logger.warning(f"Temporal split validation: {issues} potential leaks (may be same-timestamp ties)")
        else:
            logger.info("Temporal split validation passed: no leakage")

    def _build_metadata(self) -> Dict:
        """Build dataset metadata"""
        return {
            "dataset": self.dataset,
            "feedback_type": self.feedback_type,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": len(self._interactions),
            "relevance_threshold": self.relevance_threshold,
            "cold_start_threshold": self.cold_start_threshold,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "train_size": len(self._train),
            "val_size": len(self._val),
            "test_size": len(self._test),
            "sparsity": 1 - len(self._interactions) / (self.n_users * self.n_items),
            "random_seed": self.random_seed,
        }

    def _save_cache(self, cache_file: str):
        """Save processed data to cache"""
        cache_data = {
            "interactions": self._interactions,
            "user2idx": self._user2idx,
            "idx2user": self._idx2user,
            "item2idx": self._item2idx,
            "idx2item": self._idx2item,
            "train": self._train,
            "val": self._val,
            "test": self._test,
            "metadata": self._metadata,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cache saved: {cache_file}")

    def _load_cache(self, cache_file: str):
        """Load processed data from cache"""
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
        self._interactions = cache_data["interactions"]
        self._user2idx = cache_data["user2idx"]
        self._idx2user = cache_data["idx2user"]
        self._item2idx = cache_data["item2idx"]
        self._idx2item = cache_data["idx2item"]
        self._train = cache_data["train"]
        self._val = cache_data["val"]
        self._test = cache_data["test"]
        self._metadata = cache_data["metadata"]

    def _save_splits_csv(self):
        """Save splits as CSV files for direct use by downstream models"""
        self._train.to_csv(os.path.join(self.splits_dir, "train.csv"), index=False)
        self._val.to_csv(os.path.join(self.splits_dir, "val.csv"), index=False)
        self._test.to_csv(os.path.join(self.splits_dir, "test.csv"), index=False)
        self._interactions.to_csv(os.path.join(self.splits_dir, "interactions.csv"), index=False)

        with open(os.path.join(self.splits_dir, "user2idx.json"), "w") as f:
            json.dump({str(k): v for k, v in self._user2idx.items()}, f)
        with open(os.path.join(self.splits_dir, "item2idx.json"), "w") as f:
            json.dump({str(k): v for k, v in self._item2idx.items()}, f)

        with open(os.path.join(self.splits_dir, "metadata.json"), "w") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Splits saved to: {self.splits_dir}")


# ============================================================
# Convenience functions
# ============================================================

def load_splits_from_csv(splits_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load splits directly from CSV (without needing raw data).

    Usage:
        train, val, test = load_splits_from_csv("data/splits/movielens-1m")
    """
    train = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    return train, val, test


def load_metadata(splits_dir: str) -> Dict:
    """Load dataset metadata"""
    with open(os.path.join(splits_dir, "metadata.json"), "r") as f:
        return json.load(f)
