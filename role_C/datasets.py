"""
datasets.py — PyTorch Dataset 
====================================
- PointwiseDataset: use for NeuMF / TwoTower, Output directly using A's DataLoader
- SequenceDataset: use for SASRec, Build user history into a sliding window sequence
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PointwiseDataset(Dataset):
    """
    Construct (user, item, label) triples from a DataFrame.
    Directly accept the output of A's get_torch_dataset() or get_negative_samples().
    """

    def __init__(self, df: pd.DataFrame):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["item_id"].values)
        self.labels = torch.FloatTensor(df["label"].values)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            "user_id": self.users[idx],
            "item_id": self.items[idx],
            "label": self.labels[idx],
        }


class SequenceDataset(Dataset):
    """
    Construct sequence training samples for SASRec.

    For each user's temporal interactions, generate training samples using a sliding window approach:
      - item_seq:  Historical sequence (left padded to max_seq_len)
      - pos_items: Corresponding next item (positive sample sequence)
      - neg_items: Random negative sampling sequence

    Args:
        df:           Training set DataFrame (must be sorted by user_id + timestamp)
        n_items:      Total number of items (from metadata["n_items"])
        max_seq_len:  Maximum sequence length
        num_neg:      Number of negative samples for each position (take the first as neg_item)
        seed:         Random seed
    """

    def __init__(self, df: pd.DataFrame, n_items: int,
                 max_seq_len: int = 50, num_neg: int = 1, seed: int = 42):
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)

        self.item_seqs = []    # (L,) Historical sequence (left padded)
        self.pos_seqs = []     # (L,) Positive sample sequence
        self.neg_seqs = []     # (L,) Negative sample sequence

        self._build(df)

    def _build(self, df: pd.DataFrame):
        # Grouped by user, arranged chronologically
        df_sorted = df.sort_values(["user_id", "timestamp"])
        for uid, group in df_sorted.groupby("user_id"):
            items = group["item_id"].tolist()  # Temporal ordered item sequence
            pos_items_set = set(items)

            if len(items) < 2:
                continue  # At least 2 interactions are needed to form a training sample

            # Sliding window: generate L-1 samples for a sequence of length L
            for i in range(1, len(items)):
                # Historical sequence (first i items, truncated to max_seq_len)
                history = items[max(0, i - self.max_seq_len):i]
                pos_item = items[i]

                # Negative sampling: uniform random, avoiding interacted items
                neg_item = pos_item
                while neg_item in pos_items_set:
                    neg_item = self.rng.randint(1, self.n_items + 1)

                # Left padding to max_seq_len
                pad_len = self.max_seq_len - len(history)
                padded_history = [0] * pad_len + history
                # Padding is also applied to pos/neg (only the last bit is the actual value)
                padded_pos = [0] * (self.max_seq_len - 1) + [pos_item]
                padded_neg = [0] * (self.max_seq_len - 1) + [neg_item]

                self.item_seqs.append(padded_history)
                self.pos_seqs.append(padded_pos)
                self.neg_seqs.append(padded_neg)

        self.item_seqs = torch.LongTensor(self.item_seqs)
        self.pos_seqs = torch.LongTensor(self.pos_seqs)
        self.neg_seqs = torch.LongTensor(self.neg_seqs)

    def __len__(self):
        return len(self.item_seqs)

    def __getitem__(self, idx):
        return {
            "item_seq": self.item_seqs[idx],
            "pos_items": self.pos_seqs[idx],
            "neg_items": self.neg_seqs[idx],
        }


def build_user_sequences(df: pd.DataFrame, max_seq_len: int = 50) -> dict:
    """
    Construct a history sequence for each user (the entire training set history) for the inference phase.

    Returns:
        dict: {user_id: padded_sequence (List[int], length max_seq_len)}
    """
    user_seqs = {}
    df_sorted = df.sort_values(["user_id", "timestamp"])
    for uid, group in df_sorted.groupby("user_id"):
        items = group["item_id"].tolist()
        history = items[-max_seq_len:]
        pad_len = max_seq_len - len(history)
        user_seqs[uid] = [0] * pad_len + history
    return user_seqs
