# Inference
"""
inference.py — Generate a top-K recommendation list and output it to D's pkl file.
==========================================================
All models uniformly call generate_predictions(), output format:
    {user_id (int): [item_id_1, item_id_2, ..., item_id_K]}
"""

import os
import pickle
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _exclude_train_items(scores: np.ndarray,
                         user_ids: List[int],
                         train_item_sets: Dict[int, set]) -> np.ndarray:
    """
    Set the scores of items that have appeared in the user's training set to -inf to prevent recommending known items.
    """
    scores = scores.copy()
    for i, uid in enumerate(user_ids):
        known = train_item_sets.get(uid, set())
        for iid in known:
            if iid < scores.shape[1]:
                scores[i, iid] = -np.inf
    return scores


def generate_predictions_pointwise(
    model: nn.Module,
    test_user_ids: List[int],
    n_items: int,
    train_df,
    K: int = 10,
    device: str = "auto",
    batch_size: int = 512,
) -> Dict[int, List[int]]:
    """
    NeuMF / TwoTower Inference: Enumerate all items for each user and score them, then take the top-K.

    Args:
        model:         The trained model (NeuMF or TwoTower)
        test_user_ids: The list of users for whom recommendations are to be generated
        n_items:      The total number of items
        train_df:     The training set DataFrame (used to filter known items)
        K:            The length of the recommendation list
        device:      "auto" / "cuda" / "cpu"
        batch_size:    The number of users to process in each batch

    Returns:
        predictions: {user_id: [top-K item_ids]}
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Build a set of known interacted items for each user in the training set
    train_item_sets = (
        train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    all_item_ids = torch.arange(n_items, device=device)  # (n_items,)

    # Precompute item embeddings for TwoTower (if applicable)
    has_encode_item = hasattr(model, "encode_item")
    if has_encode_item:
        with torch.no_grad():
            all_item_embs = model.encode_item(all_item_ids)  # (n_items, D)
    
    predictions = {}

    for start in range(0, len(test_user_ids), batch_size):
        batch_uids = test_user_ids[start: start + batch_size]
        user_tensor = torch.LongTensor(batch_uids).to(device)  # (B,)

        with torch.no_grad():
            if has_encode_item:
                # TwoTower: Batch scoring using matrix multiplication
                scores = model.get_scores_for_all_items(
                    user_tensor, all_item_embs
                ).cpu().numpy()  # (B, n_items)
            else:
                # NeuMF: Items need to be scored one by one, using broadcasting.
                # (B, 1) x (1, n_items) — Expand to (B*n_items,) and feed into the model in batches
                B = len(batch_uids)
                user_exp = user_tensor.unsqueeze(1).expand(B, n_items).reshape(-1)
                item_exp = all_item_ids.unsqueeze(0).expand(B, n_items).reshape(-1)
                logits = model(user_exp, item_exp).reshape(B, n_items)
                scores = logits.cpu().numpy()  # (B, n_items)

        # Filter known items
        scores = _exclude_train_items(scores, batch_uids, train_item_sets)

        # Take top-K
        topk_indices = np.argpartition(scores, -K, axis=1)[:, -K:]
        for i, uid in enumerate(batch_uids):
            row_scores = scores[i, topk_indices[i]]
            sorted_order = np.argsort(-row_scores)
            predictions[uid] = topk_indices[i][sorted_order].tolist()

        if (start // batch_size) % 10 == 0:
            logger.info(
                f"  Inference progress: {min(start + batch_size, len(test_user_ids))}"
                f"/{len(test_user_ids)} users"
            )

    return predictions


def generate_predictions_sasrec(
    model: nn.Module,
    test_user_ids: List[int],
    user_sequences: Dict[int, List[int]],
    n_items: int,
    train_df,
    K: int = 10,
    device: str = "auto",
    batch_size: int = 256,
) -> Dict[int, List[int]]:
    """
    SASRec Inference: Generate top-K recommendations using user historical sequences.

    Args:
        model:          The trained SASRec model
        test_user_ids:  The list of users for whom recommendations are to be generated
        user_sequences: The {user_id: padded_seq} returned by build_user_sequences()
        n_items:        The total number of items
        train_df:       The training set DataFrame (used to filter known items)
        K:              The number of recommendations to generate
        device:        The device to run the model on
        batch_size:     The number of users to process in each batch

    Returns:
        predictions: {user_id: [top-K item_ids]}
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    train_item_sets = (
        train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    predictions = {}

    for start in range(0, len(test_user_ids), batch_size):
        batch_uids = test_user_ids[start: start + batch_size]

        # Build sequence tensor
        seqs = [user_sequences.get(uid, [0] * model.max_seq_len) for uid in batch_uids]
        seq_tensor = torch.LongTensor(seqs).to(device)  # (B, L)

        with torch.no_grad():
            scores = model.predict(seq_tensor).cpu().numpy()  # (B, n_items)

        scores = _exclude_train_items(scores, batch_uids, train_item_sets)

        topk_indices = np.argpartition(scores, -K, axis=1)[:, -K:]
        for i, uid in enumerate(batch_uids):
            row_scores = scores[i, topk_indices[i]]
            sorted_order = np.argsort(-row_scores)
            predictions[uid] = topk_indices[i][sorted_order].tolist()

        if (start // batch_size) % 10 == 0:
            logger.info(
                f"  SASRec inference: {min(start + batch_size, len(test_user_ids))}"
                f"/{len(test_user_ids)} users"
            )

    return predictions


def save_predictions(predictions: Dict[int, List[int]],
                     dataset: str,
                     model_name: str,
                     output_dir: str = "predictions"):
    """
    Save the prediction results in the format specified in document A.

    Save path: predictions/{dataset}/{model_name}.pkl
    Format:     {user_id (int): [item_id_1, ..., item_id_K]}
    """
    save_dir = os.path.join(output_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(predictions, f)

    logger.info(f"Predictions saved → {save_path}  ({len(predictions)} users)")
    return save_path
