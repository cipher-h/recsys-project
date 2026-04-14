# Run on MovieLens
"""
run_movielens.py — Training and Inference of DL Models on MovieLens 1M
=========================================================
Run:
    python run_movielens.py

Output:
    predictions/movielens-1m/neumf.pkl
    predictions/movielens-1m/two_tower.pkl
    checkpoints/ (model weights)
    results/movielens_training_curves.png
"""

import os
import json
import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─── Add the repo root directory to sys.path to ensure that modules in src and role_C can be imported ──────
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))   # role_C/
_ROOT = os.path.dirname(_HERE)                        # repo root
sys.path.insert(0, _ROOT)   # for src/
sys.path.insert(0, _HERE)   # for models/, datasets.py, trainer.py …

from src.data_loader import load_splits_from_csv, load_metadata
from models.ncf import NeuMF
from models.two_tower import TwoTowerModel
from datasets import PointwiseDataset
from trainer import PointwiseTrainer
from tuning import tune_neumf, tune_two_tower
from inference import (
    generate_predictions_pointwise,
    save_predictions,
)

# ─── log ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Global Config ───────────────────────────────────────────────────────────────
DATASET = "movielens-1m"
DATA_DIR = os.path.join(_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits", DATASET)
PRED_DIR = os.path.join(_ROOT, "predictions")
CKPT_DIR = os.path.join(_HERE, "ckpts")
RESULT_DIR = os.path.join(_HERE, "results")

K = 10
SEEDS = [42, 123, 2024]
N_OPTUNA_TRIALS = 30
N_EPOCHS = 30
BATCH_SIZE = 512
NUM_NEG = 4

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(PRED_DIR, DATASET), exist_ok=True)


def load_data():
    """Load processed data from A. Prefer to read CSV directly."""
    logger.info(f"Loading data from {SPLITS_DIR}")
    train, val, test = load_splits_from_csv(SPLITS_DIR)
    meta = load_metadata(SPLITS_DIR)
    logger.info(
        f"MovieLens 1M | n_users={meta['n_users']}, n_items={meta['n_items']} "
        f"| train={len(train)}, val={len(val)}, test={len(test)}"
    )
    return train, val, test, meta


def run_neumf(train, val, test, meta, best_params=None):
    """Train NeuMF with multiple seeds and output prediction files."""
    n_users, n_items = meta["n_users"], meta["n_items"]

    # ── Hyperparameter Tuning ──────────────────────────────────────────────────────────
    if best_params is None:
        logger.info("=== NeuMF Hyperparameter Tuning ===")
        best_params = tune_neumf(train, val, n_users, n_items,
                                 n_trials=N_OPTUNA_TRIALS, n_epochs=15)
        with open(os.path.join(RESULT_DIR, "neumf_best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2)
    else:
        logger.info(f"Using provided NeuMF params: {best_params}")

    emb_dim = best_params.get("emb_dim", 64)
    n_mlp_layers = best_params.get("n_mlp_layers", 3)
    hidden_0 = best_params.get("hidden_0", 256)
    hidden_dims = [hidden_0 // (2 ** i) for i in range(n_mlp_layers)]
    lr = best_params.get("lr", 1e-3)
    wd = best_params.get("weight_decay", 1e-5)
    batch_size = best_params.get("batch_size", BATCH_SIZE)

    # ── Multi-seed training ────────────────────────────────────────────────────────
    all_predictions = []
    for seed in SEEDS:
        logger.info(f"=== NeuMF Seed={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = NeuMF(n_users, n_items,
                      gmf_emb_dim=emb_dim,
                      mlp_emb_dim=emb_dim,
                      hidden_dims=hidden_dims)

        train_loader = DataLoader(
            PointwiseDataset(train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            PointwiseDataset(val), batch_size=batch_size
        )

        trainer = PointwiseTrainer(model, lr=lr, weight_decay=wd)
        history = trainer.fit(
            train_loader, val_loader,
            n_epochs=N_EPOCHS, patience=5,
            ckpt_path=os.path.join(CKPT_DIR, f"neumf_seed{seed}.pt")
        )

        # Save training history
        with open(os.path.join(RESULT_DIR, f"neumf_history_seed{seed}.json"), "w") as f:
            json.dump(history, f)

        # Inference
        test_user_ids = test["user_id"].unique().tolist()
        preds = generate_predictions_pointwise(
            model, test_user_ids, n_items, train, K=K
        )
        all_predictions.append(preds)

    # ── Use the predictions from the last seed as the main predictions (D will use this to compute metrics) ──────
    # Report the mean and standard deviation of the 3 seeds in the report (D will run the evaluation code separately)
    final_preds = all_predictions[-1]
    path = save_predictions(final_preds, DATASET, "neumf", output_dir=PRED_DIR)

    # Also save all seeds' results for D analysis
    with open(os.path.join(PRED_DIR, DATASET, "neumf_all_seeds.pkl"), "wb") as f:
        pickle.dump(all_predictions, f)

    logger.info(f"NeuMF done. Predictions saved to {path}")
    return final_preds


def run_two_tower(train, val, test, meta, best_params=None):
    """Train TwoTower with multiple seeds and output prediction files."""
    n_users, n_items = meta["n_users"], meta["n_items"]

    if best_params is None:
        logger.info("=== TwoTower Hyperparameter Tuning ===")
        best_params = tune_two_tower(train, val, n_users, n_items,
                                     n_trials=N_OPTUNA_TRIALS, n_epochs=15)
        with open(os.path.join(RESULT_DIR, "two_tower_best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2)

    emb_dim = best_params.get("emb_dim", 64)
    output_dim = best_params.get("output_dim", 64)
    tower_hidden = best_params.get("tower_hidden", 128)
    lr = best_params.get("lr", 1e-3)
    wd = best_params.get("weight_decay", 1e-5)
    batch_size = best_params.get("batch_size", BATCH_SIZE)
    temperature = best_params.get("temperature", 0.1)

    all_predictions = []
    for seed in SEEDS:
        logger.info(f"=== TwoTower Seed={seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TwoTowerModel(n_users, n_items,
                              emb_dim=emb_dim,
                              tower_hidden=[tower_hidden],
                              output_dim=output_dim,
                              temperature=temperature)

        train_loader = DataLoader(
            PointwiseDataset(train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            PointwiseDataset(val), batch_size=batch_size
        )

        trainer = PointwiseTrainer(model, lr=lr, weight_decay=wd)
        history = trainer.fit(
            train_loader, val_loader,
            n_epochs=N_EPOCHS, patience=5,
            ckpt_path=os.path.join(CKPT_DIR, f"two_tower_seed{seed}.pt")
        )
        with open(os.path.join(RESULT_DIR, f"two_tower_history_seed{seed}.json"), "w") as f:
            json.dump(history, f)

        test_user_ids = test["user_id"].unique().tolist()
        preds = generate_predictions_pointwise(
            model, test_user_ids, n_items, train, K=K
        )
        all_predictions.append(preds)

    final_preds = all_predictions[-1]
    path = save_predictions(final_preds, DATASET, "two_tower", output_dir=PRED_DIR)

    with open(os.path.join(PRED_DIR, DATASET, "two_tower_all_seeds.pkl"), "wb") as f:
        pickle.dump(all_predictions, f)

    logger.info(f"TwoTower done. Predictions saved to {path}")
    return final_preds


if __name__ == "__main__":
    train, val, test, meta = load_data()

    run_neumf(train, val, test, meta)
    run_two_tower(train, val, test, meta)

    logger.info("=== MovieLens 1M Done ===")
    logger.info("Outputs:")
    logger.info(f"  predictions/{DATASET}/neumf.pkl")
    logger.info(f"  predictions/{DATASET}/two_tower.pkl")
    logger.info(f"  predictions/{DATASET}/neumf_all_seeds.pkl  (D will use this for multi-seed analysis)")
    logger.info(f"  predictions/{DATASET}/two_tower_all_seeds.pkl")
