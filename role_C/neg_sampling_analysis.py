# Neg sampling analysis
"""
neg_sampling_analysis.py — Negative sampling sensitivity analysis
=============================================
The assignment explicitly requires analyzing the impact of num_neg and strategy on model performance.
This script conducts a controlled experiment using TwoTower on Last.fm (chosen for faster training).

Run:
    python neg_sampling_analysis.py

Output:
    results/neg_sampling_analysis.json
    predictions/lastfm/two_tower_neg{N}_{strategy}.pkl
"""

import os
import json
import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from src.data_loader import load_splits_from_csv, load_metadata
from run_lastfm import _negative_sample
from models.two_tower import TwoTowerModel
from datasets import PointwiseDataset
from trainer import PointwiseTrainer
from inference import generate_predictions_pointwise, save_predictions

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET = "lastfm"
DATA_DIR = os.path.join(_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits", DATASET)
PRED_DIR = os.path.join(_ROOT, "predictions")
CKPT_DIR = os.path.join(_HERE, "ckpts")
RESULT_DIR = os.path.join(_HERE, "results")

K = 10
SEED = 42
N_EPOCHS = 20

NUM_NEG_LIST = [1, 4, 10]
STRATEGY_LIST = ["uniform", "popularity"]


def main():
    train, val, test = load_splits_from_csv(SPLITS_DIR)
    meta = load_metadata(SPLITS_DIR)
    n_users, n_items = meta["n_users"], meta["n_items"]

    n_items = meta["n_items"]

    # Fixed model configuration (using reasonable default parameters, no further parameter tuning required)
    MODEL_CFG = dict(emb_dim=64, tower_hidden=[128], output_dim=64, temperature=0.1)
    TRAIN_CFG = dict(lr=1e-3, weight_decay=1e-5)

    results = {}
    os.makedirs(os.path.join(PRED_DIR, DATASET), exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    for num_neg in NUM_NEG_LIST:
        for strategy in STRATEGY_LIST:
            exp_name = f"neg{num_neg}_{strategy}"
            logger.info(f"=== Experiment: {exp_name} ===")

            torch.manual_seed(SEED)
            np.random.seed(SEED)
            rng = np.random.RandomState(SEED)
            train_neg = _negative_sample(train, n_items, num_neg, strategy, rng)
            val_neg   = _negative_sample(val,   n_items, num_neg, strategy, rng)

            train_loader = DataLoader(PointwiseDataset(train_neg),
                                      batch_size=512, shuffle=True)
            val_loader = DataLoader(PointwiseDataset(val_neg),
                                    batch_size=512)

            model = TwoTowerModel(n_users, n_items, **MODEL_CFG)
            trainer = PointwiseTrainer(model, **TRAIN_CFG)
            history = trainer.fit(
                train_loader, val_loader,
                n_epochs=N_EPOCHS, patience=4,
                ckpt_path=os.path.join(CKPT_DIR, f"lastfm_twotower_{exp_name}.pt")
            )

            test_user_ids = test["user_id"].unique().tolist()
            preds = generate_predictions_pointwise(
                model, test_user_ids, n_items, train, K=K
            )

            # Save predictions for D evaluation
            pkl_name = f"two_tower_{exp_name}"
            save_predictions(preds, DATASET, pkl_name, output_dir=PRED_DIR)

            results[exp_name] = {
                "num_neg": num_neg,
                "strategy": strategy,
                "best_val_loss": min(history["val_loss"]),
                "n_epochs_trained": len(history["val_loss"]),
                "train_loss_final": history["train_loss"][-1],
            }
            logger.info(f"  best_val_loss={results[exp_name]['best_val_loss']:.4f}")

    # Save overall results
    with open(os.path.join(RESULT_DIR, "neg_sampling_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Neg sampling analysis done.")
    logger.info("Results: results/neg_sampling_analysis.json")
    logger.info("Predictions for D to evaluate:")
    for k in results:
        logger.info(f"  predictions/{DATASET}/two_tower_{k}.pkl")


if __name__ == "__main__":
    main()
