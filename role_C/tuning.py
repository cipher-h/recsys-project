# Hyperparameter tuning
"""
tuning.py — Optuna Hyperparameter search
================================
Perform a unified hyperparameter search on NeuMF, TwoTower, and SASRec.
Use val_loss on the val set to select the best configuration, without touching the test set.
"""

import logging
import os
import tempfile
from typing import Dict

import optuna
import torch
from torch.utils.data import DataLoader

from models.ncf import NeuMF
from models.two_tower import TwoTowerModel
from models.sasrec import SASRec
from trainer import PointwiseTrainer, SASRecTrainer
from datasets import PointwiseDataset, SequenceDataset

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial_ckpt(prefix: str, trial_number: int) -> str:
    """
    Save checkpoint for each trial in the system's temporary directory to avoid relying on the ckpts/ directory.
    tempfile.gettempdir() is guaranteed to exist and be writable on all platforms.
    """
    return os.path.join(tempfile.gettempdir(), f"{prefix}_trial{trial_number}.pt")


def tune_neumf(train_df, val_df, n_users: int, n_items: int,
               n_trials: int = 30, n_epochs: int = 15) -> Dict:
    """Perform Optuna search on NeuMF."""

    def objective(trial):
        emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 128])
        n_layers = trial.suggest_int("n_mlp_layers", 2, 4)
        hidden_0 = trial.suggest_categorical("hidden_0", [128, 256])
        hidden_dims = [hidden_0 // (2 ** i) for i in range(n_layers)]
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        train_loader = DataLoader(PointwiseDataset(train_df), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(PointwiseDataset(val_df),   batch_size=batch_size)

        model = NeuMF(n_users, n_items,
                      gmf_emb_dim=emb_dim, mlp_emb_dim=emb_dim,
                      hidden_dims=hidden_dims)
        trainer = PointwiseTrainer(model, lr=lr, weight_decay=wd)
        history = trainer.fit(train_loader, val_loader,
                              n_epochs=n_epochs, patience=3,
                              ckpt_path=_trial_ckpt("neumf", trial.number))
        return min(history["val_loss"])

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    logger.info(f"NeuMF best params: {study.best_trial.params}")
    logger.info(f"NeuMF best val_loss: {study.best_value:.4f}")
    return study.best_trial.params


def tune_two_tower(train_df, val_df, n_users: int, n_items: int,
                   n_trials: int = 30, n_epochs: int = 15) -> Dict:
    """Perform Optuna search on TwoTower."""

    def objective(trial):
        emb_dim     = trial.suggest_categorical("emb_dim", [32, 64, 128])
        output_dim  = trial.suggest_categorical("output_dim", [32, 64])
        hidden      = trial.suggest_categorical("tower_hidden", [64, 128, 256])
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd          = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [256, 512, 1024])
        temperature = trial.suggest_float("temperature", 0.05, 0.5, log=True)

        train_loader = DataLoader(PointwiseDataset(train_df), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(PointwiseDataset(val_df),   batch_size=batch_size)

        model = TwoTowerModel(n_users, n_items,
                              emb_dim=emb_dim, tower_hidden=[hidden],
                              output_dim=output_dim, temperature=temperature)
        trainer = PointwiseTrainer(model, lr=lr, weight_decay=wd)
        history = trainer.fit(train_loader, val_loader,
                              n_epochs=n_epochs, patience=3,
                              ckpt_path=_trial_ckpt("twotower", trial.number))
        return min(history["val_loss"])

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    logger.info(f"TwoTower best params: {study.best_trial.params}")
    logger.info(f"TwoTower best val_loss: {study.best_value:.4f}")
    return study.best_trial.params


def tune_sasrec(train_df, val_df, n_items: int,
                n_trials: int = 20, n_epochs: int = 20) -> Dict:
    """Perform Optuna search on SASRec."""

    def objective(trial):
        emb_dim    = trial.suggest_categorical("emb_dim", [32, 64, 128])
        n_heads    = trial.suggest_categorical("n_heads", [1, 2, 4])
        n_layers   = trial.suggest_int("n_layers", 1, 3)
        max_seq_len= trial.suggest_categorical("max_seq_len", [20, 50, 100])
        dropout    = trial.suggest_float("dropout", 0.1, 0.5)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd         = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        if emb_dim % n_heads != 0:
            raise optuna.exceptions.TrialPruned()

        train_loader = DataLoader(
            SequenceDataset(train_df, n_items, max_seq_len=max_seq_len),
            batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            SequenceDataset(val_df, n_items, max_seq_len=max_seq_len),
            batch_size=batch_size)

        model = SASRec(n_items, max_seq_len=max_seq_len,
                       emb_dim=emb_dim, n_heads=n_heads,
                       n_layers=n_layers, dropout=dropout)
        trainer = SASRecTrainer(model, lr=lr, weight_decay=wd)
        history = trainer.fit(train_loader, val_loader,
                              n_epochs=n_epochs, patience=4,
                              ckpt_path=_trial_ckpt("sasrec", trial.number))
        return min(history["val_loss"])

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    logger.info(f"SASRec best params: {study.best_trial.params}")
    logger.info(f"SASRec best val_loss: {study.best_value:.4f}")
    return study.best_trial.params
