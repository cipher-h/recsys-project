# Trainer
"""
trainer.py — General training loop
==========================
Supports both NeuMF / TwoTower (pointwise loss) and SASRec (sequential BPR loss).
"""

import os
import time
import pickle
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss. Used for implicit feedback."""
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


# ─────────────────────────────────────────
# Pointwise Trainer (NeuMF / TwoTower)
# ─────────────────────────────────────────

class PointwiseTrainer:
    """
    Train NeuMF or TwoTower with BCE loss.
    Suitable for: MovieLens (explicit, with label column) and LastFM (implicit, with negative sampling).
    """

    def __init__(self, model: nn.Module, device: str = "auto",
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        logger.info(f"PointwiseTrainer initialized on {self.device}")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            user_ids = batch["user_id"].to(self.device)
            item_ids = batch["item_id"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(user_ids, item_ids)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            n += len(labels)

        return total_loss / n

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in loader:
            user_ids = batch["user_id"].to(self.device)
            item_ids = batch["item_id"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(user_ids, item_ids)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            n += len(labels)
        return total_loss / n

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            n_epochs: int = 20, patience: int = 5,
            ckpt_path: str = "best_model.pt") -> Dict:
        best_val_loss = float("inf")
        no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            logger.info(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"time={elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"  → Best model saved (val_loss={best_val_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights (if checkpoint doesn't exist, it means no epoch improved, keep current weights)
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return history


# ─────────────────────────────────────────
# SASRec Trainer 
# ─────────────────────────────────────────

class SASRecTrainer:
    """
    Train SASRec with BPR loss.
    Input is sequence samples provided by SequenceDataset.
    """

    def __init__(self, model: nn.Module, device: str = "auto",
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.bpr_loss = BPRLoss()
        logger.info(f"SASRecTrainer initialized on {self.device}")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            item_seq = batch["item_seq"].to(self.device)
            pos_items = batch["pos_items"].to(self.device)
            neg_items = batch["neg_items"].to(self.device)

            self.optimizer.zero_grad()
            pos_logits, neg_logits = self.model(item_seq, pos_items, neg_items)

            # Only compute loss for non-padding positions
            mask = pos_items != 0
            loss = self.bpr_loss(pos_logits[mask], neg_logits[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()

        return total_loss / max(n, 1)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in loader:
            item_seq = batch["item_seq"].to(self.device)
            pos_items = batch["pos_items"].to(self.device)
            neg_items = batch["neg_items"].to(self.device)
            pos_logits, neg_logits = self.model(item_seq, pos_items, neg_items)
            mask = pos_items != 0
            loss = self.bpr_loss(pos_logits[mask], neg_logits[mask])
            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()
        return total_loss / max(n, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            n_epochs: int = 30, patience: int = 5,
            ckpt_path: str = "best_sasrec.pt") -> Dict:
        best_val_loss = float("inf")
        no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            logger.info(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"time={elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"  → Best model saved (val_loss={best_val_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return history
