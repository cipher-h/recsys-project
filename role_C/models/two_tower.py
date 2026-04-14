# Two-tower model
"""
Two-Tower Model (Dual Encoder)
================================
Industry-standard candidate generation architecture.
- User tower: user_id embedding → MLP → user vector
- Item tower: item_id embedding → MLP → item vector
- Score: dot product (supports ANN retrieval)

Note: The inner product structure of the two towers is intentional 
- it allows for large-scale retrieval using approximate nearest neighbors (FAISS, etc.) during inference. 
Do not change it to cross-attention, as that would turn it into a ranking model, contradicting the design philosophy of the two towers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_tower(input_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.2) -> nn.Sequential:
    """Build the MLP tower."""
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class TwoTowerModel(nn.Module):
    """
    Two-tower model.

    Args:
        n_users: Total number of users (from metadata["n_users"])
        n_items: Total number of items (from metadata["n_items"])
        emb_dim: Embedding dimension
        tower_hidden: List of hidden dimensions for each tower
        output_dim: Final user/item vector dimension
        temperature: Softmax temperature for contrastive learning loss
    """

    def __init__(self, n_users: int, n_items: int,
                 emb_dim: int = 64,
                 tower_hidden: list = None,
                 output_dim: int = 64,
                 temperature: float = 0.1):
        super().__init__()
        if tower_hidden is None:
            tower_hidden = [128]

        self.temperature = temperature

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.user_tower = _build_tower(emb_dim, tower_hidden, output_dim)
        self.item_tower = _build_tower(emb_dim, tower_hidden, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Return L2 normalized user vector."""
        return F.normalize(self.user_tower(self.user_emb(user_ids)), dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return L2 normalized item vector."""
        return F.normalize(self.item_tower(self.item_emb(item_ids)), dim=-1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return user-item similarity score (dot product after L2 norm = cosine sim)."""
        u = self.encode_user(user_ids)   # (B, output_dim)
        v = self.encode_item(item_ids)   # (B, output_dim)
        return (u * v).sum(dim=-1)       # (B,)

    def get_scores_for_all_items(self, user_ids: torch.Tensor,
                                  all_item_embs: torch.Tensor) -> torch.Tensor:
        """
        During inference, calculate the user's scores for all items in batches.
        Used for generating top-K recommendation lists.

        Args:
            user_ids: (B,) User IDs
            all_item_embs: (n_items, output_dim) Pre-computed item vectors

        Returns:
            scores: (B, n_items)
        """
        u = self.encode_user(user_ids)          # (B, D)
        return torch.matmul(u, all_item_embs.T) # (B, n_items)
