# NCF model
"""
NCF — Neural Collaborative Filtering
=====================================
Essay: He et al. (2017) "Neural Collaborative Filtering"

Architecture: GMF branch + MLP branch, then concat → output layer
- GMF: element-wise product of user/item embeddings
- MLP: concatenate embeddings → multi-layer fully connected
- NeuMF: merge two branches
"""

import torch
import torch.nn as nn


class GMF(nn.Module):
    """Generalized Matrix Factorization — element-wise product of embeddings."""

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.output = nn.Linear(emb_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        return self.output(u * v).squeeze(-1)


class MLP(nn.Module):
    """MLP branch: concatenate embeddings, then fully-connected layers."""

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        layers = []
        input_dim = emb_dim * 2
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            input_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        x = self.mlp(torch.cat([u, v], dim=-1))
        return self.output(x).squeeze(-1)


class NeuMF(nn.Module):
    """
    NeuMF = GMF + MLP, fused at the output layer.
    """

    def __init__(self, n_users: int, n_items: int,
                 gmf_emb_dim: int = 64,
                 mlp_emb_dim: int = 64,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # GMF branch
        self.gmf_user_emb = nn.Embedding(n_users, gmf_emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items, gmf_emb_dim)

        # MLP branch
        self.mlp_user_emb = nn.Embedding(n_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_emb_dim)

        mlp_layers = []
        input_dim = mlp_emb_dim * 2
        for h in hidden_dims:
            mlp_layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            input_dim = h
        self.mlp = nn.Sequential(*mlp_layers)

        # Fusion layer: concat GMF output + MLP last hidden
        self.output = nn.Linear(gmf_emb_dim + input_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF branch
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_v = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_v  # (B, gmf_emb_dim)

        # MLP branch
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_v = self.mlp_item_emb(item_ids)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_v], dim=-1))  # (B, hidden_dims[-1])

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(fused).squeeze(-1)
