# SASRec model
"""
SASRec — Self-Attentive Sequential Recommendation
===================================================
Essay: Kang & McAuley (2018) "Self-Attentive Sequential Recommendation"

Model the user's historical interaction sequence using Transformer self-attention.
Predict the next item to interact with.

Applicable scenarios: Last.fm (temporal implicit feedback)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    """
    SASRec: Transformer-based sequential recommendation.

    Args:
        n_items: Total number of items (from metadata["n_items"])
        max_seq_len: Maximum length of user historical sequences (truncate the most recent max_seq_len items if exceeded)
        emb_dim: Item embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer blocks
        dropout: Dropout rate
    """

    def __init__(self, n_items: int,
                 max_seq_len: int = 50,
                 emb_dim: int = 64,
                 n_heads: int = 2,
                 n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim

        # Item embedding（0 is padding index）
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len + 1, emb_dim)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN, more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight, std=0.01)

    def encode_sequence(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode the user's historical sequence into a vector.

        Args:
            item_seq: (B, L) Historical item ID sequence (0 indicates padding, placed on the left)

        Returns:
            seq_emb: (B, L, emb_dim)
        """
        B, L = item_seq.shape
        positions = torch.arange(1, L + 1, device=item_seq.device).unsqueeze(0)  # (1, L)

        x = self.item_emb(item_seq) + self.pos_emb(positions)  # (B, L, D)
        x = self.dropout(self.layer_norm(x))

        # Causal mask: Unable to see future tokens
        causal_mask = torch.triu(
            torch.ones(L, L, device=item_seq.device, dtype=torch.bool), diagonal=1
        )
        # Padding mask
        pad_mask = (item_seq == 0)  # (B, L), True indicates padding

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return x  # (B, L, D)

    def forward(self, item_seq: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: torch.Tensor = None):
        """
        Called during training.

        Args:
            item_seq:  (B, L) Historical sequence (padding on the left)
            pos_items: (B, L) Positive samples (next item, padding aligned)
            neg_items: (B, L) Negative samples (optional)

        Returns:
            If neg_items is not None: (pos_logits, neg_logits) for BPR loss
            Otherwise: pos_logits for BCE loss
        """
        seq_emb = self.encode_sequence(item_seq)   # (B, L, D)
        item_all_emb = self.item_emb.weight         # (n_items+1, D)

        pos_emb = item_all_emb[pos_items]           # (B, L, D)
        pos_logits = (seq_emb * pos_emb).sum(-1)    # (B, L)

        if neg_items is not None:
            neg_emb = item_all_emb[neg_items]       # (B, L, D)
            neg_logits = (seq_emb * neg_emb).sum(-1)
            return pos_logits, neg_logits

        return pos_logits

    def predict(self, item_seq: torch.Tensor, target_items: torch.Tensor = None):
        """
        Called during inference. Uses the output from the last non-padding position of the sequence

        Args:
            item_seq:     (B, L) Historical sequence
            target_items: (B, K) Candidate items (None means score all items)

        Returns:
            scores: (B,) or (B, K) or (B, n_items)
        """
        seq_emb = self.encode_sequence(item_seq)  # (B, L, D)

        # Get the last non-padding position for each user's sequence
        lengths = (item_seq != 0).sum(dim=1) - 1   # (B,) Last valid position index
        lengths = lengths.clamp(min=0)
        last_emb = seq_emb[torch.arange(seq_emb.size(0)), lengths]  # (B, D)

        if target_items is not None:
            # Score the specified candidates
            t_emb = self.item_emb(target_items)     # (B, K, D)
            scores = (last_emb.unsqueeze(1) * t_emb).sum(-1)  # (B, K)
        else:
            # Score all items
            all_emb = self.item_emb.weight           # (n_items+1, D)
            scores = torch.matmul(last_emb, all_emb.T)  # (B, n_items+1)
            scores = scores[:, 1:]                   # Remove padding index=0

        return scores
