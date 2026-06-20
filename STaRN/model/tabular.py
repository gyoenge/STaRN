# Variant of TransTab + SAINT

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init

# input -> no pd.DataFrame, just raw tensors


class FeatureEmbedding(nn.Module):
    """Embed fixed-width numerical feature tensors (radiomics) into per-feature token embeddings.

    Adapts the TransTab numerical embedding pipeline for tensor-only input:
    - Replaces BERT-tokenized column names with a learnable per-column index embedding
      (TransTabWordEmbedding role, without tokenizer dependency)
    - Applies TransTabNumEmbedding scaling: emb = col_emb * value + bias
    - Projects through an align_layer (TransTabFeatureProcessor role)

    Output shape: (B, num_features, hidden_dim) — one token per radiomics feature.
    """

    def __init__(self,
        num_features: int,
        hidden_dim: int = 128,
        hidden_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-5,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.col_embedding = nn.Embedding(num_features, hidden_dim)
        nn_init.kaiming_normal_(self.col_embedding.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.num_bias = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.register_buffer('col_indices', torch.arange(num_features))
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_features)
        Returns:
            (B, num_features, hidden_dim)
        """
        col_emb = self.col_embedding(self.col_indices)              # (F, D)
        col_emb = self.norm(col_emb)
        col_emb = self.dropout(col_emb)
        col_emb = col_emb.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, F, D)

        feat_emb = col_emb * x.unsqueeze(-1).float() + self.num_bias  # (B, F, D)
        feat_emb = self.align_layer(feat_emb)
        return feat_emb


class ColumnAttention(nn.Module):
    """Multi-head self-attention over feature tokens with gated FFN.

    Attends over the F (feature) dimension: (B, F, D) → (B, F, D).

    Gated FFN (replaces standard FFN following the design spec):
        gate = sigmoid(W_g(x))
        val  = W_v(x)
        out  = W_o(gate ⊙ val)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim)
        self.val_proj  = nn.Linear(hidden_dim, ffn_dim)
        self.out_proj  = nn.Linear(ffn_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, D)
        Returns:
            (B, F, D)
        """
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(h))

        gate = torch.sigmoid(self.gate_proj(x))
        val  = self.val_proj(x)
        h_gate = self.drop(self.out_proj(gate * val))
        x = self.norm2(x + h_gate)
        return x


class RowAttention(nn.Module):
    """Intersample multi-head self-attention across spots with relative positional encoding.

    Transposes to (F, B, D), attends over the B (spot) dimension with a
    distance-based relative PE bias, then transposes back: (B, F, D) → (B, F, D).

    Relative PE: pairwise pixel/spatial distances are quantised into bins and
    each bin has a learnable per-head bias added to the attention logits.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        n_pos_bins: int = 32,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.drop  = nn.Dropout(dropout)

        self.num_heads  = num_heads
        self.n_pos_bins = n_pos_bins
        # +1 bucket for "no coord" (distance = 0 when all coords are -1)
        self.pos_bias = nn.Embedding(n_pos_bins + 1, num_heads)

    def _rel_pos_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute (num_heads, B, B) attention bias from 2-D spatial coords."""
        B = coords.size(0)
        diff = coords.unsqueeze(1).float() - coords.unsqueeze(0).float()  # (B, B, 2)
        dist = torch.norm(diff, dim=-1)                                    # (B, B)
        max_dist = dist.max().clamp(min=1.0)
        bins = (dist / max_dist * self.n_pos_bins).long().clamp(0, self.n_pos_bins)
        bias = self.pos_bias(bins)          # (B, B, H)
        return bias.permute(2, 0, 1)        # (H, B, B)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, F, D)
            coords: (B, 2) spatial coordinates; None disables relative PE.
        Returns:
            (B, F, D)
        """
        B, F, D = x.shape
        x_t = x.transpose(0, 1)   # (F, B, D)

        attn_mask = None
        if coords is not None:
            bias = self._rel_pos_bias(coords)                              # (H, B, B)
            attn_mask = (
                bias.unsqueeze(0)
                .expand(F, -1, -1, -1)
                .reshape(F * self.num_heads, B, B)
            )

        h, _ = self.attn(x_t, x_t, x_t, attn_mask=attn_mask)
        x_t = self.norm1(x_t + self.drop(h))
        x_t = self.norm2(x_t + self.drop(self.ffn(x_t)))
        return x_t.transpose(0, 1)   # (B, F, D)


class _ProjHead(nn.Module):
    """Two-layer MLP projection head with L2 normalisation on output."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SummaryTableModel(nn.Module):
    """SAINT-style tabular encoder for spatial radiomics Summary Tables.

    Architecture:
        Input (B, F)
          → FeatureEmbedding + [CLS] prepend  →  (B, 1+F, D)
          → ColumnAttention × num_col_layers  →  z_col (B, proj_dim)
          → RowAttention    × num_row_layers  →  z_row (B, proj_dim)
                                              →  z_out (B, proj_dim)

    Args:
        num_features:    Number of input radiomics features.
        hidden_dim:      Token embedding dimension D.
        num_col_layers:  Number of ColumnAttention blocks.
        num_row_layers:  Number of RowAttention blocks (design default: 1).
        num_heads:       Attention heads (shared for both attention types).
        ffn_dim:         Feed-forward / gated-FFN intermediate dimension.
        proj_dim:        Output projection dimension for all heads.
        dropout:         Dropout probability.
        n_pos_bins:      Relative-PE distance quantisation bins for RowAttention.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_col_layers: int = 2,
        num_row_layers: int = 1,
        num_heads: int = 8,
        ffn_dim: int = 256,
        proj_dim: int = 128,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        n_pos_bins: int = 32,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.embed = FeatureEmbedding(
            num_features=num_features,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=dropout,
            layer_norm_eps=layer_norm_eps,
            device=device,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn_init.trunc_normal_(self.cls_token, std=0.02)

        self.col_layers = nn.ModuleList([
            ColumnAttention(hidden_dim, num_heads, ffn_dim, dropout, layer_norm_eps)
            for _ in range(num_col_layers)
        ])
        self.norm_col = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.proj_col = _ProjHead(hidden_dim, proj_dim)

        self.row_layers = nn.ModuleList([
            RowAttention(hidden_dim, num_heads, ffn_dim, dropout, layer_norm_eps, n_pos_bins)
            for _ in range(num_row_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.proj_row = _ProjHead(hidden_dim, proj_dim)
        self.proj_out = _ProjHead(hidden_dim, proj_dim)

        self.device = device
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      (B, num_features) — raw radiomics features
            coords: (B, 2) spatial coordinates for relative PE; None disables PE.
        Returns:
            z_col:     (B, proj_dim) — after column-attention stage
            z_row:     (B, proj_dim) — after row-attention stage
            z_out:     (B, proj_dim) — final output, used for L_self
            token_emb: (B, num_features, hidden_dim)
        """
        h = self.embed(x)                                    # (B, F, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)      # (B, 1, D)
        h = torch.cat([cls, h], dim=1)                       # (B, 1+F, D)

        for layer in self.col_layers:
            h = layer(h)
        z_col = self.proj_col(self.norm_col(h[:, 0]))        # (B, proj_dim)

        for layer in self.row_layers:
            h = layer(h, coords)
        h = self.norm_out(h)
        z_row     = self.proj_row(h[:, 0])   # (B, proj_dim)
        z_out     = self.proj_out(h[:, 0])   # (B, proj_dim)
        token_emb = h[:, 1:]                 # (B, F, D)

        return z_col, z_row, z_out, token_emb

    def encode(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the pre-projection CLS embedding for downstream tasks (inference).

        Args:
            x:      (B, num_features)
            coords: (B, 2) optional spatial coordinates.
        Returns:
            (B, hidden_dim)
        """
        h = self.embed(x)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for layer in self.col_layers:
            h = layer(h)
        for layer in self.row_layers:
            h = layer(h, coords)
        return self.norm_out(h)[:, 0]   # (B, D)
