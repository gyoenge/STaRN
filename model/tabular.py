# Variant of TransTab + SAINT

import math
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
        # Learnable per-column embedding (replaces BERT tokenization of column names)
        self.col_embedding = nn.Embedding(num_features, hidden_dim)
        nn_init.kaiming_normal_(self.col_embedding.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Additive bias after value scaling (from TransTabNumEmbedding)
        self.num_bias = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn_init.uniform_(self.num_bias, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim))

        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.register_buffer('col_indices', torch.arange(num_features))
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_features) — raw numerical feature tensor
        Returns:
            (B, num_features, hidden_dim) — per-feature token embeddings
        """
        col_emb = self.col_embedding(self.col_indices)              # (F, D)
        col_emb = self.norm(col_emb)
        col_emb = self.dropout(col_emb)
        col_emb = col_emb.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, F, D)

        # TransTabNumEmbedding: scale column embedding by feature value, add bias
        feat_emb = col_emb * x.unsqueeze(-1).float() + self.num_bias  # (B, F, D)
        feat_emb = self.align_layer(feat_emb)
        return feat_emb


class ColumnAttention(nn.Module):
    """Standard multi-head self-attention over feature tokens within each sample.

    Attends over the F (feature) dimension: (B, F, D) → (B, F, D).
    Follows SAINT's column-wise self-attention block.
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
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, D)
        Returns:
            (B, F, D)
        """
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(h))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class RowAttention(nn.Module):
    """Intersample multi-head self-attention across spots for each feature position.

    Transposes to (F, B, D), attends over the B (spot) dimension, then transposes
    back: (B, F, D) → (B, F, D). Follows SAINT's intersample attention block.
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
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F, D)
        Returns:
            (B, F, D)
        """
        x_t = x.transpose(0, 1)        # (F, B, D) — features as sequence, spots as batch
        h, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm1(x_t + self.drop(h))
        x_t = self.norm2(x_t + self.drop(self.ffn(x_t)))
        return x_t.transpose(0, 1)     # (B, F, D)


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
          → ColumnAttention × num_col_layers  →  h_col  →  z_col (B, proj_dim)
          → RowAttention    × num_row_layers  →  h_final
              → z_row (B, proj_dim)   — for L_row distillation
              → z_out (B, proj_dim)   — for L_self contrastive

    The three projection heads share the same trunk but are kept separate so each
    loss gradient flows through a dedicated head without interfering.

    Returns:
        z_col:     (B, proj_dim) — L2-normalised, after column-attention stage
        z_row:     (B, proj_dim) — L2-normalised, after row-attention stage
        z_out:     (B, proj_dim) — L2-normalised, used for self-contrastive loss
        token_emb: (B, F, D)    — final per-feature token embeddings (for recon / probing)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_col_layers: int = 2,
        num_row_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 256,
        proj_dim: int = 128,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
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

        # Stage 1: column attention — learns within-patch feature interactions
        self.col_layers = nn.ModuleList([
            ColumnAttention(hidden_dim, num_heads, ffn_dim, dropout, layer_norm_eps)
            for _ in range(num_col_layers)
        ])
        self.norm_col = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.proj_col = _ProjHead(hidden_dim, proj_dim)

        # Stage 2: row attention — learns cross-patch neighbourhood context
        self.row_layers = nn.ModuleList([
            RowAttention(hidden_dim, num_heads, ffn_dim, dropout, layer_norm_eps)
            for _ in range(num_row_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.proj_row = _ProjHead(hidden_dim, proj_dim)   # for L_row
        self.proj_out = _ProjHead(hidden_dim, proj_dim)   # for L_self

        self.device = device
        self.to(device)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, num_features) — raw radiomics features
        Returns:
            z_col:     (B, proj_dim)
            z_row:     (B, proj_dim)
            z_out:     (B, proj_dim)
            token_emb: (B, num_features, hidden_dim)
        """
        h = self.embed(x)                                    # (B, F, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)      # (B, 1, D)
        h = torch.cat([cls, h], dim=1)                       # (B, 1+F, D)

        for layer in self.col_layers:
            h = layer(h)
        z_col = self.proj_col(self.norm_col(h[:, 0]))        # (B, proj_dim)

        for layer in self.row_layers:
            h = layer(h)
        h = self.norm_out(h)
        z_row     = self.proj_row(h[:, 0])                   # (B, proj_dim)
        z_out     = self.proj_out(h[:, 0])                   # (B, proj_dim)
        token_emb = h[:, 1:]                                 # (B, F, D)

        return z_col, z_row, z_out, token_emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-projection CLS embedding for downstream tasks.

        Args:
            x: (B, num_features)
        Returns:
            (B, hidden_dim)
        """
        h = self.embed(x)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        for layer in self.col_layers:
            h = layer(h)
        for layer in self.row_layers:
            h = layer(h)
        return self.norm_out(h)[:, 0]  # (B, D)
