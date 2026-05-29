# Variant of TransTab + SAINT

import math
import torch
import torch.nn as nn
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


class SummaryTableModel(nn.Module):
    """Tabular representation model for spatial radiomics Summary Tables.

    Interleaves ColumnAttention and RowAttention layers (SAINT-style) on top of
    FeatureEmbedding. A learnable [CLS] token is prepended; its final hidden state
    is returned as the sample-level embedding for contrastive learning / gene prediction.

    Input:  (B, num_features) — raw radiomics feature tensor
    Output: cls_emb   (B, hidden_dim)            — CLS embedding
            token_emb (B, num_features, hidden_dim) — per-feature token embeddings
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 256,
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

        # Each layer: ColumnAttention then RowAttention
        self.layers = nn.ModuleList([
            nn.ModuleList([
                ColumnAttention(hidden_dim, num_heads, ffn_dim, dropout, layer_norm_eps),
                RowAttention(hidden_dim, num_heads, ffn_dim, dropout, layer_norm_eps),
            ])
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, num_features) — raw radiomics features
        Returns:
            cls_emb:   (B, hidden_dim)
            token_emb: (B, num_features, hidden_dim)
        """
        h = self.embed(x)                                   # (B, F, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)     # (B, 1, D)
        h = torch.cat([cls, h], dim=1)                     # (B, 1+F, D)

        for col_attn, row_attn in self.layers:
            h = col_attn(h)
            h = row_attn(h)

        h = self.norm(h)
        cls_emb   = h[:, 0]    # (B, D)
        token_emb = h[:, 1:]   # (B, F, D)
        return cls_emb, token_emb
