import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_uni_masks(
    uni_emb: torch.Tensor,
    k_pos: int,
    k_neg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build top-K positive and bottom-K negative masks from UNI cosine similarity.

    Args:
        uni_emb: (B, D_uni) — UNI patch embeddings (unnormalised OK)
        k_pos:   number of top-K UNI-similar patches per anchor → positive mask
        k_neg:   number of bottom-K UNI-similar patches per anchor → negative mask

    Returns:
        pos_mask: (B, B) float — 1 at top-k positions (self excluded)
        neg_mask: (B, B) float — 1 at bottom-k positions (self excluded)
    """
    B = uni_emb.size(0)
    k_pos = min(k_pos, B - 1)
    k_neg = min(k_neg, B - 1)

    uni_norm = F.normalize(uni_emb.float(), dim=-1)
    sim = uni_norm @ uni_norm.T  # (B, B)

    eye = torch.eye(B, dtype=torch.bool, device=uni_emb.device)

    # top-k positive (exclude self)
    sim_pos = sim.masked_fill(eye, float('-inf'))
    _, top_idx = sim_pos.topk(k_pos, dim=-1)
    pos_mask = torch.zeros(B, B, device=uni_emb.device)
    pos_mask.scatter_(-1, top_idx, 1.0)

    # bottom-k negative (exclude self)
    sim_neg = sim.masked_fill(eye, float('inf'))
    _, bot_idx = sim_neg.topk(k_neg, dim=-1, largest=False)
    neg_mask = torch.zeros(B, B, device=uni_emb.device)
    neg_mask.scatter_(-1, bot_idx, 1.0)

    return pos_mask, neg_mask


def _proto(mask: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compute mean-pool prototype for each row given a selection mask.

    Args:
        mask: (B, B) — 1 at selected positions
        z:    (B, D) — L2-normalised embeddings
    Returns:
        (B, D) — L2-normalised prototypes
    """
    # mask[i, j] = 1  ⟹  patch j is selected for anchor i
    proto = (mask.unsqueeze(-1) * z.unsqueeze(0)).sum(1)   # (B, D)
    proto = proto / mask.sum(1, keepdim=True).clamp(min=1.0)
    return F.normalize(proto, dim=-1)


# ── individual losses ─────────────────────────────────────────────────────────

class SelfContrastiveLoss(nn.Module):
    """NT-Xent contrastive loss on two augmented views of the same batch.

    ℒ_self = -log( exp(sim(z^a_i, z^b_i)/τ) / Σ_j exp(sim(z^a_i, z^b_j)/τ) )

    Args:
        temperature: softmax temperature τ.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_a, z_b: (B, D) — L2-normalised embeddings from two augmented views
        Returns:
            scalar loss
        """
        B = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)              # (2B, D)
        sim = (z @ z.T) / self.temperature             # (2B, 2B)

        # Mask self-similarity on the diagonal
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float('-inf'))

        # Positive for i is i+B, and for i+B is i
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(B),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)


class _DistillContrastiveLoss(nn.Module):
    """UNI-guided binary contrastive loss (shared base for L_col and L_row).

    For each anchor i, uses UNI cosine similarity to select:
        pos_proto = mean( z[top-K UNI neighbours] )
        neg_proto = mean( z[bottom-K UNI remotes] )

    ℒ = -log( exp(sim(z_i, pos)/τ) / (exp(sim(z_i, pos)/τ) + exp(sim(z_i, neg)/τ)) )

    Args:
        k_pos:       top-K UNI-similar patches as positives.
        k_neg:       bottom-K UNI-dissimilar patches as negatives.
        temperature: softmax temperature τ.
    """

    def __init__(self, k_pos: int = 5, k_neg: int = 5, temperature: float = 0.1):
        super().__init__()
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.temperature = temperature

    def forward(self, z: torch.Tensor, uni_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:       (B, D) — L2-normalised embeddings (z_col or z_row)
            uni_emb: (B, D_uni) — UNI embeddings for positive/negative selection
        Returns:
            scalar loss
        """
        pos_mask, neg_mask = _build_uni_masks(uni_emb, self.k_pos, self.k_neg)

        pos_proto = _proto(pos_mask, z)                   # (B, D)
        neg_proto = _proto(neg_mask, z)                   # (B, D)

        pos_sim = (z * pos_proto).sum(-1) / self.temperature   # (B,)
        neg_sim = (z * neg_proto).sum(-1) / self.temperature   # (B,)

        # -log( e^pos / (e^pos + e^neg) )  =  -pos + log(e^pos + e^neg)
        loss = -pos_sim + torch.logaddexp(pos_sim, neg_sim)
        return loss.mean()


class ColDistillLoss(_DistillContrastiveLoss):
    """UNI-guided patch-level semantic ordering loss on z_col.

    Trains the column-attention representation to reflect UNI's semantic ranking
    of individual patches (patch-level distillation).
    """


class RowDistillLoss(_DistillContrastiveLoss):
    """UNI-guided neighbourhood context distillation loss on z_row.

    Trains the row-attention representation so that an anchor's embedding is
    closer to the mean context of its UNI-semantic neighbours than to the mean
    context of UNI-dissimilar remotes (group-level context distillation).
    """


# ── combined loss ─────────────────────────────────────────────────────────────

class STaRNLoss(nn.Module):
    """Combined loss: ℒ_total = w_self·ℒ_self + w_col·ℒ_col + w_row·ℒ_row

    Args:
        k_pos, k_neg:  UNI top/bottom-K for distillation selection.
        temperature:   shared softmax temperature.
        w_self:        weight for ℒ_self (representation stability).
        w_col:         weight for ℒ_col (patch-level semantic ordering).
        w_row:         weight for ℒ_row (contextual manifold, set 0 to disable).
    """

    def __init__(
        self,
        k_pos: int = 5,
        k_neg: int = 5,
        temperature: float = 0.1,
        w_self: float = 1.0,
        w_col: float = 1.0,
        w_row: float = 0.5,
    ):
        super().__init__()
        self.self_loss = SelfContrastiveLoss(temperature)
        self.col_loss  = ColDistillLoss(k_pos, k_neg, temperature)
        self.row_loss  = RowDistillLoss(k_pos, k_neg, temperature)
        self.w_self = w_self
        self.w_col  = w_col
        self.w_row  = w_row

    def forward(
        self,
        z_col:   torch.Tensor,
        z_row:   torch.Tensor,
        z_out_a: torch.Tensor,
        z_out_b: torch.Tensor,
        uni_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            z_col:   (B, D) — column-stage embedding (from view a)
            z_row:   (B, D) — row-stage embedding    (from view a)
            z_out_a: (B, D) — output embedding, view a
            z_out_b: (B, D) — output embedding, view b
            uni_emb: (B, D_uni) — UNI teacher embeddings
        Returns:
            total loss (scalar), dict of component losses
        """
        l_self = self.self_loss(z_out_a, z_out_b)
        l_col  = self.col_loss(z_col, uni_emb)
        l_row  = self.row_loss(z_row, uni_emb)

        total = self.w_self * l_self + self.w_col * l_col + self.w_row * l_row

        return total, {
            "loss":   total.item(),
            "l_self": l_self.item(),
            "l_col":  l_col.item(),
            "l_row":  l_row.item(),
        }
