import torch
import torch.nn as nn
import torch.nn.functional as F


# ── individual losses ─────────────────────────────────────────────────────────

class SelfContrastiveLoss(nn.Module):
    """NT-Xent contrastive loss on two augmented views of the same batch.

    L_self = -log( exp(sim(z^a_i, z^b_i)/τ) / Σ_j exp(sim(z^a_i, z^b_j)/τ) )

    Applied to all B spots in the batch (anchor + neighbors + globals).
    Positive pair: the two augmented views of the same spot.

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

        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float('-inf'))

        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(B),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)


class DistillLoss(nn.Module):
    """Cosine distillation loss: align Z^S (student) with Z^T (teacher).

    L_distill = 1 − cosine_similarity(Z^S, Z^T)

    Both inputs should be L2-normalised. Minimising this loss maximises
    cosine similarity between student and teacher context embeddings.
    """

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_s: (D,) or (B, D) — student Z^S (L2-normalised)
            z_t: (D,) or (B, D) — teacher Z^T (L2-normalised)
        Returns:
            scalar loss
        """
        return (1.0 - F.cosine_similarity(z_s, z_t, dim=-1)).mean()


# ── combined loss ─────────────────────────────────────────────────────────────

class STaRNLoss(nn.Module):
    """Combined loss: L_total = w_self · L_self + w_distill · L_distill

    L_self:    NT-Xent on two augmented views of all batch spots (representation stability).
    L_distill: Cosine alignment between anchor's Z^S and teacher's Z^T (UNI distillation).

    Args:
        temperature: shared softmax temperature for L_self.
        w_self:      weight for L_self.
        w_distill:   weight for L_distill (set 0 to disable).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        w_self: float = 1.0,
        w_distill: float = 1.0,
    ):
        super().__init__()
        self.self_loss    = SelfContrastiveLoss(temperature)
        self.distill_loss = DistillLoss()
        self.w_self    = w_self
        self.w_distill = w_distill

    def forward(
        self,
        z_out_a:  torch.Tensor,
        z_out_b:  torch.Tensor,
        z_s:      torch.Tensor,
        z_t:      torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            z_out_a: (B, D) — output embedding, view a  (for L_self)
            z_out_b: (B, D) — output embedding, view b  (for L_self)
            z_s:     (D,) or (1, D) — anchor's Z^S      (for L_distill)
            z_t:     (D,) or (1, D) — anchor's Z^T      (for L_distill)
        Returns:
            total loss (scalar), dict of component losses
        """
        l_self    = self.self_loss(z_out_a, z_out_b)
        l_distill = self.distill_loss(z_s, z_t)

        total = self.w_self * l_self + self.w_distill * l_distill

        return total, {
            "loss":      total.item(),
            "l_self":    l_self.item(),
            "l_distill": l_distill.item(),
        }
