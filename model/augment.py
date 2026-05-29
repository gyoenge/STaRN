import torch
import torch.nn as nn


class FeatureAugment(nn.Module):
    """Stochastic augmentation for radiomics feature vectors.

    Applies Gaussian noise and random feature masking to produce two
    independent views of the same input, used for L_self contrastive learning.

    Args:
        noise_std: Std of additive Gaussian noise (0 to disable).
        mask_prob: Per-feature zeroing probability (0 to disable).
    """

    def __init__(self, noise_std: float = 0.1, mask_prob: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.mask_prob = mask_prob

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F) — radiomics features
        Returns:
            (B, F) — augmented copy
        """
        x = x.clone().float()
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.mask_prob > 0:
            mask = torch.bernoulli(torch.full_like(x, self.mask_prob)).bool()
            x = x.masked_fill(mask, 0.0)
        return x
