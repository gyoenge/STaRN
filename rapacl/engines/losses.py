from __future__ import annotations

import torch 
import torch.nn.functional as F


#### 

def symmetric_info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

#### 

def simclr_nt_xent_loss_multi_pos(
        embeddings: torch.Tensor,  # [B, d]
        idxes,
        temperature: float = 0.07
    ) -> torch.Tensor:
        device = embeddings.device
        if not isinstance(idxes, torch.Tensor):
            idxes = torch.tensor(idxes, device=device, dtype=torch.long)

        B, d = embeddings.shape
        z = F.normalize(embeddings, dim=1)                  # 1) L2-normalize
        sim_matrix = torch.mm(z, z.t()) / temperature       # 2) (B,B) similarity matrix

        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e4)  # 3) Mask diagonal with large negative value

        pos_mask = (idxes.unsqueeze(1) == idxes.unsqueeze(0)) & (~diag_mask)
        logsumexp = torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        log_prob = sim_matrix - logsumexp

        pos_log_prob_sum = (pos_mask * log_prob).sum(dim=1)
        num_pos = pos_mask.sum(dim=1).clamp_min(1)
        pos_log_prob_mean = pos_log_prob_sum / num_pos

        loss = -pos_log_prob_mean.mean()
        return loss

def compute_multimodal_contrastive_loss_singleSimCLR(
        image_token_embedding: torch.Tensor,       # [B, 384]
        radiomics_token_embedding: torch.Tensor,   # [B, n_rad, 384]
        idxes: torch.Tensor,                       # [B]
        temperature: float = 0.07
    ):
        """
        Combine image and radiomics embeddings into a single (n_rad+1, d) tensor,
        then compute SimCLR (NT-Xent) loss with positive pairs defined by identical indices.


        Args:
            image_token_embedding: [B, d] — image embeddings
            radiomics_token_embedding: [B, n_rad, d] — radiomics tokens per tumor
            idxes: list or tensor of sample IDs (length B)
            temperature: temperature parameter for NT-Xent loss

        Returns:
            Scalar NT-Xent contrastive loss
        """
        device = image_token_embedding.device
        B, n_rad, d = radiomics_token_embedding.shape

        # 1) Flatten radiomics tokens: [B, n_rad, d] → [B*n_rad, d]
        rad_all = radiomics_token_embedding.view(B*n_rad, d)

        # 2) Concatenate with image tokens → [n_rad+1, d]
        combined = torch.cat([image_token_embedding, rad_all], dim=0)  # (B*(n_rad+1), d)

        # 3) Expand sample indices to match tokens → [B*(n_rad+1)]
        if isinstance(idxes, torch.Tensor):
            idxes_rad = idxes.repeat_interleave(n_rad)  # (B*n_rad,)
            combined_idxes = torch.cat([idxes, idxes_rad], dim=0)  # (B*(n_rad+1),)
        else:
            idxes_rad = []
            for x in idxes:
                idxes_rad.extend([x]*n_rad)
            combined_idxes = list(idxes) + idxes_rad

        # 4) Compute SimCLR loss on all tokens at once
        loss = simclr_nt_xent_loss_multi_pos(combined, combined_idxes, temperature=temperature)
        return loss
