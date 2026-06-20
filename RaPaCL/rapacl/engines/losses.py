from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_info_nce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)

    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def simclr_nt_xent_loss_multi_pos(
    embeddings: torch.Tensor,
    idxes,
    temperature: float = 0.07,
) -> torch.Tensor:
    device = embeddings.device

    if not isinstance(idxes, torch.Tensor):
        idxes = torch.tensor(idxes, device=device, dtype=torch.long)
    else:
        idxes = idxes.to(device=device, dtype=torch.long)

    bsz = embeddings.size(0)

    z = F.normalize(embeddings, dim=1)
    sim_matrix = torch.mm(z, z.t()) / temperature

    diag_mask = torch.eye(bsz, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(diag_mask, -1e4)

    pos_mask = (idxes.unsqueeze(1) == idxes.unsqueeze(0)) & (~diag_mask)

    logsumexp = torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    log_prob = sim_matrix - logsumexp

    pos_log_prob_sum = (pos_mask.float() * log_prob).sum(dim=1)
    num_pos = pos_mask.sum(dim=1).clamp_min(1)

    pos_log_prob_mean = pos_log_prob_sum / num_pos

    return -pos_log_prob_mean.mean()


def compute_multimodal_contrastive_loss_single_simclr(
    image_token_embedding: torch.Tensor,       # [B, D]
    radiomics_token_embedding: torch.Tensor,   # [B, N, D]
    idxes: torch.Tensor,                       # [B]
    temperature: float = 0.07,
) -> torch.Tensor:
    device = image_token_embedding.device

    if radiomics_token_embedding.dim() != 3:
        raise ValueError(
            "radiomics_token_embedding must be [B, N, D]. "
            f"Got shape: {tuple(radiomics_token_embedding.shape)}"
        )

    bsz, n_rad, dim = radiomics_token_embedding.shape

    if image_token_embedding.shape != (bsz, dim):
        raise ValueError(
            "image_token_embedding must be [B, D] and match radiomics token dim. "
            f"image={tuple(image_token_embedding.shape)}, "
            f"radiomics={tuple(radiomics_token_embedding.shape)}"
        )

    idxes = idxes.to(device=device, dtype=torch.long)

    rad_all = radiomics_token_embedding.reshape(bsz * n_rad, dim)

    combined = torch.cat(
        [image_token_embedding, rad_all],
        dim=0,
    )

    idxes_rad = idxes.repeat_interleave(n_rad)
    combined_idxes = torch.cat([idxes, idxes_rad], dim=0)

    return simclr_nt_xent_loss_multi_pos(
        embeddings=combined,
        idxes=combined_idxes,
        temperature=temperature,
    )


def compute_mmcl_loss(
    out: dict[str, torch.Tensor],
    idxes: torch.Tensor,
    loss_type: str = "symmetric",
    temperature: float = 0.07,
) -> torch.Tensor:
    loss_type = loss_type.lower()

    if loss_type in {"symmetric", "symmetric_info_nce", "info_nce"}:
        return symmetric_info_nce(
            out["path_z"],
            out["rad_contrast_z"],
            temperature=temperature,
        )

    if loss_type in {"multipos_nt_xent", "simclr_multi_pos", "multimodal_simclr"}:
        if "rad_token_z" not in out:
            raise KeyError(
                "MMCL_LOSS='single_simclr' requires out['rad_token_z'] "
                "with shape [B, n_rad, D]."
            )

        return compute_multimodal_contrastive_loss_single_simclr(
            image_token_embedding=out["path_z"],
            radiomics_token_embedding=out["rad_token_z"],
            idxes=idxes,
            temperature=temperature,
        )

    raise ValueError(f"Unknown MMCL_LOSS: {loss_type}")