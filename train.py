"""STaRN training entry point.

Usage:
    cd /root/workspace/STaRN
    python train.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 train.py

All hyperparameters are in configs/config.py — edit directly, no CLI flags.

Batch layout (per step):
    index 0              : anchor
    index 1..n_neighbors : spatial kNN neighbours
    index 1+n..n+n_sem   : UNI semantic neighbours
    index n+n_sem+1..B-1 : random globals

Training computes:
    Z^S = concat( z_out[anchor], mean(z_out[neighbours]), mean(z_out[globals]) )
    Z^T = AuxNeighborAttention( UNI+scFoundation[anchor : anchor+n_total_neighbours] )
    L   = w_self · L_self(z_out_a, z_out_b) + w_distill · L_distill(Z^S, Z^T)
"""

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.config import Config
from dataset.loader import HestRadiomicsDataset, build_loader
from model.augment import FeatureAugment
from model.loss import STaRNLoss
from model.tabular import SummaryTableModel
from model.teacher import AuxNeighborAttention
from utils.ddp import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    ddp_barrier,
    unwrap_model,
)


def _build_zs(
    z_out: torch.Tensor,
    n_neighbors: int,
    n_semantic: int,
) -> torch.Tensor:
    """Construct the anchor's context embedding Z^S.

    Z^S = concat( z_anchor, mean(z_neighbours), mean(z_globals) )

    Args:
        z_out:       (B, proj_dim) — L2-normalised spot embeddings for the batch.
        n_neighbors: Number of spatial neighbours (immediately after anchor).
        n_semantic:  Number of semantic neighbours (after spatial neighbours).
    Returns:
        (3 * proj_dim,) — L2-normalised Z^S for the anchor.
    """
    n_total_nbr = n_neighbors + n_semantic
    z_anchor = z_out[0]                          # (proj_dim,)
    z_nbr    = z_out[1 : 1 + n_total_nbr].mean(0)  # (proj_dim,)
    z_glob   = z_out[1 + n_total_nbr :].mean(0)    # (proj_dim,)
    z_s = torch.cat([z_anchor, z_nbr, z_glob], dim=-1)  # (3 * proj_dim,)
    return F.normalize(z_s, dim=-1)


def train():
    cfg = Config()
    distributed, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}") if distributed else torch.device(cfg.device)

    if is_main_process():
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            print(f"[DDP] world_size={world_size}")

    # ── dataset & loader ──────────────────────────────────────────────────────
    # Rank 0 builds the dataset first so it writes the .npy feature caches alone;
    # other ranks wait, then load the now-cached files (avoids concurrent writes).
    if is_main_process():
        dataset = HestRadiomicsDataset(
            sources=cfg.data_sources,
            n_genes=cfg.n_genes,
            gene_criteria=cfg.gene_criteria,
        )
    ddp_barrier()
    if not is_main_process():
        dataset = HestRadiomicsDataset(
            sources=cfg.data_sources,
            n_genes=cfg.n_genes,
            gene_criteria=cfg.gene_criteria,
        )

    loader = build_loader(
        dataset,
        batch_size=cfg.batch_size,
        n_neighbors=cfg.n_neighbors,
        n_semantic=cfg.n_semantic,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    if is_main_process():
        print(dataset)

    # ── models ────────────────────────────────────────────────────────────────
    model = SummaryTableModel(
        num_features=cfg.num_features,
        hidden_dim=cfg.hidden_dim,
        num_col_layers=cfg.num_col_layers,
        num_row_layers=cfg.num_row_layers,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
        n_pos_bins=cfg.n_pos_bins,
        device=str(device),
    )

    teacher = AuxNeighborAttention(
        uni_dim=cfg.uni_dim,
        scfound_dim=cfg.scfoundation_dim,
        fuse_dim=cfg.teacher_fuse_dim,
        num_heads=cfg.num_heads,
        zs_dim=3 * cfg.proj_dim,
        dropout=cfg.dropout,
    ).to(device)

    n_student = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_teacher = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    if is_main_process():
        print(f"SummaryTableModel    — {n_student:,} trainable params")
        print(f"AuxNeighborAttention — {n_teacher:,} trainable params")

    if distributed:
        # Both branches run the same fixed computation graph every step
        # (model: two forward calls -> one backward; teacher: one forward call).
        model = DDP(model, device_ids=[local_rank], static_graph=True)
        teacher = DDP(teacher, device_ids=[local_rank], static_graph=True)

    # ── losses & optimiser ────────────────────────────────────────────────────
    criterion = STaRNLoss(
        temperature=cfg.temperature,
        w_self=cfg.w_self,
        w_distill=cfg.w_distill,
    )
    augment = FeatureAugment(
        noise_std=cfg.noise_std,
        mask_prob=cfg.mask_prob,
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(teacher.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    n_total_nbr = cfg.n_neighbors + cfg.n_semantic

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg.epochs):
        model.train()
        teacher.train()

        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)

        running = {"loss": 0.0, "l_self": 0.0, "l_distill": 0.0}

        for step, batch in enumerate(loader):
            rad      = batch["radiomics"].to(device)        # (B, F)
            uni_emb  = batch["uni_emb"].to(device)           # (B, 1024)
            sf_emb   = batch["scfoundation_emb"].to(device)  # (B, 3072)
            coords   = batch["coord"].to(device)             # (B, 2)

            # Two augmented views for L_self
            rad_a = augment(rad)
            rad_b = augment(rad)

            # Student branch — view a used for Z^S and L_col/L_row; view b for L_self only
            _, _, z_out_a, _ = model(rad_a, coords)
            _, _, z_out_b, _ = model(rad_b, coords)

            # Z^S: anchor context representation (view a)
            z_s = _build_zs(z_out_a, cfg.n_neighbors, cfg.n_semantic)   # (3*proj_dim,)

            # Z^T: teacher context representation from UNI + scFoundation embeddings
            z_t = teacher(uni_emb, sf_emb, n_total_nbr)                  # (3*proj_dim,)

            loss, loss_dict = criterion(
                z_out_a,
                z_out_b,
                z_s.unsqueeze(0),   # (1, 3*proj_dim)
                z_t.unsqueeze(0),   # (1, 3*proj_dim)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(teacher.parameters()), 1.0
            )
            optimizer.step()

            for k in running:
                running[k] += loss_dict[k]

            if step % cfg.log_every == 0 and is_main_process():
                avg = {k: v / (step + 1) for k, v in running.items()}
                print(
                    f"epoch {epoch:3d} | step {step:4d} | "
                    f"loss={avg['loss']:.4f}  "
                    f"l_self={avg['l_self']:.4f}  "
                    f"l_distill={avg['l_distill']:.4f}"
                )

        scheduler.step()

        # ── checkpoint ────────────────────────────────────────────────────────
        if is_main_process():
            ckpt = cfg.save_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch":   epoch,
                "model":   unwrap_model(model).state_dict(),
                "teacher": unwrap_model(teacher).state_dict(),
            }, ckpt)

    if is_main_process():
        print("Training complete.")
    cleanup_ddp()


if __name__ == "__main__":
    train()
