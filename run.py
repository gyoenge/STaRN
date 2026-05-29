"""STaRN training entry point.

Usage:
    cd /root/workspace/STaRN
    python run.py

All hyperparameters are in configs/config.py — edit directly, no CLI flags.
"""

import torch

from configs.config import Config
from dataset.loader import HestRadiomicsDataset, build_loader
from model.augment import FeatureAugment
from model.loss import STaRNLoss
from model.tabular import SummaryTableModel


def train():
    cfg = Config()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    # ── dataset & loader ──────────────────────────────────────────────────────
    dataset = HestRadiomicsDataset(
        dataroot=cfg.data_root,
        sample_ids=list(cfg.sample_ids),
        n_genes=cfg.n_genes,
        gene_criteria=cfg.gene_criteria,
    )
    loader = build_loader(
        dataset,
        batch_size=cfg.batch_size,
        n_neighbors=cfg.n_neighbors,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    print(dataset)

    # ── model ─────────────────────────────────────────────────────────────────
    model = SummaryTableModel(
        num_features=cfg.num_features,
        hidden_dim=cfg.hidden_dim,
        num_col_layers=cfg.num_col_layers,
        num_row_layers=cfg.num_row_layers,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
        device=cfg.device,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SummaryTableModel — {n_params:,} trainable parameters")

    # ── losses & optimiser ────────────────────────────────────────────────────
    criterion = STaRNLoss(
        k_pos=cfg.k_pos,
        k_neg=cfg.k_neg,
        temperature=cfg.temperature,
        w_self=cfg.w_self,
        w_col=cfg.w_col,
        w_row=cfg.w_row,
    )
    augment = FeatureAugment(
        noise_std=cfg.noise_std,
        mask_prob=cfg.mask_prob,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg.epochs):
        model.train()

        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)

        running = {"loss": 0.0, "l_self": 0.0, "l_col": 0.0, "l_row": 0.0}

        for step, batch in enumerate(loader):
            rad     = batch["radiomics"].to(device)   # (B, F)
            uni_emb = batch["uni_emb"].to(device)     # (B, 1024)

            # Two augmented views for L_self; view-a also used for L_col / L_row
            rad_a = augment(rad)
            rad_b = augment(rad)

            z_col, z_row, z_out_a, _ = model(rad_a)
            _,     _,     z_out_b, _ = model(rad_b)

            loss, loss_dict = criterion(z_col, z_row, z_out_a, z_out_b, uni_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in running:
                running[k] += loss_dict[k]

            if step % cfg.log_every == 0:
                avg = {k: v / (step + 1) for k, v in running.items()}
                print(
                    f"epoch {epoch:3d} | step {step:4d} | "
                    f"loss={avg['loss']:.4f}  "
                    f"l_self={avg['l_self']:.4f}  "
                    f"l_col={avg['l_col']:.4f}  "
                    f"l_row={avg['l_row']:.4f}"
                )

        scheduler.step()

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt = cfg.save_dir / f"epoch_{epoch:03d}.pt"
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt)

    print("Training complete.")


if __name__ == "__main__":
    train()
