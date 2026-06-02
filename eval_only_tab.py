"""STaRN eval entry point — MLP gene head with LOOCV on IDC_Bench_Xenium_4.

Usage:
    cd /root/workspace/STaRN
    python eval.py

All hyperparameters are at the top of the file — edit directly, no CLI flags.

The SummaryTableModel backbone is loaded from CKPT_PATH and frozen.
Only the MLP gene head is trained per fold.  With 4 samples the LOOCV
produces 4 folds; mean ± std PCC is reported at the end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from dataset.loader import HestRadiomicsDataset, get_common_genes
from model.tabular import SummaryTableModel
from torch.utils.data import DataLoader

# ── hyperparameters ───────────────────────────────────────────────────────────

CKPT_PATH = Path("checkpoints/epoch_025.pt")

EVAL_DATA_ROOT = Path(
    "/root/workspace/datasets/hest_radiomics/IDC_Bench_Xenium_4/IDC/"
)
SAMPLE_IDS: tuple[str, ...] = ("NCBI783", "NCBI785", "TENX95", "TENX99")

N_GENES       = 250
GENE_CRITERIA = "var"

HEAD_EPOCHS       = 50
HEAD_LR           = 3e-4
HEAD_WEIGHT_DECAY = 1e-4
HEAD_HIDDEN_DIM   = 256
HEAD_DROPOUT      = 0.1

BATCH_SIZE  = 256
NUM_WORKERS = 4
LOG_EVERY   = 10

SAVE_DIR = Path("checkpoints/loocv")

# ── MLP gene head ─────────────────────────────────────────────────────────────

class MLPGeneHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_genewise_pcc(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray]:
    pred   = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    pred_c   = pred   - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    denom = (
        torch.sqrt((pred_c ** 2).sum(dim=0) * (target_c ** 2).sum(dim=0)) + eps
    )
    pcc_per_gene = (pred_c * target_c).sum(dim=0) / denom
    return pcc_per_gene.mean().item(), pcc_per_gene.numpy()


# ── helpers ───────────────────────────────────────────────────────────────────

def build_backbone(cfg: Config, device: torch.device) -> tuple[SummaryTableModel, int]:
    model = SummaryTableModel(
        num_features=cfg.num_features,
        hidden_dim=cfg.hidden_dim,
        num_col_layers=cfg.num_col_layers,
        num_row_layers=cfg.num_row_layers,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
        device=str(device),
    )
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, ckpt["epoch"]


def make_loader(
    sample_ids: list[str],
    gene_names: list[str],
    shuffle: bool,
) -> DataLoader:
    dataset = HestRadiomicsDataset(
        dataroot=EVAL_DATA_ROOT,
        sample_ids=sample_ids,
        gene_names=gene_names,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    backbone: SummaryTableModel,
    gene_head: MLPGeneHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    gene_head.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        rad  = batch["radiomics"].to(device)
        st   = batch["st"].to(device)
        feat = backbone.encode(rad)
        pred = gene_head(feat)
        all_preds.append(pred.cpu())
        all_targets.append(st.cpu())

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)

    val_mse             = F.mse_loss(preds, targets).item()
    mean_pcc, per_gene  = compute_genewise_pcc(preds, targets)
    return val_mse, mean_pcc, per_gene


def run_fold(
    fold: int,
    val_id: str,
    train_ids: list[str],
    gene_names: list[str],
    backbone: SummaryTableModel,
    cfg: Config,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    print(f"\n{'─' * 60}")
    print(f"Fold {fold}  val={val_id}  train={train_ids}")

    train_loader = make_loader(train_ids, gene_names, shuffle=True)
    val_loader   = make_loader([val_id],  gene_names, shuffle=False)
    print(
        f"  train spots: {len(train_loader.dataset)}"
        f"  val spots: {len(val_loader.dataset)}"
    )

    gene_head = MLPGeneHead(
        in_dim=cfg.hidden_dim,
        hidden_dim=HEAD_HIDDEN_DIM,
        out_dim=len(gene_names),
        dropout=HEAD_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        gene_head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    best_pcc      = -1.0
    best_epoch    = -1
    best_per_gene = None

    for epoch in range(HEAD_EPOCHS):
        gene_head.train()
        total_loss, n_seen = 0.0, 0

        for batch in train_loader:
            rad = batch["radiomics"].to(device)
            st  = batch["st"].to(device)

            with torch.no_grad():
                feat = backbone.encode(rad)

            pred = gene_head(feat)
            loss = criterion(pred, st)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * rad.size(0)
            n_seen     += rad.size(0)

        train_mse = total_loss / n_seen
        val_mse, val_pcc, per_gene = evaluate(backbone, gene_head, val_loader, device)

        is_best = val_pcc > best_pcc
        if is_best:
            best_pcc      = val_pcc
            best_epoch    = epoch
            best_per_gene = per_gene
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "fold":             fold,
                    "epoch":            epoch,
                    "gene_head":        gene_head.state_dict(),
                    "val_genewise_pcc": val_pcc,
                    "per_gene_pcc":     per_gene,
                    "gene_names":       gene_names,
                    "val_id":           val_id,
                    "train_ids":        train_ids,
                    "ckpt_path":        str(CKPT_PATH),
                },
                SAVE_DIR / f"fold_{fold}_best.pt",
            )

        if epoch % LOG_EVERY == 0 or epoch == HEAD_EPOCHS - 1 or is_best:
            marker = " *" if is_best else ""
            print(
                f"  epoch {epoch:3d} | "
                f"train_mse={train_mse:.4f} | "
                f"val_mse={val_mse:.4f} | "
                f"val_pcc={val_pcc:.4f}"
                f"{marker}"
            )

    print(f"  → best PCC={best_pcc:.4f} at epoch {best_epoch}")
    return best_pcc, best_per_gene


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg    = Config()
    device = torch.device(cfg.device)

    # gene names from the bench set
    st_paths   = [EVAL_DATA_ROOT / "st" / f"{sid}.h5ad" for sid in SAMPLE_IDS]
    gene_names = get_common_genes(st_paths, k=N_GENES, criteria=GENE_CRITERIA)
    print(f"Checkpoint : {CKPT_PATH}")
    print(f"Data root  : {EVAL_DATA_ROOT}")
    print(f"Samples    : {list(SAMPLE_IDS)}")
    print(f"Genes      : {len(gene_names)}")

    backbone, ckpt_epoch = build_backbone(cfg, device)
    print(f"Backbone loaded from epoch {ckpt_epoch}, all params frozen")

    sample_list = list(SAMPLE_IDS)
    fold_pccs: list[float]      = []
    fold_pergene: list[np.ndarray] = []

    for fold, val_id in enumerate(sample_list):
        train_ids = [s for s in sample_list if s != val_id]
        best_pcc, best_per_gene = run_fold(
            fold, val_id, train_ids, gene_names, backbone, cfg, device
        )
        fold_pccs.append(best_pcc)
        if best_per_gene is not None:
            fold_pergene.append(best_per_gene)

    # ── LOOCV summary ─────────────────────────────────────────────────────────
    mean_pcc = float(np.mean(fold_pccs))
    std_pcc  = float(np.std(fold_pccs, ddof=1) if len(fold_pccs) > 1 else 0.0)

    print(f"\n{'=' * 60}")
    print("LOOCV Results")
    print("─" * 60)
    for fold, (sid, pcc) in enumerate(zip(sample_list, fold_pccs)):
        print(f"  Fold {fold}  val={sid:10s}  PCC={pcc:.4f}")
    print("─" * 60)
    print(f"  Mean PCC : {mean_pcc:.4f}")
    print(f"  Std  PCC : {std_pcc:.4f}")

    if fold_pergene:
        mean_per_gene = np.stack(fold_pergene).mean(axis=0)
        top5 = np.argsort(mean_per_gene)[::-1][:5]
        bot5 = np.argsort(mean_per_gene)[:5]
        print("\nTop-5 genes (mean across folds):")
        for i in top5:
            print(f"  {gene_names[i]:20s}  {mean_per_gene[i]:.4f}")
        print("Bottom-5 genes (mean across folds):")
        for i in bot5:
            print(f"  {gene_names[i]:20s}  {mean_per_gene[i]:.4f}")

    print("=" * 60)
    print(f"Per-fold checkpoints: {SAVE_DIR}/fold_{{i}}_best.pt")


if __name__ == "__main__":
    main()