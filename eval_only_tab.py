"""STaRN eval — tabular-only backbone, LOOCV on IDC_Bench_Xenium_4.

Usage:
    cd /root/workspace/STaRN
    python eval_only_tab.py

Pipeline
--------
1. LOOCV over SAMPLE_IDS: train MLPGeneHead on 3 samples, eval on 1.
2. Report gene-wise PCC per fold + mean ± std.

Architecture (frozen backbone, only MLPGeneHead trained)
---------------------------------------------------------
  rad  → SummaryTableModel.encode()  → (B, RAD_DIM=128)
                                             │
                                    MLP(256) → (B, N_GENES)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import Config
from dataset.loader import HestRadiomicsDataset, _PersampleDataset, build_loader, get_common_genes
from model.tabular import SummaryTableModel

# ── hyperparameters ───────────────────────────────────────────────────────────

CKPT_PATH = Path("checkpoints/epoch_025.pt")

EVAL_DATA_ROOT = Path(
    "/root/workspace/datasets/hest_radiomics/IDC_Bench_Xenium_4/IDC/"
)
SAMPLE_IDS: tuple[str, ...] = ("NCBI783", "NCBI785", "TENX95", "TENX99")

N_GENES       = 250
GENE_CRITERIA = "var"

HEAD_EPOCHS       = 50
HEAD_LR           = 1e-4
HEAD_WEIGHT_DECAY = 1e-3
HEAD_HIDDEN_DIM   = 256
HEAD_DROPOUT      = 0.3

PATIENCE = 3

BATCH_SIZE  = 32
N_NEIGHBORS = 6
NUM_WORKERS = 4
LOG_EVERY   = 1

SAVE_DIR = Path("checkpoints/loocv")

# ── MLP gene head ─────────────────────────────────────────────────────────────

class MLPGeneHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_genewise_pcc(
    pred:   torch.Tensor,
    target: torch.Tensor,
    eps:    float = 1e-8,
) -> tuple[float, np.ndarray]:
    pred   = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    pc     = pred   - pred.mean(dim=0, keepdim=True)
    tc     = target - target.mean(dim=0, keepdim=True)
    denom  = torch.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0)) + eps
    per    = (pc * tc).sum(0) / denom
    return per.mean().item(), per.numpy()


# ── dataset patch ─────────────────────────────────────────────────────────────

def _monkey_patch_dataset_getitem_without_patch() -> None:
    """Replace _PersampleDataset.__getitem__ to skip H5 patch image loading.

    This eval script never uses the patch image — skipping the H5 random read
    removes the main DataLoader I/O bottleneck.  loader.py is left untouched.
    """
    def _getitem_no_patch(self, idx: int) -> dict:
        barcode       = self.valid_barcodes[idx]
        patch_idx     = self.patch_barcode_to_idx[barcode]
        st_idx        = self.st_barcode_to_idx[barcode]
        radiomics_idx = self.radiomics_barcode_to_idx[barcode]

        if self.patch_coords is not None:
            coord = torch.tensor(self.patch_coords[patch_idx], dtype=torch.float32)
        elif self.st_coords is not None:
            coord = torch.tensor(self.st_coords[st_idx], dtype=torch.float32)
        else:
            coord = torch.tensor([-1.0, -1.0])

        st        = torch.from_numpy(self.st_matrix[st_idx].copy())
        radiomics = torch.from_numpy(self.radiomics_matrix[radiomics_idx].copy())

        if self.uni_matrix is not None:
            uni_idx = self.uni_barcode_to_idx[barcode]
            uni_emb = torch.from_numpy(self.uni_matrix[uni_idx].copy())
        else:
            uni_emb = torch.zeros(1024)

        return {
            "idx":       idx,
            "barcode":   barcode,
            "coord":     coord,
            "patch":     torch.empty(0),
            "st":        st,
            "radiomics": radiomics,
            "uni_emb":   uni_emb,
        }

    _PersampleDataset.__getitem__ = _getitem_no_patch


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


@torch.no_grad()
def evaluate(
    head:     MLPGeneHead,
    backbone: SummaryTableModel,
    loader:   DataLoader,
    device:   torch.device,
) -> tuple[float, float, np.ndarray]:
    """Evaluate using InductiveBatchSampler loader.

    backbone.encode receives the full batch (anchor + neighbors) so row attention
    sees spatial context — same as training.  Only the anchor (index 0) is
    collected for PCC, so each spot is evaluated exactly once.
    """
    head.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        rad  = batch["radiomics"].to(device)   # (B, rad_features)
        st   = batch["st"][:1].to(device)       # anchor only
        feat = backbone.encode(rad)             # row attention sees full batch
        pred = head(feat[:1])
        all_preds.append(pred.cpu())
        all_targets.append(st.cpu())

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse             = F.mse_loss(preds, targets).item()
    mean_pcc, per_g = compute_genewise_pcc(preds, targets)
    return mse, mean_pcc, per_g


# ── LOOCV fold ────────────────────────────────────────────────────────────────

def run_fold(
    fold:       int,
    val_id:     str,
    train_ids:  list[str],
    gene_names: list[str],
    backbone:   SummaryTableModel,
    cfg:        Config,
    device:     torch.device,
) -> tuple[float, np.ndarray]:
    print(f"\n{'─' * 60}")
    print(f"Fold {fold}  val={val_id}  train={train_ids}")

    train_dataset = HestRadiomicsDataset(
        dataroot=EVAL_DATA_ROOT, sample_ids=train_ids, gene_names=gene_names,
    )
    val_dataset = HestRadiomicsDataset(
        dataroot=EVAL_DATA_ROOT, sample_ids=[val_id], gene_names=gene_names,
    )
    train_loader = build_loader(
        train_dataset, batch_size=BATCH_SIZE, n_neighbors=N_NEIGHBORS,
        num_workers=NUM_WORKERS, shuffle=True,
    )
    val_loader = build_loader(
        val_dataset, batch_size=BATCH_SIZE, n_neighbors=N_NEIGHBORS,
        num_workers=NUM_WORKERS, shuffle=False,
    )
    print(f"  train spots: {len(train_dataset)}  val spots: {len(val_dataset)}")

    head = MLPGeneHead(
        in_dim=cfg.hidden_dim,
        hidden_dim=HEAD_HIDDEN_DIM,
        out_dim=len(gene_names),
        dropout=HEAD_DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"  MLPGeneHead: {n_params:,} params")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    best_pcc      = -1.0
    best_epoch    = -1
    best_per_gene = None
    bad_epochs    = 0

    for epoch in range(HEAD_EPOCHS):
        print(f"Epoch {epoch:3d}/{HEAD_EPOCHS - 1} ... [{time.strftime('%H:%M:%S')}]")
        train_loader.batch_sampler.set_epoch(epoch)
        head.train()
        total_loss, n_seen = 0.0, 0

        for batch in train_loader:
            rad = batch["radiomics"].to(device)
            st  = batch["st"].to(device)

            with torch.no_grad():
                feat = backbone.encode(rad)   # row attention sees anchor + neighbors

            pred = head(feat)
            loss = criterion(pred, st)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * rad.size(0)
            n_seen     += rad.size(0)

        train_mse = total_loss / n_seen
        val_mse, val_pcc, per_gene = evaluate(head, backbone, val_loader, device)

        is_best = val_pcc > best_pcc
        if is_best:
            bad_epochs    = 0
            best_pcc      = val_pcc
            best_epoch    = epoch
            best_per_gene = per_gene
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "fold":             fold,
                    "epoch":            epoch,
                    "head":             head.state_dict(),
                    "val_genewise_pcc": val_pcc,
                    "per_gene_pcc":     per_gene,
                    "gene_names":       gene_names,
                    "val_id":           val_id,
                    "train_ids":        train_ids,
                    "ckpt_path":        str(CKPT_PATH),
                },
                SAVE_DIR / f"fold_{fold}_best.pt",
            )
        else:
            bad_epochs += 1

        if epoch % LOG_EVERY == 0 or epoch == HEAD_EPOCHS - 1 or is_best:
            marker = " *" if is_best else ""
            print(
                f"  epoch {epoch:3d} | "
                f"train_mse={train_mse:.4f} | "
                f"val_mse={val_mse:.4f} | "
                f"val_pcc={val_pcc:.4f}"
                f"{marker}"
            )

        # if bad_epochs >= PATIENCE:
        #     print(f"  early stopping at epoch {epoch}")
        #     break

    print(f"  → best PCC={best_pcc:.4f} at epoch {best_epoch}")
    return best_pcc, best_per_gene


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    _monkey_patch_dataset_getitem_without_patch()

    cfg    = Config()
    device = torch.device(cfg.device)

    # gene names from bench set
    st_paths   = [EVAL_DATA_ROOT / "st" / f"{sid}.h5ad" for sid in SAMPLE_IDS]
    gene_names = get_common_genes(st_paths, k=N_GENES, criteria=GENE_CRITERIA)

    print(f"\nCheckpoint : {CKPT_PATH}")
    print(f"Data root  : {EVAL_DATA_ROOT}")
    print(f"Samples    : {list(SAMPLE_IDS)}")
    print(f"Genes      : {len(gene_names)}")

    # backbone
    backbone, ckpt_epoch = build_backbone(cfg, device)
    print(f"Backbone   : epoch {ckpt_epoch}, frozen")
    print(f"Head       : rad({cfg.hidden_dim}) → {HEAD_HIDDEN_DIM} → {len(gene_names)}")

    # LOOCV
    sample_list   = list(SAMPLE_IDS)
    fold_pccs:    list[float]      = []
    fold_pergene: list[np.ndarray] = []

    for fold, val_id in enumerate(sample_list):
        train_ids       = [s for s in sample_list if s != val_id]
        best_pcc, per_g = run_fold(
            fold, val_id, train_ids, gene_names, backbone, cfg, device
        )
        fold_pccs.append(best_pcc)
        if per_g is not None:
            fold_pergene.append(per_g)

    # summary
    mean_pcc = float(np.mean(fold_pccs))
    std_pcc  = float(np.std(fold_pccs, ddof=1) if len(fold_pccs) > 1 else 0.0)

    print(f"\n{'=' * 60}")
    print("LOOCV Results  (tabular only)")
    print("─" * 60)
    for fold, (sid, pcc) in enumerate(zip(sample_list, fold_pccs)):
        print(f"  Fold {fold}  val={sid:10s}  PCC={pcc:.4f}")
    print("─" * 60)
    print(f"  Mean PCC : {mean_pcc:.4f}")
    print(f"  Std  PCC : {std_pcc:.4f}")

    if fold_pergene:
        mean_per = np.stack(fold_pergene).mean(axis=0)
        top5 = np.argsort(mean_per)[::-1][:5]
        bot5 = np.argsort(mean_per)[:5]
        print("\nTop-5 genes (mean across folds):")
        for i in top5:
            print(f"  {gene_names[i]:20s}  {mean_per[i]:.4f}")
        print("Bottom-5 genes (mean across folds):")
        for i in bot5:
            print(f"  {gene_names[i]:20s}  {mean_per[i]:.4f}")

    print("=" * 60)
    print(f"Per-fold checkpoints : {SAVE_DIR}/fold_{{i}}_best.pt")


if __name__ == "__main__":
    main()
