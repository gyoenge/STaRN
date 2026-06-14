"""STaRN evaluation — LOOCV gene expression prediction.

Usage:
    cd /root/workspace/STaRN
    python eval.py [--ckpt CKPT] [--mode MODE] [--no-neighbor]

Modes:
    fusion    (default)  concat(Tab, UNI) → MLP         — design inference pipeline
    tab_only             Tab backbone only → MLP          — ablation: Tab contribution
    uni_only             UNI embeddings only → MLP        — baseline (no backbone needed)

Flags:
    --no-neighbor   Use plain random batches instead of InductiveBatchSampler.
                    Row attention sees no spatial context (ablation ②).

Inference (fusion mode):
    rad  → backbone.encode(rad, coords)  → z_i  (B, hidden_dim)
    uni  → pre-extracted ViT-L emb       → u_i  (B, uni_dim)
    concat(z_i[:1], u_i[:1])             → MLP  → gene expression
                ↑ anchor only collected during eval; full batch feeds row attention.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import Config
from dataset.loader import (
    HestRadiomicsDataset,
    _PersampleDataset,
    build_loader,
    get_common_genes,
)
from model.tabular import SummaryTableModel

# ── evaluation constants ──────────────────────────────────────────────────────

EVAL_DATA_ROOT: Path = Path(
    "/root/workspace/datasets/hest_radiomics/IDC_Bench_Xenium_4/IDC/"
)
SAMPLE_IDS: tuple[str, ...] = ("NCBI783", "NCBI785", "TENX95", "TENX99")

N_GENES       = 250
GENE_CRITERIA = "var"

HEAD_EPOCHS       = 10
HEAD_LR           = 1e-4
HEAD_WEIGHT_DECAY = 1e-4
HEAD_HIDDEN_DIM   = 256
HEAD_DROPOUT      = 0.1

BATCH_SIZE  = 32
N_NEIGHBORS = 6
N_SEMANTIC  = 6
NUM_WORKERS = 4
LOG_EVERY   = 1


# ── dataset helper — skip H5 patch loading ───────────────────────────────────

def _patch_dataset_no_img() -> None:
    """Monkey-patch _PersampleDataset to skip patch-image H5 I/O during eval.

    The patch image is never used in eval (UNI embeddings are pre-extracted).
    Skipping the H5 random read removes the dominant DataLoader bottleneck.
    """
    def _getitem(self, idx: int) -> dict:
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

    _PersampleDataset.__getitem__ = _getitem


# ── MLP gene head ─────────────────────────────────────────────────────────────

class MLPGeneHead(nn.Module):
    """Two-layer MLP for gene expression prediction.

    Args:
        in_dim:     Input feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim:    Number of genes (output).
        dropout:    Dropout probability.
    """

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
    """Compute gene-wise Pearson Correlation Coefficient.

    Returns:
        (mean_pcc, per_gene_pcc array of shape (n_genes,))
    """
    pred   = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    pc     = pred   - pred.mean(dim=0, keepdim=True)
    tc     = target - target.mean(dim=0, keepdim=True)
    denom  = torch.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0)) + eps
    per    = (pc * tc).sum(0) / denom
    return per.mean().item(), per.numpy()


# ── backbone ──────────────────────────────────────────────────────────────────

def build_backbone(
    cfg:       Config,
    ckpt_path: Path,
    device:    torch.device,
) -> tuple[SummaryTableModel, int]:
    """Load pretrained SummaryTableModel and freeze all parameters."""
    model = SummaryTableModel(
        num_features=cfg.num_features,
        hidden_dim=cfg.hidden_dim,
        num_col_layers=cfg.num_col_layers,
        num_row_layers=cfg.num_row_layers,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        proj_dim=cfg.proj_dim,
        dropout=0.0,
        n_pos_bins=cfg.n_pos_bins,
        device=str(device),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, ckpt["epoch"]


# ── loaders ───────────────────────────────────────────────────────────────────

def make_loader(
    sample_ids:   list[str],
    gene_names:   list[str],
    cfg:          Config,
    use_neighbor: bool,
    shuffle:      bool,
) -> DataLoader:
    """Build a DataLoader for the given sample set.

    Args:
        use_neighbor: If True, use InductiveBatchSampler (spatial + semantic
                      neighbours); rows in each batch provide neighbour context
                      to the backbone's row attention.
                      If False, use a plain shuffled DataLoader (ablation ②).
    """
    dataset = HestRadiomicsDataset(
        dataroot=EVAL_DATA_ROOT,
        sample_ids=sample_ids,
        gene_names=gene_names,
    )
    if use_neighbor:
        return build_loader(
            dataset,
            batch_size=BATCH_SIZE,
            n_neighbors=N_NEIGHBORS,
            n_semantic=N_SEMANTIC,
            num_workers=NUM_WORKERS,
            shuffle=shuffle,
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


# ── feature extraction ────────────────────────────────────────────────────────

def _extract_features(
    batch:       dict,
    backbone:    Optional[SummaryTableModel],
    mode:        str,
    device:      torch.device,
    use_neighbor: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (features, st, mask) from a batch.

    Returns:
        feat:   (N, feat_dim) — features for gene head input
        st:     (N, n_genes)  — gene expression targets
        n_eval: number of spots to collect predictions for:
                  - use_neighbor: 1 (anchor only, index 0)
                  - plain loader: full batch B
    """
    rad     = batch["radiomics"].to(device)
    uni_emb = batch["uni_emb"].to(device)
    st      = batch["st"].to(device)
    coords  = batch["coord"].to(device)

    if mode == "uni_only":
        return uni_emb, st, rad.size(0)

    # backbone.encode receives full batch so row attention sees neighbour context
    with torch.no_grad():
        z = backbone.encode(rad, coords)   # (B, hidden_dim)

    if mode == "tab_only":
        feat = z
    else:  # fusion
        feat = torch.cat([z, uni_emb], dim=-1)

    return feat, st, rad.size(0)


# ── evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    head:         MLPGeneHead,
    loader:       DataLoader,
    backbone:     Optional[SummaryTableModel],
    mode:         str,
    device:       torch.device,
    use_neighbor: bool,
) -> tuple[float, float, np.ndarray]:
    """Run evaluation loop.

    When use_neighbor=True: only the anchor (index 0) in each batch is collected
    so each spot is evaluated exactly once, with its spatial/semantic neighbours
    providing row-attention context to the backbone.

    When use_neighbor=False: the full batch is collected.
    """
    head.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        feat, st, _ = _extract_features(batch, backbone, mode, device, use_neighbor)

        pred = head(feat)

        if use_neighbor and mode != "uni_only":
            # anchor only — row attention already consumed the full batch
            pred = pred[:1]
            st   = st[:1]

        all_preds.append(pred.cpu())
        all_targets.append(st.cpu())

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse             = F.mse_loss(preds, targets).item()
    mean_pcc, per_g = compute_genewise_pcc(preds, targets)
    return mse, mean_pcc, per_g


# ── LOOCV fold ────────────────────────────────────────────────────────────────

def run_fold(
    fold:         int,
    val_id:       str,
    train_ids:    list[str],
    gene_names:   list[str],
    backbone:     Optional[SummaryTableModel],
    cfg:          Config,
    mode:         str,
    use_neighbor: bool,
    device:       torch.device,
    save_dir:     Path,
    ckpt_path:    Optional[Path],
) -> tuple[float, np.ndarray]:
    print(f"\n{'─' * 60}")
    print(f"Fold {fold}  val={val_id}  train={train_ids}")

    train_loader = make_loader(train_ids, gene_names, cfg, use_neighbor, shuffle=True)
    val_loader   = make_loader([val_id],  gene_names, cfg, use_neighbor, shuffle=False)
    print(f"  train spots: {len(train_loader.dataset)}  val spots: {len(val_loader.dataset)}")

    # Input dimension depends on mode
    if mode == "uni_only":
        in_dim = cfg.uni_dim
    elif mode == "tab_only":
        in_dim = cfg.hidden_dim
    else:  # fusion
        in_dim = cfg.hidden_dim + cfg.uni_dim

    head = MLPGeneHead(
        in_dim=in_dim,
        hidden_dim=HEAD_HIDDEN_DIM,
        out_dim=len(gene_names),
        dropout=HEAD_DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"  MLPGeneHead: {n_params:,} params  in_dim={in_dim}")

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    best_pcc      = -1.0
    best_epoch    = -1
    best_per_gene = None

    for epoch in range(HEAD_EPOCHS):
        if hasattr(train_loader, "batch_sampler") and hasattr(
            train_loader.batch_sampler, "set_epoch"
        ):
            train_loader.batch_sampler.set_epoch(epoch)

        head.train()
        total_loss, n_seen = 0.0, 0

        for batch in train_loader:
            feat, st, _ = _extract_features(batch, backbone, mode, device, use_neighbor)

            pred = head(feat)
            loss = criterion(pred, st)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feat.size(0)
            n_seen     += feat.size(0)

        train_mse = total_loss / n_seen
        val_mse, val_pcc, per_gene = evaluate(
            head, val_loader, backbone, mode, device, use_neighbor
        )

        is_best = val_pcc > best_pcc
        if is_best:
            best_pcc      = val_pcc
            best_epoch    = epoch
            best_per_gene = per_gene
            save_dir.mkdir(parents=True, exist_ok=True)
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
                    "mode":             mode,
                    "use_neighbor":     use_neighbor,
                    "backbone_ckpt":    str(ckpt_path) if ckpt_path else None,
                },
                save_dir / f"fold_{fold}_best.pt",
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

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STaRN LOOCV evaluation")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to pretrained checkpoint (required for fusion / tab_only modes).",
    )
    p.add_argument(
        "--mode",
        choices=["fusion", "tab_only", "uni_only"],
        default="fusion",
        help=(
            "fusion: concat(Tab, UNI) → MLP  [default]\n"
            "tab_only: Tab backbone only → MLP\n"
            "uni_only: UNI embeddings only → MLP  (no ckpt needed)"
        ),
    )
    p.add_argument(
        "--no-neighbor",
        action="store_true",
        default=False,
        help="Use plain random batches — disables neighbourhood context (ablation ②).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    cfg    = Config()
    device = torch.device(cfg.device)

    if args.mode != "uni_only" and args.ckpt is None:
        raise SystemExit(
            f"--ckpt is required for mode '{args.mode}'.\n"
            "Example: python eval.py --ckpt checkpoints/epoch_099.pt"
        )

    use_neighbor = not args.no_neighbor
    neighbor_tag = "neighbor" if use_neighbor else "no_neighbor"
    save_dir     = Path(f"checkpoints/loocv_{args.mode}_{neighbor_tag}")

    _patch_dataset_no_img()

    # ── gene names ──────────────────────────────────────────────────────────
    st_paths   = [EVAL_DATA_ROOT / "st" / f"{sid}.h5ad" for sid in SAMPLE_IDS]
    gene_names = get_common_genes(st_paths, k=N_GENES, criteria=GENE_CRITERIA)

    # ── backbone ─────────────────────────────────────────────────────────────
    backbone:    Optional[SummaryTableModel] = None
    ckpt_epoch:  int = -1

    if args.mode != "uni_only":
        backbone, ckpt_epoch = build_backbone(cfg, args.ckpt, device)
        n_params = sum(p.numel() for p in backbone.parameters())

    if args.mode == "fusion":
        in_dim = cfg.hidden_dim + cfg.uni_dim
        in_desc = f"concat(Tab({cfg.hidden_dim}), UNI({cfg.uni_dim}))"
    elif args.mode == "tab_only":
        in_dim = cfg.hidden_dim
        in_desc = f"Tab({cfg.hidden_dim})"
    else:
        in_dim = cfg.uni_dim
        in_desc = f"UNI({cfg.uni_dim})"

    print(f"\n{'=' * 60}")
    print(f"Mode        : {args.mode}")
    print(f"Neighbour   : {'on (spatial + semantic)' if use_neighbor else 'off (ablation ②)'}")
    if args.ckpt:
        print(f"Checkpoint  : {args.ckpt}  (epoch {ckpt_epoch})")
    print(f"Data root   : {EVAL_DATA_ROOT}")
    print(f"Samples     : {list(SAMPLE_IDS)}")
    print(f"Genes       : {len(gene_names)}")
    print(f"Head input  : {in_desc} → MLP({HEAD_HIDDEN_DIM}) → {len(gene_names)}")
    print(f"Save dir    : {save_dir}")
    print(f"{'=' * 60}")

    # ── LOOCV ────────────────────────────────────────────────────────────────
    sample_list   = list(SAMPLE_IDS)
    fold_pccs:    list[float]       = []
    fold_pergene: list[np.ndarray]  = []

    for fold, val_id in enumerate(sample_list):
        train_ids       = [s for s in sample_list if s != val_id]
        best_pcc, per_g = run_fold(
            fold=fold,
            val_id=val_id,
            train_ids=train_ids,
            gene_names=gene_names,
            backbone=backbone,
            cfg=cfg,
            mode=args.mode,
            use_neighbor=use_neighbor,
            device=device,
            save_dir=save_dir,
            ckpt_path=args.ckpt,
        )
        fold_pccs.append(best_pcc)
        if per_g is not None:
            fold_pergene.append(per_g)

    # ── summary ──────────────────────────────────────────────────────────────
    mean_pcc = float(np.mean(fold_pccs))
    std_pcc  = float(
        np.std(fold_pccs, ddof=1) if len(fold_pccs) > 1 else 0.0
    )

    print(f"\n{'=' * 60}")
    print(f"LOOCV Results  [{args.mode} / {neighbor_tag}]")
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

    print(f"\n{'=' * 60}")
    print(f"Per-fold checkpoints : {save_dir}/fold_{{i}}_best.pt")


if __name__ == "__main__":
    main()
