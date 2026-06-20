"""Spatial z-score visualization: predicted vs ground-truth gene expression.

For each LOOCV fold, loads the saved head checkpoint, runs inference over the
val sample's spots, z-scores both GT and predicted expression, and saves a
side-by-side spatial scatter plot (GT | Pred) for the mean expression and the
top-5 genes by val PCC.

Usage:
    python visualize.py [--ckpt CKPT] [--top-k N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import Config
from dataset.loader import HestRadiomicsDataset, _PersampleDataset
from eval import _patch_dataset_no_img, build_backbone, MLPGeneHead

# ── paths ─────────────────────────────────────────────────────────────────────
EVAL_DATA_ROOT = Path("/root/workspace/datasets/hest_radiomics/IDC_Bench_Xenium_4/IDC/")
SAMPLE_IDS     = ("NCBI783", "NCBI785", "TENX95", "TENX99")
LOOCV_DIR      = Path("checkpoints/loocv_fusion_neighbor")
OUT_DIR        = Path("figures/spatial_viz")

BATCH_SIZE = 512
CMAP       = "RdBu_r"
VMIN, VMAX = -2.0, 2.0


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(
    backbone, head, dataset, device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    coords_l, gt_l, pred_l = [], [], []
    for batch in loader:
        rad   = batch["radiomics"].to(device)
        uni   = batch["uni_emb"].to(device)
        coord = batch["coord"]
        gt    = batch["st"]

        z    = backbone.encode(rad, coord.to(device))   # (B, hidden_dim)
        feat = torch.cat([z, uni], dim=-1)               # (B, hidden+uni)
        pred = head(feat)                                 # (B, G)

        coords_l.append(coord.numpy())
        gt_l.append(gt.numpy())
        pred_l.append(pred.cpu().numpy())

    return (
        np.concatenate(coords_l, 0).astype(np.float32),  # (N, 2)
        np.concatenate(gt_l,     0).astype(np.float32),  # (N, G)
        np.concatenate(pred_l,   0).astype(np.float32),  # (N, G)
    )


def zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


# ── plotting ──────────────────────────────────────────────────────────────────

def _scatter(ax, coords, vals, title, s):
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=vals,
        s=s, cmap=CMAP, vmin=VMIN, vmax=VMAX,
        linewidths=0, rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8, pad=3)
    return sc


def _select_gene_groups(
    per_gene_pcc: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices for top-k, mid-k (closest to median), and bot-k genes."""
    sorted_asc = np.argsort(per_gene_pcc)
    top_idx = sorted_asc[::-1][:k]
    bot_idx = sorted_asc[:k]

    median_pcc = np.median(per_gene_pcc)
    dist_to_mid = np.abs(per_gene_pcc - median_pcc)
    # Exclude already-selected top/bot indices from mid candidates
    excluded = set(top_idx.tolist()) | set(bot_idx.tolist())
    candidate_mask = np.array([i not in excluded for i in range(len(per_gene_pcc))])
    candidate_idx  = np.where(candidate_mask)[0]
    mid_order = candidate_idx[np.argsort(dist_to_mid[candidate_idx])]
    mid_idx = mid_order[:k]

    return top_idx, mid_idx, bot_idx


def plot_sample(
    sample_id:    str,
    coords:       np.ndarray,
    gt_z:         np.ndarray,
    pr_z:         np.ndarray,
    gene_names:   list[str],
    per_gene_pcc: np.ndarray,
    ckpt_epoch:   int,
    n_top:        int,
    save_path:    Path,
):
    top_idx, mid_idx, bot_idx = _select_gene_groups(per_gene_pcc, n_top)

    # Section layout: [mean] + [top-k] + [mid-k] + [bot-k]
    sections = [
        ("Top", top_idx),
        ("Mid", mid_idx),
        ("Bot", bot_idx),
    ]
    n_rows = 1 + n_top * 3

    # Dot size: inversely proportional to spot count, clamped
    s = float(np.clip(120_000 / max(len(coords), 1), 1.0, 50.0))

    # Figure height: scale subplot height by tissue aspect ratio
    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    col_w  = 4.2   # inches per column
    row_h  = col_w * (y_span / x_span) if x_span > 0 else col_w
    row_h  = float(np.clip(row_h, 2.5, 9.0))

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(col_w * 2 + 0.8, row_h * n_rows),
        constrained_layout=True,
    )
    fig.suptitle(
        f"{sample_id}  |  epoch {ckpt_epoch:03d}  |  LOOCV val  |  {len(coords):,} spots",
        fontsize=11, fontweight="bold",
    )

    # Row 0: mean z-score across all genes
    for ax, arr, lbl in zip(
        axes[0],
        [gt_z.mean(1), pr_z.mean(1)],
        ["GT  (mean z-score, 250 genes)", "Pred  (mean z-score, 250 genes)"],
    ):
        sc = _scatter(ax, coords, arr, lbl, s)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="z-score")

    # Rows 1+: top / mid / bot gene sections
    row = 1
    for section_label, indices in sections:
        for gi in indices:
            pcc  = per_gene_pcc[gi]
            name = gene_names[gi]
            for ax, arr, lbl in zip(
                axes[row],
                [gt_z[:, gi], pr_z[:, gi]],
                [
                    f"[{section_label}] GT  |  {name}",
                    f"[{section_label}] Pred  |  {name}  (PCC={pcc:.3f})",
                ],
            ):
                sc = _scatter(ax, coords, arr, lbl, s)
                plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="z-score")
            row += 1

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  type=Path,  default=Path("checkpoints/epoch_015.pt"))
    p.add_argument("--top-k", type=int,   default=5)
    return p.parse_args()


def main():
    args   = _parse()
    cfg    = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _patch_dataset_no_img()

    backbone, ckpt_epoch = build_backbone(cfg, args.ckpt, device)
    print(f"Backbone: {args.ckpt}  (epoch {ckpt_epoch})\n")

    for fold, val_id in enumerate(SAMPLE_IDS):
        fold_ckpt_path = LOOCV_DIR / f"fold_{fold}_best.pt"
        if not fold_ckpt_path.exists():
            print(f"[skip] {fold_ckpt_path} not found")
            continue

        print(f"Fold {fold} | val={val_id}")
        fold_ckpt    = torch.load(fold_ckpt_path, map_location=device, weights_only=False)
        gene_names   = fold_ckpt["gene_names"]
        per_gene_pcc = fold_ckpt["per_gene_pcc"]

        head = MLPGeneHead(
            in_dim=cfg.hidden_dim + cfg.uni_dim,
            hidden_dim=256,
            out_dim=len(gene_names),
            dropout=0.0,
        ).to(device)
        head.load_state_dict(fold_ckpt["head"])
        head.eval()

        dataset = HestRadiomicsDataset(
            sources=[(EVAL_DATA_ROOT, [val_id])],
            gene_names=gene_names,
        )
        print(f"  {len(dataset):,} spots")

        coords, gt_arr, pred_arr = infer(backbone, head, dataset, device)
        gt_z   = zscore(gt_arr)
        pred_z = zscore(pred_arr)

        save_path = OUT_DIR / f"{val_id}_ep{ckpt_epoch:03d}.png"
        plot_sample(
            sample_id=val_id,
            coords=coords,
            gt_z=gt_z,
            pr_z=pred_z,
            gene_names=gene_names,
            per_gene_pcc=per_gene_pcc,
            ckpt_epoch=ckpt_epoch,
            n_top=args.top_k,
            save_path=save_path,
        )

    print(f"\nDone — figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
