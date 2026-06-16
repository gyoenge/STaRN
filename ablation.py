"""Ablation study: UNI-only baseline vs STaRN (fusion + neighbor).

Compares two feature extraction strategies under the same prediction head.

Methods
-------
uni_only  UNI(1024)                              [no backbone, plain DataLoader]
starn     concat(STaRN(128), UNI(1024))          [pretrained backbone, plain DataLoader]

Head options
------------
--head mlp    Two-layer MLP, trained with Adam (default)
--head ridge  Ridge regression, fit on all train features at once (no epoch loop)

Gene groups in visualization
-----------------------------
[Win]  STaRN wins most        (largest positive PCC gap)
[Tie]  methods perform similarly (smallest |gap|)
[Lose] UNI-only wins          (largest negative gap)

Usage
-----
    python ablation.py --ckpt checkpoints/epoch_099.pt --head ridge
    python ablation.py --ckpt checkpoints/epoch_099.pt --head mlp --skip-eval
    python ablation.py --ckpt checkpoints/epoch_099.pt --head ridge --viz-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import Config
from dataset.loader import HestRadiomicsDataset, _PersampleDataset, get_common_genes
from eval import (
    _patch_dataset_no_img,
    build_backbone,
    MLPGeneHead,
    compute_genewise_pcc,
    run_fold,
    EVAL_DATA_ROOT,
    SAMPLE_IDS,
    N_GENES,
    GENE_CRITERIA,
    HEAD_HIDDEN_DIM,
    HEAD_DROPOUT,
    HEAD_LR,
    HEAD_WEIGHT_DECAY,
    HEAD_EPOCHS,
    NUM_WORKERS,
)
from model.tabular import SummaryTableModel

# ── method definitions ────────────────────────────────────────────────────────

METHODS = [
    {
        "name":         "uni_only",
        "mode":         "uni_only",
        "label":        "UNI-only",
        "use_backbone": False,
    },
    {
        "name":         "starn",
        "mode":         "fusion",
        "label":        "STaRN",
        "use_backbone": True,
    },
]

OUT_DIR     = Path("figures/ablation")
INFER_BATCH = 512
CMAP        = "RdBu_r"
VMIN, VMAX  = -2.0, 2.0


def _ckpt_dir(method_name: str, head: str) -> Path:
    return Path(f"checkpoints/loocv_{method_name}_{head}")


# ── feature extraction (shared by Ridge and viz) ──────────────────────────────

@torch.no_grad()
def extract_all_features(
    method:   dict,
    backbone: Optional[SummaryTableModel],
    dataset:  HestRadiomicsDataset,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (features, targets, coords) for every spot in dataset.

    Uses a plain DataLoader so features are deterministic and independent of
    batch composition.  For STaRN, the backbone's row attention sees batch-mates
    but without explicit neighbor structure — acceptable for feature extraction.

    Returns:
        feats   (N, feat_dim)   float32
        targets (N, n_genes)    float32
        coords  (N, 2)          float32
    """
    loader = DataLoader(
        dataset,
        batch_size=INFER_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )

    feats_l, targets_l, coords_l = [], [], []
    for batch in loader:
        rad     = batch["radiomics"].to(device)
        uni_emb = batch["uni_emb"].to(device)
        coord   = batch["coord"]
        gt      = batch["st"]

        if method["use_backbone"] and backbone is not None:
            z    = backbone.encode(rad, coord.to(device))
            feat = torch.cat([z, uni_emb], dim=-1)
        else:
            feat = uni_emb

        feats_l.append(feat.cpu().numpy())
        targets_l.append(gt.numpy())
        coords_l.append(coord.numpy())

    return (
        np.concatenate(feats_l,   0).astype(np.float32),
        np.concatenate(targets_l, 0).astype(np.float32),
        np.concatenate(coords_l,  0).astype(np.float32),
    )


def zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


# ── Ridge head ────────────────────────────────────────────────────────────────

class RidgeWrapper:
    """Thin torch-compatible wrapper around a fitted sklearn Ridge model.

    Stores coef (G, F) and intercept (G,) as tensors so infer_method can call
    it identically to MLPGeneHead.
    """

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, device: torch.device):
        self.coef      = torch.tensor(coef,      dtype=torch.float32, device=device)
        self.intercept = torch.tensor(intercept, dtype=torch.float32, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.coef.T + self.intercept

    def eval(self) -> "RidgeWrapper":
        return self

    def to(self, device: torch.device) -> "RidgeWrapper":
        self.coef      = self.coef.to(device)
        self.intercept = self.intercept.to(device)
        return self


def run_ridge_fold(
    fold:      int,
    val_id:    str,
    train_ids: list[str],
    gene_names: list[str],
    method:    dict,
    backbone:  Optional[SummaryTableModel],
    cfg:       Config,
    device:    torch.device,
    alpha:     float,
    save_dir:  Path,
) -> tuple[float, np.ndarray]:
    from sklearn.linear_model import Ridge

    print(f"\n{'─' * 60}")
    print(f"Fold {fold}  val={val_id}  train={train_ids}")

    train_ds = HestRadiomicsDataset(
        sources=[(EVAL_DATA_ROOT, train_ids)],
        gene_names=gene_names,
    )
    val_ds = HestRadiomicsDataset(
        sources=[(EVAL_DATA_ROOT, [val_id])],
        gene_names=gene_names,
    )
    print(f"  train spots: {len(train_ds)}  val spots: {len(val_ds)}")

    print("  extracting train features...", end=" ", flush=True)
    X_tr, y_tr, _ = extract_all_features(method, backbone, train_ds, device)
    print(f"done  {X_tr.shape}")

    print("  extracting val features...", end=" ", flush=True)
    X_val, y_val, _ = extract_all_features(method, backbone, val_ds, device)
    print(f"done  {X_val.shape}")

    print(f"  fitting Ridge(alpha={alpha})...", end=" ", flush=True)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_tr, y_tr)
    print("done")

    pred_val = model.predict(X_val).astype(np.float32)
    pred_t   = torch.from_numpy(pred_val)
    targ_t   = torch.from_numpy(y_val)
    mean_pcc, per_gene = compute_genewise_pcc(pred_t, targ_t)
    val_mse  = float(F.mse_loss(pred_t, targ_t).item())

    print(f"  val_mse={val_mse:.4f}  val_pcc={mean_pcc:.4f}")
    print(f"  → best PCC={mean_pcc:.4f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        save_dir / f"fold_{fold}_ridge.npy",
        {
            "fold":         fold,
            "val_id":       val_id,
            "train_ids":    train_ids,
            "gene_names":   gene_names,
            "coef":         model.coef_.astype(np.float32),       # (G, F)
            "intercept":    model.intercept_.astype(np.float32),  # (G,)
            "per_gene_pcc": per_gene.astype(np.float32),
            "val_genewise_pcc": float(mean_pcc),
            "alpha":        alpha,
        },
        allow_pickle=True,
    )

    return mean_pcc, per_gene


# ── MLP inference helper ──────────────────────────────────────────────────────

@torch.no_grad()
def infer_method(
    method:   dict,
    backbone: Optional[SummaryTableModel],
    head,                   # MLPGeneHead | RidgeWrapper
    dataset:  HestRadiomicsDataset,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-dataset inference; returns (coords, gt, pred) all as float32 np arrays."""
    loader = DataLoader(
        dataset,
        batch_size=INFER_BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    coords_l, gt_l, pred_l = [], [], []
    for batch in loader:
        rad     = batch["radiomics"].to(device)
        uni_emb = batch["uni_emb"].to(device)
        coord   = batch["coord"]
        gt      = batch["st"]

        if method["use_backbone"] and backbone is not None:
            z    = backbone.encode(rad, coord.to(device))
            feat = torch.cat([z, uni_emb], dim=-1)
        else:
            feat = uni_emb

        pred = head(feat)

        coords_l.append(coord.numpy())
        gt_l.append(gt.numpy())
        pred_l.append(pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred)

    return (
        np.concatenate(coords_l, 0).astype(np.float32),
        np.concatenate(gt_l,     0).astype(np.float32),
        np.concatenate(pred_l,   0).astype(np.float32),
    )


# ── gene group selection ──────────────────────────────────────────────────────

def select_gene_groups_by_gap(
    pcc_uni:  np.ndarray,
    pcc_star: np.ndarray,
    k:        int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gap = pcc_star - pcc_uni
    sorted_asc = np.argsort(gap)
    win_idx  = sorted_asc[::-1][:k]
    lose_idx = sorted_asc[:k]

    excluded  = set(win_idx.tolist()) | set(lose_idx.tolist())
    remain    = np.array([i for i in range(len(gap)) if i not in excluded])
    tie_order = remain[np.argsort(np.abs(gap[remain]))]
    tie_idx   = tie_order[:k]

    return win_idx, tie_idx, lose_idx


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
    ax.set_title(title, fontsize=7, pad=3)
    return sc


def plot_ablation_sample(
    sample_id:  str,
    coords:     np.ndarray,
    gt_z:       np.ndarray,
    uni_z:      np.ndarray,
    star_z:     np.ndarray,
    pcc_uni:    np.ndarray,
    pcc_star:   np.ndarray,
    gene_names: list[str],
    ckpt_epoch: int,
    head_type:  str,
    n_top:      int,
    save_path:  Path,
):
    win_idx, tie_idx, lose_idx = select_gene_groups_by_gap(pcc_uni, pcc_star, n_top)
    sections = [("[Win]", win_idx), ("[Tie]", tie_idx), ("[Lose]", lose_idx)]
    n_rows   = 1 + n_top * 3

    s      = float(np.clip(120_000 / max(len(coords), 1), 1.0, 50.0))
    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    col_w  = 3.5
    row_h  = col_w * (y_span / x_span) if x_span > 0 else col_w
    row_h  = float(np.clip(row_h, 2.0, 8.0))

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(col_w * 3 + 1.0, row_h * n_rows),
        constrained_layout=True,
    )

    uni_mean_pcc  = float(pcc_uni.mean())
    star_mean_pcc = float(pcc_star.mean())
    fig.suptitle(
        f"{sample_id}  |  epoch {ckpt_epoch:03d}  |  head={head_type}  |  {len(coords):,} spots\n"
        f"Mean PCC — UNI-only: {uni_mean_pcc:.4f}   STaRN: {star_mean_pcc:.4f}   "
        f"Δ={star_mean_pcc - uni_mean_pcc:+.4f}",
        fontsize=10, fontweight="bold",
    )

    for ax, arr, lbl in zip(
        axes[0],
        [gt_z.mean(1), uni_z.mean(1), star_z.mean(1)],
        [
            "GT  (mean z-score, 250 genes)",
            f"UNI-only  (mean)  PCC={uni_mean_pcc:.3f}",
            f"STaRN     (mean)  PCC={star_mean_pcc:.3f}",
        ],
    ):
        sc = _scatter(ax, coords, arr, lbl, s)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="z-score")

    row = 1
    gap = pcc_star - pcc_uni
    for sec_label, indices in sections:
        for gi in indices:
            name = gene_names[gi]
            pu, ps, dg = pcc_uni[gi], pcc_star[gi], gap[gi]
            for ax, arr, lbl in zip(
                axes[row],
                [gt_z[:, gi], uni_z[:, gi], star_z[:, gi]],
                [
                    f"{sec_label} GT  |  {name}",
                    f"{sec_label} UNI-only  PCC={pu:.3f}",
                    f"{sec_label} STaRN     PCC={ps:.3f}  (Δ={dg:+.3f})",
                ],
            ):
                sc = _scatter(ax, coords, arr, lbl, s)
                plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="z-score")
            row += 1

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


# ── comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(
    method_results: dict[str, list[float]],
    gene_names:     list[str],
    method_pergene: dict[str, list[np.ndarray]],
    head_type:      str,
) -> None:
    sample_list = list(SAMPLE_IDS)
    col_w       = 12

    print(f"\n{'=' * 70}")
    print(f"Ablation Results  —  LOOCV gene-wise PCC  [head={head_type}]")
    print("─" * 70)
    print(f"{'Sample':<12}" + "".join(f"{m['label']:>{col_w}}" for m in METHODS))
    print("─" * 70)
    for fold, sid in enumerate(sample_list):
        row = f"{sid:<12}"
        for m in METHODS:
            row += f"{method_results[m['name']][fold]:>{col_w}.4f}"
        print(row)
    print("─" * 70)

    means = {m["name"]: float(np.mean(method_results[m["name"]])) for m in METHODS}
    stds  = {m["name"]: float(np.std(method_results[m["name"]], ddof=1)) for m in METHODS}
    print(f"{'Mean':12}" + "".join(f"{means[m['name']]:>{col_w}.4f}" for m in METHODS))
    print(f"{'Std':12}"  + "".join(f"{stds[m['name']]:>{col_w}.4f}"  for m in METHODS))

    pccs_uni  = method_results["uni_only"]
    pccs_star = method_results["starn"]
    deltas    = [s - u for s, u in zip(pccs_star, pccs_uni)]
    mean_delta = float(np.mean(deltas))
    print("─" * 70)
    print(f"{'Δ(STaRN−UNI)':12}" +
          "".join(f"{d:+{col_w}.4f}" for d in deltas) +
          f"  mean={mean_delta:+.4f}")
    print(f"{'=' * 70}")

    if "uni_only" in method_pergene and "starn" in method_pergene:
        mean_uni  = np.stack(method_pergene["uni_only"]).mean(0)
        mean_star = np.stack(method_pergene["starn"]).mean(0)
        gap       = mean_star - mean_uni

        top5 = np.argsort(gap)[::-1][:5]
        bot5 = np.argsort(gap)[:5]
        print("\nTop-5 genes where STaRN gains most over UNI-only:")
        for i in top5:
            print(f"  {gene_names[i]:20s}  UNI={mean_uni[i]:.4f}  STaRN={mean_star[i]:.4f}  Δ={gap[i]:+.4f}")
        print("Top-5 genes where UNI-only outperforms STaRN:")
        for i in bot5:
            print(f"  {gene_names[i]:20s}  UNI={mean_uni[i]:.4f}  STaRN={mean_star[i]:.4f}  Δ={gap[i]:+.4f}")
        print()


# ── LOOCV runner ──────────────────────────────────────────────────────────────

def _load_fold_ckpt(
    ckpt_path: Path, head_type: str, device: torch.device
) -> tuple[float, np.ndarray, list[str]]:
    """Load PCC and gene names from a saved fold checkpoint."""
    if head_type == "ridge":
        saved = np.load(ckpt_path, allow_pickle=True).item()
    else:
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    return (
        float(saved["val_genewise_pcc"]),
        np.array(saved["per_gene_pcc"]),
        list(saved["gene_names"]),
    )


def run_ablation_eval(
    gene_names:  list[str],
    backbone:    Optional[SummaryTableModel],
    cfg:         Config,
    device:      torch.device,
    ckpt_path:   Optional[Path],
    head_type:   str,
    ridge_alpha: float,
    skip_eval:   bool,
) -> tuple[dict[str, list[float]], dict[str, list[np.ndarray]]]:
    sample_list   = list(SAMPLE_IDS)
    method_results: dict[str, list[float]]      = {m["name"]: [] for m in METHODS}
    method_pergene: dict[str, list[np.ndarray]] = {m["name"]: [] for m in METHODS}

    for m in METHODS:
        print(f"\n{'#' * 60}")
        print(f"# Method: {m['label']}  head={head_type}")
        print(f"{'#' * 60}")

        backbone_for_method = backbone if m["use_backbone"] else None
        save_dir = _ckpt_dir(m["name"], head_type)

        for fold, val_id in enumerate(sample_list):
            train_ids = [s for s in sample_list if s != val_id]

            # Check for existing checkpoint
            if head_type == "ridge":
                fold_ckpt = save_dir / f"fold_{fold}_ridge.npy"
            else:
                fold_ckpt = save_dir / f"fold_{fold}_best.pt"

            if skip_eval and fold_ckpt.exists():
                best_pcc, per_gene, _ = _load_fold_ckpt(fold_ckpt, head_type, device)
                print(f"  [skip] fold {fold} val={val_id}  PCC={best_pcc:.4f}")
            elif head_type == "ridge":
                best_pcc, per_gene = run_ridge_fold(
                    fold=fold,
                    val_id=val_id,
                    train_ids=train_ids,
                    gene_names=gene_names,
                    method=m,
                    backbone=backbone_for_method,
                    cfg=cfg,
                    device=device,
                    alpha=ridge_alpha,
                    save_dir=save_dir,
                )
            else:
                # MLP: reuse eval.py run_fold (uses InductiveBatchSampler for starn)
                use_neighbor = m["use_backbone"]   # starn=True, uni_only=False
                best_pcc, per_gene = run_fold(
                    fold=fold,
                    val_id=val_id,
                    train_ids=train_ids,
                    gene_names=gene_names,
                    backbone=backbone_for_method,
                    cfg=cfg,
                    mode=m["mode"],
                    use_neighbor=use_neighbor,
                    device=device,
                    save_dir=save_dir,
                    ckpt_path=ckpt_path if m["use_backbone"] else None,
                )

            method_results[m["name"]].append(best_pcc)
            if per_gene is not None:
                method_pergene[m["name"]].append(per_gene)

    return method_results, method_pergene


# ── visualization ─────────────────────────────────────────────────────────────

def _load_head_for_viz(
    method:    dict,
    fold:      int,
    head_type: str,
    cfg:       Config,
    gene_names: list[str],
    device:    torch.device,
) -> tuple[object, np.ndarray]:
    """Load fold checkpoint and return (head, per_gene_pcc)."""
    save_dir = _ckpt_dir(method["name"], head_type)

    if head_type == "ridge":
        ckpt_path = save_dir / f"fold_{fold}_ridge.npy"
        saved     = np.load(ckpt_path, allow_pickle=True).item()
        head      = RidgeWrapper(saved["coef"], saved["intercept"], device)
        per_gene  = np.array(saved["per_gene_pcc"])
    else:
        ckpt_path = save_dir / f"fold_{fold}_best.pt"
        saved     = torch.load(ckpt_path, map_location=device, weights_only=False)
        in_dim    = cfg.uni_dim if method["mode"] == "uni_only" else cfg.hidden_dim + cfg.uni_dim
        head      = MLPGeneHead(
            in_dim=in_dim, hidden_dim=HEAD_HIDDEN_DIM,
            out_dim=len(gene_names), dropout=0.0,
        ).to(device)
        head.load_state_dict(saved["head"])
        head.eval()
        per_gene  = np.array(saved["per_gene_pcc"])

    return head, per_gene


def run_ablation_viz(
    gene_names:  list[str],
    backbone:    Optional[SummaryTableModel],
    cfg:         Config,
    device:      torch.device,
    ckpt_epoch:  int,
    head_type:   str,
    n_top:       int,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold, val_id in enumerate(SAMPLE_IDS):
        print(f"\nFold {fold} | val={val_id}")

        method_heads: dict[str, object]      = {}
        method_pcc:   dict[str, np.ndarray]  = {}
        all_loaded = True

        for m in METHODS:
            save_dir = _ckpt_dir(m["name"], head_type)
            fold_ckpt = (
                save_dir / f"fold_{fold}_ridge.npy"
                if head_type == "ridge"
                else save_dir / f"fold_{fold}_best.pt"
            )
            if not fold_ckpt.exists():
                print(f"  [skip] missing {fold_ckpt}")
                all_loaded = False
                break

            head, per_gene           = _load_head_for_viz(m, fold, head_type, cfg, gene_names, device)
            method_heads[m["name"]]  = head
            method_pcc[m["name"]]    = per_gene

        if not all_loaded:
            continue

        dataset = HestRadiomicsDataset(
            sources=[(EVAL_DATA_ROOT, [val_id])],
            gene_names=gene_names,
        )
        print(f"  {len(dataset):,} spots")

        coords, gt_arr, pred_uni = infer_method(
            METHODS[0], None, method_heads["uni_only"], dataset, device
        )
        _, _, pred_star = infer_method(
            METHODS[1], backbone, method_heads["starn"], dataset, device
        )

        gt_z   = zscore(gt_arr)
        uni_z  = zscore(pred_uni)
        star_z = zscore(pred_star)

        save_path = OUT_DIR / f"{val_id}_ep{ckpt_epoch:03d}_{head_type}_ablation.png"
        plot_ablation_sample(
            sample_id=val_id,
            coords=coords,
            gt_z=gt_z,
            uni_z=uni_z,
            star_z=star_z,
            pcc_uni=method_pcc["uni_only"],
            pcc_star=method_pcc["starn"],
            gene_names=gene_names,
            ckpt_epoch=ckpt_epoch,
            head_type=head_type,
            n_top=n_top,
            save_path=save_path,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STaRN ablation: UNI-only vs STaRN")
    p.add_argument("--ckpt",        type=Path,  default=None,
                   help="Pretrained STaRN backbone checkpoint.")
    p.add_argument("--head",        choices=["mlp", "ridge"], default="ridge",
                   help="Prediction head: mlp (Adam-trained) or ridge (default: ridge).")
    p.add_argument("--ridge-alpha", type=float, default=1.0,
                   help="Regularisation strength for Ridge regression (default 1.0).")
    p.add_argument("--top-k",       type=int,   default=5,
                   help="Genes per section in visualization (default 5).")
    p.add_argument("--skip-eval",   action="store_true",
                   help="Skip LOOCV if per-fold checkpoints already exist.")
    p.add_argument("--viz-only",    action="store_true",
                   help="Skip LOOCV, jump straight to visualization.")
    return p.parse_args()


def main():
    args   = _parse()
    cfg    = Config()
    device = torch.device(cfg.device)

    _patch_dataset_no_img()

    st_paths   = [EVAL_DATA_ROOT / "st" / f"{sid}.h5ad" for sid in SAMPLE_IDS]
    gene_names = get_common_genes(st_paths, k=N_GENES, criteria=GENE_CRITERIA)
    print(f"Selected {len(gene_names)} common genes.")

    # Backbone
    backbone:   Optional[SummaryTableModel] = None
    ckpt_epoch: int = -1

    if args.ckpt is not None:
        backbone, ckpt_epoch = build_backbone(cfg, args.ckpt, device)
        print(f"Backbone loaded: {args.ckpt}  (epoch {ckpt_epoch})")
    else:
        # Try to recover epoch from existing starn checkpoint
        starn_dir = _ckpt_dir("starn", args.head)
        suffix = "ridge.npy" if args.head == "ridge" else "best.pt"
        for i in range(len(SAMPLE_IDS)):
            p = starn_dir / f"fold_{i}_{suffix}"
            if p.exists():
                try:
                    if args.head == "ridge":
                        saved = np.load(p, allow_pickle=True).item()
                    else:
                        saved = torch.load(p, map_location="cpu", weights_only=False)
                    ref = saved.get("backbone_ckpt")
                    if ref and Path(ref).exists():
                        backbone, ckpt_epoch = build_backbone(cfg, Path(ref), device)
                        print(f"Backbone rebuilt from {ref}  (epoch {ckpt_epoch})")
                except Exception:
                    pass
                break

    # Guard: backbone needed for STaRN training
    if not args.viz_only and not args.skip_eval and backbone is None:
        starn_dir = _ckpt_dir("starn", args.head)
        suffix    = "ridge.npy" if args.head == "ridge" else "best.pt"
        starn_ok  = all(
            (starn_dir / f"fold_{i}_{suffix}").exists()
            for i in range(len(SAMPLE_IDS))
        )
        if not starn_ok:
            raise SystemExit(
                "--ckpt is required to run STaRN evaluation.\n"
                "Use --skip-eval if STaRN checkpoints already exist."
            )

    # ── eval ────────────────────────────────────────────────────────────────
    if not args.viz_only:
        method_results, method_pergene = run_ablation_eval(
            gene_names=gene_names,
            backbone=backbone,
            cfg=cfg,
            device=device,
            ckpt_path=args.ckpt,
            head_type=args.head,
            ridge_alpha=args.ridge_alpha,
            skip_eval=args.skip_eval,
        )
        print_comparison_table(method_results, gene_names, method_pergene, args.head)

    # ── viz ─────────────────────────────────────────────────────────────────
    if backbone is None:
        print("\n[warn] No backbone available — skipping STaRN visualization.")
        return

    print(f"\nGenerating ablation visualizations → {OUT_DIR}/")
    run_ablation_viz(
        gene_names=gene_names,
        backbone=backbone,
        cfg=cfg,
        device=device,
        ckpt_epoch=ckpt_epoch,
        head_type=args.head,
        n_top=args.top_k,
    )
    print(f"\nDone — figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
