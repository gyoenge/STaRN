"""Ablation study: UNI-only vs STaRN (random batch) vs STaRN (neighbor batch).

Isolates two contributions of STaRN:
  (1) backbone feature quality    — compare UNI-only vs STaRN(random)
  (2) neighbor context in row attention — compare STaRN(random) vs STaRN(neighbor)

Methods
-------
uni_only          UNI(1024)                        plain DataLoader
starn_random      concat(STaRN(128), UNI(1024))    plain DataLoader  (random batch-mates)
starn_neighbor    concat(STaRN(128), UNI(1024))    InductiveBatchSampler, anchor-only

Head
----
--head ridge  (default)  Ridge regression — fit on all train features at once
--head mlp               MLP(256), Adam

Gene groups in visualization (4 columns: GT | UNI | STaRN-rnd | STaRN-nbr)
-----------------------------
[Win]   STaRN(neighbor) gains most over UNI-only
[Tie]   smallest |gap| between STaRN(neighbor) and UNI-only
[Lose]  UNI-only outperforms STaRN(neighbor)

Usage
-----
    python ablation.py --ckpt checkpoints/epoch_099.pt --head ridge
    python ablation.py --ckpt checkpoints/epoch_099.pt --head ridge --skip-eval
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
from eval import (
    _patch_dataset_no_img,
    build_backbone,
    MLPGeneHead,
    compute_genewise_pcc,
    run_fold,
    make_loader,
    _extract_features,
    evaluate,
    EVAL_DATA_ROOT,
    SAMPLE_IDS,
    N_GENES,
    GENE_CRITERIA,
    HEAD_HIDDEN_DIM,
    HEAD_DROPOUT,
    HEAD_WEIGHT_DECAY,
    NUM_WORKERS,
    BATCH_SIZE,
    N_NEIGHBORS,
    N_SEMANTIC,
)
from model.tabular import SummaryTableModel

# ── method registry ───────────────────────────────────────────────────────────

METHODS = [
    {
        "name":           "uni_only",
        "label":          "UNI-only",
        "use_backbone":   False,
        "neighbor_batch": False,
    },
    {
        "name":           "starn_random",
        "label":          "STaRN (random)",
        "use_backbone":   True,
        "neighbor_batch": False,
    },
    {
        "name":           "starn_neighbor",
        "label":          "STaRN (neighbor)",
        "use_backbone":   True,
        "neighbor_batch": True,
    },
]

OUT_DIR     = Path("figures/ablation")
INFER_BATCH = 512
CMAP        = "RdBu_r"
VMIN, VMAX  = -2.0, 2.0


def _ckpt_dir(method_name: str, head: str, n_genes: int = 250) -> Path:
    suffix = f"_hvg{n_genes}" if n_genes != 250 else ""
    return Path(f"checkpoints/loocv_{method_name}_{head}{suffix}")


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features_plain(
    method:   dict,
    backbone: Optional[SummaryTableModel],
    dataset:  HestRadiomicsDataset,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for all spots via plain DataLoader (no neighbor context).

    Row attention in the backbone sees batch-mates grouped by index order only.
    Returns (feats, targets, coords) each float32.
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

        if method["use_backbone"] and backbone is not None:
            z    = backbone.encode(rad, coord.to(device))
            feat = torch.cat([z, uni_emb], dim=-1)
        else:
            feat = uni_emb

        feats_l.append(feat.cpu().numpy())
        targets_l.append(batch["st"].numpy())
        coords_l.append(coord.numpy())

    return (
        np.concatenate(feats_l,   0).astype(np.float32),
        np.concatenate(targets_l, 0).astype(np.float32),
        np.concatenate(coords_l,  0).astype(np.float32),
    )


@torch.no_grad()
def extract_features_neighbor(
    backbone: SummaryTableModel,
    dataset:  HestRadiomicsDataset,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract neighbor-aware anchor features via InductiveBatchSampler (no globals).

    Feature per anchor:
        concat(z_anchor, mean(z_spatial[N_NEIGHBORS]), mean(z_semantic[N_SEMANTIC]), uni_anchor)
    dim: (2 + int(N_SEMANTIC>0)) * hidden_dim + uni_dim  =  1408

    Returns (feats, targets, coords) each float32, one row per spot.
    """
    loader = build_loader(
        dataset,
        batch_size=1 + N_NEIGHBORS + N_SEMANTIC,   # no random globals at inference
        n_neighbors=N_NEIGHBORS,
        n_semantic=N_SEMANTIC,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    feats_l, targets_l, coords_l = [], [], []
    for batch in loader:
        rad     = batch["radiomics"].to(device)
        uni_emb = batch["uni_emb"].to(device)
        coord   = batch["coord"]

        z = backbone.encode(rad, coord.to(device))             # (B, D)

        z_anchor  = z[0:1]                                     # (1, D)
        z_spatial = z[1:1+N_NEIGHBORS].mean(0, keepdim=True)  # (1, D)
        parts = [z_anchor, z_spatial]
        if N_SEMANTIC > 0:
            z_semantic = z[1+N_NEIGHBORS:1+N_NEIGHBORS+N_SEMANTIC].mean(0, keepdim=True)
            parts.append(z_semantic)
        parts.append(uni_emb[0:1])
        feat = torch.cat(parts, dim=-1)                        # (1, feat_dim)

        feats_l.append(feat.cpu().numpy())
        targets_l.append(batch["st"][0:1].numpy())
        coords_l.append(coord[0:1].numpy())

    return (
        np.concatenate(feats_l,   0).astype(np.float32),
        np.concatenate(targets_l, 0).astype(np.float32),
        np.concatenate(coords_l,  0).astype(np.float32),
    )


def dispatch_extract(
    method:   dict,
    backbone: Optional[SummaryTableModel],
    dataset:  HestRadiomicsDataset,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method["neighbor_batch"]:
        return extract_features_neighbor(backbone, dataset, device)
    return extract_features_plain(method, backbone, dataset, device)


def zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


# ── Ridge head ────────────────────────────────────────────────────────────────

class RidgeWrapper:
    """Torch-compatible wrapper around a fitted Ridge model for use in infer_method.

    Optionally holds a fitted sklearn Pipeline (StandardScaler + PCA) that is
    applied before the linear forward pass — matching the HEST-bench protocol.
    """

    def __init__(
        self,
        coef:      np.ndarray,
        intercept: np.ndarray,
        device:    torch.device,
        pipeline=None,   # fitted sklearn Pipeline or None
    ):
        self.coef      = torch.tensor(coef,      dtype=torch.float32, device=device)
        self.intercept = torch.tensor(intercept, dtype=torch.float32, device=device)
        self.pipeline  = pipeline

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.pipeline is not None:
            x = torch.tensor(
                self.pipeline.transform(x.cpu().numpy()),
                dtype=torch.float32, device=self.coef.device,
            )
        return x @ self.coef.T + self.intercept

    def eval(self) -> "RidgeWrapper":
        return self

    def to(self, device: torch.device) -> "RidgeWrapper":
        self.coef      = self.coef.to(device)
        self.intercept = self.intercept.to(device)
        return self


# ── Ridge LOOCV fold ──────────────────────────────────────────────────────────

HEST_PCA_DIM = 256   # HEST-bench default latent_dim

# ── MLP hyperparameter search ─────────────────────────────────────────────────

MLP_LR_GRID    = [1e-3, 3e-4, 1e-4, 3e-5]
MLP_MAX_EPOCHS = 50
MLP_PATIENCE   = 10


def run_mlp_fold_hparam(
    fold:       int,
    val_id:     str,
    train_ids:  list[str],
    gene_names: list[str],
    method:     dict,
    backbone:   Optional[SummaryTableModel],
    cfg:        Config,
    device:     torch.device,
    save_dir:   Path,
    ckpt_path:  Optional[Path],
    n_genes:    int = 250,
) -> tuple[float, np.ndarray]:
    """MLP fold training with LR grid search + early stopping.

    For each lr in MLP_LR_GRID, trains for up to MLP_MAX_EPOCHS with
    patience-based early stopping.  Saves the best (lr, epoch) checkpoint.
    """
    from copy import deepcopy

    mode         = "uni_only" if not method["use_backbone"] else "fusion"
    use_neighbor = method["neighbor_batch"]

    if not method["use_backbone"]:
        in_dim = cfg.uni_dim
    elif method["neighbor_batch"]:
        in_dim = (2 + (1 if N_SEMANTIC > 0 else 0)) * cfg.hidden_dim + cfg.uni_dim
    else:
        in_dim = cfg.hidden_dim + cfg.uni_dim

    print(f"\n{'─' * 60}")
    print(f"Fold {fold}  val={val_id}  train={train_ids}")

    train_loader = make_loader(train_ids, gene_names, cfg, use_neighbor, shuffle=True)
    val_loader   = make_loader([val_id],  gene_names, cfg, use_neighbor, shuffle=False)
    print(f"  train spots: {len(train_loader.dataset)}  val spots: {len(val_loader.dataset)}")
    print(f"  in_dim={in_dim}  LR grid={MLP_LR_GRID}  max_epochs={MLP_MAX_EPOCHS}  patience={MLP_PATIENCE}")

    criterion = nn.MSELoss()

    best_global_pcc      = -1.0
    best_global_per_gene = None
    best_lr_found        = None
    best_epoch_found     = -1
    best_state_dict      = None

    for lr in MLP_LR_GRID:
        head = MLPGeneHead(
            in_dim=in_dim, hidden_dim=HEAD_HIDDEN_DIM,
            out_dim=len(gene_names), dropout=HEAD_DROPOUT,
        ).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=lr, weight_decay=HEAD_WEIGHT_DECAY
        )

        best_pcc_lr      = -1.0
        best_state_lr    = None
        best_epoch_lr    = -1
        best_per_gene_lr = None
        no_improve       = 0

        for epoch in range(MLP_MAX_EPOCHS):
            if hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler, "set_epoch"
            ):
                train_loader.batch_sampler.set_epoch(epoch)

            head.train()
            for batch in train_loader:
                feat, st, _ = _extract_features(batch, backbone, mode, device, use_neighbor)
                loss = criterion(head(feat), st)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, val_pcc, per_gene = evaluate(head, val_loader, backbone, mode, device, use_neighbor)

            if val_pcc > best_pcc_lr:
                best_pcc_lr      = val_pcc
                best_state_lr    = deepcopy(head.state_dict())
                best_epoch_lr    = epoch
                best_per_gene_lr = per_gene
                no_improve       = 0
            else:
                no_improve += 1
                if no_improve >= MLP_PATIENCE:
                    break

        print(f"  lr={lr:.0e}  best_pcc={best_pcc_lr:.4f}  best_epoch={best_epoch_lr}  "
              f"(stopped at epoch {epoch})")

        if best_pcc_lr > best_global_pcc:
            best_global_pcc      = best_pcc_lr
            best_global_per_gene = best_per_gene_lr
            best_lr_found        = lr
            best_epoch_found     = best_epoch_lr
            best_state_dict      = best_state_lr

    print(f"  → best PCC={best_global_pcc:.4f}  lr={best_lr_found:.0e}  epoch={best_epoch_found}")

    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold":             fold,
            "epoch":            best_epoch_found,
            "lr":               best_lr_found,
            "head":             best_state_dict,
            "val_genewise_pcc": best_global_pcc,
            "per_gene_pcc":     best_global_per_gene,
            "gene_names":       gene_names,
            "val_id":           val_id,
            "train_ids":        train_ids,
            "mode":             mode,
            "use_neighbor":     use_neighbor,
            "backbone_ckpt":    str(ckpt_path) if ckpt_path else None,
        },
        save_dir / f"fold_{fold}_best.pt",
    )
    return best_global_pcc, best_global_per_gene


def run_ridge_fold(
    fold:       int,
    val_id:     str,
    train_ids:  list[str],
    gene_names: list[str],
    method:     dict,
    backbone:   Optional[SummaryTableModel],
    device:     torch.device,
    alpha:      float,   # kept for API compat; overridden by HEST adaptive formula
    save_dir:   Path,
    n_genes:    int = 250,
) -> tuple[float, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'─' * 60}")
    print(f"Fold {fold}  val={val_id}  train={train_ids}")

    train_ds = HestRadiomicsDataset(
        sources=[(EVAL_DATA_ROOT, train_ids)], gene_names=gene_names
    )
    val_ds = HestRadiomicsDataset(
        sources=[(EVAL_DATA_ROOT, [val_id])], gene_names=gene_names
    )
    print(f"  train spots: {len(train_ds)}  val spots: {len(val_ds)}")

    print("  extracting train features...", end=" ", flush=True)
    X_tr, y_tr, _ = dispatch_extract(method, backbone, train_ds, device)
    print(f"done  {X_tr.shape}")

    print("  extracting val features...", end=" ", flush=True)
    X_val, y_val, _ = dispatch_extract(method, backbone, val_ds, device)
    print(f"done  {X_val.shape}")

    # HEST-bench preprocessing: StandardScaler → PCA(256)
    pca_dim = min(HEST_PCA_DIM, X_tr.shape[0] - 1, X_tr.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=pca_dim, random_state=0)),
    ])
    X_tr_t  = pipe.fit_transform(X_tr)
    X_val_t = pipe.transform(X_val)
    print(f"  PCA: {X_tr.shape[1]}→{pca_dim}")

    # HEST-bench adaptive alpha: 100 / (n_features * n_genes)
    hest_alpha = 100.0 / (X_tr_t.shape[1] * y_tr.shape[1])
    print(f"  Ridge alpha={hest_alpha:.6f} (HEST adaptive)  fit_intercept=False  solver=lsqr")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Ridge(
            alpha=hest_alpha,
            fit_intercept=False,
            solver="lsqr",
            random_state=0,
            max_iter=1000,
        )
        model.fit(X_tr_t, y_tr)

    pred_val    = model.predict(X_val_t).astype(np.float32)
    mean_pcc, per_gene = compute_genewise_pcc(
        torch.from_numpy(pred_val), torch.from_numpy(y_val)
    )
    val_mse = float(F.mse_loss(torch.from_numpy(pred_val), torch.from_numpy(y_val)).item())
    print(f"  val_mse={val_mse:.4f}  val_pcc={mean_pcc:.4f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        save_dir / f"fold_{fold}_ridge.npy",
        {
            "fold":             fold,
            "val_id":           val_id,
            "train_ids":        train_ids,
            "gene_names":       gene_names,
            "coef":             model.coef_.astype(np.float32),
            "intercept":        np.atleast_1d(model.intercept_).astype(np.float32),
            "pipeline":         pipe,
            "per_gene_pcc":     per_gene.astype(np.float32),
            "val_genewise_pcc": float(mean_pcc),
            "alpha":            hest_alpha,
        },
        allow_pickle=True,
    )
    return mean_pcc, per_gene


# ── inference for visualization ───────────────────────────────────────────────

@torch.no_grad()
def infer_method(
    method:   dict,
    backbone: Optional[SummaryTableModel],
    head,                        # MLPGeneHead | RidgeWrapper
    dataset:  HestRadiomicsDataset,
    device:   torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-dataset inference returning (coords, gt, pred).

    For neighbor_batch methods, uses InductiveBatchSampler and collects
    the anchor spot only — identical to eval.py's evaluate() logic.
    """
    if method["neighbor_batch"]:
        loader = build_loader(
            dataset,
            batch_size=1 + N_NEIGHBORS + N_SEMANTIC,   # no random globals at inference
            n_neighbors=N_NEIGHBORS,
            n_semantic=N_SEMANTIC,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )
        anchor_only = True
    else:
        loader = DataLoader(
            dataset,
            batch_size=INFER_BATCH,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        anchor_only = False

    coords_l, gt_l, pred_l = [], [], []
    for batch in loader:
        rad     = batch["radiomics"].to(device)
        uni_emb = batch["uni_emb"].to(device)
        coord   = batch["coord"]
        gt      = batch["st"]

        if method["use_backbone"] and backbone is not None:
            z = backbone.encode(rad, coord.to(device))           # (B, D)
            if anchor_only:
                # Neighbor-aware anchor feature (same as extract_features_neighbor)
                z_anchor  = z[0:1]
                z_spatial = z[1:1+N_NEIGHBORS].mean(0, keepdim=True)
                parts = [z_anchor, z_spatial]
                if N_SEMANTIC > 0:
                    z_semantic = z[1+N_NEIGHBORS:1+N_NEIGHBORS+N_SEMANTIC].mean(0, keepdim=True)
                    parts.append(z_semantic)
                parts.append(uni_emb[0:1])
                feat = torch.cat(parts, dim=-1)
            else:
                feat = torch.cat([z, uni_emb], dim=-1)
        else:
            feat = uni_emb

        pred = head(feat)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu()

        if anchor_only:
            coord, gt = coord[0:1], gt[0:1]
            if isinstance(pred, torch.Tensor) and pred.shape[0] > 1:
                pred = pred[0:1]

        coords_l.append(coord.numpy())
        gt_l.append(gt.numpy() if isinstance(gt, torch.Tensor) else gt)
        pred_l.append(pred.numpy() if isinstance(pred, torch.Tensor) else pred)

    return (
        np.concatenate(coords_l, 0).astype(np.float32),
        np.concatenate(gt_l,     0).astype(np.float32),
        np.concatenate(pred_l,   0).astype(np.float32),
    )


# ── gene group selection ──────────────────────────────────────────────────────

def select_gene_groups_by_gap(
    pcc_ref:  np.ndarray,   # UNI-only
    pcc_tgt:  np.ndarray,   # STaRN (neighbor) — largest gap of interest
    k:        int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select k genes per group based on gap = pcc_tgt - pcc_ref."""
    gap       = pcc_tgt - pcc_ref
    sorted_asc = np.argsort(gap)
    win_idx   = sorted_asc[::-1][:k]
    lose_idx  = sorted_asc[:k]

    excluded  = set(win_idx.tolist()) | set(lose_idx.tolist())
    remain    = np.array([i for i in range(len(gap)) if i not in excluded])
    tie_idx   = remain[np.argsort(np.abs(gap[remain]))][:k]

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
    ax.set_title(title, fontsize=6.5, pad=3)
    return sc


def plot_ablation_sample(
    sample_id:    str,
    coords:       np.ndarray,
    gt_z:         np.ndarray,
    pred_zs:      dict[str, np.ndarray],  # name → z-scored pred (N, G)
    pccs:         dict[str, np.ndarray],  # name → per_gene_pcc (G,)
    gene_names:   list[str],
    ckpt_epoch:   int,
    head_type:    str,
    n_top:        int,
    save_path:    Path,
):
    pcc_ref = pccs["uni_only"]
    pcc_tgt = pccs["starn_neighbor"]
    win_idx, tie_idx, lose_idx = select_gene_groups_by_gap(pcc_ref, pcc_tgt, n_top)
    sections = [("[Win]", win_idx), ("[Tie]", tie_idx), ("[Lose]", lose_idx)]
    n_rows   = 1 + n_top * 3
    n_cols   = 1 + len(METHODS)  # GT + 3 methods

    s      = float(np.clip(120_000 / max(len(coords), 1), 1.0, 50.0))
    x_span = float(np.ptp(coords[:, 0]))
    y_span = float(np.ptp(coords[:, 1]))
    col_w  = 3.2
    row_h  = col_w * (y_span / x_span) if x_span > 0 else col_w
    row_h  = float(np.clip(row_h, 2.0, 8.0))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_w * n_cols + 1.0, row_h * n_rows),
        constrained_layout=True,
    )

    mean_pccs = {n: float(v.mean()) for n, v in pccs.items()}
    pcc_str   = "  ".join(
        f"{m['label']}: {mean_pccs[m['name']]:.4f}" for m in METHODS
    )
    fig.suptitle(
        f"{sample_id}  |  epoch {ckpt_epoch:03d}  |  head={head_type}  |  {len(coords):,} spots\n"
        f"Mean PCC — {pcc_str}",
        fontsize=9, fontweight="bold",
    )

    def _row_scatter(row_axes, arrays, titles):
        for ax, arr, lbl in zip(row_axes, arrays, titles):
            sc = _scatter(ax, coords, arr, lbl, s)
            plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="z-score")

    # Row 0: mean z-score
    _row_scatter(
        axes[0],
        [gt_z.mean(1)] + [pred_zs[m["name"]].mean(1) for m in METHODS],
        ["GT  (mean z-score)"] + [
            f"{m['label']}  PCC={mean_pccs[m['name']]:.3f}" for m in METHODS
        ],
    )

    # Gene rows
    row = 1
    for sec_label, indices in sections:
        for gi in indices:
            name = gene_names[gi]
            titles = [f"{sec_label} GT  |  {name}"] + [
                f"{sec_label} {m['label']}  PCC={pccs[m['name']][gi]:.3f}"
                for m in METHODS
            ]
            arrays = [gt_z[:, gi]] + [pred_zs[m["name"]][:, gi] for m in METHODS]
            _row_scatter(axes[row], arrays, titles)
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
    col_w       = 18

    print(f"\n{'=' * 80}")
    print(f"Ablation Results  —  LOOCV gene-wise PCC  [head={head_type}]")
    print("─" * 80)
    print(f"{'Sample':<12}" + "".join(f"{m['label']:>{col_w}}" for m in METHODS))
    print("─" * 80)
    for fold, sid in enumerate(sample_list):
        print(f"{sid:<12}" + "".join(
            f"{method_results[m['name']][fold]:>{col_w}.4f}" for m in METHODS
        ))
    print("─" * 80)
    means = {m["name"]: float(np.mean(method_results[m["name"]])) for m in METHODS}
    stds  = {m["name"]: float(np.std(method_results[m["name"]], ddof=1)) for m in METHODS}
    print(f"{'Mean':12}" + "".join(f"{means[m['name']]:>{col_w}.4f}" for m in METHODS))
    print(f"{'Std':12}"  + "".join(f"{stds [m['name']]:>{col_w}.4f}"  for m in METHODS))

    # Delta rows
    u = method_results["uni_only"]
    r = method_results["starn_random"]
    n = method_results["starn_neighbor"]
    print("─" * 80)
    for label, a, b in [
        ("Δ rnd−uni",  u, r),
        ("Δ nbr−rnd",  r, n),
        ("Δ nbr−uni",  u, n),
    ]:
        deltas = [bv - av for av, bv in zip(a, b)]
        print(
            f"{label:<12}" +
            "".join(f"{d:+{col_w}.4f}" for d in deltas) +
            f"  (mean {float(np.mean(deltas)):+.4f})"
        )
    print(f"{'=' * 80}")

    # Top/bottom genes by STaRN(neighbor) − UNI-only gap
    if all(k in method_pergene for k in ("uni_only", "starn_neighbor")):
        mean_uni = np.stack(method_pergene["uni_only"]).mean(0)
        mean_nbr = np.stack(method_pergene["starn_neighbor"]).mean(0)
        gap = mean_nbr - mean_uni
        print("\nTop-5 genes where STaRN(neighbor) gains most over UNI-only:")
        for i in np.argsort(gap)[::-1][:5]:
            print(f"  {gene_names[i]:20s}  UNI={mean_uni[i]:.4f}  STaRN-nbr={mean_nbr[i]:.4f}  Δ={gap[i]:+.4f}")
        print("Top-5 genes where UNI-only outperforms STaRN(neighbor):")
        for i in np.argsort(gap)[:5]:
            print(f"  {gene_names[i]:20s}  UNI={mean_uni[i]:.4f}  STaRN-nbr={mean_nbr[i]:.4f}  Δ={gap[i]:+.4f}")
        print()


# ── LOOCV runner ──────────────────────────────────────────────────────────────

def _load_fold_ckpt(
    ckpt_path: Path, head_type: str, device: torch.device
) -> tuple[float, np.ndarray, list[str]]:
    if head_type == "ridge":
        saved = np.load(ckpt_path, allow_pickle=True).item()
    else:
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    return float(saved["val_genewise_pcc"]), np.array(saved["per_gene_pcc"]), list(saved["gene_names"])


def run_ablation_eval(
    gene_names:  list[str],
    backbone:    Optional[SummaryTableModel],
    cfg:         Config,
    device:      torch.device,
    ckpt_path:   Optional[Path],
    head_type:   str,
    ridge_alpha: float,
    skip_eval:   bool,
    n_genes:     int = 250,
) -> tuple[dict[str, list[float]], dict[str, list[np.ndarray]]]:
    sample_list   = list(SAMPLE_IDS)
    method_results: dict[str, list[float]]      = {m["name"]: [] for m in METHODS}
    method_pergene: dict[str, list[np.ndarray]] = {m["name"]: [] for m in METHODS}

    for m in METHODS:
        print(f"\n{'#' * 60}")
        print(f"# Method: {m['label']}  head={head_type}")
        print(f"{'#' * 60}")

        backbone_m = backbone if m["use_backbone"] else None
        save_dir   = _ckpt_dir(m["name"], head_type, n_genes)
        fold_suffix = "ridge.npy" if head_type == "ridge" else "best.pt"

        for fold, val_id in enumerate(sample_list):
            train_ids = [s for s in sample_list if s != val_id]
            fold_ckpt = save_dir / f"fold_{fold}_{fold_suffix}"

            if skip_eval and fold_ckpt.exists():
                best_pcc, per_gene, _ = _load_fold_ckpt(fold_ckpt, head_type, device)
                print(f"  [skip] fold {fold} val={val_id}  PCC={best_pcc:.4f}")
            elif head_type == "ridge":
                best_pcc, per_gene = run_ridge_fold(
                    fold=fold, val_id=val_id, train_ids=train_ids,
                    gene_names=gene_names, method=m, backbone=backbone_m,
                    device=device, alpha=ridge_alpha, save_dir=save_dir,
                    n_genes=n_genes,
                )
            else:
                # MLP with per-fold LR grid search + early stopping
                best_pcc, per_gene = run_mlp_fold_hparam(
                    fold=fold, val_id=val_id, train_ids=train_ids,
                    gene_names=gene_names, method=m, backbone=backbone_m,
                    cfg=cfg, device=device, save_dir=save_dir,
                    ckpt_path=ckpt_path if m["use_backbone"] else None,
                    n_genes=n_genes,
                )

            method_results[m["name"]].append(best_pcc)
            if per_gene is not None:
                method_pergene[m["name"]].append(per_gene)

    return method_results, method_pergene


# ── visualization ─────────────────────────────────────────────────────────────

def _load_head_for_viz(
    method:     dict,
    fold:       int,
    head_type:  str,
    gene_names: list[str],
    cfg:        Config,
    device:     torch.device,
    n_genes:    int = 250,
) -> tuple[object, np.ndarray]:
    save_dir    = _ckpt_dir(method["name"], head_type, n_genes)
    fold_suffix = "ridge.npy" if head_type == "ridge" else "best.pt"
    ckpt_path   = save_dir / f"fold_{fold}_{fold_suffix}"

    if head_type == "ridge":
        saved    = np.load(ckpt_path, allow_pickle=True).item()
        head     = RidgeWrapper(saved["coef"], saved["intercept"], device,
                                pipeline=saved.get("pipeline"))
        per_gene = np.array(saved["per_gene_pcc"])
    else:
        saved   = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not method["use_backbone"]:
            in_dim = cfg.uni_dim
        elif method["neighbor_batch"]:
            in_dim = (2 + (1 if N_SEMANTIC > 0 else 0)) * cfg.hidden_dim + cfg.uni_dim
        else:
            in_dim = cfg.hidden_dim + cfg.uni_dim
        head    = MLPGeneHead(
            in_dim=in_dim, hidden_dim=HEAD_HIDDEN_DIM,
            out_dim=len(gene_names), dropout=0.0,
        ).to(device)
        head.load_state_dict(saved["head"])
        head.eval()
        per_gene = np.array(saved["per_gene_pcc"])

    return head, per_gene


def run_ablation_viz(
    gene_names:  list[str],
    backbone:    Optional[SummaryTableModel],
    cfg:         Config,
    device:      torch.device,
    ckpt_epoch:  int,
    head_type:   str,
    n_top:       int,
    n_genes:     int = 250,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold, val_id in enumerate(SAMPLE_IDS):
        print(f"\nFold {fold} | val={val_id}")

        heads: dict[str, object]      = {}
        pccs:  dict[str, np.ndarray]  = {}
        ok = True

        for m in METHODS:
            fold_suffix = "ridge.npy" if head_type == "ridge" else "best.pt"
            ckpt_path   = _ckpt_dir(m["name"], head_type, n_genes) / f"fold_{fold}_{fold_suffix}"
            if not ckpt_path.exists():
                print(f"  [skip] missing {ckpt_path}")
                ok = False
                break
            head, per_gene = _load_head_for_viz(m, fold, head_type, gene_names, cfg, device, n_genes)
            heads[m["name"]] = head
            pccs[m["name"]]  = per_gene

        if not ok:
            continue

        dataset = HestRadiomicsDataset(
            sources=[(EVAL_DATA_ROOT, [val_id])],
            gene_names=gene_names,
        )
        print(f"  {len(dataset):,} spots")

        coords, gt_arr, _ = infer_method(METHODS[0], None, heads["uni_only"], dataset, device)
        gt_z  = zscore(gt_arr)

        pred_zs: dict[str, np.ndarray] = {}
        for m in METHODS:
            bb  = backbone if m["use_backbone"] else None
            _, _, pred = infer_method(m, bb, heads[m["name"]], dataset, device)
            pred_zs[m["name"]] = zscore(pred)

        hvg_tag   = f"_hvg{n_genes}" if n_genes != 250 else ""
        save_path = OUT_DIR / f"{val_id}_ep{ckpt_epoch:03d}_{head_type}{hvg_tag}_ablation.png"
        plot_ablation_sample(
            sample_id=val_id,
            coords=coords,
            gt_z=gt_z,
            pred_zs=pred_zs,
            pccs=pccs,
            gene_names=gene_names,
            ckpt_epoch=ckpt_epoch,
            head_type=head_type,
            n_top=n_top,
            save_path=save_path,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STaRN ablation: UNI-only vs STaRN (random) vs STaRN (neighbor)")
    p.add_argument("--ckpt",        type=Path,  default=None)
    p.add_argument("--head",        choices=["mlp", "ridge"], default="ridge")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--n-genes",     type=int,   default=250)
    p.add_argument("--top-k",       type=int,   default=5)
    p.add_argument("--skip-eval",   action="store_true")
    p.add_argument("--viz-only",    action="store_true")
    return p.parse_args()


def main():
    args   = _parse()
    cfg    = Config()
    device = torch.device(cfg.device)

    _patch_dataset_no_img()

    st_paths   = [EVAL_DATA_ROOT / "st" / f"{sid}.h5ad" for sid in SAMPLE_IDS]
    gene_names = get_common_genes(st_paths, k=args.n_genes, criteria=GENE_CRITERIA)
    print(f"Selected {len(gene_names)} common genes.")

    backbone:   Optional[SummaryTableModel] = None
    ckpt_epoch: int = -1

    if args.ckpt is not None:
        backbone, ckpt_epoch = build_backbone(cfg, args.ckpt, device)
        print(f"Backbone loaded: {args.ckpt}  (epoch {ckpt_epoch})")
    else:
        # Try to recover backbone from existing starn_neighbor checkpoint
        nbr_dir = _ckpt_dir("starn_neighbor", args.head, args.n_genes)
        suffix  = "ridge.npy" if args.head == "ridge" else "best.pt"
        for i in range(len(SAMPLE_IDS)):
            p = nbr_dir / f"fold_{i}_{suffix}"
            if p.exists():
                try:
                    saved = np.load(p, allow_pickle=True).item() if args.head == "ridge" \
                            else torch.load(p, map_location="cpu", weights_only=False)
                    ref = saved.get("backbone_ckpt")
                    if ref and Path(ref).exists():
                        backbone, ckpt_epoch = build_backbone(cfg, Path(ref), device)
                        print(f"Backbone rebuilt from {ref}  (epoch {ckpt_epoch})")
                except Exception:
                    pass
                break

    if backbone is None and not args.viz_only and not args.skip_eval:
        raise SystemExit("--ckpt required to run STaRN evaluation.")

    if not args.viz_only:
        method_results, method_pergene = run_ablation_eval(
            gene_names=gene_names, backbone=backbone, cfg=cfg, device=device,
            ckpt_path=args.ckpt, head_type=args.head,
            ridge_alpha=args.ridge_alpha, skip_eval=args.skip_eval,
            n_genes=args.n_genes,
        )
        print_comparison_table(method_results, gene_names, method_pergene, args.head)

    if backbone is None:
        print("\n[warn] No backbone — skipping visualization.")
        return

    print(f"\nGenerating ablation visualizations → {OUT_DIR}/")
    run_ablation_viz(
        gene_names=gene_names, backbone=backbone, cfg=cfg, device=device,
        ckpt_epoch=ckpt_epoch, head_type=args.head, n_top=args.top_k,
        n_genes=args.n_genes,
    )
    print(f"\nDone — figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
