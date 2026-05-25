from __future__ import annotations

"""
python -m baselines.stnet._stnet_gene_analysis \
  --config configs/stnet.yaml \
  --folds 0,1,2,3 \
  --run_root outputs/stnet/run_YYYYMMDD_HHMMSS
"""
"""
python -m baselines.stnet._stnet_gene_analysis \
  --config baselines/configs/stnet.yaml \
  --folds 3 \
  --run_root /root/workspace/RaPaCL/outputs/stnet/hvg250/fold3/run_20260512_232358
"""


import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.common.config import load_yaml
from baselines.common.utils import (
    get_device,
    load_gene_names,
    resolve_gene_list_path,
    resolve_split_path,
    seed_everything,
)
from baselines.stnet import build_model
from baselines.stnet.dataset import STNetDataset


TARGET_GENES = [
    "FASN",
    "CEACAM6",
    "GATA3",
    "SERPINA3",
    "TACSTD2",
    "ABCC11",
]

PLOT_SPOT_SIZE = 3


def create_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def unwrap_state_dict(ckpt: Any) -> dict:
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_module_prefix(state_dict: dict) -> dict:
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def find_best_stnet_checkpoint(ckpt_dir: Path, fold: int | None = None) -> Path:
    candidates = [
        ckpt_dir / "final_model.pth",
        ckpt_dir / "best_model.pth",
        ckpt_dir / "model_best.pth",
    ]

    for path in candidates:
        if path.exists():
            return path

    pt_files = sorted(list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.pt")))
    if pt_files:
        return pt_files[-1]

    raise FileNotFoundError(f"No STNet checkpoint found in: {ckpt_dir}")


def get_prediction_from_output(output):
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for key in ["gene_pred", "pred_gene", "gene_prediction", "gene", "pred", "logits"]:
            if key in output:
                return output[key]

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item

    raise TypeError(f"Cannot extract gene prediction from output type: {type(output)}")


def get_batch_image_and_gene(batch):
    if isinstance(batch, dict):
        image = batch.get("image", batch.get("img", batch.get("patch")))
        gene = batch.get("gene", batch.get("expression", batch.get("expr", batch.get("target"))))

        if image is None:
            raise KeyError(f"Cannot find image key in batch. keys={list(batch.keys())}")
        if gene is None:
            raise KeyError(f"Cannot find gene key in batch. keys={list(batch.keys())}")

        return image, gene

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(f"Tuple/list batch must contain at least image and gene: len={len(batch)}")
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def extract_meta_from_batch(batch, batch_size: int) -> list[dict]:
    metas = []

    if not isinstance(batch, dict):
        for _ in range(batch_size):
            metas.append({})
        return metas

    meta_keys = [
        "sample_id",
        "barcode",
        "patch_idx",
        "idx",
        "coord_x",
        "coord_y",
        "x",
        "y",
    ]

    for i in range(batch_size):
        meta = {}

        for key in meta_keys:
            if key not in batch:
                continue

            value = batch[key][i]

            if torch.is_tensor(value):
                value = value.item() if value.ndim == 0 else value.detach().cpu().numpy()

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            meta[key] = value

        metas.append(meta)

    return metas


@torch.no_grad()
def predict_gene_expression(model, loader, device: torch.device):
    model.eval()

    all_preds = []
    all_targets = []
    all_meta = []

    for batch in tqdm(loader, desc="[Eval]", leave=False):
        image, target = get_batch_image_and_gene(batch)

        image = image.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()

        output = model(image)
        pred = get_prediction_from_output(output)

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())
        all_meta.extend(extract_meta_from_batch(batch, target.size(0)))

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return preds, targets, all_meta


def compute_gene_pccs(preds: np.ndarray, targets: np.ndarray, genes: list[str]) -> pd.DataFrame:
    rows = []

    for gene_idx, gene_name in enumerate(genes):
        y_true = targets[:, gene_idx]
        y_pred = preds[:, gene_idx]

        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            pcc = np.nan
        else:
            pcc = pearsonr(y_true, y_pred)[0]

        rows.append(
            {
                "gene": gene_name,
                "gene_idx": gene_idx,
                "pcc": pcc,
            }
        )

    df = pd.DataFrame(rows)
    df["pcc_rank"] = df["pcc"].rank(ascending=False, method="min").astype("Int64")
    return df.sort_values("pcc_rank").reset_index(drop=True)


def find_h5_key(f, candidates):
    found = []

    def visitor(name, obj):
        if name.split("/")[-1] in candidates:
            found.append(name)

    f.visititems(visitor)

    if not found:
        raise KeyError(f"Cannot find any of {candidates}. Available keys: {list(f.keys())}")

    return found[0]


def normalize_barcode_for_match(barcode) -> str:
    if isinstance(barcode, np.ndarray):
        barcode = barcode.item()

    if isinstance(barcode, bytes):
        barcode = barcode.decode("utf-8")

    barcode = str(barcode).strip()

    if barcode.startswith("[b'") and barcode.endswith("']"):
        barcode = barcode[3:-2]
    if barcode.startswith("b'") and barcode.endswith("'"):
        barcode = barcode[2:-1]
    if barcode.startswith('["') and barcode.endswith('"]'):
        barcode = barcode[2:-2]

    return barcode


def read_h5_barcodes_and_coords(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        barcode_key = find_h5_key(f, ["barcodes", "barcode"])
        coord_key = find_h5_key(f, ["coords", "coord", "coordinates"])

        print(f"[INFO] h5: {h5_path}")
        print(f"[INFO] barcode key: {barcode_key}")
        print(f"[INFO] coord key  : {coord_key}")

        barcodes = f[barcode_key][:]
        coords = f[coord_key][:]

    return barcodes, coords


def get_split_column(split_df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in split_df.columns:
            return col
    raise KeyError(f"Cannot find any of columns {candidates}. Available={list(split_df.columns)}")


def inject_coords_to_dataset(dataset):
    if not hasattr(dataset, "split_df") or not hasattr(dataset, "bench_data_root"):
        print("[WARN] dataset does not have split_df/bench_data_root. Skip coord injection.")
        return

    split_df = dataset.split_df

    sample_col = get_split_column(split_df, ["sample_id", "sample", "id"])
    patch_col = get_split_column(split_df, ["patches_path", "patches", "patch_path", "h5_path"])

    sample_to_h5 = {}

    for _, row in split_df.iterrows():
        sample_id = str(row[sample_col])
        h5_path = Path(dataset.bench_data_root) / str(row[patch_col])
        sample_to_h5[sample_id] = str(h5_path)

    coord_maps = {}

    for sample_id, h5_path in sample_to_h5.items():
        barcode_to_coord = {}
        barcodes, coords = read_h5_barcodes_and_coords(h5_path)

        for barcode, coord in zip(barcodes, coords):
            barcode = normalize_barcode_for_match(barcode)
            barcode_to_coord[barcode] = (int(coord[0]), int(coord[1]))

        coord_maps[sample_id] = barcode_to_coord

    if not hasattr(dataset, "samples"):
        print("[WARN] dataset does not have samples. Skip writing coords into samples.")
        return

    missing = 0

    for sample in dataset.samples:
        sample_id = str(sample.get("sample_id", sample.get("sample", "")))
        barcode = normalize_barcode_for_match(sample.get("barcode", sample.get("barcodes", "")))

        coord = coord_maps.get(sample_id, {}).get(barcode)

        if coord is None:
            sample["coord_x"] = -1
            sample["coord_y"] = -1
            missing += 1
        else:
            sample["coord_x"] = coord[0]
            sample["coord_y"] = coord[1]

    print(f"[INFO] injected coords to dataset | missing={missing}/{len(dataset.samples)}")


def save_target_gene_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    meta: list[dict],
    genes: list[str],
    target_genes: list[str],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    for gene in target_genes:
        if gene not in gene_to_idx:
            print(f"[WARN] target gene not found in gene list: {gene}")
            continue

        idx = gene_to_idx[gene]
        rows = []

        for sample_meta, y_true, y_pred in zip(meta, targets[:, idx], preds[:, idx]):
            row = dict(sample_meta)
            row["gene"] = gene
            row["gene_idx"] = idx
            row["y_true"] = float(y_true)
            row["y_pred"] = float(y_pred)
            rows.append(row)

        path = out_dir / f"{gene}_predictions.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[INFO] saved predictions: {path}")


def save_target_gene_pcc_barplot(
    target_summary: pd.DataFrame,
    pcc_df: pd.DataFrame,
    out_dir: Path,
):
    if len(target_summary) == 0:
        return

    plot_df = target_summary.sort_values("pcc", ascending=False).copy()
    baseline_pcc = pcc_df["pcc"].mean()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(plot_df["gene"], plot_df["pcc"])

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{row["pcc"]:.3f}\nRank {int(row["pcc_rank"])}',
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.axhline(
        y=baseline_pcc,
        linestyle="--",
        linewidth=2,
        label=f"Mean PCC ({baseline_pcc:.3f})",
    )

    plt.ylim(0, max(1.0, plot_df["pcc"].max() + 0.1))
    plt.ylabel("Gene-wise PCC")
    plt.xlabel("Gene")
    plt.title("STNet Target Gene-wise PCC")
    plt.legend()
    plt.tight_layout()

    path = out_dir / "target_gene_pcc_barplot.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] saved target gene PCC barplot: {path}")


def save_all_gene_sorted_pcc_barplot(
    pcc_df: pd.DataFrame,
    target_genes: list[str],
    out_dir: Path,
):
    plot_df = pcc_df.sort_values("pcc", ascending=False).reset_index(drop=True).copy()
    plot_df["rank_index"] = np.arange(1, len(plot_df) + 1)

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df["rank_index"], plot_df["pcc"], width=1.0)

    target_df = plot_df[plot_df["gene"].isin(target_genes)]
    plt.scatter(target_df["rank_index"], target_df["pcc"], s=60, zorder=3)

    for _, row in target_df.iterrows():
        plt.text(
            row["rank_index"],
            row["pcc"] + 0.02,
            f'{row["gene"]}\nRank {int(row["pcc_rank"])}',
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Genes sorted by PCC ranking")
    plt.ylabel("Gene-wise PCC")
    plt.title("STNet All Gene-wise PCC Ranking")
    plt.tight_layout()

    path = out_dir / "all_gene_sorted_pcc_ranking.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] saved all gene sorted PCC barplot: {path}")


def zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)


def symmetric_zlim(a: np.ndarray, b: np.ndarray, clip: float = 3.0) -> tuple[float, float]:
    m = max(np.max(np.abs(a)), np.max(np.abs(b)))
    m = min(float(m), clip)
    return -m, m


def save_spatial_expression_maps(
    preds: np.ndarray,
    targets: np.ndarray,
    meta: list[dict],
    genes: list[str],
    target_summary: pd.DataFrame,
    target_genes: list[str],
    out_dir: Path,
):
    meta_df = pd.DataFrame(meta)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    if len(meta_df) == 0:
        print("[WARN] meta is empty. Skip spatial expression map.")
        return

    for gene in target_genes:
        if gene not in gene_to_idx:
            print(f"[WARN] target gene not found: {gene}")
            continue

        gene_idx = gene_to_idx[gene]

        df = meta_df.copy()
        df["y_true"] = targets[:, gene_idx]
        df["y_pred"] = preds[:, gene_idx]

        summary_row = target_summary[target_summary["gene"] == gene]
        if len(summary_row) > 0:
            pcc = float(summary_row.iloc[0]["pcc"])
            rank = int(summary_row.iloc[0]["pcc_rank"])
            title_suffix = f"PCC={pcc:.3f}, Rank={rank}/{len(genes)}"
        else:
            title_suffix = ""

        if "sample_id" in df.columns:
            group_iter = df.groupby("sample_id")
        else:
            group_iter = [("all", df)]

        for sample_id, sdf in group_iter:
            if "coord_x" in sdf.columns and "coord_y" in sdf.columns and (sdf["coord_x"] >= 0).all():
                x = sdf["coord_y"].values
                y = sdf["coord_x"].values
            elif "x" in sdf.columns and "y" in sdf.columns:
                x = sdf["x"].values
                y = sdf["y"].values
            elif "patch_idx" in sdf.columns:
                grid_w = int(np.ceil(np.sqrt(sdf["patch_idx"].max() + 1)))
                x = sdf["patch_idx"].values % grid_w
                y = sdf["patch_idx"].values // grid_w
            else:
                print(f"[WARN] no spatial coordinate for {gene} / {sample_id}. Skip.")
                continue

            vmin = min(sdf["y_true"].min(), sdf["y_pred"].min())
            vmax = max(sdf["y_true"].max(), sdf["y_pred"].max())

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            sc1 = plt.scatter(x, y, c=sdf["y_true"], s=PLOT_SPOT_SIZE, marker="h", vmin=vmin, vmax=vmax)
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Ground Truth")
            plt.colorbar(sc1, fraction=0.046, pad=0.04)

            plt.subplot(1, 2, 2)
            sc2 = plt.scatter(x, y, c=sdf["y_pred"], s=PLOT_SPOT_SIZE, marker="h", vmin=vmin, vmax=vmax)
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Prediction")
            plt.colorbar(sc2, fraction=0.046, pad=0.04)

            plt.suptitle(f"{gene} | {sample_id} | {title_suffix}")
            plt.tight_layout()

            safe_sample_id = str(sample_id).replace("/", "_")
            path = out_dir / f"{gene}_spatial_gt_vs_pred_{safe_sample_id}.png"
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"[INFO] saved spatial map: {path}")

            z_true = zscore_1d(sdf["y_true"].values)
            z_pred = zscore_1d(sdf["y_pred"].values)
            z_vmin, z_vmax = symmetric_zlim(z_true, z_pred, clip=3.0)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            sc1 = plt.scatter(x, y, c=z_true, s=PLOT_SPOT_SIZE, marker="h", vmin=z_vmin, vmax=z_vmax)
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Ground Truth (z-score)")
            plt.colorbar(sc1, fraction=0.046, pad=0.04)

            plt.subplot(1, 2, 2)
            sc2 = plt.scatter(x, y, c=z_pred, s=PLOT_SPOT_SIZE, marker="h", vmin=z_vmin, vmax=z_vmax)
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Prediction (z-score)")
            plt.colorbar(sc2, fraction=0.046, pad=0.04)

            plt.suptitle(f"{gene} | {sample_id} | z-score normalized | {title_suffix}")
            plt.tight_layout()

            z_path = out_dir / f"{gene}_spatial_gt_vs_pred_zscore_{safe_sample_id}.png"
            plt.savefig(z_path, dpi=300)
            plt.close()
            print(f"[INFO] saved z-score spatial map: {z_path}")


def run_one_fold_analysis(
    cfg: dict[str, Any],
    fold: int,
    device: torch.device,
    run_root: Path,
    checkpoint_path: str | None = None,
):
    paths_cfg = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    seed_everything(cfg.get("seed", 42) + fold * 100)

    bench_data_root = paths_cfg["bench_data_root"]
    gene_list_path = resolve_gene_list_path(paths_cfg, model_cfg)
    test_split_path = resolve_split_path(paths_cfg, "test", outer_fold=fold)

    genes = load_gene_names(gene_list_path)
    model_cfg["num_genes"] = len(genes)

    print("=" * 80)
    print(f"[INFO] Fold {fold}")
    print(f"[INFO] test split: {test_split_path}")
    print(f"[INFO] gene list : {gene_list_path}")
    print(f"[INFO] num genes : {len(genes)}")
    print("=" * 80)

    test_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=str(gene_list_path),
        split_csv_path=str(test_split_path),
        transforms=None,
    )

    try:
        inject_coords_to_dataset(test_dataset)
    except Exception as e:
        print(f"[WARN] coord injection failed: {e}")

    test_loader = create_loader(
        test_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 0),
    )

    model = build_model(model_cfg).to(device)

    if checkpoint_path is None:
        ckpt_dir = run_root / "checkpoints"
        ckpt_path = find_best_stnet_checkpoint(ckpt_dir, fold)
    else:
        ckpt_path = Path(checkpoint_path)

    print(f"[INFO][Fold {fold}] checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO][Fold {fold}] missing keys: {len(missing)}")
    print(f"[INFO][Fold {fold}] unexpected keys: {len(unexpected)}")

    if missing:
        print("[MISSING][:30]")
        print("\n".join(missing[:30]))
    if unexpected:
        print("[UNEXPECTED][:30]")
        print("\n".join(unexpected[:30]))

    preds, targets, meta = predict_gene_expression(
        model=model,
        loader=test_loader,
        device=device,
    )

    fold_out_dir = run_root / "gene_analysis" / f"fold_{fold}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    pcc_df = compute_gene_pccs(preds, targets, genes)
    pcc_path = fold_out_dir / "all_gene_pcc_rank.csv"
    pcc_df.to_csv(pcc_path, index=False)
    print(f"[INFO][Fold {fold}] saved PCC rank: {pcc_path}")

    target_summary = pcc_df[pcc_df["gene"].isin(TARGET_GENES)].copy()
    target_summary = target_summary.sort_values("gene").reset_index(drop=True)

    target_summary_path = fold_out_dir / "target_gene_pcc_rank.csv"
    target_summary.to_csv(target_summary_path, index=False)
    print(f"[INFO][Fold {fold}] saved target summary: {target_summary_path}")

    save_target_gene_predictions(
        preds=preds,
        targets=targets,
        meta=meta,
        genes=genes,
        target_genes=TARGET_GENES,
        out_dir=fold_out_dir,
    )

    save_target_gene_pcc_barplot(
        target_summary=target_summary,
        pcc_df=pcc_df,
        out_dir=fold_out_dir,
    )

    save_spatial_expression_maps(
        preds=preds,
        targets=targets,
        meta=meta,
        genes=genes,
        target_summary=target_summary,
        target_genes=TARGET_GENES,
        out_dir=fold_out_dir,
    )

    save_all_gene_sorted_pcc_barplot(
        pcc_df=pcc_df,
        target_genes=TARGET_GENES,
        out_dir=fold_out_dir,
    )

    return target_summary


def main():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", type=str, required=True)
    parent.add_argument("--folds", type=str, default="0,1,2,3")
    parent.add_argument("--run_root", type=str, required=True)
    parent.add_argument("--checkpoint", type=str, default=None)

    args, unknown = parent.parse_known_args()

    cfg = load_yaml(args.config)

    cfg.setdefault("paths", {})
    cfg.setdefault("runtime", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("cv", {})

    seed_everything(cfg.get("seed", 42))

    device = get_device(cfg["runtime"])
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    run_root = Path(args.run_root)

    all_rows = []

    for fold in folds:
        summary = run_one_fold_analysis(
            cfg=cfg,
            fold=fold,
            device=device,
            run_root=run_root,
            checkpoint_path=args.checkpoint,
        )
        summary["fold"] = fold
        all_rows.append(summary)

    out_dir = run_root / "gene_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_df = pd.concat(all_rows, axis=0).reset_index(drop=True)
    final_path = out_dir / "target_gene_pcc_rank_all_folds.csv"
    final_df.to_csv(final_path, index=False)

    print("\n" + "=" * 80)
    print("[STNET TARGET GENE PCC SUMMARY]")
    print(final_df)
    print(f"Saved to: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
