# z-score 상위 patch마다 왼쪽: 전체 spatial map에서 해당 spot 강조, 오른쪽: 실제 patch image를 같이 저장

# python -m rapacl._gene_analysis_hotspot --model_type rapacl --folds 0,1,2,3
# python -m rapacl._gene_analysis_hotspot --model_type uni_mlp --folds 0,1,2,3
# python -m rapacl._gene_analysis_hotspot --model_type rapacl --folds 0 --target_genes MKI67,GATA3,CEACAM6 --top_k 20 --score_type pred

from __future__ import annotations

import os
import argparse
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from rapacl.engines.trainer_utils import set_seed
from rapacl.engines.data_utils import (
    build_dataset,
    build_loader,
    DEFAULT_DATASET_STRUCTURE,
)
from rapacl.model.rapacl import build_model
from rapacl.model.rapacl_uni import build_uni_model
from rapacl._uni_mlp import build_uni_mlp_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


TARGET_GENES = ["MKI67", "GATA3", "CEACAM6"]
DEFAULT_DATASET_STRUCTURE = DEFAULT_DATASET_STRUCTURE.copy()

PLOT_SPOT_SIZE = 3
HOTSPOT_SPOT_SIZE = 80


def get_experiment_name(model_type: str = "rapacl") -> str:
    if model_type == "uni_mlp":
        return "uni_frozen_mlp"

    backbone = train.BACKBONE.lower().strip()
    return "rapacl_uni_frozen" if backbone == "uni" else f"rapacl_{backbone}"


def get_fold_split_paths(fold: int) -> tuple[str, str]:
    split_dir = os.path.join(train.ROOT_DIR, "splits")
    train_split_csv = os.path.join(split_dir, f"train_{fold}.csv")
    val_split_csv = os.path.join(split_dir, f"test_{fold}.csv")
    return train_split_csv, val_split_csv


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


def remap_old_densenet_keys(state_dict: dict) -> dict:
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("pathomics_encoder.features."):
            k = k.replace(
                "pathomics_encoder.features.",
                "pathomics_encoder.encoder.features.",
                1,
            )
        elif k.startswith("pathomics_encoder.classifier."):
            k = k.replace(
                "pathomics_encoder.classifier.",
                "pathomics_encoder.encoder.classifier.",
                1,
            )

        new_state[k] = v

    return new_state


def find_best_stage2_checkpoint(save_dir: str) -> str:
    candidates = [
        "best_uni_mlp.pt",
        "best_densenet121_mlp.pt",
        "best_stage2_genepred.pt",
        "stage2_best.pt",
        "best_stage2.pt",
        "best_model.pt",
        "model_best.pt",
        "checkpoint_best.pt",
    ]

    for name in candidates:
        path = os.path.join(save_dir, name)
        if os.path.isfile(path):
            return path

    pt_files = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.endswith((".pt", ".pth"))
    ]

    stage2_files = [p for p in pt_files if "stage2" in os.path.basename(p).lower()]
    best_files = [p for p in pt_files if "best" in os.path.basename(p).lower()]

    if stage2_files:
        return sorted(stage2_files)[-1]
    if best_files:
        return sorted(best_files)[-1]
    if pt_files:
        return sorted(pt_files)[-1]

    raise FileNotFoundError(f"No checkpoint found in: {save_dir}")


def build_eval_model(
    device: torch.device,
    num_genes: int,
    model_type: str = "rapacl",
):
    if model_type == "uni_mlp":
        return build_uni_mlp_model(
            device=device,
            num_genes=num_genes,
        )

    backbone = train.BACKBONE.lower().strip()
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    if backbone == "uni":
        model = build_uni_model(
            device=device,
            num_genes=num_genes,
            num_radiomics_features=num_radiomics_features,
        )
    else:
        model = build_model(
            device=device,
            num_genes=num_genes,
            num_radiomics_features=num_radiomics_features,
            backbone=backbone,
        )

    return model.to(device)


def zscore_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)


def find_h5_key(f, candidates):
    found = []

    def visitor(name, obj):
        if name.split("/")[-1] in candidates:
            found.append(name)

    f.visititems(visitor)

    if len(found) == 0:
        raise KeyError(
            f"Cannot find any of {candidates}. Available keys: {list(f.keys())}"
        )

    return found[0]


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


def inject_coords_to_dataset(dataset):
    split_cols = DEFAULT_DATASET_STRUCTURE["split_csv_cols"]

    sample_to_h5 = {}

    for _, row in dataset.split_df.iterrows():
        sample_id = str(row[split_cols["sample_id"]])
        patches_h5_path = os.path.join(
            dataset.bench_data_root,
            row[split_cols["patches"]],
        )
        sample_to_h5[sample_id] = patches_h5_path

    coord_maps = {}

    for sample_id, h5_path in sample_to_h5.items():
        barcode_to_coord = {}

        barcodes, coords = read_h5_barcodes_and_coords(h5_path)

        for barcode, coord in zip(barcodes, coords):
            barcode = normalize_barcode_for_match(barcode)
            barcode_to_coord[barcode] = (int(coord[0]), int(coord[1]))

        coord_maps[sample_id] = barcode_to_coord

    missing = 0

    for sample in dataset.samples:
        sample_id = sample["sample_id"]
        barcode = normalize_barcode_for_match(sample["barcode"])

        coord = coord_maps.get(sample_id, {}).get(barcode)

        if coord is None:
            sample["coord_x"] = -1
            sample["coord_y"] = -1
            missing += 1
        else:
            sample["coord_x"] = coord[0]
            sample["coord_y"] = coord[1]

    print(f"[INFO] injected coords to dataset | missing={missing}/{len(dataset.samples)}")


@torch.no_grad()
def predict_gene_expression(model, loader, device: torch.device):
    model.eval()

    all_preds = []
    all_targets = []
    all_meta = []

    for batch in tqdm(loader, desc="[Eval]", leave=False):
        image = batch["image"].to(device, non_blocking=True).float()
        radiomics = batch["radiomics"].to(device, non_blocking=True).float()
        target = batch["gene"].to(device, non_blocking=True).float()

        if hasattr(model, "forward_gene"):
            output = model.forward_gene(
                image=image,
                radiomics=radiomics,
            )
            pred = output["pred_gene"]
        else:
            pred = model(image)

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        batch_size = target.size(0)
        for i in range(batch_size):
            meta = {}
            for key in [
                "sample_id",
                "barcode",
                "patch_idx",
                "idx",
                "coord_x",
                "coord_y",
            ]:
                if key in batch:
                    value = batch[key][i]
                    if torch.is_tensor(value):
                        value = value.item()
                    meta[key] = value
            all_meta.append(meta)

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return preds, targets, all_meta


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    dataset[idx]["image"]가 C,H,W normalized tensor일 가능성을 고려.
    시각화용으로 0~1 범위로 min-max 변환.
    """
    if torch.is_tensor(image):
        image = image.detach().cpu()

    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.permute(1, 2, 0)

    image = image.numpy().astype(np.float32)

    img_min = float(image.min())
    img_max = float(image.max())

    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)

    return image


def get_image_from_dataset(dataset, global_index: int) -> np.ndarray:
    sample = dataset[global_index]

    if "image" not in sample:
        raise KeyError("dataset item does not contain 'image' key.")

    return tensor_image_to_numpy(sample["image"])


def build_hotspot_dataframe(
    preds: np.ndarray,
    targets: np.ndarray,
    meta: list[dict],
    genes: list[str],
    target_gene: str,
    score_type: str,
) -> pd.DataFrame:
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    if target_gene not in gene_to_idx:
        raise ValueError(f"Target gene not found: {target_gene}")

    gene_idx = gene_to_idx[target_gene]

    df = pd.DataFrame(meta).copy()
    df["global_index"] = np.arange(len(df))
    df["gene"] = target_gene
    df["gene_idx"] = gene_idx
    df["y_true"] = targets[:, gene_idx]
    df["y_pred"] = preds[:, gene_idx]

    if "coord_x" not in df.columns or "coord_y" not in df.columns:
        raise KeyError("coord_x / coord_y not found. Check inject_coords_to_dataset().")

    df["z_true"] = np.nan
    df["z_pred"] = np.nan
    df["z_error_abs"] = np.nan

    if "sample_id" in df.columns:
        group_iter = df.groupby("sample_id").groups.items()
    else:
        df["sample_id"] = "all"
        group_iter = df.groupby("sample_id").groups.items()

    for _, indices in group_iter:
        indices = list(indices)

        z_true = zscore_1d(df.loc[indices, "y_true"].values)
        z_pred = zscore_1d(df.loc[indices, "y_pred"].values)

        df.loc[indices, "z_true"] = z_true
        df.loc[indices, "z_pred"] = z_pred
        df.loc[indices, "z_error_abs"] = np.abs(z_true - z_pred)

    if score_type == "pred":
        df["hotspot_score"] = df["z_pred"]
    elif score_type == "true":
        df["hotspot_score"] = df["z_true"]
    elif score_type == "error":
        df["hotspot_score"] = df["z_error_abs"]
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    return df


def save_hotspot_pair_figure(
    all_df: pd.DataFrame,
    row: pd.Series,
    patch_img: np.ndarray,
    out_path: str,
    score_type: str,
):
    sample_id = row["sample_id"]

    sdf = all_df[all_df["sample_id"] == sample_id].copy()

    # 기존 spatial plot과 동일하게 x=coord_y, y=coord_x 사용
    x = sdf["coord_y"].values
    y = sdf["coord_x"].values

    hx = row["coord_y"]
    hy = row["coord_x"]

    if score_type == "pred":
        color_values = sdf["z_pred"].values
        color_title = "Prediction z-score"
        hotspot_value = row["z_pred"]
    elif score_type == "true":
        color_values = sdf["z_true"].values
        color_title = "Ground Truth z-score"
        hotspot_value = row["z_true"]
    else:
        color_values = sdf["z_error_abs"].values
        color_title = "|GT z - Pred z|"
        hotspot_value = row["z_error_abs"]

    vmin = float(np.nanmin(color_values))
    vmax = float(np.nanmax(color_values))

    if score_type in ["pred", "true"]:
        abs_max = min(max(abs(vmin), abs(vmax)), 3.0)
        vmin, vmax = -abs_max, abs_max

    fig = plt.figure(figsize=(10, 5))

    ax1 = plt.subplot(1, 2, 1)
    sc = ax1.scatter(
        x,
        y,
        c=color_values,
        s=PLOT_SPOT_SIZE,
        marker="h",
        vmin=vmin,
        vmax=vmax,
    )

    ax1.scatter(
        [hx],
        [hy],
        s=HOTSPOT_SPOT_SIZE,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        zorder=5,
    )

    ax1.scatter(
        [hx],
        [hy],
        s=20,
        marker="x",
        c="red",
        linewidths=1.5,
        zorder=6,
    )

    ax1.invert_yaxis()
    ax1.axis("equal")
    ax1.axis("off")
    ax1.set_title(
        f"Spatial location\n{color_title}",
        fontsize=10,
    )

    cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(patch_img)
    ax2.axis("off")
    ax2.set_title("Patch image", fontsize=10)

    title = (
        f"{row['gene']} | {sample_id} | rank={int(row['hotspot_rank'])} | "
        f"idx={int(row['global_index'])} | "
        f"score={hotspot_value:.3f}\n"
        f"coord=({int(row['coord_x'])}, {int(row['coord_y'])}) | "
        f"true_z={row['z_true']:.3f}, pred_z={row['z_pred']:.3f}"
    )

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_hotspots_for_gene(
    dataset,
    preds: np.ndarray,
    targets: np.ndarray,
    meta: list[dict],
    genes: list[str],
    target_gene: str,
    score_type: str,
    top_k: int,
    out_dir: str,
):
    df = build_hotspot_dataframe(
        preds=preds,
        targets=targets,
        meta=meta,
        genes=genes,
        target_gene=target_gene,
        score_type=score_type,
    )

    valid_df = df[
        (df["coord_x"] >= 0)
        & (df["coord_y"] >= 0)
        & np.isfinite(df["hotspot_score"])
    ].copy()

    top_df = (
        valid_df.sort_values("hotspot_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    top_df["hotspot_rank"] = np.arange(1, len(top_df) + 1)

    gene_out_dir = os.path.join(out_dir, target_gene, score_type)
    os.makedirs(gene_out_dir, exist_ok=True)

    csv_path = os.path.join(
        gene_out_dir,
        f"{target_gene}_{score_type}_top{top_k}_hotspots.csv",
    )
    top_df.to_csv(csv_path, index=False)
    print(f"[INFO] saved hotspot csv: {csv_path}")

    for _, row in top_df.iterrows():
        global_index = int(row["global_index"])
        rank = int(row["hotspot_rank"])
        sample_id = str(row["sample_id"]).replace("/", "_")

        patch_img = get_image_from_dataset(dataset, global_index)

        fig_name = (
            f"rank{rank:03d}_idx{global_index}_"
            f"{sample_id}_"
            f"coordx{int(row['coord_x'])}_coordy{int(row['coord_y'])}_"
            f"score{row['hotspot_score']:.3f}.png"
        )

        fig_path = os.path.join(gene_out_dir, fig_name)

        save_hotspot_pair_figure(
            all_df=df,
            row=row,
            patch_img=patch_img,
            out_path=fig_path,
            score_type=score_type,
        )

    print(f"[INFO] saved hotspot figures: {gene_out_dir}")


def run_one_fold_hotspot_analysis(
    fold: int,
    device: torch.device,
    checkpoint_path: str | None,
    model_type: str,
    target_genes: list[str],
    score_type: str,
    top_k: int,
):
    set_seed(train.SEED + fold * 100)

    _, val_split_csv = get_fold_split_paths(fold)

    val_dataset = build_dataset(
        val_split_csv,
        use_image_augmentation=False,
    )

    inject_coords_to_dataset(val_dataset)

    val_loader, _ = build_loader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        distributed=False,
        pair_augment=False,
    )

    genes = val_dataset.genes
    num_genes = len(genes)

    exp_name = get_experiment_name(model_type)
    save_dir = os.path.join(train.OUTPUT_CHECKPOINT_DIR, exp_name, f"fold_{fold}")

    if checkpoint_path is None:
        checkpoint_path = find_best_stage2_checkpoint(save_dir)

    print(f"[INFO][Fold {fold}] checkpoint: {checkpoint_path}")

    model = build_eval_model(
        device=device,
        num_genes=num_genes,
        model_type=model_type,
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))
    state_dict = remap_old_densenet_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[MISSING]")
    print("\n".join(missing[:50]))
    print("[UNEXPECTED]")
    print("\n".join(unexpected[:50]))

    assert len(unexpected) == 0
    assert len(missing) == 0

    preds, targets, meta = predict_gene_expression(
        model=model,
        loader=val_loader,
        device=device,
    )

    fold_out_dir = os.path.join(
        train.OUTPUT_DIR,
        exp_name,
        "gene_analysis_hotspot",
        f"fold_{fold}",
    )
    os.makedirs(fold_out_dir, exist_ok=True)

    for gene in target_genes:
        if gene not in genes:
            print(f"[WARN][Fold {fold}] target gene not found: {gene}")
            continue

        save_hotspots_for_gene(
            dataset=val_dataset,
            preds=preds,
            targets=targets,
            meta=meta,
            genes=genes,
            target_gene=gene,
            score_type=score_type,
            top_k=top_k,
            out_dir=fold_out_dir,
        )


def parse_target_genes(s: str | None) -> list[str]:
    if s is None or len(s.strip()) == 0:
        return TARGET_GENES
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="e.g., 0,1,2,3",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="single checkpoint path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="rapacl",
        choices=["rapacl", "uni_mlp"],
    )
    parser.add_argument(
        "--target_genes",
        type=str,
        default="MKI67,GATA3,CEACAM6",
        help="comma-separated target genes",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        default="pred",
        choices=["pred", "true", "error"],
        help="pred: high predicted z-score, true: high GT z-score, error: high z-score disagreement",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    device = torch.device(train.DEVICE if torch.cuda.is_available() else "cpu")

    if args.folds is None:
        folds = train.SELECT_FOLDS
    else:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    target_genes = parse_target_genes(args.target_genes)

    print("=" * 80)
    print("[HOTSPOT ANALYSIS]")
    print(f"model_type  : {args.model_type}")
    print(f"folds       : {folds}")
    print(f"target_genes: {target_genes}")
    print(f"score_type  : {args.score_type}")
    print(f"top_k       : {args.top_k}")
    print("=" * 80)

    for fold in folds:
        run_one_fold_hotspot_analysis(
            fold=fold,
            device=device,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            target_genes=target_genes,
            score_type=args.score_type,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()

    