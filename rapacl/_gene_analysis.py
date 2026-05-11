# python -m rapacl._gene_analysis --model_type rapacl --folds 0,1,2,3
# python -m rapacl._gene_analysis --model_type uni_mlp --folds 0,1,2,3
# python -m rapacl._gene_analysis --model_type densenet_mlp --folds 0,1,2,3

from __future__ import annotations

import os
import json
import argparse
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from rapacl.engines.trainer_utils import set_seed
from rapacl.engines.data_utils import build_dataset, build_loader, DEFAULT_DATASET_STRUCTURE
from rapacl.model.rapacl import build_model
from rapacl.model.rapacl_uni import build_uni_model
from rapacl._uni_mlp import build_uni_mlp_model
from rapacl._densenet_mlp import build_densenet_mlp_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


TARGET_GENES = ["MKI67", "GATA3", "CEACAM6"]
DEFAULT_DATASET_STRUCTURE = DEFAULT_DATASET_STRUCTURE.copy()
PLOT_SPOT_SIZE = 3 # 0.5 (TENX99 fit) # 3 (others) 


def get_experiment_name(model_type: str = "rapacl") -> str:
    if model_type == "uni_mlp":
        return "uni_frozen_mlp"

    if model_type == "densenet_mlp":
        return "densenet121_mlp"

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

    if model_type == "densenet_mlp":
        return build_densenet_mlp_model(
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


def get_prediction_from_output(output):
    """
    모델 output 형식이 프로젝트마다 조금 다를 수 있어서 안전하게 처리.
    필요하면 여기만 네 모델 output key에 맞게 수정하면 됨.
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for key in ["gene_pred", "pred_gene", "gene_prediction", "gene", "pred", "logits"]:
            if key in output:
                return output[key]

    if isinstance(output, (tuple, list)):
        # 보통 마지막 또는 첫 번째가 gene prediction인 경우가 많음.
        # 네 모델에서 다르면 이 부분만 수정.
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item

    raise TypeError(f"Cannot extract gene prediction from output type: {type(output)}")


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
            for key in ["sample_id", "barcode", "patch_idx", "idx", "coord_x", "coord_y"]:
                if key in batch:
                    value = batch[key][i]
                    if torch.is_tensor(value):
                        value = value.item()
                    meta[key] = value
            all_meta.append(meta)

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
    df = df.sort_values("pcc_rank").reset_index(drop=True)

    return df


def find_h5_key(f, candidates):
    """
    h5 파일 안에서 candidates 중 존재하는 key를 찾음.
    최상위뿐 아니라 group 내부도 탐색.
    """
    found = []

    def visitor(name, obj):
        if name.split("/")[-1] in candidates:
            found.append(name)

    f.visititems(visitor)

    if len(found) == 0:
        raise KeyError(
            f"Cannot find any of {candidates}. "
            f"Available keys: {list(f.keys())}"
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
    """
    data_utils.py를 수정하지 않고,
    dataset.samples에 coord_x, coord_y를 후처리로 추가.
    """
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

        print("\n[DEBUG] h5 barcode examples")
        for k in list(barcode_to_coord.keys())[:5]:
            print(repr(k))

        print("\n[DEBUG] dataset barcode examples")
        for sample in dataset.samples[:5]:
            print(repr(sample["barcode"]))

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


def save_target_gene_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    meta: list[dict],
    genes: list[str],
    target_genes: list[str],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

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

        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, f"{gene}_predictions.csv")
        df.to_csv(path, index=False)
        print(f"[INFO] saved predictions: {path}")


def save_target_gene_pcc_barplot(
    target_summary: pd.DataFrame,
    pcc_df: pd.DataFrame,
    out_dir: str,
):
    plot_df = target_summary.sort_values("pcc", ascending=False).copy()

    # 전체 gene 평균 PCC baseline
    baseline_pcc = pcc_df["pcc"].mean()

    plt.figure(figsize=(6, 5))
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

    # baseline dashed line
    plt.axhline(
        y=baseline_pcc,
        linestyle="--",
        linewidth=2,
        label=f"Mean PCC ({baseline_pcc:.3f})",
    )

    plt.ylim(0, max(1.0, plot_df["pcc"].max() + 0.1))
    plt.ylabel("Gene-wise PCC")
    plt.xlabel("Gene")
    plt.title("Target Gene-wise PCC")
    plt.tight_layout()

    path = os.path.join(out_dir, "target_gene_pcc_barplot.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"[INFO] saved target gene PCC barplot: {path}")


def save_all_gene_sorted_pcc_barplot(
    pcc_df: pd.DataFrame,
    target_genes: list[str],
    out_dir: str,
):
    plot_df = pcc_df.sort_values("pcc", ascending=False).reset_index(drop=True).copy()
    plot_df["rank_index"] = np.arange(1, len(plot_df) + 1)

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df["rank_index"], plot_df["pcc"], width=1.0)

    target_df = plot_df[plot_df["gene"].isin(target_genes)]
    plt.scatter(
        target_df["rank_index"],
        target_df["pcc"],
        s=60,
        zorder=3,
    )

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
    plt.title("All Gene-wise PCC Ranking")
    plt.tight_layout()

    path = os.path.join(out_dir, "all_gene_sorted_pcc_ranking.png")
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
    out_dir: str,
):
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    meta_df = pd.DataFrame(meta)

    if "patch_idx" not in meta_df.columns:
        print("[WARN] patch_idx not found in meta. Skip spatial expression map.")
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

        # sample_id별로 따로 저장
        if "sample_id" in df.columns:
            group_iter = df.groupby("sample_id")
        else:
            group_iter = [("all", df)]

        for sample_id, sdf in group_iter:
            # 실제 spatial coordinates 사용
            if (
                "coord_x" in sdf.columns
                and "coord_y" in sdf.columns
                and (sdf["coord_x"] >= 0).all()
            ):
                # x = sdf["coord_x"].values
                # y = sdf["coord_y"].values
                x = sdf["coord_y"].values
                y = sdf["coord_x"].values

            else:
                # fallback: patch_idx 기반 grid
                grid_w = int(np.ceil(np.sqrt(sdf["patch_idx"].max() + 1)))

                x = sdf["patch_idx"].values % grid_w
                y = sdf["patch_idx"].values // grid_w

            vmin = min(sdf["y_true"].min(), sdf["y_pred"].min())
            vmax = max(sdf["y_true"].max(), sdf["y_pred"].max())

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            sc1 = plt.scatter(
                x,
                y,
                c=sdf["y_true"],
                s=PLOT_SPOT_SIZE, 
                marker="h", 
                vmin=vmin,
                vmax=vmax,
            )
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Ground Truth")
            plt.colorbar(sc1, fraction=0.046, pad=0.04)

            plt.subplot(1, 2, 2)
            sc2 = plt.scatter(
                x,
                y,
                c=sdf["y_pred"],
                s=PLOT_SPOT_SIZE,
                marker="h", 
                vmin=vmin,
                vmax=vmax,
            )
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Prediction")
            plt.colorbar(sc2, fraction=0.046, pad=0.04)

            plt.suptitle(f"{gene} | {sample_id} | {title_suffix}")
            plt.tight_layout()

            safe_sample_id = str(sample_id).replace("/", "_")
            path = os.path.join(out_dir, f"{gene}_spatial_gt_vs_pred_{safe_sample_id}.png")
            plt.savefig(path, dpi=300)
            plt.close()

            print(f"[INFO] saved spatial map: {path}")


            # =========================
            # Z-score normalized spatial map
            # =========================
            z_true = zscore_1d(sdf["y_true"].values)
            z_pred = zscore_1d(sdf["y_pred"].values)

            z_vmin, z_vmax = symmetric_zlim(z_true, z_pred, clip=3.0)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            sc1 = plt.scatter(
                x,
                y,
                c=z_true,
                s=PLOT_SPOT_SIZE,
                marker="h",
                vmin=z_vmin,
                vmax=z_vmax,
                # cmap="coolwarm",
            )
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Ground Truth (z-score)")
            plt.colorbar(sc1, fraction=0.046, pad=0.04)

            plt.subplot(1, 2, 2)
            sc2 = plt.scatter(
                x,
                y,
                c=z_pred,
                s=PLOT_SPOT_SIZE,
                marker="h",
                vmin=z_vmin,
                vmax=z_vmax,
                # cmap="coolwarm",
            )
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.axis("off")
            plt.title("Prediction (z-score)")
            plt.colorbar(sc2, fraction=0.046, pad=0.04)

            plt.suptitle(f"{gene} | {sample_id} | z-score normalized | {title_suffix}")
            plt.tight_layout()

            z_path = os.path.join(
                out_dir,
                f"{gene}_spatial_gt_vs_pred_zscore_{safe_sample_id}.png",
            )
            plt.savefig(z_path, dpi=300)
            plt.close()

            print(f"[INFO] saved z-score spatial map: {z_path}")


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


def run_one_fold_analysis(
    fold: int,
    device: torch.device,
    checkpoint_path: str | None = None,
    model_type: str = "rapacl",
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

    preds, targets, meta = predict_gene_expression(model, val_loader, device)

    fold_out_dir = os.path.join(train.OUTPUT_DIR, exp_name, "gene_analysis", f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)

    # 1. 전체 gene별 PCC + rank 저장
    pcc_df = compute_gene_pccs(preds, targets, genes)
    pcc_path = os.path.join(fold_out_dir, "all_gene_pcc_rank.csv")
    pcc_df.to_csv(pcc_path, index=False)
    print(f"[INFO][Fold {fold}] saved PCC rank: {pcc_path}")

    # 2. MKI67, GATA3, CEACAM6 summary 저장
    target_summary = pcc_df[pcc_df["gene"].isin(TARGET_GENES)].copy()
    target_summary = target_summary.sort_values("gene").reset_index(drop=True)

    target_summary_path = os.path.join(fold_out_dir, "target_gene_pcc_rank.csv")
    target_summary.to_csv(target_summary_path, index=False)
    print(f"[INFO][Fold {fold}] saved target summary: {target_summary_path}")

    # 3. MKI67, GATA3, CEACAM6 prediction 저장
    save_target_gene_predictions(
        preds=preds,
        targets=targets,
        meta=meta,
        genes=genes,
        target_genes=TARGET_GENES,
        out_dir=fold_out_dir,
    )

    # 4. Target gene PCC barplot 저장
    save_target_gene_pcc_barplot(
        target_summary=target_summary,
        pcc_df=pcc_df,
        out_dir=fold_out_dir,
    )

    # 5. Spatial expression map GT vs Prediction 저장
    save_spatial_expression_maps(
        preds=preds,
        targets=targets,
        meta=meta,
        genes=genes,
        target_summary=target_summary,
        target_genes=TARGET_GENES,
        out_dir=fold_out_dir,
    )

    # 6. 전체 gene PCC ranking barplot 저장
    save_all_gene_sorted_pcc_barplot(
        pcc_df=pcc_df,
        target_genes=TARGET_GENES,
        out_dir=fold_out_dir,
    )

    return target_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default=None, help="e.g., 0,1,2,3")
    parser.add_argument("--checkpoint", type=str, default=None, help="single checkpoint path")
    parser.add_argument(
        "--model_type",
        type=str,
        default="rapacl",
        choices=["rapacl", "uni_mlp", "densenet_mlp"],
    )
    args = parser.parse_args()

    device = torch.device(train.DEVICE if torch.cuda.is_available() else "cpu")

    if args.folds is None:
        folds = train.SELECT_FOLDS
    else:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    all_rows = []

    for fold in folds:
        summary = run_one_fold_analysis(
            fold=fold,
            device=device,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
        )
        summary["fold"] = fold
        all_rows.append(summary)

    exp_name = get_experiment_name(args.model_type)
    out_dir = os.path.join(train.OUTPUT_DIR, exp_name, "gene_analysis")
    os.makedirs(out_dir, exist_ok=True)

    final_df = pd.concat(all_rows, axis=0).reset_index(drop=True)
    final_path = os.path.join(out_dir, "target_gene_pcc_rank_all_folds.csv")
    final_df.to_csv(final_path, index=False)

    print("\n" + "=" * 80)
    print("[TARGET GENE PCC SUMMARY]")
    print(final_df)
    print(f"Saved to: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
