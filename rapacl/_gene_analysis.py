# python -m rapacl._gene_analysis --folds 0,1,2,3 

from __future__ import annotations

import os
import json
import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import pearsonr

from rapacl.engines.trainer_utils import set_seed
from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.model.rapacl import build_model
from rapacl.model.rapacl_uni import build_uni_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


TARGET_GENES = ["MKI67", "GATA3", "CEACAM6"]


def get_experiment_name() -> str:
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


def build_eval_model(device: torch.device, num_genes: int):
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

        try:
            output = model(image=image, radiomics=radiomics)
        except TypeError:
            try:
                output = model(image, radiomics)
            except TypeError:
                output = model(image)

        pred = get_prediction_from_output(output)

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        batch_size = target.size(0)
        for i in range(batch_size):
            meta = {}
            for key in ["sample_id", "barcode", "patch_idx", "idx"]:
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


def run_one_fold_analysis(fold: int, device: torch.device, checkpoint_path: str | None = None):
    set_seed(train.SEED + fold * 100)

    _, val_split_csv = get_fold_split_paths(fold)

    val_dataset = build_dataset(
        val_split_csv,
        use_image_augmentation=False,
    )

    val_loader, _ = build_loader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        distributed=False,
        pair_augment=False,
    )

    genes = val_dataset.genes
    num_genes = len(genes)

    exp_name = get_experiment_name()
    save_dir = os.path.join(train.OUTPUT_CHECKPOINT_DIR, exp_name, f"fold_{fold}")

    if checkpoint_path is None:
        checkpoint_path = find_best_stage2_checkpoint(save_dir)

    print(f"[INFO][Fold {fold}] checkpoint: {checkpoint_path}")

    model = build_eval_model(device=device, num_genes=num_genes)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO][Fold {fold}] missing keys: {len(missing)}")
    print(f"[INFO][Fold {fold}] unexpected keys: {len(unexpected)}")

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

    return target_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default=None, help="e.g., 0,1,2,3")
    parser.add_argument("--checkpoint", type=str, default=None, help="single checkpoint path")
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
        )
        summary["fold"] = fold
        all_rows.append(summary)

    exp_name = get_experiment_name()
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
