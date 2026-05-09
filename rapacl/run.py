from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
)

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from rapacl.engines.trainer_utils import (
    set_seed,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    ddp_barrier,
)
from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.engines.trainer import run_stage1, run_stage2
from rapacl.model.rapacl import build_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


def get_fold_split_paths(fold: int) -> tuple[str, str]:
    split_dir = os.path.join(train.ROOT_DIR, "splits")
    train_split_csv = os.path.join(split_dir, f"train_{fold}.csv")
    val_split_csv = os.path.join(split_dir, f"test_{fold}.csv")

    return train_split_csv, val_split_csv


def run_one_fold(
    fold: int,
    distributed: bool,
    rank: int,
    local_rank: int,
    device: torch.device,
) -> dict:
    set_seed(train.SEED + fold * 100 + rank)

    train_split_csv, val_split_csv = get_fold_split_paths(fold)

    if is_main_process():
        print("\n" + "=" * 80)
        print(f"[INFO] Start Fold {fold}")
        print(f"[INFO][Fold {fold}] train split: {train_split_csv}")
        print(f"[INFO][Fold {fold}] val split  : {val_split_csv}")
        print("=" * 80)

    train_dataset = build_dataset(
        train_split_csv,
        use_image_augmentation=train.USE_IMAGE_AUGMENTATION,
    )

    val_dataset = build_dataset(
        val_split_csv,
        use_image_augmentation=False,
    )

    train_loader, train_sampler = build_loader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        distributed=distributed,
        pair_augment=train.USE_PAIR_AUGMENT_BATCH,
    )

    val_loader, _ = build_loader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        distributed=distributed,
        pair_augment=False,
    )

    if is_main_process():
        print(f"[INFO][Fold {fold}] train samples: {len(train_dataset)}")
        print(f"[INFO][Fold {fold}] val samples: {len(val_dataset)}")

    num_genes = len(train_dataset.genes)
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    if is_main_process():
        print(f"[INFO][Fold {fold}] num_genes: {num_genes}")
        print(f"[INFO][Fold {fold}] num_radiomics_features: {num_radiomics_features}")

    model = build_model(
        device=device,
        num_genes=num_genes,
        num_radiomics_features=num_radiomics_features,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    save_dir = os.path.join(
        train.OUTPUT_CHECKPOINT_DIR,
        "rapacl_baseline",
        f"fold_{fold}",
    )

    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    ddp_barrier()

    stage1_ckpt_path = run_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        train_sampler=train_sampler,
        is_distributed=distributed,
    )

    ddp_barrier()

    best_record = run_stage2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        stage1_ckpt_path=stage1_ckpt_path,
        train_sampler=train_sampler,
        is_distributed=distributed,
    )

    ddp_barrier()

    if best_record is None:
        best_record = {
            "epoch": -1,
            "val_gene_mse": float("nan"),
            "val_genewise_pcc": float("nan"),
        }

    if is_main_process():
        fold_result = {
            "fold": fold,
            **best_record,
        }

        result_path = os.path.join(save_dir, "fold_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=4)

        print(f"[INFO][Fold {fold}] result saved to: {result_path}")
        print(f"[INFO][Fold {fold}] best_record: {fold_result}")

    if distributed:
        torch.cuda.empty_cache()

    return best_record


def summarize_cv_results(fold_results: list[dict]) -> dict:
    pccs = [r["val_genewise_pcc"] for r in fold_results]
    mses = [r["val_gene_mse"] for r in fold_results]

    mean_pcc = sum(pccs) / len(pccs)
    mean_mse = sum(mses) / len(mses)

    if len(pccs) > 1:
        std_pcc = (
            sum((x - mean_pcc) ** 2 for x in pccs) / (len(pccs) - 1)
        ) ** 0.5
        std_mse = (
            sum((x - mean_mse) ** 2 for x in mses) / (len(mses) - 1)
        ) ** 0.5
    else:
        std_pcc = 0.0
        std_mse = 0.0

    return {
        "fold_results": fold_results,
        "mean_pcc": mean_pcc,
        "std_pcc": std_pcc,
        "mean_mse": mean_mse,
        "std_mse": std_mse,
    }


def main():
    distributed, rank, local_rank, world_size = setup_ddp()

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(train.DEVICE)

    if is_main_process():
        print(f"[INFO] distributed: {distributed}")
        print(f"[INFO] world_size: {world_size}")
        print(f"[INFO] device: {device}")

    fold_results = []

    selected_folds = train.SELECT_FOLDS

    for fold in selected_folds:
        best_record = run_one_fold(
            fold=fold,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            device=device,
        )

        if is_main_process():
            fold_results.append(
                {
                    "fold": fold,
                    **best_record,
                }
            )

        ddp_barrier()
        torch.cuda.empty_cache()

    if is_main_process():
        final_result = summarize_cv_results(fold_results)

        result_dir = os.path.join(
            train.OUTPUT_CHECKPOINT_DIR,
            "rapacl_baseline",
        )
        os.makedirs(result_dir, exist_ok=True)

        result_path = os.path.join(result_dir, "cv_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4)

        print("\n" + "=" * 80)
        print("[FINAL CV RESULT]")

        for r in final_result["fold_results"]:
            print(
                f"Fold {r['fold']} | "
                f"best_epoch={r['epoch']} | "
                f"PCC={r['val_genewise_pcc']:.4f} | "
                f"MSE={r['val_gene_mse']:.6f}"
            )

        print("-" * 80)
        print(f"Mean PCC: {final_result['mean_pcc']:.4f}")
        print(f"Std PCC : {final_result['std_pcc']:.4f}")
        print(f"Mean MSE: {final_result['mean_mse']:.6f}")
        print(f"Std MSE : {final_result['std_mse']:.6f}")
        print(f"Saved to: {result_path}")
        print("=" * 80)

    cleanup_ddp()


if __name__ == "__main__":
    main()