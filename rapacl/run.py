from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True"
)

import os

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


def main():
    distributed, rank, local_rank, world_size = setup_ddp()

    set_seed(train.SEED + rank)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(train.DEVICE)

    if is_main_process():
        print(f"[INFO] distributed: {distributed}")
        print(f"[INFO] world_size: {world_size}")
        print(f"[INFO] device: {device}")

    train_dataset = build_dataset(
        train.TRAIN_SPLIT_CSV,
        use_image_augmentation=train.USE_IMAGE_AUGMENTATION,
    )

    val_dataset = build_dataset(
        train.VAL_SPLIT_CSV,
        use_image_augmentation=False,
    )

    train_loader, train_sampler = build_loader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        distributed=distributed,
        pair_augment=train.USE_PAIR_AUGMENT_BATCH,
    )

    val_loader, val_sampler = build_loader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        distributed=distributed,
        pair_augment=False,
    )

    if is_main_process():
        print(f"[INFO] train samples: {len(train_dataset)}")
        print(f"[INFO] val samples: {len(val_dataset)}")

    num_genes = len(train_dataset.genes)
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    if is_main_process():
        print(f"[INFO] num_genes: {num_genes}")
        print(f"[INFO] num_radiomics_features: {num_radiomics_features}")

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

    run_stage2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        stage1_ckpt_path=stage1_ckpt_path,
        train_sampler=train_sampler,
        is_distributed=distributed,
    )

    cleanup_ddp()


if __name__ == "__main__":
    main()
