from __future__ import annotations

import os

import torch

from rapacl.engines.trainer_utils import set_seed
from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.engines.trainer import run_stage1, run_stage2
from rapacl.model.rapacl import build_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


def main():
    set_seed(train.SEED)

    device = torch.device(train.DEVICE)
    print(f"[INFO] device: {device}")

    trainset = build_dataset(train.TRAIN_SPLIT_CSV)
    valset = build_dataset(train.VAL_SPLIT_CSV)

    train_loader = build_loader(
        trainset,
        shuffle=True,
        drop_last=False,
    )

    val_loader = build_loader(
        valset,
        shuffle=False,
        drop_last=False,
    )

    print(f"[INFO] train samples: {len(trainset)}")
    print(f"[INFO] val samples: {len(valset)}")

    num_genes = len(trainset.genes)
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    print(f"[INFO] num_genes: {num_genes}")
    print(f"[INFO] num_radiomics_features: {num_radiomics_features}")

    model = build_model(
        device=device,
        num_genes=num_genes,
        num_radiomics_features=num_radiomics_features,
    )

    save_dir = os.path.join(
        train.OUTPUT_CHECKPOINT_DIR,
        "rapacl_baseline",
    )
    os.makedirs(save_dir, exist_ok=True)

    stage1_ckpt_path = run_stage1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
    )

    run_stage2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        stage1_ckpt_path=stage1_ckpt_path,
    )


if __name__ == "__main__":
    main()