from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import rapacl.configs.default.train as train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def get_existing_stage1_checkpoint_path(save_dir: str) -> str | None:
    explicit_path = getattr(train, "STAGE1_CHECKPOINT_PATH", None)

    if explicit_path is not None and os.path.exists(explicit_path):
        return explicit_path

    candidates = [
        os.path.join(save_dir, "best_stage1_full_mmcl_recon_cls.pt"),
        os.path.join(save_dir, "best_stage1_mmcl_recon_cls.pt"),
    ]

    for ckpt_path in candidates:
        if os.path.exists(ckpt_path):
            return ckpt_path

    return None


def load_stage1_checkpoint_if_available(
    model: nn.Module,
    save_dir: str,
    device: torch.device,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    if not getattr(train, "LOAD_STAGE1_CHECKPOINT_IF_EXISTS", True):
        return False, None, None

    ckpt_path = get_existing_stage1_checkpoint_path(save_dir)

    if ckpt_path is None:
        return False, None, None

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    print(
        f"[INFO] loaded existing Stage1 checkpoint: {ckpt_path} "
        f"(epoch={ckpt.get('epoch')}, "
        f"stage={ckpt.get('stage_name')}, "
        f"recon_only={ckpt.get('recon_only')})"
    )

    return True, ckpt_path, ckpt

