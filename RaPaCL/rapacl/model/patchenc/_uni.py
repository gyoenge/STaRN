from __future__ import annotations

import os

import torch
import torch.distributed as dist
import timm
from torchvision import transforms
from huggingface_hub import hf_hub_download

import rapacl.configs.default.model_patchenc as config


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_rank0() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _barrier():
    if _is_dist_initialized():
        dist.barrier()


def _download_uni_checkpoint_if_missing(checkpoint_path: str) -> None:
    if os.path.isfile(checkpoint_path):
        return

    if _is_rank0():
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        print(f"[UNI] checkpoint not found: {checkpoint_path}")
        print("[UNI] downloading from HuggingFace: MahmoodLab/UNI")

        hf_hub_download(
            repo_id="MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=os.path.dirname(checkpoint_path),
            local_dir_use_symlinks=False,
            force_download=False,
        )

        downloaded_path = os.path.join(
            os.path.dirname(checkpoint_path),
            "pytorch_model.bin",
        )

        if downloaded_path != checkpoint_path:
            os.replace(downloaded_path, checkpoint_path)

        print(f"[UNI] downloaded checkpoint to: {checkpoint_path}")

    _barrier()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"[UNI] checkpoint file not found after download: {checkpoint_path}"
        )


def build_uni():
    model = timm.create_model(
        config.UNI_VERSION,
        img_size=config.UNI_IMG_SIZE,
        patch_size=config.UNI_PATCH_SIZE,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )

    checkpoint_path = config.UNI_CKPT_PATH

    if not checkpoint_path:
        raise ValueError("[UNI] UNI_CKPT_PATH is not set.")

    _download_uni_checkpoint_if_missing(checkpoint_path)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if _is_rank0():
        print(f"[UNI] loaded checkpoint: {checkpoint_path}")
        print(f"[UNI] missing keys: {missing}")
        print(f"[UNI] unexpected keys: {unexpected}")

    feature_dim = model.num_features

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=(0.485, 0.456, 0.406),
    #             std=(0.229, 0.224, 0.225),
    #         ),
    #     ]
    # )

    model.eval()

    return model, feature_dim #, transform
