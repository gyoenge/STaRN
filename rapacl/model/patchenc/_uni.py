from __future__ import annotations

import torch
import timm
from torchvision import transforms


def build_uni(checkpoint_path: str | None = None):
    """
    UNI ViT-L/16 encoder

    Returns:
        model: UNI encoder
        feature_dim: output embedding dimension (1024)
        transform: image transform
    """

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )

    if checkpoint_path is not None:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        # huggingface downloaded pytorch_model.bin
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=True,
        )

        print(f"[UNI] loaded checkpoint: {checkpoint_path}")
        print(f"[UNI] missing keys: {missing}")
        print(f"[UNI] unexpected keys: {unexpected}")

    feature_dim = model.num_features  # 1024

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    model.eval()

    return model, feature_dim, transform


