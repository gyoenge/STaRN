from __future__ import annotations

import torch.optim as optim


def build_optimizer(parameters, cfg: dict) -> optim.Optimizer:
    train_cfg = cfg["train"]
    optimizer_name = str(train_cfg.get("optimizer_name", "adamw")).lower()
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    momentum = float(train_cfg.get("momentum", 0.9))

    if optimizer_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    raise ValueError(
        f"Unsupported optimizer_name: '{optimizer_name}'. Choose from: sgd, adam, adamw."
    )
