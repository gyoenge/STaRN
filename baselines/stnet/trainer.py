from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader
from torchvision import transforms

from baselines.common.metrics import compute_genewise_pcc
from baselines.common.optimizer import build_optimizer
from .dataset import STNetDataset
from .stnet import build_model


def build_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    optimizer,
    criterion,
) -> float:
    model.train()
    epoch_loss = 0.0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(train_loader), 1)


def eval_fold(model: torch.nn.Module, test_loader, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return compute_genewise_pcc(all_targets, all_preds)


def select_best_epoch(
    train_df,
    bench_data_root: str,
    gene_list_path: str,
    device: torch.device,
    cfg: dict,
    logger=None,
):
    """Inner LOO CV to pick the epoch with the best mean validation PCC."""
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    num_genes = model_cfg.get("num_genes", 250)
    pretrained = model_cfg.get("pretrained", True)
    max_epochs = train_cfg.get("max_epochs", 50)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)

    loo = LeaveOneOut()
    fold_indices = list(loo.split(train_df))
    epoch_scores = {epoch: [] for epoch in range(1, max_epochs + 1)}

    for inner_fold, (tr_idx, val_idx) in enumerate(fold_indices):
        if logger:
            logger.info("\n  [Inner Fold %d/%d]", inner_fold + 1, len(fold_indices))

        inner_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        inner_train_dataset = STNetDataset(
            bench_data_root=bench_data_root,
            gene_list_path=gene_list_path,
            split_df=inner_train_df,
            transforms=build_train_transform(),
        )
        inner_val_dataset = STNetDataset(
            bench_data_root=bench_data_root,
            gene_list_path=gene_list_path,
            split_df=inner_val_df,
            transforms=None,
        )

        inner_train_loader = build_dataloader(
            inner_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        inner_val_loader = build_dataloader(
            inner_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        model = build_model({
            "num_genes": num_genes,
            "pretrained": pretrained,
            "backbone": "densenet121",
        }).to(device)

        criterion = nn.MSELoss()
        optimizer = build_optimizer(model.parameters(), cfg)

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                train_loader=inner_train_loader,
                device=device,
                optimizer=optimizer,
                criterion=criterion,
            )
            val_pearson, _ = eval_fold(model, inner_val_loader, device)
            epoch_scores[epoch].append(val_pearson)

            if logger:
                logger.info(
                    "    Epoch %02d/%d | Train Loss: %.4f | Val Pearson: %.4f",
                    epoch, max_epochs, train_loss, val_pearson,
                )

    mean_epoch_scores = {
        epoch: float(np.mean(scores)) for epoch, scores in epoch_scores.items()
    }
    best_epoch = max(mean_epoch_scores, key=mean_epoch_scores.get)
    best_score = mean_epoch_scores[best_epoch]

    if logger:
        logger.info("\n  [Epoch Selection Summary]")
        for epoch in range(1, max_epochs + 1):
            logger.info(
                "    Epoch %02d: Mean Val Pearson = %.4f", epoch, mean_epoch_scores[epoch]
            )
        logger.info(
            "  >>> Selected Best Epoch = %d (Mean Val Pearson = %.4f)", best_epoch, best_score
        )

    return best_epoch, mean_epoch_scores


def retrain_full_train(
    train_df,
    bench_data_root: str,
    gene_list_path: str,
    device: torch.device,
    num_epochs: int,
    cfg: dict,
    logger=None,
):
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    num_genes = model_cfg.get("num_genes", 250)
    pretrained = model_cfg.get("pretrained", True)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)

    full_train_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_df=train_df,
        transforms=build_train_transform(),
    )
    full_train_loader = build_dataloader(
        full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    model = build_model({
        "num_genes": num_genes,
        "backbone": "densenet121",
        "pretrained": pretrained,
    }).to(device)

    criterion = nn.MSELoss()
    optimizer = build_optimizer(model.parameters(), cfg)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=full_train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
        )
        if logger:
            logger.info(
                "  [Retrain] Epoch %02d/%d | Train Loss: %.4f", epoch, num_epochs, train_loss
            )

    return model
