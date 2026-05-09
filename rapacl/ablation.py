from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from rapacl.engines.trainer_utils import (
    set_seed,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    ddp_barrier,
)
from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.model.patchenc.build import build_patch_encoder
import rapacl.configs.default.model_patchenc as patchenc_constants
import rapacl.configs.default.train as train


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PatchGenePredictionModel(nn.Module):
    def __init__(self, patch_encoder, feat_dim, num_genes):
        super().__init__()

        self.patch_encoder = patch_encoder

        self.proj = MLPHead(
            in_dim=feat_dim,
            out_dim=train.PROJECTION_DIM,
            hidden_dim=getattr(train, "PATH_PROJ_HIDDEN_DIM", 512),
            dropout=getattr(train, "HEAD_DROPOUT", 0.1),
        )

        self.gene_head = MLPHead(
            in_dim=train.PROJECTION_DIM,
            out_dim=num_genes,
            hidden_dim=getattr(train, "GENE_HIDDEN_DIM", 512),
            dropout=getattr(train, "HEAD_DROPOUT", 0.1),
        )

    def forward(self, image):
        feat = self.patch_encoder(image)

        if isinstance(feat, dict):
            feat = feat.get("feat", feat.get("embedding"))

        if isinstance(feat, tuple):
            feat = feat[0]

        z = self.proj(feat)
        pred_gene = self.gene_head(z)

        return {
            "patch_feat": feat,
            "patch_z": z,
            "pred_gene": pred_gene,
        }


def get_fold_split_paths(fold: int) -> tuple[str, str]:
    split_dir = os.path.join(train.ROOT_DIR, "splits")
    return (
        os.path.join(split_dir, f"train_{fold}.csv"),
        os.path.join(split_dir, f"test_{fold}.csv"),
    )


def get_batch_image_gene(batch, device):
    image_keys = ["image", "images", "img", "patch", "patch_image"]
    gene_keys = ["gene", "genes", "expr", "expression", "target_gene", "y"]

    image = None
    gene = None

    for k in image_keys:
        if k in batch:
            image = batch[k]
            break

    for k in gene_keys:
        if k in batch:
            gene = batch[k]
            break

    if image is None:
        raise KeyError(f"Image key not found. batch keys: {batch.keys()}")

    if gene is None:
        raise KeyError(f"Gene key not found. batch keys: {batch.keys()}")

    return image.to(device), gene.float().to(device)


def gene_wise_pcc(pred, target, eps=1e-8):
    pred = pred.detach()
    target = target.detach()

    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    numerator = (pred * target).sum(dim=0)
    denominator = torch.sqrt((pred ** 2).sum(dim=0) * (target ** 2).sum(dim=0)) + eps

    pcc = numerator / denominator
    return pcc.mean().item()


def train_one_epoch(model, loader, optimizer, device, epoch, train_sampler=None):
    model.train()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        image, gene = get_batch_image_gene(batch, device)

        out = model(image)
        pred_gene = out["pred_gene"]

        loss = F.mse_loss(pred_gene, gene)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    preds = []
    targets = []

    for batch in loader:
        image, gene = get_batch_image_gene(batch, device)

        out = model(image)
        pred_gene = out["pred_gene"]

        loss = F.mse_loss(pred_gene, gene)

        bs = image.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        preds.append(pred_gene.detach().cpu())
        targets.append(gene.detach().cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    return {
        "val_gene_mse": total_loss / max(total_n, 1),
        "val_genewise_pcc": gene_wise_pcc(preds, targets),
    }


def build_ablation_model(device, num_genes):
    backbone = getattr(
        train,
        "PATCH_BACKBONE",
        getattr(patchenc_constants, "BACKBONE", "densenet121"),
    )

    pretrained = getattr(
        train,
        "PATCH_PRETRAINED",
        getattr(patchenc_constants, "PRETRAINED", True),
    )

    checkpoint_path = getattr(
        train,
        "PATCH_CKPT_PATH",
        getattr(patchenc_constants, "UNI_CKPT_PATH", None),
    )

    patch_encoder, feat_dim = build_patch_encoder(
        backbone=backbone,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )

    if is_main_process():
        print(f"[INFO] Ablation backbone: {backbone}")
        print(f"[INFO] Patch encoder feat_dim: {feat_dim}")

    return PatchGenePredictionModel(
        patch_encoder=patch_encoder,
        feat_dim=feat_dim,
        num_genes=num_genes,
    ).to(device)


def run_one_fold(fold, distributed, rank, local_rank, device):
    set_seed(train.SEED + fold * 100 + rank)

    train_split_csv, val_split_csv = get_fold_split_paths(fold)

    if is_main_process():
        print("\n" + "=" * 80)
        print(f"[ABLATION] Start Fold {fold}")
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

    num_genes = len(train_dataset.genes)

    if is_main_process():
        print(f"[INFO][Fold {fold}] train samples: {len(train_dataset)}")
        print(f"[INFO][Fold {fold}] val samples: {len(val_dataset)}")
        print(f"[INFO][Fold {fold}] num_genes: {num_genes}")

    model = build_ablation_model(device=device, num_genes=num_genes)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(train, "STAGE2_LR", getattr(train, "LR", 1e-4)),
        weight_decay=getattr(train, "WEIGHT_DECAY", 1e-4),
    )

    epochs = getattr(train, "STAGE2_EPOCHS", getattr(train, "MAX_EPOCHS", 50))

    save_dir = os.path.join(
        train.OUTPUT_CHECKPOINT_DIR,
        "ablation_patch_gene",
        f"fold_{fold}",
    )

    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    ddp_barrier()

    best_record = None
    best_pcc = -1e9

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            train_sampler=train_sampler,
        )

        ddp_barrier()

        val_record = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        if is_main_process():
            print(
                f"[Fold {fold}][Epoch {epoch}] "
                f"train_mse={train_loss:.6f} | "
                f"val_mse={val_record['val_gene_mse']:.6f} | "
                f"val_pcc={val_record['val_genewise_pcc']:.4f}"
            )

            if val_record["val_genewise_pcc"] > best_pcc:
                best_pcc = val_record["val_genewise_pcc"]
                best_record = {
                    "epoch": epoch,
                    **val_record,
                }

                model_to_save = model.module if distributed else model
                ckpt_path = os.path.join(save_dir, "best.pt")

                torch.save(
                    {
                        "model": model_to_save.state_dict(),
                        "epoch": epoch,
                        "best_record": best_record,
                    },
                    ckpt_path,
                )

                print(f"[INFO][Fold {fold}] best checkpoint saved: {ckpt_path}")

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

    return best_record


def summarize_cv_results(fold_results):
    pccs = [r["val_genewise_pcc"] for r in fold_results]
    mses = [r["val_gene_mse"] for r in fold_results]

    mean_pcc = sum(pccs) / len(pccs)
    mean_mse = sum(mses) / len(mses)

    if len(pccs) > 1:
        std_pcc = (sum((x - mean_pcc) ** 2 for x in pccs) / (len(pccs) - 1)) ** 0.5
        std_mse = (sum((x - mean_mse) ** 2 for x in mses) / (len(mses) - 1)) ** 0.5
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

    for fold in train.SELECT_FOLDS:
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
            "ablation_patch_gene",
        )
        os.makedirs(result_dir, exist_ok=True)

        result_path = os.path.join(result_dir, "cv_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4)

        print("\n" + "=" * 80)
        print("[FINAL ABLATION RESULT]")

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
