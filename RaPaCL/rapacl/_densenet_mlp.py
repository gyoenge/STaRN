# OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 -m rapacl._densenet_mlp 2>&1 | tee log_densenet_mlp.log
# nohup bash -c 'OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 -m rapacl._densenet_mlp' > log_densenet_mlp.log 2>&1 &

from __future__ import annotations

from datetime import datetime
import json
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
)

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from rapacl.model.rapacl import MLPHead
from rapacl.model.patchenc.build import build_patch_encoder
from rapacl.engines.trainer_utils import (
    set_seed,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    ddp_barrier,
)
from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.engines.metrics import compute_genewise_pcc
import rapacl.configs.default.train as train


class DenseNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        encoder, feat_dim = build_patch_encoder(
            backbone="densenet121",
            pretrained=True,
        )

        self.encoder = encoder
        self.out_dim = feat_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class DenseNetMLPGeneModel(nn.Module):
    def __init__(self, num_genes: int):
        super().__init__()

        self.pathomics_encoder = DenseNetEncoder()

        self.gene_head = MLPHead(
            in_dim=self.pathomics_encoder.out_dim,
            out_dim=num_genes,
            hidden_dim=train.GENE_HIDDEN_DIM,
            dropout=train.HEAD_DROPOUT,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        path_feat = self.pathomics_encoder(image)
        gene_pred = self.gene_head(path_feat)
        return gene_pred


def build_densenet_mlp_model(
    device: torch.device,
    num_genes: int,
) -> DenseNetMLPGeneModel:
    model = DenseNetMLPGeneModel(num_genes=num_genes)
    return model.to(device)


def get_experiment_name() -> str:
    return "densenet121_mlp"


def get_fold_split_paths(fold: int) -> tuple[str, str]:
    split_dir = os.path.join(train.ROOT_DIR, "splits")
    return (
        os.path.join(split_dir, f"train_{fold}.csv"),
        os.path.join(split_dir, f"test_{fold}.csv"),
    )


def unpack_batch(batch, device: torch.device):
    image = batch["image"].to(device, non_blocking=True).float()
    gene = batch["gene"].to(device, non_blocking=True).float()
    return image, gene


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    train_sampler=None,
    is_distributed: bool = False,
):
    model.train()

    if is_distributed and train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_count = 0

    iterator = loader

    for batch in iterator:
        image, gene = unpack_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(image)
        loss = criterion(pred, gene)

        loss.backward()
        optimizer.step()

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    epoch: int,
    distributed: bool = False,
):
    model.eval()

    total_loss = 0.0
    total_count = 0

    preds = []
    targets = []

    iterator = loader

    for batch in iterator:
        image, gene = unpack_batch(batch, device)

        pred = model(image)
        loss = criterion(pred, gene)

        batch_size = image.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        preds.append(pred.detach())
        targets.append(gene.detach())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    loss_tensor = torch.tensor(
        [total_loss, total_count],
        device=device,
        dtype=torch.float32,
    )

    if distributed:
        torch.distributed.all_reduce(
            loss_tensor,
            op=torch.distributed.ReduceOp.SUM,
        )

        gathered_preds = [
            torch.zeros_like(preds)
            for _ in range(torch.distributed.get_world_size())
        ]
        gathered_targets = [
            torch.zeros_like(targets)
            for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_preds, preds)
        torch.distributed.all_gather(gathered_targets, targets)

        preds = torch.cat(gathered_preds, dim=0)
        targets = torch.cat(gathered_targets, dim=0)

    val_mse = (loss_tensor[0] / loss_tensor[1].clamp_min(1)).item()

    preds = preds.cpu()
    targets = targets.cpu()

    pcc_result = compute_genewise_pcc(preds, targets)

    if isinstance(pcc_result, tuple):
        val_pcc = pcc_result[0]
        per_gene_pcc = pcc_result[1]
    else:
        val_pcc = pcc_result
        per_gene_pcc = None

    return {
        "val_gene_mse": val_mse,
        "val_genewise_pcc": float(val_pcc),
        "per_gene_pcc": per_gene_pcc,
    }


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
        pair_augment=False,
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
        print("[INFO] model: Trainable DenseNet121 + MLP gene head")

    model = build_densenet_mlp_model(
        device=device,
        num_genes=num_genes,
    )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    exp_name = get_experiment_name()
    save_dir = os.path.join(
        train.OUTPUT_CHECKPOINT_DIR,
        exp_name,
        f"fold_{fold}",
    )

    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    ddp_barrier()

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.GENE_LR if hasattr(train, "GENE_LR") else train.LR,
        weight_decay=train.WEIGHT_DECAY_STAGE2,
    )

    best_record = None
    best_pcc = -1e9

    max_epochs = (
        train.STAGE2_EPOCHS
        if hasattr(train, "STAGE2_EPOCHS")
        else train.MAX_EPOCHS
    )

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            train_sampler=train_sampler,
            is_distributed=distributed,
        )

        val_record = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            distributed=distributed,
        )

        val_mse = val_record["val_gene_mse"]
        val_pcc = val_record["val_genewise_pcc"]

        if is_main_process():
            print(
                f"[{now()}] "
                f"[Fold {fold}][Epoch {epoch}] "
                f"train_gene_mse={train_loss:.6f} | "
                f"val_gene_mse={val_mse:.6f} | "
                f"val_genewise_PCC={val_pcc:.4f}"
            )

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            best_record = {
                "epoch": epoch,
                "val_gene_mse": val_mse,
                "val_genewise_pcc": val_pcc,
            }

            if is_main_process():
                ckpt_path = os.path.join(save_dir, "best_densenet121_mlp.pt")
                state_dict = (
                    model.module.state_dict()
                    if distributed
                    else model.state_dict()
                )

                torch.save(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
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
            sum((x - mean_pcc) ** 2 for x in pccs)
            / (len(pccs) - 1)
        ) ** 0.5
        std_mse = (
            sum((x - mean_mse) ** 2 for x in mses)
            / (len(mses) - 1)
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
        print("[INFO] experiment: densenet121_mlp")

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
            get_experiment_name(),
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
