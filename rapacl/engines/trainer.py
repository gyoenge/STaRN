from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from rapacl.engines.data_utils import get_batch_tensor, get_target_label
from rapacl.engines.losses import symmetric_info_nce
from rapacl.engines.metrics import accuracy, compute_genewise_pcc
from rapacl.engines.trainer_utils import (
    freeze_module,
    unwrap_model,
    is_main_process,
    ddp_barrier,
)
from rapacl.model.rapacl import MMCLReconClsModel
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


WARMUP_RECON_EPOCHS = 5
MMCL_LAMBDA = 1.0
RECON_LAMBDA = 1.0
CLS_LAMBDA = 1.0


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def reduce_meter_sum(meter: dict[str, float], device: torch.device) -> dict[str, float]:
    if not is_dist_avail_and_initialized():
        return meter

    keys = list(meter.keys())
    values = torch.tensor(
        [meter[k] for k in keys],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    return {k: values[i].item() for i, k in enumerate(keys)}


def reduce_scalar_sum(value: float, device: torch.device) -> float:
    if not is_dist_avail_and_initialized():
        return value

    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def get_global_num_samples(loader, device: torch.device) -> int:
    local_n = len(loader.sampler) if hasattr(loader, "sampler") else len(loader.dataset)

    if not is_dist_avail_and_initialized():
        return local_n

    t = torch.tensor(local_n, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def train_contrastive_epoch(
    model: MMCLReconClsModel,
    loader,
    optimizer,
    device: torch.device,
    recon_only: bool = False,
):
    model.train()

    raw_model = unwrap_model(model)
    raw_model.pathomics_encoder.eval()

    mmcl_w = getattr(train, "MMCL_LAMBDA", MMCL_LAMBDA)
    recon_w = getattr(train, "RECON_LAMBDA", RECON_LAMBDA)
    cls_w = getattr(train, "CLS_LAMBDA", CLS_LAMBDA)
    temperature = getattr(train, "CONTRASTIVE_TEMPERATURE", 0.07)

    meter = {
        "loss": 0.0,
        "mmcl": 0.0,
        "recon": 0.0,
        "cls": 0.0,
        "acc": 0.0,
    }

    iterator = tqdm(
        loader,
        desc="stage1_train",
        leave=False,
        disable=(not is_main_process()) or (not train.USE_TQDM),
    )

    for batch in iterator:
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_target_label(batch, device)

        out = raw_model.forward_pretrain(
            image=image,
            radiomics=radiomics,
        )

        recon_loss = F.mse_loss(out["pred_radiomics"], radiomics)

        if recon_only:
            mmcl_loss = torch.zeros((), device=device)
            cls_loss = torch.zeros((), device=device)
            loss = recon_loss
        else:
            mmcl_loss = symmetric_info_nce(
                out["path_z"],
                out["rad_contrast_z"],
                temperature=temperature,
            )
            cls_loss = F.cross_entropy(out["pred_class_logits"], target_label)
            loss = mmcl_w * mmcl_loss + recon_w * recon_loss + cls_w * cls_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = image.size(0)
        meter["loss"] += loss.item() * bs
        meter["mmcl"] += mmcl_loss.item() * bs
        meter["recon"] += recon_loss.item() * bs
        meter["cls"] += cls_loss.item() * bs
        meter["acc"] += accuracy(out["pred_class_logits"].detach(), target_label) * bs

    meter = reduce_meter_sum(meter, device)
    n = get_global_num_samples(loader, device)

    return {k: v / n for k, v in meter.items()}


@torch.no_grad()
def eval_contrastive_epoch(
    model: MMCLReconClsModel,
    loader,
    device: torch.device,
):
    model.eval()

    raw_model = unwrap_model(model)

    temperature = getattr(train, "CONTRASTIVE_TEMPERATURE", 0.07)

    mmcl_w = getattr(train, "MMCL_LAMBDA", MMCL_LAMBDA)
    recon_w = getattr(train, "RECON_LAMBDA", RECON_LAMBDA)
    cls_w = getattr(train, "CLS_LAMBDA", CLS_LAMBDA)

    meter = {
        "loss": 0.0,
        "mmcl": 0.0,
        "recon": 0.0,
        "cls": 0.0,
        "acc": 0.0,
    }

    iterator = tqdm(
        loader,
        desc="stage1_val",
        leave=False,
        disable=not is_main_process(),
    )

    for batch in iterator:
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_target_label(batch, device)

        out = raw_model.forward_pretrain(
            image=image,
            radiomics=radiomics,
        )

        mmcl_loss = symmetric_info_nce(
            out["path_z"],
            out["rad_contrast_z"],
            temperature=temperature,
        )
        recon_loss = F.mse_loss(out["pred_radiomics"], radiomics)
        cls_loss = F.cross_entropy(out["pred_class_logits"], target_label)

        loss = mmcl_w * mmcl_loss + recon_w * recon_loss + cls_w * cls_loss

        bs = image.size(0)
        meter["loss"] += loss.item() * bs
        meter["mmcl"] += mmcl_loss.item() * bs
        meter["recon"] += recon_loss.item() * bs
        meter["cls"] += cls_loss.item() * bs
        meter["acc"] += accuracy(out["pred_class_logits"], target_label) * bs

    meter = reduce_meter_sum(meter, device)
    n = get_global_num_samples(loader, device)

    return {k: v / n for k, v in meter.items()}


def set_gene_eval_trainable(model: MMCLReconClsModel):
    raw_model = unwrap_model(model)

    freeze_module(raw_model.radiomics_model)
    freeze_module(raw_model.recon_head)
    freeze_module(raw_model.cls_head)

    raw_model.pathomics_encoder.train()
    raw_model.pathomics_proj.train()
    raw_model.gene_head.train()

    for p in raw_model.pathomics_encoder.parameters():
        p.requires_grad_(True)

    for p in raw_model.pathomics_proj.parameters():
        p.requires_grad_(True)

    for p in raw_model.gene_head.parameters():
        p.requires_grad_(True)


def train_gene_epoch(
    model: MMCLReconClsModel,
    loader,
    optimizer,
    device: torch.device,
):
    set_gene_eval_trainable(model)

    raw_model = unwrap_model(model)

    meter = {"mse": 0.0}

    iterator = tqdm(
        loader,
        desc="stage2_gene_train",
        leave=False,
        disable=not is_main_process(),
    )

    for batch in iterator:
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        out = raw_model.forward_gene(
            image=image,
            radiomics=radiomics,
        )

        loss = F.mse_loss(out["pred_gene"], gene)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        meter["mse"] += loss.item() * image.size(0)

    meter = reduce_meter_sum(meter, device)
    n = get_global_num_samples(loader, device)

    return {k: v / n for k, v in meter.items()}


@torch.no_grad()
def eval_gene_epoch(
    model: MMCLReconClsModel,
    loader,
    device: torch.device,
):
    model.eval()

    raw_model = unwrap_model(model)

    mse_sum = 0.0
    preds = []
    targets = []

    iterator = tqdm(
        loader,
        desc="stage2_gene_val",
        leave=False,
        disable=not is_main_process(),
    )

    for batch in iterator:
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        out = raw_model.forward_gene(
            image=image,
            radiomics=radiomics,
        )

        pred = out["pred_gene"]

        mse_sum += F.mse_loss(pred, gene, reduction="sum").item()

        preds.append(pred.detach().cpu())
        targets.append(gene.detach().cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)

    mean_pcc, pcc_per_gene = compute_genewise_pcc(pred_all, target_all)

    mse_sum = reduce_scalar_sum(mse_sum, device)
    local_numel = target_all.numel()
    global_numel = reduce_scalar_sum(float(local_numel), device)

    mse = mse_sum / global_numel

    return {
        "mse": mse,
        "mean_pcc": mean_pcc,
        "pcc_per_gene": pcc_per_gene,
    }


def run_stage1(
    model: MMCLReconClsModel,
    train_loader,
    val_loader,
    device: torch.device,
    save_dir: str,
    train_sampler=None,
    is_distributed: bool = False,
) -> str:
    raw_model = unwrap_model(model)

    best_stage1_path = os.path.join(save_dir, "best_stage1_mmcl_recon_cls.pt")
    best_stage1_full_path = os.path.join(save_dir, "best_stage1_full_mmcl_recon_cls.pt")

    stage1_params = (
        list(raw_model.radiomics_model.parameters())
        + list(raw_model.pathomics_proj.parameters())
        + list(raw_model.recon_head.parameters())
        + list(raw_model.cls_head.parameters())
    )

    optimizer_stage1 = torch.optim.AdamW(
        stage1_params,
        lr=getattr(train, "LR", 1e-4),
        weight_decay=getattr(train, "WEIGHT_DECAY", 1e-4),
    )

    best_stage1_val = float("inf")
    best_stage1_full_val = float("inf")

    stage1_epochs = getattr(train, "PRETRAIN_EPOCHS", getattr(train, "EPOCHS", 20))
    warmup_recon_epochs = getattr(train, "WARMUP_RECON_EPOCHS", WARMUP_RECON_EPOCHS)

    if is_main_process():
        print(f"[INFO] Stage1 recon warmup epochs: {warmup_recon_epochs}")

    for epoch in range(stage1_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        recon_only = epoch < warmup_recon_epochs
        stage_name = "ReconWarmup" if recon_only else "MMCLReconCls"

        train_m = train_contrastive_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_stage1,
            device=device,
            recon_only=recon_only,
        )

        val_m = eval_contrastive_epoch(
            model=model,
            loader=val_loader,
            device=device,
        )

        if is_main_process():
            print(
                f"[Stage1:{stage_name}][Epoch {epoch}] "
                f"train_loss={train_m['loss']:.4f} "
                f"mmcl={train_m['mmcl']:.4f} "
                f"recon={train_m['recon']:.4f} "
                f"cls={train_m['cls']:.4f} "
                f"acc={train_m['acc']:.4f} | "
                f"val_loss={val_m['loss']:.4f} "
                f"mmcl={val_m['mmcl']:.4f} "
                f"recon={val_m['recon']:.4f} "
                f"cls={val_m['cls']:.4f} "
                f"acc={val_m['acc']:.4f}"
            )

        if is_main_process() and val_m["loss"] < best_stage1_val:
            best_stage1_val = val_m["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "stage_name": stage_name,
                    "recon_only": recon_only,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer_stage1.state_dict(),
                    "val_metrics": val_m,
                },
                best_stage1_path,
            )

        if is_main_process() and (not recon_only) and val_m["loss"] < best_stage1_full_val:
            best_stage1_full_val = val_m["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "stage_name": stage_name,
                    "recon_only": recon_only,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer_stage1.state_dict(),
                    "val_metrics": val_m,
                },
                best_stage1_full_path,
            )

        if is_distributed:
            ddp_barrier()

    if is_distributed:
        ddp_barrier()

    load_path = (
        best_stage1_full_path
        if os.path.exists(best_stage1_full_path)
        else best_stage1_path
    )

    if not os.path.exists(load_path):
        raise FileNotFoundError("No Stage1 checkpoint found.")

    return load_path


def run_stage2(
    model: MMCLReconClsModel,
    train_loader,
    val_loader,
    device: torch.device,
    save_dir: str,
    stage1_ckpt_path: str,
    train_sampler=None,
    is_distributed: bool = False,
):
    raw_model = unwrap_model(model)

    ckpt = torch.load(stage1_ckpt_path, map_location=device)
    raw_model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if is_main_process():
        print(f"[INFO] loaded Stage1 checkpoint for gene eval: {stage1_ckpt_path}")

    set_gene_eval_trainable(model)

    optimizer_stage2 = torch.optim.AdamW(
        [
            {
                "params": raw_model.gene_head.parameters(),
                "lr": getattr(train, "GENE_LR", 1e-4),
            },
            {
                "params": raw_model.pathomics_proj.parameters(),
                "lr": getattr(train, "PATH_PROJ_LR", 1e-4),
            },
            {
                "params": raw_model.pathomics_encoder.parameters(),
                "lr": getattr(train, "PATH_ENCODER_LR", 1e-4),
            },
        ],
        weight_decay=getattr(train, "GENE_WEIGHT_DECAY", train.WEIGHT_DECAY),
    )

    best_pcc = -float("inf")
    best_record: dict[str, Any] | None = None

    stage2_epochs = getattr(train, "GENE_EPOCHS", getattr(train, "EPOCHS", 50))

    for epoch in range(stage2_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_gene_m = train_gene_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_stage2,
            device=device,
        )

        val_gene_m = eval_gene_epoch(
            model=model,
            loader=val_loader,
            device=device,
        )

        val_pcc = val_gene_m["mean_pcc"]

        if is_main_process():
            print(
                f"[Stage2][Epoch {epoch}] "
                f"train_gene_mse={train_gene_m['mse']:.6f} | "
                f"val_gene_mse={val_gene_m['mse']:.6f} "
                f"val_genewise_PCC={val_pcc:.4f} "
                f"best_PCC={best_pcc:.4f}"
            )

        if is_main_process() and val_pcc > best_pcc:
            best_pcc = val_pcc

            best_record = {
                "epoch": epoch,
                "val_gene_mse": val_gene_m["mse"],
                "val_genewise_pcc": val_pcc,
            }

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "gene_head_state_dict": raw_model.gene_head.state_dict(),
                    "best_record": best_record,
                    "feature_cols": RADIOMICS_FEATURES_NAMES,
                    "pcc_per_gene": val_gene_m["pcc_per_gene"],
                },
                os.path.join(save_dir, "best_stage2_genepred.pt"),
            )

            print(f"[INFO] saved best gene model: PCC={best_pcc:.4f}")

        if is_distributed:
            ddp_barrier()

    if is_main_process():
        print("\n========== Final Result ==========")
        print(best_record)

    return best_record
