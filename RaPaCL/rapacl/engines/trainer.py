from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from rapacl.engines.data_utils import get_batch_tensor, get_target_label
from rapacl.engines.losses import compute_mmcl_loss
from rapacl.engines.metrics import (
    accuracy,
    compute_genewise_pcc,
    uniformity,
    effective_rank,
    celltype_separability,
    multiclass_auroc_auprc,
)
from rapacl.engines.trainer_utils import (
    freeze_module,
    unwrap_model,
    is_main_process,
    ddp_barrier,
)
from rapacl.model.rapacl import MMCLReconClsModel
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def get_scheduled_mmcl_weight(
    epoch: int,
    warmup_epochs: int,
    max_weight: float,
) -> float:
    """
    Slowly ramps up MMCL weight after reconstruction warmup.

    Default:
    - epoch < warmup_epochs: 0.0
    - after warmup: cosine ramp-up
    """
    rampup_epochs = train.MMCL_RAMPUP_EPOCHS

    if epoch < warmup_epochs:
        return 0.0

    progress = (epoch - warmup_epochs + 1) / max(rampup_epochs, 1)
    progress = min(max(progress, 0.0), 1.0)

    return max_weight * 0.5 * (1.0 - math.cos(math.pi * progress))


def init_stage1_meter() -> dict[str, float]:
    return {
        "loss": 0.0,
        "mmcl": 0.0,
        "recon": 0.0,
        "cls": 0.0,
        "acc": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "uni_path": 0.0,
        "uni_rad": 0.0,
        "rank_path": 0.0,
        "rank_rad": 0.0,
        "sep_path": 0.0,
        "sep_rad": 0.0,
    }


def update_geometry_meter(
    meter: dict[str, float],
    path_z: torch.Tensor,
    rad_z: torch.Tensor,
    target_label: torch.Tensor,
    bs: int,
) -> None:
    path_z = path_z.detach()
    rad_z = rad_z.detach()
    target_label = target_label.detach()

    meter["uni_path"] += uniformity(path_z) * bs
    meter["uni_rad"] += uniformity(rad_z) * bs

    meter["rank_path"] += effective_rank(path_z) * bs
    meter["rank_rad"] += effective_rank(rad_z) * bs

    meter["sep_path"] += celltype_separability(path_z, target_label)["separation"] * bs
    meter["sep_rad"] += celltype_separability(rad_z, target_label)["separation"] * bs


def finalize_stage1_meter(
    meter: dict[str, float],
    all_logits: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    device: torch.device,
    loader,
) -> dict[str, float]:
    if len(all_logits) > 0:
        logits_all = torch.cat(all_logits, dim=0)
        targets_all = torch.cat(all_targets, dim=0)

        cls_metrics = multiclass_auroc_auprc(logits_all, targets_all)
        local_n = logits_all.size(0)

        meter["auroc"] = cls_metrics["auroc"] * local_n
        meter["auprc"] = cls_metrics["auprc"] * local_n

    meter = reduce_meter_sum(meter, device)
    n = get_global_num_samples(loader, device)

    return {k: v / n for k, v in meter.items()}


def train_contrastive_epoch(
    model: MMCLReconClsModel,
    loader,
    optimizer,
    device: torch.device,
    recon_only: bool = False,
    mmcl_w: float | None = None,
):
    model.train()

    raw_model = unwrap_model(model)
    raw_model.pathomics_encoder.eval()

    if mmcl_w is None:
        mmcl_w = train.MMCL_LAMBDA
    recon_w = train.RECON_LAMBDA
    cls_w = train.CLS_LAMBDA
    temperature = train.CONTRASTIVE_TEMPERATURE

    meter = init_stage1_meter()
    all_logits = []
    all_targets = []

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
            sample_idx = batch.get("idx", None)

            if sample_idx is None:
                sample_idx = torch.arange(image.size(0), device=device)
            elif not isinstance(sample_idx, torch.Tensor):
                sample_idx = torch.tensor(sample_idx, device=device, dtype=torch.long)
            else:
                sample_idx = sample_idx.to(device=device, dtype=torch.long)

            mmcl_loss = compute_mmcl_loss(
                out=out,
                idxes=sample_idx,
                loss_type=train.MMCL_LOSS,
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

        all_logits.append(out["pred_class_logits"].detach().cpu())
        all_targets.append(target_label.detach().cpu())

        update_geometry_meter(
            meter=meter,
            path_z=out["path_z"],
            rad_z=out["rad_contrast_z"],
            target_label=target_label,
            bs=bs,
        )

    return finalize_stage1_meter(
        meter=meter,
        all_logits=all_logits,
        all_targets=all_targets,
        device=device,
        loader=loader,
    )


@torch.no_grad()
@torch.no_grad()
def eval_contrastive_epoch(
    model: MMCLReconClsModel,
    loader,
    device: torch.device,
    mmcl_w: float | None = None,
):
    model.eval()

    raw_model = unwrap_model(model)

    temperature = train.CONTRASTIVE_TEMPERATURE

    if mmcl_w is None:
        mmcl_w = train.MMCL_LAMBDA
    recon_w = train.RECON_LAMBDA
    cls_w = train.CLS_LAMBDA

    meter = init_stage1_meter()
    all_logits = []
    all_targets = []

    iterator = tqdm(
        loader,
        desc="stage1_val",
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

        sample_idx = batch.get("idx", None)

        if sample_idx is None:
            sample_idx = torch.arange(image.size(0), device=device)
        elif not isinstance(sample_idx, torch.Tensor):
            sample_idx = torch.tensor(sample_idx, device=device, dtype=torch.long)
        else:
            sample_idx = sample_idx.to(device=device, dtype=torch.long)

        mmcl_loss = compute_mmcl_loss(
            out=out,
            idxes=sample_idx,
            loss_type=train.MMCL_LOSS,
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

        all_logits.append(out["pred_class_logits"].detach().cpu())
        all_targets.append(target_label.detach().cpu())

        update_geometry_meter(
            meter=meter,
            path_z=out["path_z"],
            rad_z=out["rad_contrast_z"],
            target_label=target_label,
            bs=bs,
        )

    return finalize_stage1_meter(
        meter=meter,
        all_logits=all_logits,
        all_targets=all_targets,
        device=device,
        loader=loader,
    )


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
        disable=(not is_main_process()) or (not train.USE_TQDM),
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
        disable=(not is_main_process()) or (not train.USE_TQDM),
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


def format_stage1_log(
    prefix: str,
    m: dict[str, float],
) -> str:
    return (
        f"{prefix}_loss={m['loss']:.4f} "
        f"mmcl={m['mmcl']:.4f} "
        f"recon={m['recon']:.4f} "
        f"cls={m['cls']:.4f} | " 
        f"acc={m['acc']:.4f} "
        f"auroc={m['auroc']:.4f} "
        f"auprc={m['auprc']:.4f} "
        f"uni_p={m['uni_path']:.3f} "
        f"uni_r={m['uni_rad']:.3f} "
        f"rank_p={m['rank_path']:.1f} "
        f"rank_r={m['rank_rad']:.1f} "
        f"sep_p={m['sep_path']:.3f} "
        f"sep_r={m['sep_rad']:.3f}"
    )


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

    # stage1_params = (
    #     list(raw_model.radiomics_model.parameters())
    #     + list(raw_model.pathomics_proj.parameters())
    #     + list(raw_model.recon_head.parameters())
    #     + list(raw_model.cls_head.parameters())
    # )

    # optimizer_stage1 = torch.optim.AdamW(
    #     stage1_params,
    #     lr=train.LR,
    #     weight_decay=train.WEIGHT_DECAY_STAGE1,
    # )

    optimizer_stage1 = torch.optim.AdamW(
        [
            {
                "params": raw_model.radiomics_model.parameters(),
                "lr": train.LR,
            },
            {
                "params": raw_model.pathomics_proj.parameters(),
                "lr": train.LR * 0.3,
            },
            {
                "params": raw_model.recon_head.parameters(),
                "lr": train.LR,
            },
            {
                "params": raw_model.cls_head.parameters(),
                "lr": train.LR,
            },
        ],
        weight_decay=train.WEIGHT_DECAY_STAGE1,
    )

    best_stage1_val = float("inf")
    best_stage1_full_val = float("inf")

    stage1_epochs = train.STAGE1_EPOCHS
    warmup_recon_epochs = train.WARMUP_RECON_EPOCHS

    if is_main_process():
        print(f"[INFO] Stage1 recon warmup epochs: {warmup_recon_epochs}")

    for epoch in range(stage1_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        recon_only = epoch < warmup_recon_epochs
        stage_name = "ReconWarmup" if recon_only else "MMCLReconCls"

        current_mmcl_w = get_scheduled_mmcl_weight(
            epoch=epoch,
            warmup_epochs=warmup_recon_epochs,
            max_weight=train.MMCL_LAMBDA,
        )

        train_m = train_contrastive_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer_stage1,
            device=device,
            recon_only=recon_only,
            mmcl_w=current_mmcl_w,
        )

        val_m = eval_contrastive_epoch(
            model=model,
            loader=val_loader,
            device=device,
            mmcl_w=current_mmcl_w,
        )

        if is_main_process():
            print(
                f"[{now_str()}] "
                f"[Stage1:{stage_name}][Epoch {epoch}] "
                f"mmcl_w={current_mmcl_w:.4f} | "
                f"{format_stage1_log('train', train_m)} | "
                f"{format_stage1_log('val', val_m)}"
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
            {"params": raw_model.gene_head.parameters(), "lr": train.GENE_LR},
            {"params": raw_model.pathomics_proj.parameters(), "lr": train.PATH_PROJ_LR},
            {"params": raw_model.pathomics_encoder.parameters(), "lr": train.PATH_ENCODER_LR},
        ],
        weight_decay=train.WEIGHT_DECAY_STAGE2,
    )

    best_pcc = -float("inf")
    best_record: dict[str, Any] | None = None

    stage2_epochs = train.STAGE2_EPOCHS

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
                f"[{now_str()}] "
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