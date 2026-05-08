# analysis.py
from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from rapacl.engines.data_utils import build_dataset, build_loader, get_batch_tensor
from rapacl.model.rapacl import build_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


PROJECT_DIR = os.path.join(os.path.expanduser("~"), "workspace", "RaPaCL")
RADTRANSTAB_PRETRAINED_DIR = os.path.join(
    PROJECT_DIR, "checkpoints", "radiomics_retrieval", "transtab"
)
OUTPUT_CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR, "checkpoints", "rapacl", "default"
)
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "outputs", "rapacl", "default"
)

CKPT_ROOT = os.path.join(OUTPUT_CHECKPOINT_DIR, "rapacl_baseline")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "posthoc_analysis")

TARGET_GENES = [
    "FASN", "FOXA1", "CEACAM6", "GATA3", "MZB1",
    "AGR3", "SERPINA3", "TACSTD2", "ABCC1", "MKI67",
]

NUM_FOLDS = train.NUM_FOLDS
DEVICE = train.DEVICE
MAX_GRADCAM_PER_GENE = 20
ATTR_BATCHES_PER_FOLD = 20


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(
        output_dir,
        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    logger = logging.getLogger("posthoc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f"Log path: {log_path}")
    return logger


def get_fold_split_paths(fold: int) -> tuple[str, str]:
    split_dir = os.path.join(train.ROOT_DIR, "splits")
    return (
        os.path.join(split_dir, f"train_{fold}.csv"),
        os.path.join(split_dir, f"test_{fold}.csv"),
    )


def get_gene_indices(genes: Iterable[str], target_genes: list[str]) -> dict[str, int]:
    genes = list(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    result = {}
    for gene in target_genes:
        if gene in gene_to_idx:
            result[gene] = gene_to_idx[gene]
        else:
            print(f"[WARN] target gene not found in dataset gene list: {gene}")

    return result


def load_fold_model(fold: int, device: torch.device, num_genes: int):
    model = build_model(
        device=device,
        num_genes=num_genes,
        num_radiomics_features=len(RADIOMICS_FEATURES_NAMES),
    )

    ckpt_path = os.path.join(
        CKPT_ROOT,
        f"fold_{fold}",
        "best_stage2_genepred.pt",
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, ckpt_path, ckpt


def find_last_conv2d(module: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m

    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    return last_conv


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, image, radiomics, gene_idx: int):
        self.model.zero_grad(set_to_none=True)

        image = image.detach().clone().requires_grad_(True)
        radiomics = radiomics.detach().clone()

        out = self.model.forward_gene(image=image, radiomics=radiomics)
        score = out["pred_gene"][:, gene_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze(1)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu(), out["pred_gene"].detach().cpu()


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu()

    if img.ndim == 3:
        if img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
    img = img.numpy()

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = np.clip(img, 0.0, 1.0)
    return img


def save_gradcam_figure(
    image: torch.Tensor,
    cam: torch.Tensor,
    save_path: str,
    title: str,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img_np = tensor_image_to_numpy(image)
    cam_np = cam.detach().cpu().numpy()

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.axis("off")
    plt.title("Patch")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_np, cmap="jet")
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(cam_np, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title("Overlay")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def collect_predictions(model, loader, device, gene_indices: dict[str, int]):
    rows = []

    for batch_idx, batch in enumerate(loader):
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        out = model.forward_gene(image=image, radiomics=radiomics)
        pred = out["pred_gene"]

        bs = image.size(0)

        for i in range(bs):
            row = {
                "batch_idx": batch_idx,
                "sample_idx": batch_idx * bs + i,
            }

            for gene_name, gene_idx in gene_indices.items():
                y_true = gene[i, gene_idx].item()
                y_pred = pred[i, gene_idx].item()

                row[f"{gene_name}_true"] = y_true
                row[f"{gene_name}_pred"] = y_pred
                row[f"{gene_name}_abs_error"] = abs(y_true - y_pred)

            rows.append(row)

    return pd.DataFrame(rows)


@torch.no_grad()
def compute_train_radiomics_mean(train_loader, device):
    total = None
    count = 0

    for batch in train_loader:
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)

        if total is None:
            total = radiomics.sum(dim=0)
        else:
            total += radiomics.sum(dim=0)

        count += radiomics.size(0)

    return total / max(count, 1)


@torch.no_grad()
def modality_ablation(
    model,
    loader,
    device,
    gene_indices: dict[str, int],
    radiomics_mean: torch.Tensor,
):
    rows = []

    for batch in loader:
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        zero_image = torch.zeros_like(image)
        mean_radiomics = radiomics_mean.view(1, -1).repeat(radiomics.size(0), 1)

        pred_full = model.forward_gene(image=image, radiomics=radiomics)["pred_gene"]
        pred_no_image = model.forward_gene(
            image=zero_image,
            radiomics=radiomics,
        )["pred_gene"]
        pred_no_radiomics = model.forward_gene(
            image=image,
            radiomics=mean_radiomics,
        )["pred_gene"]

        for gene_name, gene_idx in gene_indices.items():
            y = gene[:, gene_idx]

            mse_full = F.mse_loss(pred_full[:, gene_idx], y).item()
            mse_no_image = F.mse_loss(pred_no_image[:, gene_idx], y).item()
            mse_no_radiomics = F.mse_loss(pred_no_radiomics[:, gene_idx], y).item()

            rows.append(
                {
                    "gene": gene_name,
                    "mse_full": mse_full,
                    "mse_no_image": mse_no_image,
                    "mse_no_radiomics": mse_no_radiomics,
                    "image_contribution": mse_no_image - mse_full,
                    "radiomics_contribution": mse_no_radiomics - mse_full,
                }
            )

    df = pd.DataFrame(rows)
    return df.groupby("gene", as_index=False).mean()


def radiomics_gradient_attribution(
    model,
    loader,
    device,
    gene_indices: dict[str, int],
    max_batches: int,
):
    attr_sum = {
        gene: torch.zeros(len(RADIOMICS_FEATURES_NAMES), device=device)
        for gene in gene_indices
    }
    count = 0

    model.eval()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)

        radiomics = radiomics.detach().clone().requires_grad_(True)

        out = model.forward_gene(image=image, radiomics=radiomics)
        pred = out["pred_gene"]

        for gene_name, gene_idx in gene_indices.items():
            model.zero_grad(set_to_none=True)

            if radiomics.grad is not None:
                radiomics.grad.zero_()

            score = pred[:, gene_idx].sum()
            score.backward(retain_graph=True)

            grad = radiomics.grad.detach()
            attr = (grad * radiomics.detach()).abs().sum(dim=0)
            attr_sum[gene_name] += attr

        count += radiomics.size(0)

    rows = []
    for gene_name, values in attr_sum.items():
        values = (values / max(count, 1)).detach().cpu().numpy()

        for feature_name, value in zip(RADIOMICS_FEATURES_NAMES, values):
            rows.append(
                {
                    "gene": gene_name,
                    "feature": feature_name,
                    "importance": float(value),
                }
            )

    return pd.DataFrame(rows)


def save_top_radiomics_plot(attr_df: pd.DataFrame, save_dir: str, top_k: int = 15):
    os.makedirs(save_dir, exist_ok=True)

    for gene in sorted(attr_df["gene"].unique()):
        sub = attr_df[attr_df["gene"] == gene]
        sub = sub.sort_values("importance", ascending=False).head(top_k)

        plt.figure(figsize=(8, 5))
        plt.barh(sub["feature"][::-1], sub["importance"][::-1])
        plt.xlabel("Gradient × Input importance")
        plt.title(f"Top radiomics features - {gene}")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{gene}_top_radiomics.png")
        plt.savefig(save_path, dpi=200)
        plt.close()


def run_gradcam_for_fold(
    model,
    loader,
    device,
    gene_indices: dict[str, int],
    save_dir: str,
    logger: logging.Logger,
):
    os.makedirs(save_dir, exist_ok=True)

    target_layer = find_last_conv2d(model.pathomics_encoder)
    gradcam = GradCAM(model=model, target_layer=target_layer)

    saved_count = {gene: 0 for gene in gene_indices}

    try:
        for batch_idx, batch in enumerate(loader):
            image = get_batch_tensor(batch, ("image", "img", "patch"), device)
            radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
            gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

            bs = image.size(0)

            for gene_name, gene_idx in gene_indices.items():
                if saved_count[gene_name] >= MAX_GRADCAM_PER_GENE:
                    continue

                cam, pred = gradcam(
                    image=image,
                    radiomics=radiomics,
                    gene_idx=gene_idx,
                )

                remain = MAX_GRADCAM_PER_GENE - saved_count[gene_name]
                n_save = min(bs, remain)

                for i in range(n_save):
                    y_true = gene[i, gene_idx].item()
                    y_pred = pred[i, gene_idx].item()

                    title = (
                        f"{gene_name} | "
                        f"true={y_true:.4f}, pred={y_pred:.4f}"
                    )

                    save_path = os.path.join(
                        save_dir,
                        gene_name,
                        f"batch{batch_idx:04d}_idx{i:02d}.png",
                    )

                    save_gradcam_figure(
                        image=image[i],
                        cam=cam[i],
                        save_path=save_path,
                        title=title,
                    )

                    saved_count[gene_name] += 1

            if all(v >= MAX_GRADCAM_PER_GENE for v in saved_count.values()):
                break

    finally:
        gradcam.remove()

    logger.info(f"Grad-CAM saved counts: {saved_count}")


def run_fold(fold: int, logger: logging.Logger):
    device = torch.device(DEVICE)

    logger.info("=" * 80)
    logger.info(f"Start fold {fold}")

    train_split_csv, val_split_csv = get_fold_split_paths(fold)

    logger.info(f"Train split: {train_split_csv}")
    logger.info(f"Val split  : {val_split_csv}")

    train_dataset = build_dataset(
        train_split_csv,
        use_image_augmentation=False,
    )
    val_dataset = build_dataset(
        val_split_csv,
        use_image_augmentation=False,
    )

    train_loader, _ = build_loader(
        train_dataset,
        shuffle=False,
        drop_last=False,
        distributed=False,
        pair_augment=False,
    )

    val_loader, _ = build_loader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        distributed=False,
        pair_augment=False,
    )

    gene_indices = get_gene_indices(val_dataset.genes, TARGET_GENES)

    logger.info(f"Target gene indices: {gene_indices}")

    model, ckpt_path, ckpt = load_fold_model(
        fold=fold,
        device=device,
        num_genes=len(val_dataset.genes),
    )

    logger.info(f"Loaded checkpoint: {ckpt_path}")
    logger.info(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    logger.info(f"Checkpoint best record: {ckpt.get('best_record', 'N/A')}")

    fold_dir = os.path.join(ANALYSIS_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    logger.info("Collecting gene predictions...")
    pred_df = collect_predictions(
        model=model,
        loader=val_loader,
        device=device,
        gene_indices=gene_indices,
    )
    pred_path = os.path.join(fold_dir, "target_gene_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions: {pred_path}")

    logger.info("Computing train radiomics mean...")
    radiomics_mean = compute_train_radiomics_mean(train_loader, device)

    logger.info("Running modality ablation...")
    ablation_df = modality_ablation(
        model=model,
        loader=val_loader,
        device=device,
        gene_indices=gene_indices,
        radiomics_mean=radiomics_mean,
    )
    ablation_path = os.path.join(fold_dir, "modality_ablation.csv")
    ablation_df.to_csv(ablation_path, index=False)
    logger.info(f"Saved modality ablation: {ablation_path}")

    logger.info("Running radiomics gradient attribution...")
    attr_df = radiomics_gradient_attribution(
        model=model,
        loader=val_loader,
        device=device,
        gene_indices=gene_indices,
        max_batches=ATTR_BATCHES_PER_FOLD,
    )
    attr_path = os.path.join(fold_dir, "radiomics_gradient_input_attribution.csv")
    attr_df.to_csv(attr_path, index=False)
    logger.info(f"Saved radiomics attribution: {attr_path}")

    plot_dir = os.path.join(fold_dir, "radiomics_top_features")
    save_top_radiomics_plot(attr_df, plot_dir)
    logger.info(f"Saved radiomics top feature plots: {plot_dir}")

    logger.info("Running Grad-CAM...")
    gradcam_dir = os.path.join(fold_dir, "gradcam")
    run_gradcam_for_fold(
        model=model,
        loader=val_loader,
        device=device,
        gene_indices=gene_indices,
        save_dir=gradcam_dir,
        logger=logger,
    )
    logger.info(f"Saved Grad-CAM results: {gradcam_dir}")

    summary = {
        "fold": fold,
        "checkpoint": ckpt_path,
        "target_genes": list(gene_indices.keys()),
        "gene_indices": gene_indices,
        "prediction_csv": pred_path,
        "modality_ablation_csv": ablation_path,
        "radiomics_attribution_csv": attr_path,
        "gradcam_dir": gradcam_dir,
    }

    summary_path = os.path.join(fold_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Saved summary: {summary_path}")
    logger.info(f"Done fold {fold}")


def aggregate_results(logger: logging.Logger):
    logger.info("Aggregating fold results...")

    ablation_all = []
    attr_all = []

    for fold in range(NUM_FOLDS):
        fold_dir = os.path.join(ANALYSIS_DIR, f"fold_{fold}")

        ablation_path = os.path.join(fold_dir, "modality_ablation.csv")
        attr_path = os.path.join(fold_dir, "radiomics_gradient_input_attribution.csv")

        if os.path.exists(ablation_path):
            df = pd.read_csv(ablation_path)
            df["fold"] = fold
            ablation_all.append(df)

        if os.path.exists(attr_path):
            df = pd.read_csv(attr_path)
            df["fold"] = fold
            attr_all.append(df)

    if ablation_all:
        ablation_df = pd.concat(ablation_all, axis=0)
        ablation_mean = ablation_df.groupby("gene", as_index=False).mean(numeric_only=True)

        save_path = os.path.join(ANALYSIS_DIR, "all_folds_modality_ablation_mean.csv")
        ablation_mean.to_csv(save_path, index=False)
        logger.info(f"Saved aggregated ablation: {save_path}")

    if attr_all:
        attr_df = pd.concat(attr_all, axis=0)
        attr_mean = (
            attr_df.groupby(["gene", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["gene", "importance"], ascending=[True, False])
        )

        save_path = os.path.join(ANALYSIS_DIR, "all_folds_radiomics_attribution_mean.csv")
        attr_mean.to_csv(save_path, index=False)
        logger.info(f"Saved aggregated attribution: {save_path}")

        save_top_radiomics_plot(
            attr_mean,
            os.path.join(ANALYSIS_DIR, "all_folds_radiomics_top_features"),
        )


def main():
    logger = setup_logger(ANALYSIS_DIR)

    logger.info(f"PROJECT_DIR: {PROJECT_DIR}")
    logger.info(f"CKPT_ROOT: {CKPT_ROOT}")
    logger.info(f"ANALYSIS_DIR: {ANALYSIS_DIR}")
    logger.info(f"DEVICE: {DEVICE}")
    logger.info(f"TARGET_GENES: {TARGET_GENES}")

    for fold in range(NUM_FOLDS):
        run_fold(fold, logger)

    aggregate_results(logger)
    logger.info("All post-hoc analysis completed.")


if __name__ == "__main__":
    main()