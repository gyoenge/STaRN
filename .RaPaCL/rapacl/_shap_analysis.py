# python -m rapacl._shap_analysis --model_type rapacl --folds 0,1,2,3 --background_size 10 --explain_size 5
# python -m rapacl._shap_analysis --model_type uni_mlp --folds 0,1,2,3
# python -m rapacl._shap_analysis --model_type densenet_mlp --folds 0,1,2,3
# python -m rapacl._shap_analysis --model_type radtranstab_mlp --folds 0,1,2,3

from __future__ import annotations

import os
import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import shap

from rapacl.engines.trainer_utils import set_seed
from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.model.rapacl import build_model
from rapacl.model.rapacl_uni import build_uni_model
from rapacl._uni_mlp import build_uni_mlp_model
from rapacl._densenet_mlp import build_densenet_mlp_model
from rapacl._radtranstab_mlp import build_radtranstab_mlp_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


TARGET_GENES = ["FASN", "CEACAM6", "GATA3", "SERPINA3", "TACSTD2", "ABCC11"]


def get_experiment_name(model_type: str = "rapacl") -> str:
    if model_type == "uni_mlp":
        return "uni_frozen_mlp"

    if model_type == "densenet_mlp":
        return "densenet121_mlp"

    if model_type == "radtranstab_mlp":
        return "radtranstab_mlp"

    backbone = train.BACKBONE.lower().strip()
    return "rapacl_uni_frozen" if backbone == "uni" else f"rapacl_{backbone}"


def get_fold_split_paths(fold: int) -> tuple[str, str]:
    split_dir = os.path.join(train.ROOT_DIR, "splits")
    train_split_csv = os.path.join(split_dir, f"train_{fold}.csv")
    val_split_csv = os.path.join(split_dir, f"test_{fold}.csv")
    return train_split_csv, val_split_csv


def unwrap_state_dict(ckpt: Any) -> dict:
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_module_prefix(state_dict: dict) -> dict:
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def remap_old_densenet_keys(state_dict: dict) -> dict:
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("pathomics_encoder.features."):
            k = k.replace(
                "pathomics_encoder.features.",
                "pathomics_encoder.encoder.features.",
                1,
            )
        elif k.startswith("pathomics_encoder.classifier."):
            k = k.replace(
                "pathomics_encoder.classifier.",
                "pathomics_encoder.encoder.classifier.",
                1,
            )

        new_state[k] = v

    return new_state


def find_best_stage2_checkpoint(save_dir: str) -> str:
    candidates = [
        "best_radtranstab_mlp.pt",
        "best_uni_mlp.pt",
        "best_densenet121_mlp.pt",
        "best_stage2_genepred.pt",
        "stage2_best.pt",
        "best_stage2.pt",
        "best_model.pt",
        "model_best.pt",
        "checkpoint_best.pt",
    ]

    for name in candidates:
        path = os.path.join(save_dir, name)
        if os.path.isfile(path):
            return path

    pt_files = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.endswith((".pt", ".pth"))
    ]

    stage2_files = [p for p in pt_files if "stage2" in os.path.basename(p).lower()]
    best_files = [p for p in pt_files if "best" in os.path.basename(p).lower()]

    if stage2_files:
        return sorted(stage2_files)[-1]
    if best_files:
        return sorted(best_files)[-1]
    if pt_files:
        return sorted(pt_files)[-1]

    raise FileNotFoundError(f"No checkpoint found in: {save_dir}")


class RapaclImageOnlyWrapper(nn.Module):
    def __init__(self, model, gene_idx, fixed_radiomics):
        super().__init__()
        self.model = model
        self.gene_idx = gene_idx
        self.register_buffer("fixed_radiomics", fixed_radiomics.mean(dim=0, keepdim=True))

    def forward(self, image):
        radiomics = self.fixed_radiomics.repeat(image.size(0), 1)

        output = self.model.forward_gene(
            image=image,
            radiomics=radiomics,
        )

        pred = output["pred_gene"]
        return pred[:, self.gene_idx].unsqueeze(1)

class RapaclRadiomicsKernelWrapper:
    def __init__(
        self,
        model,
        gene_idx,
        fixed_image,
        device,
        max_batch_size: int = 8,
    ):
        self.model = model
        self.gene_idx = gene_idx
        self.device = device
        self.max_batch_size = max_batch_size

        # 대표 이미지 1장만 고정
        self.fixed_image = fixed_image[:1].to(device)

    def __call__(self, radiomics_np):
        self.model.eval()

        outputs = []

        for start in range(0, len(radiomics_np), self.max_batch_size):
            end = start + self.max_batch_size

            radiomics = torch.tensor(
                radiomics_np[start:end],
                dtype=torch.float32,
                device=self.device,
            )

            image = self.fixed_image.repeat(
                radiomics.size(0),
                1,
                1,
                1,
            )

            with torch.no_grad():
                output = self.model.forward_gene(
                    image=image,
                    radiomics=radiomics,
                )

                pred = output["pred_gene"][:, self.gene_idx]
                outputs.append(pred.detach().cpu())

            del image, radiomics, output, pred
            torch.cuda.empty_cache()

        return torch.cat(outputs, dim=0).numpy()


def build_eval_model(
    device: torch.device,
    num_genes: int,
    model_type: str,
    feature_cols: list[str] | None = None,
):
    if model_type == "uni_mlp":
        return build_uni_mlp_model(
            device=device,
            num_genes=num_genes,
        )

    if model_type == "densenet_mlp":
        return build_densenet_mlp_model(
            device=device,
            num_genes=num_genes,
        )

    if model_type == "radtranstab_mlp":
        assert feature_cols is not None
        return build_radtranstab_mlp_model(
            device=device,
            num_genes=num_genes,
            feature_cols=feature_cols,
        )

    backbone = train.BACKBONE.lower().strip()
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    if backbone == "uni":
        model = build_uni_model(
            device=device,
            num_genes=num_genes,
            num_radiomics_features=num_radiomics_features,
        )
    else:
        model = build_model(
            device=device,
            num_genes=num_genes,
            num_radiomics_features=num_radiomics_features,
            backbone=backbone,
        )

    return model.to(device)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = strip_module_prefix(unwrap_state_dict(ckpt))
    state_dict = remap_old_densenet_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[MISSING]")
    print("\n".join(missing[:50]))
    print("[UNEXPECTED]")
    print("\n".join(unexpected[:50]))

    assert len(unexpected) == 0, f"Unexpected keys exist: {unexpected[:10]}"
    assert len(missing) == 0, f"Missing keys exist: {missing[:10]}"

    model.eval()
    return model


def extract_feature_tensor(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in [
            "pathomics_proj",
            "path_proj",
            "projected",
            "proj",
            "feat",
            "embedding",
            "cls",
            "path_cls",
        ]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        for v in output.values():
            if torch.is_tensor(v) and v.ndim == 2:
                return v

        raise KeyError(f"Cannot find tensor feature in dict keys: {list(output.keys())}")

    if isinstance(output, (tuple, list)):
        for v in output:
            if torch.is_tensor(v) and v.ndim == 2:
                return v

    raise TypeError(f"Cannot extract feature tensor from type: {type(output)}")



def collect_shap_batches(
    loader,
    device: torch.device,
    background_size: int,
    explain_size: int,
):
    images = []
    radiomics = []
    genes = []
    meta_rows = []

    total_needed = background_size + explain_size

    for batch in tqdm(loader, desc="[Collect SHAP batches]", leave=False):
        image = batch["image"].float()
        rad = batch["radiomics"].float()
        gene = batch["gene"].float()

        images.append(image)
        radiomics.append(rad)
        genes.append(gene)

        batch_size = gene.size(0)

        for i in range(batch_size):
            row = {}
            for key in ["sample_id", "barcode", "patch_idx", "idx", "coord_x", "coord_y"]:
                if key in batch:
                    value = batch[key][i]
                    if torch.is_tensor(value):
                        value = value.item()
                    row[key] = value
            meta_rows.append(row)

        if sum(x.size(0) for x in genes) >= total_needed:
            break

    images = torch.cat(images, dim=0)[:total_needed].to(device)
    radiomics = torch.cat(radiomics, dim=0)[:total_needed].to(device)
    genes = torch.cat(genes, dim=0)[:total_needed].to(device)
    meta_rows = meta_rows[:total_needed]

    bg_image = images[:background_size]
    bg_radiomics = radiomics[:background_size]

    ex_image = images[background_size:background_size + explain_size]
    ex_radiomics = radiomics[background_size:background_size + explain_size]
    ex_gene = genes[background_size:background_size + explain_size]
    ex_meta = meta_rows[background_size:background_size + explain_size]

    return {
        "bg_image": bg_image,
        "bg_radiomics": bg_radiomics,
        "ex_image": ex_image,
        "ex_radiomics": ex_radiomics,
        "ex_gene": ex_gene,
        "ex_meta": ex_meta,
    }


class GeneOutputWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        gene_idx: int,
    ):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.gene_idx = gene_idx

    def forward(self, *inputs):
        if self.model_type == "radtranstab_mlp":
            radiomics = inputs[0]
            pred = self.model(radiomics)

        elif self.model_type in ["uni_mlp", "densenet_mlp"]:
            image = inputs[0]
            pred = self.model(image)

        else:
            image, radiomics = inputs

            if hasattr(self.model, "forward_gene"):
                output = self.model.forward_gene(
                    image=image,
                    radiomics=radiomics,
                )
                pred = output["pred_gene"]
            else:
                pred = self.model(image, radiomics)

        return pred[:, self.gene_idx].unsqueeze(1)


def normalize_shap_values(shap_values):
    """
    SHAP 반환 형식을 통일.
    - single input: np.ndarray 또는 list
    - multi input : list[np.ndarray]
    - output dim 포함 가능
    """
    if isinstance(shap_values, tuple):
        shap_values = list(shap_values)

    if isinstance(shap_values, list):
        normalized = []
        for sv in shap_values:
            sv = np.asarray(sv)

            if sv.ndim >= 3 and sv.shape[-1] == 1:
                sv = np.squeeze(sv, axis=-1)

            normalized.append(sv)

        return normalized

    shap_values = np.asarray(shap_values)

    if shap_values.ndim >= 3 and shap_values.shape[-1] == 1:
        shap_values = np.squeeze(shap_values, axis=-1)

    return shap_values


def build_explainer(wrapper, background_inputs, prefer: str = "gradient"):
    """
    DeepExplainer가 모델 구조에 따라 실패할 수 있어서 GradientExplainer fallback.
    """
    try:
        if prefer == "deep":
            explainer = shap.DeepExplainer(wrapper, background_inputs)
        else:
            explainer = shap.GradientExplainer(wrapper, background_inputs)

        return explainer

    except Exception as e:
        print(f"[WARN] {prefer} explainer failed: {repr(e)}")
        print("[INFO] fallback to GradientExplainer")

        return shap.GradientExplainer(wrapper, background_inputs)


def save_radiomics_shap_summary(
    shap_values: np.ndarray,
    input_values: np.ndarray,
    feature_names: list[str],
    out_dir: str,
    prefix: str,
):
    os.makedirs(out_dir, exist_ok=True)

    shap_values = np.asarray(shap_values)
    input_values = np.asarray(input_values)

    if shap_values.ndim != 2:
        print(f"[WARN] radiomics SHAP ndim is not 2: {shap_values.shape}")
        return

    mean_abs = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    csv_path = os.path.join(out_dir, f"{prefix}_radiomics_shap_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"[INFO] saved: {csv_path}")

    plt.figure()
    shap.summary_plot(
        shap_values,
        input_values,
        feature_names=feature_names,
        show=False,
        max_display=min(30, len(feature_names)),
    )
    path = os.path.join(out_dir, f"{prefix}_radiomics_shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved: {path}")

    plt.figure()
    shap.summary_plot(
        shap_values,
        input_values,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=min(30, len(feature_names)),
    )
    path = os.path.join(out_dir, f"{prefix}_radiomics_shap_bar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved: {path}")


def save_image_shap_heatmaps(
    shap_values: np.ndarray,
    images: torch.Tensor,
    out_dir: str,
    prefix: str,
    max_images: int = 12,
):
    os.makedirs(out_dir, exist_ok=True)

    shap_values = np.asarray(shap_values)
    image_np = images.detach().cpu().numpy()

    if shap_values.ndim != 4:
        print(f"[WARN] image SHAP ndim is not 4: {shap_values.shape}")
        return

    # image: [B, C, H, W], shap: [B, C, H, W]
    if image_np.shape[1] in [1, 3]:
        image_vis = np.transpose(image_np, (0, 2, 3, 1))
        shap_heat = np.mean(np.abs(shap_values), axis=1)
    else:
        print(f"[WARN] unexpected image shape: {image_np.shape}")
        return

    n = min(max_images, image_vis.shape[0])

    for i in range(n):
        img = image_vis[i]
        heat = shap_heat[i]

        # percentile normalize
        vmax = np.percentile(heat, 99)
        vmin = np.percentile(heat, 5)

        heat = np.clip(heat, vmin, vmax)
        heat = (heat - vmin) / (vmax - vmin + 1e-8)

        img = img.astype(np.float32)
        img = img - img.min()
        img = img / (img.max() + 1e-8)

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Patch")

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(
            heat,
            alpha=0.7,
            cmap="jet",
        )
        plt.axis("off")
        plt.title("Mean |SHAP| heatmap")

        plt.tight_layout()

        path = os.path.join(out_dir, f"{prefix}_image_shap_{i:03d}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[INFO] saved image SHAP heatmaps: {out_dir}")


def save_prediction_check(
    wrapper: nn.Module,
    inputs,
    y_true: torch.Tensor,
    gene_idx: int,
    out_dir: str,
    prefix: str,
):
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        pred = wrapper(*inputs).detach().cpu().numpy().reshape(-1)

    true = y_true[:, gene_idx].detach().cpu().numpy().reshape(-1)

    df = pd.DataFrame(
        {
            "y_true": true,
            "y_pred": pred,
        }
    )

    path = os.path.join(out_dir, f"{prefix}_prediction_check.csv")
    df.to_csv(path, index=False)
    print(f"[INFO] saved: {path}")


def run_shap_for_gene(
    model: nn.Module,
    model_type: str,
    gene: str,
    gene_idx: int,
    batch_data: dict,
    feature_names: list[str],
    out_dir: str,
    explainer_type: str,
):
    print("\n" + "=" * 80)
    print(f"[SHAP] gene={gene} | gene_idx={gene_idx}")
    print("=" * 80)

    wrapper = GeneOutputWrapper(
        model=model,
        model_type=model_type,
        gene_idx=gene_idx,
    ).eval()

    # =========================================================
    # Case 1. RaPaCL multimodal model
    # - image: Gradient SHAP
    # - radiomics: Kernel SHAP
    # =========================================================
    if model_type == "rapacl":
        input_for_wrapper = (
            batch_data["ex_image"],
            batch_data["ex_radiomics"],
        )

        save_prediction_check(
            wrapper=wrapper,
            inputs=input_for_wrapper,
            y_true=batch_data["ex_gene"],
            gene_idx=gene_idx,
            out_dir=out_dir,
            prefix=gene,
        )

        # -------------------------
        # 1) Image SHAP
        # -------------------------
        image_wrapper = RapaclImageOnlyWrapper(
            model=model,
            gene_idx=gene_idx,
            fixed_radiomics=batch_data["bg_radiomics"],
        ).eval()

        try:
            image_explainer = shap.GradientExplainer(
                image_wrapper,
                batch_data["bg_image"],
            )

            image_shap = image_explainer.shap_values(batch_data["ex_image"])
            image_shap = normalize_shap_values(image_shap)

            save_image_shap_heatmaps(
                shap_values=image_shap,
                images=batch_data["ex_image"],
                out_dir=out_dir,
                prefix=gene,
            )

        except Exception as e:
            print(f"[WARN] image GradientExplainer failed: {repr(e)}")
            print("[WARN] skip image SHAP for this gene.")

        # -------------------------
        # 2) Radiomics SHAP
        # -------------------------
        bg_rad_np = batch_data["bg_radiomics"].detach().cpu().numpy()
        ex_rad_np = batch_data["ex_radiomics"].detach().cpu().numpy()

        rad_predict_fn = RapaclRadiomicsKernelWrapper(
            model=model,
            gene_idx=gene_idx,
            fixed_image=batch_data["bg_image"],
            device=batch_data["bg_radiomics"].device,
            max_batch_size=4,
        )

        rad_explainer = shap.KernelExplainer(
            rad_predict_fn,
            bg_rad_np,
        )

        radiomics_shap = rad_explainer.shap_values(
            ex_rad_np,
            # nsamples=min(500, 2 * ex_rad_np.shape[1] + 100),
            nsamples=50,
        )

        radiomics_shap = normalize_shap_values(radiomics_shap)

        save_radiomics_shap_summary(
            shap_values=radiomics_shap,
            input_values=ex_rad_np,
            feature_names=feature_names,
            out_dir=out_dir,
            prefix=gene,
        )

        print(f"[INFO] completed SHAP for gene: {gene}")
        return

    # =========================================================
    # Case 2. Radiomics-only model
    # =========================================================
    if model_type == "radtranstab_mlp":
        background_inputs = batch_data["bg_radiomics"]
        explain_inputs = batch_data["ex_radiomics"]
        input_for_wrapper = (explain_inputs,)

        save_prediction_check(
            wrapper=wrapper,
            inputs=input_for_wrapper,
            y_true=batch_data["ex_gene"],
            gene_idx=gene_idx,
            out_dir=out_dir,
            prefix=gene,
        )

        # TransTab 내부에서 tensor -> DataFrame 변환이 있으면 gradient가 끊길 수 있음
        bg_np = background_inputs.detach().cpu().numpy()
        ex_np = explain_inputs.detach().cpu().numpy()

        def predict_fn(x_np):
            x = torch.tensor(
                x_np,
                dtype=torch.float32,
                device=background_inputs.device,
            )
            with torch.no_grad():
                pred = wrapper(x)
            return pred.detach().cpu().numpy().reshape(-1)

        explainer = shap.KernelExplainer(predict_fn, bg_np)
        shap_values = explainer.shap_values(
            ex_np,
            nsamples=min(500, 2 * ex_np.shape[1] + 100),
        )
        shap_values = normalize_shap_values(shap_values)

        save_radiomics_shap_summary(
            shap_values=shap_values,
            input_values=ex_np,
            feature_names=feature_names,
            out_dir=out_dir,
            prefix=gene,
        )

        print(f"[INFO] completed SHAP for gene: {gene}")
        return

    # =========================================================
    # Case 3. Image-only models
    # =========================================================
    if model_type in ["uni_mlp", "densenet_mlp"]:
        background_inputs = batch_data["bg_image"]
        explain_inputs = batch_data["ex_image"]
        input_for_wrapper = (explain_inputs,)

        save_prediction_check(
            wrapper=wrapper,
            inputs=input_for_wrapper,
            y_true=batch_data["ex_gene"],
            gene_idx=gene_idx,
            out_dir=out_dir,
            prefix=gene,
        )

        explainer = build_explainer(
            wrapper=wrapper,
            background_inputs=background_inputs,
            prefer=explainer_type,
        )

        shap_values = explainer.shap_values(explain_inputs)
        shap_values = normalize_shap_values(shap_values)

        save_image_shap_heatmaps(
            shap_values=shap_values,
            images=batch_data["ex_image"],
            out_dir=out_dir,
            prefix=gene,
        )

        print(f"[INFO] completed SHAP for gene: {gene}")
        return

    raise ValueError(f"Unsupported model_type: {model_type}")


def run_one_fold_shap(
    fold: int,
    device: torch.device,
    model_type: str,
    checkpoint_path: str | None,
    background_size: int,
    explain_size: int,
    explainer_type: str,
):
    set_seed(train.SEED + fold * 100)

    _, val_split_csv = get_fold_split_paths(fold)

    val_dataset = build_dataset(
        val_split_csv,
        use_image_augmentation=False,
    )

    val_loader, _ = build_loader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        distributed=False,
        pair_augment=False,
    )

    genes = val_dataset.genes
    num_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    exp_name = get_experiment_name(model_type)
    save_dir = os.path.join(train.OUTPUT_CHECKPOINT_DIR, exp_name, f"fold_{fold}")

    if checkpoint_path is None:
        checkpoint_path = find_best_stage2_checkpoint(save_dir)

    print(f"[INFO][Fold {fold}] checkpoint: {checkpoint_path}")

    model = build_eval_model(
        device=device,
        num_genes=num_genes,
        model_type=model_type,
        feature_cols=val_dataset.feature_cols,
    )

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    batch_data = collect_shap_batches(
        loader=val_loader,
        device=device,
        background_size=background_size,
        explain_size=explain_size,
    )

    feature_names = getattr(val_dataset, "feature_cols", None)
    if feature_names is None:
        feature_names = RADIOMICS_FEATURES_NAMES

    fold_out_dir = os.path.join(
        train.OUTPUT_DIR,
        exp_name,
        "shap_analysis",
        f"fold_{fold}",
    )
    os.makedirs(fold_out_dir, exist_ok=True)

    for gene in TARGET_GENES:
        if gene not in gene_to_idx:
            print(f"[WARN] target gene not found: {gene}")
            continue

        gene_out_dir = os.path.join(fold_out_dir, gene)
        os.makedirs(gene_out_dir, exist_ok=True)

        run_shap_for_gene(
            model=model,
            model_type=model_type,
            gene=gene,
            gene_idx=gene_to_idx[gene],
            batch_data=batch_data,
            feature_names=feature_names,
            out_dir=gene_out_dir,
            explainer_type=explainer_type,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--folds", type=str, default=None, help="e.g., 0,1,2,3")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument(
        "--model_type",
        type=str,
        default="rapacl",
        choices=["rapacl", "uni_mlp", "densenet_mlp", "radtranstab_mlp"],
    )

    parser.add_argument("--background_size", type=int, default=64)
    parser.add_argument("--explain_size", type=int, default=64)

    parser.add_argument(
        "--explainer_type",
        type=str,
        default="gradient",
        choices=["gradient", "deep"],
        help="DeepExplainer may fail for some custom modules. GradientExplainer is safer.",
    )

    args = parser.parse_args()

    device = torch.device(train.DEVICE if torch.cuda.is_available() else "cpu")

    if args.folds is None:
        folds = train.SELECT_FOLDS
    else:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    for fold in folds:
        run_one_fold_shap(
            fold=fold,
            device=device,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            background_size=args.background_size,
            explain_size=args.explain_size,
            explainer_type=args.explainer_type,
        )


if __name__ == "__main__":
    main()
