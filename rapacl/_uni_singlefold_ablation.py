# fold0 에 대해 
# 1. UNI -> PCA(256) -> Ridge Regression (HEST-Bench style)
# 2. UNI -> PCA(256) -> MLP 
# 3. concat([UNI -> PCA(256)], [rapacl-UNI-proj], [rapacl-RadTransTab-proj]) -> MLP 
# gene-wise PCC 비교 

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from rapacl.engines.data_utils import build_dataset, build_loader
from rapacl.engines.trainer_utils import set_seed
from rapacl.model.rapacl_uni import build_uni_model
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES

import rapacl.configs.default.train as train


FOLD = 0
PCA_DIM = 256
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
MLP_EPOCHS = 100
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256


def get_split_paths(fold: int) -> tuple[str, str]:
    split_dir = os.path.join(train.ROOT_DIR, "splits")
    return (
        os.path.join(split_dir, f"train_{fold}.csv"),
        os.path.join(split_dir, f"test_{fold}.csv"),
    )


def get_ckpt_path(fold: int) -> str:
    fold_dir = Path(train.OUTPUT_CHECKPOINT_DIR) / "rapacl_uni_frozen" / f"fold_{fold}"

    candidates = [
        fold_dir / "best_stage1_full_mmcl_recon_cls.pt",
        fold_dir / "best_stage1_mmcl_recon_cls.pt",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"No stage1 checkpoint found in {fold_dir}")


def unwrap_batch(batch: dict, device: torch.device):
    image = batch["image"].to(device)
    gene = batch["gene"].float().to(device)

    # 프로젝트 Dataset key에 맞게 필요 시 수정
    radiomics = batch.get("radiomics", batch.get("features", None))
    if radiomics is None:
        raise KeyError("Batch must contain `radiomics` or `features`.")
    radiomics = radiomics.float().to(device)

    return image, radiomics, gene


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()

    uni_raw_list = []
    path_proj_list = []
    rad_proj_list = []
    y_list = []

    for batch in loader:
        image, radiomics, gene = unwrap_batch(batch, device)

        # 1) frozen UNI raw embedding
        uni_raw = model.pathomics_encoder(image)

        # 2) RaPaCL pathomics projection
        path_proj = model.pathomics_proj(uni_raw)

        # 3) RaPaCL radiomics projection
        rad_out = model.radiomics_model(radiomics)

        if isinstance(rad_out, dict):
            rad_proj = (
                rad_out.get("contrastive")
                or rad_out.get("z")
                or rad_out.get("proj")
                or rad_out.get("embedding")
            )
        elif isinstance(rad_out, (tuple, list)):
            # 일반적으로 contrastive/projection embedding이 앞쪽에 있다고 가정
            rad_proj = rad_out[0]
        else:
            rad_proj = rad_out

        if rad_proj is None:
            raise RuntimeError("Could not infer radiomics projection from radiomics_model output.")

        uni_raw_list.append(uni_raw.detach().cpu().numpy())
        path_proj_list.append(path_proj.detach().cpu().numpy())
        rad_proj_list.append(rad_proj.detach().cpu().numpy())
        y_list.append(gene.detach().cpu().numpy())

    return {
        "uni_raw": np.concatenate(uni_raw_list, axis=0),
        "path_proj": np.concatenate(path_proj_list, axis=0),
        "rad_proj": np.concatenate(rad_proj_list, axis=0),
        "y": np.concatenate(y_list, axis=0),
    }


def gene_wise_pcc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    pccs = []

    for g in range(y_true.shape[1]):
        yt = y_true[:, g]
        yp = y_pred[:, g]

        yt = yt - yt.mean()
        yp = yp - yp.mean()

        denom = np.sqrt((yt ** 2).sum()) * np.sqrt((yp ** 2).sum()) + eps
        pccs.append(float((yt * yp).sum() / denom))

    return float(np.mean(pccs)), pccs


class GeneMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(x_train, y_train, x_val, y_val, device):
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)

    ds = torch.utils.data.TensorDataset(x_train, y_train)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = GeneMLP(
        in_dim=x_train.shape[1],
        out_dim=y_train.shape[1],
        hidden_dim=train.GENE_HIDDEN_DIM,
        dropout=train.HEAD_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=MLP_WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    best_pred = None
    best_pcc = -999.0

    for epoch in range(MLP_EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t).cpu().numpy()

        mean_pcc, _ = gene_wise_pcc(y_val, val_pred)

        if mean_pcc > best_pcc:
            best_pcc = mean_pcc
            best_pred = val_pred

        if epoch % 10 == 0 or epoch == MLP_EPOCHS - 1:
            print(
                f"[MLP][Epoch {epoch:03d}] "
                f"train_mse={total_loss / len(ds):.6f} "
                f"val_pcc={mean_pcc:.4f}"
            )

    return best_pred, best_pcc


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    mean_pcc, gene_pccs = gene_wise_pcc(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"\n[{name}]")
    print(f"Gene-wise PCC: {mean_pcc:.4f}")
    print(f"MSE          : {mse:.6f}")

    return {
        "method": name,
        "gene_wise_pcc": mean_pcc,
        "mse": float(mse),
        "gene_pccs": gene_pccs,
    }


def main():
    set_seed(train.SEED)

    device = torch.device(train.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    train_csv, val_csv = get_split_paths(FOLD)

    train_dataset = build_dataset(train_csv, use_image_augmentation=False)
    val_dataset = build_dataset(val_csv, use_image_augmentation=False)

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

    num_genes = len(train_dataset.genes)
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    print(f"[INFO] fold: {FOLD}")
    print(f"[INFO] train samples: {len(train_dataset)}")
    print(f"[INFO] val samples  : {len(val_dataset)}")
    print(f"[INFO] num_genes: {num_genes}")
    print(f"[INFO] num_radiomics_features: {num_radiomics_features}")

    model = build_uni_model(
        device=device,
        num_genes=num_genes,
        num_radiomics_features=num_radiomics_features,
    )

    ckpt_path = get_ckpt_path(FOLD)
    print(f"[INFO] load checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] missing keys: {missing}")
    print(f"[INFO] unexpected keys: {unexpected}")

    print("[INFO] extracting train features...")
    train_feat = extract_features(model, train_loader, device)

    print("[INFO] extracting val features...")
    val_feat = extract_features(model, val_loader, device)

    y_train = train_feat["y"]
    y_val = val_feat["y"]

    results = []

    # ------------------------------------------------------------
    # 1. UNI -> PCA(256) -> Ridge Regression
    # ------------------------------------------------------------
    scaler_uni = StandardScaler()
    x_train_uni = scaler_uni.fit_transform(train_feat["uni_raw"])
    x_val_uni = scaler_uni.transform(val_feat["uni_raw"])

    pca = PCA(n_components=PCA_DIM, random_state=train.SEED)
    x_train_pca = pca.fit_transform(x_train_uni)
    x_val_pca = pca.transform(x_val_uni)

    print(f"[INFO] PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(x_train_pca, y_train)
    pred_ridge = ridge.predict(x_val_pca)

    results.append(
        evaluate(
            "UNI_PCA256_Ridge",
            y_val,
            pred_ridge,
        )
    )

    # ------------------------------------------------------------
    # 2. UNI -> PCA(256) -> MLP
    # ------------------------------------------------------------
    scaler_pca = StandardScaler()
    x_train_pca_scaled = scaler_pca.fit_transform(x_train_pca)
    x_val_pca_scaled = scaler_pca.transform(x_val_pca)

    pred_mlp_pca, _ = train_mlp(
        x_train_pca_scaled,
        y_train,
        x_val_pca_scaled,
        y_val,
        device,
    )

    results.append(
        evaluate(
            "UNI_PCA256_MLP",
            y_val,
            pred_mlp_pca,
        )
    )

    # ------------------------------------------------------------
    # 3. concat([UNI PCA256], [RaPaCL UNI proj], [RaPaCL RadTransTab proj]) -> MLP
    # ------------------------------------------------------------
    x_train_concat = np.concatenate(
        [
            x_train_pca,
            train_feat["path_proj"],
            train_feat["rad_proj"],
        ],
        axis=1,
    )
    x_val_concat = np.concatenate(
        [
            x_val_pca,
            val_feat["path_proj"],
            val_feat["rad_proj"],
        ],
        axis=1,
    )

    scaler_concat = StandardScaler()
    x_train_concat = scaler_concat.fit_transform(x_train_concat)
    x_val_concat = scaler_concat.transform(x_val_concat)

    print(f"[INFO] concat dim: {x_train_concat.shape[1]}")

    pred_mlp_concat, _ = train_mlp(
        x_train_concat,
        y_train,
        x_val_concat,
        y_val,
        device,
    )

    results.append(
        evaluate(
            "UNI_PCA256_RaPaCLPathProj_RaPaCLRadProj_MLP",
            y_val,
            pred_mlp_concat,
        )
    )

    save_dir = Path(train.OUTPUT_CHECKPOINT_DIR) / "rapacl_uni_frozen" / f"fold_{FOLD}"
    save_path = save_dir / "uni_singlefold_ablation_result.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fold": FOLD,
                "checkpoint": ckpt_path,
                "pca_dim": PCA_DIM,
                "results": results,
            },
            f,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("[FINAL SINGLE-FOLD ABLATION RESULT]")
    for r in results:
        print(
            f"{r['method']} | "
            f"PCC={r['gene_wise_pcc']:.4f} | "
            f"MSE={r['mse']:.6f}"
        )
    print(f"Saved to: {save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()