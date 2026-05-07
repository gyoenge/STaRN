from __future__ import annotations

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES
from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import freeze_module

import rapacl.configs.default.train as train


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNet121PathomicsEncoder(nn.Module):
    def __init__(self, out_dim: int = 1024, pretrained: bool = True):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)

        self.features = backbone.features
        self.out_dim = 1024
        self.proj = nn.Identity() if out_dim == self.out_dim else nn.Linear(self.out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = x.float()

        if x.max() > 2:
            x = x / 255.0

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device,
        ).view(1, 3, 1, 1)

        x = (x - mean) / std

        feat = self.features(x)
        feat = F.relu(feat, inplace=True)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)

        return self.proj(feat)


class MMCLReconClsModel(nn.Module):
    def __init__(
        self,
        radiomics_model: nn.Module,
        pathomics_encoder: nn.Module,
        pathomics_proj: nn.Module,
        recon_head: nn.Module,
        cls_head: nn.Module,
        gene_head: nn.Module,
        feature_cols: list[str],
    ):
        super().__init__()

        self.radiomics_model = radiomics_model
        self.pathomics_encoder = pathomics_encoder
        self.pathomics_proj = pathomics_proj
        self.recon_head = recon_head
        self.cls_head = cls_head
        self.gene_head = gene_head
        self.feature_cols = feature_cols

    def _to_dataframe(self, radiomics: torch.Tensor | pd.DataFrame) -> pd.DataFrame:
        if isinstance(radiomics, pd.DataFrame):
            return radiomics

        return pd.DataFrame(
            radiomics.detach().cpu().numpy(),
            columns=self.feature_cols,
        )

    def encode_radiomics(
        self,
        radiomics: torch.Tensor | pd.DataFrame,
    ) -> dict[str, torch.Tensor]:
        x_df = self._to_dataframe(radiomics)

        feat = self.radiomics_model.input_encoder(x_df)
        feat = self.radiomics_model.contrastive_token(**feat)
        feat = self.radiomics_model.cls_token(**feat)

        enc = self.radiomics_model.encoder(**feat)

        # sequence: [CLS, feature tokens..., CONTRASTIVE]
        rad_cls_h = enc[:, 0, :]
        rad_contrast_h = enc[:, -1, :]
        rad_contrast_z = self.radiomics_model.projection_head(rad_contrast_h)

        return {
            "rad_cls_h": rad_cls_h,
            "rad_contrast_h": rad_contrast_h,
            "rad_contrast_z": rad_contrast_z,
        }

    @torch.no_grad()
    def encode_pathomics_frozen(self, image: torch.Tensor) -> torch.Tensor:
        self.pathomics_encoder.eval()
        return self.pathomics_encoder(image)

    def encode_pathomics_projected(
        self,
        image: torch.Tensor,
        freeze_encoder: bool = True,
    ) -> dict[str, torch.Tensor]:
        if freeze_encoder:
            with torch.no_grad():
                path_cls = self.encode_pathomics_frozen(image)
        else:
            path_cls = self.pathomics_encoder(image)

        path_z = self.pathomics_proj(path_cls)

        return {
            "path_cls": path_cls,
            "path_z": path_z,
        }

    def forward_pretrain(
        self,
        image: torch.Tensor,
        radiomics: torch.Tensor | pd.DataFrame,
    ):
        rad = self.encode_radiomics(radiomics)
        path = self.encode_pathomics_projected(image, freeze_encoder=True)

        pred_radiomics = self.recon_head(rad["rad_contrast_z"])
        pred_class_logits = self.cls_head(rad["rad_cls_h"])

        return {
            **rad,
            **path,
            "pred_radiomics": pred_radiomics,
            "pred_class_logits": pred_class_logits,
        }

    def forward_gene(
        self,
        image: torch.Tensor,
        radiomics: torch.Tensor | pd.DataFrame,
    ):
        rad = self.encode_radiomics(radiomics)
        path = self.encode_pathomics_projected(image, freeze_encoder=False)

        fused = torch.cat(
            [path["path_z"], rad["rad_contrast_z"]],
            dim=1,
        )

        pred_gene = self.gene_head(fused)

        return {
            **rad,
            **path,
            "fused": fused,
            "pred_gene": pred_gene,
        }


def build_scratch_radiomics_model(device: torch.device):
    return build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=train.NUM_CELLTYPE_CLASSES,
        hidden_dropout_prob=train.DROPOUT,
        projection_dim=train.PROJECTION_DIM,
        activation=train.ACTIVATION,
        ape_drop_rate=0.0,
        device=device,
        num_sub_cols=train.NUM_SUB_COLS,
    ).to(device)


def build_radiomics_model(device: torch.device):
    rad_ckpt = train.RADTRANSTAB_PRETRAINED_DIR

    if rad_ckpt is not None:
        print(f"[INFO] load rad TransTab checkpoint: {rad_ckpt}")

    model = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=train.NUM_CELLTYPE_CLASSES,
        hidden_dropout_prob=train.DROPOUT,
        projection_dim=train.PROJECTION_DIM,
        activation=train.ACTIVATION,
        ape_drop_rate=0.0,
        device=device,
        num_sub_cols=train.NUM_SUB_COLS,
    ).to(device)

    if rad_ckpt is not None:
        load_radiomics_backbone_except_clf(
            model=model,
            checkpoint_path=rad_ckpt,
            device=device,
        )

    return model


def load_radiomics_backbone_except_clf(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    ckpt_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    state_dict = torch.load(ckpt_file, map_location=device)

    filtered = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("clf.fc.")
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print("[INFO] loaded rad TransTab backbone checkpoint")
    print("[INFO] skipped keys: clf.fc.*")
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)


def build_model(
    device: torch.device,
    num_genes: int,
    num_radiomics_features: int,
):
    radiomics_model = build_radiomics_model(device)

    pathomics_encoder = DenseNet121PathomicsEncoder(
        out_dim=train.PATHOMICS_DIM,
        pretrained=True,
    ).to(device)

    freeze_module(pathomics_encoder)

    pathomics_proj = MLPHead(
        in_dim=train.PATHOMICS_DIM,
        out_dim=train.PROJECTION_DIM,
        hidden_dim=train.PATH_PROJ_HIDDEN_DIM,
        dropout=train.HEAD_DROPOUT,
    ).to(device)

    recon_head = MLPHead(
        in_dim=train.PROJECTION_DIM,
        out_dim=num_radiomics_features,
        hidden_dim=train.RECON_HIDDEN_DIM,
        dropout=train.HEAD_DROPOUT,
    ).to(device)

    cls_head = MLPHead(
        in_dim=train.HIDDEN_DIM,
        out_dim=train.NUM_CELLTYPE_CLASSES,
        hidden_dim=train.CLS_HIDDEN_DIM,
        dropout=train.HEAD_DROPOUT,
    ).to(device)

    gene_head = MLPHead(
        in_dim=train.PROJECTION_DIM + train.PROJECTION_DIM,
        out_dim=num_genes,
        hidden_dim=train.GENE_HIDDEN_DIM,
        dropout=train.HEAD_DROPOUT,
    ).to(device)

    return MMCLReconClsModel(
        radiomics_model=radiomics_model,
        pathomics_encoder=pathomics_encoder,
        pathomics_proj=pathomics_proj,
        recon_head=recon_head,
        cls_head=cls_head,
        gene_head=gene_head,
        feature_cols=RADIOMICS_FEATURES_NAMES,
    ).to(device)

