from __future__ import annotations

import torch
import torch.nn as nn

from rapacl.model.rapacl import (
    MLPHead,
    MMCLReconClsModel,
    build_radiomics_model,
)
from rapacl.model.patchenc.build import build_patch_encoder
from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES
from rapacl.engines.trainer_utils import freeze_module

import rapacl.configs.default.train as train


class FrozenUNIEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        encoder, feat_dim = build_patch_encoder(
            backbone="uni",
            pretrained=True,
        )

        self.encoder = encoder
        self.out_dim = feat_dim

        freeze_module(self.encoder)
        self.encoder.eval()

    def train(self, mode: bool = True):
        # 항상 eval 유지
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        return self.encoder(image)


def build_uni_model(
    device: torch.device,
    num_genes: int,
    num_radiomics_features: int,
):
    radiomics_model = build_radiomics_model(device)

    pathomics_encoder = FrozenUNIEncoder().to(device)

    pathomics_proj = MLPHead(
        in_dim=pathomics_encoder.out_dim,
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
