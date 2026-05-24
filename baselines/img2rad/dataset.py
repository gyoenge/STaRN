from __future__ import annotations

import torch
from torch.utils.data import Dataset

from baselines.common.dataset import STNetDataset

__all__ = ["STNetDataset", "RadiomicsTargetDataset", "GeneWithRadiomicsDataset"]


class RadiomicsTargetDataset(Dataset):
    def __init__(self, base_dataset: STNetDataset, radiomics_targets: torch.Tensor):
        if len(base_dataset) != len(radiomics_targets):
            raise ValueError(
                f"Length mismatch: len(base_dataset)={len(base_dataset)}, "
                f"len(radiomics_targets)={len(radiomics_targets)}"
            )
        self.base_dataset = base_dataset
        self.radiomics_targets = radiomics_targets

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, _ = self.base_dataset[idx]
        target = self.radiomics_targets[idx]
        return img, target


class GeneWithRadiomicsDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        radiomics_features: torch.Tensor,
    ):
        if len(base_dataset) != len(radiomics_features):
            raise ValueError(
                f"Length mismatch: len(base_dataset)={len(base_dataset)}, "
                f"len(radiomics_features)={len(radiomics_features)}"
            )
        self.base_dataset = base_dataset
        self.radiomics_features = radiomics_features

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, gene_target = self.base_dataset[idx]
        raw_radiomics = self.radiomics_features[idx]
        return img, raw_radiomics, gene_target
