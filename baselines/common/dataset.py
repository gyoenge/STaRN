from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hest.bench.st_dataset import H5PatchDataset, load_adata

from .utils import load_gene_names


class STNetDataset(Dataset):
    """
    Patch image -> gene expression dataset shared by stnet and img2rad.

    Required split CSV columns: patches_path, expr_path.
    Optional column: sample_id (defaults to "unknown" if absent).
    """

    def __init__(
        self,
        bench_data_root: str,
        gene_list_path: str,
        split_csv_path: Optional[str] = None,
        split_df: Optional[pd.DataFrame] = None,
        transforms=None,
        normalize_expression: bool = True,
    ) -> None:
        super().__init__()

        if (split_csv_path is None) == (split_df is None):
            raise ValueError("Provide exactly one of split_csv_path or split_df.")

        self.bench_data_root = bench_data_root
        self.split_df = (
            split_df.reset_index(drop=True).copy()
            if split_df is not None
            else pd.read_csv(split_csv_path)
        )

        self.genes = load_gene_names(gene_list_path)
        self.transforms = transforms
        self.normalize_expression = normalize_expression

        self.images: list[np.ndarray] = []
        self.targets: list[torch.Tensor] = []
        self.patch_meta: list[dict[str, str]] = []

        self._build_samples()

    def _build_samples(self) -> None:
        has_sample_id = "sample_id" in self.split_df.columns

        for _, row in self.split_df.iterrows():
            sample_id = str(row["sample_id"]) if has_sample_id else "unknown"
            patches_h5_path = os.path.join(self.bench_data_root, row["patches_path"])
            expr_path = os.path.join(self.bench_data_root, row["expr_path"])

            if not os.path.isfile(patches_h5_path):
                raise FileNotFoundError(f"Patch file not found: {patches_h5_path}")
            if not os.path.isfile(expr_path):
                raise FileNotFoundError(f"Expr file not found: {expr_path}")

            patch_dataset = H5PatchDataset(patches_h5_path)

            slide_imgs: list[np.ndarray] = []
            slide_barcodes: list[str] = []

            for i in range(len(patch_dataset)):
                item = patch_dataset[i]
                chunk_imgs = item["imgs"]
                chunk_barcodes = item["barcodes"]

                if isinstance(chunk_imgs, torch.Tensor):
                    chunk_imgs = chunk_imgs.numpy()
                if isinstance(chunk_barcodes, torch.Tensor):
                    chunk_barcodes = chunk_barcodes.numpy()

                if chunk_imgs.ndim == 3:
                    chunk_imgs = np.expand_dims(chunk_imgs, axis=0)
                    chunk_barcodes = [chunk_barcodes]

                for barcode, img in zip(chunk_barcodes, chunk_imgs):
                    barcode_str = (
                        barcode.decode("utf-8") if isinstance(barcode, bytes) else str(barcode)
                    )
                    slide_barcodes.append(barcode_str)
                    slide_imgs.append(img)

            adata_df = load_adata(
                expr_path,
                genes=self.genes,
                barcodes=slide_barcodes,
                normalize=self.normalize_expression,
            )

            for j in range(len(slide_imgs)):
                self.images.append(slide_imgs[j])
                self.targets.append(
                    torch.tensor(adata_df.iloc[j].values, dtype=torch.float32)
                )
                self.patch_meta.append({"sample_id": sample_id, "barcode": slide_barcodes[j]})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        target = self.targets[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(np.array(img))

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0

        return img, target
