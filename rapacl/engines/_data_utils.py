# For HEST-IDC-Radiomics Dataset

import json
import os
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from hest.bench.st_dataset import H5PatchDataset, load_adata


DEFAULT_DATASET_STRUCTURE = {
    "barcode_col": "barcode",

    "split_csv_cols": {
        "patches": "patches_path",
        "expr": "expr_path",
        "sample_id": "sample_id",
    },

    "radiomics_data": {
        "dir": "radiomics_feature",
        "format": "parquet",
        "barcode_col": "barcode",
        "patch_idx_col": "patch_idx",
    },

    "patch_data": {
        "imgs_key": "imgs",
        "barcodes_key": "barcodes",
    },

    "sample_keys": {
        "image": "image",
        "gene": "gene",
        "radiomics": "radiomics",
        "barcode": "barcode",
        "patch_idx": "patch_idx",
        "sample_id": "sample_id",
        "target_label": "target_label",
        "target_distribution": "target_distribution",
    },

    "target": {
        "label_cols": ("target_label", "label", "celltype_label"),
        "distribution_col": "target_distribution",
    },
}


class HestRadiomicsDataset(torch.utils.data.Dataset):
    """
    HEST-IDC-Radiomics Dataset

    Returns:
        {
            "image": Tensor[C, H, W],
            "gene": Tensor[num_genes],
            "radiomics": Tensor[num_features],
            "target_label": LongTensor scalar,          # optional
            "target_distribution": Tensor[num_classes], # optional
            "barcode": str,
            "patch_idx": int,
            "sample_id": str,
        }
    """

    def __init__(
        self,
        bench_data_root: str,
        gene_list_path: str,
        feature_list_path: str,
        radiomics_dir: str = DEFAULT_DATASET_STRUCTURE["radiomics_data"]["dir"],
        split_csv_path: Optional[str] = None,
        split_df: Optional[pd.DataFrame] = None,
        transforms=None,
        normalize_gene: bool = True,
        radiomics_fillna: float = 0.0,
        radiomics_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if (split_csv_path is None) == (split_df is None):
            raise ValueError("Provide exactly one of split_csv_path or split_df.")

        self.split_df = (
            split_df.reset_index(drop=True).copy()
            if split_df is not None
            else pd.read_csv(split_csv_path)
        )

        with open(gene_list_path, "r", encoding="utf-8") as f:
            gene_info = json.load(f)
        self.genes = gene_info["genes"]

        with open(feature_list_path, "r", encoding="utf-8") as f:
            self.feature_cols = [line.strip() for line in f if line.strip()]

        self.bench_data_root = bench_data_root
        self.radiomics_dir = radiomics_dir
        self.transforms = transforms
        self.normalize_gene = normalize_gene
        self.radiomics_fillna = radiomics_fillna
        self.radiomics_dtype = radiomics_dtype

        self.samples: list[dict] = []

        self._build_samples()

    @staticmethod
    def _normalize_barcode(barcode) -> str:
        barcode = str(barcode)

        if barcode.startswith("b'") and barcode.endswith("'"):
            barcode = barcode[2:-1]

        barcode = barcode.replace("-1", "")

        if "_" in barcode:
            barcode = barcode.split("_")[-1]

        return barcode

    def _build_samples(self) -> None:
        split_cols = DEFAULT_DATASET_STRUCTURE["split_csv_cols"]
        rad_cfg = DEFAULT_DATASET_STRUCTURE["radiomics_data"]
        keys = DEFAULT_DATASET_STRUCTURE["sample_keys"]
        target_cfg = DEFAULT_DATASET_STRUCTURE["target"]

        for _, row in self.split_df.iterrows():
            patches_h5_path = os.path.join(
                self.bench_data_root,
                row[split_cols["patches"]],
            )
            expr_path = os.path.join(
                self.bench_data_root,
                row[split_cols["expr"]],
            )

            sample_id = self._infer_sample_id(row, patches_h5_path)

            radiomics_path = os.path.join(
                self.bench_data_root,
                self.radiomics_dir,
                f"{sample_id}.{rad_cfg['format']}",
            )

            self._check_file(patches_h5_path, "Patch")
            self._check_file(expr_path, "Expr")
            self._check_file(radiomics_path, "Radiomics")

            patch_items = self._load_patches(patches_h5_path)
            barcodes = [item[keys["barcode"]] for item in patch_items]

            gene_df = load_adata(
                expr_path,
                genes=self.genes,
                barcodes=barcodes,
                normalize=self.normalize_gene,
            )

            radiomics_df = pd.read_parquet(radiomics_path)

            rad_barcode_col = rad_cfg["barcode_col"]
            if rad_barcode_col not in radiomics_df.columns:
                raise KeyError(
                    f"Radiomics barcode column '{rad_barcode_col}' not found in {radiomics_path}. "
                    f"Available columns: {list(radiomics_df.columns)[:20]}"
                )

            radiomics_df[rad_barcode_col] = radiomics_df[rad_barcode_col].apply(
                self._normalize_barcode
            )
            radiomics_df = radiomics_df.set_index(rad_barcode_col)

            missing_features = [
                col for col in self.feature_cols if col not in radiomics_df.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Missing radiomics feature columns in {radiomics_path}: "
                    f"{missing_features[:10]}"
                )

            for i, item in enumerate(patch_items):
                barcode = self._normalize_barcode(item[keys["barcode"]])

                if barcode not in radiomics_df.index:
                    continue

                rad_row = radiomics_df.loc[barcode]

                radiomics_values = (
                    rad_row[self.feature_cols]
                    .astype(float)
                    .fillna(self.radiomics_fillna)
                    .values
                )

                sample = {
                    keys["image"]: item[keys["image"]],
                    keys["gene"]: torch.tensor(
                        gene_df.iloc[i].values,
                        dtype=torch.float32,
                    ),
                    keys["radiomics"]: torch.tensor(
                        radiomics_values,
                        dtype=self.radiomics_dtype,
                    ),
                    keys["barcode"]: barcode,
                    keys["patch_idx"]: self._get_patch_idx(rad_row, item),
                    keys["sample_id"]: sample_id,
                }

                for label_col in target_cfg["label_cols"]:
                    if label_col in rad_row.index:
                        sample[keys["target_label"]] = torch.tensor(
                            int(rad_row[label_col]),
                            dtype=torch.long,
                        )
                        break

                dist_col = target_cfg["distribution_col"]
                if dist_col in rad_row.index:
                    sample[keys["target_distribution"]] = self._parse_distribution(
                        rad_row[dist_col]
                    )

                self.samples.append(sample)

    def _get_patch_idx(self, rad_row: pd.Series, item: dict) -> int:
        rad_cfg = DEFAULT_DATASET_STRUCTURE["radiomics_data"]
        keys = DEFAULT_DATASET_STRUCTURE["sample_keys"]

        patch_idx_col = rad_cfg["patch_idx_col"]

        if patch_idx_col in rad_row.index:
            return int(rad_row[patch_idx_col])

        return int(item[keys["patch_idx"]])

    def _load_patches(self, patches_h5_path: str) -> list[dict]:
        patch_cfg = DEFAULT_DATASET_STRUCTURE["patch_data"]
        keys = DEFAULT_DATASET_STRUCTURE["sample_keys"]

        patch_dataset = H5PatchDataset(patches_h5_path)
        patch_items: list[dict] = []

        for i in range(len(patch_dataset)):
            chunk = patch_dataset[i]

            chunk_imgs = chunk[patch_cfg["imgs_key"]]
            chunk_barcodes = chunk[patch_cfg["barcodes_key"]]

            if isinstance(chunk_imgs, torch.Tensor):
                chunk_imgs = chunk_imgs.numpy()
            if isinstance(chunk_barcodes, torch.Tensor):
                chunk_barcodes = chunk_barcodes.numpy()

            if chunk_imgs.ndim == 3:
                chunk_imgs = np.expand_dims(chunk_imgs, axis=0)
                chunk_barcodes = [chunk_barcodes]

            for barcode, img in zip(chunk_barcodes, chunk_imgs):
                barcode_str = self._to_str_barcode(barcode)

                patch_items.append(
                    {
                        keys["image"]: img,
                        keys["barcode"]: barcode_str,
                        keys["patch_idx"]: len(patch_items),
                    }
                )

        return patch_items

    def _parse_distribution(self, value) -> torch.Tensor:
        if isinstance(value, str):
            value = json.loads(value)

        if isinstance(value, dict):
            value = list(value.values())

        value = np.asarray(value, dtype=np.float32)
        return torch.tensor(value, dtype=torch.float32)

    def _infer_sample_id(self, row: pd.Series, patches_h5_path: str) -> str:
        sample_id_col = DEFAULT_DATASET_STRUCTURE["split_csv_cols"]["sample_id"]

        if sample_id_col in row and pd.notna(row[sample_id_col]):
            return str(row[sample_id_col])

        filename = os.path.basename(patches_h5_path)
        return os.path.splitext(filename)[0]

    @staticmethod
    def _to_str_barcode(barcode) -> str:
        if isinstance(barcode, bytes):
            return barcode.decode("utf-8")

        if isinstance(barcode, np.ndarray):
            barcode = barcode.item()
            if isinstance(barcode, bytes):
                return barcode.decode("utf-8")

        return str(barcode)

    @staticmethod
    def _check_file(path: str, name: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        keys = DEFAULT_DATASET_STRUCTURE["sample_keys"]

        sample = self.samples[idx].copy()
        img = sample[keys["image"]]

        if self.transforms is not None:
            img = self.transforms(img)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(np.array(img))

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        img = img.float()

        if img.max() > 1.0:
            img = img / 255.0

        sample[keys["image"]] = img
        return sample


def build_dataset(split_csv_path: str):
    return HestRadiomicsDataset(
        bench_data_root=train.ROOT_DIR,
        split_csv_path=split_csv_path,
        gene_list_path=train.GENE_LIST_PATH,
        feature_list_path=train.FEATURE_LIST_PATH,
        radiomics_dir=getattr(
            train,
            "RADIOMICS_DIR",
            DEFAULT_DATASET_STRUCTURE["radiomics_data"]["dir"],
        ),
    )


def build_loader(dataset, shuffle: bool, drop_last: bool = False):
    return DataLoader(
        dataset,
        batch_size=train.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=train.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_batch_tensor(
    batch: dict[str, Any],
    names: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    for name in names:
        if name in batch:
            return batch[name].to(device, non_blocking=True).float()

    raise KeyError(
        f"None of keys {names} found in batch. "
        f"Available keys: {list(batch.keys())}"
    )


def get_target_label(
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    for name in DEFAULT_DATASET_STRUCTURE["target"]["label_cols"]:
        if name in batch:
            return batch[name].to(device, non_blocking=True).long()

    raise KeyError(
        f"target label key not found. "
        f"Available keys: {list(batch.keys())}"
    )

