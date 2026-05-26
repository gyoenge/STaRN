from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence
import hashlib

import h5py
import numpy as np
import torch
import scanpy as sc
from PIL import Image
import bisect

from torch.utils.data import (
    Dataset,
    ConcatDataset,
    DataLoader,
    Sampler,
)
from torchvision import transforms


# ── cache helpers ─────────────────────────────────────────────────────────────

def _gene_hash(gene_names: Sequence[str]) -> str:
    """gene 목록의 8자리 해시 — 캐시 파일명 구분용."""
    return hashlib.md5(",".join(sorted(gene_names)).encode()).hexdigest()[:8]


def _load_or_build_cache(
    h5ad_path: Path,
    cache_path: Path,
    gene_names: Optional[List[str]],
    barcode_col: Optional[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """h5ad를 읽어 X를 float32 .npy로 저장하고 memmap으로 반환한다.

    Args:
        h5ad_path: 원본 .h5ad 경로.
        cache_path: 저장할 .npy 경로.
        gene_names: subset할 gene 목록. None이면 전체 var 사용.
        barcode_col: obs에서 barcode를 읽을 컬럼명. None이면 obs.index 사용.

    Returns:
        (memmap array shape (N, F), barcode list)
    """
    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()

    if gene_names is not None:
        adata = adata[:, list(gene_names)].copy()

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)

    if not cache_path.exists():
        np.save(cache_path, X)

    barcodes = (
        list(adata.obs[barcode_col].astype(str))
        if barcode_col and barcode_col in adata.obs.columns
        else list(adata.obs_names.astype(str))
    )

    return np.load(cache_path, mmap_mode="r"), barcodes


# ── dataset ───────────────────────────────────────────────────────────────────

class _PersampleDataset(Dataset):
    def __init__(
        self,
        dataroot: str | Path,
        sample_id: str,
        gene_names: Optional[Sequence[str]] = None,
        transform=None,
        cache_dir: Optional[Path] = None,
    ):
        self.root = Path(dataroot)
        self.sample_id = sample_id
        self.gene_names = list(gene_names) if gene_names is not None else None

        self.patches_path  = self.root / "patches"  / f"{sample_id}.h5"
        self.st_path       = self.root / "st"        / f"{sample_id}.h5ad"
        self.radiomics_path = self.root / "radiomics" / f"{sample_id}.h5ad"

        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.patches_h5 = None

        self._cache_dir = Path(cache_dir) if cache_dir else self.root / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._init_patches()
        self._init_st()
        self._init_radiomics()
        self._align_barcodes()

    # ── init ──────────────────────────────────────────────────────────────────

    def _init_patches(self):
        with h5py.File(self.patches_path, "r") as f:
            self.patches_barcodes = [
                b.decode() if isinstance(b, bytes) else str(b)
                for b in f["barcode"][:].reshape(-1)
            ]
            if "coords" in f:
                self.patch_coords = f["coords"][:]
            elif "coord" in f:
                self.patch_coords = f["coord"][:]
            elif "spatial" in f:
                self.patch_coords = f["spatial"][:]
            else:
                self.patch_coords = None

        self.patch_barcode_to_idx = {b: i for i, b in enumerate(self.patches_barcodes)}

    def _init_st(self):
        ghash = _gene_hash(self.gene_names) if self.gene_names else "full"
        cache_path = self._cache_dir / f"{self.sample_id}_st_{ghash}.npy"

        self.st_matrix, self.st_barcodes = _load_or_build_cache(
            h5ad_path=self.st_path,
            cache_path=cache_path,
            gene_names=self.gene_names,
            barcode_col=None,  # ST: obs.index = barcode
        )
        self.st_barcode_to_idx = {b: i for i, b in enumerate(self.st_barcodes)}

        # spatial coords — lightweight, keep in RAM
        adata = sc.read_h5ad(self.st_path)
        if "spatial" in adata.obsm:
            self.st_coords = adata.obsm["spatial"]
        elif {"coord_x", "coord_y"}.issubset(adata.obs.columns):
            self.st_coords = adata.obs[["coord_x", "coord_y"]].to_numpy()
        else:
            self.st_coords = None

    def _init_radiomics(self):
        cache_path = self._cache_dir / f"{self.sample_id}_radiomics.npy"

        self.radiomics_matrix, self.radiomics_barcodes = _load_or_build_cache(
            h5ad_path=self.radiomics_path,
            cache_path=cache_path,
            gene_names=None,        # radiomics: 전체 390 feature 사용
            barcode_col="barcode",  # radiomics: obs['barcode'] 컬럼
        )
        self.radiomics_barcode_to_idx = {b: i for i, b in enumerate(self.radiomics_barcodes)}

    def _align_barcodes(self):
        patch_set = set(self.patches_barcodes)
        st_set    = set(self.st_barcodes)
        rad_set   = set(self.radiomics_barcodes)
        self.valid_barcodes = sorted(patch_set & st_set & rad_set)

    # ── dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.valid_barcodes)

    def _open_patch_h5(self):
        if self.patches_h5 is None:
            self.patches_h5 = h5py.File(self.patches_path, "r")

    def __getitem__(self, idx):
        self._open_patch_h5()

        barcode      = self.valid_barcodes[idx]
        patch_idx    = self.patch_barcode_to_idx[barcode]
        st_idx       = self.st_barcode_to_idx[barcode]
        radiomics_idx = self.radiomics_barcode_to_idx[barcode]

        if "img" in self.patches_h5:
            patch_arr = self.patches_h5["img"][patch_idx]
        elif "imgs" in self.patches_h5:
            patch_arr = self.patches_h5["imgs"][patch_idx]
        elif "patches" in self.patches_h5:
            patch_arr = self.patches_h5["patches"][patch_idx]
        else:
            raise KeyError(f"patch image key not found. keys={list(self.patches_h5.keys())}")

        patch = self.transform(Image.fromarray(patch_arr))

        if self.patch_coords is not None:
            coord = torch.tensor(self.patch_coords[patch_idx], dtype=torch.float32)
        elif self.st_coords is not None:
            coord = torch.tensor(self.st_coords[st_idx], dtype=torch.float32)
        else:
            coord = torch.tensor([-1.0, -1.0])

        # memmap 행 → tensor (복사 최소화)
        st        = torch.from_numpy(self.st_matrix[st_idx].copy())
        radiomics = torch.from_numpy(self.radiomics_matrix[radiomics_idx].copy())

        return {
            "idx":       idx,
            "barcode":   barcode,
            "coord":     coord,
            "patch":     patch,
            "st":        st,
            "radiomics": radiomics,
        }

    def __del__(self):
        if getattr(self, "patches_h5", None) is not None:
            self.patches_h5.close()


# ── gene helpers ──────────────────────────────────────────────────────────────

def get_common_genes(
    st_paths: Sequence[Path],
    k: int = 250,
    criteria: str = "var",
) -> List[str]:
    """샘플 간 공통 유전자 중 상위 k개를 반환한다.

    Args:
        st_paths: 각 샘플의 ST .h5ad 파일 경로 목록.
        k: 선택할 유전자 수.
        criteria: 'var' (발현 분산) | 'mean' (평균 발현량).

    Returns:
        상위 k개 공통 유전자 이름 목록.
    """
    from hest import get_k_genes

    adatas = [sc.read_h5ad(p) for p in st_paths]
    return get_k_genes(adatas, k=k, criteria=criteria)


# ── public dataset ────────────────────────────────────────────────────────────

class HestRadiomicsDataset(Dataset):
    """여러 샘플을 하나의 Dataset으로 연결하는 wrapper.

    gene_names를 지정하지 않으면 hest.get_k_genes로 샘플 간 공통 유전자를 자동 선택한다.
    첫 로드 시 ST / radiomics X를 .npy 캐시로 저장하고, 이후 memmap으로 접근한다.

    Args:
        dataroot: 데이터 루트 디렉토리.
        sample_ids: 불러올 sample ID 목록.
        gene_names: 사용할 gene 목록. None이면 n_genes / gene_criteria 기준으로 자동 선택.
        n_genes: gene_names=None일 때 선택할 유전자 수 (기본 250).
        gene_criteria: gene_names=None일 때 선택 기준 — 'var' | 'mean' (기본 'var').
        cache_dir: .npy 캐시 저장 경로. None이면 dataroot/.cache 사용.
        transform: 패치 이미지 transform.
    """

    def __init__(
        self,
        dataroot: str | Path,
        sample_ids: Sequence[str],
        gene_names: Optional[Sequence[str]] = None,
        n_genes: int = 250,
        gene_criteria: str = "var",
        cache_dir: Optional[Path] = None,
        transform=None,
    ):
        self.sample_ids = list(sample_ids)
        dataroot = Path(dataroot)
        cache_dir = Path(cache_dir) if cache_dir else dataroot / ".cache"

        if gene_names is None:
            st_paths = [dataroot / "st" / f"{sid}.h5ad" for sid in self.sample_ids]
            gene_names = get_common_genes(st_paths, k=n_genes, criteria=gene_criteria)

        self.gene_names = list(gene_names)
        self.datasets = [
            _PersampleDataset(dataroot, sid, self.gene_names, transform, cache_dir)
            for sid in self.sample_ids
        ]
        self._concat = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx = bisect.bisect_right(self._concat.cumulative_sizes, idx)
        item = self._concat[idx]
        item["sample_id"] = self.sample_ids[dataset_idx]
        return item

    def __repr__(self) -> str:
        lines = [f"HestRadiomicsDataset(n_samples={len(self.datasets)}, n_spots={len(self)})"]
        for sid, ds in zip(self.sample_ids, self.datasets):
            lines.append(f"  {sid}: {len(ds)} spots")
        return "\n".join(lines)


class InductiveBatchSampler(Sampler):
    """Batch sampler for neighborhood-aware contrastive training.

    Each batch has exactly ``batch_size`` spots laid out as:

        [ anchor | n_neighbors spatial-kNN spots | n_globals random spots ]

    so ``batch_size = 1 + n_neighbors + n_globals``.

    The kNN graph is built once at construction from the pixel / spatial
    coordinates stored in each ``_PersampleDataset``.  Neighbours are
    always drawn from *within the same sample* so they are genuinely
    spatially adjacent.  Global negatives are drawn from the entire
    concatenated dataset.

    Args:
        dataset: An initialised ``HestRadiomicsDataset``.
        batch_size: Total number of spots per batch.
        n_neighbors: Spatial kNN neighbours per anchor.
        shuffle: Permute anchor order each epoch.  Call ``set_epoch``
            before each epoch when training with multiple epochs.
        seed: Base RNG seed.  Epoch number is added to it automatically.
    """

    def __init__(
        self,
        dataset: HestRadiomicsDataset,
        batch_size: int,
        n_neighbors: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if batch_size <= n_neighbors + 1:
            raise ValueError(
                f"batch_size ({batch_size}) must be > n_neighbors + 1 ({n_neighbors + 1})"
            )
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.n_neighbors = n_neighbors
        self.n_globals   = batch_size - 1 - n_neighbors
        self.shuffle     = shuffle
        self.seed        = seed
        self._epoch      = 0

        self._build_knn()

    # ── kNN graph ──────────────────────────────────────────────────────────────

    def _build_knn(self):
        """Precompute per-spot spatial kNN stored as global dataset indices."""
        from sklearn.neighbors import NearestNeighbors

        # _knn[global_idx] = int64 array of global neighbour indices
        self._knn: list[np.ndarray] = []

        offset = 0
        for ds in self.dataset.datasets:
            n = len(ds)

            # Coordinates aligned to valid_barcodes order
            if ds.patch_coords is not None:
                coords = np.stack([
                    ds.patch_coords[ds.patch_barcode_to_idx[bc]]
                    for bc in ds.valid_barcodes
                ])
            elif ds.st_coords is not None:
                coords = np.stack([
                    ds.st_coords[ds.st_barcode_to_idx[bc]]
                    for bc in ds.valid_barcodes
                ])
            else:
                # No spatial information: empty neighbour arrays
                self._knn.extend([np.empty(0, dtype=np.int64)] * n)
                offset += n
                continue

            k = min(self.n_neighbors, n - 1)
            _, indices = (
                NearestNeighbors(n_neighbors=k + 1)
                .fit(coords)
                .kneighbors(coords)
            )   # (n, k+1) — column 0 is self

            for local_idx in range(n):
                global_nbrs = (indices[local_idx, 1 : k + 1] + offset).astype(np.int64)
                self._knn.append(global_nbrs)

            offset += n

    # ── Sampler API ───────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int):
        """Advance the internal epoch counter so each epoch uses a different shuffle."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        rng     = np.random.default_rng(self.seed + self._epoch)
        n_total = len(self.dataset)

        order = rng.permutation(n_total) if self.shuffle else np.arange(n_total)

        for anchor in map(int, order):
            nbrs = self._knn[anchor]

            # -- spatial neighbours --
            if len(nbrs) == 0:
                # fallback when no coordinates are available
                neighbors = rng.integers(n_total, size=self.n_neighbors).tolist()
            elif len(nbrs) >= self.n_neighbors:
                neighbors = rng.choice(nbrs, self.n_neighbors, replace=False).tolist()
            else:
                # fewer stored neighbours than requested → sample with replacement
                neighbors = rng.choice(nbrs, self.n_neighbors, replace=True).tolist()

            # -- random globals --
            # Collision with anchor/neighbours is negligible at scale (k << N)
            globals_idx = rng.choice(n_total, self.n_globals, replace=False).tolist()

            yield [anchor] + neighbors + globals_idx


# ── loader factory ────────────────────────────────────────────────────────────

def build_loader(
    dataset: HestRadiomicsDataset,
    batch_size: int,
    n_neighbors: int,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Wrap a ``HestRadiomicsDataset`` with ``InductiveBatchSampler``.

    Args:
        dataset: Initialised dataset.
        batch_size: Total spots per batch (anchor + neighbours + globals).
        n_neighbors: Spatial kNN neighbours per anchor.
        num_workers: DataLoader worker processes.
        shuffle: Shuffle anchors each epoch.
        seed: Base RNG seed.

    Returns:
        A ``DataLoader`` configured with ``batch_sampler``.
    """
    sampler = InductiveBatchSampler(
        dataset,
        batch_size=batch_size,
        n_neighbors=n_neighbors,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
