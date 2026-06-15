from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence
import hashlib

import h5py
import numpy as np
import torch
import torch.distributed as dist
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

    barcodes = (
        list(adata.obs[barcode_col].astype(str))
        if barcode_col and barcode_col in adata.obs.columns
        else list(adata.obs_names.astype(str))
    )

    needs_save = not cache_path.exists()
    if not needs_save:
        cached = np.load(cache_path, mmap_mode="r")
        if np.isnan(cached).any():
            needs_save = True

    if needs_save:
        # Fill NaN with per-column mean (PyRadiomics produces NaN for edge-case spots)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = col_means[np.where(nan_mask)[1]]
        np.save(cache_path, X)

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
        self._init_uni()
        self._init_scfoundation()
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

    def _init_uni(self):
        """Load pre-extracted UNI ViT-L embeddings (DATA_ROOT/embeddings/{sid}_uni.npy).

        File format (saved by .test/extract_uni.py):
            np.save(path, {"barcodes": np.array([...]), "X": np.ndarray (N, 1024)})
        """
        uni_path = self.root / "embeddings" / f"{self.sample_id}_uni.npy"
        if not uni_path.exists():
            self.uni_matrix = None
            self.uni_barcode_to_idx = {}
            return

        data = np.load(uni_path, allow_pickle=True).item()
        uni_barcodes = [str(b) for b in data["barcodes"]]
        self.uni_matrix = data["X"].astype(np.float32)
        self.uni_barcode_to_idx = {b: i for i, b in enumerate(uni_barcodes)}

    def _init_scfoundation(self):
        """Load pre-extracted scFoundation embeddings (DATA_ROOT/embeddings/{sid}_scfoundation.npy).

        File format (saved by .test/extract_scfoundation.py):
            np.save(path, {"barcodes": np.array([...]), "X": np.ndarray (N, 3072)})
        """
        sf_path = self.root / "embeddings" / f"{self.sample_id}_scfoundation.npy"
        if not sf_path.exists():
            self.scfoundation_matrix = None
            self.scfoundation_barcode_to_idx = {}
            return

        data = np.load(sf_path, allow_pickle=True).item()
        sf_barcodes = [str(b) for b in data["barcodes"]]
        self.scfoundation_matrix = data["X"].astype(np.float32)
        self.scfoundation_barcode_to_idx = {b: i for i, b in enumerate(sf_barcodes)}

    def _align_barcodes(self):
        sets = [
            set(self.patches_barcodes),
            set(self.st_barcodes),
            set(self.radiomics_barcodes),
        ]
        if self.uni_matrix is not None:
            sets.append(set(self.uni_barcode_to_idx.keys()))
        if self.scfoundation_matrix is not None:
            sets.append(set(self.scfoundation_barcode_to_idx.keys()))
        self.valid_barcodes = sorted(set.intersection(*sets))

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

        st        = torch.from_numpy(self.st_matrix[st_idx].copy())
        radiomics = torch.from_numpy(self.radiomics_matrix[radiomics_idx].copy())

        if self.uni_matrix is not None:
            uni_idx = self.uni_barcode_to_idx[barcode]
            uni_emb = torch.from_numpy(self.uni_matrix[uni_idx].copy())
        else:
            uni_emb = torch.zeros(1024)

        if self.scfoundation_matrix is not None:
            sf_idx = self.scfoundation_barcode_to_idx[barcode]
            scfoundation_emb = torch.from_numpy(self.scfoundation_matrix[sf_idx].copy())
        else:
            scfoundation_emb = torch.zeros(3072)

        return {
            "idx":              idx,
            "barcode":          barcode,
            "coord":            coord,
            "patch":            patch,
            "st":               st,
            "radiomics":        radiomics,
            "uni_emb":          uni_emb,
            "scfoundation_emb": scfoundation_emb,
        }

    def __del__(self):
        h5 = getattr(self, "patches_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass


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
    """여러 샘플(여러 data root에 걸칠 수 있음)을 하나의 Dataset으로 연결하는 wrapper.

    gene_names를 지정하지 않으면 hest.get_k_genes로 샘플 간 공통 유전자를 자동 선택한다.
    첫 로드 시 ST / radiomics X를 .npy 캐시로 저장하고, 이후 memmap으로 접근한다.

    Args:
        sources: (dataroot, sample_ids) 쌍의 목록. 서로 다른 데이터 root(예: Xenium / Visium)를
            섞어서 하나의 데이터셋으로 합칠 수 있다. radiomics feature 레이아웃(390-dim, 순서)이
            모든 source에서 동일해야 한다.
        gene_names: 사용할 gene 목록. None이면 n_genes / gene_criteria 기준으로 자동 선택.
        n_genes: gene_names=None일 때 선택할 유전자 수 (기본 250).
        gene_criteria: gene_names=None일 때 선택 기준 — 'var' | 'mean' (기본 'var').
        cache_dir: .npy 캐시 저장 경로. None이면 각 source의 dataroot/.cache 사용.
        transform: 패치 이미지 transform.
    """

    def __init__(
        self,
        sources: Sequence[tuple[str | Path, Sequence[str]]],
        gene_names: Optional[Sequence[str]] = None,
        n_genes: int = 250,
        gene_criteria: str = "var",
        cache_dir: Optional[Path] = None,
        transform=None,
    ):
        self.sources = [(Path(root), list(sids)) for root, sids in sources]
        self.sample_ids = [sid for _, sids in self.sources for sid in sids]

        if gene_names is None:
            st_paths = [
                root / "st" / f"{sid}.h5ad"
                for root, sids in self.sources
                for sid in sids
            ]
            gene_names = get_common_genes(st_paths, k=n_genes, criteria=gene_criteria)

        self.gene_names = list(gene_names)
        self.datasets = [
            _PersampleDataset(
                root, sid, self.gene_names, transform,
                Path(cache_dir) if cache_dir else root / ".cache",
            )
            for root, sids in self.sources
            for sid in sids
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

        [ anchor | n_neighbors spatial-kNN | n_semantic UNI-semantic-kNN | n_globals random ]

    so ``batch_size = 1 + n_neighbors + n_semantic + n_globals``.

    The spatial kNN graph is built from pixel / spatial coordinates.
    The semantic kNN graph is built from pre-extracted UNI embeddings (cosine similarity);
    FAISS is used when available, otherwise falls back to sklearn NearestNeighbors.
    Neighbours are always drawn from the *entire* dataset (cross-sample allowed for
    semantic neighbours; spatial neighbours remain within the same sample).

    Args:
        dataset:    An initialised ``HestRadiomicsDataset``.
        batch_size: Total number of spots per batch.
        n_neighbors: Spatial kNN neighbours per anchor.
        n_semantic:  UNI-similarity semantic neighbours per anchor (0 to disable).
        shuffle:    Permute anchor order each epoch.
        seed:       Base RNG seed.
    """

    def __init__(
        self,
        dataset: HestRadiomicsDataset,
        batch_size: int,
        n_neighbors: int,
        n_semantic: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        n_context = 1 + n_neighbors + n_semantic
        if batch_size <= n_context:
            raise ValueError(
                f"batch_size ({batch_size}) must be > 1 + n_neighbors + n_semantic ({n_context})"
            )
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.n_neighbors = n_neighbors
        self.n_semantic  = n_semantic
        self.n_globals   = batch_size - n_context
        self.shuffle     = shuffle
        self.seed        = seed
        self._epoch      = 0

        if dist.is_available() and dist.is_initialized():
            self.rank       = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank       = 0
            self.world_size = 1

        self._build_knn()
        if n_semantic > 0:
            self._build_semantic_index()
        else:
            self._semantic_knn = None

    # ── spatial kNN ───────────────────────────────────────────────────────────

    def _build_knn(self):
        """Precompute per-spot spatial kNN stored as global dataset indices."""
        from sklearn.neighbors import NearestNeighbors

        self._knn: list[np.ndarray] = []

        offset = 0
        for ds in self.dataset.datasets:
            n = len(ds)

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

    # ── semantic kNN (UNI cosine) ─────────────────────────────────────────────

    def _build_semantic_index(self):
        """Build ANN index over pre-extracted UNI embeddings (cosine similarity).

        Uses FAISS (IndexFlatIP on L2-normalised vectors) when available,
        otherwise falls back to sklearn NearestNeighbors with cosine metric.
        """
        # Collect UNI embeddings aligned to valid_barcodes order
        parts = []
        for ds in self.dataset.datasets:
            n = len(ds)
            if ds.uni_matrix is None:
                parts.append(np.zeros((n, 1024), dtype=np.float32))
            else:
                vecs = np.stack([
                    ds.uni_matrix[ds.uni_barcode_to_idx[bc]]
                    for bc in ds.valid_barcodes
                ])
                parts.append(vecs.astype(np.float32))

        all_uni = np.concatenate(parts, axis=0)   # (N_total, 1024)

        # L2 normalise for cosine similarity via inner product
        norms = np.linalg.norm(all_uni, axis=1, keepdims=True).clip(min=1e-8)
        all_uni_norm = all_uni / norms

        k = min(self.n_semantic, len(all_uni_norm) - 1)

        try:
            import faiss
            d = all_uni_norm.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(all_uni_norm)
            _, indices = index.search(all_uni_norm, k + 1)  # col 0 = self
            self._semantic_knn = [
                indices[i, 1 : k + 1].astype(np.int64)
                for i in range(len(all_uni_norm))
            ]
        except ImportError:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
            nn.fit(all_uni_norm)
            _, indices = nn.kneighbors(all_uni_norm)
            self._semantic_knn = [
                indices[i, 1 : k + 1].astype(np.int64)
                for i in range(len(all_uni_norm))
            ]

    # ── Sampler API ───────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int):
        """Advance the internal epoch counter so each epoch uses a different shuffle."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.world_size))

    def __iter__(self):
        rng     = np.random.default_rng(self.seed + self._epoch)
        n_total = len(self.dataset)

        order = rng.permutation(n_total) if self.shuffle else np.arange(n_total)
        order = order[self.rank :: self.world_size]  # disjoint anchor slice per DDP rank

        for anchor in map(int, order):
            # -- spatial neighbours --
            nbrs = self._knn[anchor]
            if len(nbrs) == 0:
                spatial = rng.integers(n_total, size=self.n_neighbors).tolist()
            elif len(nbrs) >= self.n_neighbors:
                spatial = rng.choice(nbrs, self.n_neighbors, replace=False).tolist()
            else:
                spatial = rng.choice(nbrs, self.n_neighbors, replace=True).tolist()

            # -- semantic neighbours --
            if self._semantic_knn is not None:
                sem_nbrs = self._semantic_knn[anchor]
                if len(sem_nbrs) >= self.n_semantic:
                    semantic = rng.choice(sem_nbrs, self.n_semantic, replace=False).tolist()
                else:
                    semantic = rng.choice(sem_nbrs, self.n_semantic, replace=True).tolist()
            else:
                semantic = []

            # -- random globals --
            globals_idx = rng.choice(n_total, self.n_globals, replace=False).tolist()

            yield [anchor] + spatial + semantic + globals_idx


# ── loader factory ────────────────────────────────────────────────────────────

def build_loader(
    dataset: HestRadiomicsDataset,
    batch_size: int,
    n_neighbors: int,
    n_semantic: int = 0,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Wrap a ``HestRadiomicsDataset`` with ``InductiveBatchSampler``.

    Args:
        dataset:     Initialised dataset.
        batch_size:  Total spots per batch (anchor + neighbours + globals).
        n_neighbors: Spatial kNN neighbours per anchor.
        n_semantic:  UNI-similarity semantic neighbours per anchor (0 to disable).
        num_workers: DataLoader worker processes.
        shuffle:     Shuffle anchors each epoch.
        seed:        Base RNG seed.

    Returns:
        A ``DataLoader`` configured with ``batch_sampler``.
    """
    sampler = InductiveBatchSampler(
        dataset,
        batch_size=batch_size,
        n_neighbors=n_neighbors,
        n_semantic=n_semantic,
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
