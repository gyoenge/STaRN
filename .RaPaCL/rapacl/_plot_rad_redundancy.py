# radiomics raw input에 대해 
# - PCA analysis 시각화 
# - feature간 correlation 분석 시각화 
# - raw radiomics space t-SNE / UMAP 시각화

# raw radiomics matrix를 모아서 → 결측/상수 feature 정리 → PCA/correlation/t-SNE/UMAP plot 저장 

"""
python -m rapacl._plot_rad_redundancy \
  --splits train \
  --folds 0,1,2,3 \
  --max_samples 400000
"""


from __future__ import annotations

import os
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    umap = None

from rapacl.configs.default.radiomics_columns import RADIOMICS_FEATURES_NAMES
from rapacl.engines.data_utils import DEFAULT_DATASET_STRUCTURE
import rapacl.configs.default.train as train


def load_radiomics_from_splits(
    root_dir: str,
    folds: list[int],
    split: str,
    feature_cols: list[str],
    radiomics_dir: str = "radiomics_features",
) -> pd.DataFrame:
    dfs = []

    split_cols = DEFAULT_DATASET_STRUCTURE["split_csv_cols"]
    rad_cfg = DEFAULT_DATASET_STRUCTURE["radiomics_data"]

    for fold in folds:
        split_path = os.path.join(
            root_dir,
            "splits",
            f"{split}_{fold}.csv",
        )

        split_df = pd.read_csv(split_path)

        for _, row in split_df.iterrows():
            sample_id = str(row[split_cols["sample_id"]])

            radiomics_path = os.path.join(
                root_dir,
                radiomics_dir,
                f"{sample_id}.{rad_cfg['format']}",
            )

            if not os.path.isfile(radiomics_path):
                print(f"[WARN] missing radiomics file: {radiomics_path}")
                continue

            rad_df = pd.read_parquet(radiomics_path)

            barcode_col = rad_cfg["barcode_col"]

            if barcode_col not in rad_df.columns:
                raise KeyError(
                    f"barcode column '{barcode_col}' "
                    f"not found in {radiomics_path}"
                )

            missing = [
                c for c in feature_cols
                if c not in rad_df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing radiomics columns in "
                    f"{radiomics_path}: {missing[:10]}"
                )

            rad = rad_df[feature_cols].copy()

            rad["sample_id"] = sample_id
            rad["fold"] = fold
            rad["split"] = split

            if barcode_col in rad_df.columns:
                rad["barcode"] = rad_df[barcode_col].astype(str)

            dfs.append(rad)

    if len(dfs) == 0:
        raise RuntimeError("No radiomics dataframe loaded.")

    return pd.concat(dfs, axis=0, ignore_index=True)


def clean_radiomics(df: pd.DataFrame, feature_cols: list[str]):
    x = df[feature_cols].copy()

    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    nan_ratio = x.isna().mean()
    valid_cols = nan_ratio[nan_ratio < 0.5].index.tolist()

    x = x[valid_cols]
    x = x.fillna(x.median())

    std = x.std(axis=0)
    valid_cols = std[std > 1e-8].index.tolist()
    x = x[valid_cols]

    meta = df.drop(columns=feature_cols, errors="ignore")

    return x, meta, valid_cols


def standardize_features(x: pd.DataFrame):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.values)
    return x_scaled


def plot_pca(x_scaled, meta, out_dir: str):
    pca = PCA(n_components=min(20, x_scaled.shape[1]), random_state=0)
    z = pca.fit_transform(x_scaled)

    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    # PCA scatter
    plt.figure(figsize=(7, 6))
    folds = sorted(meta["fold"].unique())

    for fold in folds:
        mask = meta["fold"].values == fold
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=8,
            alpha=0.55,
            label=f"fold {fold}",
        )

    plt.xlabel(f"PC1 ({explained[0] * 100:.2f}%)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.2f}%)")
    plt.title("PCA of Raw Radiomics Features")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=300)
    plt.close()

    # Scree plot
    plt.figure(figsize=(8, 5))
    xs = np.arange(1, len(explained) + 1)
    plt.plot(xs, explained, marker="o", label="Explained variance")
    plt.plot(xs, cum_explained, marker="o", label="Cumulative variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_explained_variance.png"), dpi=300)
    plt.close()

    np.save(os.path.join(out_dir, "pca_embedding.npy"), z)
    np.save(os.path.join(out_dir, "pca_explained_variance_ratio.npy"), explained)


def plot_correlation(x: pd.DataFrame, out_dir: str):
    corr = x.corr(method="pearson").values

    plt.figure(figsize=(10, 9))
    im = plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Feature-wise Pearson Correlation of Raw Radiomics")
    plt.xlabel("Radiomics Feature")
    plt.ylabel("Radiomics Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_correlation_heatmap.png"), dpi=300)
    plt.close()

    abs_corr = np.abs(corr)
    upper = abs_corr[np.triu_indices_from(abs_corr, k=1)]

    plt.figure(figsize=(7, 5))
    plt.hist(upper, bins=50)
    plt.xlabel("|Pearson correlation|")
    plt.ylabel("Feature pair count")
    plt.title("Distribution of Absolute Feature Correlations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_correlation_distribution.png"), dpi=300)
    plt.close()

    corr_df = pd.DataFrame(corr, index=x.columns, columns=x.columns)
    corr_df.to_csv(os.path.join(out_dir, "feature_correlation_matrix.csv"))


def random_subsample(x_scaled, meta, max_samples: int, seed: int = 0):
    n = x_scaled.shape[0]

    if n <= max_samples:
        return x_scaled, meta.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=max_samples, replace=False)
    indices = np.sort(indices)

    return x_scaled[indices], meta.iloc[indices].reset_index(drop=True)


def stratified_subsample_by_sample_id(
    x_scaled,
    meta,
    max_per_sample: int = 1000,
    seed: int = 0,
):
    if "sample_id" not in meta.columns:
        raise KeyError("sample_id not found in meta.")

    rng = np.random.default_rng(seed)
    selected = []

    for sample_id, idx in meta.groupby("sample_id").groups.items():
        idx = np.asarray(list(idx))

        if len(idx) > max_per_sample:
            idx = rng.choice(idx, size=max_per_sample, replace=False)

        selected.extend(idx.tolist())

    selected = np.array(sorted(selected))

    return x_scaled[selected], meta.iloc[selected].reset_index(drop=True)


def reduce_by_pca(x_vis, pca_dim: int, seed: int = 0):
    if pca_dim is None:
        return x_vis

    if x_vis.shape[1] <= pca_dim:
        return x_vis

    pca = PCA(n_components=pca_dim, random_state=seed)
    return pca.fit_transform(x_vis)


def plot_2d_embedding(
    z,
    meta,
    title: str,
    save_path: str,
    color_col: str = "fold",
    point_size: float = 2.0,
    alpha: float = 0.30,
):
    plt.figure(figsize=(8, 7))

    if color_col not in meta.columns:
        raise KeyError(f"{color_col} not found in meta.")

    groups = sorted(meta[color_col].dropna().unique())

    for group in groups:
        mask = meta[color_col].values == group
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=point_size,
            alpha=alpha,
            label=str(group),
        )

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend(markerscale=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def build_embedding_variants(
    x_scaled,
    meta,
    max_samples: int,
    max_per_sample: int,
    pca_dim: int,
):
    x_random, meta_random = random_subsample(
        x_scaled,
        meta,
        max_samples=max_samples,
    )

    x_strat, meta_strat = stratified_subsample_by_sample_id(
        x_scaled,
        meta,
        max_per_sample=max_per_sample,
    )

    return [
        {
            "name": "random",
            "x": x_random,
            "meta": meta_random,
            "use_pca": False,
        },
        {
            "name": f"random_pca{pca_dim}",
            "x": x_random,
            "meta": meta_random,
            "use_pca": True,
        },
        {
            "name": "stratified_sampleid",
            "x": x_strat,
            "meta": meta_strat,
            "use_pca": False,
        },
        {
            "name": f"stratified_sampleid_pca{pca_dim}",
            "x": x_strat,
            "meta": meta_strat,
            "use_pca": True,
        },
    ]


def plot_tsne_variants(
    x_scaled,
    meta,
    out_dir: str,
    max_samples: int = 5000,
    max_per_sample: int = 1000,
    pca_dim: int = 30,
):
    variants = build_embedding_variants(
        x_scaled=x_scaled,
        meta=meta,
        max_samples=max_samples,
        max_per_sample=max_per_sample,
        pca_dim=pca_dim,
    )

    for variant in variants:
        name = variant["name"]
        x_vis = variant["x"]
        meta_vis = variant["meta"]

        if variant["use_pca"]:
            x_input = reduce_by_pca(x_vis, pca_dim=pca_dim)
        else:
            x_input = x_vis

        print(f"[INFO] t-SNE variant: {name}, n={x_input.shape[0]}, dim={x_input.shape[1]}")

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=0,
        )

        z = tsne.fit_transform(x_input)

        plot_2d_embedding(
            z=z,
            meta=meta_vis,
            title=f"t-SNE of Raw Radiomics Space ({name})",
            save_path=os.path.join(out_dir, f"tsne_raw_radiomics_{name}_by_fold.png"),
            color_col="fold",
            point_size=2.0,
            alpha=0.25,
        )

        if "sample_id" in meta_vis.columns:
            plot_2d_embedding(
                z=z,
                meta=meta_vis,
                title=f"t-SNE of Raw Radiomics Space ({name})",
                save_path=os.path.join(out_dir, f"tsne_raw_radiomics_{name}_by_sample_id.png"),
                color_col="sample_id",
                point_size=2.0,
                alpha=0.25,
            )

        np.save(os.path.join(out_dir, f"tsne_embedding_{name}.npy"), z)


def plot_umap_variants(
    x_scaled,
    meta,
    out_dir: str,
    max_samples: int = 5000,
    max_per_sample: int = 1000,
    pca_dim: int = 30,
):
    if umap is None:
        print("[WARN] umap-learn is not installed. Skip UMAP.")
        return

    variants = build_embedding_variants(
        x_scaled=x_scaled,
        meta=meta,
        max_samples=max_samples,
        max_per_sample=max_per_sample,
        pca_dim=pca_dim,
    )

    for variant in variants:
        name = variant["name"]
        x_vis = variant["x"]
        meta_vis = variant["meta"]

        if variant["use_pca"]:
            x_input = reduce_by_pca(x_vis, pca_dim=pca_dim)
        else:
            x_input = x_vis

        print(f"[INFO] UMAP variant: {name}, n={x_input.shape[0]}, dim={x_input.shape[1]}")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=0,
        )

        z = reducer.fit_transform(x_input)

        plot_2d_embedding(
            z=z,
            meta=meta_vis,
            title=f"UMAP of Raw Radiomics Space ({name})",
            save_path=os.path.join(out_dir, f"umap_raw_radiomics_{name}_by_fold.png"),
            color_col="fold",
            point_size=2.0,
            alpha=0.25,
        )

        if "sample_id" in meta_vis.columns:
            plot_2d_embedding(
                z=z,
                meta=meta_vis,
                title=f"UMAP of Raw Radiomics Space ({name})",
                save_path=os.path.join(out_dir, f"umap_raw_radiomics_{name}_by_sample_id.png"),
                color_col="sample_id",
                point_size=2.0,
                alpha=0.25,
            )

        np.save(os.path.join(out_dir, f"umap_embedding_{name}.npy"), z)


def subsample(x_scaled, meta, max_samples: int):
    n = x_scaled.shape[0]

    if n <= max_samples:
        return x_scaled, meta.reset_index(drop=True)

    rng = np.random.default_rng(0)
    indices = rng.choice(n, size=max_samples, replace=False)
    indices = np.sort(indices)

    return x_scaled[indices], meta.iloc[indices].reset_index(drop=True)


def get_block_level1(feature: str) -> str:
    # morph 계열: morph_perimetersurfaceratio_maximum
    if feature.startswith("morph_"):
        parts = feature.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return "morph"

    # patch/cellseg 계열:
    # patch_original_glszm_GrayLevelNonUniformity
    # cellseg_all_original_firstorder_Energy
    parts = feature.split("_")

    if feature.startswith("patch_original_"):
        return "_".join(parts[:3])

    if feature.startswith("cellseg_all_original_"):
        return "_".join(parts[:4])

    return "other"


def get_block_level2(feature: str) -> str:
    if feature.startswith("patch_original_"):
        return "patch_original"

    if feature.startswith("cellseg_all_original_"):
        return "cellseg_all_original"

    if feature.startswith("morph_"):
        return "morph"

    return "other"


def get_sorted_features_by_block(
    features: list[str],
    level: int = 1,
) -> list[str]:
    if level == 1:
        key_fn = get_block_level1
    elif level == 2:
        key_fn = get_block_level2
    else:
        raise ValueError("level must be 1 or 2")

    return sorted(features, key=lambda f: (key_fn(f), f))


def plot_block_correlation_heatmap(
    x: pd.DataFrame,
    out_dir: str,
    level: int = 1,
    method: str = "pearson",
):
    if level == 1:
        block_fn = get_block_level1
    elif level == 2:
        block_fn = get_block_level2
    else:
        raise ValueError("level must be 1 or 2")

    sorted_features = get_sorted_features_by_block(
        list(x.columns),
        level=level,
    )

    x_sorted = x[sorted_features]
    corr = x_sorted.corr(method=method).values

    blocks = [block_fn(f) for f in sorted_features]

    boundary_positions = []
    block_centers = []
    block_labels = []

    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            end = i
            boundary_positions.append(end - 0.5)
            block_centers.append((start + end - 1) / 2)
            block_labels.append(blocks[start])
            start = i

    plt.figure(figsize=(14, 12))
    im = plt.imshow(
        corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)

    for pos in boundary_positions[:-1]:
        plt.axhline(pos, color="black", linewidth=0.8)
        plt.axvline(pos, color="black", linewidth=0.8)

    plt.xticks(
        block_centers,
        block_labels,
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        block_centers,
        block_labels,
        fontsize=7,
    )

    plt.title(f"Block-wise Radiomics Correlation Heatmap - Level {level}")
    plt.tight_layout()

    save_path = os.path.join(
        out_dir,
        f"feature_correlation_block_level{level}.png",
    )
    plt.savefig(save_path, dpi=300)
    plt.close()

    corr_df = pd.DataFrame(
        corr,
        index=sorted_features,
        columns=sorted_features,
    )
    corr_df.to_csv(
        os.path.join(
            out_dir,
            f"feature_correlation_block_level{level}.csv",
        )
    )

    block_df = pd.DataFrame(
        {
            "feature": sorted_features,
            "block": blocks,
        }
    )
    block_df.to_csv(
        os.path.join(
            out_dir,
            f"feature_block_assignment_level{level}.csv",
        ),
        index=False,
    )

    print(f"[INFO] saved block correlation heatmap: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=train.ROOT_DIR)
    parser.add_argument("--splits", type=str, default="train")
    parser.add_argument("--folds", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    folds = [int(x) for x in args.folds.split(",")]

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(
            train.OUTPUT_DIR,
            "radiomics_redundancy",
            args.splits,
        )

    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] root_dir: {args.root_dir}")
    print(f"[INFO] splits: {args.splits}")
    print(f"[INFO] folds: {folds}")
    print(f"[INFO] out_dir: {out_dir}")

    df = load_radiomics_from_splits(
        root_dir=args.root_dir,
        folds=folds,
        split=args.splits,
        feature_cols=RADIOMICS_FEATURES_NAMES,
    )

    x, meta, valid_cols = clean_radiomics(df, RADIOMICS_FEATURES_NAMES)
    x_scaled = standardize_features(x)

    print(f"[INFO] samples: {x.shape[0]}")
    print(f"[INFO] valid radiomics features: {x.shape[1]}")

    pd.Series(valid_cols).to_csv(
        os.path.join(out_dir, "valid_radiomics_features.csv"),
        index=False,
        header=["feature"],
    )

    print("[INFO] plot_pca")
    plot_pca(x_scaled, meta, out_dir)

    print("[INFO] plot_correlation")
    plot_correlation(x, out_dir)
    print("[INFO] plot_block_correlation_heatmap level1")
    plot_block_correlation_heatmap(
        x=x,
        out_dir=out_dir,
        level=1,
    )
    print("[INFO] plot_block_correlation_heatmap level2")
    plot_block_correlation_heatmap(
        x=x,
        out_dir=out_dir,
        level=2,
    )

    print("[INFO] plot_tsne_variants")
    plot_tsne_variants(
        x_scaled,
        meta,
        out_dir,
        max_samples=args.max_samples,
        max_per_sample=1000,
        pca_dim=30,
    )

    print("[INFO] plot_umap_variants")
    
    plot_umap_variants(
        x_scaled,
        meta,
        out_dir,
        max_samples=args.max_samples,
        max_per_sample=1000,
        pca_dim=30,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

