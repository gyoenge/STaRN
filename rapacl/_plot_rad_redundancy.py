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


def plot_tsne(x_scaled, meta, out_dir: str, max_samples: int = 5000):
    x_vis, meta_vis = subsample(x_scaled, meta, max_samples)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=0,
    )
    z = tsne.fit_transform(x_vis)

    plot_2d_embedding(
        z=z,
        meta=meta_vis,
        title="t-SNE of Raw Radiomics Space",
        save_path=os.path.join(out_dir, "tsne_raw_radiomics.png"),
    )

    np.save(os.path.join(out_dir, "tsne_embedding.npy"), z)


def plot_umap(x_scaled, meta, out_dir: str, max_samples: int = 5000):
    if umap is None:
        print("[WARN] umap-learn is not installed. Skip UMAP.")
        return

    x_vis, meta_vis = subsample(x_scaled, meta, max_samples)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=0,
    )
    z = reducer.fit_transform(x_vis)

    plot_2d_embedding(
        z=z,
        meta=meta_vis,
        title="UMAP of Raw Radiomics Space",
        save_path=os.path.join(out_dir, "umap_raw_radiomics.png"),
    )

    np.save(os.path.join(out_dir, "umap_embedding.npy"), z)


def subsample(x_scaled, meta, max_samples: int):
    n = x_scaled.shape[0]

    if n <= max_samples:
        return x_scaled, meta.reset_index(drop=True)

    rng = np.random.default_rng(0)
    indices = rng.choice(n, size=max_samples, replace=False)
    indices = np.sort(indices)

    return x_scaled[indices], meta.iloc[indices].reset_index(drop=True)


def plot_2d_embedding(z, meta, title: str, save_path: str):
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

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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

    plot_pca(x_scaled, meta, out_dir)
    plot_correlation(x, out_dir)
    plot_tsne(x_scaled, meta, out_dir, max_samples=args.max_samples)
    plot_umap(x_scaled, meta, out_dir, max_samples=args.max_samples)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

