"""Embedding extraction for IDC_Others_Visium_10.

Extracts the per-spot embeddings required by model/teacher.py (AuxNeighborAttention)
and the semantic-kNN sampler (dataset/loader.py InductiveBatchSampler):

    - UNI ViT-L patch embeddings   -> embeddings/{sid}_uni.npy          (N, 1024)
    - scFoundation spot embeddings -> embeddings/{sid}_scfoundation.npy (N, 3072)

Output format (consumed by dataset/loader.py _PersampleDataset):
    np.save(path, {"barcodes": np.array([...]), "X": np.ndarray (N, D)})

Usage:
    cd /root/workspace/STaRN
    python extractemb.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import timm
import torch
from PIL import Image
from torchvision import transforms

# ── config ────────────────────────────────────────────────────────────────────

DATA_ROOT  = Path("/root/workspace/datasets/hest_radiomics/IDC_Others_Visium_10/IDC")
SAMPLE_IDS = [
    "NCBI681", "NCBI682", "NCBI683", "NCBI684", "NCBI776",
    "TENX13", "TENX14", "TENX39", "TENX53", "TENX68",
]

EMB_ROOT = DATA_ROOT / "embeddings"
EMB_ROOT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UNI
UNI_CKPT       = Path("/root/workspace/rapacl-results/checkpoints/UNI/vit_large_patch16_224.bin")
UNI_BATCH_SIZE = 128

# scFoundation
SCFOUND_DIR     = Path("/root/workspace/clones/scFoundation/model")
GENE_INDEX_PATH = SCFOUND_DIR / "OS_scRNA_gene_index.19264.tsv"
TGTHIGHRES      = "t4096"
VERSION         = "rde"
CKPT_NAME       = "01B-resolution"


# ── helpers ───────────────────────────────────────────────────────────────────

def _decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


# ── UNI extraction ───────────────────────────────────────────────────────────

def extract_uni():
    print("[UNI] loading model ...")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(UNI_CKPT, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    for sid in SAMPLE_IDS:
        out_path = EMB_ROOT / f"{sid}_uni.npy"
        if out_path.exists():
            print(f"[UNI] {sid} — already exists, skipping.")
            continue

        patch_path = DATA_ROOT / "patches" / f"{sid}.h5"
        print(f"\n[UNI] {sid} ...")

        with h5py.File(patch_path, "r") as f:
            barcodes = [_decode(b) for b in f["barcode"][:].reshape(-1)]
            imgs     = f["img"]
            n        = len(barcodes)

            feats = []
            for start in range(0, n, UNI_BATCH_SIZE):
                end        = min(start + UNI_BATCH_SIZE, n)
                batch_imgs = imgs[start:end]
                tensors    = torch.stack([tfm(Image.fromarray(img)) for img in batch_imgs])
                with torch.inference_mode():
                    out = model(tensors.to(DEVICE)).cpu().numpy()
                feats.append(out)

                if (start // UNI_BATCH_SIZE) % 5 == 0:
                    print(f"  {end}/{n}")

        embeddings = np.concatenate(feats, axis=0).astype(np.float32)
        np.save(out_path, {"barcodes": np.array(barcodes), "X": embeddings})
        print(f"  saved → {out_path}  shape: {embeddings.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── scFoundation extraction ──────────────────────────────────────────────────

def extract_scfoundation():
    sys.path.insert(0, str(SCFOUND_DIR))
    cwd = os.getcwd()
    os.chdir(SCFOUND_DIR)  # get_embedding.py reads ./OS_scRNA_gene_index.19264.tsv relatively
    try:
        from get_embedding import main_gene_selection

        gene_list = list(pd.read_csv(GENE_INDEX_PATH, header=0, delimiter="\t")["gene_name"])

        for sid in SAMPLE_IDS:
            out_path = EMB_ROOT / f"{sid}_scfoundation.npy"
            if out_path.exists():
                print(f"[scFoundation] {sid} — already exists, skipping.")
                continue

            st_path = DATA_ROOT / "st" / f"{sid}.h5ad"
            print(f"\n[scFoundation] {sid} ...")

            adata = sc.read_h5ad(st_path)
            adata.obs_names_make_unique()

            mask = ~(
                adata.var_names.str.startswith("NegControl") |
                adata.var_names.str.startswith("UnassignedCodeword")
            )
            adata = adata[:, mask].copy()

            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X_df = pd.DataFrame(X.astype(np.float32), index=adata.obs_names, columns=adata.var_names)

            X_aligned, to_fill_columns, _ = main_gene_selection(X_df, gene_list)
            print(f"  aligned: {X_aligned.shape}  filled: {len(to_fill_columns)}")

            expressed_mask = (X_aligned.values > 0).any(axis=1)
            n_dropped = (~expressed_mask).sum()
            if n_dropped > 0:
                print(f"  dropping {n_dropped} all-zero cells")
                X_aligned = X_aligned[expressed_mask]

            barcodes = list(X_aligned.index.astype(str))

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir    = Path(tmpdir)
                csv_path  = tmpdir / f"{sid}.csv"
                save_path = tmpdir / "out"
                save_path.mkdir()

                X_aligned.to_csv(csv_path)

                cmd = [
                    sys.executable,
                    str(SCFOUND_DIR / "get_embedding.py"),
                    "--task_name",      sid,
                    "--input_type",     "singlecell",
                    "--output_type",    "cell",
                    "--pool_type",      "all",
                    "--tgthighres",     TGTHIGHRES,
                    "--data_path",      str(csv_path),
                    "--save_path",      str(save_path),
                    "--pre_normalized", "F",
                    "--version",        VERSION,
                    "--ckpt_name",      CKPT_NAME,
                ]

                result = subprocess.run(cmd, cwd=str(SCFOUND_DIR), check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"[{sid}] get_embedding.py failed (exit {result.returncode})")

                out_files = list(save_path.glob("*.npy"))
                assert len(out_files) == 1, f"expected 1 .npy, got: {out_files}"
                embeddings = np.load(out_files[0]).astype(np.float32)

            np.save(out_path, {"barcodes": np.array(barcodes), "X": embeddings})
            print(f"  saved → {out_path}  shape: {embeddings.shape}")
    finally:
        os.chdir(cwd)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_uni()
    extract_scfoundation()
    print("\nDone.")
