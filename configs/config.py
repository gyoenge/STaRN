from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # ── data ──────────────────────────────────────────────────────────────────
    data_root: Path = Path(
        "/root/workspace/datasets/hest_radiomics/IDC_Others_Xenium_11/IDC/"
    ).resolve()

    sample_ids: Tuple[str, ...] = (
        "TENX191", "TENX192", "TENX193",
        "TENX195", "TENX196", "TENX197", "TENX198", "TENX199",
        "TENX200", "TENX201", "TENX202",
    )

    n_genes: int = 250
    gene_criteria: str = "var"          # 'var' | 'mean'

    # ── loader ────────────────────────────────────────────────────────────────
    batch_size: int = 32                # anchor + n_neighbors + n_semantic + n_globals
    n_neighbors: int = 6               # spatial kNN neighbours per anchor
    n_semantic: int = 6                # UNI-similarity semantic neighbours per anchor
    num_workers: int = 4

    # ── model ─────────────────────────────────────────────────────────────────
    num_features: int = 390             # radiomics feature dimension
    uni_dim: int = 1024                 # UNI ViT-L output dimension
    scfoundation_dim: int = 3072        # scFoundation embedding dimension
    teacher_fuse_dim: int = 1024        # AuxNeighborAttention fused (UNI + scFoundation) dim
    hidden_dim: int = 128
    num_col_layers: int = 2
    num_row_layers: int = 1             # design spec: Row Attention Block × 1
    num_heads: int = 8
    ffn_dim: int = 256
    proj_dim: int = 128
    dropout: float = 0.1
    n_pos_bins: int = 32               # relative-PE distance quantisation bins

    # ── loss ──────────────────────────────────────────────────────────────────
    temperature: float = 0.1
    w_self: float = 1.0                # weight for L_self (NT-Xent)
    w_distill: float = 1.0            # weight for L_distill (Z^S ↔ Z^T cosine)

    # ── augmentation ──────────────────────────────────────────────────────────
    noise_std: float = 0.1
    mask_prob: float = 0.1

    # ── training ──────────────────────────────────────────────────────────────
    device: str = "cuda:0"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    save_dir: Path = Path("checkpoints")
    log_every: int = 10
