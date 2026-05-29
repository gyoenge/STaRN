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
    batch_size: int = 32                # anchor + n_neighbors + n_globals
    n_neighbors: int = 6
    num_workers: int = 4

    # ── model ─────────────────────────────────────────────────────────────────
    num_features: int = 390             # radiomics feature dimension
    uni_dim: int = 1024                 # UNI ViT-L output dimension
    hidden_dim: int = 128
    num_col_layers: int = 2
    num_row_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 256
    proj_dim: int = 128
    dropout: float = 0.1

    # ── loss ──────────────────────────────────────────────────────────────────
    k_pos: int = 5                      # top-K UNI semantic neighbours → positive
    k_neg: int = 5                      # bottom-K UNI semantic remotes  → negative
    temperature: float = 0.1
    w_self: float = 1.0                 # weight for L_self
    w_col: float = 1.0                  # weight for L_col
    w_row: float = 0.5                  # weight for L_row (optional term)

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
