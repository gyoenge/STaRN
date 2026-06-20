# Baselines

Two ST gene expression prediction baselines built on a shared common library.

---

## Package Structure

```
baselines/
├── common/             # Shared utilities
│   ├── dataset.py      # STNetDataset (unified patch → gene dataset)
│   ├── metrics.py      # compute_genewise_pcc
│   ├── optimizer.py    # build_optimizer (sgd / adam / adamw)
│   ├── utils.py        # load_gene_names, get_device, resolve_split_path, ...
│   ├── config.py       # load_yaml, parse_common_args, apply_cli_overrides
│   └── logger.py       # setup_logger
├── stnet/              # STNet baseline
│   ├── run.py          # Entry point (train / eval / tuning modes)
│   ├── stnet.py        # Model definition
│   ├── trainer.py      # train_one_epoch, eval_fold, select_best_epoch, ...
│   ├── dataset.py      # Re-exports STNetDataset from common
│   └── _stnet_gene_analysis.py  # Per-gene PCC analysis + spatial plots
├── img2rad/            # Img2Rad baseline
│   ├── run.py          # Entry point (train / eval / all modes)
│   ├── main.py         # Thin wrapper → run.py (backward compat)
│   ├── model.py        # PatchImgEncoder, ImgToRadiomicsModel, FusionGeneModel
│   ├── trainer.py      # Two-stage training (img→rad, then rad+img→gene)
│   ├── evaluator.py    # Per-fold PCC evaluation + aggregation
│   ├── dataset.py      # RadiomicsTargetDataset, GeneWithRadiomicsDataset
│   ├── loader.py       # build_radiomics_dataloaders, build_gene_dataloaders
│   ├── cache.py        # Radiomics parquet loading + barcode alignment
│   ├── engine.py       # train_epoch, evaluate_loss, predict_all
│   └── inspect.py      # Parquet inspection utility
├── configs/
│   ├── stnet.yaml
│   └── img2rad.yaml
└── scripts/
    ├── run_img2rad_arch_ablation.sh
    └── run_img2rad_filter-prefix_ablation.sh
```

---

## STNet

Image-only baseline: DenseNet121 patch encoder → linear gene head.

### Run

```bash
cd /root/workspace/RaPaCL
```

**Train** (single fold, saves `final_model.pth`):
```bash
python -m baselines.stnet.run \
  --config ./baselines/configs/stnet.yaml \
  --mode train
```

**Eval** (loads checkpoint, computes per-gene PCC):
```bash
python -m baselines.stnet.run \
  --config ./baselines/configs/stnet.yaml \
  --mode eval
```

**Tuning** (LOO inner CV for epoch selection, then retrain + outer test):
```bash
python -m baselines.stnet.run \
  --config ./baselines/configs/stnet.yaml \
  --mode tuning
```

**Gene analysis** (per-gene PCC ranking + spatial expression maps):
```bash
python -m baselines.stnet._stnet_gene_analysis \
  --config ./baselines/configs/stnet.yaml \
  --folds 0,1,2,3 \
  --run_root /path/to/outputs/stnet/run_YYYYMMDD_HHMMSS
```

### Key Config (`configs/stnet.yaml`)

| Section | Key | Description |
|---|---|---|
| `paths` | `bench_data_root` | Root of HEST benchmark data |
| `paths` | `output_root` | Where run outputs are written |
| `paths` | `gene_list_path` | Override auto-resolved gene list path |
| `paths` | `train_split_csv` / `test_split_csv` | Override auto-resolved split paths |
| `model` | `backbone` | `densenet121` |
| `model` | `num_genes` | Number of HVGs to predict |
| `model` | `genes_criteria` | Gene selection criterion (e.g. `var`) |
| `model` | `pretrained` | Use ImageNet pretrained weights |
| `train` | `optimizer_name` | `sgd` / `adam` / `adamw` |
| `train` | `max_epochs` | Training epochs |
| `cv` | `outer_folds` | List of fold indices for tuning mode |
| `runtime` | `device` | `cuda` / `cpu` |
| `runtime` | `checkpoint_path` | Checkpoint to load in eval mode |

Split CSV paths are resolved automatically as `{bench_data_root}/splits/{train,test}_{fold}.csv` unless overridden.
Gene list path is resolved as `{bench_data_root}/{genes_criteria}_{num_genes}genes.json` unless overridden.

---

## Img2Rad

Two-stage baseline: (1) train DenseNet121 to predict radiomics features from patches (`ImgToRadiomicsModel`), (2) fuse image embedding + radiomics representation to predict gene expression (`FusionGeneModel`).

### Fusion modes

| `fusion_mode` | Description |
|---|---|
| `img_radpred` | concat(img_emb, predicted radiomics) |
| `img_radhidden` | concat(img_emb, intermediate radiomics hidden) |
| `img_rawrad` | concat(img_emb, ground-truth radiomics from parquet) |

### Run

```bash
cd /root/workspace/RaPaCL
```

**Train + eval** (full pipeline):
```bash
python -m baselines.img2rad.run \
  --config ./baselines/configs/img2rad.yaml \
  --mode all
```

**Train only**:
```bash
python -m baselines.img2rad.run \
  --config ./baselines/configs/img2rad.yaml \
  --mode train
```

**Eval only** (requires trained checkpoints in `paths.output_root`):
```bash
python -m baselines.img2rad.run \
  --config ./baselines/configs/img2rad.yaml \
  --mode eval
```

**Inspect radiomics parquet**:
```bash
python -m baselines.img2rad.inspect \
  --config ./baselines/configs/img2rad.yaml \
  --mode parquet \
  --show_columns
```

**Ablation studies**:
```bash
bash ./baselines/scripts/run_img2rad_arch_ablation.sh
bash ./baselines/scripts/run_img2rad_filter-prefix_ablation.sh
```

### Key Config (`configs/img2rad.yaml`)

| Section | Key | Description |
|---|---|---|
| `paths` | `bench_data_root` | Root of HEST benchmark data |
| `paths` | `output_root` | Where run outputs are written (`run_{timestamp}/` subdirs) |
| `paths` | `stnet_ckpt_dir` | Directory of pretrained STNet backbone weights |
| `model` | `fusion_mode` | `img_radpred` / `img_radhidden` / `img_rawrad` |
| `model` | `freeze_img2rad` | Freeze stage-1 weights during gene training |
| `model` | `radiomics_dim` | Override auto-inferred radiomics dimension |
| `model` | `radiomics_head_hidden_dims` | Hidden dims for img→rad MLP |
| `model` | `gene_head_hidden_dims` | Hidden dims for fusion→gene MLP |
| `train` | `num_epochs_img2rad` | Stage-1 training epochs |
| `train` | `num_epochs_gene` | Stage-2 training epochs |
| `cv` | `outer_folds` | List of fold indices to run |
| `data` | `radiomics_parquet_dir` | Directory of per-sample radiomics `.parquet` files |
| `data` | `radiomics_valid_prefixes` | PyRadiomics feature prefixes to include |
| `data` | `radiomics_apply_train_split_scaling` | Apply train-split z-score normalization to radiomics |

Each run creates a timestamped directory under `output_root`: `run_{timestamp}/checkpoints/`, `run_{timestamp}/results/`.
Radiomics features are aligned to patch barcodes at load time. Constant features (zero variance on train split) are automatically dropped.

---

## Common Module

Shared code used by both baselines:

- **`STNetDataset`** (`common/dataset.py`) — loads H5 patch images and `.h5ad` gene expression for a split CSV. Requires columns `patches_path`, `expr_path`; `sample_id` is optional (defaults to `"unknown"`). The `patch_meta` attribute (list of `{sample_id, barcode}` dicts) is used by the img2rad radiomics alignment pipeline.
- **`build_optimizer`** (`common/optimizer.py`) — creates SGD / Adam / AdamW from `cfg["train"]`. Used by both baselines.
- **`resolve_split_path`** (`common/utils.py`) — resolves train/test CSV path from config; falls back to `{bench_data_root}/splits/{kind}_{fold}.csv`.
- **`resolve_gene_list_path`** (`common/utils.py`) — resolves gene list JSON path from config; falls back to `{bench_data_root}/{criteria}_{n}genes.json`.
- **`compute_genewise_pcc`** (`common/metrics.py`) — computes per-gene Pearson correlation and mean PCC across all genes.
- **`load_gene_names`** (`common/utils.py`) — reads gene list from JSON (supports `genes`, `gene_names`, `var_genes` keys, or bare list).
- **`get_device`** (`common/utils.py`) — resolves `runtime.device` from config, falls back to CPU if CUDA unavailable.
