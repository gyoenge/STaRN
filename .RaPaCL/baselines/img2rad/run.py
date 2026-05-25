from __future__ import annotations

import json
from pathlib import Path

from baselines.common.config import apply_cli_overrides, load_yaml, parse_common_args
from baselines.common.logger import setup_logger
from baselines.common.utils import (
    ensure_dir,
    get_device,
    resolve_gene_list_path,
    save_yaml,
    seed_everything,
)
from baselines.img2rad.evaluator import run_all_folds_pcc_eval
from baselines.img2rad.trainer import run_all_folds_training


def run_train_mode(cfg: dict, gene_list_path: str, device, ckpt_dir: Path, logger) -> list:
    logger.info("=" * 80)
    logger.info("Start training")
    train_reports = run_all_folds_training(
        cfg=cfg,
        gene_list_path=gene_list_path,
        device=device,
        logger=logger,
    )
    with open(ckpt_dir / "train_reports.json", "w", encoding="utf-8") as f:
        json.dump(train_reports, f, indent=2)
    return train_reports


def run_eval_mode(
    cfg: dict,
    gene_list_path: str,
    device,
    timestamp: str,
    logger,
    train_reports: list | None = None,
):
    logger.info("=" * 80)
    logger.info("Start evaluation")

    radiomics_dim = cfg["model"].get("radiomics_dim")
    if radiomics_dim is None:
        if not train_reports:
            raise ValueError(
                "model.radiomics_dim is not set. "
                "Either run training first (mode=all) or set model.radiomics_dim in config."
            )
        radiomics_dim = int(train_reports[0]["radiomics_dim"])

    aggregate = run_all_folds_pcc_eval(
        cfg=cfg,
        gene_list_path=gene_list_path,
        radiomics_dim=int(radiomics_dim),
        device=device,
        timestamp=timestamp,
        logger=logger,
    )

    if aggregate is not None:
        logger.info(
            "Final macro mean PCC across folds = %.6f",
            aggregate["macro_mean_pcc_across_folds"],
        )
    return aggregate


def main() -> None:
    args = parse_common_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed_everything(cfg.get("seed", 42))

    paths_cfg = cfg.setdefault("paths", {})
    cfg.setdefault("runtime", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("cv", {})

    output_root = paths_cfg.get("output_root", "outputs/img2rad")
    log_dir = paths_cfg.get("log_dir", "logs/img2rad")

    timestamp, logger = setup_logger(log_dir=log_dir, name="img2rad")

    run_root = ensure_dir(Path(output_root) / f"run_{timestamp}")
    ckpt_dir = ensure_dir(run_root / "checkpoints")
    result_dir = ensure_dir(run_root / "results")  # noqa: F841

    cfg["paths"]["checkpoint_dir"] = str(ckpt_dir)

    save_yaml(cfg, run_root / "resolved_config.yaml")

    device = get_device(cfg["runtime"])
    gene_list_path = str(resolve_gene_list_path(paths_cfg, cfg["model"]))

    logger.info("Mode: %s", args.mode)
    logger.info("Device: %s", device)
    logger.info("gene_list_path: %s", gene_list_path)
    logger.info("Run root: %s", run_root)

    if "bench_data_root" not in paths_cfg:
        raise ValueError("paths.bench_data_root is required.")

    mode = getattr(args, "mode", "all")
    train_reports = None

    if mode in ("train", "all"):
        train_reports = run_train_mode(cfg, gene_list_path, device, ckpt_dir, logger)

    if mode in ("eval", "all"):
        run_eval_mode(cfg, gene_list_path, device, timestamp, logger, train_reports)

    logger.info("=" * 80)
    logger.info("Pipeline finished. Run root: %s", run_root)


if __name__ == "__main__":
    main()
