#!/usr/bin/env python3
"""Run the verified L1W inference + evaluation workflow.

This script wraps the workflow described in:
  - docs/segmentation_pipeline.md
  - docs/evaluation.md

It supports two L1W modes:
  1) raw_forest: doc-faithful pipeline on data/pipeline/L1W/forest/L1W.laz
     and evaluation against data/benchmark/L1W_voxelized01_for_eval.laz.
  2) eval_subset: quick sanity mode that runs the pipeline directly on
     data/benchmark/L1W_voxelized01_for_eval.laz and evaluates on the same GT.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit(
        "PyYAML is required for tools/pipeline/run_l1w_workflow.py. "
        "Please activate the TreeLearn conda environment before running it."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_BASE_CFG = REPO_ROOT / "configs" / "pipeline" / "pipeline.yaml"
EVAL_BASE_CFG = REPO_ROOT / "configs" / "evaluation" / "evaluate.yaml"
GENERATED_CFG_DIR = REPO_ROOT / "work_dirs" / "generated_l1w_workflow"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def repo_rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def default_results_dir(mode: str) -> str:
    if mode == "raw_forest":
        return "results_l1w_raw"
    if mode == "eval_subset":
        return "results_l1w_eval_subset"
    raise ValueError(f"Unsupported mode: {mode}")


def resolve_checkpoint(user_value: str | None) -> Path:
    if user_value:
        ckpt = Path(user_value)
        return ckpt if ckpt.is_absolute() else (REPO_ROOT / ckpt)

    base_cfg = load_yaml(PIPELINE_BASE_CFG)
    ckpt = Path(base_cfg["pretrain"])
    return ckpt if ckpt.is_absolute() else (REPO_ROOT / ckpt)


def build_mode_paths(mode: str, results_dir_name: str) -> dict[str, Path]:
    if mode == "raw_forest":
        pipeline_input = REPO_ROOT / "data" / "pipeline" / "L1W" / "forest" / "L1W.laz"
        gt_forest = REPO_ROOT / "data" / "benchmark" / "L1W_voxelized01_for_eval.laz"
        base_dir = REPO_ROOT / "data" / "pipeline" / "L1W"
        pred_forest = base_dir / results_dir_name / "full_forest" / "L1W.laz"
        pointwise_npz = base_dir / results_dir_name / "pointwise_results" / "pointwise_results.npz"
    elif mode == "eval_subset":
        pipeline_input = REPO_ROOT / "data" / "benchmark" / "L1W_voxelized01_for_eval.laz"
        gt_forest = REPO_ROOT / "data" / "benchmark" / "L1W_voxelized01_for_eval.laz"
        base_dir = REPO_ROOT / "data"
        pred_forest = base_dir / results_dir_name / "full_forest" / "L1W_voxelized01_for_eval.laz"
        pointwise_npz = base_dir / results_dir_name / "pointwise_results" / "pointwise_results.npz"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    evaluation_dir = pred_forest.parent / "evaluation"
    return {
        "pipeline_input": pipeline_input,
        "gt_forest": gt_forest,
        "pred_forest": pred_forest,
        "pointwise_npz": pointwise_npz,
        "evaluation_dir": evaluation_dir,
    }


def validate_required_inputs(mode: str, checkpoint: Path, paths: dict[str, Path]) -> list[tuple[str, Path]]:
    missing: list[tuple[str, Path]] = []
    if not checkpoint.exists():
        missing.append(("checkpoint", checkpoint))
    if not paths["pipeline_input"].exists():
        missing.append(("pipeline_input", paths["pipeline_input"]))
    if not paths["gt_forest"].exists():
        missing.append(("gt_forest", paths["gt_forest"]))
    return missing


def download_hints(mode: str) -> list[str]:
    hints = []
    if mode == "raw_forest":
        hints.append(
            "python tree_learn/util/download.py --dataset_name benchmark_dataset "
            "--root_folder data/pipeline/L1W/forest"
        )
    hints.append(
        "python tree_learn/util/download.py --dataset_name benchmark_dataset_evaluation "
        "--root_folder data/benchmark"
    )
    return hints


def write_pipeline_config(
    mode: str,
    pipeline_input: Path,
    checkpoint: Path,
    results_dir_name: str,
    reuse_tiles: bool,
) -> Path:
    cfg = load_yaml(PIPELINE_BASE_CFG)
    cfg["forest_path"] = repo_rel(pipeline_input)
    cfg["pretrain"] = repo_rel(checkpoint)
    cfg["tile_generation"] = not reuse_tiles
    cfg.setdefault("save_cfg", {})
    cfg["save_cfg"]["results_dir"] = results_dir_name
    cfg["save_cfg"]["return_type"] = "original"
    cfg["save_cfg"]["save_formats"] = ["laz"]

    config_path = GENERATED_CFG_DIR / f"pipeline_l1w_{mode}.yaml"
    dump_yaml(cfg, config_path)
    return config_path


def write_eval_config(mode: str, pred_forest: Path, gt_forest: Path) -> Path:
    cfg = load_yaml(EVAL_BASE_CFG)
    cfg.setdefault("paths", {})
    cfg["paths"]["pred_forest_path"] = repo_rel(pred_forest)
    cfg["paths"]["gt_forest_path"] = repo_rel(gt_forest)

    config_path = GENERATED_CFG_DIR / f"evaluate_l1w_{mode}.yaml"
    dump_yaml(cfg, config_path)
    return config_path


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verified L1W inference + evaluation workflow runner."
    )
    parser.add_argument(
        "--mode",
        choices=["raw_forest", "eval_subset"],
        default="raw_forest",
        help=(
            "raw_forest follows docs/segmentation_pipeline.md; "
            "eval_subset is a quick pipeline+eval sanity mode on the evaluation subset itself."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path. Defaults to the pretrain path currently configured in configs/pipeline/pipeline.yaml.",
    )
    parser.add_argument(
        "--results-dir-name",
        type=str,
        default=None,
        help="Relative results directory name under the pipeline base dir. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--reuse-tiles",
        action="store_true",
        help="Reuse existing tiles/features/voxelized artifacts instead of regenerating them.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip pipeline inference and only run evaluation.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation and only run pipeline inference.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved files, generated configs, and commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir_name = args.results_dir_name or default_results_dir(args.mode)
    checkpoint = resolve_checkpoint(args.checkpoint)
    paths = build_mode_paths(args.mode, results_dir_name)

    print("=== Verified L1W workflow ===")
    print(f"mode              : {args.mode}")
    print(f"checkpoint        : {checkpoint}")
    print(f"pipeline_input    : {paths['pipeline_input']}")
    print(f"gt_forest         : {paths['gt_forest']}")
    print(f"pred_forest       : {paths['pred_forest']}")
    print(f"pointwise_npz     : {paths['pointwise_npz']}")
    print(f"evaluation_dir    : {paths['evaluation_dir']}")

    missing = validate_required_inputs(args.mode, checkpoint, paths)
    if missing:
        print("\n[ERROR] Missing required files:")
        for kind, path in missing:
            print(f"  - {kind}: {path}")
        print("\n[HINT] Download / prepare the required L1W files first:")
        for hint in download_hints(args.mode):
            print(f"  {hint}")
        print("  # Also make sure your checkpoint exists, e.g. data/model_weights/epoch_781.pth")
        return 2

    pipeline_cfg_path = write_pipeline_config(
        args.mode,
        paths["pipeline_input"],
        checkpoint,
        results_dir_name,
        args.reuse_tiles,
    )
    eval_cfg_path = write_eval_config(args.mode, paths["pred_forest"], paths["gt_forest"])

    print("\nGenerated configs:")
    print(f"  - {pipeline_cfg_path}")
    print(f"  - {eval_cfg_path}")

    if not args.skip_pipeline:
        pipeline_cmd = [
            sys.executable,
            "tools/pipeline/pipeline.py",
            "--config",
            repo_rel(pipeline_cfg_path),
        ]
        run_command(pipeline_cmd, args.dry_run)

    if not args.skip_eval:
        if not args.dry_run and not paths["pred_forest"].exists():
            print(f"[ERROR] Expected prediction file not found after pipeline: {paths['pred_forest']}")
            return 3
        eval_cmd = [
            sys.executable,
            "tools/evaluation/evaluate.py",
            "--config",
            repo_rel(eval_cfg_path),
        ]
        run_command(eval_cmd, args.dry_run)

    print("\nExpected outputs:")
    print(f"  - pointwise results : {paths['pointwise_npz']}")
    print(f"  - full forest pred  : {paths['pred_forest']}")
    print(f"  - evaluation dir    : {paths['evaluation_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
