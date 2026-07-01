#!/usr/bin/env python3
"""Run L1W pipeline + evaluation for multiple checkpoints and summarize results."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit(
        "PyTorch is required for tools/pipeline/run_l1w_checkpoint_sweep.py. "
        "Please activate the TreeLearn conda environment before running it."
    ) from exc

from run_l1w_workflow import (  # noqa: E402
    GENERATED_CFG_DIR,
    PIPELINE_BASE_CFG,
    REPO_ROOT,
    build_mode_paths,
    default_results_dir,
    load_yaml,
)


DEFAULT_EPOCHS = [680, 720, 781]
DEFAULT_TRAIN_WORK_DIR = REPO_ROOT / "work_dirs" / "train_20260627"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the verified L1W workflow for multiple checkpoints and summarize evaluation results."
    )
    parser.add_argument(
        "--mode",
        choices=["raw_forest", "eval_subset"],
        default="raw_forest",
        help="Use raw_forest for the doc-faithful L1W workflow, or eval_subset for quick sanity checks.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=DEFAULT_EPOCHS,
        help=f"Checkpoint epochs to evaluate (default: {' '.join(map(str, DEFAULT_EPOCHS))}).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory containing epoch_XXX.pth files from training. "
            "If omitted, the script first checks the current pipeline pretrain path, "
            f"then {DEFAULT_TRAIN_WORK_DIR}, then data/model_weights."
        ),
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default=None,
        help="Prefix for per-checkpoint results directories. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a checkpoint if its evaluation_results.pt already exists.",
    )
    parser.add_argument(
        "--reuse-tiles",
        action="store_true",
        help="Reuse existing tiles even for the first checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved checkpoints and commands without running them.",
    )
    return parser.parse_args()


def unique_existing_dirs(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            deduped.append(path)
    return deduped


def resolve_search_dirs(user_checkpoint_dir: str | None) -> list[Path]:
    dirs: list[Path] = []
    if user_checkpoint_dir:
        dirs.append(Path(user_checkpoint_dir) if Path(user_checkpoint_dir).is_absolute() else REPO_ROOT / user_checkpoint_dir)

    pipeline_cfg = load_yaml(PIPELINE_BASE_CFG)
    pretrain = Path(pipeline_cfg["pretrain"])
    pretrain = pretrain if pretrain.is_absolute() else (REPO_ROOT / pretrain)
    dirs.append(pretrain.parent)
    dirs.append(DEFAULT_TRAIN_WORK_DIR)
    dirs.append(REPO_ROOT / "data" / "model_weights")
    dirs.append(REPO_ROOT / "work_dirs")
    return unique_existing_dirs(dirs)


def resolve_checkpoint(epoch: int, search_dirs: list[Path], preferred_dir: Path | None = None) -> Path:
    expected_name = f"epoch_{epoch}.pth"

    if preferred_dir is not None:
        preferred_candidate = preferred_dir / expected_name
        if preferred_candidate.is_file():
            return preferred_candidate

    pipeline_cfg = load_yaml(PIPELINE_BASE_CFG)
    configured_pretrain = Path(pipeline_cfg["pretrain"])
    configured_pretrain = configured_pretrain if configured_pretrain.is_absolute() else (REPO_ROOT / configured_pretrain)
    if configured_pretrain.name == expected_name and configured_pretrain.is_file():
        return configured_pretrain

    direct_matches: list[Path] = []
    recursive_matches: list[Path] = []
    for directory in search_dirs:
        direct = directory / expected_name
        if direct.is_file():
            direct_matches.append(direct)
            continue
        recursive_matches.extend(sorted(directory.rglob(expected_name)))

    candidates: list[Path] = []
    seen = set()
    for path in direct_matches + recursive_matches:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        search_text = "\n".join(f"  - {directory}" for directory in search_dirs)
        raise FileNotFoundError(
            f"Could not find {expected_name}. Searched:\n{search_text}"
        )

    candidate_text = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(
        f"Found multiple candidates for {expected_name}. "
        f"Please pass --checkpoint-dir to disambiguate:\n{candidate_text}"
    )


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def results_prefix_for_mode(mode: str) -> str:
    if mode == "raw_forest":
        return "results_l1w_raw"
    if mode == "eval_subset":
        return "results_l1w_eval_subset"
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_one(epoch: int, checkpoint_path: Path, evaluation_results_path: Path, pred_forest_path: Path) -> dict[str, object]:
    results = torch.load(evaluation_results_path, map_location="cpu")
    detection = results["detection_results"]
    segmentation = results["segmentation_results"]
    return {
        "epoch": epoch,
        "checkpoint_path": str(checkpoint_path),
        "pred_forest_path": str(pred_forest_path),
        "evaluation_results_path": str(evaluation_results_path),
        "completeness": float(detection["completeness"]),
        "omission_error_rate": float(detection["omission_error_rate"]),
        "commission_error_rate": float(detection["commission_error_rate"]),
        "f1_score": float(detection["f1_score"]),
        "precision": float(segmentation["precision"]),
        "recall": float(segmentation["recall"]),
        "iou": float(segmentation["iou"]),
    }


def write_summary(rows: list[dict[str, object]], output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    md_path = output_stem.with_suffix(".md")

    fieldnames = [
        "epoch",
        "checkpoint_path",
        "pred_forest_path",
        "evaluation_results_path",
        "completeness",
        "omission_error_rate",
        "commission_error_rate",
        "f1_score",
        "precision",
        "recall",
        "iou",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_by_iou = max(rows, key=lambda row: row["iou"])
    best_by_f1 = max(rows, key=lambda row: row["f1_score"])
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# L1W checkpoint sweep summary\n\n")
        f.write(f"- Best by segmentation IoU: epoch {best_by_iou['epoch']} ({best_by_iou['iou']:.1f})\n")
        f.write(f"- Best by detection F1: epoch {best_by_f1['epoch']} ({best_by_f1['f1_score']:.1f})\n\n")
        f.write("| epoch | completeness | omission | commission | f1 | precision | recall | iou |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                "| {epoch} | {completeness:.1f} | {omission_error_rate:.1f} | {commission_error_rate:.1f} | "
                "{f1_score:.1f} | {precision:.1f} | {recall:.1f} | {iou:.1f} |\n".format(**row)
            )
        f.write("\n## Paths\n\n")
        for row in rows:
            f.write(f"- epoch {row['epoch']}: `{row['checkpoint_path']}`\n")
            f.write(f"  pred: `{row['pred_forest_path']}`\n")
            f.write(f"  eval: `{row['evaluation_results_path']}`\n")

    return csv_path, md_path


def main() -> int:
    args = parse_args()
    prefix = args.results_prefix or results_prefix_for_mode(args.mode)
    search_dirs = resolve_search_dirs(args.checkpoint_dir)
    preferred_dir = None
    if args.checkpoint_dir:
        preferred_dir = Path(args.checkpoint_dir)
        preferred_dir = preferred_dir if preferred_dir.is_absolute() else (REPO_ROOT / preferred_dir)

    print("=== L1W checkpoint sweep ===")
    print(f"mode        : {args.mode}")
    print(f"epochs      : {args.epochs}")
    print(f"search_dirs :")
    for directory in search_dirs:
        print(f"  - {directory}")

    rows: list[dict[str, object]] = []
    for run_idx, epoch in enumerate(args.epochs):
        checkpoint_path = resolve_checkpoint(epoch, search_dirs, preferred_dir=preferred_dir)
        results_dir_name = f"{prefix}_epoch{epoch}"
        paths = build_mode_paths(args.mode, results_dir_name)
        evaluation_results_path = paths["evaluation_dir"] / "evaluation_results.pt"

        print(f"\n--- epoch {epoch} ---")
        print(f"checkpoint_path      : {checkpoint_path}")
        print(f"results_dir_name     : {results_dir_name}")
        print(f"pred_forest_path     : {paths['pred_forest']}")
        print(f"evaluation_results   : {evaluation_results_path}")

        if args.skip_existing and evaluation_results_path.exists():
            print("[SKIP] Existing evaluation results found.")
        else:
            workflow_cmd = [
                sys.executable,
                "tools/pipeline/run_l1w_workflow.py",
                "--mode",
                args.mode,
                "--checkpoint",
                str(checkpoint_path),
                "--results-dir-name",
                results_dir_name,
            ]
            if args.dry_run:
                workflow_cmd.append("--dry-run")
            if args.reuse_tiles or run_idx > 0:
                workflow_cmd.append("--reuse-tiles")
            run_command(workflow_cmd, args.dry_run)

        if args.dry_run:
            continue
        if not evaluation_results_path.exists():
            raise FileNotFoundError(
                f"Expected evaluation results were not created: {evaluation_results_path}"
            )

        rows.append(summarize_one(epoch, checkpoint_path, evaluation_results_path, paths["pred_forest"]))

    if args.dry_run:
        print("\n[DRY-RUN] No summary files were written.")
        return 0

    if not rows:
        print("[WARN] No completed runs to summarize.")
        return 1

    output_stem = GENERATED_CFG_DIR / f"checkpoint_sweep_{args.mode}"
    csv_path, md_path = write_summary(rows, output_stem)
    print("\nSummary files:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
