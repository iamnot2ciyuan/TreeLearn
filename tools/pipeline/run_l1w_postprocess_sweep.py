#!/usr/bin/env python3
"""Sweep L1W post-processing parameters from saved pointwise predictions.

This script is intended to diagnose whether high commission error comes from
inference-time grouping thresholds rather than the trained checkpoint itself.
Run a normal pipeline once to create pointwise_results.npz, then this script
reuses those pointwise logits/offsets for many grouping configurations.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit(
        "PyTorch is required for tools/pipeline/run_l1w_postprocess_sweep.py. "
        "Please activate the TreeLearn conda environment before running it."
    ) from exc

from run_l1w_workflow import (  # noqa: E402
    EVAL_BASE_CFG,
    GENERATED_CFG_DIR,
    PIPELINE_BASE_CFG,
    REPO_ROOT,
    build_mode_paths,
    dump_yaml,
    load_yaml,
    repo_rel,
)
from tree_learn.util import (  # noqa: E402
    assign_remaining_points_nearest_neighbor,
    get_config,
    get_coords_within_shape,
    get_hull_buffer,
    load_data,
    make_labels_consecutive,
    propagate_preds,
    propagate_preds_hash_full,
    propagate_preds_hash_vox,
    save_data,
)
from tree_learn.util.pipeline import get_instances  # noqa: E402


NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "nil", "~", "na"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse saved L1W pointwise predictions and sweep grouping/post-processing "
            "parameters without rerunning the neural network."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["raw_forest", "eval_subset"],
        default="eval_subset",
        help="Must match the mode used to create the source pointwise predictions.",
    )
    parser.add_argument(
        "--source-results-dir-name",
        required=True,
        help=(
            "Existing results directory containing pointwise_results/pointwise_results.npz, "
            "for example results_l1w_eval_subset_epoch720_base."
        ),
    )
    parser.add_argument(
        "--pointwise-npz",
        default=None,
        help="Optional explicit path to pointwise_results.npz. Overrides --source-results-dir-name.",
    )
    parser.add_argument(
        "--results-prefix",
        default="postprocess_sweep",
        help="Prefix for per-run result directories and summary files.",
    )
    parser.add_argument("--tree-conf-thresh", type=float, nargs="+", default=[0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--tau-min", type=int, nargs="+", default=[35, 45, 55, 70])
    parser.add_argument("--tau-vert", type=parse_optional_float, nargs="+", default=[0.35])
    parser.add_argument("--tau-off", type=parse_optional_float, nargs="+", default=[6.0])
    parser.add_argument(
        "--outer-remove",
        type=parse_optional_float,
        nargs="+",
        default=[None],
        help="Optional edge buffer values in meters; use none or ~ for no outer removal.",
    )
    parser.add_argument(
        "--checkpoint-label",
        default="",
        help="Free-form label written to the summary, e.g. epoch720.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run if its evaluation_results.pt already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned runs and generated paths; do not write predictions or evaluate.",
    )
    return parser.parse_args()


def source_pointwise_path(mode: str, source_results_dir_name: str) -> Path:
    paths = build_mode_paths(mode, source_results_dir_name)
    return paths["pointwise_npz"]


def format_num(value: float | int | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, int):
        return str(value)
    return str(value).replace(".", "p")


def run_id_for(params: dict[str, Any]) -> str:
    return (
        f"tc{format_num(params['tree_conf_thresh'])}_"
        f"tm{format_num(params['tau_min'])}_"
        f"tv{format_num(params['tau_vert'])}_"
        f"to{format_num(params['tau_off'])}_"
        f"or{format_num(params['outer_remove'])}"
    )


def write_pipeline_config(mode: str, results_dir_name: str, params: dict[str, Any]) -> Path:
    paths = build_mode_paths(mode, results_dir_name)
    cfg = load_yaml(PIPELINE_BASE_CFG)
    cfg["forest_path"] = repo_rel(paths["pipeline_input"])
    cfg["tile_generation"] = False
    cfg.setdefault("save_cfg", {})
    cfg["save_cfg"]["results_dir"] = results_dir_name
    cfg["save_cfg"]["return_type"] = "original"
    cfg["save_cfg"]["save_formats"] = ["laz"]
    cfg["save_cfg"]["save_treewise"] = False
    cfg["save_cfg"]["save_pointwise"] = False

    cfg.setdefault("grouping", {})
    cfg["grouping"]["tree_conf_thresh"] = params["tree_conf_thresh"]
    cfg["grouping"]["tau_min"] = params["tau_min"]
    cfg["grouping"]["tau_vert"] = params["tau_vert"]
    cfg["grouping"]["tau_off"] = params["tau_off"]

    cfg.setdefault("shape_cfg", {})
    cfg["shape_cfg"]["outer_remove"] = params["outer_remove"]

    config_path = GENERATED_CFG_DIR / f"pipeline_l1w_postprocess_{mode}_{results_dir_name}.yaml"
    dump_yaml(cfg, config_path)
    return config_path


def write_eval_config(mode: str, results_dir_name: str) -> Path:
    paths = build_mode_paths(mode, results_dir_name)
    cfg = load_yaml(EVAL_BASE_CFG)
    cfg.setdefault("paths", {})
    cfg["paths"]["pred_forest_path"] = repo_rel(paths["pred_forest"])
    cfg["paths"]["gt_forest_path"] = repo_rel(paths["gt_forest"])

    config_path = GENERATED_CFG_DIR / f"evaluate_l1w_postprocess_{mode}_{results_dir_name}.yaml"
    dump_yaml(cfg, config_path)
    return config_path


def load_pointwise(pointwise_npz: Path) -> dict[str, np.ndarray]:
    if not pointwise_npz.exists():
        raise FileNotFoundError(
            f"Missing pointwise predictions: {pointwise_npz}\n"
            "Run tools/pipeline/run_l1w_workflow.py once with --skip-eval first."
        )
    with np.load(pointwise_npz, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def apply_outer_remove(config: Any, arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], Any | None]:
    outer_remove = getattr(config.shape_cfg, "outer_remove", None)
    if not outer_remove:
        return arrays, None

    hull_buffer_large = get_hull_buffer(
        arrays["coords"][:, :2],
        config.shape_cfg.alpha,
        buffersize=outer_remove,
    )
    mask_coords_within_hull_buffer_large = get_coords_within_shape(arrays["coords"], hull_buffer_large)
    masks_inner_coords = np.logical_not(mask_coords_within_hull_buffer_large)
    filtered = {
        key: value[masks_inner_coords] if len(value) == len(arrays["coords"]) else value
        for key, value in arrays.items()
    }
    return filtered, hull_buffer_large


def save_full_forest_prediction(
    mode: str,
    config: Any,
    results_dir_name: str,
    instance_preds: np.ndarray,
    pointwise: dict[str, np.ndarray],
    hull_buffer_large: Any | None,
) -> Path:
    paths = build_mode_paths(mode, results_dir_name)
    pipeline_input = paths["pipeline_input"]
    base_dir = pipeline_input.parent.parent
    plot_name = pipeline_input.stem
    voxelized_data_dir = base_dir / f"forest_voxelized{config.sample_generation.voxel_size}"

    original_data = load_data(str(pipeline_input))
    xyz = original_data[:, :3].astype(np.float64)
    xyz_mean = np.mean(xyz, 0).astype(np.float64)
    coords_to_return = xyz - xyz_mean

    return_type = config.save_cfg.return_type
    not_yet_propagated = np.zeros(len(coords_to_return), dtype=bool)
    if return_type == "original":
        hash_mapping_path = voxelized_data_dir / f"{plot_name}_hash_mapping.pkl"
        with hash_mapping_path.open("rb") as f:
            hash_mapping = pickle.load(f)
        preds_to_return, not_yet_propagated = propagate_preds_hash_full(
            pointwise["coords"],
            instance_preds,
            coords_to_return,
            hash_mapping,
        )
    elif return_type == "voxelized":
        voxelized_forest_path = voxelized_data_dir / f"{plot_name}.npz"
        coords_to_return = load_data(str(voxelized_forest_path))[:, :3]
        preds_to_return, not_yet_propagated = propagate_preds_hash_vox(
            pointwise["coords"],
            instance_preds,
            coords_to_return,
        )
    elif return_type == "voxelized_and_filtered":
        coords_to_return = pointwise["coords"]
        preds_to_return = instance_preds
    else:
        raise ValueError(f"Unsupported return_type: {return_type}")

    if hull_buffer_large is not None:
        mask_coords_to_return_within_hull_buffer_large = get_coords_within_shape(
            coords_to_return,
            hull_buffer_large,
        )
        masks_inner_coords_to_return = np.logical_not(mask_coords_to_return_within_hull_buffer_large)
        coords_to_return = coords_to_return[masks_inner_coords_to_return]
        preds_to_return = preds_to_return[masks_inner_coords_to_return]
        not_yet_propagated = not_yet_propagated[masks_inner_coords_to_return]

    if not_yet_propagated.any():
        preds_to_return[not_yet_propagated] = propagate_preds(
            pointwise["coords"],
            instance_preds,
            coords_to_return[not_yet_propagated],
            n_neighbors=5,
        )

    if return_type in {"original", "voxelized"}:
        coords_to_return = coords_to_return.astype(np.float64) + xyz_mean

    full_dir = paths["pred_forest"].parent
    full_dir.mkdir(parents=True, exist_ok=True)
    save_data(
        np.hstack([coords_to_return, preds_to_return.reshape(-1, 1)]),
        "laz",
        plot_name,
        str(full_dir),
    )
    return paths["pred_forest"]


def postprocess_once(mode: str, config_path: Path, pointwise_npz: Path, results_dir_name: str) -> Path:
    config = get_config(str(config_path))
    pointwise = load_pointwise(pointwise_npz)
    pointwise, hull_buffer_large = apply_outer_remove(config, pointwise)

    tree_class_in_dataset = getattr(config.grouping, "tree_class_in_dataset", 0)
    instance_preds = get_instances(
        pointwise["coords"],
        pointwise["offset_predictions"],
        pointwise["semantic_prediction_logits"],
        config.grouping,
        pointwise["input_feats"][:, -1],
        tree_class_in_dataset,
        NON_TREES_LABEL_IN_GROUPING,
        NOT_ASSIGNED_LABEL_IN_GROUPING,
        START_NUM_PREDS,
    )
    instance_preds_after_initial_clustering = np.copy(instance_preds)

    tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
    instance_preds[tree_mask] = assign_remaining_points_nearest_neighbor(
        pointwise["coords"][tree_mask] + pointwise["offset_predictions"][tree_mask],
        instance_preds[tree_mask],
        NOT_ASSIGNED_LABEL_IN_GROUPING,
    )

    if hull_buffer_large is not None:
        tree_mask = instance_preds != NON_TREES_LABEL_IN_GROUPING
        instance_preds[tree_mask], _ = make_labels_consecutive(instance_preds[tree_mask], start_num=1)

    paths = build_mode_paths(mode, results_dir_name)
    post_dir = paths["pred_forest"].parent.parent / "postprocess_results"
    post_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        post_dir / "postprocess_results.npz",
        coords=pointwise["coords"],
        offset_predictions=pointwise["offset_predictions"],
        semantic_prediction_logits=pointwise["semantic_prediction_logits"],
        input_feats=pointwise["input_feats"],
        instance_preds=instance_preds,
        instance_preds_after_initial_clustering=instance_preds_after_initial_clustering,
    )
    return save_full_forest_prediction(mode, config, results_dir_name, instance_preds, pointwise, hull_buffer_large)


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def summarize_one(
    checkpoint_label: str,
    run_name: str,
    params: dict[str, Any],
    evaluation_results_path: Path,
    pred_forest_path: Path,
) -> dict[str, Any]:
    results = torch.load(evaluation_results_path, map_location="cpu")
    detection = results["detection_results"]
    segmentation = results["segmentation_results"]
    return {
        "checkpoint_label": checkpoint_label,
        "run_name": run_name,
        "pred_forest_path": str(pred_forest_path),
        "evaluation_results_path": str(evaluation_results_path),
        "tree_conf_thresh": params["tree_conf_thresh"],
        "tau_min": params["tau_min"],
        "tau_vert": params["tau_vert"],
        "tau_off": params["tau_off"],
        "outer_remove": params["outer_remove"],
        "completeness": float(detection["completeness"]),
        "omission_error_rate": float(detection["omission_error_rate"]),
        "commission_error_rate": float(detection["commission_error_rate"]),
        "f1_score": float(detection["f1_score"]),
        "precision": float(segmentation["precision"]),
        "recall": float(segmentation["recall"]),
        "iou": float(segmentation["iou"]),
    }


def write_summary(rows: list[dict[str, Any]], output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    md_path = output_stem.with_suffix(".md")

    fieldnames = [
        "checkpoint_label",
        "run_name",
        "tree_conf_thresh",
        "tau_min",
        "tau_vert",
        "tau_off",
        "outer_remove",
        "completeness",
        "omission_error_rate",
        "commission_error_rate",
        "f1_score",
        "precision",
        "recall",
        "iou",
        "pred_forest_path",
        "evaluation_results_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_by_f1 = max(rows, key=lambda row: row["f1_score"])
    best_low_commission = min(rows, key=lambda row: (row["commission_error_rate"], -row["iou"]))
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# L1W post-processing sweep summary\n\n")
        if rows[0].get("checkpoint_label"):
            f.write(f"- Checkpoint: `{rows[0]['checkpoint_label']}`\n")
        f.write(
            "- Best by detection F1: "
            f"`{best_by_f1['run_name']}` "
            f"(F1={best_by_f1['f1_score']:.1f}, Ecom={best_by_f1['commission_error_rate']:.1f}, "
            f"Cov={best_by_f1['iou']:.1f})\n"
        )
        f.write(
            "- Lowest commission: "
            f"`{best_low_commission['run_name']}` "
            f"(Ecom={best_low_commission['commission_error_rate']:.1f}, F1={best_low_commission['f1_score']:.1f}, "
            f"Cov={best_low_commission['iou']:.1f})\n\n"
        )
        f.write("| run | tree_conf | tau_min | tau_vert | tau_off | outer_remove | C | Eom | Ecom | F1 | Prec. | Rec. | Cov. |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                "| {run_name} | {tree_conf_thresh} | {tau_min} | {tau_vert} | {tau_off} | {outer_remove} | "
                "{completeness:.1f} | {omission_error_rate:.1f} | {commission_error_rate:.1f} | "
                "{f1_score:.1f} | {precision:.1f} | {recall:.1f} | {iou:.1f} |\n".format(**row)
            )
        f.write("\n## Paths\n\n")
        for row in rows:
            f.write(f"- `{row['run_name']}` pred: `{row['pred_forest_path']}`\n")
            f.write(f"  eval: `{row['evaluation_results_path']}`\n")

    return csv_path, md_path


def main() -> int:
    args = parse_args()
    pointwise_npz = Path(args.pointwise_npz) if args.pointwise_npz else source_pointwise_path(args.mode, args.source_results_dir_name)
    pointwise_npz = pointwise_npz if pointwise_npz.is_absolute() else (REPO_ROOT / pointwise_npz)

    combos = list(
        itertools.product(
            args.tree_conf_thresh,
            args.tau_min,
            args.tau_vert,
            args.tau_off,
            args.outer_remove,
        )
    )

    print("=== L1W post-processing parameter sweep ===")
    print(f"mode                 : {args.mode}")
    print(f"source_results_dir   : {args.source_results_dir_name}")
    print(f"pointwise_npz        : {pointwise_npz}")
    print(f"results_prefix       : {args.results_prefix}")
    print(f"num_runs             : {len(combos)}")

    if not args.dry_run and not pointwise_npz.exists():
        raise FileNotFoundError(
            f"Pointwise file not found: {pointwise_npz}\n"
            "Create it first with tools/pipeline/run_l1w_workflow.py --skip-eval."
        )

    rows: list[dict[str, Any]] = []
    for combo in combos:
        params = {
            "tree_conf_thresh": combo[0],
            "tau_min": combo[1],
            "tau_vert": combo[2],
            "tau_off": combo[3],
            "outer_remove": combo[4],
        }
        rid = run_id_for(params)
        results_dir_name = f"{args.results_prefix}_{rid}"
        paths = build_mode_paths(args.mode, results_dir_name)
        evaluation_results_path = paths["evaluation_dir"] / "evaluation_results.pt"

        print(f"\n--- {results_dir_name} ---")
        print(f"params      : {params}")
        print(f"pred_forest : {paths['pred_forest']}")
        print(f"eval_result : {evaluation_results_path}")

        pipeline_cfg_path = write_pipeline_config(args.mode, results_dir_name, params)
        eval_cfg_path = write_eval_config(args.mode, results_dir_name)
        print(f"pipeline_cfg: {pipeline_cfg_path}")
        print(f"eval_cfg    : {eval_cfg_path}")

        if args.skip_existing and evaluation_results_path.exists():
            print("[SKIP] Existing evaluation result found.")
        elif args.dry_run:
            continue
        else:
            postprocess_once(args.mode, pipeline_cfg_path, pointwise_npz, results_dir_name)
            eval_cmd = [
                sys.executable,
                "tools/evaluation/evaluate.py",
                "--config",
                repo_rel(eval_cfg_path),
            ]
            run_command(eval_cmd, args.dry_run)

        if args.dry_run:
            continue
        if not evaluation_results_path.exists():
            raise FileNotFoundError(f"Expected evaluation result missing: {evaluation_results_path}")
        rows.append(
            summarize_one(
                args.checkpoint_label,
                results_dir_name,
                params,
                evaluation_results_path,
                paths["pred_forest"],
            )
        )

    if args.dry_run:
        print("\n[DRY-RUN] No predictions, evaluations, or summaries were written.")
        return 0
    if not rows:
        print("[WARN] No completed rows to summarize.")
        return 1

    output_stem = GENERATED_CFG_DIR / f"postprocess_sweep_{args.mode}_{args.results_prefix}"
    csv_path, md_path = write_summary(rows, output_stem)
    print("\nSummary files:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
