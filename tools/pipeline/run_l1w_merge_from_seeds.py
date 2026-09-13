#!/usr/bin/env python3
"""Fast duplicate-seed merge sweep: reuse saved HDBSCAN seeds, only merge + NN + eval.

Avoids re-running HDBSCAN (the expensive step). Source must contain
postprocess_results/postprocess_results.npz with instance_preds_after_initial_clustering.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

from run_l1w_duplicate_seed_sweep import run_id_for, write_pipeline_config, write_summary  # noqa: E402
from run_l1w_postprocess_sweep import (  # noqa: E402
    GENERATED_CFG_DIR,
    NON_TREES_LABEL_IN_GROUPING,
    NOT_ASSIGNED_LABEL_IN_GROUPING,
    REPO_ROOT,
    START_NUM_PREDS,
    apply_outer_remove,
    build_mode_paths,
    format_num,
    get_config,
    load_pointwise,
    make_labels_consecutive,
    parse_optional_float,
    repo_rel,
    run_command,
    save_full_forest_prediction,
    summarize_one,
    write_eval_config,
)
from tree_learn.util import assign_remaining_points_nearest_neighbor  # noqa: E402
from tree_learn.util.pipeline import suppress_duplicate_seed_clusters  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["raw_forest", "eval_subset"], default="eval_subset")
    p.add_argument(
        "--source-postprocess-dir",
        required=True,
        help="Existing run dir with postprocess_results/postprocess_results.npz (e.g. pp_epoch720_tmin_...).",
    )
    p.add_argument("--results-prefix", default="dupmerge_ep720")
    p.add_argument("--checkpoint-label", default="epoch720")
    p.add_argument("--tree-conf-thresh", type=float, default=0.50)
    p.add_argument("--tau-min", type=int, default=100)
    p.add_argument("--tau-vert", type=parse_optional_float, default=0.35)
    p.add_argument("--tau-off", type=parse_optional_float, default=6.0)
    p.add_argument("--outer-remove", type=parse_optional_float, default=None)
    p.add_argument("--focused", action="store_true")
    p.add_argument("--merge-seed-dist", type=parse_optional_float, nargs="+", default=[None])
    p.add_argument("--merge-height-overlap", type=parse_optional_float, nargs="+", default=[None])
    p.add_argument("--merge-weak-seed-frac", type=parse_optional_float, nargs="+", default=[None])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def focused_merge_combos() -> list[tuple[float | None, float | None, float | None]]:
    return [
        # baseline already evaluated as mdnone; keep for skip-existing reuse
        (None, None, None),
        (1.00, 0.60, 0.25),  # safe: ~6 merges, targets 298-like
        (1.00, 0.75, 0.25),  # slightly stricter height
        (3.00, 0.75, 0.55),  # aggressive: also targets 252-like (~92 merges)
    ]


def postprocess_from_saved_seeds(
    mode: str,
    config_path: Path,
    source_npz: Path,
    results_dir_name: str,
    merge_seed_dist: float | None,
    merge_height_overlap: float | None,
    merge_weak_seed_frac: float | None,
) -> tuple[Path, int]:
    config = get_config(str(config_path))
    arrays = load_pointwise(source_npz)
    if "instance_preds_after_initial_clustering" not in arrays:
        raise KeyError(f"Missing instance_preds_after_initial_clustering in {source_npz}")

    pointwise = {
        "coords": arrays["coords"],
        "offset_predictions": arrays["offset_predictions"],
        "semantic_prediction_logits": arrays["semantic_prediction_logits"],
        "input_feats": arrays["input_feats"],
    }
    pointwise, hull_buffer_large = apply_outer_remove(config, pointwise)

    # Outer-remove may have filtered coords; re-align seed labels if needed.
    # Source postprocess was typically run with outer_remove=None, so lengths match.
    init = arrays["instance_preds_after_initial_clustering"].astype(np.int64)
    if len(init) != len(pointwise["coords"]):
        raise ValueError(
            f"Length mismatch after outer_remove: seeds={len(init)} coords={len(pointwise['coords'])}. "
            "Use a source run with the same outer_remove setting."
        )

    seeds = init.copy()
    seeds, n_merges = suppress_duplicate_seed_clusters(
        pointwise["coords"],
        pointwise["offset_predictions"],
        seeds,
        NOT_ASSIGNED_LABEL_IN_GROUPING,
        NON_TREES_LABEL_IN_GROUPING,
        merge_seed_dist,
        merge_height_overlap,
        merge_weak_seed_frac,
        start_num_preds=START_NUM_PREDS,
    )
    instance_preds_after_initial_clustering = seeds.copy()

    tree_mask = seeds != NON_TREES_LABEL_IN_GROUPING
    instance_preds = seeds.copy()
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
        n_seed_merges=np.asarray([n_merges], dtype=np.int64),
    )
    pred_path = save_full_forest_prediction(
        mode, config, results_dir_name, instance_preds, pointwise, hull_buffer_large
    )
    return pred_path, n_merges


def main() -> int:
    args = parse_args()
    src = Path(args.source_postprocess_dir)
    src = src if src.is_absolute() else REPO_ROOT / src
    source_npz = src / "postprocess_results" / "postprocess_results.npz"
    if not source_npz.exists():
        raise FileNotFoundError(source_npz)

    if args.focused:
        merge_combos = focused_merge_combos()
    else:
        import itertools

        merge_combos = []
        for md, mh, mw in itertools.product(
            args.merge_seed_dist, args.merge_height_overlap, args.merge_weak_seed_frac
        ):
            n_none = sum(v is None for v in (md, mh, mw))
            if n_none in (0, 3):
                merge_combos.append((md, mh, mw))

    print("=== Fast merge-from-seeds sweep ===", flush=True)
    print(f"source_npz: {source_npz}", flush=True)
    print(f"num_runs   : {len(merge_combos)}", flush=True)

    rows: list[dict[str, Any]] = []
    for md, mh, mw in merge_combos:
        params = {
            "tree_conf_thresh": args.tree_conf_thresh,
            "tau_min": args.tau_min,
            "tau_vert": args.tau_vert,
            "tau_off": args.tau_off,
            "outer_remove": args.outer_remove,
            "cluster_selection_epsilon": 0.0,
            "merge_seed_dist": md,
            "merge_height_overlap": mh,
            "merge_weak_seed_frac": mw,
        }
        rid = run_id_for(params)
        results_dir_name = f"{args.results_prefix}_{rid}"
        paths = build_mode_paths(args.mode, results_dir_name)
        evaluation_results_path = paths["evaluation_dir"] / "evaluation_results.pt"

        print(f"\n--- {results_dir_name} ---", flush=True)
        print(f"params: md={md} mh={mh} mw={mw}", flush=True)

        pipeline_cfg_path = write_pipeline_config(args.mode, results_dir_name, params)
        eval_cfg_path = write_eval_config(args.mode, results_dir_name)

        if args.skip_existing and evaluation_results_path.exists():
            print("[SKIP] existing eval", flush=True)
            n_merges = -1
        elif args.dry_run:
            continue
        else:
            _, n_merges = postprocess_from_saved_seeds(
                args.mode,
                pipeline_cfg_path,
                source_npz,
                results_dir_name,
                md,
                mh,
                mw,
            )
            print(f"n_seed_merges: {n_merges}", flush=True)
            run_command(
                [sys.executable, "tools/evaluation/evaluate.py", "--config", repo_rel(eval_cfg_path)],
                dry_run=False,
            )

        if args.dry_run:
            continue
        row = summarize_one(
            args.checkpoint_label,
            results_dir_name,
            params,
            evaluation_results_path,
            paths["pred_forest"],
        )
        row.update(
            {
                "cluster_selection_epsilon": 0.0,
                "merge_seed_dist": md,
                "merge_height_overlap": mh,
                "merge_weak_seed_frac": mw,
                "n_seed_merges": n_merges,
            }
        )
        print(
            f"metrics: C={row['completeness']:.1f} Ecom={row['commission_error_rate']:.1f} "
            f"F1={row['f1_score']:.1f} Cov={row['iou']:.1f}",
            flush=True,
        )
        rows.append(row)

    if args.dry_run:
        return 0
    if not rows:
        return 1

    csv_path, md_path = write_summary(
        rows, GENERATED_CFG_DIR / f"dupmerge_sweep_{args.mode}_{args.results_prefix}"
    )
    # also write n_seed_merges column
    extra_csv = GENERATED_CFG_DIR / f"dupmerge_sweep_{args.mode}_{args.results_prefix}_with_merges.csv"
    with extra_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary:\n  {csv_path}\n  {md_path}\n  {extra_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
