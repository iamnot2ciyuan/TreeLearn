#!/usr/bin/env python3
"""Sweep duplicate-seed suppression / HDBSCAN cluster_selection_epsilon on L1W.

Reuses saved pointwise predictions. Baseline grouping stays at the finalized
tree_conf/tau_min/tau_vert/tau_off; only seed-merge / CSE knobs vary.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from typing import Any

from run_l1w_postprocess_sweep import (  # noqa: E402
    GENERATED_CFG_DIR,
    REPO_ROOT,
    build_mode_paths,
    dump_yaml,
    format_num,
    load_yaml,
    parse_optional_float,
    postprocess_once,
    repo_rel,
    run_command,
    source_pointwise_path,
    summarize_one,
    write_eval_config,
)
from run_l1w_workflow import PIPELINE_BASE_CFG  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["raw_forest", "eval_subset"], default="eval_subset")
    p.add_argument("--source-results-dir-name", required=True)
    p.add_argument("--pointwise-npz", default=None)
    p.add_argument("--results-prefix", default="dupseed_sweep")
    p.add_argument("--checkpoint-label", default="epoch720")
    p.add_argument("--tree-conf-thresh", type=float, default=0.50)
    p.add_argument("--tau-min", type=int, default=100)
    p.add_argument("--tau-vert", type=parse_optional_float, default=0.35)
    p.add_argument("--tau-off", type=parse_optional_float, default=6.0)
    p.add_argument("--outer-remove", type=parse_optional_float, default=None)
    p.add_argument(
        "--merge-seed-dist",
        type=parse_optional_float,
        nargs="+",
        default=[None],
        help="Offset-shifted seed-center merge distances (m); use none to disable.",
    )
    p.add_argument(
        "--merge-height-overlap",
        type=parse_optional_float,
        nargs="+",
        default=[None],
        help="Min seed Z-interval overlap; use none to disable.",
    )
    p.add_argument(
        "--merge-weak-seed-frac",
        type=parse_optional_float,
        nargs="+",
        default=[None],
        help="Max min/max seed-count ratio; use none to disable.",
    )
    p.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        nargs="+",
        default=[0.0],
        help="HDBSCAN cluster_selection_epsilon values (m).",
    )
    p.add_argument(
        "--focused",
        action="store_true",
        help=(
            "Ignore merge/CSE lists and run a small calibrated set: "
            "baseline; md=0.5/1.0/3.0 with mh/mw; CSE-only 0.25/0.35/0.50."
        ),
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_id_for(params: dict[str, Any]) -> str:
    return (
        f"tc{format_num(params['tree_conf_thresh'])}_"
        f"tm{format_num(params['tau_min'])}_"
        f"cse{format_num(params['cluster_selection_epsilon'])}_"
        f"md{format_num(params['merge_seed_dist'])}_"
        f"mh{format_num(params['merge_height_overlap'])}_"
        f"mw{format_num(params['merge_weak_seed_frac'])}"
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
    cfg["grouping"]["cluster_selection_epsilon"] = params["cluster_selection_epsilon"]
    cfg["grouping"]["merge_seed_dist"] = params["merge_seed_dist"]
    cfg["grouping"]["merge_height_overlap"] = params["merge_height_overlap"]
    cfg["grouping"]["merge_weak_seed_frac"] = params["merge_weak_seed_frac"]

    cfg.setdefault("shape_cfg", {})
    cfg["shape_cfg"]["outer_remove"] = params["outer_remove"]

    config_path = GENERATED_CFG_DIR / f"pipeline_l1w_dupseed_{mode}_{results_dir_name}.yaml"
    dump_yaml(cfg, config_path)
    return config_path


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
        "cluster_selection_epsilon",
        "merge_seed_dist",
        "merge_height_overlap",
        "merge_weak_seed_frac",
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
    # summarize_one does not yet include merge fields; enrich here.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["f1_score"])
    low_c = min(rows, key=lambda r: (r["commission_error_rate"], -r["iou"]))
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# L1W duplicate-seed / CSE sweep\n\n")
        f.write(
            f"- Best F1: `{best['run_name']}` "
            f"(F1={best['f1_score']:.1f}, Ecom={best['commission_error_rate']:.1f}, Cov={best['iou']:.1f})\n"
        )
        f.write(
            f"- Lowest Ecom: `{low_c['run_name']}` "
            f"(Ecom={low_c['commission_error_rate']:.1f}, F1={low_c['f1_score']:.1f}, Cov={low_c['iou']:.1f})\n\n"
        )
        f.write(
            "| run | cse | md | mh | mw | C | Eom | Ecom | F1 | Cov |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for row in rows:
            f.write(
                "| {run_name} | {cluster_selection_epsilon} | {merge_seed_dist} | "
                "{merge_height_overlap} | {merge_weak_seed_frac} | "
                "{completeness:.1f} | {omission_error_rate:.1f} | {commission_error_rate:.1f} | "
                "{f1_score:.1f} | {iou:.1f} |\n".format(**row)
            )
    return csv_path, md_path


def main() -> int:
    args = parse_args()
    pointwise_npz = Path(args.pointwise_npz) if args.pointwise_npz else source_pointwise_path(
        args.mode, args.source_results_dir_name
    )
    pointwise_npz = pointwise_npz if pointwise_npz.is_absolute() else (REPO_ROOT / pointwise_npz)

    if args.focused:
        # Calibrated from commission-vs-adjacent analysis on tm100:
        # - 0.15–0.50 m is too tight (closest commission sc-dist ≈ 0.96 m)
        # - md=1.0/mh=0.60/mw=0.25 safely hits 298–299 only
        # - md=3.0/mh=0.75/mw=0.55 also targets 252–253 (some adj risk)
        combos = [
            (0.0, None, None, None),
            (0.0, 0.50, 0.60, 0.25),
            (0.0, 1.00, 0.60, 0.25),
            (0.0, 1.00, 0.75, 0.25),
            (0.0, 3.00, 0.75, 0.55),
            (0.25, None, None, None),
            (0.35, None, None, None),
            (0.50, None, None, None),
        ]
    else:
        # Cartesian product. Merge is active only when all three merge_* are non-None.
        combos = list(
            itertools.product(
                args.cluster_selection_epsilon,
                args.merge_seed_dist,
                args.merge_height_overlap,
                args.merge_weak_seed_frac,
            )
        )

    print("=== L1W duplicate-seed / CSE sweep ===")
    print(f"pointwise : {pointwise_npz}")
    print(f"num_runs  : {len(combos)}")

    if not args.dry_run and not pointwise_npz.exists():
        raise FileNotFoundError(pointwise_npz)

    rows: list[dict[str, Any]] = []
    for cse, md, mh, mw in combos:
        # Skip incoherent partial-merge settings (any-but-not-all null).
        merge_vals = (md, mh, mw)
        n_none = sum(v is None for v in merge_vals)
        if n_none not in (0, 3):
            print(f"[SKIP] partial merge config cse={cse} md={md} mh={mh} mw={mw}")
            continue

        params = {
            "tree_conf_thresh": args.tree_conf_thresh,
            "tau_min": args.tau_min,
            "tau_vert": args.tau_vert,
            "tau_off": args.tau_off,
            "outer_remove": args.outer_remove,
            "cluster_selection_epsilon": cse,
            "merge_seed_dist": md,
            "merge_height_overlap": mh,
            "merge_weak_seed_frac": mw,
        }
        rid = run_id_for(params)
        results_dir_name = f"{args.results_prefix}_{rid}"
        paths = build_mode_paths(args.mode, results_dir_name)
        evaluation_results_path = paths["evaluation_dir"] / "evaluation_results.pt"

        print(f"\n--- {results_dir_name} ---")
        print(f"params: {params}")

        pipeline_cfg_path = write_pipeline_config(args.mode, results_dir_name, params)
        eval_cfg_path = write_eval_config(args.mode, results_dir_name)

        if args.skip_existing and evaluation_results_path.exists():
            print("[SKIP] existing eval")
        elif args.dry_run:
            continue
        else:
            postprocess_once(args.mode, pipeline_cfg_path, pointwise_npz, results_dir_name)
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
                "cluster_selection_epsilon": cse,
                "merge_seed_dist": md,
                "merge_height_overlap": mh,
                "merge_weak_seed_frac": mw,
            }
        )
        rows.append(row)

    if args.dry_run:
        print("[DRY-RUN] done")
        return 0
    if not rows:
        print("[WARN] no rows")
        return 1

    csv_path, md_path = write_summary(
        rows, GENERATED_CFG_DIR / f"dupseed_sweep_{args.mode}_{args.results_prefix}"
    )
    print(f"\nSummary:\n  {csv_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
