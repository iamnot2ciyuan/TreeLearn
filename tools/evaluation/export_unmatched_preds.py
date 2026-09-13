#!/usr/bin/env python3
"""Export unmatched predicted instances with diagnostic features and class hypotheses.

Computes per-instance geometry / confidence / stability stats for all unmatched
predictions from a finished evaluation_results.pt, then assigns a hypothesized
failure mode to help diagnose commission / over-segmentation sources:

  - overseg_duplicate   : same GT already matched / shared by other preds
  - crown_branch_frag   : elevated, small canopy/branch fragment
  - understory_veg      : low height / filtered by precision (likely unlabeled veg)
  - plot_edge_fragment  : near plot hull boundary
  - boundary_overpush   : sits between neighboring trees with moderate GT overlap
  - other               : residual / ambiguous
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point

from tree_learn.util import get_hull, load_data, save_data

REPO_ROOT = Path(__file__).resolve().parents[2]
NON_TREE = 0
TREE_CLASS = 0  # semantic class index for tree in logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-results",
        type=Path,
        default=REPO_ROOT
        / "data/pp_epoch720_tmin_tc0p5_tm100_tv0p35_to6p0_ornone/full_forest/evaluation/evaluation_results.pt",
    )
    p.add_argument(
        "--pred-forest",
        type=Path,
        default=REPO_ROOT
        / "data/pp_epoch720_tmin_tc0p5_tm100_tv0p35_to6p0_ornone/full_forest/L1W_voxelized01_for_eval.laz",
    )
    p.add_argument(
        "--gt-forest",
        type=Path,
        default=REPO_ROOT / "data/benchmark/L1W_voxelized01_for_eval.laz",
    )
    p.add_argument(
        "--pointwise",
        type=Path,
        default=None,
        help=(
            "Pointwise/postprocess npz with instance_preds and "
            "instance_preds_after_initial_clustering. "
            "Default: <pred-forest-dir>/../postprocess_results/postprocess_results.npz "
            "if present, else epoch720 base pointwise_results.npz."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <eval-results-dir>/unmatched_pred_diagnosis",
    )
    p.add_argument("--edge-buffer-m", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--export-laz", action="store_true", default=True, help="Also write unmatched points laz")
    p.add_argument("--no-export-laz", action="store_false", dest="export_laz")
    return p.parse_args()


def softmax_tree_prob(logits: np.ndarray, tree_class: int = TREE_CLASS) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e[:, tree_class] / e.sum(axis=1)).astype(np.float32)


def instance_geometry(coords: np.ndarray) -> dict[str, float]:
    """coords: (N,3) for one instance."""
    if len(coords) == 0:
        return {
            "n_points": 0,
            "z_min": np.nan,
            "z_max": np.nan,
            "tree_height": np.nan,
            "vertical_span": np.nan,
            "crown_diameter": np.nan,
            "xy_extent": np.nan,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "centroid_z": np.nan,
            "base_x": np.nan,
            "base_y": np.nan,
            "base_z": np.nan,
        }
    xyz = coords.astype(np.float64)
    z = xyz[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    height = z_max - z_min
    # vertical span: 5th–95th percentile to reduce outlier sensitivity
    z05, z95 = np.percentile(z, [5, 95])
    vertical_span = float(z95 - z05)

    xy = xyz[:, :2]
    centroid = xyz.mean(axis=0)
    # trunk base ≈ mean of lowest 5% points
    n_base = max(1, int(0.05 * len(xyz)))
    base_idx = np.argpartition(z, n_base - 1)[:n_base]
    base = xyz[base_idx].mean(axis=0)

    # crown diameter from points in upper 50% of height
    upper = xyz[z >= (z_min + 0.5 * height)] if height > 0 else xyz
    if len(upper) >= 3:
        uc = upper[:, :2].mean(axis=0)
        rad = np.linalg.norm(upper[:, :2] - uc, axis=1)
        crown_diameter = float(2.0 * np.percentile(rad, 90))
    else:
        rad = np.linalg.norm(xy - centroid[:2], axis=1)
        crown_diameter = float(2.0 * np.percentile(rad, 90)) if len(rad) else 0.0

    xy_extent = float(np.ptp(xy, axis=0).max()) if len(xy) else 0.0
    return {
        "n_points": int(len(coords)),
        "z_min": z_min,
        "z_max": z_max,
        "tree_height": float(height),
        "vertical_span": vertical_span,
        "crown_diameter": crown_diameter,
        "xy_extent": xy_extent,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "base_x": float(base[0]),
        "base_y": float(base[1]),
        "base_z": float(base[2]),
    }


def build_instance_index(labels: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    unique, start, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
    out: dict[int, np.ndarray] = {}
    for u, s, c in zip(unique, start, counts):
        if int(u) == NON_TREE:
            continue
        out[int(u)] = order[s : s + c]
    return out


def nearest_gt_stats(
    pred_xy: np.ndarray,
    pred_base_xy: np.ndarray,
    gt_centroids: dict[int, np.ndarray],
    gt_bases: dict[int, np.ndarray],
    gt_heights: dict[int, float],
) -> dict[str, float]:
    if not gt_centroids:
        return {
            "nearest_gt_id": np.nan,
            "dist_centroid_to_nearest_gt": np.nan,
            "dist_base_to_nearest_gt": np.nan,
            "nearest_gt_height": np.nan,
            "second_nearest_gt_id": np.nan,
            "dist_centroid_to_second_gt": np.nan,
        }
    ids = np.array(list(gt_centroids.keys()), dtype=np.int64)
    cents = np.stack([gt_centroids[i][:2] for i in ids])
    bases = np.stack([gt_bases[i][:2] for i in ids])
    d_c = np.linalg.norm(cents - pred_xy, axis=1)
    d_b = np.linalg.norm(bases - pred_base_xy, axis=1)
    i1 = int(np.argmin(d_c))
    order = np.argsort(d_c)
    i2 = int(order[1]) if len(order) > 1 else i1
    return {
        "nearest_gt_id": int(ids[i1]),
        "dist_centroid_to_nearest_gt": float(d_c[i1]),
        "dist_base_to_nearest_gt": float(d_b[i1]),
        "nearest_gt_height": float(gt_heights[int(ids[i1])]),
        "second_nearest_gt_id": int(ids[i2]),
        "dist_centroid_to_second_gt": float(d_c[i2]),
    }


def classify_row(row: dict[str, Any], edge_buffer_m: float) -> str:
    """Heuristic failure-mode label for one unmatched prediction."""
    is_commission = bool(row["is_commission"])
    corr_gt = row["corresponding_gt"]
    dist = row["dist_centroid_to_nearest_gt"]
    dist2 = row["dist_centroid_to_second_gt"]
    height = row["tree_height"]
    gt_h = row["nearest_gt_height"]
    base_rel = (row["base_z"] - row["z_min"]) if np.isfinite(row["base_z"]) else 0.0
    # relative base elevation vs nearest GT height
    elev_ratio = np.nan
    if np.isfinite(height) and np.isfinite(gt_h) and gt_h > 1e-6:
        # how high the fragment sits relative to GT tree
        elev_ratio = (row["z_min"] - (row["nearest_gt_base_z"] if "nearest_gt_base_z" in row else row["z_min"])) / gt_h
        # better: z_min of pred vs base of GT — use stored nearest_gt_base_z if present
    if "nearest_gt_base_z" in row and np.isfinite(row["nearest_gt_base_z"]) and np.isfinite(gt_h) and gt_h > 1e-6:
        elev_ratio = (row["z_min"] - row["nearest_gt_base_z"]) / gt_h

    edge_dist = row["dist_to_plot_boundary"]
    n_share = row["n_preds_sharing_corr_gt"]
    gt_already_matched = bool(row["corr_gt_already_matched"])
    mean_conf = row["mean_tree_conf"]
    n_pts = row["n_points"]

    # 1) plot edge
    if np.isfinite(edge_dist) and edge_dist <= edge_buffer_m:
        return "plot_edge_fragment"

    # 2) understory / unlabeled veg (precision-filtered or tiny/low)
    if (not is_commission) and (
        (np.isfinite(height) and height < 4.0)
        or (np.isfinite(mean_conf) and mean_conf < 0.55)
        or n_pts < 80
    ):
        return "understory_veg"
    if np.isfinite(height) and height < 3.0 and (not is_commission or n_pts < 150):
        return "understory_veg"

    # 3) oversegmentation / duplicate of an already-detected tree
    if is_commission and (gt_already_matched or n_share >= 2):
        return "overseg_duplicate"
    if (
        is_commission
        and np.isfinite(dist)
        and dist < 3.0
        and np.isfinite(gt_h)
        and np.isfinite(height)
        and height < 0.7 * gt_h
    ):
        return "overseg_duplicate"

    # 4) crown / branch fragment: elevated base, moderate size, near a GT
    if (
        np.isfinite(elev_ratio)
        and elev_ratio > 0.25
        and np.isfinite(dist)
        and dist < 8.0
        and np.isfinite(height)
        and height < 0.6 * (gt_h if np.isfinite(gt_h) and gt_h > 0 else 1e9)
    ):
        return "crown_branch_frag"
    if (
        np.isfinite(row["z_min"])
        and np.isfinite(gt_h)
        and np.isfinite(dist)
        and dist < 6.0
        and row["vertical_span"] < 0.45 * gt_h
        and n_pts < 2000
        and elev_ratio is not np.nan
        and np.isfinite(elev_ratio)
        and elev_ratio > 0.2
    ):
        return "crown_branch_frag"

    # 5) boundary over-push: between two GTs, commission or strong geometric proximity to both
    if (
        np.isfinite(dist)
        and np.isfinite(dist2)
        and dist < 6.0
        and dist2 < 8.0
        and abs(dist2 - dist) < 4.0
        and is_commission
    ):
        return "boundary_overpush"
    if (
        np.isfinite(dist)
        and np.isfinite(dist2)
        and dist < 5.0
        and dist2 < 7.0
        and row.get("mean_abs_offset_xy", np.nan) == row.get("mean_abs_offset_xy", np.nan)
        and np.isfinite(row.get("mean_abs_offset_xy", np.nan))
        and row["mean_abs_offset_xy"] > 1.5
        and is_commission
    ):
        return "boundary_overpush"

    # fallback: commission near GT → overseg; else understory-like residual
    if is_commission and np.isfinite(dist) and dist < 5.0:
        return "overseg_duplicate"
    if not is_commission:
        return "understory_veg"
    return "other"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    counts = Counter(r["hypothesis"] for r in rows)
    comm = [r for r in rows if r["is_commission"]]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Unmatched prediction instance diagnosis\n\n")
        f.write(f"- Pred forest: `{meta['pred_forest']}`\n")
        f.write(f"- GT forest: `{meta['gt_forest']}`\n")
        f.write(f"- Eval results: `{meta['eval_results']}`\n")
        f.write(f"- Unmatched preds (all): **{len(rows)}**\n")
        f.write(f"- Commission (precision-filtered): **{len(comm)}**\n")
        f.write(f"- Matched preds: {meta.get('n_matched_preds')}\n\n")
        f.write("## Hypothesis counts\n\n")
        f.write("| hypothesis | n | n_commission |\n| --- | --- | --- |\n")
        for hyp, n in sorted(counts.items(), key=lambda x: -x[1]):
            nc = sum(1 for r in rows if r["hypothesis"] == hyp and r["is_commission"])
            f.write(f"| {hyp} | {n} | {nc} |\n")
        f.write("\n## Hypothesis definitions\n\n")
        f.write("- `overseg_duplicate`: same tree split / duplicate of an already matched GT\n")
        f.write("- `crown_branch_frag`: elevated canopy/branch fragment near a real tree\n")
        f.write("- `understory_veg`: understory / unlabeled vegetation (often precision-filtered)\n")
        f.write("- `plot_edge_fragment`: incomplete instance near plot boundary\n")
        f.write("- `boundary_overpush`: sits between neighboring trees (offset/boundary push)\n")
        f.write("- `other`: ambiguous residual\n\n")
        f.write("## Commission instances (detail)\n\n")
        f.write(
            "| pred_id | corr_gt | hyp | n_pts | height | vspan | crown_d | conf | stability | "
            "dist_gt | edge_dist |\n"
        )
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in sorted(comm, key=lambda x: x["pred_id"]):
            f.write(
                f"| {r['pred_id']} | {r['corresponding_gt']} | {r['hypothesis']} | {r['n_points']} | "
                f"{r['tree_height']:.1f} | {r['vertical_span']:.1f} | {r['crown_diameter']:.1f} | "
                f"{r['mean_tree_conf']:.3f} | {r['cluster_stability']:.3f} | "
                f"{r['dist_centroid_to_nearest_gt']:.2f} | {r['dist_to_plot_boundary']:.2f} |\n"
            )


def main() -> int:
    args = parse_args()
    eval_path = args.eval_results if args.eval_results.is_absolute() else REPO_ROOT / args.eval_results
    pred_path = args.pred_forest if args.pred_forest.is_absolute() else REPO_ROOT / args.pred_forest
    gt_path = args.gt_forest if args.gt_forest.is_absolute() else REPO_ROOT / args.gt_forest
    if args.pointwise is None:
        candidate = pred_path.parent.parent / "postprocess_results" / "postprocess_results.npz"
        fallback = (
            REPO_ROOT
            / "data/results_l1w_eval_subset_epoch720_base/pointwise_results/pointwise_results.npz"
        )
        pw_path = candidate if candidate.exists() else fallback
    else:
        pw_path = args.pointwise if args.pointwise.is_absolute() else REPO_ROOT / args.pointwise
    out_dir = args.out_dir or (eval_path.parent / "unmatched_pred_diagnosis")
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] evaluation: {eval_path}")
    results = torch.load(eval_path, map_location="cpu")
    det = results["detection_results"]
    non_matched_preds = np.asarray(det["non_matched_preds"], dtype=np.int64)
    corr_gts = np.asarray(det["non_matched_preds_corresponding_gt"], dtype=np.float64)
    matched_preds = set(int(x) for x in np.asarray(det["matched_preds"]))
    matched_gts = set(int(x) for x in np.asarray(det["matched_gts"]))
    commission_ids = set(int(x) for x in np.asarray(det["non_matched_preds_filtered"]))

    corr_map = {int(p): (None if np.isnan(g) else int(g)) for p, g in zip(non_matched_preds, corr_gts)}
    share_counts: dict[int | None, int] = defaultdict(int)
    for g in corr_map.values():
        if g is not None:
            share_counts[g] += 1

    print(f"[LOAD] pred forest: {pred_path}")
    pred = load_data(str(pred_path))
    pred_coords = pred[:, :3].astype(np.float64)
    pred_labels = pred[:, 3].astype(np.int64)
    print(f"[LOAD] gt forest: {gt_path}")
    gt = load_data(str(gt_path))
    gt_coords = gt[:, :3].astype(np.float64)
    gt_labels = gt[:, 3].astype(np.int64)

    print("[INDEX] building instance indices...")
    pred_idx = build_instance_index(pred_labels)
    gt_idx = build_instance_index(gt_labels)

    gt_centroids: dict[int, np.ndarray] = {}
    gt_bases: dict[int, np.ndarray] = {}
    gt_heights: dict[int, float] = {}
    gt_base_z: dict[int, float] = {}
    for gid, idxs in gt_idx.items():
        geom = instance_geometry(gt_coords[idxs])
        gt_centroids[gid] = np.array([geom["centroid_x"], geom["centroid_y"], geom["centroid_z"]])
        gt_bases[gid] = np.array([geom["base_x"], geom["base_y"], geom["base_z"]])
        gt_heights[gid] = geom["tree_height"]
        gt_base_z[gid] = geom["base_z"]

    # plot hull for edge distance
    print("[HULL] computing plot boundary...")
    tree_xy = pred_coords[pred_labels != NON_TREE][:, :2]
    if len(tree_xy) > 100_000:
        rng = np.random.default_rng(0)
        tree_xy = tree_xy[rng.choice(len(tree_xy), 100_000, replace=False)]
    try:
        hull = get_hull(tree_xy, args.alpha)
        hull_boundary = hull.boundary
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] hull failed ({exc}); edge distances set to nan")
        hull_boundary = None

    # pointwise confidence + stability + offset
    print(f"[LOAD] pointwise: {pw_path}")
    pw = np.load(pw_path)
    pw_inst = pw["instance_preds"].astype(np.int64)
    pw_init = pw["instance_preds_after_initial_clustering"].astype(np.int64)
    pw_logits = pw["semantic_prediction_logits"]
    pw_off = pw["offset_predictions"].astype(np.float32)
    tree_prob = softmax_tree_prob(pw_logits, TREE_CLASS)

    pw_idx = build_instance_index(pw_inst)
    conf_map: dict[int, float] = {}
    stab_map: dict[int, float] = {}
    off_xy_map: dict[int, float] = {}
    off_z_map: dict[int, float] = {}
    for pid, idxs in pw_idx.items():
        conf_map[pid] = float(tree_prob[idxs].mean())
        n_final = len(idxs)
        n_seed = int(np.sum(pw_init[idxs] == pid))
        stab_map[pid] = float(n_seed / n_final) if n_final else np.nan
        off = pw_off[idxs]
        off_xy_map[pid] = float(np.linalg.norm(off[:, :2], axis=1).mean())
        off_z_map[pid] = float(np.abs(off[:, 2]).mean())

    print(f"[COMPUTE] features for {len(non_matched_preds)} unmatched preds...")
    rows: list[dict[str, Any]] = []
    unmatched_point_masks = []
    hyp_label_code = {
        "overseg_duplicate": 1,
        "crown_branch_frag": 2,
        "understory_veg": 3,
        "plot_edge_fragment": 4,
        "boundary_overpush": 5,
        "other": 6,
    }

    for pid in non_matched_preds:
        pid = int(pid)
        idxs = pred_idx.get(pid)
        if idxs is None or len(idxs) == 0:
            geom = instance_geometry(np.zeros((0, 3)))
        else:
            geom = instance_geometry(pred_coords[idxs])
            unmatched_point_masks.append((idxs, pid))

        nearest = nearest_gt_stats(
            np.array([geom["centroid_x"], geom["centroid_y"]]),
            np.array([geom["base_x"], geom["base_y"]]),
            gt_centroids,
            gt_bases,
            gt_heights,
        )
        ng = nearest["nearest_gt_id"]
        nearest_gt_base_z = gt_base_z[int(ng)] if ng == ng and int(ng) in gt_base_z else np.nan

        if hull_boundary is not None and np.isfinite(geom["centroid_x"]):
            edge_dist = float(np.asarray(hull_boundary.distance(Point(geom["centroid_x"], geom["centroid_y"]))).reshape(-1)[0])
        else:
            edge_dist = np.nan

        corr_gt = corr_map.get(pid)
        row: dict[str, Any] = {
            "pred_id": pid,
            "is_commission": pid in commission_ids,
            "corresponding_gt": corr_gt if corr_gt is not None else np.nan,
            "corr_gt_already_matched": bool(corr_gt is not None and corr_gt in matched_gts),
            "n_preds_sharing_corr_gt": int(share_counts[corr_gt]) if corr_gt is not None else 0,
            **geom,
            **nearest,
            "nearest_gt_base_z": nearest_gt_base_z,
            "dist_to_plot_boundary": edge_dist,
            "mean_tree_conf": conf_map.get(pid, np.nan),
            "cluster_stability": stab_map.get(pid, np.nan),
            "mean_abs_offset_xy": off_xy_map.get(pid, np.nan),
            "mean_abs_offset_z": off_z_map.get(pid, np.nan),
        }
        row["hypothesis"] = classify_row(row, args.edge_buffer_m)
        rows.append(row)

    rows.sort(key=lambda r: (not r["is_commission"], r["hypothesis"], r["pred_id"]))

    csv_path = out_dir / "unmatched_preds_features.csv"
    md_path = out_dir / "unmatched_preds_summary.md"
    json_path = out_dir / "unmatched_preds_hypothesis_counts.json"
    write_csv(csv_path, rows)
    write_md(
        md_path,
        rows,
        {
            "pred_forest": str(pred_path),
            "gt_forest": str(gt_path),
            "eval_results": str(eval_path),
            "n_matched_preds": len(matched_preds),
        },
    )
    counts = Counter(r["hypothesis"] for r in rows)
    json_path.write_text(json.dumps(dict(counts), indent=2), encoding="utf-8")

    if args.export_laz and unmatched_point_masks:
        print("[EXPORT] writing unmatched points laz...")
        keep = np.zeros(len(pred_coords), dtype=bool)
        hyp_attr = np.zeros(len(pred_coords), dtype=np.int32)
        for idxs, pid in unmatched_point_masks:
            keep[idxs] = True
            hyp = next(r["hypothesis"] for r in rows if r["pred_id"] == pid)
            hyp_attr[idxs] = hyp_label_code[hyp]
        export = np.hstack(
            [
                pred_coords[keep],
                pred_labels[keep].reshape(-1, 1),
                hyp_attr[keep].reshape(-1, 1),
            ]
        )
        # save_data expects xyz + label; extra column not supported — write via laspy-compatible 4-col and note hyp in treeID remap
        # Store hypothesis code in a side npy alongside standard laz of unmatched trees.
        save_data(export[:, :4], "laz", "unmatched_preds", str(out_dir))
        np.save(out_dir / "unmatched_preds_hypothesis_codes.npy", export[:, 4].astype(np.int32))
        (out_dir / "hypothesis_code_legend.txt").write_text(
            "\n".join(f"{v}: {k}" for k, v in hyp_label_code.items()) + "\n",
            encoding="utf-8",
        )

    print("\n=== Hypothesis counts ===")
    for hyp, n in sorted(counts.items(), key=lambda x: -x[1]):
        nc = sum(1 for r in rows if r["hypothesis"] == hyp and r["is_commission"])
        print(f"  {hyp:22s}  n={n:4d}  (commission={nc})")
    print(f"\nWrote:\n  {csv_path}\n  {md_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
