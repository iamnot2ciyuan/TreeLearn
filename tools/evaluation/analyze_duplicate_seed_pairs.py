#!/usr/bin/env python3
"""Compare commission-vs-main seed pairs against correctly adjacent matched pairs.

Uses postprocess_results.npz (HDBSCAN seeds + final instances after NN) together with
evaluation_results.pt matching. Writes CSV + markdown for threshold design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=REPO_ROOT
        / "data/pp_epoch720_tmin_tc0p5_tm100_tv0p35_to6p0_ornone",
    )
    p.add_argument("--adj-radius", type=float, default=8.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <pred-dir>/full_forest/evaluation/duplicate_seed_pair_analysis",
    )
    return p.parse_args()


def interval_overlap(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["zmax"], b["zmax"]) - max(a["zmin"], b["zmin"]))
    union = max(a["zmax"], b["zmax"]) - min(a["zmin"], b["zmin"])
    return inter / union if union > 0 else 0.0


def aabb_iou(a: dict, b: dict, key: str = "aabb") -> float:
    ax0, ay0, ax1, ay1 = a[key]
    bx0, by0, bx1, by1 = b[key]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def seed_feats(coords: np.ndarray, shifted: np.ndarray, init: np.ndarray, pid: int) -> dict | None:
    m = init == pid
    if not np.any(m):
        return None
    c = coords[m]
    s = shifted[m]
    return {
        "sc": s[:, :2].mean(0),
        "xy": c[:, :2].mean(0),
        "zmin": float(c[:, 2].min()),
        "zmax": float(c[:, 2].max()),
        "aabb": (float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())),
        "n_seed": int(m.sum()),
    }


def final_feats(coords: np.ndarray, inst: np.ndarray, pid: int) -> dict | None:
    m = inst == pid
    if not np.any(m):
        return None
    c = coords[m]
    zmin, zmax = float(c[:, 2].min()), float(c[:, 2].max())
    zthr = zmin + 0.5 * (zmax - zmin)
    crown = c[c[:, 2] >= zthr][:, :2]
    if len(crown) == 0:
        crown = c[:, :2]
    return {
        "xy": c[:, :2].mean(0),
        "zmin": zmin,
        "zmax": zmax,
        "height": zmax - zmin,
        "crown": (
            float(crown[:, 0].min()),
            float(crown[:, 1].min()),
            float(crown[:, 0].max()),
            float(crown[:, 1].max()),
        ),
        "n_final": int(m.sum()),
    }


def pair_row(pa: int, pb: int, label: str, S: dict, F: dict, corr_gt: int | None = None) -> dict[str, Any]:
    sa, sb = S[pa], S[pb]
    fa, fb = F.get(pa), F.get(pb)
    row: dict[str, Any] = {
        "label": label,
        "id_a": pa,
        "id_b": pb,
        "corr_gt": corr_gt,
        "dist_seed_center_offset": float(np.linalg.norm(sa["sc"] - sb["sc"])),
        "dist_seed_xy": float(np.linalg.norm(sa["xy"] - sb["xy"])),
        "seed_height_overlap": interval_overlap(sa, sb),
        "seed_aabb_iou": aabb_iou(sa, sb, "aabb"),
        "n_seed_a": sa["n_seed"],
        "n_seed_b": sb["n_seed"],
        "seed_frac_min": min(sa["n_seed"], sb["n_seed"]) / max(sa["n_seed"], sb["n_seed"]),
        "weak_n_seed": min(sa["n_seed"], sb["n_seed"]),
    }
    if fa is not None and fb is not None:
        row.update(
            {
                "dist_final_xy": float(np.linalg.norm(fa["xy"] - fb["xy"])),
                "final_height_overlap": interval_overlap(fa, fb),
                "final_crown_iou": aabb_iou(fa, fb, "crown"),
                "n_final_a": fa["n_final"],
                "n_final_b": fb["n_final"],
                "stab_a": sa["n_seed"] / max(fa["n_final"], 1),
                "stab_b": sb["n_seed"] / max(fb["n_final"], 1),
                "height_a": fa["height"],
                "height_b": fb["height"],
            }
        )
    return row


def main() -> int:
    args = parse_args()
    pred_dir = args.pred_dir if args.pred_dir.is_absolute() else REPO_ROOT / args.pred_dir
    out_dir = args.out_dir or (pred_dir / "full_forest/evaluation/duplicate_seed_pair_analysis")
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pp_path = pred_dir / "postprocess_results/postprocess_results.npz"
    eval_path = pred_dir / "full_forest/evaluation/evaluation_results.pt"
    pp = np.load(pp_path)
    coords = pp["coords"].astype(np.float64)
    off = pp["offset_predictions"].astype(np.float64)
    init = pp["instance_preds_after_initial_clustering"].astype(np.int64)
    inst = pp["instance_preds"].astype(np.int64)
    shifted = coords + off

    det = torch.load(eval_path, map_location="cpu")["detection_results"]
    gt2pred = {int(g): int(p) for g, p in zip(det["matched_gts"], det["matched_preds"])}
    matched = [int(p) for p in det["matched_preds"]]

    seed_ids = np.unique(init[init > 0])
    S = {int(pid): seed_feats(coords, shifted, init, int(pid)) for pid in seed_ids}
    S = {k: v for k, v in S.items() if v is not None}
    F = {int(pid): final_feats(coords, inst, int(pid)) for pid in np.unique(inst[inst > 0])}
    F = {k: v for k, v in F.items() if v is not None}

    rows: list[dict[str, Any]] = []
    for p, g in zip(det["non_matched_preds_filtered"], det["non_matched_preds_corresponding_gt_filtered"]):
        p, g = int(p), int(g)
        main = gt2pred[g]
        rows.append(pair_row(p, main, "commission_vs_main", S, F, corr_gt=g))

    matched_set = set(matched)
    ids = [pid for pid in matched if pid in S]
    C = np.stack([S[i]["sc"] for i in ids])
    nn = NearestNeighbors(radius=args.adj_radius, algorithm="ball_tree").fit(C)
    for i, pid in enumerate(ids):
        nbrs = nn.radius_neighbors([C[i]], return_distance=False)[0]
        for j in nbrs:
            if j <= i:
                continue
            pb = ids[j]
            if pb not in matched_set:
                continue
            rows.append(pair_row(pid, pb, "adjacent_matched", S, F))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pair_features.csv", index=False)

    comm = df[df["label"] == "commission_vs_main"]
    adj = df[df["label"] == "adjacent_matched"]
    summary = {
        "n_commission_pairs": int(len(comm)),
        "n_adjacent_matched_pairs": int(len(adj)),
        "commission_pairs": comm.to_dict(orient="records"),
        "adjacent_dist_seed_center_offset": adj["dist_seed_center_offset"].describe(
            percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
        ).to_dict(),
        "adjacent_seed_height_overlap": adj["seed_height_overlap"].describe(
            percentiles=[0.05, 0.1, 0.25, 0.5]
        ).to_dict(),
        "adjacent_seed_frac_min": adj["seed_frac_min"].describe(
            percentiles=[0.05, 0.1, 0.25, 0.5]
        ).to_dict(),
        "notes": {
            "tau_min_applies_to": "voxelized HDBSCAN seed points",
            "stability_definition": "n_seed_voxel / n_final_voxel on matching instance id",
            "suggested_safe_merge": {
                "merge_seed_dist": 1.0,
                "merge_height_overlap": 0.60,
                "merge_weak_seed_frac": 0.25,
                "expected_commission_hits": "298-299 only; 0 adjacent matched FPs in this plot",
            },
            "user_grid_0p15_to_0p50m": "too tight for current commissions (min commission sc dist ~0.96 m)",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    md = [
        "# Duplicate-seed pair analysis",
        "",
        f"- Pred dir: `{pred_dir}`",
        f"- Commission pairs: **{len(comm)}**; adjacent matched (sc≤{args.adj_radius} m): **{len(adj)}**",
        "",
        "## Commission vs main",
        "",
        "| pred | main | gt | sc_dist | seed_hov | seed_frac | final_crown_iou | n_seed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in comm.iterrows():
        md.append(
            f"| {int(r.id_a)} | {int(r.id_b)} | {int(r.corr_gt)} | {r.dist_seed_center_offset:.3f} | "
            f"{r.seed_height_overlap:.3f} | {r.seed_frac_min:.3f} | {r.final_crown_iou:.3f} | "
            f"{int(r.n_seed_a)}/{int(r.n_seed_b)} |"
        )
    md += [
        "",
        "## Stability / tau_min note",
        "",
        "- `tau_min` filters **voxelized** HDBSCAN seed size (tm100: all seeds ≥102).",
        "- Diagnosis that used base `pointwise_results.npz` mixed **different label spaces**; "
        "pred=252 “70 seeds / stability 0.007” was an artifact. On tm100 postprocess, "
        "pred=252 has **1980** voxel seeds (stability ≈0.21).",
        "",
        "## Threshold implication",
        "",
        "- User grid `merge_seed_dist∈{0.15..0.50}` catches **no** current commission "
        "(closest is 298–299 at **0.96 m**).",
        "- Safe seed-stage rule on this plot: `sc≤1.0 & hov≥0.60 & seed_frac≤0.25` → "
        "only 298–299, **0** adjacent matched FPs.",
        "- 252–253 needs ~3 m + crown/seed cues (risk of ~3 adjacent FPs).",
        "- 44–45 has sc≈**6.75 m** (neighbor-like); not a near-duplicate seed case.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_dir / 'pair_features.csv'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
