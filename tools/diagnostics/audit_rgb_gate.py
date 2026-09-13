#!/usr/bin/env python3
"""Read-only RGB-gate audit for TreeLearn.

Produces:
  work_dirs/rgb_gate_audit/data_audit.json
  work_dirs/rgb_gate_audit/checkpoint_audit.json
  work_dirs/rgb_gate_audit/forward_audit.json
  work_dirs/rgb_gate_audit/report.md

Does not modify model code, training data, or tiles. Does not run full train/infer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "work_dirs" / "rgb_gate_audit"
DEFAULT_CKPT = REPO / "work_dirs" / "train_20260627" / "epoch_720.pth"
SIGMOID_NEG6 = float(1.0 / (1.0 + math.exp(6.0)))  # ~0.002472623


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--train-root", type=Path, default=REPO / "data/train/random_crops/npz")
    p.add_argument("--val-root", type=Path, default=REPO / "data/val/tiles/npz")
    p.add_argument("--tile-root", type=Path, default=REPO / "data/tiles/npz")
    p.add_argument("--train-sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--skip-forward", action="store_true")
    p.add_argument(
        "--multi-tile",
        action="store_true",
        help="Run stratified multi-tile full-backbone causal + gradient audit.",
    )
    p.add_argument("--n-tiles", type=int, default=20)
    p.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Per-tile point cap for GPU causal tests (full tile if smaller).",
    )
    p.add_argument(
        "--multitile-out",
        type=Path,
        default=OUT_DIR / "multitile",
        help="Output directory for multi-tile audit artifacts.",
    )
    return p.parse_args()


def jdump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def percentile_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {k: float("nan") for k in ["min", "max", "mean", "std", "p01", "p05", "p50", "p95", "p99"]}
    qs = np.percentile(x, [1, 5, 50, 95, 99])
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p50": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
    }


def plot_prefix(name: str) -> str:
    stem = Path(name).stem
    # e.g. L1W_7, A1N_1000, L1W_voxelized01_for_eval_12
    if "_voxelized" in stem:
        return stem.split("_voxelized")[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def audit_npz_dir(
    root: Path,
    sample_n: int | None,
    seed: int,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label,
        "root": str(root),
        "exists": root.exists(),
    }
    if not root.exists():
        return result

    files = sorted([p for p in root.glob("*.npz") if p.is_file()])
    result["n_files_total"] = len(files)
    prefixes = Counter(plot_prefix(p.name) for p in files)
    result["prefixes"] = dict(sorted(prefixes.items(), key=lambda kv: (-kv[1], kv[0])))
    result["n_prefixes"] = len(prefixes)

    rng = random.Random(seed)
    if sample_n is None or sample_n >= len(files):
        chosen = files
        result["sampled"] = False
        result["n_checked"] = len(files)
    else:
        chosen = rng.sample(files, sample_n)
        result["sampled"] = True
        result["n_checked"] = sample_n
        result["sample_seed"] = seed

    missing_colors = []
    bad_shape = []
    nan_inf = []
    all_zero = []
    low_var = []
    not_01 = []
    exception_files = []

    # reservoir for global RGB stats (subsample points)
    rgb_chunks: list[np.ndarray] = []
    zero_frac_r = []
    zero_frac_g = []
    zero_frac_b = []
    nunique_r = []
    nunique_g = []
    nunique_b = []
    max_points_for_stats = 2_000_000
    points_collected = 0

    for i, p in enumerate(chosen):
        if (i + 1) % 25 == 0 or (i + 1) == len(chosen):
            print(f"  [{label}] {i+1}/{len(chosen)}", flush=True)
        try:
            with np.load(p) as data:
                fileset = set(data.files)
                if "colors" not in fileset:
                    missing_colors.append(str(p))
                    continue
                c = np.asarray(data["colors"])
                # Length check against points only every file but avoid loading full points when huge:
                n_points = None
                if "points" in fileset:
                    # shape only via .shape on memmap-like; for npz this still decompresses points.
                    # Use colors length as primary; verify points on a sparse schedule.
                    if (i % 10 == 0) or c.ndim != 2:
                        n_points = int(np.asarray(data["points"]).shape[0])
                if c.ndim != 2 or c.shape[1] != 3 or (n_points is not None and c.shape[0] != n_points):
                    bad_shape.append(
                        {
                            "path": str(p),
                            "shape": list(c.shape),
                            "n_points": n_points,
                        }
                    )
                    continue
                c = c.astype(np.float32, copy=False)
                if not np.isfinite(c).all():
                    nan_inf.append(str(p))
                if float(np.max(np.abs(c))) == 0.0:
                    all_zero.append(str(p))
                # low variance on a subsample for speed
                if c.shape[0] > 50_000:
                    step = max(1, c.shape[0] // 50_000)
                    c_stat = c[::step]
                else:
                    c_stat = c
                if float(np.std(c_stat)) < 1e-4:
                    low_var.append(str(p))
                if float(c_stat.min()) < -1e-6 or float(c_stat.max()) > 1.0 + 1e-3:
                    not_01.append({"path": str(p), "min": float(c_stat.min()), "max": float(c_stat.max())})

                zero_frac_r.append(float(np.mean(c_stat[:, 0] == 0)))
                zero_frac_g.append(float(np.mean(c_stat[:, 1] == 0)))
                zero_frac_b.append(float(np.mean(c_stat[:, 2] == 0)))
                for i_ch, bucket in enumerate((nunique_r, nunique_g, nunique_b)):
                    col = c_stat[:, i_ch]
                    if col.size > 100_000:
                        col = col[:: max(1, col.size // 100_000)]
                    bucket.append(int(np.unique(col).size))

                if points_collected < max_points_for_stats:
                    take = min(len(c_stat), max_points_for_stats - points_collected)
                    if take < len(c_stat):
                        idx = np.linspace(0, len(c_stat) - 1, take, dtype=np.int64)
                        rgb_chunks.append(c_stat[idx])
                    else:
                        rgb_chunks.append(c_stat)
                    points_collected += take
        except Exception as exc:  # noqa: BLE001
            exception_files.append({"path": str(p), "reason": str(exc)})

    result["missing_colors"] = {"count": len(missing_colors), "paths": missing_colors[:50]}
    result["bad_shape"] = {"count": len(bad_shape), "examples": bad_shape[:20]}
    result["nan_inf"] = {"count": len(nan_inf), "paths": nan_inf[:50]}
    result["all_zero_rgb"] = {"count": len(all_zero), "paths": all_zero[:50]}
    result["low_variance_rgb"] = {"count": len(low_var), "paths": low_var[:50]}
    result["outside_0_1"] = {"count": len(not_01), "examples": not_01[:20]}
    result["exceptions"] = {"count": len(exception_files), "examples": exception_files[:20]}
    result["channel_zero_fraction_mean"] = {
        "R": float(np.mean(zero_frac_r)) if zero_frac_r else float("nan"),
        "G": float(np.mean(zero_frac_g)) if zero_frac_g else float("nan"),
        "B": float(np.mean(zero_frac_b)) if zero_frac_b else float("nan"),
    }
    result["channel_nunique_mean"] = {
        "R": float(np.mean(nunique_r)) if nunique_r else float("nan"),
        "G": float(np.mean(nunique_g)) if nunique_g else float("nan"),
        "B": float(np.mean(nunique_b)) if nunique_b else float("nan"),
    }

    if rgb_chunks:
        rgb = np.concatenate(rgb_chunks, axis=0)
        result["rgb_stats_all_channels"] = percentile_stats(rgb)
        result["rgb_stats_per_channel"] = {
            "R": percentile_stats(rgb[:, 0]),
            "G": percentile_stats(rgb[:, 1]),
            "B": percentile_stats(rgb[:, 2]),
        }
        result["n_points_in_stats"] = int(rgb.shape[0])
        result["values_in_0_1"] = bool(rgb.min() >= -1e-6 and rgb.max() <= 1.0 + 1e-3)
    else:
        result["rgb_stats_all_channels"] = None
        result["n_points_in_stats"] = 0

    return result


def audit_tile_root_exclusivity(tile_root: Path) -> dict[str, Any]:
    if not tile_root.exists():
        return {"exists": False, "root": str(tile_root)}
    files = sorted(tile_root.glob("*.npz"))
    prefixes = Counter(plot_prefix(p.name) for p in files)
    non_l1w = [str(p) for p in files if not p.name.startswith("L1W")]
    return {
        "exists": True,
        "root": str(tile_root),
        "n_files": len(files),
        "prefixes": dict(prefixes),
        "only_l1w": len(non_l1w) == 0 and len(files) > 0,
        "non_l1w_count": len(non_l1w),
        "non_l1w_paths_sample": non_l1w[:30],
    }


def align_val_tiles_to_forest(
    val_root: Path,
    forest_npz: Path,
    n_tiles: int = 5,
    seed: int = 0,
    match_tol: float = 1e-3,
) -> dict[str, Any]:
    # Prefer the val-split voxelized forest when caller passes pipeline forest path.
    preferred = REPO / "data/val/forest_voxelized0.1/L1W.npz"
    if preferred.exists() and "pipeline" in str(forest_npz):
        forest_npz = preferred

    out: dict[str, Any] = {
        "forest_npz": str(forest_npz),
        "forest_exists": forest_npz.exists(),
        "n_tiles_requested": n_tiles,
        "match_tol_m": match_tol,
    }
    if not val_root.exists() or not forest_npz.exists():
        return out

    from sklearn.neighbors import NearestNeighbors

    files = sorted(val_root.glob("*.npz"))
    rng = random.Random(seed)
    chosen = rng.sample(files, min(n_tiles, len(files)))
    out["tiles"] = [str(p) for p in chosen]

    forest = np.load(forest_npz)
    f_pts = forest["points"].astype(np.float64)
    if "colors" not in forest.files:
        out["error"] = "forest missing colors"
        return out
    f_col = forest["colors"].astype(np.float32)

    tile_reports = []
    for tp in chosen:
        d = np.load(tp)
        pts = d["points"].astype(np.float64)
        cols = d["colors"].astype(np.float32)
        center = np.asarray(d["center"], dtype=np.float64)
        if center.ndim > 1:
            center = center.reshape(-1)[:3]
        world = pts + center.reshape(1, 3)

        # Subsample query points for speed
        n = len(world)
        q_idx = np.linspace(0, n - 1, min(n, 20_000), dtype=np.int64)
        q = world[q_idx]
        q_col = cols[q_idx]

        # Restrict forest to tile AABB + margin
        margin = 2.0
        mins = q.min(axis=0) - margin
        maxs = q.max(axis=0) + margin
        mask = np.all((f_pts >= mins) & (f_pts <= maxs), axis=1)
        f_sub = f_pts[mask]
        c_sub = f_col[mask]
        if len(f_sub) == 0:
            tile_reports.append(
                {
                    "tile": str(tp),
                    "n_points": int(n),
                    "n_queried": int(len(q)),
                    "match_rate": 0.0,
                    "n_matched": 0,
                    "rgb_absdiff_mean": float("nan"),
                    "rgb_absdiff_p95": float("nan"),
                    "center": center.tolist(),
                    "forest_subset_n": 0,
                }
            )
            continue

        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(f_sub)
        dist, ind = nn.kneighbors(q, return_distance=True)
        dist = dist.reshape(-1)
        ind = ind.reshape(-1)
        matched_mask = dist <= match_tol
        matched = int(matched_mask.sum())
        diffs = np.abs(q_col[matched_mask] - c_sub[ind[matched_mask]]).mean(axis=1) if matched else np.array([])
        tile_reports.append(
            {
                "tile": str(tp),
                "n_points": int(n),
                "n_queried": int(len(q)),
                "match_rate": float(matched / max(len(q), 1)),
                "n_matched": matched,
                "mean_nn_dist": float(dist.mean()),
                "rgb_absdiff_mean": float(np.mean(diffs)) if diffs.size else float("nan"),
                "rgb_absdiff_p95": float(np.percentile(diffs, 95)) if diffs.size else float("nan"),
                "center": center.tolist(),
                "forest_subset_n": int(len(f_sub)),
            }
        )
    out["tile_alignment"] = tile_reports
    out["mean_match_rate"] = float(np.mean([t["match_rate"] for t in tile_reports])) if tile_reports else float("nan")
    return out


def code_path_audit() -> dict[str, Any]:
    """Static findings from reading current sources (post git pull)."""
    model_yaml = (REPO / "configs/_modular/model.yaml").read_text(encoding="utf-8")
    pipeline_yaml = (REPO / "configs/pipeline/pipeline.yaml").read_text(encoding="utf-8")
    train_yaml = (REPO / "configs/training/train.yaml").read_text(encoding="utf-8")
    train_log = REPO / "work_dirs/train_20260627/20260627_145122.log"
    log_snip = ""
    if train_log.exists():
        text = train_log.read_text(encoding="utf-8", errors="ignore").splitlines()
        log_snip = "\n".join([ln for ln in text[:30] if "missing" in ln or "pretrain" in ln or "rgb_" in ln or "Munch" in ln][:10])

    return {
        "use_rgb_gate_in_model_yaml": "use_rgb_gate: True" in model_yaml or "use_rgb_gate: true" in model_yaml,
        "pipeline_pretrain_line": next((ln.strip() for ln in pipeline_yaml.splitlines() if ln.strip().startswith("pretrain:")), None),
        "pipeline_inherits_model_yaml": "configs/_modular/model.yaml" in pipeline_yaml,
        "train_pretrain_line": next((ln.strip() for ln in train_yaml.splitlines() if ln.strip().startswith("pretrain:")), None),
        "dataset_missing_colors_fills_zeros": True,
        "sample_generator_missing_colors_fills_zeros": True,
        "pipeline_missing_rgb_channels_fills_zeros": True,
        "fusion_formula": "fused = F_geo + F_geo * sigmoid(gate)  # amplify-only residual; cannot suppress below F_geo",
        "gate_init": "rgb_gate.weight=0, rgb_gate.bias=-6 => sigmoid≈0.00247 at start",
        "colors_in_collate_batch_key": "colors",
        "v2p_alignment_comment": "backbone_feats = backbone_output.features[v2p_map]; colors are point-order; fusion after v2p_map",
        "fused_feeds_both_heads": True,
        "optimizer_includes_all_requires_grad": "build_optimizer(filter requires_grad=True) includes rgb_encoder/rgb_gate when use_rgb_gate=True",
        "train_log_pretrain_missing_rgb_keys": log_snip,
        "notes": [
            "No separate use_rgb_gate flag existed historically; after pull, model.yaml defaults use_rgb_gate=True.",
            "configs/pipeline/pipeline.yaml still points pretrain to data/model_weights/epoch_781.pth (not necessarily epoch_720).",
            "HAIS pretrain load logged missing rgb_encoder/rgb_gate keys (random/init values used at train start).",
        ],
    }


def checkpoint_and_gate_mlp_audit(ckpt: Path, color_samples: np.ndarray) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    out: dict[str, Any] = {"checkpoint": str(ckpt), "exists": ckpt.exists()}
    if not ckpt.exists():
        out["error"] = "checkpoint missing"
        return out

    state = torch.load(str(ckpt), map_location="cpu")
    out["top_keys"] = list(state.keys())
    out["epoch"] = int(state.get("epoch", -1))
    net = state["net"] if "net" in state else state
    keys = list(net.keys())
    rgb_enc = sorted(k for k in keys if k.startswith("rgb_encoder."))
    rgb_gate = sorted(k for k in keys if k.startswith("rgb_gate."))
    out["rgb_encoder_keys"] = rgb_enc
    out["rgb_gate_keys"] = rgb_gate
    expected_enc = [
        "rgb_encoder.0.weight",
        "rgb_encoder.0.bias",
        "rgb_encoder.2.weight",
        "rgb_encoder.2.bias",
    ]
    expected_gate = ["rgb_gate.weight", "rgb_gate.bias"]
    out["has_all_rgb_encoder"] = all(k in net for k in expected_enc)
    out["has_all_rgb_gate"] = all(k in net for k in expected_gate)

    if out["has_all_rgb_gate"]:
        w = net["rgb_gate.weight"].detach().float().cpu().numpy()
        b = net["rgb_gate.bias"].detach().float().cpu().numpy()
        out["rgb_gate_weight"] = {
            "shape": list(w.shape),
            "l1": float(np.sum(np.abs(w))),
            "l2": float(np.sqrt(np.sum(w * w))),
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "max_abs": float(np.max(np.abs(w))),
            "near_zero_init": bool(float(np.max(np.abs(w))) < 1e-6),
        }
        out["rgb_gate_bias"] = {
            "shape": list(b.shape),
            "min": float(np.min(b)),
            "mean": float(np.mean(b)),
            "max": float(np.max(b)),
            "near_neg6_init": bool(float(np.max(np.abs(b + 6.0))) < 1e-3),
        }
    else:
        out["rgb_gate_weight"] = None
        out["rgb_gate_bias"] = None

    # Rebuild RGB MLP only and run gate distribution.
    if out["has_all_rgb_encoder"] and out["has_all_rgb_gate"] and color_samples.size:
        enc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        gate = nn.Linear(16, 32)
        with torch.no_grad():
            enc[0].weight.copy_(net["rgb_encoder.0.weight"])
            enc[0].bias.copy_(net["rgb_encoder.0.bias"])
            enc[2].weight.copy_(net["rgb_encoder.2.weight"])
            enc[2].bias.copy_(net["rgb_encoder.2.bias"])
            gate.weight.copy_(net["rgb_gate.weight"])
            gate.bias.copy_(net["rgb_gate.bias"])
            x = torch.from_numpy(color_samples.astype(np.float32))
            # cap
            if x.shape[0] > 100_000:
                idx = torch.linspace(0, x.shape[0] - 1, 100_000).long()
                x = x[idx]
            g = torch.sigmoid(gate(enc(x))).numpy()
        out["gate_output"] = {
            "n_points": int(g.shape[0]),
            "channels": int(g.shape[1]),
            "overall": percentile_stats(g),
            "per_channel_mean_stats": percentile_stats(g.mean(axis=1)),
            "frac_gate_lt_0_01": float(np.mean(g < 0.01)),
            "channel_means": g.mean(axis=0).tolist(),
            "channel_stds": g.std(axis=0).tolist(),
            "ref_sigmoid_neg6": SIGMOID_NEG6,
            "mean_near_init": bool(abs(float(g.mean()) - SIGMOID_NEG6) < 5e-4),
            "std_near_zero": bool(float(g.std()) < 1e-4),
        }
        out["n_color_points_used"] = int(g.shape[0])
    else:
        out["gate_output"] = None

    # Compare vs random init signature
    out["dormant_weight_bias_heuristic"] = bool(
        out.get("rgb_gate_weight", {}) and out["rgb_gate_weight"].get("near_zero_init")
        and out.get("rgb_gate_bias", {}) and out["rgb_gate_bias"].get("near_neg6_init")
    )
    return out


class _FeatBag:
    def __init__(self, features):
        self.features = features


def _pick_val_tile(val_root: Path, seed: int, max_points: int = 250_000) -> Path:
    """Pick a val tile without scanning every NPZ (that can take many minutes on disk).

    Prefer file-size heuristic, then verify point count on a shortlist.
    """
    files = sorted(val_root.glob("*.npz"), key=lambda p: p.stat().st_size)
    if not files:
        raise FileNotFoundError(f"no tiles under {val_root}")
    # Shortlist ~20 mid-small files by on-disk size, then load only those.
    n_files = len(files)
    lo = max(0, n_files // 10)
    hi = min(n_files, lo + 40)
    shortlist = files[lo:hi] or files[:20]
    rng = random.Random(seed)
    rng.shuffle(shortlist)
    scored = []
    for p in shortlist[:20]:
        try:
            with np.load(p) as d:
                n = int(d["points"].shape[0])
            scored.append((abs(n - max_points // 2), n, p))
        except Exception:
            continue
    if not scored:
        return files[seed % len(files)]
    scored.sort()
    candidates = [t for t in scored if t[1] <= max_points] or scored[:5]
    return candidates[seed % len(candidates)][2]


def forward_causal_audit_gpu(ckpt: Path, val_root: Path, seed: int = 0) -> dict[str, Any]:
    """Full sparse-backbone causal tests on CUDA."""
    import torch
    import torch.nn.functional as F
    from tree_learn.model import TreeLearn
    from tree_learn.util import get_config

    assert torch.cuda.is_available()
    out: dict[str, Any] = {
        "mode": "gpu_full_model",
        "reason": "Full TreeLearn.forward on CUDA with real val batch; RGB counterfactuals swap only colors.",
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    if not ckpt.exists():
        out["error"] = "checkpoint missing"
        return out

    cfg = get_config(str(REPO / "configs/training/train.yaml"))
    model = TreeLearn(**cfg.model).cuda()
    state = torch.load(str(ckpt), map_location="cpu")
    net = state["net"]
    missing, unexpected = model.load_state_dict(net, strict=False)
    out["load_missing_keys"] = list(missing)
    out["load_unexpected_keys"] = list(unexpected)
    out["use_rgb_gate"] = bool(model.use_rgb_gate)
    out["checkpoint_epoch"] = int(state.get("epoch", -1))

    tile = _pick_val_tile(val_root, seed=seed, max_points=80_000)
    data = np.load(tile)
    xyz = data["points"].astype(np.float32)
    colors_np = data["colors"].astype(np.float32)
    feats = data["feat"].astype(np.float32) if "feat" in data.files else np.ones((len(xyz), 1), np.float32)
    if feats.ndim == 1:
        feats = feats[:, None]
    inst = data["instance_label"].astype(np.int64)
    n_full = len(xyz)
    # hard cap for speed / VRAM (causal RGB test does not need full tile)
    n = min(n_full, 50_000)
    if n < n_full:
        idx = np.linspace(0, n_full - 1, n, dtype=np.int64)
        xyz, colors_np, feats, inst = xyz[idx], colors_np[idx], feats[idx], inst[idx]

    sem = np.zeros(n, dtype=np.int64)
    sem[inst == 0] = 1
    # offset labels dummy (zeros) — enough for loss path if needed
    off_lbl = np.zeros((n, 3), dtype=np.float32)
    masks = np.ones(n, dtype=bool)

    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_batch(colors_arr: np.ndarray) -> dict[str, Any]:
        return {
            "coords": torch.from_numpy(xyz),
            "input_feats": torch.from_numpy(feats),
            "colors": torch.from_numpy(colors_arr.astype(np.float32)),
            "batch_ids": torch.zeros(n, dtype=torch.long),
            "batch_size": 1,
            "semantic_labels": torch.from_numpy(sem),
            "instance_labels": torch.from_numpy(inst),
            "offset_labels": torch.from_numpy(off_lbl),
            "masks_off": torch.from_numpy(masks),
            "masks_sem": torch.from_numpy(masks),
            "masks_inner": torch.from_numpy(masks),
            "centers": torch.zeros((n, 3), dtype=torch.float32),
        }

    colors = colors_np
    mean_rgb = np.repeat(colors.mean(axis=0, keepdims=True), n, axis=0)
    zero_rgb = np.zeros_like(colors)
    perm = np.random.permutation(n)
    shuffle_rgb = colors[perm]

    def run(colors_arr: np.ndarray, grad: bool = False) -> dict[str, Any]:
        batch = make_batch(colors_arr)
        if grad:
            model.train()
            model.zero_grad(set_to_none=True)
            output = model(batch, return_loss=False)
            # Manual scalar loss keeps rgb_gate_mask in the same graph without unpacking get_loss tuple.
            logits = output["semantic_prediction_logits"]
            offs = output["offset_predictions"]
            sem_t = batch["semantic_labels"].to(logits.device)
            loss = F.cross_entropy(logits, sem_t) + offs.pow(2).mean()
            loss.backward()
            named = {
                k: float(v.grad.norm().item()) if v.grad is not None else 0.0
                for k, v in model.named_parameters()
            }

            def pick(prefix: str) -> float:
                for k, v in named.items():
                    if k.startswith(prefix) and k.endswith("weight"):
                        return v
                return 0.0

            grads = {
                "rgb_encoder.0.weight": named.get("rgb_encoder.0.weight", 0.0),
                "rgb_encoder.2.weight": named.get("rgb_encoder.2.weight", 0.0),
                "rgb_gate.weight": named.get("rgb_gate.weight", 0.0),
                "rgb_gate.bias": named.get("rgb_gate.bias", 0.0),
                "semantic_linear_first_weight": pick("semantic_linear"),
                "offset_linear_first_weight": pick("offset_linear"),
            }
            gate = output["rgb_gate_mask"].detach()
            model.eval()
            return {
                "loss": float(loss.detach().item()),
                "grads": grads,
                "rgb_to_sem_grad_ratio": float(grads["rgb_gate.weight"] / (grads["semantic_linear_first_weight"] + 1e-12)),
                "rgb_to_off_grad_ratio": float(grads["rgb_gate.weight"] / (grads["offset_linear_first_weight"] + 1e-12)),
                "gate_mean": float(gate.mean().item()),
                "gate_std": float(gate.std().item()),
            }

        model.eval()
        with torch.no_grad():
            output = model(batch, return_loss=False)
            gate = output["rgb_gate_mask"]
            logits = output["semantic_prediction_logits"]
            offs = output["offset_predictions"]
            probs = torch.softmax(logits, dim=-1)[:, 0]
            return {
                "gate_mean": float(gate.mean().item()),
                "gate_std": float(gate.std().item()),
                "gate_min": float(gate.min().item()),
                "gate_max": float(gate.max().item()),
                "logits": logits.detach().cpu(),
                "probs": probs.detach().cpu(),
                "offsets": offs.detach().cpu(),
            }

    grad_stats = run(colors, grad=True)
    torch.cuda.empty_cache()
    base = run(colors, grad=False)
    zero = run(zero_rgb, grad=False)
    shuf = run(shuffle_rgb, grad=False)
    meanr = run(mean_rgb, grad=False)

    def compare(tag: str, other: dict[str, Any]) -> dict[str, Any]:
        logits_mae = float((base["logits"] - other["logits"]).abs().mean())
        logits_max = float((base["logits"] - other["logits"]).abs().max())
        prob_mae = float((base["probs"] - other["probs"]).abs().mean())
        off_l2 = float((base["offsets"] - other["offsets"]).norm(dim=-1).mean())
        pred_base = base["logits"].argmax(dim=-1)
        pred_oth = other["logits"].argmax(dim=-1)
        cls_change = float((pred_base != pred_oth).float().mean())
        return {
            "tag": tag,
            "gate_mean": other["gate_mean"],
            "gate_std": other["gate_std"],
            "semantic_logits_mae": logits_mae,
            "semantic_logits_max_abs": logits_max,
            "tree_prob_mae": prob_mae,
            "offset_mean_l2": off_l2,
            "semantic_class_change_frac": cls_change,
        }

    out["tile"] = str(tile)
    out["n_points"] = int(n)
    out["n_points_full_tile"] = int(n_full)
    out["gradient"] = grad_stats
    out["counterfactual"] = {
        "original": {"gate_mean": base["gate_mean"], "gate_std": base["gate_std"]},
        "vs_zero_rgb": compare("zero", zero),
        "vs_shuffle_rgb": compare("shuffle", shuf),
        "vs_mean_rgb": compare("mean", meanr),
    }
    cf = out["counterfactual"]
    out["activity_signals"] = {
        "gate_mean": base["gate_mean"],
        "gate_std": base["gate_std"],
        "gate_near_init_mean": bool(abs(base["gate_mean"] - SIGMOID_NEG6) < 5e-4),
        "shuffle_logits_mae": cf["vs_shuffle_rgb"]["semantic_logits_mae"],
        "shuffle_changes_predictions": bool(
            cf["vs_shuffle_rgb"]["semantic_class_change_frac"] > 1e-4
            or cf["vs_shuffle_rgb"]["semantic_logits_mae"] > 1e-5
            or cf["vs_shuffle_rgb"]["offset_mean_l2"] > 1e-5
        ),
        "rgb_gate_grad_nonzero": bool(grad_stats["grads"].get("rgb_gate.weight", 0.0) > 0.0),
    }
    return out


def forward_causal_audit(ckpt: Path, val_root: Path, seed: int = 0) -> dict[str, Any]:
    """Prefer full GPU model; fall back to CPU forward_head-only."""
    import torch

    if torch.cuda.is_available():
        try:
            return forward_causal_audit_gpu(ckpt, val_root, seed=seed)
        except Exception as exc:  # noqa: BLE001
            cpu_out = forward_causal_audit_cpu(ckpt, val_root, seed=seed)
            cpu_out["gpu_attempt_error"] = str(exc)
            return cpu_out
    return forward_causal_audit_cpu(ckpt, val_root, seed=seed)


def forward_causal_audit_cpu(ckpt: Path, val_root: Path, seed: int = 0) -> dict[str, Any]:
    """CPU forward_head causal tests (no full sparse backbone)."""
    import torch
    import torch.nn.functional as F
    from tree_learn.model import TreeLearn
    from tree_learn.util import get_config

    out: dict[str, Any] = {
        "mode": "cpu_forward_head_only",
        "reason": "CUDA unavailable or GPU full-model failed; tests RGB gate via forward_head with fixed geo feats.",
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not ckpt.exists():
        out["error"] = "checkpoint missing"
        return out

    cfg = get_config(str(REPO / "configs/training/train.yaml"))
    model = TreeLearn(**cfg.model)
    state = torch.load(str(ckpt), map_location="cpu")
    net = state["net"]
    missing, unexpected = model.load_state_dict(net, strict=False)
    out["load_missing_keys"] = list(missing)
    out["load_unexpected_keys"] = list(unexpected)
    out["use_rgb_gate"] = bool(model.use_rgb_gate)
    out["checkpoint_epoch"] = int(state.get("epoch", -1))
    model.eval()

    files = sorted(val_root.glob("*.npz"))
    rng = random.Random(seed)
    tile = Path(rng.choice(files))
    data = np.load(tile)
    colors_np = data["colors"].astype(np.float32)
    # subsample for memory
    n_full = len(colors_np)
    n = min(80_000, n_full)
    idx = np.linspace(0, n_full - 1, n, dtype=np.int64)
    colors_np = colors_np[idx]
    # labels for a cheap supervised loss
    inst = data["instance_label"][idx]
    sem = np.zeros(n, dtype=np.int64)
    sem[inst == 0] = 1  # NON_TREE -> 1
    # tree class 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    C = int(cfg.model.channels)
    # Fixed geometric features (point-order identity v2p)
    backbone_feats = torch.randn(n, C)
    v2p = torch.arange(n, dtype=torch.long)
    bag = _FeatBag(backbone_feats)

    colors = torch.from_numpy(colors_np)
    mean_rgb = colors.mean(dim=0, keepdim=True).expand_as(colors)
    zero_rgb = torch.zeros_like(colors)
    perm = torch.randperm(n)
    shuffle_rgb = colors[perm]

    def run(colors_t: torch.Tensor, grad: bool = False) -> dict[str, Any]:
        if grad:
            model.train()
            model.zero_grad(set_to_none=True)
            out_h = model.forward_head(bag, v2p, colors_t)
            logits = out_h["semantic_prediction_logits"]
            offs = out_h["offset_predictions"]
            sem_t = torch.from_numpy(sem)
            loss = F.cross_entropy(logits, sem_t) + offs.pow(2).mean()
            loss.backward()
            named = {
                k: float(v.grad.norm().item()) if v.grad is not None else 0.0
                for k, v in model.named_parameters()
            }
            def pick(prefix: str) -> float:
                for k, v in named.items():
                    if k.startswith(prefix) and k.endswith("weight"):
                        return v
                return 0.0
            grads = {
                "rgb_encoder.0.weight": named.get("rgb_encoder.0.weight", 0.0),
                "rgb_encoder.2.weight": named.get("rgb_encoder.2.weight", 0.0),
                "rgb_gate.weight": named.get("rgb_gate.weight", 0.0),
                "rgb_gate.bias": named.get("rgb_gate.bias", 0.0),
                "semantic_linear_first_weight": pick("semantic_linear"),
                "offset_linear_first_weight": pick("offset_linear"),
            }
            gate = out_h["rgb_gate_mask"].detach()
            model.eval()
            return {
                "loss": float(loss.item()),
                "grads": grads,
                "rgb_to_sem_grad_ratio": float(grads["rgb_gate.weight"] / (grads["semantic_linear_first_weight"] + 1e-12)),
                "rgb_to_off_grad_ratio": float(grads["rgb_gate.weight"] / (grads["offset_linear_first_weight"] + 1e-12)),
                "gate_mean": float(gate.mean()),
                "gate_std": float(gate.std()),
            }
        with torch.no_grad():
            out_h = model.forward_head(bag, v2p, colors_t)
            gate = out_h["rgb_gate_mask"]
            logits = out_h["semantic_prediction_logits"]
            offs = out_h["offset_predictions"]
            probs = torch.softmax(logits, dim=-1)[:, 0]
            return {
                "gate_mean": float(gate.mean()),
                "gate_std": float(gate.std()),
                "gate_min": float(gate.min()),
                "gate_max": float(gate.max()),
                "logits": logits,
                "probs": probs,
                "offsets": offs,
            }

    grad_stats = run(colors, grad=True)
    base = run(colors, grad=False)
    zero = run(zero_rgb, grad=False)
    shuf = run(shuffle_rgb, grad=False)
    meanr = run(mean_rgb, grad=False)

    def compare(tag: str, other: dict[str, Any]) -> dict[str, Any]:
        logits_mae = float((base["logits"] - other["logits"]).abs().mean())
        logits_max = float((base["logits"] - other["logits"]).abs().max())
        prob_mae = float((base["probs"] - other["probs"]).abs().mean())
        off_l2 = float((base["offsets"] - other["offsets"]).norm(dim=-1).mean())
        pred_base = base["logits"].argmax(dim=-1)
        pred_oth = other["logits"].argmax(dim=-1)
        cls_change = float((pred_base != pred_oth).float().mean())
        return {
            "tag": tag,
            "gate_mean": other["gate_mean"],
            "gate_std": other["gate_std"],
            "semantic_logits_mae": logits_mae,
            "semantic_logits_max_abs": logits_max,
            "tree_prob_mae": prob_mae,
            "offset_mean_l2": off_l2,
            "semantic_class_change_frac": cls_change,
        }

    out["tile"] = str(tile)
    out["n_points"] = int(n)
    out["n_points_full_tile"] = int(n_full)
    out["gradient"] = grad_stats
    out["counterfactual"] = {
        "original": {"gate_mean": base["gate_mean"], "gate_std": base["gate_std"]},
        "vs_zero_rgb": compare("zero", zero),
        "vs_shuffle_rgb": compare("shuffle", shuf),
        "vs_mean_rgb": compare("mean", meanr),
    }
    # activity heuristics
    cf = out["counterfactual"]
    out["activity_signals"] = {
        "gate_mean": base["gate_mean"],
        "gate_std": base["gate_std"],
        "gate_near_init_mean": bool(abs(base["gate_mean"] - SIGMOID_NEG6) < 5e-4),
        "shuffle_logits_mae": cf["vs_shuffle_rgb"]["semantic_logits_mae"],
        "shuffle_changes_predictions": bool(cf["vs_shuffle_rgb"]["semantic_class_change_frac"] > 1e-4
                                            or cf["vs_shuffle_rgb"]["semantic_logits_mae"] > 1e-5
                                            or cf["vs_shuffle_rgb"]["offset_mean_l2"] > 1e-5),
        "rgb_gate_grad_nonzero": bool(grad_stats["grads"].get("rgb_gate.weight", 0.0) > 0.0),
    }
    return out


def decide_verdict(data_audit: dict, ckpt_audit: dict, fwd: dict | None) -> str:
    # INVALID_DATA
    issues = 0
    checked = 0
    for key in ("train", "val", "tiles"):
        block = data_audit.get(key) or {}
        if not block.get("exists"):
            continue
        checked += int(block.get("n_checked") or 0)
        issues += int(block.get("missing_colors", {}).get("count") or 0)
        issues += int(block.get("all_zero_rgb", {}).get("count") or 0)
        issues += int(block.get("bad_shape", {}).get("count") or 0)
    if checked > 0 and issues / max(checked, 1) > 0.5:
        return "INVALID_DATA"

    if not ckpt_audit.get("exists") or not ckpt_audit.get("has_all_rgb_gate"):
        return "INCONCLUSIVE"

    gate = ckpt_audit.get("gate_output") or {}
    dormant_w = bool(ckpt_audit.get("dormant_weight_bias_heuristic"))
    dormant_out = bool(gate.get("mean_near_init") and gate.get("std_near_zero"))

    if fwd is None or fwd.get("error"):
        if dormant_w or dormant_out:
            return "DORMANT"
        return "INCONCLUSIVE"

    sig = fwd.get("activity_signals") or {}
    if sig.get("shuffle_changes_predictions") and sig.get("rgb_gate_grad_nonzero") and not sig.get("gate_near_init_mean"):
        return "ACTIVE"
    if (dormant_w or dormant_out or sig.get("gate_near_init_mean")) and not sig.get("shuffle_changes_predictions"):
        return "DORMANT"
    if sig.get("shuffle_changes_predictions") or (sig.get("rgb_gate_grad_nonzero") and gate and float(gate.get("overall", {}).get("std", 0)) > 1e-4):
        return "ACTIVE"
    return "INCONCLUSIVE"


def write_report(
    path: Path,
    verdict: str,
    code: dict,
    data: dict,
    ckpt: dict,
    fwd: dict | None,
) -> None:
    gate = ckpt.get("gate_output") or {}
    lines = []
    lines.append("# RGB Gate Audit Report")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Checkpoint: `{ckpt.get('checkpoint')}`")
    lines.append("")
    lines.append("## 1. 最终判断")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("## 2. 代码链路检查")
    lines.append("")
    lines.append(f"- `configs/_modular/model.yaml` 中 `use_rgb_gate=True`: **{code.get('use_rgb_gate_in_model_yaml')}**")
    lines.append(f"- pipeline 配置: `{code.get('pipeline_pretrain_line')}`；继承 model.yaml: **{code.get('pipeline_inherits_model_yaml')}**")
    lines.append(f"- Dataset 缺 `colors` 时静默填零: **{code.get('dataset_missing_colors_fills_zeros')}**")
    lines.append(f"- SampleGenerator / pipeline 缺 RGB 通道时填零: **{code.get('sample_generator_missing_colors_fills_zeros')}** / **{code.get('pipeline_missing_rgb_channels_fills_zeros')}**")
    lines.append(f"- 融合公式: `{code.get('fusion_formula')}`（只能增强，不能抑制到低于 F_geo）")
    lines.append(f"- 初始化: `{code.get('gate_init')}`")
    lines.append(f"- colors→`batch['colors']`→`forward_head`；`v2p_map` 后与点序对齐再融合: **是**")
    lines.append(f"- fused_feats 同时进入 semantic_linear 与 offset_linear: **{code.get('fused_feeds_both_heads')}**")
    lines.append(f"- optimizer 包含全部 requires_grad 参数（含 rgb_*）: **是**")
    lines.append("")
    lines.append("训练日志（HAIS pretrain）摘录：")
    lines.append("```")
    lines.append(code.get("train_log_pretrain_missing_rgb_keys") or "(none)")
    lines.append("```")
    lines.append("")
    lines.append("## 3. RGB 数据质量")
    lines.append("")
    for split in ("train", "val", "tiles", "tile_exclusivity", "val_forest_alignment"):
        block = data.get(split)
        if not block:
            continue
        lines.append(f"### {split}")
        lines.append("```json")
        # compact
        compact = {k: block[k] for k in block if k not in ("missing_colors", "all_zero_rgb", "low_variance_rgb") or True}
        # shorten path lists in report
        b2 = dict(block)
        for pk in ("missing_colors", "all_zero_rgb", "low_variance_rgb", "nan_inf"):
            if pk in b2 and isinstance(b2[pk], dict) and "paths" in b2[pk]:
                b2[pk] = {"count": b2[pk]["count"], "paths_sample_n": len(b2[pk].get("paths", []))}
        lines.append(json.dumps(b2, indent=2, ensure_ascii=False, default=_json_default)[:8000])
        lines.append("```")
        lines.append("")

    lines.append("## 4. checkpoint 门控权重")
    lines.append("")
    lines.append(f"- epoch: **{ckpt.get('epoch')}**")
    lines.append(f"- 含全部 rgb_encoder 键: **{ckpt.get('has_all_rgb_encoder')}**")
    lines.append(f"- 含全部 rgb_gate 键: **{ckpt.get('has_all_rgb_gate')}**")
    lines.append(f"- weight 统计: `{json.dumps(ckpt.get('rgb_gate_weight'), ensure_ascii=False)}`")
    lines.append(f"- bias 统计: `{json.dumps(ckpt.get('rgb_gate_bias'), ensure_ascii=False)}`")
    lines.append(f"- 仍接近 weight=0 / bias=-6: **{ckpt.get('dormant_weight_bias_heuristic')}**")
    lines.append("")
    lines.append("## 5. 门控输出分布")
    lines.append("")
    lines.append(f"- 参考 sigmoid(-6)=**{SIGMOID_NEG6:.8f}**")
    lines.append(f"- gate 输出: `{json.dumps(gate, ensure_ascii=False, default=_json_default)[:4000]}`")
    lines.append("")
    lines.append("## 6. 梯度检查")
    lines.append("")
    if fwd and "gradient" in fwd:
        lines.append(f"- 模式: `{fwd.get('mode')}`")
        lines.append(f"- tile: `{fwd.get('tile')}` n={fwd.get('n_points')}")
        lines.append(f"- grads: `{json.dumps(fwd['gradient'].get('grads'), ensure_ascii=False)}`")
        lines.append(
            f"- rgb_gate.weight / semantic_first = **{fwd['gradient'].get('rgb_to_sem_grad_ratio')}**; "
            f"/ offset_first = **{fwd['gradient'].get('rgb_to_off_grad_ratio')}**"
        )
    else:
        lines.append("- 未能完成梯度检查。")
    lines.append("")
    lines.append("## 7. 原始/全零/打乱 RGB 反事实结果")
    lines.append("")
    if fwd and "counterfactual" in fwd:
        lines.append("```json")
        lines.append(json.dumps(fwd["counterfactual"], indent=2, ensure_ascii=False, default=_json_default))
        lines.append("```")
        lines.append(f"- activity_signals: `{json.dumps(fwd.get('activity_signals'), ensure_ascii=False)}`")
    else:
        lines.append("- 未能完成反事实测试。")
    lines.append("")
    lines.append("## 8. 发现的问题（按严重程度）")
    lines.append("")
    problems = []
    if code.get("fusion_formula"):
        problems.append("P1 融合式 `F+F*g` 只能增强，不能抑制几何特征；门控语义是 residual boost 而非真正的 soft mask。")
    if ckpt.get("dormant_weight_bias_heuristic") or (gate.get("mean_near_init") and gate.get("std_near_zero")):
        problems.append("P1 gate 权重/输出仍接近初始化 sigmoid(-6)，颜色分支可能未学到有效调制。")
    if data.get("train", {}).get("all_zero_rgb", {}).get("count", 0):
        problems.append(f"P2 训练 NPZ 存在全零 RGB: {data['train']['all_zero_rgb']['count']} / checked")
    if data.get("train", {}).get("missing_colors", {}).get("count", 0):
        problems.append(f"P1 训练 NPZ 缺 colors: {data['train']['missing_colors']['count']}")
    problems.append("P2 Dataset/SampleGenerator/pipeline 在缺 colors 时静默填零，易掩盖数据问题。")
    problems.append("P2 weight=0、bias=-6 使初始梯度经 sigmoid'(-6)≈0.00247 缩放，收敛偏慢，易长期休眠。")
    problems.append("P3 默认 pipeline.yaml 的 pretrain 指向 epoch_781.pth，与审计用的 epoch_720 不一定一致。")
    if fwd and fwd.get("mode") == "cpu_forward_head_only":
        problems.append("P3 本机无 NVIDIA 驱动，反事实/梯度测试仅覆盖 forward_head（固定几何特征），非完整 sparse backbone。")
    for i, p in enumerate(problems, 1):
        lines.append(f"{i}. {p}")
    lines.append("")
    lines.append("## 9. 是否需要修改门控初始化、融合公式或数据缓存")
    lines.append("")
    if verdict in ("DORMANT", "INVALID_DATA"):
        lines.append("- **需要**：考虑改融合为 `F*(1+g)` 以外的可抑制形式（如 `F*(epsilon+(1-epsilon)*g)` 或 `F + MLP(rgb)`），并放宽初始化（如 bias=-2~-3）。")
        lines.append("- **需要**：审计并修复静默填零；确保缓存 Tile/crops 含对齐 RGB。")
    elif verdict == "ACTIVE":
        lines.append("- 数据与门控已有响应；仍建议评估融合公式是否需要可抑制形式，以及初始化是否过保守。")
    else:
        lines.append("- 证据不足；优先补齐 GPU 全链路前向后再决定是否改公式/初始化。")
    lines.append("")
    lines.append("## 10. 下一步最小实验方案")
    lines.append("")
    lines.append("1. 在有 GPU 的机器上对同一 val batch 跑完整 `model(batch)` 反事实（原/零/打乱 RGB）。")
    lines.append("2. 若判定 DORMANT：将 `rgb_gate.bias` 初始化改为 -2 或 -3，短训 20–50 epoch，对比 gate 分布与 val 指标。")
    lines.append("3. 可选：改融合为 `fused = backbone * (1 + gate)` 保持增强语义但记录 gate 统计；或试 `backbone + rgb_delta`。")
    lines.append("4. 用 `analyze` 脚本确认 `data/tiles/npz` 与训练 crops 无混样地后，再开完整 L1W 推理对比。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_color_samples(paths: list[Path], max_points: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = random.Random(seed)
    chunks = []
    left = max_points
    for p in paths:
        if left <= 0:
            break
        try:
            with np.load(p) as d:
                if "colors" not in d.files:
                    continue
                c = d["colors"].astype(np.float32)
                if c.ndim != 2 or c.shape[1] != 3:
                    continue
                take = min(len(c), left, 20_000)
                if take < len(c):
                    idx = rng.sample(range(len(c)), take)
                    c = c[idx]
                chunks.append(c)
                left -= len(c)
        except Exception:
            continue
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def npz_array_shape(path: Path, key: str = "points") -> tuple[int, ...]:
    """Read array shape from NPZ without loading payload (zip + npy header)."""
    import zipfile

    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(f"{key}.npy") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _, _ = np.lib.format._read_array_header(f, version)  # type: ignore[attr-defined]
    return tuple(int(x) for x in shape)


def select_stratified_l1w_tiles(
    tile_root: Path,
    n_tiles: int = 20,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Deterministically pick L1W tiles stratified by point-count tertiles."""
    files = sorted(
        [p for p in tile_root.glob("L1W_*.npz") if p.is_file()],
        key=lambda p: p.name,
    )
    if not files:
        files = sorted([p for p in tile_root.glob("*.npz") if p.is_file()], key=lambda p: p.name)
    if len(files) < n_tiles:
        raise FileNotFoundError(f"Need >= {n_tiles} tiles under {tile_root}, found {len(files)}")

    indexed: list[tuple[int, Path]] = []
    for p in files:
        try:
            n = int(npz_array_shape(p, "points")[0])
        except Exception:
            continue
        indexed.append((n, p))
    indexed.sort(key=lambda t: (t[0], t[1].name))
    if len(indexed) < n_tiles:
        raise RuntimeError(f"Only indexed {len(indexed)} readable tiles")

    n = len(indexed)
    tertile_cuts = [0, n // 3, (2 * n) // 3, n]
    strata = [
        indexed[tertile_cuts[i] : tertile_cuts[i + 1]]
        for i in range(3)
        if tertile_cuts[i + 1] > tertile_cuts[i]
    ]
    # evenly allocate 20 across strata (e.g. 7/7/6)
    base, rem = divmod(n_tiles, len(strata))
    quotas = [base + (1 if i < rem else 0) for i in range(len(strata))]

    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for si, (stratum, q) in enumerate(zip(strata, quotas)):
        label = ("small", "medium", "large")[si]
        # even stride within sorted stratum, then deterministic offset by seed
        if q >= len(stratum):
            picks = list(stratum)
        else:
            step = len(stratum) / q
            offset = rng.random() * step
            idxs = [min(len(stratum) - 1, int(offset + i * step)) for i in range(q)]
            # dedupe while preserving order
            seen = set()
            picks = []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    picks.append(stratum[i])
            # fill if collisions
            j = 0
            while len(picks) < q and j < len(stratum):
                if j not in seen:
                    seen.add(j)
                    picks.append(stratum[j])
                j += 1
        for n_pts, path in picks:
            chosen.append(
                {
                    "tile": str(path),
                    "name": path.name,
                    "n_points_full": int(n_pts),
                    "stratum": label,
                }
            )

    chosen.sort(key=lambda d: (d["n_points_full"], d["name"]))
    # if over-selected due to stratum fill quirks, trim deterministically
    if len(chosen) > n_tiles:
        chosen = chosen[:n_tiles]
    return chosen


def _summarize_scalar_list(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ["mean", "std", "median", "min", "max", "q25", "q75"]}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def _head_first_weight_grad(named: dict[str, float], prefix: str) -> float:
    for k, v in named.items():
        if k.startswith(prefix) and k.endswith("weight"):
            return float(v)
    return 0.0


def _load_tile_for_audit(tile: Path, max_points: int, seed: int) -> dict[str, Any]:
    data = np.load(tile)
    xyz = data["points"].astype(np.float32)
    colors = data["colors"].astype(np.float32)
    feats = data["feat"].astype(np.float32) if "feat" in data.files else np.ones((len(xyz), 1), np.float32)
    if feats.ndim == 1:
        feats = feats[:, None]
    inst = data["instance_label"].astype(np.int64)
    n_full = len(xyz)
    subsampled = n_full > max_points
    if subsampled:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_full, size=max_points, replace=False))
        xyz, colors, feats, inst = xyz[idx], colors[idx], feats[idx], inst[idx]
    n = len(xyz)
    sem = np.zeros(n, dtype=np.int64)
    sem[inst == 0] = 1  # NON_TREE
    tree_mask = inst != 0
    return {
        "xyz": xyz,
        "colors": colors,
        "feats": feats,
        "inst": inst,
        "sem": sem,
        "tree_mask": tree_mask,
        "n": n,
        "n_full": n_full,
        "subsampled": subsampled,
    }


def _make_batch_from_arrays(arr: dict[str, Any], colors_np: np.ndarray) -> dict[str, Any]:
    import torch

    n = arr["n"]
    return {
        "coords": torch.from_numpy(arr["xyz"]),
        "input_feats": torch.from_numpy(arr["feats"]),
        "colors": torch.from_numpy(colors_np.astype(np.float32)),
        "batch_ids": torch.zeros(n, dtype=torch.long),
        "batch_size": 1,
        "semantic_labels": torch.from_numpy(arr["sem"]),
        "instance_labels": torch.from_numpy(arr["inst"]),
        "offset_labels": torch.zeros((n, 3), dtype=torch.float32),
        "masks_off": torch.ones(n, dtype=torch.bool),
        "masks_sem": torch.ones(n, dtype=torch.bool),
        "masks_inner": torch.ones(n, dtype=torch.bool),
        "centers": torch.zeros((n, 3), dtype=torch.float32),
    }


def _gate_stats(gate_t) -> dict[str, float]:
    g = gate_t.detach().float().reshape(-1).cpu().numpy()
    qs = np.percentile(g, [5, 50, 95, 99])
    return {
        "gate_mean": float(np.mean(g)),
        "gate_std": float(np.std(g)),
        "gate_p05": float(qs[0]),
        "gate_p50": float(qs[1]),
        "gate_p95": float(qs[2]),
        "gate_p99": float(qs[3]),
        "gate_lt_0p01_frac": float(np.mean(g < 0.01)),
    }


def _compare_preds(
    base: dict[str, Any],
    other: dict[str, Any],
    tree_mask: np.ndarray,
) -> dict[str, float]:
    import torch

    logits_mae = float((base["logits"] - other["logits"]).abs().mean())
    prob_mae = float((base["probs"] - other["probs"]).abs().mean())
    off_l2 = float((base["offsets"] - other["offsets"]).norm(dim=-1).mean())
    pred_b = base["logits"].argmax(dim=-1)
    pred_o = other["logits"].argmax(dim=-1)
    flip = pred_b != pred_o
    flip_frac = float(flip.float().mean())
    tree_t = torch.from_numpy(tree_mask.astype(bool))
    nontree_t = ~tree_t
    flip_tree = float(flip[tree_t].float().mean()) if tree_t.any() else float("nan")
    flip_nontree = float(flip[nontree_t].float().mean()) if nontree_t.any() else float("nan")
    return {
        "semantic_logits_mae": logits_mae,
        "tree_prob_mae": prob_mae,
        "offset_mean_l2": off_l2,
        "semantic_class_change_frac": flip_frac,
        "tree_class_change_frac": flip_tree,
        "nontree_class_change_frac": flip_nontree,
    }


def run_one_tile_causal(model, tile_meta: dict[str, Any], max_points: int, seed: int) -> dict[str, Any]:
    """Eval-mode full-backbone RGB counterfactuals for one tile (no grads)."""
    import torch

    tile = Path(tile_meta["tile"])
    arr = _load_tile_for_audit(tile, max_points=max_points, seed=seed)
    colors = arr["colors"]
    rng = np.random.default_rng(seed)
    zero_rgb = np.zeros_like(colors)
    mean_rgb = np.repeat(colors.mean(axis=0, keepdims=True), arr["n"], axis=0)
    shuffle_rgb = colors[rng.permutation(arr["n"])]

    def forward_eval(colors_np: np.ndarray) -> dict[str, Any]:
        batch = _make_batch_from_arrays(arr, colors_np)
        model.eval()
        with torch.no_grad():
            out = model(batch, return_loss=False)
            logits = out["semantic_prediction_logits"]
            offs = out["offset_predictions"]
            gate = out["rgb_gate_mask"]
            probs = torch.softmax(logits, dim=-1)[:, 0]
            return {
                "logits": logits.detach().cpu(),
                "probs": probs.detach().cpu(),
                "offsets": offs.detach().cpu(),
                "gate": gate.detach().cpu(),
            }

    base = forward_eval(colors)
    shuf = forward_eval(shuffle_rgb)
    zero = forward_eval(zero_rgb)
    meanr = forward_eval(mean_rgb)

    row = {
        "name": tile_meta["name"],
        "tile": str(tile),
        "stratum": tile_meta["stratum"],
        "n_points_full": int(arr["n_full"]),
        "n_points_used": int(arr["n"]),
        "subsampled": bool(arr["subsampled"]),
        "n_tree_gt": int(arr["tree_mask"].sum()),
        "n_nontree_gt": int((~arr["tree_mask"]).sum()),
    }
    row.update(_gate_stats(base["gate"]))
    for tag, other in (("shuffle", shuf), ("zero", zero), ("mean", meanr)):
        cmp_ = _compare_preds(base, other, arr["tree_mask"])
        for k, v in cmp_.items():
            row[f"{tag}_{k}"] = v
        row[f"{tag}_gate_mean"] = float(other["gate"].float().mean())
        row[f"{tag}_gate_std"] = float(other["gate"].float().std())
    return row


def run_one_tile_gradient(model, tile_meta: dict[str, Any], max_points: int, seed: int) -> dict[str, Any]:
    """Single forward+backward (no optimizer.step) for gradient norms."""
    import torch
    import torch.nn.functional as F

    tile = Path(tile_meta["tile"])
    arr = _load_tile_for_audit(tile, max_points=max_points, seed=seed)
    batch = _make_batch_from_arrays(arr, arr["colors"])
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(batch, return_loss=False)
    logits = out["semantic_prediction_logits"]
    offs = out["offset_predictions"]
    loss = F.cross_entropy(logits, batch["semantic_labels"].to(logits.device)) + offs.pow(2).mean()
    loss.backward()
    named = {
        k: float(v.grad.norm().item()) if v.grad is not None else 0.0
        for k, v in model.named_parameters()
    }
    sem_g = _head_first_weight_grad(named, "semantic_linear")
    off_g = _head_first_weight_grad(named, "offset_linear")
    grads = {
        "rgb_encoder.0.weight": named.get("rgb_encoder.0.weight", 0.0),
        "rgb_encoder.2.weight": named.get("rgb_encoder.2.weight", 0.0),
        "rgb_gate.weight": named.get("rgb_gate.weight", 0.0),
        "rgb_gate.bias": named.get("rgb_gate.bias", 0.0),
        "semantic_linear_first_weight": sem_g,
        "offset_linear_first_weight": off_g,
    }
    model.eval()
    model.zero_grad(set_to_none=True)
    return {
        "name": tile_meta["name"],
        "stratum": tile_meta["stratum"],
        "n_points_used": int(arr["n"]),
        "n_points_full": int(arr["n_full"]),
        "loss": float(loss.detach().item()),
        "grads": grads,
        "rgb_to_sem_grad_ratio": float(grads["rgb_gate.weight"] / (sem_g + 1e-12)),
        "rgb_to_off_grad_ratio": float(grads["rgb_gate.weight"] / (off_g + 1e-12)),
        "rgb_encoder_and_gate_nonzero": bool(
            grads["rgb_encoder.0.weight"] > 0
            and grads["rgb_encoder.2.weight"] > 0
            and grads["rgb_gate.weight"] > 0
        ),
    }


def decide_multitile_verdict(rows: list[dict[str, Any]], gradients: list[dict[str, Any]]) -> str:
    if not rows:
        return "INVALID"
    # noise floor: class flip > 0.1% or logits MAE > 1e-3 or offset L2 > 1e-3
    active_flags = []
    for r in rows:
        active = (
            r.get("shuffle_semantic_class_change_frac", 0) > 1e-3
            or r.get("shuffle_semantic_logits_mae", 0) > 1e-3
            or r.get("shuffle_offset_mean_l2", 0) > 1e-3
        )
        active_flags.append(bool(active))
    n_active = sum(active_flags)
    n = len(active_flags)
    if gradients and not all(g.get("rgb_encoder_and_gate_nonzero") for g in gradients):
        # still allow ACTIVE_* if causal is strong, but mark heterogeneous risk via INVALID only if no causal
        if n_active == 0:
            return "INVALID"
    if n_active >= int(math.ceil(0.7 * n)):
        return "ACTIVE_STABLE"
    if n_active >= 1:
        return "ACTIVE_HETEROGENEOUS"
    return "DORMANT"


def write_multitile_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_multitile_report(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    gradients: list[dict[str, Any]],
    ckpt_cfg: dict[str, Any],
) -> None:
    v = summary["verdict"]
    agg = summary["aggregates"]
    lines = [
        "# RGB Gate Multi-Tile Audit",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Mode: `{summary.get('mode')}` / GPU: `{summary.get('gpu_name')}`",
        f"- Tiles: {summary.get('n_tiles')} (max_points={summary.get('max_points')})",
        f"- Subsampled tiles: {summary.get('n_subsampled')} / {summary.get('n_tiles')}",
        "",
        "## 1. 最终判断",
        "",
        f"**{v}**",
        "",
        "## 2. 两件事必须分开看",
        "",
        "1. **RGB Gate 确实参与预测**：由打乱/均值 RGB 反事实下 logits、offset、类别翻转是否系统性变化证明。",
        "2. **RGB Gate 对最终指标是否有正向贡献**：本审计**不能**回答；需 A0/A1/A2/A3 消融（有/无 RGB gate 的完整训推对比）。",
        "",
        "全零 RGB 是分布外测试：高翻转率只说明「模型对 RGB 缺失敏感」，**不等于** RGB 带来性能增益。",
        "",
        "## 3. Checkpoint 统一",
        "",
        f"- 已修正 `configs/pipeline/pipeline.yaml` pretrain → `{ckpt_cfg.get('pipeline_pretrain_after')}`",
        f"- 正式 eval_subset 配置 `pipeline_l1w_eval_subset.yaml`: `{ckpt_cfg.get('formal_eval_subset_pretrain')}`",
        "- 消融 configs 的 epoch_1000：**实验设计本来如此，未改**。",
        "- `work_dirs/generated_l1w_workflow/*` 中大量 `epoch_781`：多为 postprocess/dupseed **复用已有 pointwise** 时从旧 pipeline 模板继承的字段，**不驱动重新推理**；正式推理结果目录为 `results_l1w_eval_subset_epoch720*`。",
        "",
        "## 4. Tile 抽样",
        "",
        "| name | stratum | n_full | n_used | subsampled |",
        "|---|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['stratum']} | {r['n_points_full']} | {r['n_points_used']} | {r['subsampled']} |"
        )
    lines += [
        "",
        "## 5. 跨 Tile 汇总（按 Tile 统计量再汇总，非点合并）",
        "",
        "### Shuffle RGB（相对原始）",
        "",
        "| metric | mean | std | median | min | max | q25 | q75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "shuffle_semantic_logits_mae",
        "shuffle_tree_prob_mae",
        "shuffle_offset_mean_l2",
        "shuffle_semantic_class_change_frac",
        "shuffle_tree_class_change_frac",
        "shuffle_nontree_class_change_frac",
        "gate_mean",
        "gate_std",
        "gate_lt_0p01_frac",
    ):
        s = agg.get(key, {})
        lines.append(
            f"| {key} | {s.get('mean', float('nan')):.6g} | {s.get('std', float('nan')):.6g} | "
            f"{s.get('median', float('nan')):.6g} | {s.get('min', float('nan')):.6g} | "
            f"{s.get('max', float('nan')):.6g} | {s.get('q25', float('nan')):.6g} | "
            f"{s.get('q75', float('nan')):.6g} |"
        )
    lines += [
        "",
        "### Zero RGB（OOD；仅作敏感性，不作增益证据）",
        "",
        f"- class_change mean={agg.get('zero_semantic_class_change_frac', {}).get('mean')}",
        f"- logits_mae mean={agg.get('zero_semantic_logits_mae', {}).get('mean')}",
        "",
        "### Mean RGB",
        "",
        f"- class_change mean={agg.get('mean_semantic_class_change_frac', {}).get('mean')}",
        f"- logits_mae mean={agg.get('mean_semantic_logits_mae', {}).get('mean')}",
        "",
        "## 6. 梯度稳定性（小/中/大各 1）",
        "",
    ]
    for g in gradients:
        gd = g["grads"]
        lines.append(
            f"- **{g['name']}** ({g['stratum']}, n={g['n_points_used']}): "
            f"enc0={gd['rgb_encoder.0.weight']:.4g}, enc2={gd['rgb_encoder.2.weight']:.4g}, "
            f"gate_w={gd['rgb_gate.weight']:.4g}, gate_b={gd['rgb_gate.bias']:.4g}, "
            f"sem={gd['semantic_linear_first_weight']:.4g}, off={gd['offset_linear_first_weight']:.4g}, "
            f"ratios={g['rgb_to_sem_grad_ratio']:.4g}/{g['rgb_to_off_grad_ratio']:.4g}, "
            f"nonzero={g['rgb_encoder_and_gate_nonzero']}"
        )
    lines += [
        "",
        f"- 三 Tile RGB encoder+gate 均非零: **{all(g.get('rgb_encoder_and_gate_nonzero') for g in gradients)}**",
        "",
        "## 7. 是否可开始 A0/A1/A2/A3",
        "",
        summary.get("ablation_readiness_note", ""),
        "",
        "## 8. 判定依据摘要",
        "",
        f"- tiles_with_clear_shuffle_effect: {summary.get('n_tiles_active')}/{summary.get('n_tiles')}",
        f"- verdict rule: >=70% → ACTIVE_STABLE; >=1 → ACTIVE_HETEROGENEOUS; 0 → DORMANT",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_multitile_audit(
    ckpt: Path,
    tile_root: Path,
    out_dir: Path,
    n_tiles: int = 20,
    max_points: int = 50_000,
    seed: int = 0,
    ckpt_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    from tree_learn.model import TreeLearn
    from tree_learn.util import get_config

    out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("multi-tile audit requires CUDA full backbone")

    print(f"[multitile] indexing tiles under {tile_root}", flush=True)
    selected = select_stratified_l1w_tiles(tile_root, n_tiles=n_tiles, seed=seed)
    jdump(out_dir / "selected_tiles.json", selected)

    print(f"[multitile] load model {ckpt}", flush=True)
    cfg = get_config(str(REPO / "configs/training/train.yaml"))
    model = TreeLearn(**cfg.model).cuda()
    state = torch.load(str(ckpt), map_location="cpu")
    missing, unexpected = model.load_state_dict(state["net"], strict=False)
    model.eval()

    rows: list[dict[str, Any]] = []
    for i, meta in enumerate(selected):
        print(
            f"[multitile] causal {i+1}/{len(selected)} {meta['name']} "
            f"({meta['stratum']}, n_full={meta['n_points_full']})",
            flush=True,
        )
        row = run_one_tile_causal(model, meta, max_points=max_points, seed=seed)
        rows.append(row)
        torch.cuda.empty_cache()

    # one small / medium / large for grads
    by_stratum = {s: [m for m in selected if m["stratum"] == s] for s in ("small", "medium", "large")}
    grad_metas = []
    for s in ("small", "medium", "large"):
        if by_stratum[s]:
            # pick median-sized within stratum
            grad_metas.append(sorted(by_stratum[s], key=lambda m: m["n_points_full"])[len(by_stratum[s]) // 2])

    gradients: list[dict[str, Any]] = []
    for meta in grad_metas:
        print(f"[multitile] gradient {meta['name']} ({meta['stratum']})", flush=True)
        gradients.append(run_one_tile_gradient(model, meta, max_points=max_points, seed=seed))
        torch.cuda.empty_cache()

    metric_keys = [
        k
        for k in rows[0].keys()
        if isinstance(rows[0][k], (int, float, np.floating, np.integer))
        and k not in ("n_points_full", "n_points_used", "n_tree_gt", "n_nontree_gt", "subsampled")
    ]
    aggregates = {k: _summarize_scalar_list([float(r[k]) for r in rows]) for k in metric_keys}

    active_flags = [
        (
            r.get("shuffle_semantic_class_change_frac", 0) > 1e-3
            or r.get("shuffle_semantic_logits_mae", 0) > 1e-3
            or r.get("shuffle_offset_mean_l2", 0) > 1e-3
        )
        for r in rows
    ]
    verdict = decide_multitile_verdict(rows, gradients)
    ready = verdict in ("ACTIVE_STABLE", "ACTIVE_HETEROGENEOUS") and all(
        g.get("rgb_encoder_and_gate_nonzero") for g in gradients
    )
    readiness = (
        "具备开始 A0/A1/A2/A3 正式消融的**前提条件**（跨 Tile 证明 RGB 参与预测，且梯度通路畅通）。"
        "仍需你确认后再启动四组训练；本审计不替代消融对最终指标贡献的结论。"
        if ready
        else "暂不建议直接开四组消融：跨 Tile 因果或梯度证据不足，先排查数据/配置。"
    )

    summary = {
        "verdict": verdict,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "load_missing_keys": list(missing),
        "load_unexpected_keys": list(unexpected),
        "mode": "gpu_full_model_multitile",
        "gpu_name": torch.cuda.get_device_name(0),
        "tile_root": str(tile_root),
        "n_tiles": len(rows),
        "n_subsampled": int(sum(1 for r in rows if r["subsampled"])),
        "max_points": max_points,
        "seed": seed,
        "n_tiles_active": int(sum(active_flags)),
        "aggregates": aggregates,
        "gradients": gradients,
        "ablation_ready": bool(ready),
        "ablation_readiness_note": readiness,
        "checkpoint_config": ckpt_cfg or {},
        "per_tile": rows,
    }
    write_multitile_csv(out_dir / "per_tile_results.csv", rows)
    jdump(out_dir / "summary.json", summary)
    write_multitile_report(out_dir / "report.md", summary, rows, gradients, ckpt_cfg or {})
    print(f"[multitile] VERDICT {verdict}", flush=True)
    return summary


def audit_checkpoint_configs() -> dict[str, Any]:
    """Classify pretrain references (fix-needed vs by-design)."""
    import re

    pipeline = REPO / "configs/pipeline/pipeline.yaml"
    text = pipeline.read_text(encoding="utf-8")
    m = re.search(r"pretrain:\s*['\"]?([^'\"\n]+)", text)
    pipeline_pretrain = m.group(1).strip() if m else None

    ablations = {}
    for p in sorted((REPO / "configs/pipeline/ablations").glob("*.yaml")):
        t = p.read_text(encoding="utf-8")
        mm = re.search(r"pretrain:\s*['\"]?([^'\"\n]+)", t)
        ablations[p.name] = mm.group(1).strip() if mm else None

    formal = REPO / "work_dirs/generated_l1w_workflow/pipeline_l1w_eval_subset.yaml"
    formal_pretrain = None
    if formal.exists():
        mm = re.search(r"pretrain:\s*['\"]?([^'\"\n]+)", formal.read_text(encoding="utf-8"))
        formal_pretrain = mm.group(1).strip() if mm else None

    generated = {"stale_inherited_postprocess": [], "other_non720": [], "uses_720": []}
    gen_dir = REPO / "work_dirs/generated_l1w_workflow"
    if gen_dir.exists():
        for p in sorted(gen_dir.glob("*.yaml")):
            t = p.read_text(encoding="utf-8")
            mm = re.search(r"pretrain:\s*['\"]?([^'\"\n]+)", t)
            if not mm:
                continue
            pret = mm.group(1).strip()
            if "720" in pret:
                generated["uses_720"].append(p.name)
            elif any(x in pret for x in ("781", "1000", "680", "675")):
                if any(k in p.name for k in ("dupseed", "postprocess", "dupmerge", "cse_")):
                    generated["stale_inherited_postprocess"].append({"file": p.name, "pretrain": pret})
                else:
                    generated["other_non720"].append({"file": p.name, "pretrain": pret})

    compare = REPO / "configs/pipeline/pipeline_compare_675.yaml"
    compare_pretrain = None
    if compare.exists():
        mm = re.search(r"pretrain:\s*['\"]?([^'\"\n]+)", compare.read_text(encoding="utf-8"))
        compare_pretrain = mm.group(1).strip() if mm else None

    return {
        "pipeline_pretrain_after": pipeline_pretrain,
        "formal_eval_subset_pretrain": formal_pretrain,
        "ablations_by_design": ablations,
        "pipeline_compare_675": compare_pretrain,
        "generated": {
            "n_uses_720": len(generated["uses_720"]),
            "n_stale_inherited_postprocess": len(generated["stale_inherited_postprocess"]),
            "stale_inherited_postprocess_sample": generated["stale_inherited_postprocess"][:8],
            "other_non720": generated["other_non720"],
        },
        "classification": {
            "需要修正": [
                "configs/pipeline/pipeline.yaml（已改为 work_dirs/train_20260627/epoch_720.pth）",
            ],
            "实验设计本来如此": [
                "configs/pipeline/ablations/*.yaml → 各自 train_ablation_*/epoch_1000.pth",
                "configs/pipeline/pipeline_compare_675.yaml → epoch_675.pth",
                "work_dirs/generated_l1w_workflow/pipeline_l1w_raw_forest.yaml → epoch_680（历史 raw 实验）",
            ],
            "继承字段但不驱动推理": [
                "generated_* postprocess/dupseed yaml 中的 epoch_781：脚本复用 source pointwise，"
                "pretrain 来自修正前的 pipeline 模板，不重新跑 backbone。",
            ],
        },
    }


def main() -> int:
    args = parse_args()
    if args.multi_tile:
        out_dir = args.multitile_out if args.multitile_out.is_absolute() else REPO / args.multitile_out
        tile_root = args.val_root if args.val_root.exists() else args.tile_root
        ckpt_cfg = audit_checkpoint_configs()
        jdump(out_dir / "checkpoint_config_audit.json", ckpt_cfg)
        summary = run_multitile_audit(
            ckpt=args.checkpoint if args.checkpoint.is_absolute() else REPO / args.checkpoint,
            tile_root=tile_root,
            out_dir=out_dir,
            n_tiles=args.n_tiles,
            max_points=args.max_points,
            seed=args.seed,
            ckpt_cfg=ckpt_cfg,
        )
        print(json.dumps({k: summary[k] for k in (
            "verdict", "n_tiles", "n_tiles_active", "n_subsampled", "ablation_ready", "gpu_name"
        )}, indent=2, ensure_ascii=False))
        print(f"report: {out_dir / 'report.md'}")
        return 0

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = {
        "git_status": "git status --short",
        "git_pull": "git pull origin main",
        "audit": f"{sys.executable} tools/diagnostics/audit_rgb_gate.py --checkpoint {args.checkpoint}",
    }

    print("[1/5] code path audit", flush=True)
    code = code_path_audit()

    print("[2/5] RGB data audit", flush=True)
    data = {
        "train": audit_npz_dir(args.train_root, args.train_sample, args.seed, "train_random_crops"),
        "val": audit_npz_dir(args.val_root, None, args.seed, "val_tiles"),
        "tiles": audit_npz_dir(args.tile_root, None, args.seed + 1, "infer_tiles"),
        "tile_exclusivity": audit_tile_root_exclusivity(args.tile_root),
        "forests_dir_exists": (REPO / "data/train/forests").exists(),
        "benchmark_laz_exists": (REPO / "data/benchmark/L1W_voxelized01_for_eval.laz").exists(),
    }
    # alignment: val tiles must match val forest (not pipeline L1W.npz)
    forest_npz = REPO / "data/val/forest_voxelized0.1/L1W.npz"
    if not forest_npz.exists():
        forest_npz = REPO / "data/pipeline/L1W/forest_voxelized0.1/L1W.npz"
    print("[2b] val tile <-> forest RGB alignment (5 tiles)", flush=True)
    data["val_forest_alignment"] = align_val_tiles_to_forest(
        args.val_root, forest_npz, n_tiles=5, seed=args.seed
    )
    jdump(out_dir / "data_audit.json", data)

    print("[3/5] checkpoint + gate MLP", flush=True)
    # gather colors from checked val files preferentially
    val_files = sorted(args.val_root.glob("*.npz")) if args.val_root.exists() else []
    train_files = sorted(args.train_root.glob("*.npz")) if args.train_root.exists() else []
    color_src = (val_files[:20] + train_files[:30]) if val_files or train_files else []
    colors = collect_color_samples(color_src, max_points=100_000, seed=args.seed)
    ckpt = checkpoint_and_gate_mlp_audit(args.checkpoint, colors)
    ckpt["commands"] = commands
    jdump(out_dir / "checkpoint_audit.json", ckpt)

    fwd = None
    if not args.skip_forward:
        print("[4/5] forward/grad/counterfactual", flush=True)
        try:
            fwd = forward_causal_audit(args.checkpoint, args.val_root, seed=args.seed)
        except Exception as exc:  # noqa: BLE001
            fwd = {"error": str(exc), "mode": "failed"}
        jdump(out_dir / "forward_audit.json", fwd)
    else:
        jdump(out_dir / "forward_audit.json", {"skipped": True})

    print("[5/5] verdict + report", flush=True)
    verdict = decide_verdict(data, ckpt, fwd)
    meta = {
        "verdict": verdict,
        "checkpoint": str(args.checkpoint),
        "checkpoint_exists": args.checkpoint.exists(),
        "out_dir": str(out_dir),
        "commands": commands,
    }
    jdump(out_dir / "verdict.json", meta)
    write_report(out_dir / "report.md", verdict, code, data, ckpt, fwd)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
