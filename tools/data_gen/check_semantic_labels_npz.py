#!/usr/bin/env python3
"""
Randomly inspect NPZ files and validate semantic_labels quality.

Checks for each sampled file:
1) `semantic_labels` field exists
2) labels contain both 0 (non-tree) and 1 (tree)
3) labels are not all zeros (dead data)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spot-check .npz files for valid semantic_labels."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/train"),
        help="Directory to recursively search for .npz files (default: data/train).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many files to sample (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    return parser.parse_args()


def check_one_file(npz_path: Path) -> tuple[bool, str]:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "semantic_labels" not in data:
                return False, "missing semantic_labels"

            labels = np.asarray(data["semantic_labels"]).reshape(-1)
            if labels.size == 0:
                return False, "semantic_labels is empty"

            unique_vals = set(np.unique(labels).tolist())
            if 0 not in unique_vals or 1 not in unique_vals:
                return (
                    False,
                    f"semantic_labels does not contain both 0 and 1, got {sorted(unique_vals)}",
                )

            if np.all(labels == 0):
                return False, "all labels are 0 (dead data)"

        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to read npz: {exc}"


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir

    if args.num_samples <= 0:
        print("[ERROR] --num-samples must be > 0")
        return 2

    if not data_dir.exists():
        print(f"[ERROR] data directory not found: {data_dir}")
        return 2

    npz_files = sorted(data_dir.rglob("*.npz"))
    if not npz_files:
        print(f"[ERROR] no .npz files found under: {data_dir}")
        return 2

    k = min(args.num_samples, len(npz_files))
    random.seed(args.seed)
    sampled_files = random.sample(npz_files, k=k)

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] total_npz_files={len(npz_files)}")
    print(f"[INFO] sampled={k}, seed={args.seed}")
    print("-" * 80)

    failed = []
    for path in sampled_files:
        ok, msg = check_one_file(path)
        rel = path.relative_to(data_dir) if path.is_relative_to(data_dir) else path
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {rel} -> {msg}")
        if not ok:
            failed.append(path)

    print("-" * 80)
    print(f"[SUMMARY] pass={k - len(failed)}, fail={len(failed)}, sampled={k}")
    if failed:
        print("[SUMMARY] some sampled files are invalid.")
        return 1

    print("[SUMMARY] all sampled files are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
