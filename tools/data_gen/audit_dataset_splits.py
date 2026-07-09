#!/usr/bin/env python3
"""Audit TreeLearn dataset preprocessing and split leakage risks.

This script performs two levels of checks:
1. Config-level checks: compare the declared train/val/pipeline/eval sources.
2. Filesystem-level checks: if processed split directories exist, inspect the
   generated chunk metadata and look for source-plot overlap across splits.

It is intentionally conservative: warnings indicate that reported metrics may
be optimistic or scientifically weak, not necessarily that the code is broken.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on user env
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
POINT_CLOUD_SUFFIXES = {".las", ".laz", ".npy", ".npz", ".txt"}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}


@dataclass
class Finding:
    severity: str
    title: str
    detail: str


@dataclass
class PathSummary:
    label: str
    path: Path
    exists: bool
    kind: str
    canonical_plots: set[str] = field(default_factory=set)


@dataclass
class ProcessedSplitSummary:
    label: str
    path: Path
    exists: bool
    chunk_count: int = 0
    source_plot_counts: Counter[str] = field(default_factory=Counter)
    missing_metadata_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit preprocessing, split hygiene, and leakage risk for TreeLearn."
    )
    parser.add_argument(
        "--train-config",
        default="configs/training/train.yaml",
        help="Training config path (default: configs/training/train.yaml).",
    )
    parser.add_argument(
        "--train-data-config",
        default="configs/data_gen/gen_train_data.yaml",
        help="Training data-generation config path.",
    )
    parser.add_argument(
        "--val-data-config",
        default="configs/data_gen/gen_val_data.yaml",
        help="Validation data-generation config path.",
    )
    parser.add_argument(
        "--pipeline-config",
        default="configs/pipeline/pipeline.yaml",
        help="Inference pipeline config path.",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/evaluation/evaluate.yaml",
        help="Evaluation config path.",
    )
    parser.add_argument(
        "--compute-sha256",
        action="store_true",
        help="When overlapping raw files are found, also compare SHA256 digests.",
    )
    parser.add_argument(
        "--fail-on-high-risk",
        action="store_true",
        help="Exit with code 2 if any HIGH-severity finding is detected.",
    )
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        ruby_script = (
            "require 'json'; require 'yaml'; "
            "data = YAML.safe_load(File.read(ARGV[0]), aliases: true); "
            "puts JSON.generate(data || {})"
        )
        try:
            completed = subprocess.run(
                ["ruby", "-e", ruby_script, str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                "This audit needs either PyYAML in Python or a system Ruby with YAML support."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            raise SystemExit(f"Failed to parse YAML config {path}: {stderr}") from exc
        data = json.loads(completed.stdout or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML config {path}, got {type(data).__name__}.")
    return data


def merge_override(default_cfg: Any, override_cfg: Any) -> None:
    if not isinstance(default_cfg, dict) or not isinstance(override_cfg, dict):
        return
    for key, value in override_cfg.items():
        if isinstance(value, dict) and key in default_cfg and isinstance(default_cfg[key], dict):
            merge_override(default_cfg[key], value)
        else:
            default_cfg[key] = value


def load_treelearn_config(path_value: str | Path) -> dict[str, Any]:
    path = resolve_repo_path(path_value)
    cfg = load_yaml(path)
    default_args = cfg.pop("default_args", None)
    if default_args is not None:
        for default_arg in default_args:
            default_cfg = load_yaml(resolve_repo_path(default_arg))
            default_cfg = copy.deepcopy(default_cfg)
            for key in list(cfg.keys()):
                if key in default_cfg:
                    merge_override(default_cfg[key], cfg[key])
            cfg.update(default_cfg)
    return cfg


def normalize_plot_id(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"_original_idx$", "", stem)
    stem = re.sub(r"_voxelized\d+_for_eval$", "", stem)
    stem = re.sub(r"_voxelized\d+$", "", stem)
    stem = re.sub(r"_for_eval$", "", stem)
    stem = re.sub(r"_eval$", "", stem)
    return stem


def summarize_declared_path(label: str, path_value: str | Path, kind: str) -> PathSummary:
    path = resolve_repo_path(path_value)
    exists = path.exists()
    canonical_plots: set[str] = set()

    if path.suffix.lower() in POINT_CLOUD_SUFFIXES:
        canonical_plots.add(normalize_plot_id(path.name))
    elif path.is_dir():
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in POINT_CLOUD_SUFFIXES:
                canonical_plots.add(normalize_plot_id(child.name))

    return PathSummary(
        label=label,
        path=path,
        exists=exists,
        kind=kind,
        canonical_plots=canonical_plots,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_source_plot(npz_path: Path) -> tuple[str, bool]:
    json_path = npz_path.parent.parent / "json" / f"{npz_path.stem}.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            plot_name = data.get("plot_name")
            if isinstance(plot_name, str) and plot_name:
                return normalize_plot_id(plot_name), True
        except (OSError, json.JSONDecodeError):
            pass

    inferred = re.sub(r"_\d+$", "", npz_path.stem)
    return normalize_plot_id(inferred), False


def scan_processed_split(label: str, path_value: str | Path) -> ProcessedSplitSummary:
    path = resolve_repo_path(path_value)
    if not path.exists():
        return ProcessedSplitSummary(label=label, path=path, exists=False)

    npz_files = sorted(path.glob("*.npz"))
    summary = ProcessedSplitSummary(label=label, path=path, exists=True, chunk_count=len(npz_files))

    for npz_path in npz_files:
        source_plot, has_metadata = infer_source_plot(npz_path)
        summary.source_plot_counts[source_plot] += 1
        if not has_metadata:
            summary.missing_metadata_count += 1
    return summary


def add_finding(findings: list[Finding], severity: str, title: str, detail: str) -> None:
    findings.append(Finding(severity=severity, title=title, detail=detail))


def format_plot_set(plot_ids: set[str] | list[str]) -> str:
    values = sorted(set(plot_ids))
    return ", ".join(values) if values else "<none detected>"


def audit_configs(
    train_raw: PathSummary,
    val_raw: PathSummary,
    eval_gt: PathSummary,
    pipeline_input: PathSummary,
    findings: list[Finding],
    compute_sha256: bool,
) -> None:
    if val_raw.canonical_plots and eval_gt.canonical_plots:
        overlap = val_raw.canonical_plots & eval_gt.canonical_plots
        if overlap:
            add_finding(
                findings,
                "HIGH",
                "Validation and evaluation appear to use the same forest",
                (
                    f"{val_raw.path} and {eval_gt.path} normalize to the same plot id(s): "
                    f"{format_plot_set(overlap)}. Using the same forest for checkpoint selection and final "
                    "reporting biases the benchmark upward."
                ),
            )

    if train_raw.canonical_plots and val_raw.canonical_plots:
        overlap = train_raw.canonical_plots & val_raw.canonical_plots
        if overlap:
            add_finding(
                findings,
                "HIGH",
                "Train and validation raw sources overlap",
                (
                    f"Raw train sources in {train_raw.path} and validation source {val_raw.path} share "
                    f"plot id(s): {format_plot_set(overlap)}."
                ),
            )

    if train_raw.canonical_plots and eval_gt.canonical_plots:
        overlap = train_raw.canonical_plots & eval_gt.canonical_plots
        if overlap:
            add_finding(
                findings,
                "HIGH",
                "Train and evaluation raw sources overlap",
                (
                    f"Raw train sources in {train_raw.path} and evaluation GT {eval_gt.path} share "
                    f"plot id(s): {format_plot_set(overlap)}."
                ),
            )

    if val_raw.exists and eval_gt.exists and val_raw.path.is_file() and eval_gt.path.is_file() and compute_sha256:
        if normalize_plot_id(val_raw.path.name) == normalize_plot_id(eval_gt.path.name):
            same_hash = sha256_file(val_raw.path) == sha256_file(eval_gt.path)
            if same_hash:
                add_finding(
                    findings,
                    "HIGH",
                    "Validation and evaluation files are byte-identical",
                    f"{val_raw.path} and {eval_gt.path} have the same SHA256 digest.",
                )

    if pipeline_input.path == eval_gt.path:
        add_finding(
            findings,
            "MEDIUM",
            "Pipeline is configured to run directly on the evaluation file",
            (
                f"pipeline forest_path and evaluation gt_forest_path are both {pipeline_input.path}. "
                "This is acceptable for a quick sanity check, but it is not the cleanest final benchmark setup."
            ),
        )
    elif pipeline_input.canonical_plots and eval_gt.canonical_plots:
        overlap = pipeline_input.canonical_plots & eval_gt.canonical_plots
        if overlap and "benchmark" in str(pipeline_input.path):
            add_finding(
                findings,
                "MEDIUM",
                "Pipeline input is the benchmark subset, not the raw held-out forest",
                (
                    f"pipeline forest_path={pipeline_input.path} overlaps evaluation GT plot id(s): "
                    f"{format_plot_set(overlap)} and lives under a benchmark path. This is useful for fast "
                    "debugging, but benchmark-subset inference can look slightly better than a doc-faithful "
                    "full-forest run."
                ),
            )

    if val_raw.path.suffix.lower() in POINT_CLOUD_SUFFIXES:
        add_finding(
            findings,
            "MEDIUM",
            "Validation is based on a single forest",
            (
                f"Validation source is a single file: {val_raw.path}. This is not leakage by itself, but "
                "it makes model selection noisy and weakens scientific claims about generalization."
            ),
        )
    elif val_raw.exists and val_raw.path.is_dir() and len(val_raw.canonical_plots) <= 1:
        add_finding(
            findings,
            "MEDIUM",
            "Validation appears to cover only one plot",
            (
                f"Validation directory {val_raw.path} exposes {len(val_raw.canonical_plots)} plot id(s): "
                f"{format_plot_set(val_raw.canonical_plots)}."
            ),
        )


def audit_processed_splits(
    train_processed: ProcessedSplitSummary,
    val_processed: ProcessedSplitSummary,
    findings: list[Finding],
) -> None:
    if not train_processed.exists or not val_processed.exists:
        missing = []
        if not train_processed.exists:
            missing.append(str(train_processed.path))
        if not val_processed.exists:
            missing.append(str(val_processed.path))
        add_finding(
            findings,
            "INFO",
            "Processed split directories are missing",
            "Could not inspect generated npz chunks because these paths do not exist: " + ", ".join(missing),
        )
        return

    overlap = set(train_processed.source_plot_counts) & set(val_processed.source_plot_counts)
    if overlap:
        add_finding(
            findings,
            "HIGH",
            "Processed train/validation chunks come from the same source plot",
            (
                f"train={train_processed.path} and val={val_processed.path} share source plot id(s): "
                f"{format_plot_set(overlap)}. This is a direct split-leakage risk."
            ),
        )

    if train_processed.missing_metadata_count > 0 or val_processed.missing_metadata_count > 0:
        add_finding(
            findings,
            "LOW",
            "Some processed chunks were audited via filename inference",
            (
                f"{train_processed.label}: {train_processed.missing_metadata_count} chunks without usable json metadata; "
                f"{val_processed.label}: {val_processed.missing_metadata_count}. The overlap audit still ran, "
                "but confidence is slightly lower than with explicit plot_name metadata."
            ),
        )


def print_path_summary(summary: PathSummary) -> None:
    print(f"[{summary.label}]")
    print(f"  kind   : {summary.kind}")
    print(f"  path   : {summary.path}")
    print(f"  exists : {'yes' if summary.exists else 'no'}")
    print(f"  plots  : {format_plot_set(summary.canonical_plots)}")


def print_processed_summary(summary: ProcessedSplitSummary) -> None:
    print(f"[{summary.label}]")
    print(f"  path        : {summary.path}")
    print(f"  exists      : {'yes' if summary.exists else 'no'}")
    if summary.exists:
        print(f"  chunk_count : {summary.chunk_count}")
        print(f"  source_plots: {format_plot_set(list(summary.source_plot_counts.keys()))}")
        print(f"  no-metadata : {summary.missing_metadata_count}")


def print_findings(findings: list[Finding]) -> None:
    ordered = sorted(findings, key=lambda item: SEVERITY_ORDER[item.severity])
    if not ordered:
        print("No findings.")
        return

    for finding in ordered:
        print(f"- [{finding.severity}] {finding.title}")
        print(f"  {finding.detail}")


def determine_verdict(findings: list[Finding]) -> str:
    if any(item.severity == "HIGH" for item in findings):
        return "HIGH RISK: at least one leakage or benchmark-reuse issue was detected."
    if any(item.severity == "MEDIUM" for item in findings):
        return "CAUTION: no direct leakage found, but the evaluation design is scientifically weak."
    if any(item.severity == "LOW" for item in findings):
        return "MOSTLY CLEAN: only minor audit limitations or reproducibility caveats were found."
    return "CLEAN: no overlap signals were detected by this audit."


def main() -> int:
    args = parse_args()

    train_cfg = load_treelearn_config(args.train_config)
    train_data_cfg = load_treelearn_config(args.train_data_config)
    val_data_cfg = load_treelearn_config(args.val_data_config)
    pipeline_cfg = load_treelearn_config(args.pipeline_config)
    eval_cfg = load_treelearn_config(args.eval_config)

    train_raw = summarize_declared_path(
        "train_raw",
        Path(train_data_cfg["base_dir"]) / "forests",
        "raw training forests",
    )
    val_raw = summarize_declared_path(
        "val_raw",
        val_data_cfg["forest_path"],
        "raw validation forest",
    )
    pipeline_input = summarize_declared_path(
        "pipeline_input",
        pipeline_cfg["forest_path"],
        "pipeline inference input",
    )
    eval_gt = summarize_declared_path(
        "eval_gt",
        eval_cfg["paths"]["gt_forest_path"],
        "evaluation ground truth",
    )

    train_processed = scan_processed_split("train_processed", train_cfg["dataset_train"]["data_root"])
    val_processed = scan_processed_split("val_processed", train_cfg["dataset_test"]["data_root"])

    findings: list[Finding] = []
    audit_configs(train_raw, val_raw, eval_gt, pipeline_input, findings, args.compute_sha256)
    audit_processed_splits(train_processed, val_processed, findings)

    print("=== TreeLearn Split Audit ===")
    print(f"repo_root: {REPO_ROOT}")
    print()
    print("=== Declared Sources ===")
    print_path_summary(train_raw)
    print_path_summary(val_raw)
    print_path_summary(pipeline_input)
    print_path_summary(eval_gt)
    print()
    print("=== Processed Splits ===")
    print_processed_summary(train_processed)
    print_processed_summary(val_processed)
    print()
    print("=== Findings ===")
    print_findings(findings)
    print()
    print("=== Verdict ===")
    print(determine_verdict(findings))
    print(
        "Interpretation: HIGH means leakage or benchmark reuse is likely; MEDIUM means the code path can "
        "still produce valid numbers, but those numbers should not be treated as an unbiased final benchmark."
    )

    if args.fail_on_high_risk and any(item.severity == "HIGH" for item in findings):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
