#!/usr/bin/env python3
"""GPU full-model RGB-gate causal audit (one val batch).

Updates:
  work_dirs/rgb_gate_audit/forward_audit_gpu.json
  and refreshes the counterfactual/gradient sections in report.md / verdict notes.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import build_dataloader, get_config, get_root_logger

OUT = REPO / "work_dirs" / "rgb_gate_audit"
CKPT = REPO / "work_dirs" / "train_20260627" / "epoch_720.pth"
SIGMOID_NEG6 = float(1.0 / (1.0 + math.exp(6.0)))
SEED = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def named_grad_l2(model: torch.nn.Module) -> dict[str, float]:
    out = {}
    for k, v in model.named_parameters():
        if v.grad is None:
            out[k] = 0.0
        else:
            out[k] = float(v.grad.detach().norm().item())
    return out


def pick_first_weight(named: dict[str, float], prefix: str) -> float:
    for k, v in named.items():
        if k.startswith(prefix) and k.endswith("weight"):
            return v
    return 0.0


def run_infer(model, batch, colors: torch.Tensor) -> dict:
    b = dict(batch)
    b["colors"] = colors
    with torch.no_grad():
        out = model(b, return_loss=False)
    gate = out["rgb_gate_mask"]
    logits = out["semantic_prediction_logits"]
    offs = out["offset_predictions"]
    probs = torch.softmax(logits, dim=-1)[:, 0]
    return {
        "gate_mean": float(gate.mean().item()),
        "gate_std": float(gate.std().item()),
        "gate_min": float(gate.min().item()),
        "gate_max": float(gate.max().item()),
        "logits": logits.detach(),
        "probs": probs.detach(),
        "offsets": offs.detach(),
        "gate": gate.detach(),
    }


def compare(tag: str, base: dict, other: dict) -> dict:
    logits_mae = float((base["logits"] - other["logits"]).abs().mean().item())
    logits_max = float((base["logits"] - other["logits"]).abs().max().item())
    prob_mae = float((base["probs"] - other["probs"]).abs().mean().item())
    off_l2 = float((base["offsets"] - other["offsets"]).norm(dim=-1).mean().item())
    cls_change = float((base["logits"].argmax(-1) != other["logits"].argmax(-1)).float().mean().item())
    gate_mae = float((base["gate"] - other["gate"]).abs().mean().item())
    return {
        "tag": tag,
        "gate_mean": other["gate_mean"],
        "gate_std": other["gate_std"],
        "gate_mae_vs_original": gate_mae,
        "semantic_logits_mae": logits_mae,
        "semantic_logits_max_abs": logits_max,
        "tree_prob_mae": prob_mae,
        "offset_mean_l2": off_l2,
        "semantic_class_change_frac": cls_change,
    }


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required for this script"
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = get_config(str(REPO / "configs/training/train.yaml"))
    logger = get_root_logger(str(OUT / "gpu_audit.log"))
    model = TreeLearn(**cfg.model).cuda()
    state = torch.load(str(CKPT), map_location="cpu")
    missing, unexpected = model.load_state_dict(state["net"], strict=False)
    print("epoch", state.get("epoch"), "use_rgb_gate", model.use_rgb_gate)
    print("missing", missing)
    print("unexpected", unexpected)

    # One deterministic val sample (same tile preference as CPU audit if present)
    val_set = TreeDataset(**cfg.dataset_test, logger=logger)
    # Prefer L1W_276 if available
    prefer = None
    for i, p in enumerate(val_set.data_paths):
        if Path(p).name == "L1W_276.npz":
            prefer = i
            break
    if prefer is None:
        prefer = SEED % len(val_set)
    sample = val_set[prefer]
    batch = val_set.collate_fn([sample])
    # move to cuda via model forward (cuda_cast), but colors variants need to be on same device as batch tensors after cast
    # First push batch tensors to cuda explicitly for color surgery
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.cuda(non_blocking=True)

    colors = batch["colors"]
    n = colors.shape[0]
    print("batch points", n, "tile", Path(val_set.data_paths[prefer]).name)

    # ---- gradients on full model ----
    model.train()
    model.zero_grad(set_to_none=True)
    batch["current_epoch"] = int(state.get("epoch", 720))
    loss, loss_dict = model(batch, return_loss=True)
    loss.backward()
    named = named_grad_l2(model)
    grads = {
        "rgb_encoder.0.weight": named.get("rgb_encoder.0.weight", 0.0),
        "rgb_encoder.2.weight": named.get("rgb_encoder.2.weight", 0.0),
        "rgb_gate.weight": named.get("rgb_gate.weight", 0.0),
        "rgb_gate.bias": named.get("rgb_gate.bias", 0.0),
        "semantic_linear_first_weight": pick_first_weight(named, "semantic_linear"),
        "offset_linear_first_weight": pick_first_weight(named, "offset_linear"),
    }
    grad_block = {
        "loss": float(loss.detach().item()),
        "loss_terms": {k: float(v.detach().item()) for k, v in loss_dict.items() if torch.is_tensor(v)},
        "grads": grads,
        "rgb_to_sem_grad_ratio": grads["rgb_gate.weight"] / (grads["semantic_linear_first_weight"] + 1e-12),
        "rgb_to_off_grad_ratio": grads["rgb_gate.weight"] / (grads["offset_linear_first_weight"] + 1e-12),
    }
    print("grads", json.dumps(grads, indent=2))

    # ---- counterfactuals ----
    model.eval()
    torch.manual_seed(SEED)
    zero = torch.zeros_like(colors)
    mean_rgb = colors.mean(dim=0, keepdim=True).expand_as(colors)
    shuffle = colors[torch.randperm(n, device=colors.device)]

    base = run_infer(model, batch, colors)
    z = run_infer(model, batch, zero)
    s = run_infer(model, batch, shuffle)
    m = run_infer(model, batch, mean_rgb)

    result = {
        "mode": "gpu_full_model",
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(CKPT),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "use_rgb_gate": bool(model.use_rgb_gate),
        "load_missing_keys": list(missing),
        "load_unexpected_keys": list(unexpected),
        "tile": Path(val_set.data_paths[prefer]).name,
        "n_points": int(n),
        "ref_sigmoid_neg6": SIGMOID_NEG6,
        "gradient": grad_block,
        "counterfactual": {
            "original": {"gate_mean": base["gate_mean"], "gate_std": base["gate_std"], "gate_min": base["gate_min"], "gate_max": base["gate_max"]},
            "vs_zero_rgb": compare("zero", base, z),
            "vs_shuffle_rgb": compare("shuffle", base, s),
            "vs_mean_rgb": compare("mean", base, m),
        },
        "activity_signals": {
            "gate_mean": base["gate_mean"],
            "gate_std": base["gate_std"],
            "gate_near_init_mean": abs(base["gate_mean"] - SIGMOID_NEG6) < 5e-4,
            "shuffle_logits_mae": float((base["logits"] - s["logits"]).abs().mean().item()),
            "shuffle_changes_predictions": bool(
                float((base["logits"] - s["logits"]).abs().mean().item()) > 1e-5
                or float((base["offsets"] - s["offsets"]).norm(dim=-1).mean().item()) > 1e-5
                or float((base["logits"].argmax(-1) != s["logits"].argmax(-1)).float().mean().item()) > 1e-4
            ),
            "rgb_gate_grad_nonzero": grads["rgb_gate.weight"] > 0.0,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    # verdict under GPU evidence
    sig = result["activity_signals"]
    if (
        sig["shuffle_changes_predictions"]
        and sig["rgb_gate_grad_nonzero"]
        and not sig["gate_near_init_mean"]
    ):
        verdict = "ACTIVE"
    elif sig["gate_near_init_mean"] and not sig["shuffle_changes_predictions"]:
        verdict = "DORMANT"
    else:
        verdict = "ACTIVE" if sig["shuffle_changes_predictions"] else "INCONCLUSIVE"
    result["verdict_from_gpu"] = verdict

    out_path = OUT / "forward_audit_gpu.json"
    # strip non-json tensors already converted
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("wrote", out_path)
    print("verdict_from_gpu", verdict)
    print(json.dumps(result["counterfactual"], indent=2))

    # Append short section to report.md
    report = OUT / "report.md"
    section = f"""

## 11. GPU 完整模型复测（补跑）

- 设备: `{result['device']}`
- tile: `{result['tile']}`，点数: **{result['n_points']}**
- 判定（GPU）: **{verdict}**
- gate mean/std: **{base['gate_mean']:.6f}** / **{base['gate_std']:.6f}**
- 梯度: rgb_gate.weight=**{grads['rgb_gate.weight']:.6f}**，encoder.0=**{grads['rgb_encoder.0.weight']:.6f}**，semantic_first=**{grads['semantic_linear_first_weight']:.6f}**，offset_first=**{grads['offset_linear_first_weight']:.6f}**
- 相对梯度: gate/sem=**{grad_block['rgb_to_sem_grad_ratio']:.4f}**，gate/off=**{grad_block['rgb_to_off_grad_ratio']:.4f}**
- 打乱 RGB: logits MAE=**{result['counterfactual']['vs_shuffle_rgb']['semantic_logits_mae']:.6f}**，offset L2=**{result['counterfactual']['vs_shuffle_rgb']['offset_mean_l2']:.6f}**，类别变化率=**{result['counterfactual']['vs_shuffle_rgb']['semantic_class_change_frac']:.6f}**
- 全零 RGB: logits MAE=**{result['counterfactual']['vs_zero_rgb']['semantic_logits_mae']:.6f}**，offset L2=**{result['counterfactual']['vs_zero_rgb']['offset_mean_l2']:.6f}**
- 详情: `work_dirs/rgb_gate_audit/forward_audit_gpu.json`
"""
    with report.open("a", encoding="utf-8") as f:
        f.write(section)

    # Update verdict.json note
    vpath = OUT / "verdict.json"
    meta = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    meta["verdict"] = verdict
    meta["gpu_retest"] = {
        "verdict_from_gpu": verdict,
        "forward_audit_gpu": str(out_path),
        "device": result["device"],
    }
    vpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    # Inspect model return for loss packaging if needed
    raise SystemExit(main())
