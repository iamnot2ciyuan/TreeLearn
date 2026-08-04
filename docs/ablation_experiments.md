# TreeLearn 改进模块消融实验方案

本文档记录当前仓库针对原始 TreeLearn 的两项改进及对应消融实验设计：

- `rgb_gate`：基于点级 RGB 的 late residual gating 融合模块。
- `boundary_push`：训练阶段用于增强相邻树边界分离的 push loss 约束。

实验目标是回答三个问题：

- 单独加入 `rgb_gate` 是否能提升语义预测、offset 预测或最终实例分割结果。
- 单独加入 `boundary_push` 是否能改善相邻树冠、交叠边界处的实例分离。
- 两个模块同时开启时是否存在叠加收益，或者其中一个模块已经解释了主要收益。

## 实验矩阵

四组实验只改变 `model.use_rgb_gate` 和 `model.use_boundary_push_constraint`，其余训练设置保持一致。

| 实验 ID | 实验含义 | RGB gate | Boundary push | 训练配置 |
| --- | --- | --- | --- | --- |
| A0 | 原始 TreeLearn 基线 | 关闭 | 关闭 | `configs/training/ablations/train_ablation_baseline.yaml` |
| A1 | 仅加入 RGB gate | 开启 | 关闭 | `configs/training/ablations/train_ablation_rgb_gate.yaml` |
| A2 | 仅加入 boundary push | 关闭 | 开启 | `configs/training/ablations/train_ablation_boundary_push.yaml` |
| A3 | 完整改进模型 | 开启 | 开启 | `configs/training/ablations/train_ablation_full.yaml` |

## 控制变量

为保证消融结果可以解释，四组训练配置保持以下因素一致：

- 数据路径：训练集使用 `data/train/random_crops/npz`，验证集使用 `data/val/tiles/npz`。
- 优化器：`AdamW`，学习率 `0.0001`，`weight_decay=0.001`。
- 学习率调度：`CosineLRScheduler`，`t_initial=1000`，`warmup_t=100`。
- 训练轮数：`epochs=1000`。
- 每轮样本数：`examples_per_epoch=500`。
- batch size：训练 `2`，验证 `1`。
- 预训练权重：`data/model_weights/hais_ckpt_spconv2.pth`。
- 随机种子：`seed=42`。
- 保存与验证频率：每 `20` 个 epoch 保存和验证一次。

## 代码开关

`rgb_gate` 的开关位于模型配置：

```yaml
model:
  use_rgb_gate: true
```

关闭时，模型直接使用原始 sparse U-Net backbone 输出特征，不要求 RGB gate 参与推理。

`boundary_push` 的开关同样位于模型配置：

```yaml
model:
  use_boundary_push_constraint: true
```

关闭时，训练 loss 中不再加入 `boundary_push_loss`，但普通 semantic loss、offset loss 和已有 geometry constraint 保持不变。

## 训练指令

建议四组实验分别在独立终端或任务管理器中运行。若显存有限，按顺序逐个跑。

```bash
python tools/training/train.py --config configs/training/ablations/train_ablation_baseline.yaml
```

```bash
python tools/training/train.py --config configs/training/ablations/train_ablation_rgb_gate.yaml
```

```bash
python tools/training/train.py --config configs/training/ablations/train_ablation_boundary_push.yaml
```

```bash
python tools/training/train.py --config configs/training/ablations/train_ablation_full.yaml
```

默认训练输出目录如下：

| 实验 ID | 输出目录 | 默认最终 checkpoint |
| --- | --- | --- |
| A0 | `work_dirs/train_ablation_baseline` | `work_dirs/train_ablation_baseline/epoch_1000.pth` |
| A1 | `work_dirs/train_ablation_rgb_gate` | `work_dirs/train_ablation_rgb_gate/epoch_1000.pth` |
| A2 | `work_dirs/train_ablation_boundary_push` | `work_dirs/train_ablation_boundary_push/epoch_1000.pth` |
| A3 | `work_dirs/train_ablation_full` | `work_dirs/train_ablation_full/epoch_1000.pth` |

## 推理指令

以下 pipeline 配置默认读取各自实验的 `epoch_1000.pth`。如果最终选择其他 epoch，需要先修改对应 pipeline 配置中的 `pretrain` 字段。

```bash
python tools/pipeline/pipeline.py --config configs/pipeline/ablations/pipeline_ablation_baseline.yaml
```

```bash
python tools/pipeline/pipeline.py --config configs/pipeline/ablations/pipeline_ablation_rgb_gate.yaml
```

```bash
python tools/pipeline/pipeline.py --config configs/pipeline/ablations/pipeline_ablation_boundary_push.yaml
```

```bash
python tools/pipeline/pipeline.py --config configs/pipeline/ablations/pipeline_ablation_full.yaml
```

默认推理输出目录如下：

| 实验 ID | Pipeline 配置 | 结果目录 |
| --- | --- | --- |
| A0 | `configs/pipeline/ablations/pipeline_ablation_baseline.yaml` | `data/results_ablation_baseline` |
| A1 | `configs/pipeline/ablations/pipeline_ablation_rgb_gate.yaml` | `data/results_ablation_rgb_gate` |
| A2 | `configs/pipeline/ablations/pipeline_ablation_boundary_push.yaml` | `data/results_ablation_boundary_push` |
| A3 | `configs/pipeline/ablations/pipeline_ablation_full.yaml` | `data/results_ablation_full` |

## 评估指令

```bash
python tools/evaluation/evaluate.py --config configs/evaluation/evaluate_ablation_baseline.yaml
```

```bash
python tools/evaluation/evaluate.py --config configs/evaluation/evaluate_ablation_rgb_gate.yaml
```

```bash
python tools/evaluation/evaluate.py --config configs/evaluation/evaluate_ablation_boundary_push.yaml
```

```bash
python tools/evaluation/evaluate.py --config configs/evaluation/evaluate_ablation_full.yaml
```

评估结果会写入各自预测结果目录下的 `evaluation` 子目录，并在日志中输出聚合指标。

## 推荐记录指标

训练阶段建议记录：

- `train/semantic_loss`
- `train/offset_loss`
- `train/geometry_loss`
- `train/boundary_push_loss`
- `train/boundary_push_active_pairs`
- `val/acc`
- `val/Offset_MAE`

最终实例分割评估建议记录：

- detection precision / recall / F1。
- instance segmentation precision / recall / F1。
- commission error 和 omission error。
- 分层或分区指标，例如 `xy_partition` 与 `z_partition` 下的细粒度结果。
- 失败案例可视化，重点关注相邻树冠、树干近邻、冠层交叠和边界树。

## 结果表模板

| 实验 ID | RGB gate | Boundary push | Checkpoint | Detection F1 | Segmentation F1 | Precision | Recall | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 关闭 | 关闭 | `epoch_1000.pth` |  |  |  |  | baseline |
| A1 | 开启 | 关闭 | `epoch_1000.pth` |  |  |  |  | RGB gate 贡献 |
| A2 | 关闭 | 开启 | `epoch_1000.pth` |  |  |  |  | boundary push 贡献 |
| A3 | 开启 | 开启 | `epoch_1000.pth` |  |  |  |  | 完整模型 |

## 结论解释方式

建议按以下方式拆解收益：

- `A1 - A0`：衡量 `rgb_gate` 的单独贡献。
- `A2 - A0`：衡量 `boundary_push` 的单独贡献。
- `A3 - A0`：衡量完整改进模型相对原始 TreeLearn 的总收益。
- `A3 - A1`：在已有 RGB gate 的基础上，额外加入 boundary push 的收益。
- `A3 - A2`：在已有 boundary push 的基础上，额外加入 RGB gate 的收益。

如果 `A3` 不优于 `A1` 或 `A2`，需要检查两个模块是否存在训练冲突，例如 boundary push 权重过强、RGB gate 收敛过慢、或二者都在影响 offset head 的稳定性。

## 科学性注意事项

当前消融配置沿用仓库已有的 L1W quick-evaluation 设置，适合做内部对比和代码 sanity check。但如果 L1W 已经参与过验证、checkpoint 选择或参数调优，就不能把 L1W 上的结果表述为严格独立测试集性能。

正式论文或报告建议使用完全独立的 held-out forest：

- 不参与训练。
- 不参与验证。
- 不参与 checkpoint 选择。
- 不参与 grouping、threshold 或 loss 权重调参。

更换最终测试集时，需要同步更新：

- pipeline 配置中的 `forest_path`。
- evaluation 配置中的 `pred_forest_path`。
- evaluation 配置中的 `gt_forest_path`。
