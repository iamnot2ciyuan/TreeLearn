# L1W Commission Error 诊断实验方案

本文档用于验证 HTP-SegNet 在 L1W 上 commission error 偏高的来源。当前结果显示，模型的 completeness 和 omission error 稳定，precision、recall 与 coverage 也不差，但 commission error 明显高于原始 TreeLearn noisy。该现象通常来自两类原因：

1. 推理后处理过于宽松，导致低置信树点或小碎簇被保留下来。
2. Boundary_push 训练约束过强，使 offset-shifted 空间中过度分离，从而把真实单木拆成多个候选实例。

## 实验目标

第一阶段只调整推理参数，不重训模型。若 commission error 能明显下降，说明主要问题来自后处理阈值；若不同后处理组合均无法降低 commission error，则再进入第二阶段调整 Boundary_push 训练参数。

评价阈值保持不变，尤其不要修改 `configs/evaluation/evaluate.yaml` 中的 `min_precision_for_pred=0.5`，否则无法与原始 TreeLearn noisy 结果进行干净对比。

## 阶段 A：固定 checkpoint，做后处理参数 sweep

推荐先固定 epoch 720。该 checkpoint 在已有结果中 detection F1 最好，且 commission error 低于 epoch 781 和 epoch 1000。

### 1. 在远程主机更新代码

```bash
cd /path/to/TreeLearn
git pull
conda activate TreeLearn
```

如果远程主机上还没有对应 checkpoint，需要先把本地训练输出复制到远程，例如：

```bash
mkdir -p data/model_weights
cp work_dirs/train_20260627/epoch_720.pth data/model_weights/epoch_720.pth
cp work_dirs/train_20260627/epoch_781.pth data/model_weights/epoch_781.pth
```

### 2. 为 epoch 720 生成一次 pointwise 预测

后处理 sweep 会复用该文件：

```bash
python tools/pipeline/run_l1w_workflow.py \
  --mode eval_subset \
  --checkpoint data/model_weights/epoch_720.pth \
  --results-dir-name results_l1w_eval_subset_epoch720_base \
  --skip-eval
```

确认生成：

```bash
ls data/results_l1w_eval_subset_epoch720_base/pointwise_results/pointwise_results.npz
```

### 3. 运行小规模后处理 sweep

```bash
python tools/pipeline/run_l1w_postprocess_sweep.py \
  --mode eval_subset \
  --source-results-dir-name results_l1w_eval_subset_epoch720_base \
  --checkpoint-label epoch720 \
  --results-prefix pp_epoch720 \
  --tree-conf-thresh 0.45 0.50 0.55 0.60 \
  --tau-min 35 45 55 70 \
  --tau-vert 0.35 \
  --tau-off 6.0
```

输出汇总位于：

```text
work_dirs/generated_l1w_workflow/postprocess_sweep_eval_subset_pp_epoch720.md
work_dirs/generated_l1w_workflow/postprocess_sweep_eval_subset_pp_epoch720.csv
```

### 4. 加一轮更严格 seed 过滤

如果第一轮中 `tree_conf_thresh` 和 `tau_min` 能降低 Ecom，但仍偏高，再扫 `tau_vert` 和 `tau_off`：

```bash
python tools/pipeline/run_l1w_postprocess_sweep.py \
  --mode eval_subset \
  --source-results-dir-name results_l1w_eval_subset_epoch720_base \
  --checkpoint-label epoch720 \
  --results-prefix pp_epoch720_seedfilter \
  --tree-conf-thresh 0.55 \
  --tau-min 55 70 \
  --tau-vert 0.35 0.40 0.45 \
  --tau-off 6.0 5.0 4.5
```

### 5. 可选：检查边界点影响

如果怀疑 commission error 主要来自林地边界或未标注边缘树，运行：

```bash
python tools/pipeline/run_l1w_postprocess_sweep.py \
  --mode eval_subset \
  --source-results-dir-name results_l1w_eval_subset_epoch720_base \
  --checkpoint-label epoch720 \
  --results-prefix pp_epoch720_outer \
  --tree-conf-thresh 0.55 \
  --tau-min 55 \
  --tau-vert 0.35 \
  --tau-off 6.0 \
  --outer-remove none 0.3 0.5
```

## 阶段 B：用 epoch 781 复核是否是训练造成的

epoch 781 的 coverage 最高，但 commission error 更高。对它做同样的后处理 sweep：

```bash
python tools/pipeline/run_l1w_workflow.py \
  --mode eval_subset \
  --checkpoint data/model_weights/epoch_781.pth \
  --results-dir-name results_l1w_eval_subset_epoch781_base \
  --skip-eval

python tools/pipeline/run_l1w_postprocess_sweep.py \
  --mode eval_subset \
  --source-results-dir-name results_l1w_eval_subset_epoch781_base \
  --checkpoint-label epoch781 \
  --results-prefix pp_epoch781 \
  --tree-conf-thresh 0.45 0.50 0.55 0.60 \
  --tau-min 35 45 55 70 \
  --tau-vert 0.35 \
  --tau-off 6.0
```

## 判断标准

如果后处理 sweep 中出现以下情况，则优先认为问题来自推理参数：

- Ecom 相比原设置下降 3 到 5 个百分点以上。
- Coverage 下降不超过 1 个百分点。
- Recall 仍保持在 94 左右或以上。
- F1 明显高于当前 epoch 720 的 95.4。

如果所有后处理组合仍然保持高 Ecom，则更可能是训练阶段 Boundary_push 过强或 RGB gate 放大局部差异。此时进入训练参数调整。

## 阶段 C：Boundary_push 重训组合

若阶段 A/B 不能解决问题，建议只跑一个保守训练配置：

```yaml
model:
  boundary_push_weight: 0.20
  boundary_push_warmup_epochs: 50
  boundary_push_xy_radius: 1.0
  boundary_push_z_radius: 2.5
  boundary_push_margin: 0.60
  boundary_push_max_negatives: 8
```

该组合的目标是降低边界推离强度，减少真实单木被过度拆分，同时保留 Boundary_push 对树冠交叠区域的分离作用。

## 需要发回分析的文件

把以下文件内容发回来即可：

```text
work_dirs/generated_l1w_workflow/postprocess_sweep_eval_subset_pp_epoch720.md
work_dirs/generated_l1w_workflow/postprocess_sweep_eval_subset_pp_epoch720_seedfilter.md
work_dirs/generated_l1w_workflow/postprocess_sweep_eval_subset_pp_epoch781.md
```

如果只跑了第一轮，先发 `pp_epoch720.md` 也可以。重点是表格中的 `Ecom`、`F1`、`Precision`、`Recall` 和 `Coverage`。
