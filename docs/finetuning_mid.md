# 改造后 TreeLearn 的 mid 微调指南

更新：2026-11-01。任务是从服务器 `train_20260627/epoch_1000.pth` 初始化，完整保留已训练的 RGB gate 和 boundary push 设置，扩大人工标注数据范围。本文优先于旧指南中“仅 FOR-instance 微调”的默认流程。服务器执行提示词见 [server_finetune_mid_prompt.md](server_finetune_mid_prompt.md)。

## 1. mid 的定义与本次实验目标

原论文 §2.3.2、表 2 的 mid 使用 FOR-instance 的 train、val、test 加 LAUTx，单独保留 NIBIO plot 16 验证；L1W 和 Wytham 不加入训练。本次对齐这个**数据范围**，模型仍是我们的 RGB gate + boundary push 版本，初始化、优化器设置也与原论文不同，应命名为“改造模型的 mid 数据范围微调”，不能称为原版 mid 的完整复现。

正式训练清单应包含：

- FOR-instance 所有原 split 的样地，排除 NIBIO plot 16。原 split 作为来源信息保留，不再作为本实验 train/test 用途。
- 扩展版 LAUTx 的六个样地：p1、p2、p4、p10、p12、p14。
- 验证只有 NIBIO plot 16；必须核查它也未被初始 epoch1000 用于训练。如果被用过，不能声称是干净的 held-out 验证，需先报告并重新设计验证协议。
- L1W、Wytham 不加入 mid 训练；是否能作独立测试，还需核查 epoch1000 及其初始化、选权重和调参的历史。

**CULS plot 2 的身份会改变**：核对它在 FOR-instance 实际文件清单中的对应项；如果按照 mid 定义加入训练，原来“独立 CULS plot 2”的评估只能改叫训练样地诊断。若决定继续留出它，实验应标为 `mid-minus-CULS2`，不能既声称完整 mid 数据范围，又声称它是独立测试。

上一轮结果已显示检测和实例分割可能有取舍。本次应比较同一个 epoch1000 与 mid 后的模型；epoch720 可以保留作论文模型参照，但不能代替微调初始化作为对照。也不要从上一轮第 2 轮的 best_offset 接着训，否则同时改变了初始化和数据范围。

## 2. 先核查环境和已有运行

所有命令在服务器仓库根目录、原训练环境中运行。先检查 Git 工作区、GPU、磁盘和原 checkpoint；复用现有环境和已校验的数据，不执行会删除环境的 setup/setup.sh。不要覆盖已有实验目录。

```bash
git fetch origin codex/treelearn-finetuning-guide
git status --short
nvidia-smi
df -h .
```

如果服务器当前不是本分支，保留服务器本地改动，在独立 checkout/worktree 中运行。独立 checkout 中的数据和原权重路径要重新解析到服务器已有文件，不要把相对路径找不到误判为模型缺失。

定位 `work_dirs/train_20260627/epoch_1000.pth`，记录实际绝对路径、SHA256、checkpoint epoch、原训练配置、代码版本及 RGB 参数。核对 backbone、heads、rgb_hidden_channels、输入通道、voxel_size 和 spconv 格式。geometry loss 不能单凭 checkpoint 推断，应读取原训练配置。

## 3. 下载数据与保存清单

### FOR-instance

优先复用上一轮已经解压并校验的完整数据包，而不是只复用上一轮的训练目录。如果需要下载：

```bash
mkdir -p data/downloads/FORinstance
curl --fail --location --retry 3 --continue-at - \
  'https://zenodo.org/records/8287792/files/FORinstance_dataset.zip?download=1' \
  --output data/downloads/FORinstance/FORinstance_dataset.zip
echo '01fd03ba4068648e21fb4591fbff34c4  data/downloads/FORinstance/FORinstance_dataset.zip' | md5sum -c -
unzip -n data/downloads/FORinstance/FORinstance_dataset.zip -d data/downloads/FORinstance
```

`unzip -n` 不覆盖已有解压文件；若以前解压中断，要核实或换新目录重新解压。读取真实的 `data_split_metadata.csv`，不要猜列名或仅凭文件名中的数字选择 NIBIO plot 16。跨来源可能同名，建立唯一名称并保留映射与原扩展名。

### LAUTx（使用扩展版完整森林）

```bash
python tools/data_gen/download_finetune_assets.py --dataset QUTUWU
python tools/data_gen/download_finetune_assets.py --dataset QUTUWU \
  --files p1.laz p2.laz p4.laz p10.laz p12.laz p14.laz \
  --root data/downloads/LAUTx_mid
```

这六个原分辨率文件的元数据总大小约 1.97 GB；脚本会核验文件大小和 MD5，保存下载收据。不要将 `_vox0.1` 版本和同一原分辨率样地同时作为两份训练样地。下载器没有断点续传功能，`.part` 重试时会重下；不要把 `.part` 放进森林目录。

生成 `data/finetune_mid/split_manifest.csv`，至少记录 dataset、原 split、原绝对路径、标准化 plot ID、实际用途、链接路径、原文件校验和、是否有有效 RGB、颜色来源、初始模型接触历史。正式训练清单必须与“FOR 全量排除 NIBIO16 + LAUTx 六样地”逐项核对，不能悄悄少一个来源。

## 4. RGB、标签与独立目录

沿用原指南的标签约定，但必须逐来源检查。读取器要求 treeID 和 classification；内部实例 -1=忽略、0=非树、正整数=树，网络内部语义树=0、非树=1。保留地面、低植被、树基和边缘上下文。

**数据名称不保证带真实 RGB。** 检查每个样地的颜色维度、实际非零覆盖、方差、数值范围、来源及 points/colors 对齐情况，包括 LAUTx。不得随机赋色或拿 intensity 复制成 RGB；无颜色时流程补零仅保证代码能运行，不保证 RGB 模型有效适配。

若部分必需样地没有真实颜色，优先核实是否有合法、可配准的对应颜色数据。若无法满足，应明确报告“完整 mid 数据范围 + 真实 RGB 条件”尚不满足，不要静默删样地、关闭 gate 或把零颜色运行报告为已完成的 RGB-mid 实验。若另行设计有/无颜色混合训练，需明确缺失模态策略、代码变更和实验命名；当前配置没有这种策略。

准备全新的隔离目录，可符号链接原始数据：

```text
data/finetune_mid/train/forests/       FOR（除 NIBIO16）+ LAUTx 六样地
data/finetune_mid/val/forest/          仅 NIBIO16
data/finetune_mid_pilot/train/forests/ 与正式训练相同的样地清单
data/finetune_mid/train/random_crops/npz/
data/finetune_mid/val/tiles/npz/
```

森林目录只能放可读取的数据文件。元数据 CSV、收据、ZIP、日志、子目录放到目录外。链接名用唯一来源前缀（例如 FOR_NIBIO、LAUTx），并保留真实 `.las`/`.laz` 扩展名；不同源的 plot2 不能覆盖彼此。

## 5. 预处理：先 pilot，再正式 crops

配置文件：

- `configs/data_gen/gen_finetune_mid_pilot.yaml`：2,000 个请求 crops，独立 pilot 根目录。
- `configs/data_gen/gen_finetune_mid_train.yaml`：30,000 个请求 crops，独立正式根目录。
- `configs/data_gen/gen_finetune_mid_val.yaml`：将 forest_path 改为真实 NIBIO16 文件名，保留 `data/finetune_mid/val/forest/<实际文件名>` 的目录层级。

下采样 0.1 m、随机 crop 35×35 m、监督中央 8×8 m、verticality 邻域 0.6 m。验证 stride=1 是内区边长的比例，约 8 m 步长。生成器按 occupancy 面积分配每个样地的请求 crop 数，**不是**每个数据集等量采样；拒绝低占用候选后实际生成数可能少于请求值。报告每个来源的实际 crop 数、监督点数及 RGB 覆盖。

```bash
python tools/data_gen/check_forests_files.py data/finetune_mid/train/forests
python tools/data_gen/gen_train_data.py --config configs/data_gen/gen_finetune_mid_pilot.yaml
python tools/data_gen/gen_val_data.py --config configs/data_gen/gen_finetune_mid_val.yaml
```

check_forests_files 主要检测可读性，不能代替标签/RGB 检查。pilot 后检查 NPZ 的 points、feat、colors、instance_label，点行对齐、有限数值、有效内区树点与非树点、offset 合理性和磁盘占用。旧 `check_semantic_labels_npz.py` 不是当前 NPZ 的正确验证器。

正式数据生成前，按 pilot 实测占用估算 30,000 crops、缓存和 checkpoints 的总空间，留出余量；可能需要数百 GB。不要因空间不足静默改小配置后仍称与预设预算一致。清单、数据和参数固定后，生成正式数据：

```bash
python tools/data_gen/gen_train_data.py --config configs/data_gen/gen_finetune_mid_train.yaml
```

现有生成器不会自动读取训练配置的 seed，以上直接命令的 crop 随机性未固定。需要可复现生成时，在服务器通过一个记录在报告中的 Python wrapper 设置 random/numpy/torch seed=42 后 runpy 执行生成器，或添加明确的生成 seed 参数；不要声称训练的 seed 自动控制了数据生成。保存已生成 crop 清单和配置，后续比较复用同一份。

生成后做 raw/processed 的 train/val 来源审计，必须确认 NIBIO16 不在训练清单和缓存内。复用 `audit_dataset_splits.py` 时同时指定 mid 的 train、data-gen、val 配置和实际 pipeline/evaluation 配置；默认 L1W 配置只能检查其声明路径，不能证明 checkpoint 历史独立。

## 6. 微调配置与执行命令

新配置 `configs/training/finetune_mid.yaml` 保留上轮 full 微调参数，仅改变数据根目录和明确的原始 checkpoint 路径：RGB gate、boundary push 开启，geometry 配置仍需对照 epoch1000 原训练设置；严格完整加载，50 epoch，每轮最多 500 crops，batch 2，AdamW lr=1e-5→1e-6，学习率 warmup 3 epoch，FP32，clip=1.0，每轮验证并保存 best_offset。

有不少于 500 个训练 crops 时，每轮 250 次更新，总计约 12,500 次更新。30,000 是磁盘上的样本池，不表示每轮遍历 30,000 个。sample 数、batch、轮数或 loader 变化时重新核对实际更新次数。

boundary_push_warmup_epochs 主配置仍为 15，保持与上一轮一致，避免把增加数据与改变 loss 调度混为一个因素。按当前代码 epoch1 的 push 权重为 0、epoch2 约为 0.0233，epoch16 起为 0.35；之前选中 epoch2 不代表满权重 push 已验证。可另做预先声明的 `warmup=0` 对照，但必须使用相同 mid 数据与初始化、单独目录，并在验证集选方案；不默认增加这次任务。

原论文使用 lr=5e-4→5e-5、无学习率 warmup；它是原模型的设置。本次先保留低学习率，不能把“mid”理解成要给已训练的改造模型直接套用原论文学习率。

创建服务器运行配置副本并核实实际 checkpoint（相对路径须按服务器实际位置修改）：

```bash
python - <<'PY'
from pathlib import Path
import yaml
source = Path('configs/training/finetune_mid.yaml')
cfg = yaml.safe_load(source.read_text())
checkpoint = Path(cfg['pretrain']).resolve(strict=True)
cfg['pretrain'] = str(checkpoint)
Path('configs/training/finetune_mid.server.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
# Smoke run uses the same original checkpoint, separate data root and run directory.
cfg['dataset_train']['data_root'] = 'data/finetune_mid_pilot/train/random_crops/npz'
cfg['epochs'] = 2
cfg['examples_per_epoch'] = 20
cfg['scheduler']['t_initial'] = 2
cfg['scheduler']['warmup_t'] = 0
Path('configs/training/finetune_mid.smoke.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
MID_RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
python tools/training/train.py --config configs/training/finetune_mid.smoke.yaml \
  --work_dir "finetune_mid_smoke_${MID_RUN_TAG}"
```

严格加载检查必须覆盖 input_conv、unet、heads、rgb_encoder、rgb_gate。缺失/尺寸不符先恢复匹配的原代码或作可核实的格式转换，不能放宽 strict 掩盖。确认 loss 有限、RGB/backbone 参数更新、验证正常后，**重新从 epoch1000** 开始正式训练：

```bash
python tools/training/train.py --config configs/training/finetune_mid.server.yaml \
  --work_dir "finetune_mid_${MID_RUN_TAG}"
tensorboard --logdir work_dirs
```

MID_RUN_TAG 在同一 shell 中复用；跨 shell 时明确设置为记录的 run ID。长任务可由已有 tmux/任务调度器托管并保存日志。首次微调用 pretrain，不要 `--resume epoch_1000.pth`；仅恢复本次 mid 训练时使用：

```bash
python tools/training/train.py --config configs/training/finetune_mid.server.yaml \
  --work_dir "finetune_mid_${MID_RUN_TAG}" \
  --resume "work_dirs/finetune_mid_${MID_RUN_TAG}/epoch_25.pth"
```

epoch25 只是恢复示例，使用实际存在且属于同一实验的 checkpoint。现有 resume 没完整恢复 RNG/scaler 状态；记录这一限制。不要复用上一轮 full 的 best_offset 文件或输出目录。

## 7. 选择权重与测评

NIBIO16 最低验证 offset 决定本次 best_offset，不能用 L1W/Wytham/CULS 结果倒选 epoch。最好先记录 epoch1000 在 NIBIO16 上的 epoch0 指标；训练器的 best_offset 默认只在微调 epoch1 以后比较，因此要同时保留“没有优于初始化”的可能性，不能强称微调成功。

为了分析增加数据的影响，预先固定同一份 L1W eval_subset 和 grouping（含 CSE=0.5），对比原 epoch1000、上轮 FOR-only best_offset、这轮 mid best_offset；epoch720 作为额外参照。不同权重运行在独立 pipeline 输出目录，核对预测文件对应的 checkpoint SHA256，不能复用旧预测缓存。报告检测 F1、Completeness/Omission/Commission、实例 Precision/Recall/Coverage、TP/FN/有效 FP 整数计数和逐树 IoU。

L1W 和既有 grouping 已被反复用于开发时，这部分应叫固定协议下的开发/诊断比较，不声称新的盲测。Wytham 是原论文 mid 的另一个测试样地，本次不加入训练，可在核查完整训练历史后作为候选独立测试；如果缺标注、真实 RGB 或处理条件不一致，先说明限制，不随意改模型开关。

服务器应从上一轮实际运行配置创建各权重独立的 pipeline/evaluation 配置，显式写入各自 checkpoint、完整 model 参数、固定 grouping、forest_path、save_cfg.results_dir，以及评估的 paths.pred_forest_path/gt_forest_path。然后分别调用：

```bash
python tools/pipeline/pipeline.py --config <本次权重的服务器pipeline配置路径>
python tools/evaluation/evaluate.py --config <对应预测的服务器evaluation配置路径>
```

尖括号内容必须替换为服务器实际保存的配置路径，不能原样执行。当前 pipeline 会在输入文件旁写入居中后的同名 NPZ，并使用基于森林父目录的共享 tiles 缓存；给不同输入协议使用独立根目录，首次推理确认 tile_generation 和缓存来源。同一输入、采样和模型无关预处理已核验一致时可以顺序复用 tiles，但不能复用旧预测或并发改写同一缓存。

CULS2 如已加入 mid 训练，其成绩必须标注为 in-sample 诊断。若无可证明独立的样地，应报告验证与诊断成绩，并明确独立测试尚缺，不能挪用训练样地填充。

完整森林推理与 L1W eval_subset 直接推理是不同协议，分别命名。与旧结果比较时先严格保持同一协议，不能把不同输入点集下的数字混在同一提升幅度中。推理模型参数（尤其 RGB hidden width）与训练一致；不要盲目继承当前 modular 默认值。

## 8. 服务器交付

保存 `docs/results/finetune_mid_server_run.md`（存在则用新 run ID），包括 manifest、RGB审计、下载收据、数据生成配置和计数、原 checkpoint/训练历史、实际运行配置、更新次数、boundary有效权重、最佳 epoch、训练曲线、微调前后成绩、误检/漏检与逐树案例、日志及权重绝对路径。数值不得先填占位或预期提升。

服务器可修复实际代码/环境问题，但须记录每项修改；若先前 FOR-only 运行在服务器有未回传的修复，应先审查复用，避免用本地旧文件覆盖。数据和权重不提交 Git。任务跨会话时提供任务标识、日志路径和恢复命令，不把“已启动”当作“已完成”。

本地只提供并检查配置/指南，不代替服务器的 RGB数据检查、GPU试跑或最终实验结果。

## 依据

- [原论文 §2.3.2、表2与表3](https://arxiv.org/html/2309.08471v3#S2.SS3.SSS2)
- [FOR-instance 官方数据与原 split](https://zenodo.org/records/8287792)
- [扩展 LAUTx/Wytham 数据](https://doi.org/10.25625/QUTUWU)
- 本仓库 `tools/training/train.py`、`tree_learn/model/tree_learn.py`、`tree_learn/util/parser.py`、`tools/data_gen/gen_train_data.py`。
