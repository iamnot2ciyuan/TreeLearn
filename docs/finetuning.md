# 当前仓库的 TreeLearn 微调方案

核查日期：2026-10-12。依据原论文、官方数据说明，以及本仓库实际代码。文中的推荐配置是应用适配起点，尚未经过本项目 GPU 实验验证。

## 1. 先确定做哪一种微调

原版 `docs/training.md` 是用自动标签训练基础模型的指南：SoftGroup/HAIS 权重 → 自动分割森林数据 → TreeLearn。真正的树木模型微调是：**已训练的 TreeLearn checkpoint → 人工标注森林 → 重新优化语义和 offset 预测**。不需要重新设计网络或重新训练全部基础数据。

本仓库已经改过模型：有 RGB gate、同树几何一致性 loss、跨树 boundary push；原版说明没有覆盖这些改动。尤其现有 A0 配置仍继承 `use_geometry_constraint: true`，因此只是关闭 RGB/boundary 的内部对照，不能直接称为严格原版基线。

建议分两条路线：

- **应用适配（先做）**：从 `model_weights_with_small_20241213.pth` 出发，在新的人工标注样地上微调。适合需要识别小树的任务；若只考虑 >10 m 树，可用 `model_weights_20241213.pth`。这两个发布权重已有训练历史，测试集必须核查是否被初始化权重用过。
- **论文实验**：所有对照从同一未接触最终测试样地的基础权重出发。官方 `model_weights_dirty.pth` 可作为候选，但它有旧版去噪预处理历史，且主要识别 >10 m 树。严格复现时需匹配预处理；若无法确认干净的初始化来源，则用 18 个自动标注样地重新做基础训练。不能把已在 L1W 上微调的发布权重用于宣称 L1W 独立泛化。

官方数据 ReadMe 明确记录：旧 `model_weights_finetuned` 在 benchmark 上微调；diverse 从该权重继续微调；20241213 又从 diverse 继续微调并去掉统计离群过滤。原版 `docs/segmentation_pipeline.md` 也说明 L1W 已参与默认 checkpoint 训练。论文最终版的 noisy/small/mid/large 实验设计与这些发布文件的沿袭历史应分开记录。

## 2. 论文做法和本次起步配置

论文 §2.3.2 使用人工标注的 FOR-instance、LAUTx，以及视实验设置加入的 Wytham。small 使用 42 个 FOR-instance 训练样地；mid 加入该数据其他 split 和 LAUTx；large 再加入 Wytham。Wytham 作为测试集时不能用 large 设置。

论文微调训练 12,500 次 optimizer 更新，AdamW、batch size 2、weight decay 0.001、余弦学习率 5e-4 → 5e-5、无学习率 warmup；使用独立 NIBIO plot 16 验证并按最低 offset loss 选权重。论文生成总共 30,000 个微调 crops。

我们新增的独立配置使用保守起点：50 epoch、每轮最多 500 crops、batch size 2、lr 1e-4 → 1e-5、3 epoch 学习率 warmup、FP32、裁剪阈值 1.0。若训练样本不少于 500 个，每轮 250 次更新，50 轮约 12,500 次更新；若样本不足，实际更新数会减少。它是适配方案，不是论文参数复现，也不保证最优。

- `configs/training/finetune_baseline.yaml`：RGB、geometry、boundary 全关闭，作为当前代码内的几何模型对照。
- `configs/training/finetune_full.yaml`：三项改进开启。boundary 沿用当前仓库参数，15 epoch warmup；这些数值还需独立验证集验证。学习率 warmup 与 boundary loss warmup 是不同机制。

需要严格按论文学习率实验时，将 `lr` 改为 0.0005、`lr_min` 改为 0.00005、`warmup_t` 改为 0。保留当前 50 轮更新预算。复现还需对齐数据、预处理、模型和聚类，不能仅改学习率。

## 3. 数据下载

### 3.1 FOR-instance：优先用于人工标注微调

官方 [Zenodo 8287792](https://zenodo.org/records/8287792) 提供约 1.6 GB ZIP，并在包内 `data_split_metadata.csv` 给出划分。先下载并解压：

```bash
mkdir -p data/downloads/FORinstance
curl --fail --location --retry 3 --continue-at - \
  'https://zenodo.org/records/8287792/files/FORinstance_dataset.zip?download=1' \
  --output data/downloads/FORinstance/FORinstance_dataset.zip
echo '01fd03ba4068648e21fb4591fbff34c4  data/downloads/FORinstance/FORinstance_dataset.zip' | md5sum -c -
unzip data/downloads/FORinstance/FORinstance_dataset.zip -d data/downloads/FORinstance
```

上述校验命令用于 Linux 训练机。读取包内 CSV 后，按完整样地分配；不要随机划分已经生成的 crops。论文路线单独保留 NIBIO plot 16，且排除在训练列表之外。实际包内相对路径和 CSV 列名尚未下载核对，因此这里不提供猜测列名的自动复制脚本。移动样地时保留 `split_manifest.csv`，记录源文件、数据版本、原 split、实际用途和重命名映射。跨区域/跨采集批次测试比同林分随机划分更有解释力。

### 3.2 TreeLearn / LAUTx / Wytham

新增 `tools/data_gen/download_finetune_assets.py`：从平台当前元数据解析文件 ID，校验文件大小和 MD5，并保存下载收据。仅列目录时不下载：

```bash
python tools/data_gen/download_finetune_assets.py --dataset VPMPID
python tools/data_gen/download_finetune_assets.py --dataset QUTUWU
python tools/data_gen/download_finetune_assets.py --dataset VPMPID \
  --files model_weights_with_small_20241213.pth ReadMe.txt --root data/model_weights
```

LAUTx 的六个原分辨率样地可以这样获取。下面先放在下载区，随后根据实验划分挑选训练样地，不能未经核查全部放入训练目录：

```bash
python tools/data_gen/download_finetune_assets.py --dataset QUTUWU \
  --files p1.laz p2.laz p4.laz p10.laz p12.laz p14.laz --root data/downloads/LAUTx
python tools/data_gen/download_finetune_assets.py --dataset QUTUWU \
  --files wytham.laz --root data/downloads/Wytham
python tools/data_gen/download_finetune_assets.py --dataset VPMPID \
  --files L1W.laz L1W_voxelized01_for_eval.laz evaluated_trees.txt --root data/downloads/L1W
```

如果要做新的自动标签基础训练，原有命令仍可用于 18 个样地：

```bash
python tree_learn/util/download.py --dataset_name automatically_segmented_data --root_folder data/train/forests
```

核查发现原下载器的 diverse 文件 ID 写成 `1JMEQV`，而平台当前 persistent ID 为 `C1AHQW`。新下载器通过元数据避开固定 ID 过期；旧下载器暂未整体改写。

## 4. 数据格式与标注

必须同时有树实例标签和非树监督，只给分类标签或裸 XYZ 不能完成这项微调。

- LAS/LAZ：当前读取器要求 `treeID` 和 `classification`，字段名按代码大小写匹配。`treeID > 0` 表示树实例；`classification` 的 1/2 被视作地面/非树；无 treeID 且不属于 1/2 的点设为忽略。平台 TreeLearn 数据使用 classification 2=非树、3=未标注、4=树。检查不要给同一点同时写非零 treeID 和非树分类，否则当前读取器会覆盖为非树。
- NPY：优先明确使用四列 `[x, y, z, instance_label]`，或七列 `[x, y, z, instance_label, r, g, b]`。**不要使用七列 RGB 在前、标签在后的格式**：当前启发式会把整数 R 值误认成标签。TXT 读取还有表头推断问题，建议自定义数据转为 NPY/NPZ。
- 原森林 NPZ：`points: (N,3)`、`labels: (N,)` 或 `instance_label: (N,)`、可选 `colors: (N,3)`。
- 原始实例标签：`-1` 未标注、`0` 非树、正整数为各棵树；同一森林中一棵树必须使用同一 ID。不同样地可以重复 ID。
- 坐标以米为单位；保留地面、低植被与边缘上下文。训练 crops 内 XY 会居中；不要仅保存树冠或裁掉树基。offset 标签由 crop 内每棵树最低约 0.5 m 点的均值生成，缺失树基会使监督目标错误。

**语义编码不要弄反**：`TreeDataset` 会从 `instance_label` 动态生成语义，网络内部树=0、非树=1。生成的 crops 不需要 `semantic_labels` 字段。现有 `check_semantic_labels_npz.py` 要求该字段，且说明的编码相反，不适合检查当前生成器的输出。

RGB 支路不能靠补零验证效果。LAS/LAZ 有 red/green/blue 时按 8/16 bit 数值归一化；没有颜色时流程会补零，网络仍能运行，但无法证明 RGB 提供了信息。首先统计真实颜色覆盖率、方差和归一化范围，再决定是否运行 full/RGB 消融；若只有 XYZ，就先跑 baseline 或关闭 RGB 的 geometry+boundary 版本。颜色也要与 points 同行对齐。

## 5. 划分及预处理

目录约定如下，训练目录直接放样地文件，避免额外文档和子目录（生成器会遍历目录所有条目）：

```text
data/finetune/train/forests/         人工标注训练样地
data/finetune/val/forest/            独立验证样地
data/finetune/test/forest/           独立测试样地
data/finetune/train/random_crops/npz/  生成的训练 crops
data/finetune/val/tiles/npz/           生成的验证 tiles
```

将验证样地路径写入 `configs/data_gen/gen_finetune_val.yaml`。默认 `validation.laz` 只是待放置的文件名，代码不会自动选样地。若重命名 NIBIO plot 16 为该名称，必须保留映射；建议直接将配置改为原文件名，以便划分审计识别来源。多验证样地需逐一生成 tiles 到同一验证根目录，配置中的每个 `forest_path` 应为 `data/finetune/val/forest/<原文件名>`。

```bash
python tools/data_gen/check_forests_files.py data/finetune/train/forests
python tools/data_gen/gen_train_data.py --config configs/data_gen/gen_finetune_train.yaml
python tools/data_gen/gen_val_data.py --config configs/data_gen/gen_finetune_val.yaml
```

训练预处理沿用：0.1 m 下采样 → verticality（0.6 m 邻域）→ occupancy 图 → 按覆盖面积分配随机 35×35 m crops。全 crop 提供上下文，只在中央 8×8 m 计算监督。验证用 8 m 内区加每边 13.5 m 上下文，总宽约 35 m。**stride 是内区边长的比例，不是米**：1 大约对应 8 m 步长；0.5 大约对应 4 m 步长。新验证配置使用 1，与原验证指南一致。

新训练生成配置先设 2,000 crops。这是试跑上限，occupancy 拒绝候选后实际数量可能更少。原指南提示 25,000 crops 约需 700 GB，实际随点密度和存储压缩变化；先测少量 crops 占用再扩容，不要直接生成数万份。生成器仍计算 verticality，即使网络 `use_feats: false`；该特征也用于下游聚类。

更换原始标注、颜色或采样参数时，使用新数据根目录或清理对应旧缓存后重生成。当前缓存存在检查不验证全部参数，重复运行也可能遗留旧 crops。训练验证目录只能放有效 NPZ，因为 `TreeDataset` 未过滤其他文件。

划分审计可复用：

```bash
python tools/data_gen/audit_dataset_splits.py \
  --train-config configs/training/finetune_baseline.yaml \
  --train-data-config configs/data_gen/gen_finetune_train.yaml \
  --val-data-config configs/data_gen/gen_finetune_val.yaml \
  --fail-on-high-risk
```

以上仍默认读取现有 L1W pipeline/evaluation 配置；正式测试时须同时传入你自己的 `--pipeline-config` 和 `--eval-config`。审计只能发现部分显式来源重叠，不能证明预训练权重未接触测试数据；保留 provenance 是必要的。

## 6. 微调脚本怎么运行

继续用现有 `tools/training/train.py` 即可。`pretrain` 只加载模型，优化器和 epoch 从头开始，这正是微调所需；`--resume` 还恢复优化器和 epoch，用于同一实验断点续训。

```bash
python tools/training/train.py --config configs/training/finetune_baseline.yaml
# 有真实 RGB 后运行改进模型
python tools/training/train.py --config configs/training/finetune_full.yaml
```

输出分别在 `work_dirs/finetune_baseline` 和 `work_dirs/finetune_full`。新增 `save_best: true` 后每次验证保存最低验证 offset loss 的 `best_offset.pth`，不用盲选 epoch_50。TensorBoard 仍记录 `val/Offset_MAE`，但实际计算的是平均三维 L2 offset 误差，不能将名称当作逐坐标 MAE。

```bash
python tools/training/train.py --config configs/training/finetune_full.yaml \
  --resume work_dirs/finetune_full/epoch_25.pth
tensorboard --logdir work_dirs
```

现有 resume 没保存 AMP scaler、随机数和 scheduler 完整状态，因此不是逐步精确重放。当前 scheduler 根据 epoch 重建；FP32 微调不依赖 scaler 状态。续训时保持同一实验目录；best checkpoint 的历史比较从该目录已有 `best_offset.pth` 恢复。若从较早 checkpoint 分叉新实验，使用新的 `--work_dir`，否则旧实验的 best 可能继续保留。

### 权重兼容检查必须看日志

当前加载器读取 checkpoint 的 `net`，`strict=False`，遇到尺寸不匹配会跳过。官方几何权重加载到 full 时，`rgb_encoder`、`rgb_gate` 缺失是预期的：新支路按模型代码初始化。geometry/boundary 是损失约束，没有需要从 checkpoint 读取的新参数。

但如果 `input_conv`、`unet`、`output_layer` 或语义/offset head 大量缺失或尺寸不匹配，应停止试跑，核对 checkpoint、spconv 排列、channels、num_blocks 和输入维数。不能看到训练开始就以为成功迁移。不要随意启用 use_coords/use_feats；即使形状兼容，输入含义改变也会影响预训练适配。本次尚未下载权重，不能保证每个历史发布文件都与当前 spconv 完全兼容。

### 是否冻结骨干

先全网络低学习率微调。数据极少时可另做冻结实验：`fixed_modules: [input_conv, unet, output_layer]`，先学习 head/RGB，再解除冻结全量训练。冻结状态也会让这些模块的 BatchNorm 保持 eval。第二阶段仍通过 `pretrain` 指向第一阶段 checkpoint，创建新优化器；不要以冻结参数组的 optimizer 去 resume 解冻实验。分阶段增加了变量，首轮不建议采用。

## 7. 开跑前检查与最终评估

先少量 crops、1–2 个 epoch 试跑（同时缩短 scheduler 周期和 warmup），确认：有监督的内区树点、非树点和正常 offset；loss 有限；参数确实更新；训练/验证/测试的原始来源无重叠；full 的 RGB 不是全零。正式消融用同一数据、初始化、种子和更新预算，至少比较几何 baseline、仅 RGB、仅 boundary、完整模型。geometry 应单独控制；若全组固定开启 geometry，则明确叫作加入 geometry 的内部对照。

训练器现在修正了 AMP 梯度裁剪顺序：先 `scaler.unscale_(optimizer)` 再 clip，避免裁剪被缩放后的梯度。本次微调默认 FP32；原 train.yaml 的 `fp16: True` 与旁边“关闭半精度”的注释矛盾，应以 YAML 值为准。

最终评估复用原 pipeline 和 evaluate，但创建自己的配置，将 pipeline 的 `pretrain` 改为选出的 `best_offset.pth`，模型开关与训练一致，`forest_path` 指向测试森林；evaluation 的 `paths.pred_forest_path`、`paths.gt_forest_path` 指向对应预测与 GT。推理不需要开启 loss 约束来改变 forward，但 RGB 开关必须匹配权重。先固定聚类参数，所有方法一视同仁；若调参，只在验证集做。

L1W 输入优先完整 `L1W.laz`（或保留完整上下文的 voxelized 版本），最终 GT 用 `L1W_voxelized01_for_eval.laz`，评估限定完整标注树。直接拿 for_eval 文件作模型输入会改变上下文条件。记录 detection F1、实例分割 precision/recall/coverage、omission/commission，以及模型初始化、选权重和调参是否接触过该测试样地。

## 8. 当前完成与待验证项

已阅读原模型/训练/数据/推理/评估文档，并核对实际读取器、生成器、loss、加载器和训练循环；已联网核查两个 GRO 数据集的文件元数据与 ReadMe。当前本地 `data` 只有服务试验输入，没有训练 crops 或预训练权重；系统 Python 也没有 torch/numpy/spconv。**本次提供脚本与配置并做静态/轻量检查，没有下载 GB 级点云，没有执行 GPU 训练，权重形状、RGB 有效性、实际 crop 数量和收敛仍待训练机验证。**

下一步顺序：准备 Linux NVIDIA CUDA 环境 → 选目标样地及干净测试集 → 下载权重与人工标注数据 → 保留划分清单 → 生成少量 crops → baseline 试跑 → 有真实 RGB 时运行 full → 验证集选 checkpoint → 独立测试。

## 来源

- [原论文 v3，尤其 §2.3](https://arxiv.org/html/2309.08471v3#S2.SS3)
- [原版 training guide](https://github.com/ecker-lab/TreeLearn/blob/main/docs/training.md)
- [TreeLearn 官方数据](https://doi.org/10.25625/VPMPID)、[数据 ReadMe](https://data.goettingen-research-online.de/api/access/datafile/118510)
- [扩展 LAUTx / Wytham 数据](https://doi.org/10.25625/QUTUWU)
- [FOR-instance 数据、标签与 split](https://zenodo.org/records/8287792)
- 仓库：`tree_learn/model/tree_learn.py`、`tree_learn/dataset/dataset.py`、`tree_learn/util/data_preparation.py`、`tree_learn/util/train.py`、`tools/training/train.py`。
