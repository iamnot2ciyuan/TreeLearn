# 给服务器上 Cursor 的任务提示词

请执行我们改造后的 TreeLearn 模型微调：**初始化必须使用服务器上我们已训练好的 epoch1000.pth，完整保留其 RGB gate 权重**。仓库保存名可能为 epoch_1000.pth，请找到实际文件。请直接完成数据下载/核验、预处理、试跑、正式微调和结果报告。

## 1. 同步代码，核实已训练模型

- 安全地 fetch GitHub 仓库 iamnot2ciyuan/TreeLearn 的 `codex/treelearn-finetuning-guide` 分支，阅读最新 `docs/finetuning.md`、本文件和 `configs/training/finetune_full.yaml`。保留服务器已有改动、数据和运行中的任务；必要时使用独立 worktree。
- 查找我们已有的 epoch1000.pth / epoch_1000.pth，记录绝对路径、文件大小、SHA256、checkpoint epoch、来源实验和训练日志。多个候选时结合日志、原训练配置和 rgb_encoder/rgb_gate 参数确定目标，不要只按文件名随意挑选。
- 查找该 checkpoint 对应的原模型定义和原训练配置，核对 channels、num_blocks、dim_feat、use_coords/use_feats、rgb_hidden_channels、voxel_size、spconv 版本，以及 geometry/boundary loss 是否开启。当前 finetune_full 是候选模板，应以我们的实际原训练配置为准保持模型和 loss 设置一致。
- 初始化仅使用我们的 epoch1000。不要替换为官方 TreeLearn/SoftGroup 权重，不要重新初始化 RGB encoder/gate，不需要先训练关闭 RGB 的 baseline。
- 核查 GPU/显存、CUDA、Python、torch、spconv、磁盘和已有数据；复用原训练环境。不要执行会删除已有环境的 setup/setup.sh。

## 2. 下载并检查适合 RGB 模型的数据

- 优先考察 FOR-instance 人工标注数据；根据指南从官方 Zenodo 下载 ZIP，核验大小和 MD5。LAUTx/Wytham/TreeLearn 点云可用 tools/data_gen/download_finetune_assets.py 获取并校验，但不需要下载官方模型权重作为初始化。先复用并核验已有数据。
- 在大量生成 crops 前检查每个候选来源的实际字段：米制 XYZ、treeID/classification 或等价实例标签、真实 RGB、颜色范围/方差/覆盖率。数据集名字不保证有 RGB。必须保留 RGB gate；若数据无颜色，寻找其他带真实颜色和人工实例标签的数据，或核实已有可对齐的影像颜色来源。不能关闭 RGB、随机赋色或把补零当成完成 RGB 模型微调；若最终无法取得必要数据，明确报告具体缺项和已尝试来源。
- 核查我们 epoch1000 的原训练/验证样地、初始化来源及 checkpoint 选择过程，再划分新数据。按完整样地划分 train/val/test，不要将同一样地的 crops 随机分开。原始 metadata 中 NIBIO plot 16 可作为候选验证样地，但必须核查它是否已用于原模型训练。保存 split_manifest.csv 与颜色核验记录。L1W 是否独立应根据我们 checkpoint 的实际历史判断，不能只由官方权重历史推断。

## 3. 预处理与微调

- 将样地放在分离的 data/finetune/train/forests、val/forest、test/forest 目录，保持原始文件名并记录来源。修改 gen_finetune_val.yaml 中占位路径为真实验证森林。按代码核查标签映射，保留非树/未标注点及树基上下文。
- 使用 gen_finetune_train.yaml / gen_finetune_val.yaml，先少量生成，检查 points、feat、colors、instance_label 行数对齐、内区监督、offset 有效性和颜色统计，再生成正式数据。保留 0.1 m 下采样、35 m 上下文及 8 m 监督内区，与原模型设置核对。数据/参数变化后不要复用不匹配缓存。
- 将 finetune_full.yaml 的 pretrain 写成核实后的 epoch1000 绝对路径，保留 require_pretrain: true、pretrain_strict: true、use_rgb_gate: true。严格核对主干、heads、rgb_encoder、rgb_gate 的全部参数名称与形状。加载失败先恢复匹配的代码/配置或核实必要格式转换，不能靠 strict=False 跳过已训练参数。
- **首次微调用 pretrain，不要用 --resume epoch1000.pth**。本次重新建立低学习率优化器，微调 epoch 从 1 开始；resume 仅用于本次微调过程中生成的 checkpoint。使用新 work_dir，保留原 epoch1000。
- 全模型低学习率微调，建议先采用 lr=1e-5、lr_min=1e-6、3 epoch 学习率 warmup、FP32、梯度裁剪 1.0、50 epoch、每轮最多 500 crops。与原学习率、GPU显存和有效数据量核对后可调整，记录原因。geometry/boundary 的权重及 warmup 保持与原设置一致或明确记录经验证的调整。
- 先独立目录跑 1–2 epoch，确认严格加载成功、loss 有限、backbone/head/RGB gate 参数有更新、颜色传递正常、验证和 best_offset.pth 保存正常，再正式训练。故障先排查修复，记录 OOM/NaN/权重转换及修复。保留同一验证集上 epoch1000 的微调前指标作参照。

## 4. 评估与交付

- 使用验证 offset loss 最低的 best_offset.pth，并用相同数据与推理/聚类设置对比 epoch1000 与微调后权重。测试数据不能参与微调、选 checkpoint 或调参；无法证明独立性时明确描述限制。
- 报告实际配置、数据版本/划分、RGB统计、初始 checkpoint SHA256、加载检查、微调前后验证 offset、最佳 epoch、checkpoint 绝对路径、GPU耗时，以及可执行的推理/评估命令。若有独立标注测试样地，再报告 detection F1、实例 precision/recall/coverage、omission/commission。
- 将结果和运行记录写入 docs/results/finetune_server_run.md 或不覆盖已有文件的新报告。数据和 checkpoints 留在服务器，不提交 Git。若训练跨会话运行，留下任务/进程标识、日志和续训命令；不能只启动任务就宣称微调完成。
