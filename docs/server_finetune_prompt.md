# 给服务器上 Cursor 的任务提示词

请在服务器上完成这个 TreeLearn 仓库的**人工标注数据微调**，包括下载、数据核验与划分、预处理、训练、验证和结果报告。请直接执行任务；遇到环境或数据问题先排查并修复，不要只写计划。不要覆盖服务器已有的未提交改动、数据或 checkpoints，也不要把大数据和权重提交到 Git。

## 仓库与依据

1. 在现有仓库运行 `git status --short`、`git remote -v`，确认当前状态；安全地 `git fetch origin`，读取远端分支 `codex/treelearn-finetuning-guide` 上的 `docs/finetuning.md`、`docs/server_finetune_prompt.md`、微调配置和下载脚本。如果现有工作区有未提交改动或运行中的训练任务，保留它们，使用独立 worktree 或隔离目录执行。不要强制切分支或清理工作区。
2. 以 `docs/finetuning.md` 为主要操作依据，同时核查实际代码和论文。那份指南中的参数是起步配置；根据服务器 GPU 显存、环境和数据情况调整时，记录所有修改与原因。
3. 先报告 GPU 型号/显存、CUDA、PyTorch、spconv、Python、可用磁盘空间、仓库 commit 和现有可复用数据。优先使用已有兼容环境，避免执行会删除已有 conda 环境的 `setup/setup.sh`。如需新建环境，选择独立名称，安装与服务器 CUDA 匹配的 spconv。

## 数据下载与实验划分

4. 使用 `tools/data_gen/download_finetune_assets.py` 从 GRO.data 下载 TreeLearn 模型权重和需要的 LAUTx/Wytham/L1W 数据，逐个验证脚本保存的 MD5 和大小。FOR-instance 从 Zenodo 官方记录 `https://zenodo.org/records/8287792` 下载并核验 ZIP MD5；参考指南命令。若已有文件，先验校验并复用。检查该脚本的默认行为：只列目录；提供 `--files` 才下载。下载失败时保留 `.part` 证据、重试并报告，不把不完整文件当作有效数据。
5. 解压 FOR-instance，读取其中的 `data_split_metadata.csv`，列出**真实**目录层级、列名、原始 split 和每个样地的文件名。先确定正式训练/验证/测试样地。以完整原始样地为单位划分，不要把同一样地的 crops 分到不同 split。优先将论文使用的 NIBIO plot 16 留作验证，并从训练样地排除；如果数据包路径/命名不同，按实际元数据核实。保存 `split_manifest.csv`，记录数据集、源文件、原始 split、新用途、校验信息、是否曾被初始 checkpoint 使用。保留至少一个完全独立测试样地；若无法证明初始权重没接触测试样地，就明确只报告应用适配结果，不宣称独立泛化。
6. 本次优先微调 FOR-instance 人工标注数据。LAUTx/Wytham 可以用于后续扩展实验，但加入某数据集训练后，不要再将其作为独立测试。`model_weights_with_small_20241213.pth` 是默认应用适配起点，能覆盖小树；其历史可能已接触 L1W 和公开人工标注数据，所以 **L1W 只能作为诊断/复现检查，不能作为该权重的独立测试成绩**。若确有新的、未被初始化权重使用的测试数据，优先在其上做最终评估。

## 标注检查与预处理

7. 抽样读取每个来源的 LAS/LAZ，确认米制 XYZ、`treeID`、`classification`、实际实例数、树/非树/忽略点比例，以及 RGB 是否真实存在、是否有有效方差。核查 `tree_learn/util/data_preparation.py` 的读取规则：树 ID > 0，非树 classification 1/2，其他无树 ID 点忽略。发现标签冲突先修正映射或转换脚本，并记录转换前后的计数。不要把 RGB 常零的点云称为有颜色数据。
8. 建立 `data/finetune/train/forests`、`data/finetune/val/forest` 等分离目录；只放目标样地数据，不要混入文档、压缩包或其他类型文件。原始文件可用符号链接，必须保留源文件名或在 manifest 中记录映射。确认验证配置 `configs/data_gen/gen_finetune_val.yaml` 的 `forest_path` 指向实际独立验证样地；默认的 `validation.laz` 只是占位名称。
9. 先生成少量训练 crops 和验证 tiles，核对 `points`、`feat`、`colors`、`instance_label`、内区树/非树点比例、offset 监督是否有效及磁盘占用，然后再生成正式数据。当前训练 crop 配置初始为 2,000，不保证最终生成正好 2,000。若数据/参数变化，用新缓存目录或清理对应缓存并重生成。运行划分审计，并人工核对权重历史；审计不会发现预训练权重的样地泄漏。

## 训练与评估

10. 先检查 `configs/training/finetune_baseline.yaml` 和 `configs/training/finetune_full.yaml` 的模型配置与权重格式；从权重载入日志确认 `input_conv`、`unet`、两个 head 有合理覆盖率。full 的新增 RGB 参数缺失是预期情况；主干大量缺失不是。先跑 1–2 epoch 的短试验，确认损失有限、参数更新、GPU 显存安全、验证正常及 `best_offset.pth` 真正保存，再用正式配置训练。短试验用独立 `work_dir`，不要污染正式目录。
11. 正式运行 baseline；若 RGB 有真实信息，再运行 full。两组应使用相同训练/验证数据、同一初始权重、种子和更新预算。若 RGB 不可用，请关闭 RGB 并建立 geometry+boundary 的配置，与 baseline 比较；写明不能评估 RGB gate。需要调 batch/crops/轮数以适应显存时，让两组实际 optimizer 更新次数可比。出现 NaN、OOM 或权重不兼容，修复后重新试验，记录改动。
12. 选择最低验证 offset loss 的 `best_offset.pth`，不要默认选最后一轮。记录每组：训练曲线、最佳 epoch、验证 offset、checkpoint 路径、模型配置、权重来源、数据版本、样地名单、GPU 与运行时间。推理和评估时模型 RGB 开关应与训练一致；pipeline 的 `pretrain` 指向选出的权重，先在验证集上固定聚类参数，再一次性用于独立测试。报告 detection F1、实例 precision/recall/coverage、omission/commission，并附失败案例。如果只有 L1W 诊断数据，就报告诊断结果，并明确其被权重历史污染的限制。
13. 将服务器上的 `split_manifest.csv`、实际使用的配置副本、训练日志摘要、数据校验结果、最佳权重路径和最终评估写入一份 `docs/results/finetune_server_run.md`（如该目录已有同名文件，请另起名称）。大文件留在服务器并提供绝对路径，不要提交数据、训练输出或 checkpoints。最后清楚列出已经完成的步骤、尚未完成的步骤和具体阻碍；若运行超出当前会话时间，保持任务在服务器上可恢复，并提供进程/任务管理器标识及续训命令。
