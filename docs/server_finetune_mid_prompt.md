# 服务器 Cursor：从我们 epoch1000 开始的 mid 微调

请直接在服务器执行本任务，完成数据下载、mid 划分、RGB/标签核验、预处理、试跑、正式微调与报告。保留已有数据、权重、本地代码修复和运行中的任务，不要只给出计划。

1. 安全 fetch `iamnot2ciyuan/TreeLearn` 的 `codex/treelearn-finetuning-guide`，读取最新 `docs/finetuning_mid.md` 和本文件。检查当前 Git 状态；有本地改动时使用独立 checkout/worktree 或审查后有选择地应用新增 mid 文件，不能覆盖上一轮服务器修复。复用原训练环境，检查 GPU/CUDA/spconv/磁盘，不运行会删 conda 环境的 setup/setup.sh。

2. 初始化使用已经明确的 `train_20260627/epoch_1000.pth`（仓库相对候选 `work_dirs/train_20260627/epoch_1000.pth`）。核实服务器实际绝对路径、SHA256、epoch、原配置与模型代码。完整保留 RGB encoder/gate、backbone 和 heads，保留 boundary push；geometry 和其他设置按原训练记录核对。不要用官方权重、epoch720 或上一轮第2轮 best_offset 替代。本次从相同 epoch1000 独立开始，以便比较扩大数据的影响。

3. 按论文 mid 数据范围建立 manifest：FOR-instance 的全部原 train/val/test 样地，**排除 NIBIO plot16**，加扩展 LAUTx 六样地 p1/p2/p4/p10/p12/p14。NIBIO16 单独验证，L1W/Wytham 不加入训练。复用已验证的 FOR ZIP 或从官方 Zenodo 8287792 下载、校验 MD5。LAUTx 使用 `tools/data_gen/download_finetune_assets.py --dataset QUTUWU --files p1.laz p2.laz p4.laz p10.laz p12.laz p14.laz --root data/downloads/LAUTx_mid` 下载并校验。不要同时重复使用同一样地的原分辨率和voxelized版。

4. 读取包内实际 metadata CSV，不猜字段或路径。记录原 split、新用途、原绝对路径、唯一 plot ID、校验和、RGB来源、原模型接触历史。确认 NIBIO16 未用于 epoch1000 训练；否则先报告协议问题。核对 CULS plot2 的实际映射：完整 mid 若将其纳入训练，它就不能再当独立测试。若另行留出，必须将实验命名为 mid-minus-CULS2 并说明偏离，不能静默改划分。

5. 在大量生成前逐来源核查人工实例标签、树/非树/忽略编码、米制坐标、完整森林上下文和真实 RGB 的覆盖/方差/范围/点行对齐。数据名称不保证有 RGB。必须完整保留已训练 RGB gate，不得随机赋色、拿 intensity 伪造 RGB 或静默补零后声称完成 RGB-mid。某必需来源缺颜色时，先核实可配准真实颜色；若无法满足，报告具体缺项，不静默删样地或关 gate。优先阅读上一轮运行的 RGB审计及实际修复。

6. 使用隔离目录 data/finetune_mid 与 data/finetune_mid_pilot。pilot 与正式训练原森林清单一致、缓存分离；val/forest 只有 NIBIO16。修改 `gen_finetune_mid_val.yaml` 为实际文件名和扩展名。先用 `gen_finetune_mid_pilot.yaml` 请求2000 crops、生成验证tiles，核验有效内区监督、offset、colors和磁盘占用。容量充分后用 `gen_finetune_mid_train.yaml` 请求30000正式crops。固定并记录生成随机种子（现有生成器不自动读取训练 seed），保存实际每样地crop计数；它按occupancy面积采样而非数据集等量。不要把pilot混入正式数据，不复用旧FOR-only缓存。

7. 使用 `configs/training/finetune_mid.yaml` 创建服务器配置副本，pretrain填写核实后的epoch1000绝对路径，严格加载所有参数，保留require_pretrain/pretrain_strict和RGB开关。默认 lr=1e-5→1e-6、学习率warmup3、batch2、50epoch、每轮最多500crops、FP32、clip1.0，每轮验证。主实验boundary warmup=15，与上一轮一致；它在epoch2有效权重约0.0233、epoch16起0.35，报告实际日志。不要同时默默改变loss调度与数据范围。结构/损失与原训练不一致时先核对修正并记录原因。

8. 按指南生成独立的1–2epoch smoke配置，检查严格加载、loss有限、backbone/head/RGB参数确实更新、真实颜色进入模型、验证和best保存正常。正式训练重新从epoch1000启动，绝不能从smoke或旧best_offset开始；首次使用pretrain，不要 `--resume epoch_1000.pth`。使用新的run目录，通过现有tmux/调度器持久运行并保存日志。resume仅用于本次mid自身checkpoint。不得修改训练预算/采样数量却不记录实际更新次数。

9. 在NIBIO16上记录epoch1000的微调前offset，并以最低验证offset选mid best_offset。不要用L1W、Wytham或CULS结果倒选epoch或调参。用固定L1W eval_subset+原grouping（CSE=0.5）比较epoch1000、之前FOR-only best、mid best，epoch720作额外参照；每个权重单独预测目录、核查配置与SHA256，避免读取旧结果。报告detection F1、completeness/omission/commission、实例precision/recall/coverage、TP/FN/有效FP计数及逐树IoU变化。Wytham未参加mid训练，可在核查原模型历史和数据RGB条件后作另一个候选测试。已经参与开发的L1W标为诊断；CULS2若参与训练标为in-sample，不得声称独立泛化。

10. 保存 `docs/results/finetune_mid_server_run.md` 或带run ID的新报告，包含下载收据、manifest、RGB审计、实际配置、crop计数、初始化来源、训练曲线与boundary有效权重、最佳epoch、指标、失败案例、服务器日志和checkpoint绝对路径。数据和权重留在服务器不进Git。遇到故障先排查修复并记录；如果当前会话无法覆盖完整训练，提供任务标识、日志和续训命令，并如实区分已完成、运行中和受阻步骤。

本任务对齐论文mid的数据范围，使用我们改造后的模型与低学习率，不声称原版论文参数/结果已复现。不要把扩大数据范围自动等同于性能必然提升。
