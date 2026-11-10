# Patch-STR 服务器复核报告（2026-11-11）

**结论：REJECT。** 服务器已于 2026-10-08 完成与本次要求相同的固定 200/100 Patch-STR 实验。本次重新执行服务器端验证与统计，结果与归档一致；没有将旧训练冒充新训练，也没有重复启动正式训练。

实验源项目为 `/home/zyt/deep_final`，正式产物位于 `/home/zyt/deep_final/artifacts/patch_str_20261008/`，发布镜像为 `/home/zyt/pureDeep-github-upload`。所有模型测试、训练集 normalization 重拟合、validation 推理和统计重算均在服务器进行。本机只通过 SSH 调度并保存脚本、日志及聚合报告；没有取回原始数据、clinical 个体表、checkpoint、缓存或个体预测。

## 本次实际验证

| 阶段 | 本次服务器端操作 | 结果 |
| --- | --- | --- |
| Repository / freeze | 校验 21 个冻结文件、原 STR source/config/split/checkpoint/prediction 哈希；源项目与发布镜像比对；核验六个阶段 tag 和 preformal commit 内容 | PASS |
| P0 identity | 45 个 checkpoint 全 validation 重新推理；15 次 train-only normalization 重拟合；与当前 source 的 Phase2 重训模型逐 tensor 比对 | PASS，最大概率差 1.11×10⁻¹⁶ |
| Extraction | 数量、start、first/last、tail、无重复、越界检查、padding 隔离、确定性、autograd | PASS；976→9、2000→19 patches |
| P1 / P2 correctness | 各 45 个 P0 checkpoint 的 zero-init/reload；三 seeds 的原参数和 CPU/CUDA RNG；共用 patch 参数初始化；两次 backward/update；absent wrist/activity、padded samples/patches 隔离 | PASS |
| Smoke | 使用原 stage auditor 复核两份已有 one-batch/one-epoch smoke 产物 | PASS；本次未重训 smoke |
| Trained candidate inference | P1/P2 全部 90 个正式 best checkpoint 严格加载并全 validation 推理，逐 subject 在服务器内对齐概率、标签、预测和六指标 | PASS，最大概率差 1.11×10⁻¹⁶ |
| Paired analysis | 原冻结分析代码重新审计 90 个候选 run，重算 135 行指标、seed-first 15 单元、全部配对统计与保留判定；输出在内存中与归档逐项比较 | PASS，仍为 REJECT |

所有验证前后均重新检查冻结哈希。本次写入函数被替换为内存比较，未覆盖历史 JSON/CSV；没有修改服务器正式代码、权重或协议。P0 审计原脚本的“已有审计禁止覆盖”检查仅在此只读封装中跳过，所有实质验证断言保留，所有输出写入均被拦截为比较。没有新训练 recipe、结构、统计规则或保留门槛变更，没有访问 outer-test 信号、预测或性能。

本次使用服务器已有 `mfam_analysis` 环境，没有安装 Skill、插件或依赖。最初探测的 `mfam` 环境缺少 pandas，切换既有环境后全部复核通过。

## 基线与历史审查

正式 STR-01 的 temporal、norm-free Learned-Moment、64-D wrist、258-D bilateral fusion、activity-ID attention 和固定 11×16 structured residual readout 均保留。原 temporal dilation 为 `[1,2,4]`。本轮 patch 分支仅在原 wrist token 上增加 zero-init residual，不替换原路径。

输入沿用既有 L1 preprocessing、trim48、6-channel Acc+Gyro 与 `[11,2,6,1]` train-fold normalization。当前数据集记录的是 `activity_lengths[B,11]`：loader 要求有效双腕 shape 一致，因此同一 activity 的真长度适用于其有效双腕；subject forward 按真长度 slice 后仅把有效腕送入 encoder。不存在从 batch padding 构造 patch 的路径。

P0 六项归档值与用户给出的近似值一致。早期 whole-tree source hash 与当前 source 不同，因此不声称历史整棵源码相同；已通过当前 source 的 Phase2 45 次重训 state 精确一致、45 个 P0 完整 validation 重算及 train-only norm 精确一致建立 matched 可复现性。

已核对历史 Frequency-Prior 的 REJECT、gyro rFFT 与 long-dilation 停止记录、Phase2 16 个单因素训练/augmentation 对照。总体 retained best 仍是原 Frozen WSSL-STR（BA 0.719644、AUROC 0.759553），activity scaling 为 INCONCLUSIVE、不保留。本实验仍只比较 STR，不叠加 WSSL 或 Frequency。

**避免重复的审计结论：完全相同的 Patch 实验确实已经做过。** 历史完整执行了 P1/P2 各 15 splits × seeds 42/43/44，共 90 次 scratch training；本次采用验证与复用，而不是再次训练同一配置。

## 六指标

每个 split 先对三个 seed 的指标取均值，再对 15 个 split 等权汇总；不是概率 ensemble。

| 条件 | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.730718 | 0.697189 | 0.717631 | 0.686068 | 0.777864 | 0.616515 |
| P1 | 0.728773 | 0.696424 | 0.720416 | 0.684746 | 0.774540 | 0.618308 |
| P2 | 0.730912 | 0.696581 | 0.721293 | 0.686044 | 0.779374 | 0.613789 |

## 三项主要 BA 比较

差值为比例，乘 100 才是百分点。95% CI 为冻结 seed=20261008、10,000 次 paired split bootstrap。Wilcoxon 为双侧，主要 BH family=3。

| 比较 | mean ΔBA | 胜/平/负 | bootstrap 95% CI | paired dz | rank-biserial | Wilcoxon p | BH q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1−P0 | −0.000765 | 8/0/7 | [−0.004240, +0.002423] | −0.112078 | −0.100000 | 0.761536 | 0.761536 |
| P2−P0 | −0.000608 | 5/2/8 | [−0.002851, +0.001566] | −0.134194 | −0.098901 | 0.753079 | 0.761536 |
| P2−P1 | +0.000157 | 5/0/10 | [−0.002495, +0.003238] | +0.026964 | −0.100000 | 0.733220 | 0.761536 |

15 个划分共享 subjects 和 training pools，只能描述 repeated development robustness，不能把 45 runs 当独立样本，也不能当独立外部推断。

## AUROC 与 recall

| 比较 | mean ΔAUROC | 胜/平/负 | bootstrap 95% CI | Wilcoxon p | 探索性 BH q（18 比较） |
| --- | --- | --- | --- | --- | --- |
| P1−P0 | +0.002785 | 8/0/7 | [−0.001252, +0.007276] | 0.302795 | 0.778617 |
| P2−P0 | +0.003662 | 11/0/4 | [+0.000435, +0.007246] | 0.072998 | 0.778617 |
| P2−P1 | +0.000876 | 7/0/8 | [−0.000975, +0.002843] | 0.524475 | 0.806332 |

P2−P0 的 AUROC bootstrap 区间为正，但 Wilcoxon/BH 不支持同等结论，两类证据分别保留；AUROC 不能替代预注册主要 BA 门槛。P2−P0 的 PD Recall +0.001510、DD Recall −0.002726；两 Recall 护栏通过，但不能据此宣称同时改善两类。Macro-F1 Δ=−0.000024，幅度极小，违反冻结的非下降门槛；不能解释为临床上明显恶化。

## 稳定性、epoch、参数与耗时

| 条件 | BA seed-mean SD | BA mean within-split seed SD | BA split SD | AUROC mean within-split seed SD | seed-pair prediction agreement | seed-pair probability Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | 0.011699 | 0.030011 | 0.031579 | 0.032350 | 0.762814 | 0.600027 |
| P1 | 0.014022 | 0.031382 | 0.031809 | 0.029649 | 0.765304 | 0.600725 |
| P2 | 0.015624 | 0.030743 | 0.030864 | 0.030520 | 0.765329 | 0.604629 |

| 条件 | 参数量 | best epoch 均值 | stop epoch 均值 | 历史训练+预测 wall time/次 | 45 次 wall time 之和 |
| --- | --- | --- | --- | --- | --- |
| P0 | 75,524 | 13.49 | 25.49 | 本次未重训 | 本次未重训 |
| P1 | 80,340 | 13.56 | 25.56 | 19.14 秒 | 861.16 秒 |
| P2 | 80,661 | 13.80 | 25.80 | 20.17 秒 | 907.55 秒 |

耗时来自服务器历史逐 epoch/run 日志，本次重新核对。原 `timing_summary.csv` 中名称含 `total` 的列是“每 run 总时长再取 45 run 均值”，不是整个条件总耗时；本报告明确区分单次均值与 45 次之和。控制器历史总历时约 30 分 12 秒，另含调度和审计开销。P0 原硬件/runtime 与候选不保证严格配对，不能据此作严格速度优劣结论。

三条件均在 best 后出现 validation CE 上升和 BA 下降；没有证据表明 patch 消除了后期退化。没有更改 max epochs 或 patience。online train loss/BA 包含 dropout，不能据此计算 eval-mode 泛化 gap。完整逐 seed 六指标、全部稳定性指标、best/last 轨迹及同 seed 跨模型一致性见原完整报告。

## Q1–Q4 与停止决定

1. **Q1：Patch representation 是否优于 STR？** P1−P0 未建立稳定增量；BA 均值略负，CI 跨 0。
2. **Q2：显式 patch order 是否提供额外价值？** P2−P1 未建立稳定增量；只有 5/15 胜、10/15 负。P2 同时增加顺序卷积、attention pooling 和 321 参数，本身也不是纯 order/capacity 因果对照。
3. **Q3：是否正式保留？** **REJECT**。冻结规则重算未变；P2−P0 BA 非正、胜数不足、CI 跨 0、主要 BH 不通过，F1 亦未满足严格非下降。拒绝保留不等于统计证明有害。
4. **Q4：失败属于哪种？** 当前固定 local patch 配置未建立稳定增量；不属于“已证明 patch 有用、仅 order 无额外作用”。

停止 P3、WSSL+Patch、patch/stride/hidden/pooling rescue；不执行未满足条件的 attention、order shuffle、DD subtype 分析，不重新开启 Frequency。只能说：**当前固定 2s / 1s stride Patch-STR implementation 未建立稳定增量价值。** 不能推出所有 patch representation 或长时序建模对 PD/DD 无用。

## Git 与交付证据

六阶段服务器本机 commit/tag 均核验通过：audit `849a5446`、extraction `90915de4`、P1 `60fcfb2c`、P2 `918ffd31`、preformal `b06f166a`、final `8abab727`。preformal Git 内容与 21 文件冻结 SHA 一致，最终镜像工作树干净。历史 push receipt 存在。

**只读复核阶段未能实时确认 GitHub remote main/tag。** 当时服务器原代理 `127.0.0.1:18743` 连接被拒绝；临时绕过代理后仍无法解析 `github.com`。本机 tag 核验和历史 receipt 不冒充当时的远端确认。后续用户明确要求 GitHub 同步；同步状态和远端核验另见本目录 README 及外部同步 receipt，不改变上述实验结果。

本地已有的 11 份报告/聚合结果已与服务器 SHA256 比对一致。所有五项本次复核结果见 [VERIFICATION.json](VERIFICATION.json)，原完整报告见 [PATCH_STR_REPORT.md](../patch_str_20261008/PATCH_STR_REPORT.md)。

只读复核阶段新增本地文件为本目录的 SSH 调度/复核脚本、五份日志、verification manifest 和本报告；正式冻结实验产物始终未修改。后续同步阶段将独立复核目录保存到服务器，按用户授权仅发布代码、报告和聚合证据，排除数据、checkpoint、缓存和个体预测。没有安装或删除操作。
