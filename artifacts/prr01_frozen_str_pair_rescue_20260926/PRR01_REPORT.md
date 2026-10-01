# PRR-01：Frozen STR Pair-Rescue Recoverability Diagnosis

2026-09-26。**诊断完成；STR-01、V8-GN、H1 及既有正式结果保持冻结；没有训练新 deep model。** 只使用固定 15 个 inner-development splits、STR seeds 42/43/44、每 split 唯一的确定性 H1 预测、DSG-01 frozen STR representations 与 RGD-01 train-only H1 artifacts。没有访问 outer signal、label、prediction 或 metric。

## 资产、smoke 与协议

`asset_audit.csv` 核验 45 个 STR split-seed train/validation archive，subject ID、label、train/validation 分离、11×258 activity raw、258 subject、176 structured ordered、2D logits 均符合已冻结 DSG-01 资产。H1 每 split 只有一份正式预测；三个 STR seeds 先在 split 内聚合，15 split 为主要统计单位，不把 H1 复制视作独立 seed。Stable-error primary 与 DSG-01/RGD-01 完全一致：先聚合三个 seed，再看四次 validation appearance，STR stable-error/stable-correct/unstable 为 74/284/32。

RGD-01 原始 H1 `probability_dd` 决定 pair categories；只有 residual target 使用与 RGD-01 一样裁剪至 `[10⁻⁸,1−10⁻⁸]` 后的 H1 logit。修正极端概率的 ties 后，本轮 100,695 个 seed-pair 的 `both_correct/both_wrong/H1 rescue/STR rescue/tie` 与 RGD-01 **逐条 0 差异**。`smoke.log` 在一个 split-seed 通过；`verify.log` 全量通过：45 模型实例、6 组方法、1,800 次训练集置换 probe、390 个唯一 subject，pair net gain÷pair count 与 AUROC 增量最大差 `3.23×10⁻¹⁶`。没有修改 frozen forward 或 checkpoint，因此沿用 DSG-01 已完成的 checkpoint/logits consistency 验证。

每个 split-seed 仅在 inner-train 上计算 H1/STR logit 均值和标准差，target 为 `H1_train_z_logit − STR_train_z_logit`。所有方法用相同 Ridge `α=1`；表示组先 train-only StandardScaler 和固定 PCA-16（随机种子 0），再与 STR score covariate 拼接。Validation corrected score 为 `STR_train_mean + STR_train_sd × (STR_validation_z + predicted_residual)`。A 是原 STR score，B 是 score-only residual，C/D/E/F 分别为 activity raw、original subject、structured ordered、combined decision input。BA 仅以 corrected DD−PD logit `0` 为固定阈值描述，未在 validation 调阈值。全部结果仅为 frozen diagnostic，不能替换正式模型性能。

## 主结果：真实 pair rescue 与副作用

下表为 15 split 的均值，三个 STR seeds 先在 split 内平均。每 split 约 2,238 个 PD–DD pairs、327 个 H1 rescue pairs。`net pair gain` 严格定义为**全部原 STR 排错配对被纠正 − 全部原 STR 排对配对被破坏**，与 AUROC 增量×pair count 一致；`targeted net` 单独定义为**H1-rescue 配对被纠正 − 全部原 STR 排对配对被破坏**。两者不能混用。

| 方法 | AUROC | Δ vs STR/score-only | H1-rescue 恢复率 | 恢复/未恢复 H1-rescue（pairs/split） | 原 STR 排对被破坏（pairs/split） | harm率 | net pair gain | targeted net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A STR | 0.7176 | 0 | 0 | 0/326.6 | 0 | 0 | 0 | 0 |
| B score-only | 0.7176 | 0 | 0 | 0/326.6 | 0 | 0 | 0 | 0 |
| C activity raw | **0.7299** | **+0.0123** | 22.87% | 74.3/252.3 | 73.0 | 4.57% | **+27.5** | +1.3 |
| D subject | 0.7234 | +0.0057 | 23.91% | 78.4/248.2 | 103.9 | 6.51% | +12.9 | −25.5 |
| E structured | 0.7263 | +0.0087 | 25.28% | 82.8/243.8 | 95.4 | 5.98% | +19.6 | −12.6 |
| F decision | 0.7240 | +0.0064 | 25.73% | 84.8/241.8 | 106.0 | 6.66% | +14.4 | **−21.2** |

B 的 score-only correction 是单调分数变换，故 ranking 与 A 相同；BA 因固定阈值可变化。C 的原 STR 正确配对保留率为 95.43%，但新增破坏仍接近其 H1-rescue 恢复数。C 的**总体**净增益来自还纠正了部分 `both_wrong` pairs：全部原错恢复 100.5、原对破坏 73.0、净 +27.5。C 的 targeted net 仅 `+1.29` pairs/split，15-split bootstrap 95% CI `[−12.53,+14.67]`、BH q=0.720；不能说 H1 原有 gap 的目标配对被稳定净救回。F 的 targeted net `−21.24`（CI `[−33.93,−8.87]`，BH q=0.0124）。

相对 B 的预定四表示×三主要量度 BH：C 的 AUROC +0.0123（11/15 split，paired bootstrap 95% CI `[+0.0056,+0.0188]`，q=0.0107）、恢复率 +0.2287（15/15，CI `[0.2070,0.2521]`，q=0.0002）、总净配对 +27.5（11/15，CI `[+12.7,+41.9]`，q=0.0107）。D 的 AUROC 与净配对均未通过 BH；E 虽 11/15 为正，但 AUROC/净配对的 primary-family q=0.0619；F 只有 9/15 AUROC 为正，Δ +0.0064，CI `[−0.0002,+0.0133]`、q=0.203。四种表示的 H1-rescue **gross** 恢复率都高于 B，但不能只据 gross recovery 判定 readout gap。

BA 与 AUROC 的方向不同：A 原 STR 固定阈值 BA `0.6972`，C `0.6682`，F `0.6540`；相对 B 的 C/F BA 也分别低 `0.0122/0.0264`。这说明当前 correction 未形成可替代正式分类器的 threshold behavior。正式 STR/H1 performance 不受此诊断表影响。

## Hard-subject / stable-error sensitivity

只作机制描述；stable-error status 由既有 validation outcome 定义，不能当作独立预测分层或新的正式性能。C activity raw 在**涉及任一 STR stable-error subject** 的 H1-rescue pairs 中恢复率 `18.85%`，原 STR 正确 pairs harm `10.26%`，targeted net `−8.0` pairs/split（bootstrap CI `[−16.58,+0.24]`，仅 5/15 split 为正）；其余 pairs 恢复率 `33.46%`、harm `2.01%`、targeted net `+9.29`（CI `[+1.27,+17.20]`）。F decision 的对应 hard-group 恢复率 `21.93%`、harm `13.75%`、targeted net `−15.91`（CI `[−24.82,−6.51]`）。

以 390 个唯一受试者汇总 pair involvement，C 对 stable-error/stable-correct 的平均 H1-rescue recovery fraction 为 `0.173/0.311`，harm fraction `0.138/0.035`；F 分别为 `0.206/0.344` 和 `0.167/0.053`。因此整体 AUROC 的少量提升主要没有解决 RGD-01 所指向的 persistent hard-subject ranking burden。这里的 “overlap” 仍是表示/错误现象，不是由独立临床 phenotype metadata 确证的病理机制。

## 训练集置换对照

对每 split-seed、每表示做 10 次 **inner-train 内 PD 与 DD 各自独立的 residual-target permutation**，保持类组成和 validation 完全不变，然后用同一 PCA-16/Ridge 拟合。C 真实与置换均值的 AUROC Δ 分别为 `+0.0123/+0.0005`，配对差 `+0.0118`（12/15 split，95% CI `[+0.0049,+0.0185]`，BH q=0.0161）；总净配对 `+27.5/+1.1`，差 `+26.4`（q=0.0167）；H1-rescue 恢复率 `22.87%/8.79%`，差 `14.08` 个百分点（q=0.0002）。故 C 的宽泛 ranking gain 超过此预定 negative control。F 虽 gross recovery 超过置换，AUROC/net 对照 q=0.0765，未形成稳定的 decision-stage 净证据。10 次 permutation 只检验这一类打乱方式，不穷尽所有伪相关来源。

## Pair-level 条件性描述

仅描述 C 与 F，不做 activity/feature-family 搜索。C 的 recovered H1-rescue pairs 的原 STR margin 均值约 `−0.392`，未恢复为 `−1.368`；H1 probability margin 分别 `0.474/0.351`。前者更靠近 STR 排序边界；未恢复 pairs 有 stable-error subject 参与的比例 `75.1%`，恢复组 `57.7%`。新破坏 pairs 的原 STR margin 为 `+0.417`，约 `69.6%` 涉及 stable-error。Predicted residual pair margin 的组间方向由“恢复/破坏”的结果定义产生，不能作为独立机制证据。Activity representation 相对 train mean 的 L2 magnitude 在 C recovered/unrecovered/newly broken 三组约 `41.7/41.4/42.2`，未见明确分离；不支持重新寻找关键 activity。

## 按用户七个问题的判断

1. **能部分 gross 恢复，不能稳定净恢复目标 H1-rescued pairs。** C/F 分别 gross 恢复 22.9%/25.7%，但 C targeted net 接近零，F 为负。
2. **相对 score-only baseline，C 在总体 AUROC、总净配对与 gross recovery 上显著改善。** D/E/F 的净 AUROC 未稳健通过 primary BH；目标配对的 targeted net 没有相应改善。
3. **Activity raw 是整体 ranking gain 最明确的 stage。** C 相对 F 的 AUROC 差 `+0.0059` 未达显著（15 split Wilcoxon p=0.151），故不能断言两 stage 有统计可辨的真实优劣；但 F 自身没有稳定净收益。
4. **恢复没有主要发生在 persistent hard pairs。** C 的 hard-group gross recovery 18.9%，低于非 hard 33.5%，且 hard-group targeted net 为负。
5. **伴随重要损害。** C 每 split 新破坏约 73 个原本 STR 排对 pairs，几乎抵消约 74 个 H1-rescue 恢复；F 新破坏约 106 个，多于约 85 个目标恢复。
6. **整体 corrected ranking 有有限而稳定的正收益，仅 C 成立。** C AUROC +0.0123、11/15，超过置换；剩余正式 H1−C gap 约 `0.0216`。这一收益没有伴随 BA 改善或 hard-group targeted net 改善。
7. **最符合“pre-aggregation representation 中存在部分未利用的宽泛 ranking information，同时 persistent hard-subject limitation 尚未解除”。** Activity raw 的正结果与从 activity 到 combined decision 的压缩/信息利用损失相容，但 C−F 直接 AUROC 比较不显著，不能证明压缩是因果原因。现有证据不支持将剩余 H1 gap 定位为单纯 decision/readout utilization problem；是否为根本 representation gap 或临床 phenotype overlap 仍证据不足。

## 决策与统计边界

预设的五项宽泛判据（优于 score-only、gross rescue、总净 gain、AUROC 多数 split 提升、超过置换）**在 activity raw stage 满足**，所以可以承认冻结 STR 的前聚合表示含有当前最终分数未充分利用的部分 ranking information。可是 combined decision stage 没有稳定净收益；activity raw 的 target-specific net 几乎为零，stable-error pairs 仍易受损。按照 PRR-01 关于“仅 activity raw 有信息而 combined decision 不能恢复”和“persistent hard pairs 未能稳定恢复”的停止分支，**本轮不启动新的 lightweight readout、decision head 或任何 deep model training**。后续如继续，也应先在独立于 validation outcome 的设计中明确区分前聚合信息保留与 hard-subject overlap，而不是重开 activity/attention/gating/structured-readout/frequency/contrastive/invariance 搜索。

15 split 共享 390 个 development subjects 且彼此有重叠，split bootstrap 不是独立外部验证；三 seed 不是 45 个独立样本，配对数量也不是独立样本。H1 单确定性模型与 STR 三 seed 不对称。PCA-16 与 α=1 是冻结诊断选择，没有做 validation 搜索；仍有 in-sample 基础 train score 与 held-out validation score 的分布差异。Class-conditional target permutation 是负对照，不替代外部队列或因果干预。BA 使用固定阈值仅作诊断。

## 产物

本目录 `scripts/` 包含 `audit.py`、`run.py`、`summarize.py`、`subject_burden.py`、`verify.py`；`analysis/` 包含 45 与 15 split metrics、pair-level 503,475 行五组 corrected 方法记录、subject-level records 与 390 unique-subject pair burden、CI/BH、hard-subject sensitivity、negative-control 1,800 行与 conditional attribution。`protocol.json` 逐项记录数据边界、score 与 net-gain 定义；`audit.log`、`smoke.log`、`run.log`、`summary.log`、`subject_burden.log`、`verify.log` 保留执行与核验记录。没有覆盖 DSG-01、RGD-01 或任何正式实验产物。
