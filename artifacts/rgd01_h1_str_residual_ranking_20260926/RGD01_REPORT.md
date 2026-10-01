# RGD-01：H1–STR residual ranking gap 与互补性诊断

日期：2026-09-26。状态：**development-only diagnosis 完成；STR-01、V8-GN、H1 正式产物保持冻结；未训练新 deep model。** 全程仅使用固定 5×3 inner-development splits、既有 checkpoint 的 DSG-01 冻结表示、正式 development predictions 与 train-only H1 特征。没有读取或使用 outer 信号、标签、预测或指标。所有新增脚本、表格、日志均位于本目录。

## 资产与口径审查

- STR-01/V8-GN：15 split × seeds 42/43/44，共 45 组 validation predictions；DSG-01 归档的 180 份 train/validation frozen representation NPZ。H1 正式档案是 **15 split × 1 个确定性预测**，不存在 H1 的 42/43/44 三种子正式结果。不能把同一 H1 预测重复视为独立 seed；主统计先将 3 个 STR seeds 在 split 内聚合，再以 15 split 为单位。
- 各 split 的 H1 与 STR/V8 validation subject ID 和 label 完全一致，train/validation ID 不重叠。H1 正类为 DD=1，正式分数为 `probability_dd`；STR 分数为 `final_logits[DD]−final_logits[PD]`，与 DD probability 排序等价。按正式 H1 代码在 inner-train 重新拟合，15 split 的 validation probability 与正式文件最大差 `1.11×10⁻¹⁶`。
- Stable-error primary 与 DSG-01 一致：**先聚合每次 validation 的 3 seed，再看同一受试者的四次 validation appearance；error rate≥0.75 为 stable-error，≤0.25 为 stable-correct**。STR 为 74 / 284 / 32（stable-error / stable-correct / unstable）。历史 12 个 seed-fold appearance 定义仅作 sensitivity，未用于本报告主表。
- 旧 `/home/zyt/deep_final/README.md` 开头仍为历史 V8 归档说明；后续 STR-01/DSG-01 正式追加记录已将 STR-01 设为当前冻结主模型。本轮以冻结正式结果与最新章节为准，未改写历史结果。

## A. AUROC 的 PD–DD 配对分解

对每个 split、STR seed 枚举 validation 中全部 PD–DD pairs；100,695 个 seed-pair 实例只是计算材料，**不是 100,695 个独立统计样本**。pair categories 的和、PD×DD 数量及逐 seed pairwise accuracy=AUROC 均通过一致性检查。

| 每 split 均值（3 seed 先平均） | 数值 |
|---|---:|
| validation PD–DD pairs | 2,237.7 |
| H1 AUROC / STR AUROC | 0.7515 / 0.7176 |
| H1−STR AUROC | **+0.0339**；13/15 split 为正 |
| 两者都排对 / 都排错 | 1,353.7 / 306.1 pairs |
| H1 对、STR 错 | 326.6 pairs，14.59% |
| STR 对、H1 错 | 249.7 pairs，11.17% |
| H1 净收益 | +77.0 pairs/split |

受试者分层配对 bootstrap、每个 split 保持 seed 配对后平均：H1−STR gap 95% CI `[+0.0098,+0.0567]`（2,000 次）；15 split 重采样的 CI `[+0.0161,+0.0507]`（10,000 次）。这证明 development ranking gap，不代表独立外部队列效果。

## B. 受试者负担与 hard-subject sensitivity

以唯一 subject 聚合其全部 validation appearances 和 STR seed 后，stable-correct、stable-error、unstable 的 STR pair-error fraction 分别为 `0.195/0.561/0.414`；对应 H1 为 `0.177/0.473/0.360`。H1 rescue fraction 分别 `0.119/0.225/0.199`，STR rescue fraction `0.101/0.137/0.145`；净 H1 rescue fraction 为 `0.018/0.089/0.054`。stable-error 比 stable-correct 的净 rescue 高 `0.071`，subject-level BH `q=0.0029`，但 status 由 STR 错误定义，比较包含选择效应。

跨全部 seed-pair 实例，**涉及 STR stable-error subject** 的 H1 rescue/STR rescue 为 `10,460/6,827`，净 `+3,633`；其余 pairs 为 `4,238/4,408`，净 `−170`。全体净 `+3,463`。因此 H1 的净 AUROC 优势集中在少数 persistent hard subjects 所参与的 pairs，而非广泛的 PD–DD pairs 均匀修正。

仅作为事后敏感性：移除 74 名 primary stable-error subjects 后，H1−STR gap 从 `+0.0339` 变为 `−0.0031`；移除按 STR pair burden 排序的 top 5%（20 人）/10%（39 人）后，gap 分别为 `+0.0198/+0.0134`。这些集合由 validation 结果定义，移除后 class/pair 组成改变，**不应称作新的正式性能或 causal phenotype effect**。这里的 “phenotype-overlap” 是与既有 DSG-01 持续对侧表示结论一致的工作解释，并无独立临床表型确证。

## C. 分数与排序互补性

同一 validation subjects 上，STR–H1 score 的 split mean Spearman `0.541`、Kendall `0.386`；全部受试者 pair 的排序分歧率 `25.83%`，PD–DD margin Spearman `0.551`，基于各训练集尺度的绝对标准化 margin 差异 `0.883`。两者非简单单调缩放；H1-rescue 和 STR-rescue 双向存在。不过分歧不自动等于有益互补，需要 train-only fusion 验证。

每个 split 仅用 inner-train subjects 的两个 frozen score 拟合标准化 C=1 logistic（A：STR，B：H1，C：STR+H1），无 validation 调参。15 split 的 validation 结果：

| 指标 | STR-only | H1-only | STR+H1 | 相对 STR / H1 的配对增量 |
|---|---:|---:|---:|---:|
| AUROC | 0.7176 | 0.7519 | **0.7621** | +0.0445（14/15；95% CI +0.0303～+0.0574；BH q=0.0017）/ +0.0103（12/15；CI +0.0051～+0.0152；q=0.0067） |
| BA | 0.6484 | 0.6770 | **0.6917** | +0.0433（11/15；CI +0.0204～+0.0669；q=0.0090）/ +0.0146（11/15；CI +0.0039～+0.0264；q=0.0353） |

H1-only train-only logistic 与正式 H1 原概率的 AUROC 极小差异来自 logit 输入裁剪及极端概率的数值 ties（正式分数中有 1.0），不可替换正式 H1 performance；正式 gap 仍按原概率计算。Fusion 是机制诊断，**不是新的正式候选模型**。两个基础模型的 train scores 是各自在 inner-train 上的 in-sample scores，而 validation scores 是 held-out；这会影响融合系数，故不将表中数值推断为可部署性能。

## D. H1 residual 在冻结 STR 表示中的可读性

目标为 H1 train-z-logit 减 STR train-z-logit；所有均值、尺度、PCA、Ridge 均仅在各 split inner-train 拟合。每个 probe 都控制 STR final score。固定 Ridge α=1，统一使用 train-only **PCA-16** 比较 activity raw、original subject、structured ordered 与 combined decision input；没有依据 validation 搜索 PCA 维数或超参数。

Score-only baseline 的 validation residual R²/MAE/Spearman 为 `0.083/0.560/0.419`。加入 activity raw 后为 `0.171/0.525/0.496`，R²/MAE/Spearman 的提升为 `+0.0878/+0.0347/+0.0772`，BH q=`0.0058/0.0015/0.0040`。加入 combined decision input 后为 `0.153/0.528/0.456`，R²/MAE 改善 `+0.0701/+0.0318`，q=`0.0154/0.0028`；rank correlation 改善 `+0.0369`，q=`0.392`。Subject 与 structured ordered 各自也改善 R²、MAE，但 Spearman 未过 BH。

未降维、直接用高维 standardized representation 的 Ridge 敏感性出现相反结果：activity raw 有较小正收益，subject/structured/combined 的 R² 均比 score-only 差，说明直接线性拟合容易受维数/方差控制影响。**固定 PCA-16 结果支持 H1 residual score 的一部分已进入 STR frozen representation，甚至 decision input；不证明现有 readout 是 gap 的唯一成因，也不证明可恢复 H1 的具体 rescued ranking pairs。** 这种 residual target 与真正的 disease-label ranking 仍有距离。

## E. 条件性 activity、wrist、H1 feature family 归因

因 H1 rescue 在 stable-error pairs 上重复出现，进一步把 H1 标准化特征×logistic 系数按原 H1 schema 分解；每 split 重新拟合的 H1 validation 概率与正式文件精确复现，所有 feature-group contribution 加总与 H1 logit 一致。比较 H1-rescue 与 STR-rescue pairs 的 H1 DD−PD logit margin 贡献，先 seed 再 split 聚合，并在 family（6）、activity（11）、wrist（2）内作 BH。

六个 H1 family 与左右腕的条件性贡献差均过 BH；11 个 activity 中 10 个过 BH，CrossArms 未通过（q=0.083）。**这些不是 activity/family 富集的独立证据**：分组本来就按 H1 是否排对定义，所有 H1 score 的组成部分都可能因选择条件而整体偏正；两腕和多数 activity 同向更不能支持单一“最佳 activity”或重开 activity/频率支路搜索。现有证据只能描述 H1 rescue 并非由一个独特 activity/wrist/family 集中解释；没有因果归因。

## 最终六个问题

1. **主要是 hard-subject dominated。** H1 gap +0.0339，但净配对收益全部来自涉及 STR primary stable-error 的 pairs；移除这些 subject 后 gap 约为零且略反向。不能独立确证临床 phenotype overlap。
2. **排序错误有互补性。** 两方向 rescue 都多，score Spearman 0.541，排序分歧 25.83%；同时 H1 的净优势明显非均匀。
3. **train-only 融合稳定改善 development AUROC。** 相对 STR-only +0.0445（14/15），相对 H1-only +0.0103（12/15）；这是诊断，不改变正式模型或结果。
4. **部分 H1 residual 信息可从冻结 STR 表示读取。** 固定 PCA-16 的 activity 与 combined decision input 均改善 residual R²/MAE；未降维 probe 不稳，且尚未证明可以救回具体错误 pair。
5. **最符合“hard-subject / persistent overlap 主导，同时存在部分 decision/readout 利用不足”的混合解释。** “representation information gap”或“现有 readout 是唯一瓶颈”均证据不足；真实 phenotype overlap 也缺临床元数据独立验证。
6. **当前不足以批准一项新的网络结构训练。** 尽管融合给出正的互补性证据，尚未用 train-only residual probe 证明具体 H1-rescued PD–DD pairs 可由 STR frozen decision representation 稳定恢复，也没有独立 phenotype 验证。下一步如继续，只能先做预注册、development-only 的 frozen representation pair-rescue 可恢复性诊断；在获得明确机制证据前继续冻结 STR-01，停止 consistency/invariance、DANN/MMD/prototype/contrastive、structured readout、gating、attention、activity/frequency/decision-head 搜索。

## 验证、文件与限制

`scripts/audit.py` 核验 15 split、三个 STR seed、正式对齐与 H1 精确复现；`scripts/ranking.py` 计算 pair decomposition/score correlation/bootstrap；`scripts/burden.py` 计算唯一 subject burden 和 leave-hard-out；`scripts/fusion_residual.py` 做 train-only fusion 与原维数 residual probe；`scripts/residual_pca16_sensitivity.py` 做固定维数敏感性；`scripts/conditional_attribution.py` 做条件性归因；`scripts/smoke.py` 做结果总检。`smoke.log` 为 PASS：15 split、45 STR seed-splits、100,695 pair 实例、45 fusion rows、225 PCA probe rows，H1 max reproduction error `1.11×10⁻¹⁶`。既有 DSG-01 已验证 frozen checkpoints/extraction logits 与正式 logits 完全一致；本阶段**没有修改 model forward 或 checkpoint**，故无需变更前后 logits consistency test。

全部数值表、pair 明细、subject ID 明细、CI、BH 与 protocol JSON 位于 `analysis/`；H1 train-only 重拟合分数位于 `h1_reproduction/`。15 split 来自同一 390 人开发数据并存在重复 subject，split bootstrap CI 不是独立样本或外部泛化保证。H1 单确定性运行与 STR 三 seed 不对称。stable-error 划分和 top-hard 排序用 validation outcome，相关比较仅是机制敏感性。任何正式 STR/V8/H1 结果均未覆盖。
