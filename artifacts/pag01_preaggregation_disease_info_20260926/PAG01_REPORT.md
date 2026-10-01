# PAG-01：Pre-Aggregation Disease Information Preservation & Ambiguity Diagnosis

2026-09-26。**PAG-01 development-only 冻结诊断完成；ALLOW_NEW_TRAINING = NO。** 仅使用 fixed 5×3 inner-development splits、STR-01 seeds 42/43/44 的既有 frozen train/validation representations 与 final logits。没有使用 outer information、H1 residual、H1-rescue pairs，也没有训练新的深度模型或修改正式产物。

## 资产、固定方法与验证

45 个 STR split-seed 实例的 subject ID、label、train/validation 分离及 five stages 维度核验通过：bilateral activity raw/context 均 11×258D，original subject 258D，structured ordered 176D，combined decision 为 258+176D。Primary stable-error 仍采用 DSG-01 的 seed-first 四次 validation appearance 口径（STR 74 stable-error、284 stable-correct、32 unstable），**只作事后敏感性**。每个 split 三 seed 先聚合，15 split 是主要统计单位；390 subjects、100,695 个 PD–DD seed-pair、45 seed-runs 和 permutation 次数都不是独立统计样本。

Score-only baseline 为 STR final DD−PD logit 的 train-only 标准化 logistic disease probe；每个 representation 在控制同一 score 后，加入 train-only StandardScaler + 固定 PCA-16，再拟合固定 C=1/liblinear logistic。PCA-8/32 是预定敏感性，未按 validation 选维数。Inner-train 5-fold subject-level cross-fitting 分别拟合 score-only 与 score+activity raw probe，二者 OOF DD logit 的差为 incremental disease signal；validation 的真实 signal 用 full inner-train 拟合的两 probe 差计算。各 stage 用 score covariate + train-only PCA + Ridge α=1 预测该 OOF signal，再在 validation 检查 R²/MAE/Spearman 及 baseline disease logit+预测 signal 的 AUROC。

Negative control 每 split-seed-stage 做 10 次，仅在 inner-train STR final score deciles 内打乱 PCA representation 的受试者对应关系；validation 无拟合、无置换。Ambiguity 只由 inner-train score-only probe 的 5-fold OOF **绝对 disease logit margin** 的三分位阈值给出：最小为 high，中间为 medium，最大为 low；同一训练集阈值应用到 full inner-train probe 的 validation margin。类别标签/正确性不用于 ambiguity 阈值选择。PD–DD pair 的 ambiguity 取成员中较高的一档。Stable-error 仅作敏感性。

`audit.log`、单实例 `smoke.log` 与全量 `verify.log` 均 PASS。全量保存 720 disease-probe rows、675 signal-probe rows、2,250 score-conditional permutations、23,400 primary subject-stage records 和 100,695 validation pair records。Score-only validation AUROC 与冻结 STR 正式分数 **45/45 完全相等**；raw activity pair net gain/pairs 与 AUROC 增量最大差 `2.31×10⁻¹⁶`。未修改 forward/checkpoint，沿用 DSG-01 的原有 logits consistency 证明。

## 1. Final score 之外的 disease information

PCA-16，15 split mean：

| Stage（均控制 final score） | AUROC | ΔAUROC vs score-only | BA | Macro-F1 | PD Recall | DD Recall | log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| final-score-only | 0.7176 | 0 | 0.6484 | 0.6541 | 0.8563 | 0.4404 | **0.6686** |
| bilateral activity raw | **0.7350** | **+0.0174** | 0.6598 | 0.6645 | 0.8460 | 0.4735 | 0.7983 |
| activity context | 0.7321 | +0.0145 | 0.6583 | 0.6631 | 0.8451 | 0.4714 | 0.8009 |
| original subject embedding | 0.6964 | **−0.0212** | 0.6378 | 0.6390 | 0.8117 | 0.4639 | 0.9202 |
| structured ordered | 0.7202 | +0.0026 | 0.6579 | 0.6598 | 0.8219 | 0.4939 | 0.8656 |
| combined decision input | 0.7137 | −0.0039 | 0.6543 | 0.6530 | 0.8068 | 0.5019 | 1.0096 |

Activity raw 的 ΔAUROC `+0.0174`，12/15 split 改善；paired bootstrap 95% CI `[+0.0078,+0.0261]`、paired rank-biserial `0.733`、5-stage AUROC BH q=0.0171。Context `+0.0145`，13/15，q=0.0084；subject `−0.0212`，14/15 恶化，q=0.0084。Structured 和 combined 的 ΔAUROC 不稳定。原分数 log loss 最低，加入 representation 的线性 probe 虽改善 raw/context ranking，却明显损害概率质量；**不能作为正式模型或新训练收益估计**。BA、Macro-F1 和两类 recall 均已逐 split 保存，不能仅以 AUROC 解释阈值行为。

PCA-8/16/32 下 activity-raw ΔAUROC 分别 `+0.0098/+0.0174/+0.0223`，每档 11–12/15 split 为正，按各档完整指标家族 BH q=`0.0146/0.0237/0.0092`。效果大小随维数变化，不是对 PCA 维数不敏感的常数；三档均保留正方向。

### Conditional permutation

Primary PCA-16 中，activity raw 的真实 ΔAUROC 为 `+0.0174`，训练集相近 STR-score 层内置换均值 `+0.0064`；真实−置换 `+0.0110`，11/15 split，bootstrap CI `[+0.0048,+0.0176]`，BH q=0.0311。Context 对应 `+0.0145` vs `+0.0054`，q=0.0269。Subject 的真实/置换均约 `−0.021`；structured 和 combined 的真实−置换差未通过 BH。控制说明 raw/context 的增量不只是 score 分层内任意表示的效果，但单一置换设计不构成信息流失的因果证明。

## 2. Incremental disease signal 的 stage progression

Primary PCA-16 的 validation 结果：

| Stage | signal R² | MAE↓ | Spearman | 预测 signal 修正后的 AUROC Δ vs score-only |
|---|---:|---:|---:|---:|
| activity raw | **0.834** | **0.488** | **0.919** | **+0.0195** |
| activity context | 0.794 | 0.548 | 0.891 | +0.0176 |
| original subject | 0.456 | 0.879 | 0.648 | +0.0042 |
| structured ordered | 0.555 | 0.785 | 0.730 | +0.0084 |
| combined decision | 0.544 | 0.800 | 0.716 | +0.0071 |

直接配对，而不是比较各自显著性：activity raw−combined decision 的 disease-probe AUROC 差 **+0.0213**（13/15，CI `[+0.0137,+0.0288]`，同档 source 内 BH q=0.0005，rank-biserial 0.950）；signal R² 差 `+0.290`、MAE 优势 `+0.312`、Spearman 差 `+0.203`（均 15/15，q≈0.0001）；signal 修正后 AUROC 差 `+0.0124`（14/15，q=0.0038）。PCA-8/32 的 raw−decision AUROC 差分别 `+0.0101/+0.0328`，均过 BH；signal R² 差 `+0.148/+0.236`，也均过 BH。

但**不是逐级单调衰减**：raw→context disease AUROC 仅降 `0.0029`（CI 跨零）；context→subject 降 `0.0357`（14/15，q=0.0002）；subject→combined decision **回升 `0.0173`**（即 subject−decision `−0.0173`，CI `[−0.0258,−0.0091]`，q=0.0051）。Signal R² 也在 subject→decision 回升 `0.088`（15/15）。四个串联 stage 的严格单调 AUROC 仅 2/15 split、signal R² 为 0/15；PCA-8/32 同样不呈稳定单调序列。Structured ordered 是并行分支，不能强行排进一条单链。证据支持**activity/context 到 subject 的局部可读性下降，final decision 未完全恢复 activity raw 的增量**；不支持“信息在每个 aggregation/decision 步骤持续递减”的强 information-loss/compression 叙述。

## 3. Train-derived ambiguity 与 persistent errors

Validation 按固定 train-OOF margin 阈值分组，high/medium/low 每 split 平均人数 `48.0/28.4/27.6`。STR error rate 为 `37.50%/22.23%/14.59%`；high−medium `+15.27` 个百分点（15/15，CI `[+11.79,+18.63]`，BH q<0.001），high−low `+22.91` 个百分点（15/15，CI `[+20.12,+25.70]`，q<0.001）。因此无需 validation correctness 定义的 high-ambiguity 分组能识别更高错误负担，但它直接基于与 STR final score 同源的置信 margin，不能视作独立 phenotype 量测。

Raw probe 在 high/medium/low 内的平均 AUROC 增益分别 `+0.0236/+0.0431/+0.0362`；high 的 15-split CI `[−0.0008,+0.0490]`、BH q=0.107，medium/low 的 q=`0.025/0.032`。然而 high−medium、high−low 的 AUROC 增益差均**未通过 BH**（q=0.203/0.330），所以不能宣称 raw 信息只改善 low/medium 而完全无法改善 high。按 pair 最高 ambiguity 分组，high/medium/low 的原错恢复率 `19.8%/24.0%/34.6%`、harm率 `6.1%/6.7%/10.2%`、net gain/pair `+0.0113/+0.0297/+0.0362`。High 的恢复率显著低于 medium/low（BH q=0.013/0.007），net gain/pair 低于 medium（q=0.038）；high 自身的 net gain CI 触零，q=0.107。High 不是“绝对不可恢复”，只是改进较弱且错误负担持续较高。

Stable-error 只作为 sensitivity：high/medium/low 出现 stable-error subject 的比例为 `22.9%/17.6%/13.6%`，说明 train-derived ambiguity 与 persistent errors 部分重合，并非两个已证明独立的机制。涉及 stable-error subject 的 pairs 原错恢复率 `19.5%`、harm `11.1%`；不涉及者为 `33.5%/4.1%`。但两类 pair 的原错基率不同，stable-error pair 的 net gain/pair 仍可为正；不能以 gross/net 单一指标推出“完全无法改善”。

## 关键 evidence boundary

虽然 disease probe 做了 inner-train subject-level 5-fold cross-fitting，**冻结 STR 模型在这些 inner-train subjects 上已训练过**。因此 OOF probe logits 不是深度模型层面的 OOF prediction。这个分布错位可直接观察：train OOF score-only AUROC 均值 `0.9541`，held-out validation 仅 `0.7176`；raw probe 对应 `0.9475` vs `0.7350`。增量 signal 的 train OOF target 与 validation full-train target、ambiguity 的 OOF 阈值与 validation margin 可能因此不完全同尺度。不能把 high ambiguity 视为独立、经过跨受试者校准的 phenotype 状态，也不能把高 signal R² 单独当作训练新网络的因果证据。

15 split 来自同一 390 人开发数据、subjects 重复出现；bootstrap CI 与 Wilcoxon/BH 只是固定开发协议内证据，不是外部队列保证。五类表示虽然 PCA 维数固定，仍有不同原始维度和相关性；线性 probe 优势不能证明原表示没有可由更复杂函数恢复的信息。本轮禁止新结构搜索，也未进行任何调参或深度训练。

## 对最终八问的结论

1. **是。** Activity raw 在控制 final STR score 后有稳定且可在 held-out validation 泛化的增量 PD/DD ranking information：ΔAUROC +0.0174，12/15，BH q=0.0171。
2. **是。** Raw 的增量超过相近 STR-score 层内 conditional permutation：真实−置换 +0.0110，q=0.0311。
3. **不是逐级稳定变难。** Activity/context→subject 有明显下降，但 subject→combined decision 又部分回升；强单调衰减假设失败。
4. **直接差异成立。** Raw−combined decision AUROC +0.0213、13/15、q=0.0005；signal R² +0.290、15/15。PCA-8/32 同方向。
5. **只支持局部 attenuation，不足以支持整体单调 information-loss/compression 机制。** 现有 readout、分支混合与 train/validation score 分布错位无法被本轮线性诊断拆成因果信息损失。
6. **是。** Train-derived high-ambiguity 的 validation STR error rate 37.5%，显著高于 medium 22.2% 和 low 14.6%。
7. **不能断言完全无法改善。** High 的 raw gain 平均为正但不稳定，pair recovery 较低，net gain CI 触零；高错误负担仍在，且 high 与 stable-error 并非已证独立类别。
8. **证据不足。** 预设训练闸门需要增量信息、conditional control、稳定 stage-wise attenuation、raw-vs-decision 直接比较四项同时成立。第 3 项的逐级稳定衰减不成立，而且 train-deep-score 的 in-sample/validation 错位显著。**ALLOW_NEW_TRAINING = NO。** 停止把本轮结果直接转为 activity-to-subject information-preservation、auxiliary loss、new decision head 或其他 deep training；保留冻结 STR-01。

## 产物

独立目录内 `scripts/audit.py`、`run.py`、`summarize.py`、`verify.py`、`protocol.json` 与 `analysis/` 保存各 stage 原始 45 实例和 seed-first 15-split 表、全部指标、paired difference、SD/median、rank-biserial、bootstrap 95% CI、BH、PCA-8/16/32、conditional permutation、subject/pair ambiguity records、stable-error sensitivity；`audit.log`、`smoke.log`、`run.log`、`summary.log`、`verify.log` 保存执行与核验。所有正结果、负结果和 NA/限制均为新独立产物，未覆盖 DSG-01/RGD-01/PRR-01 或正式模型记录。
