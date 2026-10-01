# V8-GN Bilateral Fusion Component Diagnosis

日期：2026-09-20  
实验编号：BFC-01  
范围：冻结 V8-GN、固定 inner-development protocol 的离线诊断

## 1. 目的和证据边界

本分析比较冻结 wrist encoder 输出构成的四个同维64维表示：Left、Right、双腕 Mean 和 raw AbsDiff。所有 scaler、probe 和统计量仅使用当前 inner-train 拟合；三个 seed 在相同 `(outer, inner)` split 内先求均值，再以15个 split 为统计单位。未训练新模型、未修改 backbone/training recipe、未访问 outer-test。

Passive probe 可以判断各分量携带的信息和 shift，但不能证明某一分量对当前冻结分类头性能的因果贡献。因果判断仍需要后续一次严格的 fusion ablation。

## 2. Wrist mask 审计

全部45个 seed-fold 的 train 和 validation 中：

- missing left = 0；
- missing right = 0；
- incomplete bilateral activity = 0；
- missing wrist rate = 0%；
- 每个 split 的 wrist-mask pattern 只有一种，即全部 `[1,1]`。

因此 wrist mask 在当前数据中是常量，只保留接口兼容意义。它没有可用于疾病判别、个体识别或 shift 解释的变化，本报告不对该2维分量赋予额外含义。

## 3. 四个64维分量的统一比较

下表为15个独立 split 的均值±split SD。

| 分量 | Probe BA | Probe AUROC | Centroid BA | Cosine silhouette | Disease variance | Within-disease variance |
|---|---:|---:|---:|---:|---:|---:|
| Left | 0.5937±0.0381 | 0.6309±0.0352 | 0.5927±0.0307 | 0.0217±0.0137 | 0.0243±0.0112 | 0.9757±0.0112 |
| Right | 0.5933±0.0305 | 0.6364±0.0252 | 0.5934±0.0290 | 0.0189±0.0116 | 0.0233±0.0104 | 0.9767±0.0104 |
| Mean | **0.6048±0.0199** | **0.6474±0.0229** | **0.5965±0.0307** | **0.0247±0.0150** | **0.0281±0.0140** | 0.9719±0.0140 |
| AbsDiff | 0.5828±0.0354 | 0.6226±0.0326 | 0.5962±0.0249 | 0.0121±0.0071 | 0.0166±0.0048 | **0.9834±0.0048** |

Mean 在平均 BA、AUROC、silhouette 和 disease-between variance 上均最高。Left 与 Right 没有稳定差异：AUROC差为 -0.0055（Left-Right，p=0.847），BA差为+0.0004（p=1.0）。

Mean 相对 AbsDiff：

- BA +0.0220，12/15 split 更高，p=0.0125；
- AUROC +0.0248，13/15更高，p=0.00671；
- silhouette +0.0126，14/15更高，p=0.00043；
- disease-between variance +0.0115，14/15更高，p=0.00043。

AbsDiff 的 AUROC 仍为0.6226，说明它不是完全没有疾病相关信息，但它在四个分量中疾病判别和相对疾病方差最弱，不能被视为主要 bilateral disease signal。

## 4. Subject-specific information

平均随机 Top-1 机会约为0.0096。

| 分量 | Activity probe | Subject Top-1 | Subject Top-5 | Same-subject similarity | Subject similarity gap |
|---|---:|---:|---:|---:|---:|
| Left | 0.6310 | 0.0427 | 0.1419 | 0.1905 | 0.1572 |
| Right | 0.6606 | 0.0432 | 0.1430 | 0.1859 | 0.1519 |
| Mean | **0.6673** | **0.0501** | **0.1590** | **0.2167** | **0.1853** |
| AbsDiff | 0.4646 | 0.0375 | 0.1224 | 0.1191 | 0.0876 |

Mean 在13/15 split 上具有最高 Top-1，并在15/15 split 上具有最高 same-subject similarity 和 similarity gap。AbsDiff 在11/15 split 上具有最低 Top-1，并在15/15 split 上具有最低 same-subject similarity/gap。

Mean 相对 AbsDiff：Top-1 +0.0126、same-subject similarity +0.0977、similarity gap +0.0977，三项均 p=0.000061。

因此 raw AbsDiff 并没有携带最强的、跨 activity 稳定的 subject-specific signature；稳定个体信息主要存在于左右腕共享的 Mean 表示中。

## 5. Train→unseen-validation shift

| 分量 | Disease-controlled domain AUC | Normalized mean shift | CORAL shift |
|---|---:|---:|---:|
| Left | 0.5527±0.0312 | 0.1175±0.0207 | 0.4052±0.0550 |
| Right | **0.5335±0.0320** | 0.1205±0.0260 | 0.4315±0.0874 |
| Mean | 0.5555±0.0354 | **0.1173±0.0233** | **0.3751±0.0713** |
| AbsDiff | **0.5693±0.0436** | 0.1203±0.0128 | **0.5238±0.0386** |

AbsDiff 的 CORAL shift 最高，在11/15 split 上为四者最高；Mean 的 CORAL 最低。AbsDiff-Mean CORAL 差为+0.1486，15/15 split 均更高，p=0.000061。Domain AUC 的 AbsDiff-Mean 差为+0.0137，但一致性不足，p=0.188；mean shift 四分量相近。

这意味着 AbsDiff 的主要异常是 covariance/distribution sensitivity，而不是稳定 subject identity。它可能包含 activity-dependent噪声、左右尺度差异或其他不稳定因素；具体来源尚未验证。

## 6. 完整258维 fusion 相对 Mean

结合上一阶段使用相同 splits 的 bilateral activity 诊断：

| 指标 | Mean 64D | Full fusion 258D | Full-Mean matched delta |
|---|---:|---:|---:|
| Probe BA | 0.6048 | 0.6070 | +0.0022，p=0.720 |
| Probe AUROC | 0.6474 | 0.6549 | +0.0074，p=0.135 |
| Disease silhouette | 0.0247 | 0.0189 | -0.0058，14/15下降，p=0.00018 |
| Disease variance | 0.0281 | 0.0232 | -0.0049，13/15下降，p=0.00061 |
| Subject Top-1 | 0.0501 | 0.0632 | +0.0131，15/15上升，p=0.000061 |
| Domain AUC | 0.5555 | 0.5839 | +0.0283，13/15上升，p=0.00201 |
| CORAL shift | 0.3751 | 0.4652 | +0.0900，15/15上升，p=0.000061 |

完整 fusion 相对 Mean 只有小且不显著的 disease probe 增益，却稳定增加 domain shift 和检索型个体可恢复性，并降低相对疾病几何。由于完整 fusion 维度更高且当前分类头直接使用它，这仍是描述性比较，不是 mean-only 性能等价的证明。

## 7. 分量指标与最终 BA

四分量 disease probe、subject retrieval 和大部分 shift 指标与最终 BA 均无显著关系。唯一在本报告全部相关检验的 Benjamini-Hochberg FDR 后仍保留的关系是：

- Left domain-probe AUC 与 BA：ρ=-0.7786，p=0.000627，FDR q=0.0351。

Left CORAL（ρ=-0.6464，p=0.0092）和 Mean CORAL（ρ=-0.5964，p=0.0189）在未校正分析中为负相关，但FDR后不显著。Left shift 与较低 BA 的关系是值得复核的线索，不足以证明因果关系。

## 8. 对核心问题的回答

### 实验确认

1. Wrist mask 在当前数据中完全恒定，不携带可分析信息。
2. Mean 是四个等维分量中疾病信息最强的表示，也是跨 activity 稳定 subject information 最强的表示。
3. Left 与 Right 的疾病判别能力总体相近，没有稳定单侧优势。
4. AbsDiff 携带一定疾病信息，但显著弱于 Mean；它的稳定 subject retrieval/similarity 也是四者最低。
5. AbsDiff 的 CORAL shift 显著最高，说明它对 train→validation 协方差变化最敏感。
6. 完整 fusion 相对 Mean 的疾病 probe 增益很小且不显著，但 shift 显著增加。

### 合理推测

1. 当前 bilateral fusion 的大部分可恢复疾病信息来自左右腕共享成分，即 Mean；Left/Right 提供少量互补，AbsDiff 不是主要病理不对称通道。
2. Raw AbsDiff 更像不稳定的跨域差异来源，而不是稳定 subject identity 来源。
3. 完整 fusion 中重复放入 Left、Right、Mean 可能提高线性容量和身份可恢复性，却没有等比例增加疾病几何。

### 尚未验证

1. Mean-only 输入能否保持冻结模型的最终 BA/AUROC；passive probe 不能代替重新训练后的因果 ablation。
2. AbsDiff 的 shift 是否由幅值尺度、活动执行、左右佩戴差异或其他因素造成。
3. 删除 AbsDiff 是否一定有益；它可能仍包含与其他分量互补的非线性疾病信息。

## 9. 下一步决策

诊断结果足以支持一个、且仅一个优先因果实验：

> **Mean-only bilateral fusion ablation**：将 activity 表示限制为双腕共享 Mean，保持其余 backbone、训练 recipe、splits、seeds 和评估完全不变。

该实验直接检验“完整拼接的微小信息增益是否值得其显著增加的 shift”。在 Mean-only 因果结果出来之前，不建议并行搜索 scale-normalized AbsDiff、reliability gating 或多种新融合结构。

