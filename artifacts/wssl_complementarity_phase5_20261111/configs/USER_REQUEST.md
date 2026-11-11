# Phase 5：WSSL–Reference Complementarity and Representation Reliability Analysis

你现在是一名具有丰富研究经验的计算机视觉、可穿戴 IMU 时序分析、深度学习、异常检测及帕金森病数字生物标志物研究人员。

当前研究任务：

基于 PADS 数据集，利用双腕 IMU 传感器数据进行受试者级 PD（Parkinson's Disease）与 DD（Differential Diagnosis）二分类。

GitHub：
https://github.com/chouytong/pureDeep

本轮正式起点：

Commit:
38c89251a442bb245a77d4bffc539da050fcdeb1

本地服务器：
/home/zyt/deep_final

当前主线：
/home/zyt/deep_final/mainline

本轮历史实验目录：
/home/zyt/deep_final/artifacts/wssl_reference_audit_20261111

当前最佳模型：
Frozen WSSL-STR

当前正式指标：

Accuracy = 0.745604
BA = 0.719644
AUROC = 0.759553
Macro-F1 = 0.706589
PD Recall = 0.782344
DD Recall = 0.656945

本轮必须继续保留 Frozen WSSL-STR，不允许因为某个探索性指标较高就替换主模型。

---

## 一、当前已知实验结论

必须先阅读并理解：

- artifacts/wssl_reference_audit_20261111/FINAL_REPORT.md
- artifacts/wssl_reference_audit_20261111/STATISTICAL_APPENDIX.md
- artifacts/wssl_reference_audit_20261111/LITERATURE_VERIFICATION.md
- artifacts/wssl_reference_audit_20261111/REPRODUCE.md
- README.md
- AGENTS.md
- CONTEXT_HANDOFF
- 所有相关历史实验报告。

重点掌握以下结果。

### 1. Frozen WSSL-STR

当前 retained single-model。

45 个正式 checkpoint 已完成完整 validation 推理复核，最大概率误差为 1.11e-16。

不重复训练。

### 2. CE 审计

固定分母 CE 没有完成正式训练。

入口门槛未通过，尤其是实际 TF32 梯度精度误差。

这是 NOT_RUN_GATE_FAIL，不是性能负结果。

本轮不重新启动该候选。

### 3. Acc/Gyro 模态诊断

local representation 没有表现出稳定增量：

ΔBA = +0.000476
ΔAUROC = +0.000166

不能据此判断 Gyro 没有疾病信息。

但暂时不再训练新的 Gyro 分支。

### 4. 温度校准

NLL 有明确改善：

0.616720 → 0.577543

Brier 和 ECE 的置信区间跨零。

仅保留 calibration diagnostic，不改变正式模型。

### 5. 双参考典型性

Relative AUROC = 0.735639。

相比 PD-only：

ΔAUROC = +0.067150

相比 DD-only：

ΔAUROC = +0.236204

但仍低于 WSSL AUROC = 0.759553。

### 6. 两分数融合

Fusion BA = 0.627638。

Fusion DD Recall = 0.327707。

相对 WSSL：

ΔBA = -0.092006。

正式 REJECT。

禁止重新搜索融合系数、阈值和分类头。

---

## 二、本轮核心研究假设

只研究以下问题：

**H1：双参考 relative representation 包含 WSSL 未充分利用的部分疾病排序信息，但与 WSSL 的互补性不足以形成稳定的分类净收益。**

**H2：双参考融合失败可能与分数尺度、OOF/full-model 分布迁移、类别偏置或参考库覆盖差异有关。**

这些是假设，不是已确认结论。

必须寻找能够支持或否定它们的证据。

**H3：Frozen 与 last-block adapted WSSL 的性能差异可能来自部分受试者的表征变化，而非所有受试者一致改善。**

只利用已有 checkpoint 检查，不重新训练。

---

## 三、Phase 5A：分数与表征互补性诊断

这是本轮主要工作。

### 3.1 资产审计

找到并复用：

- 原 Frozen WSSL checkpoint。
- Last-block adapted checkpoint。
- 135 个历史 deep-OOF 模型及预测。
- 本轮 relative reference bank。
- 本轮 reference score。
- 各 split 的 subject ID、label、seed 和 context。
- 已有 normalization、Scaler、PCA 和模型来源记录。

逐项验证 SHA256、训练 ID、验证 ID、标签和来源。

不得仅根据文件名判断资产合法。

不得重新生成历史结果。

不存在的资产标记 MISSING。

无法证明来源合法的资产标记 BLOCKED。

### 3.2 四类分数联合分析

分析：

- WSSL final DD logit/probability。
- PD reference distance。
- DD reference distance。
- Relative score。

统一正类方向为 DD。

检查：

1. 均值、标准差和分位数。
2. PD/DD 条件分布。
3. 两两相关系数。
4. Spearman 相关。
5. score margin。
6. 不同 context 的分数分布。
7. OOF 与 full-model validation 的分布差异。

所有拟合状态必须保留。

不得跨不同 encoder 坐标系直接拼接表征。

### 3.3 Subject-level error complementarity

以受试者 ID 对齐预测。

构建：

- WSSL correct / Relative correct。
- WSSL correct / Relative wrong。
- WSSL wrong / Relative correct。
- WSSL wrong / Relative wrong。

分别计算 PD、DD。

定义：

Net Rescue = WSSL wrong and Relative correct - WSSL correct and Relative wrong。

同时报告：

- gross rescue。
- newly broken。
- net rescue。
- PD net rescue。
- DD net rescue。
- 原始 WSSL stable-error 组表现。
- 原始 WSSL stable-correct 组表现。
- unstable 组表现。

必须沿用已有 stable-error 定义，并单独标明不同 aggregation 口径。

不得根据 validation 错误名单重新训练或调整样本权重。

### 3.4 双参考距离覆盖性

利用已有：

u(x) = min(d_PD(x), d_DD(x))

诊断：

- 高双类距离受试者的 WSSL 错误率。
- 低双类距离受试者的 WSSL 错误率。
- PD/DD 分类别结果。
- 与 WSSL absolute margin 的相关性。
- 与 relative score 的相关性。

仅作描述性诊断。

不能将高距离直接解释为异常疾病、错标或临床严重程度。

### 3.5 Phase 5A 输出

必须生成：

- SCORE_DISTRIBUTION_REPORT.md
- SUBJECT_COMPLEMENTARITY_REPORT.md
- REFERENCE_COVERAGE_REPORT.md
- analysis/score_distribution.csv
- analysis/subject_complementarity.csv
- analysis/reference_coverage.csv
- analysis/phase5a_integrity.json

完成后给出 H1 的支持、否定或证据不足结论。

不得自动进入训练阶段。

---

## 四、Phase 5B：融合失败机制分析

只复核已有 fusion。

不训练新模型。

### 4.1 分析原融合器

完整恢复并报告：

- logistic 系数。
- intercept。
- 输入标准化参数。
- 输入分数的方向。
- 正类标签编码。
- 拟合样本类别比例。
- OOF 预测来源。
- validation 预测来源。

检查是否存在：

- 评分方向错误。
- 类别编码不一致。
- 标准化错误。
- 不同 fold 的表示混用。
- OOF 与 full-model 分布迁移。
- 参考库覆盖不一致。
- 某一评分支配最终决策。

不得通过重新拟合系数来修复结果。

### 4.2 分析类别偏置

重点解释：

为什么 AUROC 点估计略高于 WSSL，而 DD Recall 大幅下降？

分别从：

- 排序能力。
- 概率尺度。
- 类别先验。
- 固定 0.5 决策规则。
- OOF/full score distribution shift。

进行分析。

需要明确哪些是事实，哪些只是合理推测。

### 4.3 输出

- FUSION_FAILURE_ANALYSIS.md
- analysis/fusion_parameter_audit.csv
- analysis/oof_full_distribution_shift.csv
- analysis/fusion_class_bias.csv

最终判断是否存在可信的机制解释。

不允许以事后修改阈值的方式获得新结果。

---

## 五、Phase 5C：Frozen 与 Adapted WSSL 的受试者级变化

复用已有两个模型。

### 5.1 配对错误变化

对齐每个 split、seed、subject。

统计：

- 原模型正确、适配后正确。
- 原模型正确、适配后错误。
- 原模型错误、适配后正确。
- 原模型错误、适配后错误。

分别计算 PD、DD。

### 5.2 表征漂移

使用已有冻结和适配表示。

分析：

- cosine similarity。
- L2 feature displacement。
- representation norm。
- prediction logit displacement。
- prediction flip。
- logit margin change。

检查这些变化是否与受试者纠错或错误增加相关。

所有分析必须在相同输入、相同受试者和相同预处理下进行。

不得重新提取或训练 HarNet，除非现有合法特征确实不存在；如不存在，先报告缺口，不自动重跑。

### 5.3 输出

- ADAPTATION_SUBJECT_ANALYSIS.md
- REPRESENTATION_DRIFT_REPORT.md
- analysis/adaptation_error_transitions.csv
- analysis/representation_drift.csv

不得因为观察到某些受试者改善就提出 validation-informed reweighting。

---

## 六、数据访问与统计约束

本轮必须吸收上轮暴露的审计问题。

历史 outer-test 汇总文字接触和整表/整数组读取偏差必须永久保留。

不能再声明本轮完全没有接触 outer 信息。

本轮禁止：

- 新建 outer-test loader。
- 使用 outer-test 指标选模型。
- 将 outer 信息用于拟合。
- 根据 outer 结果改变规则。
- 重新解释旧 outer 结果为本轮独立验证。

优先采用逐受试者数据读取及 ID allowlist。

对每个拟合对象记录：

- fit subject IDs。
- query subject IDs。
- excluded subject IDs。
- source checkpoint hash。
- source feature hash。
- normalization hash。
- split ID。
- seed。
- context。

统计要求：

- 同 split 内先聚合三个 seed 的正式模型指标。
- 15 splits 配对分析。
- Bootstrap 和 Wilcoxon 均报告。
- BH family 必须预先明确。
- 报告 effect size。
- 报告 seed stability。
- 同时提供 subject-cluster bootstrap。
- 不将 activity、window、seed 或重复 subject appearance 当成独立患者。

必须区分：

1. 固定已拟合模型下的不确定性。
2. 不同 seed 的模型不确定性。
3. 不同 split 的开发稳定性。
4. 真正独立外部泛化的不确定性。

不能把前三者冒充第四者。

---

## 七、本轮严格禁止的操作

1. 不训练新深度模型。
2. 不重新训练 Frozen WSSL。
3. 不重新训练 HarNet。
4. 不重新搜索 LoRA、Adapter、Gate。
5. 不重新训练 Patch-STR。
6. 不重新搜索 CE、focal、sampler。
7. 不重新搜索 fusion coefficient。
8. 不重新搜索 classification threshold。
9. 不利用 validation 错误名单进行训练。
10. 不访问 outer-test performance。
11. 不覆盖历史实验结果。
12. 不上传受保护的数据、权重、缓存或逐受试者预测。

本轮深度训练预算固定为：

0。

如果发现确实需要新的训练才能回答某个问题，只能记录为下一阶段候选，不自行启动。

---

## 八、最终需要回答的问题

请不要仅仅罗列运行结果。

以研究人员的视角给出：

1. Dual-reference 是否真正包含 WSSL 未充分利用的信息？
2. 这些信息能否稳定地修复 WSSL 原有错误？
3. Fusion 失败的直接数值原因是什么？
4. 是否有证据支持 OOF/full-model 分布迁移解释？
5. Frozen 与 adapted WSSL 的受试者级改善是否具有一致机制？
6. 当前模型最大的未解决问题是什么？
7. 哪些问题值得开展新训练，哪些应该彻底停止？
8. 是否出现了新的、可检验且具有方法学贡献潜力的研究假设？

最后给出：

- 实验状态表。
- 证据强度表。
- 下一阶段候选方向优先级表。
- 每个候选的研究假设、现有证据、单一改动、成本和停止条件。
- 完整的 correctness 与 provenance 审计报告。

**本轮目标是提高研究结论的可信度，而不是为了增加实验数量而增加实验。**

所有产物放在独立目录：

/home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111/

不得修改 mainline 和历史正式产物。

最终先在服务器完成审计，再按项目现有 Git 发布规范发布代码、协议、报告和聚合结果。

不要上传私有数据、checkpoint、cache、逐受试者预测或受保护 manifest。

完成后停止，不自动开展下一轮实验。