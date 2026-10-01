# DSG-01：Cross-subject Disease Subspace Stability Diagnosis

日期：2026-09-26。状态：**完成只读诊断；不启动 activity-conditioned cross-subject disease consistency training。** 本报告只使用既有 V8-GN/STR-01 的 45+45 个 fixed inner-development checkpoint 与对应 train/validation 数据。未训练模型、未改 forward、recipe、split、checkpoint 或既有正式结果。所有新文件位于本独立 artifact 目录。

## 1. 资产审查与 smoke test

- 固定 `5 outer development contexts × 3 inner folds`，每模型 seeds 42/43/44 各 15 个 checkpoint，共 90 个，均 strict load 成功。split 文件 SHA-256 为 `b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`。每个 normalization 的 `fitted_on=explicit_training_subjects_only` 且保存的 subject IDs 与该 inner-train 集合完全一致；train/validation 无交叠。
- activity 顺序固定为 CrossArms、DrinkGlas、Entrainment、HoldWeight、LiftHold、PointFinger、Relaxed、RelaxedTask、StretchHold、TouchIndex、TouchNose；腕顺序为 left/right。数据张量按 `[B,11,2,6,T]` 输入。
- **文档/接口命名差异：**现有 `forward()` 返回的 `activity_embeddings` 是 bilateral 258D features 加 activity-ID embedding 并经过 LayerNorm 后的 context；STR residual 实际使用此前的 `11×258D` bilateral activity features。新提取脚本通过 aggregator 的只读 forward-pre-hook 获取后者，同时保存两种 stage。原 subject embedding 为 258D，STR projected tokens 为 `11×16D`，ordered representation 为 176D；base/structured/final logits 均为 2D，`final=base+structured`。V8 的 258D subject embedding 是 pre-logit，base/final logits 相同。
- 同一 checkpoint、同一输入上，hook 前后 V8 与 STR 的 base/final logits 最大差均为 **0**；STR residual logits 差为 **0**，final 分解最大误差 `2.98e-8`。180 个新 train/validation archive 与旧正式归档按 subject ID、label 对齐，final logits 最大差 **0**。见 `smoke_test.json` 与 `extraction/extraction_checks.csv`。
- 每个 embedding archive 明确含 subject ID、label、activity mask、split/seed/role（文件名）；`extraction/records.csv.gz` 为每个向量提供 model、seed、outer、inner、train/validation role、subject ID、label、activity ID、representation type 和 archive row index。全部匹配依赖 ID，未依赖数组位置。

## 2. 预设分析与统计边界

- A：在两个实例共同的相同 **training subjects** 上算 Linear CKA，输出 45×45 pairwise matrices。既保存普通 CKA，也以 unbiased-HSIC CKA 对不同维度比较作有限样本偏差核查。CKA 描述整体几何，不能单独解释为 disease-specific 稳定性。
- B/D：各实例只用自己的 inner-train 计算 PD–DD centroid direction 并拟合 balanced linear logistic disease probe。跨实例先在各自 training subjects 上无标签固定 PCA-16，然后用共同 training subjects 做无标签 Orthogonal Procrustes；报告 centroid/probe direction cosine 和两维 disease subspace 的 principal angles。原 latent raw cosine 仅存描述项。固定 PCA-16 避免 258/176/2838D 在约 104 个共同训练受试者上直接 Procrustes 的秩不足；该降维会丢失部分方向信息，是解释边界。
- C：A→B 的 probe 在 A train 拟合，先做上述无标签 alignment，只对 `B.validation − A.train` 受试者评分；同一批合格 B.validation 同时由 B native probe 评分。要求至少 20 人且每类至少 5 人，否则 NA。STR decision stage 的 630 个同 seed 跨 split 有向比较中，540 个有效；90 个同 outer context 不同 inner fold 的比较合格受试者为 0，已标记 NA。有效集 40–66 人，至少 PD 26、DD 8。记录两类人数、native/transfer BA/AUROC 与 drop。
- E：三个 seed 先在相同 `(outer,inner)` 聚合，以 15 个 split 为推断单位。Spearman ρ 本身为效应量，bootstrap 95% CI 以 15 个 split 重采样；每组 30 个 stability/performance 相关做 BH-FDR。模型和 activity 配对比较也保留负结果与 BH-FDR。
- 稳定错分 primary 固定为：先在同一 validation appearance 聚合三个 seed，再按每名受试者的四次 appearance，错误率 ≥0.75 为 stable-error、≤0.25 为 stable-correct。历史 12 个 individual seed-fold appearance 定义仅作 sensitivity；两套人数不混用。
- **outer 边界：**没有建立 outer loader，没有读取 outer 信号、临床标签、prediction、metric 或 outer-final artifact，也没有用 outer 做拟合或模型选择。既有 inner-stage `split.json` 含 test-subject 列表字段，但脚本仅取 train/validation subject IDs；本轮不解释或输出该列表。

## 3. A/B/C：跨实例稳定性与 transfer

下表的 CKA 为 unbiased-HSIC 版本；direction/probe cosine 均在无标签 PCA-16 + Procrustes 后比较。数值为同 seed、不同 split 的 15-split 汇总。BA drop 是同一合格 target validation 子集上的 `native−transfer`。

| Model / stage | CKA | centroid cosine | probe cosine | native probe BA / AUROC | transfer BA drop |
|---|---:|---:|---:|---:|---:|
| V8 activity context, 11×258D | 0.7914 | 0.9076 | 0.7962 | 0.6757 / 0.7335 | 0.0226 |
| V8 subject, 258D | 0.5102 | 0.9166 | 0.5245 | 0.6164 / 0.6590 | 0.0307 |
| STR activity context, 11×258D | 0.7881 | 0.9083 | 0.8404 | 0.6752 / 0.7427 | 0.0104 |
| STR original subject, 258D | 0.5456 | 0.9298 | 0.5608 | 0.6147 / 0.6559 | 0.0139 |
| STR structured ordered, 176D | 0.7153 | 0.9263 | 0.6683 | 0.6611 / 0.7166 | 0.0307 |
| STR combined decision, 434D | 0.7013 | 0.9334 | 0.7215 | 0.6582 / 0.7057 | 0.0294 |

STR combined vs V8 subject 的 train-only probe BA/AUROC 增量为 `+0.0418/+0.0468`（15 split paired BH `q=0.0298/0.0028`），与正式分类结果 STR BA/AUROC `0.6972/0.7176` 高于 V8 `0.6707/0.6800` 一致，确认更强的 decision-accessible disease separability。普通 CKA 在这两个 stage 为 0.7277/0.5351；unbiased 核查后仍为 0.7013/0.5102（差 `+0.1911`，15/15，`q=0.0005`）。CKA 维度不同，不能把这项差直接解释为 disease-specific 不变性。

STR combined 的 centroid/probe direction cosine 为 0.9334/0.7215，V8 subject 为 0.9166/0.5245；配对增量 `+0.0167/+0.1970`，BH `q=0.0215/0.0007`。两维 principal-angle 均值由 43.10°降至 37.70°（`q=0.0007`）。所以 STR 不仅更可分，而且在本固定 PCA-16 对齐下**并未恶化**决策表示的方向一致性。

但 probe transfer 仍有 drop：STR combined BA drop `0.0294`，14/15 split 为正，Wilcoxon `p=0.00018`；V8 subject 为 `0.0307`，14/15 为正。二者差 `−0.0013`，BH `q=0.9642`；AUROC drop 为 `0.0147` vs `0.0203`，差异亦未通过 BH。STR 并未清除跨实例 probe transfer 问题。STR combined 有效目标集上的 native/transfer BA 为 `0.6712/0.6418`，AUROC 为 `0.7187/0.7040`。这包含 probe 参数、PCA 和 alignment 的迁移误差，不能全归因于 backbone instability。

同 split、不同 seed 对照显示 STR combined centroid cosine `0.9647`，跨 split 为 `0.9334`（差 `−0.0314`，15/15，BH `q<0.001`），说明训练受试者组成相关的 centroid 方向变化可以检测到；但 probe cosine 为 `0.7135` vs `0.7215`，无显著差异。普通/校正 CKA 的跨 split 均值反而高于同 split 跨 seed（校正 CKA `0.7013` vs `0.6419`），提示随机初始化/训练本身贡献明显。对照组共同训练受试者数约 208，而跨 split 约 104，不能把两个组的绝对差当作纯粹的受试者组成因果效应。

## 4. D：activity-conditioned disease structure

11 个 activity 均完成 raw bilateral 258D、activity-context 258D 和 STR projected-token 16D 的 train-only centroid/probe 与对齐分析。raw/context 的平均 centroid direction cosine：V8 约 `0.746`，STR 约 `0.722`；按 split 先聚合 11 个 activity，STR−V8 为 `−0.0240/−0.0244`（各 1/15 split 改善，8 项 aggregate-family BH `q=0.00073`）。但平均 probe direction cosine、probe BA、AUROC 无系统差异（均 `q>0.55`）。

单 activity 的方向差异有升有降：raw stage 中 DrinkGlas `+0.0104`、LiftHold `+0.0225`，Entrainment `−0.0444`、PointFinger `−0.1615`、TouchIndex `−0.0527` 通过各指标 11-activity BH；context stage 4 项通过，方向同样混合。**没有任何单 activity 的 probe BA/AUROC 或 probe direction 改善通过 BH。** STR 收益不能描述成全面更稳定的 distributed activity-conditioned disease structure；也不能根据单 activity 差异删活动或重启 readout 搜索。

## 5. E：稳定性与最终分类性能

每模型/stage 的 6 个 predictor × 5 个正式指标，在 15 个 seed-first split 上分析；**90 个相关中无一通过各组 BH-FDR**。重点 STR combined：

| Predictor | Outcome | Spearman ρ | bootstrap 95% CI | BH q |
|---|---|---:|---:|---:|
| unbiased CKA | BA | +0.068 | [−0.415, +0.529] | 0.838 |
| unbiased CKA | AUROC | +0.036 | [−0.454, +0.486] | 0.899 |
| unbiased CKA | DD recall | +0.373 | [−0.129, +0.753] | 0.397 |
| centroid direction cosine | DD recall | −0.097 | [−0.517, +0.353] | 0.838 |
| probe direction cosine | DD recall | −0.352 | [−0.749, +0.235] | 0.397 |
| transfer BA drop | BA | +0.354 | [−0.204, +0.769] | 0.397 |
| transfer BA drop | DD recall | +0.164 | [−0.431, +0.675] | 0.729 |

这些区间宽，不能当作“没有关系”的等效性证明；当前数据不支持“stability 改善会稳定提升 BA/AUROC/DD recall”的前提。15 个 split 还共享受试者，split bootstrap CI 应视为开发协议内的不确定性描述，而非新队列的置信区间。

## 6. Stable-error primary 与 sensitivity

Primary seed-first：V8 为 stable-correct 266、stable-error 85、unstable 39；STR 为 284、74、32。历史 12 seed-fold sensitivity 分别有 54 与 42 名 stable-error，仅用于与旧报告对照。

以独立 subject 为单位汇总四次 validation appearance，STR stable-error 相比 stable-correct 的 true-label disease-axis margin、centroid-distance margin、train-only probe margin 和 base/structured/final logit margins 都更偏向错误类；这些组差异 BH 显著，但 final logits 参与 stable-error 定义，本身具有循环性，不能作为机制证明。更有区分力的是跨 appearance 一致性：STR stable-error 的 disease-axis/probe margin 为正的 appearance 比例为 `0.152/0.189`，unstable 为 `0.539/0.508`；stable-error 的 probe-margin SD 为 `2.143`，**低于** unstable 的 `2.918`（BH `q=0.0032`），disease-axis SD 没有显著升高。V8 的同类结果方向相似。

因此持续错分更符合“多个 split 均看到对侧类别样的表示/PD–DD 表型或标签空间重叠”，而非仅由剧烈 split-to-split 表示波动造成；这只是与现有 phenotype/data audit 相容的机制判断。缺少严重度、用药、共病、逐活动执行质量和外部队列，不能证明临床重叠或标签错误。

## 7. 八个问题的结论与停止决定

1. **更强 disease separability？** 是。正式分类及 train-only probe 均支持 STR combined 优于 V8 subject；STR original subject path 单独不改善。
2. **稳定的 cross-subject/cross-split instability？** 有可检测的 centroid 方向变化和 probe transfer loss，但无证据将其归为一个独立、稳定的 disease-discriminative subspace 故障；probe direction 与同 split 跨 seed 对照未分离，alignment/抽样也有影响。
3. **STR 改善 instability？** 对 decision representation 的 CKA 和对齐 probe/centroid direction 有改善；transfer BA drop 与 V8 持平，activity 方向平均下降。因此不是全面改善。
4. **与 BA/AUROC/DD recall 的稳定关系？** 未建立；预设相关均未通过 BH，重点 DD recall CI 跨 0。
5. **最明显 stage？** `11×258D activity context → 258D original subject` 的跨实例 CKA 降幅最大（V8 `0.791→0.510`，STR `0.788→0.546`）；STR ordered path 令最终 decision input 回到 `0.701`，但该 stage 仍有 transfer drop。
6. **activity-conditioned 系统差异？** centroid direction 平均更低约 0.024，单 activity 有升有降；probe BA/AUROC 和 probe direction 无系统提升。
7. **persistent errors？** 更符合持续对侧类别表示/phenotype overlap，而非单纯表示不稳定；临床原因尚不可判定。
8. **启动 consistency experiment？** **否。** 虽有可检测变化与 transfer drop，但未同时满足“明确 disease-specific instability”及“该 instability 与分类退化稳定相关”两个门槛。停止本轮 consistency regularization 方向；不以 DSG-01 为依据训练 DANN、MMD、prototype、supervised contrastive 或新模块。

## 8. 限制与可复核文件

分析完全复用既有 development 分割，同一受试者在不同 context 重复出现；DD 小亚型不足以用于稳定亚组推断。PCA-16、Procrustes、linear probe 与 transfer 的量度不能被解释为模型内真实的唯一 disease subspace；CKA 衡量全体变异，并非疾病专属。低维 aligned direction 与原高维 cosine 不同，后者只存作描述。不同维度之间的 CKA 虽做 unbiased 核查，仍需避免因果解释。

新增脚本：`scripts/extract.py`、`scripts/analyze.py`、`scripts/cka_bias_audit.py`、`scripts/summarize.py`、`scripts/error_consistency.py`、`scripts/seed_split_control.py`。核心产物：`asset_audit.json`、`smoke_test.json`、`extraction/embeddings/`、`extraction/records.csv.gz`、`analysis/cka_unbiased_matrix_*.csv`、`analysis/alignment_transfer_pairs.csv`、`analysis/activity_*`、`analysis/split_metrics_15.csv`、`analysis/stability_classification_correlations.csv`、`analysis/stable_error_*` 和 `analysis/seed_split_control_*`。原始日志也保留。未改 `/home/zyt/MFAM` 或 `deep_final/foundation_validation` 的冻结代码；诊断产物放在 `deep_final/artifacts`，因为该目录持有现行正式 STR/V8 checkpoints 与报告。
