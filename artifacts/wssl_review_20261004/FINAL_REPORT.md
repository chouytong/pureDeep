# WSSL-STR 现行模型审查与单一 learning-rate 对照：最终报告

2026-10-04。**COMPLETE / REJECT LR1e-4；保留原 frozen WSSL-STR、LR2e-4。** 本轮审查现有模型及其recipe，没有增加模型结构、外部encoder、loss或augmentation，也没有读取outer performance。

## 1. 当前保留配置与结论

WSSL-STR = STR-01 + official frozen HarNet10 final1024D wrist residual。11activities、双腕顺序、STR/structured路径、train-only normalization、balanced CE、原0.5/argmax决策、普通BA checkpoint选择规则不变。143,172 classifier-side trainable +10,457,408 frozen HarNet =10,600,580 total parameters。

|Metric|保留原LR2e-4|唯一对照LR1e-4|对照−原配置|
|---|---:|---:|---:|
|Accuracy|.745604|.741614|−.003990|
|BA|.719644|.725133|+.005489|
|AUROC|.759553|.758559|−.000994|
|Macro-F1|.706589|.706159|−.000431|
|PD Recall|.782344|.764914|−.017430|
|DD Recall|.656945|.685352|+.028408|

低LR有BA/DD正收益，同时存在PD识别的trade-off；没有共同提高BA/AUROC并满足预固定其他门槛。因此不替代正式baseline。不能把本结果写成“低LR没有任何收益”，也不能证明2e-4或AdamW为全局/近全局最优。

## 2. 四方面审查：实际覆盖

### 数据预处理

- 390development metadata、4,290activity记录、8,580腕侧raw/processed契约全部PASS：单位、通道、左右腕/活动顺序、有限值、递增时间戳、有效976/2000长度、trim48及Gyro逐点一致。
- 重建10,920固定SSL窗口输入，SHA、clip计数、cache subject-ID映射/activity order/window counts一致。没有重新运行HarNet或全部真实Acc solver；不能宣称本轮独立重新提取feature或证明临床语义完整性。
- 15个split仅由inner-train重新拟合mean/std，与45正式checkpoints逐位一致；balanced CE weights只由本fold训练标签计算。
- STR的Acc L1去趋势+Gyro路径与SSL raw Acc(g)路径有意不同。实际ADMM参数lambda50/rho40/max5000，不是函数默认值；已有275batch solver报告全部收敛。
- 均匀索引重采样没有按真实timestamps插值：4条大gap，腕侧最大clock offset .731371秒，Hz范围99.206547–100.807653。clip2,044/32,310,720值(.006326%)。这些是质量近似/限制，不是已证明的疾病类偏差；没有据此删记录或改输入。

### 模型设计与特征融合

实际路径：local wrist64 + LayerNorm1024→Linear1024→64 frozen-SSL residual，再经原bilateral258/activity context/subject258和structured11×16 ordered176，final logits=subject+structured。未发现需要修改正式结构的实际计算错误。

45冻结checkpoint各一个8subject validation batch的旧/新前向与重载逐位一致；archive probability最大误差1.11e-16。显式SSL优先、按ID重排、finite无效腕扰动检查通过。首个checkpoint独立副本参数/cache/RNG隔离通过。fresh training-only两步scratch梯度连通检查通过，未报告其性能或保存训练模型。原闭包copy问题使用此前已验证的独立接口解决；原正式权重不改。

此覆盖不是每个validation record/all possible inputs，也不能证明固定融合理论最优。有限cache和正确ID配对仍是契约；shape check不能识别同形状错subject数据，NaN×mask不能清除NaN。当前cache有限、ID映射正确。

### 训练充分性与超参数覆盖

此前EMA普通参数对照已把45原WSSL训练数值轨迹、selected weights、停止epoch及完整validation predictions exact复现；本轮重新核验原source/cache/split/checkpoint/prediction/norm身份，复用该证据，**没有再训练45原baseline**。

原模型真实best/last eval（Dropout关闭）：train BA .913406→.999546，validation BA .719644→.658764；train AUROC .964805→.999988，validation .759553→.745135；train主CE .239857→.013802，validation .679387→1.099845。45/45都best+12停止，0/45到max50。这支持后期过拟合，不支持由epoch cap造成的训练不足。BA-best邻域峰值受BA选择条件影响，不证明stopping rule全局最优。仅best/last权重可用，不能恢复中间checkpoint或历史完整EMA。

真实recipe为FP32/AdamW lr2e-4/wd1e-4/betas(.9,.999)、batch8、cosine50/minLR1e-6、clip5、balanced CE/no smoothing、dropout(.1,.1,.2)、strict ordinary BA best/patience12。通用YAML的AMP=true被STR覆写false；OOF threshold声明不等于实际primary推理，当前为argmax/.5，OOF属于已拒绝诊断。旧本地foundation副本缺当前structured模块，本轮以服务器代码/正式资产为准。

Phase2 LR/WD/loss/augmentation针对STR-only，Phase3C的低LR针对HarNet last block，不是frozen-WSSL classifier LR。因此填补这个真实覆盖缺口，不能将历史STR负结果自动归为WSSL结果。其他未比较超参数的最优性仍未知，未据此开展grid。

## 3. 唯一受控测试及技术验收

固定协议 `LR_CONTROL_PROTOCOL.md` 与source/runner/analysis/launcher哈希在正式结果前冻结并Git归档。只改变classifier initial LR2e-4→1e-4，三seed42/43/44×15固定划分全部45完成，没有新reference refit。原architecture/HarNet/cache/normalization/loss/optimizer其他参数/batch/scheduler floor/threshold/max50/patience12不变，各自按同一普通BA规则选epoch。

初始化旧/新构造的全部参数及训练batch logits exact；0optimizer updates。原engine一个batch/一个epoch smoke通过。单类smoke AUROC None审计边界在正式前修正并重复smoke；首轮产物保留，不删除，不改变forward/recipe/gate。正式后未修改候选或统计规则。

最终45stage全部audit PASS，8核心artifact SHA齐全，full-current-source/metric/loss/config/official HarNet/cache/split guard PASS。原45checkpoint/prediction/normalization保持锁定SHA。训练正常terminal、GPU无compute进程。无EMA、encoder adaptation、outer loader或其他候选。见`analysis/final_integrity_audit.json`。

LR改变也改变cosine轨迹、相对floor、有效AdamW每步shrinkage及BA-selected训练时长；本轮是固定recipe中的initial LR总效果，不单独归因于其中某个机制。

## 4. 配对统计：三seed先平均，15split为单位

|Metric|mean delta|improve/tie/worse|paired bootstrap95%CI|Cohen dz|Wilcoxon p|BH q(6项探索)|BH q(primary)|
|---|---:|---|---|---:|---:|---:|---:|
|Accuracy|−.003990|6/1/8|[−.012139,+.004521]|−.2330|.4143|.6215|—|
|BA|+.005489|9/0/6|[+.000352,+.011512]|.4861|.0946|.2231|.1892|
|AUROC|−.000994|6/0/9|[−.007181,+.005459]|−.0766|.5995|.7194|.5995|
|Macro-F1|−.000431|8/0/7|[−.006884,+.006345]|−.0320|.9780|.9780|—|
|PD Recall|−.017430|5/0/10|[−.034942,+.001872]|−.4625|.1116|.2231|—|
|DD Recall|+.028408|11/1/3|[+.006280,+.049196]|.6518|.0278|.1669|—|

10,000paired bootstrap draws，RNG20261004。BA正bootstrap CI不能等同于Wilcoxon/BH通过；DD raw p/CI支持正方向，但六指标BH未确认。AUROC/F1/PD的CI跨0，不称其显著恶化。保持预固定point-estimate retention门槛，不后验更换统计方法。

Joint gate失败项：BA仅9/15(<10)；AUROC mean负且仅6/15；F1 mean轻微负；PD Recall drop .01743超过允许.01。通过项：BA mean/CI正，DD降幅限制、BA/AUROC within-split seedSD≤1.25reference。**REJECT**不依赖事后新增criterion。

## 5. 种子波动、轨迹与错误变化

|项目|原2e-4|低LR1e-4|
|---|---:|---:|
|BA splitSD|.034563|.031054|
|AUROC splitSD|.043921|.041510|
|BA mean within-split seedSD|.021603|.022919|
|AUROC mean within-split seedSD|.021741|.022825|
|BA SD of seed means|.006714|.007247|
|AUROC SD of seed means|.001812|.004762|
|selected epoch range/median|2–24 /10|4–34 /12|
|stop epoch range/median|14–36 /22|16–46 /24|
|epoch50 cap hits|0/45|0/45|
|best online train mainCE|.291046|.358802|
|best validation mainCE|.679387|.631339|
|last online train mainCE|.019575|.102103|
|last validation mainCE|1.099845|.858135|
|three-seed prediction unanimity|.732973|.699774|
|mean pair seed prediction agreement|.821982|.799849|
|mean pair seed score Spearman|.773440|.782181|

低LR的primary within-seed SD微升6.1%/5.0%，通过原1.25限制，不能事后用另一SD定义替换gate。Accuracy/F1/recall的其他SD也全部保存在`lr_summary.csv`。上述新LR训练loss是online Dropout-active值，不冒充train eval或量化其泛化差距；验证loss下降不证明校准最优或新疾病信息。

每split先平均seed后再平均15split的描述性error计数：

|类别|原错误数|低LR错误数|被纠正|新增错误|net corrected|
|---|---:|---:|---:|---:|---:|
|PD|16.022|17.311|3.844|5.133|−1.289|
|DD|10.444|9.578|2.400|1.533|+.867|

这是每validation unit的平均计数，**不是独立人数或整个cohort独立被纠正人数**。表现更符合PD/DD识别trade-off；不能将其归因为特定activity、phenotype、aggregation信息损失或融合机制。本轮未启动这些mechanism/subgroup搜索。

## 6. 最终回答与停止决定

1. **当前模型可重现，未发现新的实质预处理/融合计算bug。** 已修copy hazard的独立接口可继续用于相同架构。
2. **原WSSL不是明显训练不足。** 实际后期训练更拟合、validation更差；没有cap截断依据。
3. **LR确实影响class-recognition trade-off。** 降为1e-4改善BA/DD，但不满足BA/AUROC联合稳定收益，且PD下降超过限制。
4. **保留原frozen WSSL-STR、LR2e-4。** 它是此已测试历史与joint objective下的retained best，不是全局最优性证明。
5. **停止本次LR coefficient/WD补偿/patience/extraepoch/optimizer/scheduler扩展和与EMA等候选组合。** 不自动训练新模型，不重新打开已停止readout/attention/gating/adapter/domain/frequency/invariance/loss/sampling/augmentation/information-preservation路线。
6. 下一个有效工作是整理现有论文方法/实验与规划真正独立的验证；本轮没有足够联合证据支持继续扩大训练或机制模块搜索。

## 7. 统计、临床与development/outer边界

15splits共享subjects/contexts，45seed-runs、窗口、activity/pairs不是独立样本。CI/p/BH/dz仅描述重复development稳健性，不能当作独立外部证据。多阶段复用同一development产生selection optimism。物理metadata一致、solver收敛、有限值不证明各设备校准或全部运动phenotype完整保留。

FOE01和历史隔离outer信息没有用于本轮选择、调试或验证；不把旧outer结论归于新的WSSL/低LR。本轮提供实现正确性和development-side recipe证据，不新增独立generalization或机制证明。没有H1、handcrafted、teacher/student或分析性probe候选。

## 8. 文件、版本控制及验收证据

新增独立六个脚本：`test_preprocessing_contract.py`、`audit_assets_normalization.py`、`test_fusion_properties.py`、`run_lr_control.py`、`launch_lr_matrix.py`、`analyze_lr_control.py`。分项审查/协议/smoke/本报告与小型aggregate CSV/JSON均独立归档。原formal source/code/calculation/checkpoints/results不修改。数据、HarNet/cache、权重、individual predictions/labels、raw logs不上传Git。

- 审查PASS commit `758159f8d9709e43fa6ff2f92ec8d38f13ae8e89`，tag `wssl-review-audit-pass-20261004`。
- 正式前protocol/smoke commit `81fe71537a1a5cd22bc172d4e42d5bf6dc7bf9b2`，tag `wssl-review-lr-protocol-smoke-20261004`。
- Final closure commit/push/tag在README/handoff追加本真实结果后验证，外部publication receipt记录最终SHA，避免自引用commit哈希。
- 实际统计：`lr_seed_split_metrics.csv`、`lr_15split_seedfirst.csv`、`lr_summary.csv`、`lr_paired.csv`、`lr_prediction_agreement.csv`、`lr_errors_45runs.csv`、`lr_errors_15split_seedfirst.csv`、`lr_error_summary.csv`、`lr_decision.json`。
- 独立结果与completion复核：`FINAL_INTERPRETATION_REVIEW.md`、`COMPLETION_EVIDENCE_REVIEW.md`。最终source/input/45stage完整性见`final_integrity_audit.json`。
