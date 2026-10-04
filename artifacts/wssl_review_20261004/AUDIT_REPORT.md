# WSSL-STR 现行模型重现审查

日期：2026-10-04。正式来源为服务器 `/home/zyt/deep_final`；本轮独立目录 `artifacts/wssl_review_20261004`。不读取 outer performance，不改变正式权重、输入、结构、分类规则或历史结果。

## 1. 当前可保留的最佳配置

Frozen WSSL-STR：STR-01 + official frozen HarNet10 final1024D wrist residual。分类器侧143,172个可训练参数，HarNet10固定10,457,408个参数，总数10,600,580。

|Accuracy|BA|AUROC|Macro-F1|PD Recall|DD Recall|
|---:|---:|---:|---:|---:|---:|
|.745604|.719644|.759553|.706589|.782344|.656945|

三个seed先在各split聚合，再对15个fixed development splits等权汇总。此前EMA同轨迹45次普通训练的数值日志、最佳权重、停止轮数和完整validation predictions均exact复现；本轮重新核验对应正式资产SHA和ID/label。无需为审查再次重复45次原recipe训练。旧本地 `code/MFAM-pure-deep/foundation_validation` 是不含当前structured模块的旧副本，本轮以服务器和 `current_source/` 快照核验。

## 2. 实际执行及验收

|检查|实际覆盖|结果|
|---|---|---|
|固定数据顺序、ID、label、split、recipe、45正式checkpoints/predictions哈希|15split×3seed|PASS|
|仅inner-train重新计算标准化|15次refit，对比45个checkpoint|mean/std逐位一致；class weights由本fold训练标签计算|
|原始metadata与processed契约|390 subjects，4,290 activity records，8,580 wrists|单位/通道/腕顺序/finite/时间戳递增/trim/Gyro逐点一致|
|重建SSL输入及ID映射|10,920个固定窗口|输入SHA、window counts、clip计数和cache SHA一致|
|旧/新融合前向、显式SSL优先、按ID重排、finite无效腕扰动、重载|45 checkpoints，每个一个8subject真实validation batch|logits逐位一致；archive probability最大误差1.11e-16|
|独立副本参数/cache/RNG|首个冻结checkpoint，一个真实batch|PASS，副本变更不改变原模型|
|梯度连通|fresh scratch，仅inner-train一个batch、两步|主干/SSL projection/structured梯度finite且非零；zero-init后LayerNorm第二步获梯度；cache无梯度|

原checkpoints未进行优化器更新；scratch不输出性能或保留训练权重。没有构造outer loader。预处理没有重跑HarNet，也没有重新求解全部真实Acc去趋势；因此PASS不能扩大为临床语义完整性、独立重提取feature或全Acc solver重新验证。

执行证据：`analysis/assets_normalization_audit.json`、`normalization_15refits.csv`、`assets_45checkpoints.csv`、`runtime_recipe_45checkpoints.csv`、`preprocessing_contract.json`、`fusion_properties.json`、`fusion_properties_45checkpoints.csv`。分项报告：`PREPROCESSING_AUDIT.md`、`FUSION_CODE_AUDIT.md`、`TRAINING_EVIDENCE_AUDIT.md`。

## 3. 模型设计与融合判断

64D local wrist + LayerNorm1024→Linear1024→64 frozen SSL residual，之后原 bilateral258/activity context/subject258/structured11×16 ordered176路径完全不变。最终logits为subject与structured之和。当前融合计算无多余随机状态，显式SSL接口与历史权重兼容。原闭包导致的deepcopy副本串用问题已在独立接口修复，本轮验证通过；没有发现新的实际计算bug。

有限cache和ID契约仍必要：同形状但错subject的SSL张量不能由shape check识别；NaN乘mask不能清除NaN。当前cache全部finite、ID查找/重排正确，不据此增加网络模块。固定residual本身是否全局最优无法由正确性测试证明；已失败的gate/adapter/last-block/domain/readout路线继续关闭。

## 4. 预处理边界

STR输入为完整记录Acc L1去趋势后trim48 + 原Gyro，train-only `[11,2,6,1]`标准化。实际ADMM配置lambda50/rho40/max5000，不是函数默认值；已有275batch审计全部收敛。SSL为raw Acc(g) trim48，先clip±3g，976点edge-pad至1000或2000点固定前/后窗口，100→30Hz polyphase重采样；外部encoder固定eval，没有PADS跨subject拟合。

本轮8580条核验：median effective Hz范围99.206547–100.807653；4条timestamp gap超过10倍median dt，左右腕最大clock offset .731371秒；clip仅2,044/32,310,720个值(.006326%)。当前重采样按均匀索引，没有按真实timestamps插值；这是明确的采样近似，尚无疾病类性能损害证据，不自动改输入或删除记录。滤波后可能overshoot，clip前边界不等于输出严格±3g。

## 5. 训练是否充分

已有真实best/last eval及45份日志：best epoch2–24，停止epoch14–36，45/45均best+12，0/45达到max50。

|实际checkpoint|train eval BA|validation BA|train eval AUROC|validation AUROC|train eval CE|validation CE|
|---|---:|---:|---:|---:|---:|---:|
|best|.913406|.719644|.964805|.759553|.239857|.679387|
|last|.999546|.658764|.999988|.745135|.013802|1.099845|

这支持后期过拟合，不支持“因为epoch cap训练不足”。Dropout-active online train CE .291046与best eval CE .239857分开解释。最佳点按BA选中，因此邻域BA峰值是选择条件，不证明停止规则最优。仅best/last权重可用，中间日志不等于中间checkpoint或可恢复历史EMA。

## 6. 超参数选取覆盖及文档差异

实际WSSL recipe：AdamW lr2e-4/wd1e-4/betas.9,.999，batch8，cosine50/minLR1e-6，clip5，FP32，balanced CE无smoothing，dropout .1/.1/.2，普通validation BA严格提高选择best，patience12，固定argmax(DD probability>0.5，tie=PD)。

- 通用base YAML写AMP=true，但STR扩展配置/真实checkpoint为false；无实际冲突。
- 通用threshold source写inner_oof_only，但当前inner inference执行固定argmax/.5；OOF阈值仅属于已拒绝Phase3C诊断。报告按实际执行规则记录，不修改历史配置。
- Phase2的LR/WD/loss/sampling/augmentation全部针对STR-only。Phase3C的2e-5只用于HarNet last block，不是frozen WSSL分类器LR对照。Phase3D/3E/4/EMA沿用2e-4。因此不能据STR负结果声称当前WSSL recipe已接近全局最优。
- 历史完整矩阵审计见`HISTORICAL_MATRIX_AUDIT.csv/json`；引用baseline在不同报告中重复出现是复用，不能算独立重现。Phase2数值gate有事后明确的时间限制，不改写成全程预注册。

## 7. 本轮审查后的唯一受控测试

在当前用户授权的“审查后测试”范围，只检验现有frozen WSSL分类器初始LR **2e-4 vs预先固定1e-4**。这填补target-specific覆盖缺口，没有证据预设会提高。保留网络、外部encoder、输入、归一化、loss、batch、scheduler形式/最低LR、阈值与BA选择/停止规则；不同时改变其他变量，不加EMA、不重开loss/augmentation/模块。固定协议与gate须先冻结再训练；正式结果写独立目录，旧baseline仍冻结。若失败停止该LR对照，不追加LR系数或patience补救。

## 8. 统计与研究边界

45runs不是45独立样本，15split共享subjects；bootstrap/paired检验只描述重复development稳健性，不是独立外部验证。已有多轮相同development选择存在selection optimism。本轮不使用FOE01、历史隔离outer、H1或diagnostic probes制定规则。当前最佳是已测试候选中的retained best，不能宣称找到全局最优预处理、融合或training recipe。
