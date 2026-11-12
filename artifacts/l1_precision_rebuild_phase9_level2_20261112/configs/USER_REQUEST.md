# Phase 9 Level 2：L1 数值正确性修复、独立数据版本重建与关键模型复核

## 一、任务定位与本轮授权

你是一名具有丰富研究经验的数值优化、可穿戴 IMU 信号处理、时序深度学习和帕金森病数字生物标志物研究人员。

请在当前 PADS PD/DD 项目中，实际完成以下工作：

1. 修复 Acc L1 去趋势求解器的数值精度问题。
2. 消除有限迭代停止导致的跨记录、跨受试者批次依赖。
3. 在新目录生成处理后的独立数据版本。
4. 对新数据按原规则重新计算各 inner-train 的 normalization。
5. 使用新数据重新训练 Frozen WSSL-STR。
6. 随后重新训练 STR-01，复核新预处理下外部 SSL 表征的增量。
7. 分别报告数值正确性、数据协议正确性和分类性能变化。

本轮不是继续只读审计，也不是只写实施方案。

本提示词明确授权执行上述修复、真实信号重处理、train-only normalization 重计算及两组关键训练，但必须严格按照阶段门槛依次执行。

正式训练预算：
- Frozen WSSL-STR：15 splits × 3 seeds = 45 个正式 run。
- STR-01：15 splits × 3 seeds = 45 个正式 run。
- 合计最多 90 个正式 run。
- 不增加其他结构、损失、阈值或滤波强度候选。

WSSL 完成后，只要正确性、来源和训练执行检查通过，就继续开展已登记的 STR 复核。
不能因为 WSSL 指标没有提高，就取消 STR 对照；也不能因为指标较高，就跳过正确性检查。

如果遇到真实的权限、来源、资源或正确性阻断，应停止受影响阶段并报告 BLOCKED。
不能越过门槛，也不能虚构完成状态。

## 二、项目位置与固定依据

GitHub：
https://github.com/chouytong/pureDeep

Phase 9 审计正式依据提交：
4da80813a98586acf23a89507359e4ca93689289

项目根目录：
/home/zyt/deep_final

冻结主线：
/home/zyt/deep_final/mainline

现有数据：
/home/zyt/deep_final/data

Phase 9 审计目录：
/home/zyt/deep_final/artifacts/pads_raw_preprocessing_audit_20261112

本轮新实验根目录：
/home/zyt/deep_final/artifacts/l1_precision_rebuild_phase9_level2_20261112

请首先核对服务器、GitHub 和上述依据提交的状态。

如果 main 已出现后续提交：
- 记录当前 SHA。
- 比较与本次依据提交之间的相关差异。
- 不自动覆盖用户后续修改。
- 不将不同版本的模型、配置和数据处理代码混用。
- 与本任务有关的实质冲突应记录并阻断对应步骤。

保留模型名称 Frozen WSSL-STR，但必须明确：
Frozen 指 HarNet10 冻结，不是将整个任务内分类网络冻结。

## 三、必须阅读的材料

优先阅读并交叉核对：

1. AGENTS.md。
2. mainline/README.md。
3. CONTEXT_HANDOFF.md。
4. conclusions 中当前模型及正式报告索引。
5. Phase 9 FINAL_REPORT.md。
6. L1_SOLVER_CORRECTNESS_REPORT.md。
7. LEVEL2_EXPERIMENT_PROTOCOL.md。
8. DATA_PIPELINE_MAP.md。
9. CORRECTNESS_AND_PROVENANCE.md。
10. ACCELERATION_SEMANTICS_ADDENDUM.md。
11. SENSOR_COORDINATE_AND_UNIT_AUDIT.md。
12. FILTER_AND_SPECTRAL_AUDIT.md。
13. STR-01 与 Frozen WSSL-STR 的实际正式训练配置和来源记录。
14. 原始数据准备入口、L1 实现、normalization、Dataset、collate、模型和评估代码。
15. Phase 8 的保护清单和 Phase 9 更新后的保护清单。

重点检查实际文件，而不是只根据报告文字推测：

mainline/src/datasets/preprocessing.py

以及报告指向的原始生产数据准备脚本。

注意：
preprocessing.py 的函数默认参数，不一定等于原生产准备时传入的实际参数。
必须从实际调用、配置和历史日志恢复真实参数。

不要重新进行一轮无关文献调研，也不要重复全部 Phase 3–8 模型审计。

## 四、已确认问题与不能扩大解释的部分

已知问题：

A. 旧 L1 数学目标为：

P(x) = 0.5 * ||y - x||_2^2 + 50 * ||D2 x||_1

B. 旧实现使用整个 batch 的联合残差和联合停止条件。

C. 同一条合成信号独立处理与混批处理，trend 可出现约 0.0082g 差异。

D. Phase 9 对历史真实输出的分析发现大量保守目标次优见证。

E. 原生产处理在 subject split 之前合批，存在跨记录数值依赖。

这些发现不等于：

- 原始 PADS 大部分信号损坏。
- 所有被标记记录都是噪声。
- 已经证明诊断标签泄漏。
- 已经证明历史分类指标虚高。
- 已经证明精确 L1 一定提高分类性能。
- 已经证明 L1 去趋势本身是最适合疾病分类的处理。

本轮解决的是：
“将原本声明的数学处理实现为可验证、逐记录独立的数值过程。”

不是：
“搜索一套让 BA 最高的新滤波器。”

## 五、本轮协议修订必须显式归档

生成 PROTOCOL_AMENDMENT.md，至少写明两项修订。

修订一：
原 Level 2 的性能实验预算为 45 个 WSSL run。
本轮增加 45 个 STR-01 run，用于复核修复后 SSL 增量。
新增对照在任何新分类结果出现前登记。

修订二：
Acc 是否包含完整重力仍可保留为 UNKNOWN，但仅限以下条件同时成立时，允许继续固定目标的数值修复及对照：

- 数值输入、单位、通道和原始文件来源可验证。
- 不补加重力。
- 不改变坐标轴。
- 不改变原 L1 目标。
- 不改变 HarNet 输入。
- 不将修复解释为已验证的物理重力去除。

如果连原始文件版本、数值输入单位或通道定义都无法核验，则不能继续。

涉及重力重建、姿态变换、删除 L1、替换滤波器或改变 HarNet 输入语义的工作仍然不在本轮授权范围内。

旧协议、旧结论和旧草稿均保留，不能重写成“原来已经授权并执行”。

## 六、保护历史资产与建立独立工程入口

不得修改或覆盖：

- 原始 PADS 文件。
- 历史处理信号。
- 历史 normalization。
- 历史 split manifest。
- 正式模型源码快照。
- 旧 checkpoint。
- 旧预测和报告。
- 冻结的 HarNet 特征。
- 当前模型指针或历史模型身份。

本轮在独立目录增加实现和训练入口。

推荐结构：

l1_precision_rebuild_phase9_level2_20261112/
    PROTOCOL.md
    PROTOCOL_AMENDMENT.md
    README.md
    scripts/
    tests/
    configs/
    audit/
    analysis/
    reports/
    logs/
    figures/
    private/
        manifests/
        numerical_certificates/
        processed_signals/
        normalization/
        checkpoints/
        predictions/
        subject_diagnostics/

private 目录默认不发布。

可以复用只读 mainline 代码，或建立有来源 hash 的隔离源码副本。
禁止通过硬链接修改导致历史文件被连带修改。
禁止修改全局安装包来改变旧实验行为。

训练入口必须显式接收：
- 数据版本。
- manifest。
- normalization 路径。
- 模型类型。
- split。
- seed。
- 输出目录。

不能依赖“当前默认路径”隐式切换旧数据和新数据。

训练前检查数据版本与 normalization 的绑定关系。
错误组合必须直接报错，不能自动回退。

## 七、阶段 A：实现逐轴独立、具有精度证书的 L1 求解器

### A1. 保持原数学目标

固定：

P(x) = 0.5 * sum((y - x)^2)
       + 50 * sum(abs(D2 x))

D2 使用原项目的索引二阶差分：
x[t] - 2*x[t+1] + x[t+2]

不得擅自：
- 加入时间间隔缩放。
- 改成一阶差分。
- 改成其他边界条件。
- 改变 lambda。
- 对不同活动设置不同 lambda。
- 先标准化信号却不相应证明目标等价。
- 改变原始单位或幅值。

生产求解使用 float64。
最后仍按原数据格式保存 float32 residual，量化误差单独检查。

### A2. 修复独立性

每条 Acc 轴作为一个独立 series。

允许共享：
- 仅依赖长度、rho 等固定配置的矩阵。
- 稀疏矩阵分解。
- 不含患者数值信息的只读缓存。

禁止共享：
- 全批次停止判定。
- 由其他记录决定的容差。
- 跨记录 warm start。
- 跨记录的自适应 rho 统计。
- 由其他患者决定的最终输出迭代次数。

向量化实现允许存在，但必须：
- 每条轴独立计算收敛证书。
- 独立决定完成状态。
- 已完成轴固定其接受结果。
- 后续其他轴继续迭代不能改变已完成轴的结果。

不能仅把 max_iterations 调大或把旧全批次 tolerance 调小，就宣布问题已修复。

### A3. 每条轴必须具备可核验的数值证书

不能只保存 converged=True。

至少保存：

- primal objective。
- dual objective。
- primal-dual gap。
- dual feasibility。
- stationarity residual。
- iterations。
- final stopping reason。
- solver config hash。
- source hash。

可采用以下与原目标一致的 dual 形式作为核对式：

设 q 的长度为 T-2，且 ||q||_infinity <= 50：

D(q) = y^T D2^T q - 0.5 * ||D2^T q||_2^2

gap = P(x) - D(q)

必须通过单元测试核对符号、维度和公式。
不要将 ADMM 的 scaled dual 直接当作真实 dual 而不做正确换算。

如将 dual 投影到可行域，用投影后的 q 计算证书，并保留原始量的诊断。
不能用不可行 dual 生成虚假的 gap。

继承 Phase 9 的精度要求：

gap <= 1e-8 + 1e-4 * E_aff

先核对旧协议及审计代码对 E_aff 的定义。
如果旧材料没有给出明确系数约定，则在新协议中明确固定：

E_aff = 0.5 * ||y - x_aff||_2^2

其中 x_aff 是该条信号自身的合法仿射参考。
不得使用跨记录统计量。

保留原始 gap，不能通过简单 max(gap, 0) 隐藏错误。
对于浮点舍入范围内的微小负值，提前定义数值容忍规则并单独记录。
超过该规则的负 gap 应视为证书实现错误。

证书在完整原始序列上计算，不能只对裁剪后的片段计算来替代原目标证书。

### A4. 验证参考与求解器选择

优先在现有 ADMM 实现基础上修复独立停止和证书。

允许用环境中已有的独立凸优化工具，在小规模合成例上进行交叉核对。
该工具仅作为数值参考，不作为另一组分类模型候选。

不要升级或替换当前全局依赖。
任何环境变更必须隔离并记录，未经授权不安装新的系统级依赖。

最终生产实现只能锁定一种。
不得根据下游 BA 选择求解器、容差或迭代参数。

在真实完整数据导出前冻结：
- 生产源码。
- solver 配置。
- 容差。
- 初始化。
- 最大迭代与失败策略。
- 依赖版本。
- 源码与配置 hash。

达到最大迭代但未通过证书时必须返回明确失败。
禁止静默回退旧算法、仿射参考或未经验证的其他滤波方法。

## 八、阶段 B：正确性与有界计算成本验收

### B1. 必须覆盖的测试

1. 常数输入。
2. 线性斜坡。
3. 已知低幅 KKT 合成例。
4. 多幅值正弦。
5. 多频率正弦。
6. 多频混合与阶跃合成输入。
7. 1024 点和 2048 点实际长度。
8. 非有限值和非法长度的拒绝。
9. 单条处理与混批处理。
10. 同一批次不同排列。
11. 不同批大小。
12. 同一记录三个 Acc 轴分开与一起处理。
13. 重复运行。
14. 进程重启、缓存命中与未命中。
15. 原始信号 = trend + residual 的数值一致性。
16. float32 保存与重新读取的误差。
17. 原始 Gyro 在最终格式转换和裁剪后保持一致。
18. 未通过证书时不能生成可用于训练的成功标记。

固定独立性门槛：
相同输入在单独、混批、重排等情况下，trend/residual 最大绝对差不超过 1e-7g。

注意：
该门槛检验输出独立性，不等于 primal-dual gap 自动保证逐点误差也为 1e-7g。
这两类检查分别报告。

求解证书针对保存前 float64 解。
对 float32 保存后的 residual 另行报告量化误差及其保守影响，不能声称量化完全无损。
若超出预登记的保存误差范围，停止并排查，不暗中改变 dtype 或原数学目标。

### B2. 真实小规模验收

合成测试通过后，允许按预先登记的确定性规则抽取少量当前授权的原始记录。

选择依据可以是：
- 已知活动。
- 原始长度。
- 固定 hash 顺序。
- 与分类结果无关的幅值覆盖。

不能依据：
- 旧模型错误名单。
- 修复后的预测。
- 哪种记录最容易取得有利结果。

小规模验收必须覆盖实际两种主要长度，记录：
- 每轴证书。
- 耗时。
- 内存。
- batch 不变性。
- 新旧信号差异。
- 输出长度和通道一致性。

不得用小规模真实样本选择分类性能更高的 solver 设置。

### B3. 计算预算

继承原 Level 2 建议：
solver QA 与候选信号导出的累计计算预算上限为 6 CPU-hours，按进程及子进程 CPU 时间累计，而非仅看墙钟时间。

先使用有界 benchmark 估计完整处理成本。

如果预计明显超过预算：
- 完成当前可验证部分。
- 报告实测成本和剩余预计需求。
- 标记 BLOCKED_COMPUTE_BUDGET。
- 不降低精度门槛，不擅自扩大预算。

这属于资源阻断，不属于算法性能负结果。

## 九、阶段 C：生成独立的新处理数据版本

只有阶段 A/B 全部通过才开始。

数据版本建议命名：
pads_l1_lambda50_certified_v1

### C1. 保持处理顺序

本轮固定：

原始完整 Acc
    -> 在完整长度上求解同一 L1 目标
    -> residual = raw Acc - trend
    -> 按原规则裁去前 48 点
    -> float32 保存

Gyro：
原始 Gyro
    -> 按原规则裁去前 48 点
    -> float32 保存

不要同时改成“先裁后滤”。
不要统一删除前 1–5 秒。
不要插值修补时间戳间隙。
不要修改左右腕同步。
不要旋转坐标轴。
不要补加 1g。
不要新增带通或小波去噪。

保持：
- 原 11 项活动。
- 原左右腕顺序。
- 原 Acc XYZ + Gyro XYZ 顺序。
- 原 subject label mapping。
- 原完整变长输入。
- 原 padding、mask 和真实长度规则。

正常情况下原始 1024/2048 点应对应裁剪后的 976/2000 点；逐条核验，不靠假定补造。

### C2. 访问范围

使用现有正式 development split 的授权受试者集合建立读取 allowlist。

先明确各 split 的 train/validation/excluded 成员。
不得读取某个当前 run 的 excluded 预测或性能。

如果全体授权 development 受试者的并集为 390 人：
- 可对这 390 人逐记录生成确定性、与 cohort 数值无关的新信号。
- 同一处理信号可被不同合法 split 复用。
- 不能将其称为 390 名独立测试受试者。
- 不扩展到健康对照或其他未授权记录。

独立求解器与跨患者统计完全分离。
不能为了“批量处理效率”恢复全队列停止准则。

### C3. 失败处理

所有拟用于训练/验证的 Acc 轴必须通过数值证书。

出现失败时：
- 保存失败日志和私有定位信息。
- 不静默删除记录或患者。
- 不混用部分新 L1 和部分旧 L1。
- 不生成“完整数据版本通过”的标记。
- 停止完整模型训练，定位修复后按版本规则重新验收。

不可通过删除失败的 DD 患者来改变队列。

### C4. 数据来源保存

私有 manifest 至少包括：
- 原始文件 hash。
- patient/activity/wrist/channel 映射。
- 输入长度与输出长度。
- solver source/config hash。
- 证书文件 hash。
- 输出 hash。
- 版本号。
- 创建时间。
- 状态。

采用原子写入和可审计断点续跑。
续跑只能复用 hash、配置和证书全部匹配的已完成文件。

## 十、阶段 D：重算 train-only normalization 并连接训练链

### D1. 归一化

在新信号上调用原正式 normalization 算法。

保持：
- 统计维度。
- 活动/腕侧/通道顺序。
- 方差与 epsilon 定义。
- 真实长度处理。
- 不含 padding 的统计规则。

每个唯一 inner split 只用对应 inner-train 受试者重新计算。
15 个 split 分别保存，三个 seed 复用该 split 的统计量。

STR 与 WSSL 若使用相同的本地输入和统计规则，应引用同一份该 split 的新 normalization，避免额外差异。

禁止：
- 全队列 fit。
- validation fit。
- 每人单独归一化。
- 用旧 normalization 作为本轮主实验输入统计。
- 搜索新的归一化方法。

### D2. HarNet 保持不变

HarNet raw Acc 路径绕过 L1，本轮不改它。

保持：
- 原官方权重。
- 原 raw Acc 解析。
- 原裁剪与上下文规则。
- 原 clipping。
- 原 100->30 Hz 重采样。
- 原零延拓。
- 原特征聚合。
- 原冻结策略。

复用经过来源核验的既有 HarNet cache，不重新提取表示。

必须区分：
- HarNet raw Acc 冻结特征：允许复用。
- 由旧 STR 处理信号产生的 supervised embedding：不能用于代替新输入训练。

优先使用可逐受试者读取的合法缓存。

如果只能访问原整队列压缩特征：
先寻找已经存在的合法逐人版本。
若确需一次纯格式拆分，只允许处理已授权的 PD/DD 队列，必须记录实际 I/O 范围，不做任何拟合、统计选型或 outer 评估，并披露不是物理上的逐 split 隔离。

不得通过“先整数组读取再筛选”声称完全没有接触其他成员特征。

### D3. 接口正确性

训练前通过：
- subject/label 映射一致。
- 原 split 不变。
- 每条信号版本一致。
- normalization 版本匹配。
- HarNet cache 来源匹配。
- padding/mask 合成不变性。
- 输出 logits 形状正确且有限。
- 冻结参数正确。
- checkpoint 保存/恢复一致。
- 数据或 normalization 版本不匹配时直接报错。

## 十一、阶段 E：重新训练 Frozen WSSL-STR

只运行一个新候选：

WSSL-CERTIFIED-L1

历史对照：

WSSL-LEGACY-L1

历史 WSSL 参考指标：

Accuracy = 0.745604
BA = 0.719644
AUROC = 0.759553
Macro-F1 = 0.706589
PD Recall = 0.782344
DD Recall = 0.656945

以上只是历史汇总校验值，配对统计必须使用真实历史 split/seed 结果，不得用一个均值复制成45行。

训练要求：

- 同15个subject-level development splits。
- seeds42/43/44。
- 同一Frozen WSSL-STR架构。
- HarNet冻结。
- 原classifier-side参数按原规则训练。
- 原训练初始化规则，而不是从旧最佳监督checkpoint继续fine-tune。
- 原balanced CE。
- 原采样、增强和batch组成规则。
- 原validation checkpoint selection。
- 原DD概率判定和tie规则。

Phase9协议记录的recipe为：
batch=8
AdamW
lr=2e-4
weight_decay=1e-4
cosine
max_epochs=50
gradient_clip=5
patience=12

必须与实际正式配置逐项核验。
AMP、TF32、精度、cuDNN、worker和determinism设置也应记录并保持匹配。

若文字与真实归档执行配置冲突，报告冲突并依据可追溯执行证据锁定，不能静默采用有利版本。

不能同时：
- 换loss。
- 改class weight。
- 加SupCon。
- 加activity interaction。
- 改阈值。
- 改HarNet。
- 搜索learning rate。
- 换checkpoint selection规则。

正式执行45个run。

允许极小的合成输入训练smoke test验证梯度与恢复，但须单独记账，不用于性能选择。
真实训练失败和重试也必须记录，不能只报告成功次数。

完成后先生成WSSL修复对照报告。

不要根据中途某个seed的高低提前保留、拒绝或改变方案。

## 十二、阶段 F：重新训练 STR-01，复核 SSL 增量

本阶段在开始任何新分类训练前已经登记。

WSSL阶段完成且流程无正确性问题后执行，不以WSSL是否涨点作为入口条件。

候选：

STR-CERTIFIED-L1

历史对照：

STR-LEGACY-L1

历史STR参考指标：

Accuracy = 0.730718
BA = 0.697189
AUROC = 0.717631
Macro-F1 = 0.686068
PD Recall = 0.777864
DD Recall = 0.616515

使用正式STR-01的真实架构和训练recipe，保持与历史来源一致。
不是删掉WSSL分支后随意训练一个近似STR模型。

使用：
- 同一新数据版本。
- 同15个split。
- 同seeds42/43/44。
- 对应split的同规则新normalization。
- 原STR训练及checkpoint选择规则。

执行45个正式run。

最终形成四格对照：

                         Legacy L1        Certified L1
STR-01                   历史结果         本轮45run
Frozen WSSL-STR          历史结果         本轮45run

必须回答三个主要问题：

1. 修复L1后，WSSL发生什么变化？
2. 修复L1后，STR发生什么变化？
3. 在同一新数据版本下，WSSL相对于STR是否仍有增量？

可预登记一个探索性差中差：

(WSSL_new - STR_new) - (WSSL_old - STR_old)

但不能将它解释为唯一隔离出的泄漏效应或临床因果效应。

## 十三、统计与结果解释

### 统计对象

延续正式口径：
- 每个split内先平均三个seed的单模型指标。
- 再对15个split等权汇总。
- 不把seed概率ensemble混入单模型主表。

报告：
Accuracy、BA、AUROC、Macro-F1、PD Recall、DD Recall、
NLL、Brier、ECE。

沿用历史指标定义，特别核对：
- 正类为DD。
- Brier采用何种二分类/双列定义。
- ECE分箱规则。
- NLL数值处理。
- tie判定。

不能为使结果更好而改变统计定义。

### 配对与不确定性

报告：
- mean/median差值。
- split wins/ties/losses。
- 10000次paired split bootstrap。
- Wilcoxon p与效应量。
- seed稳定性。
- 10000次subject-cluster bootstrap。

所有重复appearance随受试者一起重采样。
明确cluster分析对应固定已训练模型，不代表外部泛化。

### 多重比较

在训练前登记并冻结：

C1 主比较：
WSSL_new vs WSSL_old
主要family：BA/AUROC，共2项。
次要family：其余7指标，共7项。

C2 关键复核：
STR_new vs STR_old；
WSSL_new vs STR_new。
两项比较的BA/AUROC构成4项family；
其余指标构成14项family。

探索性差中差单独列出预登记family，不混入主检验。

缺失项本身报告NA，进行保守完整family校正时按p=1占位，不能删除不利或无法检验项缩小family。

不因为某一种bootstrap区间更有利而更换主口径。

### 性能保留标准

对于“是否宣布优于历史WSSL”，沿用预先登记的开发标准：

- meanΔBA>0。
- 至少10/15split的BA提高。
- 配对95%CI下界>0。
- AUROC和Macro-F1均不下降。
- PD/DD Recall各自平均下降不超过0.01。
- BA/AUROC的within-split seed SD不超过baseline的1.25倍。
- correctness/provenance通过。

报告p/q，但不要在看完结果后增加或删除门槛。

这些只是development模型提升标准，不是临床有效性证明。

### 正确性与性能必须分开判定

最终必须分别给出：

NUMERICAL_CORRECTNESS：
PASS / FAIL / BLOCKED

RECORD_INDEPENDENCE：
PASS / FAIL / BLOCKED

DATA_AND_TRAINING_PROVENANCE：
PASS / FAIL / BLOCKED

PERFORMANCE_PROMOTION：
PASS / NOT_ESTABLISHED / DEGRADATION / BLOCKED

数值修复通过但性能未提高，不得写成“L1修复失败”。
可以写“数值与独立性修复成功，未建立性能提升”。

旧结果保留为legacy参照。
不能因为旧指标更高，就恢复旧批次耦合作为已验证独立的处理方式。

也不能仅凭新旧分差断言旧指标虚高了多少。

## 十四、设备分层与信号变化分析

仅作为预登记的辅助分析，不用于选型。

设备metadata中Series3与Series4的类别分布不均衡，历史报告已披露。

本轮报告：
- 各设备标记内PD/DD的样本数。
- 修复前后各组Recall及不确定性。
- 新旧信号差异在各设备标记中的描述性分布。
- 无法估计的split内指标标NA。

同一患者重复appearance不能增加患者样本量。

小组样本稀少时：
- 不做强因果解释。
- 不发布可能识别个体的细分表。
- 不把组间差异解释成硬件必然造成的偏差。

禁止：
- 因结果不好删除某个设备组。
- 仅报告Series4的有利结果。
- 把设备型号加入模型输入。
- 使用设备分层结果修改lambda或阈值。

信号变化分析可以包含：
残差RMS、trend norm、新旧差异、边界与内部差异、预固定频域描述。

这些都是处理差异，不等于疾病信息增益。
不得据此搜索新频带、裁剪或滤波方案。

## 十五、参数量、推理时间与预处理成本

统一报告：

1. 总参数。
2. 可训练参数。
3. 冻结参数。
4. 每条轴L1求解时间。
5. 每条腕记录处理时间。
6. 每受试者预处理时间。
7. 纯模型forward时间。
8. 端到端推理时间及明确的计时边界。
9. 吞吐量和峰值内存。

区分：
- 使用HarNet缓存的训练/推理场景。
- 从原始数据开始的在线处理场景。

本轮没有授权重新提取全部HarNet特征。
如果端到端实测需要超出本轮范围，则明确报告未测，不能把缓存forward时间写成端到端延迟。

统一硬件、精度、batch size、warm-up与重复次数。
GPU计时进行正确同步。
CPU预处理与GPUforward分开记录。

架构不变应核对参数量一致，而不是推测。
不能将修复后的CPU开销隐藏在GPU模型时间之外。

## 十六、工程执行与停止条件

使用可恢复的阶段化入口，提供：
- audit。
- solver-test。
- pilot。
- build-data。
- fit-normalization。
- train-wssl。
- train-str。
- analyze。
- integrity-check。
- publish-dry-run。

参数名与脚本实现保持一致，并在README给出真实可执行命令。
不要只写伪代码。

训练入口必须检查上游阶段的有效完成标记和hash。

以下情况停止对应后续阶段：

- 数据/模型/配置来源无法证明。
- 任何待用轴未通过证书。
- 批次不变性未通过。
- 保护文件发生未授权变化。
- 数据版本混用。
- normalization包含validation成员。
- HarNet缓存不对应原记录。
- 输入shape或mask不一致。
- CPU预算不足。
- 训练资源或授权不足。

以下情况不能作为提前停止STR关键复核的理由：

- WSSL某个seed下降。
- WSSL平均BA没有提升。
- 早期几个split结果不理想。

除原训练早停规则外，不按结果进行实验级提前淘汰。

若发现新的真实代码错误，应保存失败版本、日志和修订记录；不能静默删除失败run后仅保留新结果。

## 十七、需要交付的产物

至少生成：

PROTOCOL.md
PROTOCOL_AMENDMENT.md
README.md

reports/
    FINAL_REPORT.md
    SOLVER_IMPLEMENTATION_REPORT.md
    SOLVER_CORRECTNESS_REPORT.md
    DATA_REBUILD_REPORT.md
    NORMALIZATION_PROVENANCE_REPORT.md
    WSSL_RETRAIN_REPORT.md
    STR_RETRAIN_REPORT.md
    SSL_INCREMENT_REVALIDATION_REPORT.md
    DEVICE_STRATIFIED_ANALYSIS.md
    COST_AND_LATENCY_REPORT.md
    CORRECTNESS_AND_PROVENANCE.md
    LIMITATIONS_AND_NEXT_STEPS.md

analysis/
    numerical_certificate_summary.csv
    batch_independence_summary.csv
    data_version_integrity.json
    normalization_integrity.json
    experiment_state.csv
    metrics_by_split_seed.csv
    paired_statistics.csv
    seed_stability.csv
    subject_cluster_summary.csv
    device_stratified_summary.csv
    cost_summary.csv
    final_integrity.json

公开CSV只能包含安全的聚合信息。
完整逐轴证书、ID映射、逐人预测、信号和checkpoint存入private。

最终分类结果主表必须有：

- STR-LEGACY-L1。
- WSSL-LEGACY-L1。
- STR-CERTIFIED-L1。
- WSSL-CERTIFIED-L1。

如果某项未运行，明确写NOT_RUN或BLOCKED，不填估计值。

最终回答：

1. 旧数值反例是否复现？
2. 新solver是否达到原目标精度？
3. 是否消除了跨记录停止依赖？
4. 所有新数据是否通过证书和格式检查？
5. 新旧normalization如何变化？
6. WSSL的九项指标如何变化？
7. STR的九项指标如何变化？
8. 修复后WSSL相对STR的增量是否仍成立？
9. 性能变化是否呈现设备条件差异？
10. 当前有哪些问题已解决，哪些仍UNKNOWN？
11. 正确性结论与性能结论分别是什么？

## 十八、发布、保护与任务结束

沿用项目既有保护清单，并为新增资产建立独立清单。

最终重新核验：
- 原始数据不变。
- legacy处理信号不变。
- mainline和历史配置不变。
- 历史checkpoint不变。
- 原HarNet cache不变。
- 原划分不变。
- 新数据与新normalization来源闭合。

按项目既有Git发布规范同步：
- 新源码。
- 配置。
- tests。
- 协议。
- 聚合统计。
- 正面与负面结果。
- 审计和限制说明。

不得上传：
- 原始或新处理信号。
- 私有ID及映射。
- 逐人预测。
- 个体信号图。
- checkpoint。
- HarNet缓存。
- 完整逐轴证书。
- 私有保护清单。

不要自动修改历史冻结的“当前模型”指针。
在新报告中明确区分：
- legacy历史模型。
- 数值修复后的研究基线。
- 是否通过性能提升门槛。

发布后核验远端与服务器公开快照一致。
没有实际发布或回读成功，不能声称完成。

完成后停止。

不自动启动：
- Activity Interaction复训。
- SupCon复训。
- Adapter或LoRA。
- 双参考融合。
- 阈值搜索。
- lambda搜索。
- HarNet边界line对照。
- 坐标变换、重力重建或时间戳修复。
- 新的outer evaluation。

本轮最终目标：
在保留历史记录的前提下，建立数值正确、逐记录独立的新L1处理版本，并用WSSL与STR两个关键模型重新检验性能及外部SSL增量。