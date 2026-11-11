# PADS：Frozen WSSL-STR 主线审计与双参考典型性诊断

你是一名负责可复现实验的研究工程师。请直接在现有授权环境中执行本任务，而不是仅返回建议。主任务是改进现有 Frozen WSSL-STR；异常检测/参考分布学习是独立探索，不得自动替代主模型。
在服务器上面进行修改和执行，完成之后及时提交GitHub

本次授权：

1. 只读审计、历史结果复核、正确性测试、冻结模型推理、预注册的轻量诊断。
2. 完成下文 A0–A2 和 B0–B2 的适用部分。
3. 只有 A0 的预先登记闸门通过，才允许一个“固定分母 balanced CE”WSSL 训练候选；最多 45 个正式候选 run。
4. 最多一个、满足完整 OOF 来源条件的两分数轻量融合诊断。
5. 不授权新 HarNet Adapter/LoRA/全量微调、Patch 网络、预训练、广泛结构搜索或 outer evaluation。可以给这些方向写后续协议，但本次不训练。

保护要求：

- 不修改冻结模型、历史源码快照、split、归一化、checkpoint、预测和负结果；新代码和输出放独立目录。
- 先记录受保护资产 hash，结束时再次核验；不能通过覆盖旧报告“修正历史”。
- 不读取 outer-test 信号、标签、预测、指标用于任何选择；必要的 partition ID/provenance 元数据只用于边界断言，不加载对应数据。
- 不调用会自动启动 outer-final/test 的通用完整 nested-CV 入口。使用经过核验的 inner-only runner。
- 私有原始信号、临床个体表、cache、checkpoint、逐受试者预测只留服务器；公开输出仅代码、配置、报告、小型聚合结果。不得记录密码或令牌。
- 不自行安装或升级依赖来改变正式运行环境；资产/依赖缺失时记录 BLOCKED，并继续可完成的无风险部分。
- 不自动 commit/push；如已有明确发布授权，遵循仓库既有发布流程，不 force push。

## 1. 历史状态审计：先查是否做过，再决定是否执行

阅读所有相关 FINAL_REPORT、REPORT、PROTOCOL、decision.json、manifest、correctness 与完整性报告，不能只看 README 开头或单个数值表。

至少覆盖：

- V8/V8-GN、STR-01、WSR-01、prototype/invariant representation、strong-backbone benchmark；
- DSG-01、RGD-01、PRR-01、PAG-01；
- phase2_str_training_20260926；
- phase3a_puredeep_20260928；
- phase3b_external_ssl_20260928；
- phase3c_ssl_adaptation_20260930；
- phase3d_ssl_adaptation_20261001；
- phase3e_external_priors_20261001；
- phase4_window_dynamics_20261002；
- wssl_ema_20261002；
- wssl_review_20261004；
- frequency_prior_str_20261007；
- wssl_contribution_20261007；
- wssl_activity_scaling_20261008；
- patch_str_20261008；
- patch_str_recheck_20261111。

形成 EXPERIMENT_LEDGER.csv/md，字段包括：\
实验ID、目标基线、唯一改动、预期/实际run数量、split/seed覆盖、正式结论日期、retained/rejected/inconclusive/not-run/blocked、主要指标、拒绝原因、协议与来源文件、来源hash、是否复用了既有baseline。

特别核对：

- Frozen WSSL-STR 是当前 retained single-model。
- HarNet layer5 部分解冻、middle+final、输出特征 Adapter、gated fusion、train-only masked domain adaptation 已经完成但未保留。
- 真正深度模型层面的 OOF 阈值诊断已完成，不能当作新实验重跑。
- Phase3D 的条件 activity gate 当时未执行，但后续独立的 11-scalar activity scaling 已执行；不能笼统说“activity scaling 从未试过”。
- Activity scaling 是 INCONCLUSIVE / DO NOT RETAIN，不改写成“已证明有害”。
- Patch P1/P2 已在 2026-10-08 完成；2026-11-11 是复核，不是新训练。
- Phase C 独立 loss numerical/gradient audit 在当时的条件树中没有启动；本次先确认后续 main 是否补做过。
- Full HarNet fine-tuning 没做过不等于应该立即做。
- “prototype alignment loss”与“冻结表征的类别参考距离分类器”不是同一实验。
- 拒绝某固定配置，不等于否定整个方法家族；但也不能仅换名字、宽度、rank 或超参数重跑失败实验。

## 2. 锁定并复现现有基线，不重复训练

预期 retained baseline（以现场正式资产为准）：\
Accuracy 0.745604\
Balanced Accuracy 0.719644\
AUROC 0.759553\
Macro-F1 0.706589\
PD Recall 0.782344\
DD Recall 0.656945

PD=0，DD=1；固定原 argmax/.5 规则，包括 tie 行为。\
15 个固定 development splits，seeds 42/43/44。\
主汇总为“同 split 内平均三个 seed 的指标，再汇总 15 splits”，不是概率 ensemble。

预期模型：

- Local Acc+Gyro 六通道 temporal + learned-moment wrist encoder，64D。
- Raw Acc 经既有 trim/clip/window/resampling 后进入 frozen HarNet10，1024D。
- 原 LayerNorm + Linear(1024,64) 投影输出加到 local wrist token。
- 原 bilateral 258D、activity/subject path、11×16 ordered STR path 不变，最终 logits 相加。
- 分类器侧 143,172 trainable，HarNet 10,457,408 frozen。
- 不把模型中未启用的可选分支当成实际模型。

实际核验：

1. 45 个正式 baseline checkpoint、配置、预测、split、train-only normalization、cache ID/活动/腕/通道顺序及 hash。
2. 从实际 import/runtime 路径核对模型、loss、metrics、runner；不能使用过期 foundation 副本。
3. 尽量复用已完成的精确复现，不再次训练 45 个 baseline。
4. 如需重算推理，在独立输出位置比较完整验证集概率、决策与指标；明确归档是否有 logits，不能伪称比较不存在的字段。
5. train-only normalization 适用到所有数据拟合变换：Scaler、PCA、参考库、原型、校准器、融合器。
6. 核验 135 个历史 deep-OOF 模型/预测是否实际保存及可用，不从报告中的“训练过”推断资产必然存在。

正确性测试：

- 真实 shape、真长度切片、padding/无效腕/无效活动隔离；
- subject ID 重排一致、错误 ID/重复ID/标签错位显式失败；
- checkpoint reload、独立实例参数/cache/RNG隔离；
- zero-residual 必须屏蔽 Linear(LayerNorm(SSL)) 的完整输出，包括 bias；不得用 zero SSL input 替代；
- 禁止直接 deepcopy 旧闭包式 monkeypatch wrapper；使用已验证的独立显式接口；
- 冻结 encoder 参数、BN 状态和缓存不更新；
- 新 fit 对象的训练ID审计、outer access sentinel；
- 数值有限性、必要的梯度与掩码测试。\
  所有错误都先定位，不能改 tolerance 以掩盖身份或来源问题。

## 3. 方向 A：现有模型的低成本诊断

### A0：补做 balanced CE 数值与梯度审计

先查本实验是否已有完整结论。若已有，复核并复用，不重新训练。

追踪实际 class weights、采样器、batch size、drop_last、最后不足 batch、loss reduction、梯度裁剪和 epoch loss 汇总实现。对照当前安装版本的 PyTorch 文档，不为匹配在线最新版而升级环境。

对于 hard class-index targets、无 smoothing 的实际路径，验证：\
L_batch = sum_i w[y_i] \* CE_i / sum_i w[y_i]\
与固定目标：\
L_fixed = sum_i w[y_i] \* CE_i / (B \* mean_train(w[y]))\
其中 mean_train(w[y]) 仅由该 inner-train 全部受试者求得，B 是实际 batch 大小。

必须指出：

- 默认 weighted mean 不是 PyTorch bug；
- 同一 batch 内两种 reduction 的梯度只相差一个公共标量，不改变该 batch 内的相对类别权重；
- 有限随机 batch 下，期望 batch-ratio 目标与固定分母的全局平衡经验风险不必相同；
- 这不自动证明它导致 DD 错误，也不保证更改后提高指标。

测试范围：

- 手算小 batch、单类 batch、batch=1、balanced batch、最后小 batch；
- 未加权/加权情形、实际 hard-label 路径；soft-label 情形单独解释，不能混用公式；
- autograd 数值一致性；
- 固定模型/固定输入下的 logit-gradient、参数梯度、裁剪前后范数；
- 真正 sampler 下类别出现比例、归一化系数分布、类别总损失系数；
- epoch 加权 loss 汇总是否准确反映声明的风险；
- synthetic 或 train-only 数据诊断；不能用 validation 选择“更好”的 reduction。

在任何候选性能出现之前，冻结 A0_PROTOCOL 与入口判据。建议入口要求同时满足：\
a. 运行路径确实采用随机 batch 的 self-normalized weighted mean；\
b. 项目声明的目标确实是 train-distribution 上的全局 class-balanced risk；\
c. 使用实际 inner-train batch 组成，能重复验证目标与更新缩放的差异，并完整报告效应大小，而非仅猜测；\
d. 历史未进行同一 denominator 对照；\
e. correctness 与资产完整性全部通过。\
满足这些条件仅允许一次方法对照，不表示已证明有害或有效。

若条件满足，允许本次唯一 deep training 候选：\
原 WSSL + 固定分母 balanced CE，其他全部保持：\
原 split/seed、相同随机初始化、原自然采样、原类别权重、HarNet frozen/cache、全部输入与 normalization、原 optimizer/LR/WD/batch/scheduler、原 BA 选点和停止规则、原 .5 决策。\
训练从原同 seed scratch 初始化，不从已训练最佳 checkpoint warm-start。\
不同时改权重、sampler、batch、LR、smoothing、focal、SAM、EMA、augmentation 或模型结构。\
运行 15×3=45；完整后一次性分析。\
如果入口不满足，本次 deep training 次数为 0，不自动切换到第二候选。

### A1：当前错误与模态信息诊断

复用已经完成的 WSSL contribution、45 个 last-block adapted checkpoint/预测及当前 frozen baseline，不重复 37-condition 贡献矩阵。

错误组必须统一：\
主要定义：每个 validation appearance 先均值三个 seed 的概率，再按每人四次 appearance 的错误比例分类；>=.75 为 stable-error，<=.25 为 stable-correct。\
归档对应当前 WSSL stable-error 68（PD35/DD33）。\
“全部四次都错”的 38（PD14/DD24），以及全部12次都错的12，属于不同敏感性定义，不能混成主口径。\
现场复核不一致时先找定义或来源差异，不强行匹配数字。

分析：

- 两类及已有 DD source categories 的错误分布、分数、margin、seed/split稳定性；
- adapted vs frozen 的 corrected/newly-wrong/net，按 PD/DD 分开；
- 当前 local、SSL residual、bilateral/activity、subject/structured 表征的来源与可用性；
- 只在训练集拟合一个预先固定的低容量诊断，检验 local 六通道表示在 Acc-only SSL 和当前分数之外是否含有验证集增量信息。\
  诊断需先锁定一种表示、降维/正则和比较，不扫描各层/各活动找最好结果。

严格解释：

- local 是混合 Acc+Gyro，不是 Gyro-only；local 的增量不能自动归因 Gyro。
- 输入置零/置换主要是依赖性或分布外扰动诊断，不等同重新训练的单模态模型，也不是疾病因果证据。
- 单 probe 的交叉拟合不能消除底层 encoder 曾用这些 subjects 训练的事实。
- 如果缺少真正模态隔离的表征，明确“尚不能确认 Gyro 独有增量”，不要为了补表格立即训练新 encoder。
- 验证集定义的错误组、活动贡献和亚型结果不能用于采样、删人、重加权或逐患者路由。
- 现有输出 Adapter 和 layer5 适配负结果不等同所有 PEFT 均无效，但本次不做新的 rank/layer/width 搜索。\
  如发现有稳定、具体的参数漂移/互补性证据，可以只写一个后续受约束内部适配协议，不执行。

### A2：概率与分数迁移审计，而非重新搜索阈值

计算原概率的 unweighted NLL、Brier、固定10等宽bins ECE、可靠性曲线、校准截距/斜率，并给出分级、分seed/split结果与不确定性。斜率/截距作为诊断拟合与作为可部署校准器要分开。加权训练CE不能冒充proper NLL；小样本ECE不能作为唯一指标。

复用历史 true deep-OOF 资产，检查：

- 每条 query 的 encoder 训练、normalizer、checkpoint selection、PCA/bank/calibrator 全部来源；
- OOF 小训练池/固定 epoch10 与 full-train/best-epoch 模型分数尺度的差异；
- 不混合使用曾见过当前目标验证受试者的其他 context 模型；
- 不把 probe-only OOF 当成 end-to-end OOF。

原 .5 决策保留。不得重复 Phase3C OOF 阈值网格。\
如 OOF 来源充分，可做一个固定方法的正温度校准诊断，T仅在当前 target inner-train 的合格OOF分数上拟合；报告对full-model验证分数迁移的局限。不得声称单个二分类模型的正温度校准能提高其 AUROC 或改变 .5 分类。\
校准器不满足来源条件时标记 BLOCKED；不能使用 validation 拟合T。\
校准改善与判别改善分开保留，不把概率改良冒充新最佳分类模型。

## 4. 方向 B：独立冻结参考分布实验

### B0：先区分任务定义

建立两类实验，命名不能混淆：

I. 严格单类学习：

- PD-only：任何 PADS 数据拟合的变换/PCA/bank/阈值只用 inner-train PD。
- DD-only：同理只用 inner-train DD。
- 外部独立冻结 HarNet 可复用。
- 阈值由该参考类的 subject-crossfit 距离95%分位数固定，不用另一类调优。
- 更大PD-reference偏离对应DD；更大DD-reference偏离对应PD。
- 若encoder用过PADS两类标签，就不能称端到端严格one-class。

II. 共同表征中的单参考与双参考比较：

- 在完全相同的训练集拟合距离空间中计算 PD-reference、DD-reference、relative typicality。
- 这是监督的 class-conditional/reference-based 分类，不是“完全无监督发现PD”。
- PD与DD均为疾病人群，不能把任一类等同生理正常。

### B1：预先固定的最小可行实现

首先使用现有外部 frozen HarNet10 每 activity/wrist 的1024D cache，不训练 encoder。\
第一阶段使用已保存的真实聚合特征；聚合cache不能反推出不存在的2秒patch。\
只有实际保存了窗口级特征时，才能按真实 window metadata 做一个后续局部诊断；不能把976点记录虚构成两个完整10秒窗口。

在查看新验证结果前固定：

- 每个 activity/wrist 独立参考空间；
- train-only StandardScaler + PCA，维数固定 min(16, 合法rank上限)，再L2归一化；
- PCA不whiten，不做维度搜索；记录零范数/退化情形的确定处理；
- cosine距离，k=5个不同参考受试者；不足5时预先规定NA/blocked，不能事后换k；
- 每个受试者对同一context贡献等权，不能让多窗口者更占优势；
- common-space主要分析中，两类参考库人数一致，以固定seed/hash抽样，不能按结果选库；完整训练库作为一个预定敏感性比较；
- strict one-class路径不使用另一类的样本数来选择预处理或参考库；
- 顺序固定、双腕先等权、活动再等权，无活动/腕筛选或可训练gate。

DD为正类：\
s_PD = d_PD\
s_DD = -d_DD\
s_relative = d_PD - d_DD\
相对分数越大越倾向DD。\
共同空间的相对分类边界先固定为0；距离不是概率。\
同时保存两个距离；min(d_PD,d_DD)可作为“离两类参考均远”的描述性指标，不自动解释为临床异常。

增加一个同空间的固定nearest-centroid prototype对照，用于判断kNN的局部多模态结构是否必要；不训练prototype alignment loss。\
不要加入 OCSVM/IsolationForest/Autoencoder/距离/聚类超参数网格。

仅在所有bank、scaler、PCA、query的ID与encoder hash审计通过后执行。\
参考库与query必须来自同一个encoder坐标系；不能将不同fold模型的embedding直接拼成一个bank。\
所有训练query在bank中必须排除自身全部窗口/活动副本；严格OOF还需整个拟合链排除该subject。

B报告同时给出：

- 两个严格one-class模型；
- 共同空间中的三种reference score；
- prototype对照；
- AUROC、固定决策点的全部分类指标、类别净纠错；
- 银行人数控制敏感性与rank/零范数等失败情况。\
  对于未做合法概率映射的距离分数，NLL/Brier/ECE标NA，不把任意sigmoid包装成校准概率。

### B2：是否补救 WSSL，而不是只画聚类图

在全部验证受试者上，报告 WSSL 与 reference score 的相关/分歧、PD和DD各自 corrected/newly-wrong/net；同时报告全部人的配对指标。\
不能仅在WSSL错误组上报告“救回率”，忽略对原正确样本的伤害。\
不将pair/patch/window个数作为独立样本量。

只有共同空间的 relative score 相对两种单参考在预定配对评估中有稳定优势，并且表现出WSSL之外的增量，才允许考虑一次融合。\
“稳定优势”的诊断闸门在结果前固定：对两种单参考的 mean delta AUROC 均>0、各至少10/15 splits正、主要配对CI下界均>0；保留完整多重比较结果与重叠split限制。

融合还必须满足：

- 有可审计的整个模型链out-of-fold训练分数，而非encoder训练内分数；
- 对每个目标inner split，融合fit的全部来源模型不得见过该split的validation subjects；
- reference距离的OOF算法与deployment算法一致，包括库人数/距离尺度的差异审计；
- 至多一个固定正则的两分数logistic融合（原WSSL logit + relative score），不搜索权重、模型或阈值；
- baseline也在相同subject/split/seed评价。\
  若无合格OOF分数，停止在互补性诊断，不用训练内分数拟合后声称稳健融合。

融合若有效，称为“WSSL + reference scoring pipeline”，不能把它当作原143,172参数单模型的原生结果。\
本次不训练专门patch/局部表征网络，不自动进入PaAno triplet/temporal adjacency，也不复活已被拒绝的Patch P3。\
后续局部表征训练须另有明确证据：现有评分有可重复互补、局部而非活动身份/采样数量造成增量、且收益不能由当前冻结表示与简单readout解释。此处只写协议，不执行。

## 5. 配对统计与决策

所有候选使用相同subject splits；45单元完整后分析，不按最好seed/split选模型。\
准确区分：

- 三个seed指标先平均；
- 三seed概率ensemble；
- 四次appearance的患者错误分组。\
  三个口径不能互换。

至少报告：\
Accuracy、BA、AUROC、Macro-F1、PD Recall、DD Recall；\
mean/median delta、wins/ties/losses、split SD、各seed均值、within-split seed SD；\
原始p、预登记family的BH q、effect size、paired bootstrap CI；\
校准指标及NA原因；参数量、训练/推理资源和新增训练次数。

按历史规则复现seed-first 15split paired bootstrap（固定seed、10,000次）、Wilcoxon、BH，不改历史结果。\
必须注明15splits共享受试者和训练池，不能作为15个独立外部样本，也不能把45runs当45独立样本。\
新增subject-cluster配对bootstrap时，每个subject的全部appearance一起重采样，按类别分层；它是固定已拟合模型下的条件性不确定性，不覆盖重训练与多轮开发选择偏差。\
报告按5个outer context的分组敏感性，但不访问对应outer test。\
不得把某CI跨0、另一个不跨0当成两者之间显著不同。

本次唯一CE候选的工程保留门槛在训练前冻结：

- mean ΔBA >0；
- 至少10/15 splits ΔBA>0；
- 预定paired ΔBA CI下界>0；
- mean AUROC和Macro-F1不下降；
- PD和DD mean recall下降分别不超过0.01；
- BA/AUROC mean within-split seed SD均不超过baseline的1.25倍；
- 全部correctness/provenance通过。\
  这些是重复development中的保留规则，不等同独立验证。p/q照实报告，不做事后解释性替换。\
  任何额外实际效应下限或统计门槛只能在结果前登记，不得观察结果后改变。

候选未满足联合门槛，不更新retained baseline：\
明确REJECT或INCONCLUSIVE及理由；不得靠单一显著指标、机制probe、重建loss、t-SNE/CKA或某类recall上涨弥补不通过。\
校准sidecar以预登记NLL/Brier改良为主，不冒充BA/AUC提升。\
B融合使用同样的判别和recall/seed护栏，但作为独立pipeline评价。\
若所有新诊断为负，保留原WSSL，停止本轮，不自动扩展搜索。

## 6. 论文核查与方法学贡献

只采用可核查原始论文、正式论文集、官方代码/文档。对论文记录任务、训练/测试受试者隔离、模态、是否本项目已测试、借鉴点、失配点。\
近期优先核查：

- PaAno, ICLR 2026:\
  [https://proceedings.iclr.cc/paper_files/paper/2026/hash/5991785875cf5eee9c29e2ed7d26d1c0-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5991785875cf5eee9c29e2ed7d26d1c0-Abstract-Conference.html)
- XTransfer, ICML 2026:\
  [https://proceedings.mlr.press/v306/zhang26iy.html](https://proceedings.mlr.press/v306/zhang26iy.html)
- Bio-Inspired Self-Supervised Learning for Wrist-worn Accelerometer Data, ICML 2026:\
  [https://proceedings.mlr.press/v306/tarale26a.html](https://proceedings.mlr.press/v306/tarale26a.html)
- Cycle-Aware Masked Self-Supervised Learning for Parkinson's Disease Diagnosis Using Wearable Time-Series Data in Daily Activities:\
  DOI 10.1109/TBME.2026.3712830；核对正式全文和发表状态。\
  经典基线而非新颖性来源：
- Out-of-Distribution Detection with Deep Nearest Neighbors, ICML 2022.
- On Calibration of Modern Neural Networks, ICML 2017.

注意：

- BioPM在本项目已做过受限acc neural-encoder导出的测试，不作为未测试的新方案，也不将该负结果泛化到完整BioPM。
- PaAno正常时间点异常检测与PD/DD受试者分类不等价；其局部归一化、正负样本定义、邻接假设不能直接照搬。
- Cycle-aware mask与现有随机/连续mask不同，但需先证明本数据相关动作的周期结构和失败机制，不自动新训练。
- ICML级别、JCR分区和年份须核对；无法核实写待核验，不把预印本冒充SCI Q1/Q2或CCF正式论文。
- 不用其他数据定义/标签预算/subject聚合不一致论文的高accuracy给本项目设虚假的可比目标。
- 一个CE reduction对照或普通kNN本身不自动构成论文创新；贡献需由稳定增量、任务特定机制、完整对照和独立验证支持。

## 7. 输出与完成标准

在 artifacts/wssl_reference_audit\_<实际日期>/ 下输出，不覆盖既有目录：\
AUDIT_REPORT.md\
EXPERIMENT_LEDGER.csv/md\
NEGATIVE_RESULTS_DO_NOT_REPEAT.md\
CANDIDATE_PRIORITY.md\
PROTOCOL.md\
BASELINE_REPRODUCTION.md\
LOSS_NUMERICAL_AUDIT.md\
MODALITY_ERROR_DIAGNOSTICS.md\
CALIBRATION_DIAGNOSTICS.md\
TYPICALITY_REPORT.md\
FINAL_REPORT.md\
decision.json\
configs/、scripts/、tests/、analysis/ 与分级私有输出。

保存命令、运行环境、Git SHA、源码/配置hash、checkpoint provenance、fit-ID manifests、测试日志和全部负结果；完成后核对正式资产不变。

最后中文汇报：

1. 实际完成/复用/跳过/blocked的工作及原因；
2. 发现的事实、待证假设、未验证信息；
3. 两张表：实验状态和候选优先级；
4. baseline复现误差、全部指标与配对统计；
5. 本轮新增deep training实际次数；
6. A是否有符合门槛的改进，B是否有独立互补，不自动替代主线；
7. 当前retained配置是否变化；
8. 复现命令、输出位置、具体下一停止点。

不能承诺后台继续。执行环境不足时交付已经完成的审计、可运行代码、明确BLOCKED清单，不伪造结果，不重复训练凑数量。