# Phase A：冻结 WSSL contribution audit（运行前固定）

日期：2026-10-07。只执行 Phase A；任何结果都不自动启动 Phase B/C。不训练、不反向传播、不修改正式模型、checkpoint、recipe、阈值、HarNet 或 cache；不访问 outer 信号、prediction 或 performance。

## 资产和复现闸门

唯一模型为 Phase3B 正式 Frozen WSSL-STR（143,172 classifier-side parameters，10,457,408 frozen HarNet parameters），45 个 development best checkpoints，15 fixed inner-development splits，seeds42/43/44，原11 activities、left/right顺序、train-only normalization，DD probability >.5，tie=PD。

先完成全部45完整 validation inference：ID/label/split/cache/activity/wrist/config/hash匹配、15次原inner-train normalization refit对45checkpoint逐位一致、正式预测的两个概率列与decision/六指标一致。正式 CSV 没有 logits：不声称与不存在的归档logit逐位比较；通过原历史 forward、新独立接口及复用已编码表示的诊断 forward 的实际 logits逐位比较。归档概率仅允许 CSV 浮点序列化误差≤1e-12，指标≤1e-12，三前向之间要求torch.equal。任一失败先停止并解决，不能继续 masked inference。

完整复现闸门通过以后，独立诊断脚本/统计/判定规则哈希与Git冻结；然后执行 masked inference。所有文件置于新独立目录，不覆盖原资产。

## 反事实条件

完整模型及35个用户指定反事实：11 activity-off（双腕）、left-off、right-off、22 activity×wrist-off。另固定一个 all-WSSL-off 诊断锚点，帮助说明训练后共同适配；它不是独立训练STR，不是新候选。

在 `Linear(LayerNorm(SSL))` 的输出后将指定 residual严格置零，包括projection bias。STR wrist原表示、wrist/activity有效mask、length、bilateral/activity/structured路径和全部权重不变。不把SSL输入置零当作residual-off。编码得到的原wrist与projected SSL residual复用于多个反事实；first real validation batch逐条件与投影输出hook的直接前向精确比较，并检查完整状态/缓存不变。所有计算eval+no_grad；不保存新checkpoint。

## 统计与保存

Contribution=metric_full−metric_masked，正表示去掉该路径后性能下降。保存每个context/inner/seed/condition的完整与masked六指标、delta；先均值三个seed，再按15split等权汇总mean、median、正/零/负split数、splitSD、10,000 paired split bootstrap95%CI（RNG20261007）。保留seed profile、split/seed variability和rank agreement。不把activity、subject重复出现、45runs、pairs、bootstrap抽样次数当独立统计样本；CI仅描述重复development。

不对35mask条件做显著性筛选，不用某一activity的p值挑结构。主要判定BA profile，AUROC与两类Recall辅助检查。预先固定如下**诊断性**分流阈值，非已知最优或临床界值：

- CASE A / PROCEED（仅提出Phase B资格）：11个activity平均BA contribution的max−min≥.005；同一spread的split bootstrap下界>.0025；三个seed activity profile的pairwise Spearman均>0且median≥.5；每个split与其余14split平均profile的Spearman中位数≥.3，至少10/15为正；至少两个activity在≥10/15split中正贡献。满足全部条件才称足够稳定的activity heterogeneity。max/min只用于整体profile描述，不选择activity；所有11scalar都应统一初始化1，若用户以后批准Phase B。
- CASE B / STOP scaling：spread bootstrap上界<.005，说明在本次预定实用精度内无大的BA activity差异；不是用不显著替代equivalence，也不证明11activity贡献严格相同。不自动做loss study。
- 其余 CASE C / INCONCLUSIVE：分布可不均匀，但稳定性不足或精度不够；停止用该诊断设计activity/wrist/MLP/attention gate。不能调整诊断阈值挽救结论。

Wrist差异直接比较left contribution−right contribution的15split delta/CI与三个seed方向；不据此做wrist选择。22interaction条件描述同样输出，不作新的候选或独立显著性结论。

## Subject与纯STR联系

逐validation appearance保存私有subject ID、label、split/seed、p_full/p_masked、prediction、raw DD probability delta与true-label-oriented delta。仅描述label、boundary effects、corrected/harmed与分布，不从错误列表重采样或调参。

**主分组与DSG/RGD/PRR/PAG正式定义一致**：先均值三个seed的每次validation概率，每人四次development validation appearance；error rate≥.75 stable-error、≤.25 stable-correct，其余unstable。以独立训练原STR定义primary groups（预期74/284/32）；WSSL定义的同口径groups作为辅助。EMA后续“全部四次错/全部四次对”是不同口径，只列sensitivity；历史全部12次individual预测亦仅sensitivity，各人数不混表。分组包含validation correctness选择效应，不是独立临床表型或前瞻性ambiguity。

严格按ID/label/split/seed对齐原STR与WSSL正式predictions，计算WSSL−STR六指标及p_WSSL−p_STR，与每activity residual removal effect的相关/错误变化仅作描述。不将mask贡献相加当作总gain的因果分解；WSSL训练过程中STR分支也发生共同适配，all-off仍不是独立训练STR。

报告中文，包含四表、BA/AUROC bar plot和CI、所有负贡献/不稳定性/限制、Q1/Q2/Q3、CASE及PROCEED/STOP/INCONCLUSIVE理由。所有代码/报告/小型汇总追加README/handoff并Git提交推送；数据/缓存/feature/逐subject预测/权重不公开。
