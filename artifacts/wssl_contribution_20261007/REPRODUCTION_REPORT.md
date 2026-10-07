# Phase A：完整复现与屏蔽接口验收

2026-10-07，**PASS，尚未运行正式反事实贡献矩阵**。

全部45个正式 WSSL checkpoint 完整 validation inference通过：subject ID/label/split、两个归档概率列、固定.5决策和六指标一致；最大归档概率差异 1.11e-16。15次原inner-train normalization refit对45checkpoint mean/std逐位一致。SSL按subject ID索引，只使用当前validation行，并检查activity顺序/两腕/finite/window count与whole cache SHA。保留模型143,172参数，HarNet frozen cache不修改、不重提取。

复现指标：Accuracy 0.745604、BA 0.719644、AUROC 0.759553、Macro-F1 0.706589、PD Recall 0.782344、DD Recall 0.656945。每split三seed先均值；45单元均同split等权，结果一致。

归档正式CSV没有logits，因此不能声称对照了不存在的归档logits。所有validation batches均比较历史原forward、新独立显式SSL接口、复用已编码wrist/residual后的诊断forward，logits逐位相同；同时完整归档概率/decision/metrics一致。该接口只是独立诊断代码，不修改正式forward源文件，也不声称当前源文件字节与历史树完全相同。

37个预定条件（full+11activity+2wrist+22activity×wrist+1all-off锚点）在首个真实8subject validation batch中，快速诊断forward与在原投影输出处加mask的直接hook前向logits逐位相同。零SSL输入与真正zero-residual不同，其最大logit差异 0.00214644；所以必须在Linear(LayerNorm(SSL))之后关闭residual包括bias。STR原wrist表示保留。重载/ID重排、模型state与cache保持检查PASS。

没有optimizer、backward或任何训练（包括scratch训练）；没有outer loader/outer outcome/performance。只有train-only normalization refit需要inner-train signals；inference仅validation。缓存NPZ在解压时可能读取整个存储数组，但生成tensor前按当前validation ID选择，其他行不进入模型或统计；这不应描述成对outer cache的分析。

主stable-error分组采用DSG/RGD/PRR/PAG四次seed-first appearance的≥.75/≤.25规则，以原STR定义primary groups，EMA严格all4口径单列sensitivity。后续不根据分组调参。Protocol与四个诊断/绘图脚本及闸门已SHA冻结，反事实性能尚未计算。阶段Git推送完成后开始正式counterfactual inference，不启动Phase B/C。
