# MF-01 Mean-only bilateral fusion ablation

日期：2026-09-20  
结论：拒绝 Mean-only；继续冻结 V8-GN Full bilateral fusion。

## 研究问题与单因素改动

本实验检验：当前 `[Left, Right, Mean, AbsDiff, wrist mask]` 258维双腕融合，相比仅使用
64维 masked bilateral Mean，是否提供稳定且必要的增量疾病信息。

唯一改动是将 activity-level bilateral fusion 改为 Mean-only。除后续层输入维度随之从258
调整为64外，wrist encoder、activity attention、loss、AdamW、LR=2e-4、batch=8、
WD=1e-4、cosine scheduler、数据处理、固定split和评价协议均不变。参数由71,026降至
55,700，减少15,326（21.58%）。

## 统计协议

- seed 42/43/44；每个seed使用相同的5×3 fixed inner-development splits；
- 先在同一split内聚合三个seed，再以15个split作为主要配对统计单位；
- 不把45个相关seed-fold观测当作独立样本；
- 不使用outer-test进行模型选择或结论推导；
- 配对显著性仅作为补充证据，主要依据预登记的多指标保留规则。

## 分类结果

| 模型 | Accuracy | BA | Macro-Precision | Macro-F1 | AUROC | PD recall | DD recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8-GN Full | 0.7154 | 0.6707 | 0.6746 | 0.6631 | 0.6800 | 0.7782 | 0.5631 |
| MF-01 Mean-only | 0.6939 | 0.6494 | 0.6587 | 0.6367 | 0.6671 | 0.7571 | 0.5417 |
| Δ | -0.0214 | -0.0213 | -0.0159 | -0.0264 | -0.0129 | -0.0212 | -0.0214 |

Mean-only三seed BA为0.6514、0.6557、0.6411（均值0.6494，样本SD 0.0075）；
AUROC为0.6707、0.6876、0.6429（均值0.6671，SD 0.0226）。

15-split配对结果：

| 指标 | 平均差 | Mean-only胜出 | Wilcoxon p |
|---|---:|---:|---:|
| Accuracy | -0.0214 | 5/15 | 0.1578 |
| BA | -0.0213 | 4/15 | 0.0554 |
| Macro-F1 | -0.0264 | 3/15 | 0.0353 |
| AUROC | -0.0129 | 6/15 | 0.3591 |
| DD recall | -0.0214 | 5/15 | 0.5245 |

## 表示诊断

| 指标 | V8-GN | Mean-only | Δ | Wilcoxon p |
|---|---:|---:|---:|---:|
| Disease probe BA | 0.6126 | 0.6129 | +0.0004 | 0.9341 |
| Disease probe AUROC | 0.6599 | 0.6623 | +0.0024 | 0.9780 |
| Disease centroid BA | 0.6337 | 0.6221 | -0.0115 | 0.1688 |
| Disease silhouette Euclidean | 0.0501 | 0.0530 | +0.0030 | 0.8469 |
| Subject retrieval Top-1 | 0.0632 | 0.0503 | -0.0129 | 0.000061 |
| Subject similarity gap | 0.1509 | 0.2065 | +0.0556 | 0.000061 |
| Disease-controlled domain AUC | 0.6334 | 0.6096 | -0.0238 | 0.2524 |
| Normalized mean shift | 0.1143 | 0.1189 | +0.0046 | 0.4543 |
| CORAL shift | 0.4727 | 0.3899 | -0.0828 | 0.0020 |

## 结论分级

实验确认：Mean-only明显降低参数、subject retrieval和CORAL shift，但分类BA、Macro-F1、
AUROC及两类recall均下降；15个split中BA仅4胜；subject similarity gap和mean shift没有
同步改善。Mean-only不满足预登记保留规则。

合理推测：Full fusion中的Left/Right/AbsDiff存在非线性互补信息。单分量linear probe相近，
不足以证明这些分量对端到端分类冗余；表示可线性读出性与已训练分类头的有效利用不是同一命题。

尚未验证：互补信息具体来自哪个分量或交互、是否可用更受控的融合方式保留。根据本实验的
单因素约束和停止规则，不继续搜索这些变体。

最终决策：拒绝MF-01，继续以 V8-GN Full fusion + 已冻结training recipe作为唯一backbone。

## 协议偏差说明

seed 42首次调用通用nested runner时，该runner在完成15个inner split后自动运行outer-final/test。
相关outer产物已被隔离，未被用于选择、比较、阈值确定或本文任何指标。seed 42的开发结果随后
仅由15个inner checkpoint重建；seed 43/44均由development-only runner执行，协议文件明确
记录outer-test loader、signal和prediction未访问。该事实必须保留在provenance中。

