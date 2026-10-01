# MF-01 Mean-only Bilateral Fusion Ablation — Complete

日期：2026-09-20  
状态：三seed inner-development训练、配对比较和表示诊断均已完成。  
决策：拒绝Mean-only；保留V8-GN Full fusion冻结backbone。

## 完成内容

- 唯一结构变量：258维full bilateral fusion → 64维masked bilateral mean；
- Wrist encoder、activity attention、loss、AdamW、LR=2e-4、batch=8、WD=1e-4、
  cosine scheduler、数据处理和固定splits不变；
- seed 42/43/44，每个seed 15个inner-development split；
- 三seed先按相同split聚合，以15个split进行配对比较；
- 分类结果和representation diagnostics均已完成；
- 参数：71,026 → 55,700，减少21.58%；
- 最终只读测试14 passed；artifact SHA256校验全部通过。

## 核心结果

V8-GN / Mean-only：

- BA：0.6707 / 0.6494（Δ=-0.0213，4/15 split胜出）；
- AUROC：0.6800 / 0.6671（Δ=-0.0129）；
- Macro-F1：0.6631 / 0.6367（Δ=-0.0264）；
- PD recall：0.7782 / 0.7571；
- DD recall：0.5631 / 0.5417。

Mean-only降低subject retrieval Top-1和CORAL shift，但subject similarity gap增加且normalized
mean shift轻微增加，分布敏感性没有一致改善。完整结论见`MF01_FINAL_REPORT.md`。

## 协议说明

seed 42首次运行通用nested runner时，在inner开发完成后自动产生outer-final/test产物；这些
outer产物未用于模型选择或本实验任何数值，seed 42开发汇总仅由15个inner checkpoint重建。
seed 43/44使用development-only runner。该执行偏差保留在provenance中。

## 停止规则

Mean-only未达到预登记保留标准，因此不替换V8-GN，也不继续AbsDiff normalization、gating、
weighted mean或其他fusion变体搜索。
