# BFC-01 归档清单

冻结 V8-GN bilateral fusion component diagnosis；仅使用 inner-development train/validation，未训练模型或访问 outer-test。

- `BILATERAL_FUSION_COMPONENT_DIAGNOSIS_REPORT.md`：正式报告、证据分级与后续决策。
- `run_bilateral_component_diagnosis.py`：逐分量提取、train-only probe、15-split 聚合和配对统计代码。
- `protocol.json`：分析边界及统计协议。
- `component_metrics_45_seed_folds.csv`：45个 seed-fold 原始分量指标。
- `component_metrics_15_independent_splits.csv`：三个 seed 先聚合后的15个统计单位。
- `component_metric_summary.csv`：各分量均值、split SD与范围。
- `component_pairwise_comparisons.csv`：分量间 matched-split 比较。
- `component_split_winners.csv`：每个 split 的最高/最低分量。
- `component_metric_head_ba_relationships.csv`：与最终BA的相关及BH-FDR结果。
- `wrist_mask_audit_*.csv`：缺失腕与mask常量审计。
- `extraction_equivalence_checks.csv`：与冻结模型 forward 的一致性验证。
- `SHA256SUMS`：归档哈希。

逐 fold 压缩表示位于：

`/home/zyt/deep_final/foundation_validation/outputs/bilateral_component_diagnosis/v8_gn_frozen_20260920/embeddings`

大体积 embedding 不在 artifact 目录重复复制。
