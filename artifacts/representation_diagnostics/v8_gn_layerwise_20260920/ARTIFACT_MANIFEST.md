# LRD-01 归档清单

该归档对应冻结 V8-GN 的 layer-wise representation diagnosis。模型未训练、架构未修改，所有分析仅使用固定 inner-development train/validation。

- `LAYERWISE_REPRESENTATION_DIAGNOSIS_REPORT.md`：正式方法学报告和结论分级。
- `run_layerwise_representation_diagnosis.py`：逐层提取、诊断、seed 聚合和 matched-split 统计脚本。
- `protocol.json`：数据边界、层定义和统计单位。
- `layer_fold_metrics_45_seed_folds.csv`：45个 seed-fold 的描述性原始层指标。
- `layer_metrics_15_independent_splits.csv`：三个 seed 先聚合后的15个主要统计单位。
- `layer_metric_summary.csv`：各层指标均值、split SD和范围。
- `adjacent_layer_transitions.csv`：相邻层 matched-split 差值与 Wilcoxon 检验。
- `layer_metric_head_ba_relationships.csv`：各层诊断指标与最终 BA 的15-split Spearman 关系。
- `extraction_equivalence_checks.csv`：自定义提取路径与冻结 forward 的一致性检查。
- `dd_subtype_majority_baseline_*.csv`：DD 亚型中心分类的多数类解释基准。
- `SHA256SUMS`：归档文件哈希。

逐 fold 压缩 embedding 位于：

`/home/zyt/deep_final/foundation_validation/outputs/layerwise_representation_diagnosis/v8_gn_frozen_20260920/embeddings`

该目录保留 train subject-level 各层表示、validation subject-level 各层表示，以及 validation wrist/activity token 表示。大体积 embedding 不在 artifact 目录重复复制。
