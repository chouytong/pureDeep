# ED-01 Artifact Manifest

- `ERROR_ORIENTED_DIAGNOSIS_REPORT.md`：完整方法、结果、证据边界和停止决策。
- `run_error_oriented_diagnosis.py`：只读冻结表示提取、train-only probe与统计代码。
- `results/protocol.json`：inner-only协议和outer隔离声明。
- `results/summary.json`：主要stable-error与stable-correct结果。
- `results/subject_appearances_45_seed_folds.csv`：逐validation appearance证据指标。
- `results/subject_group_metrics_15_splits.csv`：三seed聚合后的15 split分组指标。
- `results/subject_group_comparisons_bh.csv`：总体分组配对检验及BH-FDR。
- `results/subject_label_group_metrics_15_splits.csv`：PD/DD分层分组指标。
- `results/subject_label_comparisons_bh.csv`：PD/DD分层配对检验及BH-FDR。
- `results/activity_group_evidence_15_splits.csv`：activity×wrist/component×group证据。
- `results/activity_comparisons_bh.csv`：activity级多重校正比较。
- `results/metric_relationships_15_splits.csv`：证据冲突与最终margin/error关系。
- `results/extraction_equivalence_checks.csv`：冻结提取与原forward等价性检查。
- `SHA256SUMS.txt`：以上文件完整性校验。

本目录不包含或引用outer-final checkpoint、outer prediction或outer metric。
