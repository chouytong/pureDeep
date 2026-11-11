# 当前结论与证据入口

当前保留 **Frozen WSSL-STR**；主线问题是 PADS PD vs DD subject-level supervised classification，不是连续严重度回归。以下结论均来自 development，不能宣称独立外部或临床泛化。

| 模型 | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| STR-01（基线） | 0.730718 | 0.697189 | 0.717631 | 0.686068 | 0.777864 | 0.616515 |
| Frozen WSSL-STR（当前保留） | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |

| 实验 | 状态 | 正式报告 |
|---|---|---|
| STR-01 | retained matched baseline | [STR01_REPORT](../artifacts/str01_structured_token_residual_20260921/STR01_REPORT.md) |
| Frozen WSSL-STR | retained current single-model | [PHASE3B_REPORT](../artifacts/phase3b_external_ssl_20260928/PHASE3B_REPORT.md) |
| WSSL 贡献与复现 | descriptive evidence，未训练新候选 | [REPRODUCTION_REPORT](../artifacts/wssl_contribution_20261007/REPRODUCTION_REPORT.md) |
| WSSL classifier LR 对照 | REJECT，保留原 recipe | [FINAL_REPORT](../artifacts/wssl_review_20261004/FINAL_REPORT.md) |
| Activity scaling | INCONCLUSIVE，不保留 | [FINAL_REPORT](../artifacts/wssl_activity_scaling_20261008/FINAL_REPORT.md) |
| Frequency-Prior STR | REJECT | [FINAL_REPORT](../artifacts/frequency_prior_str_20261007/FINAL_REPORT.md) |
| 固定 200/100 Patch-STR | REJECT；停止 P3/WSSL+Patch | [PATCH_STR_REPORT](../artifacts/patch_str_20261008/PATCH_STR_REPORT.md) |
| Patch 2026-11-11 复核 | 与原归档一致，未重复正式训练 | [RECHECK_REPORT](../artifacts/patch_str_recheck_20261111/RECHECK_REPORT.md) |

[全部主线阶段报告索引](EXPERIMENT_REPORT_INDEX.csv) 统一列出原始正式文件位置，不复制或改写历史统计。索引不包含已封存的 outer 产物。历史归档保留原位，避免破坏冻结 hash、checkpoint 引用和报告图片相对路径。

本次目录整理校验了 PADS 原始/处理后数据副本、冻结 cache、90 个开发 checkpoint 的概率/预测/六指标、15 次训练集 normalization、原正式文件 hash。没有重新训练、模型选择、threshold tuning 或 outer 分析。详细数字见 [VERIFICATION](organization_20261111/VERIFICATION.json) 与 [COPY_AUDIT](organization_20261111/COPY_AUDIT.json)。

[当前模型机器可读清单](CURRENT_MODEL.json) · [目录整理验收报告](organization_20261111/ORGANIZATION_REPORT.md)
