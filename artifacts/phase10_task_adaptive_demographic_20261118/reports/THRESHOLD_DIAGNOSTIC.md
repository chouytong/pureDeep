# 阈值描述性诊断

{"status": "BLOCKED_NO_CERTIFIED_TRAIN_ONLY_OOF", "validation_curves": "DESCRIPTIVE_ONLY", "threshold_changed": false, "proper_losses_and_AUROC_invariant_to_threshold": true, "new_OOF_fits": 0, "legacy_OOF_not_used": true}

验证曲线不用于选择阈值。TP/FP/TN/FN、Recall/Precision/BA/F1均保存。固定概率的AUROC、proper NLL、Brier和top-confidence ECE不随阈值改变，不能归因为阈值改进。

固定15个重叠development splits；同人重复出现。三seed不是独立患者队列。人口记录和逐人预测不公开，正式阈值/模型指针不改。历史outer文字接触和多轮development选择偏差保留；本轮未读取outer性能。
