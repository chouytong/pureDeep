# Stage 3 原型对齐实验归档说明

本目录归档 2026-09-19 完成的 activity-conditioned disease prototype alignment 实验。所有模型选择与表示诊断均基于固定 inner-development protocol，未使用 outer-test 进行设计或选择。

## 归档内容

- `PROTOTYPE_ALIGNMENT_REPORT.md`：完整方法、指标、15-split 配对统计、结论与证据边界。
- `src/`：本阶段使用的模型输出接口、原型损失、训练循环和 nested-development 训练实现。
- `configs/`：PA01（λ=0.01）和 PA02（λ=0.05）的 seed 42/43/44 配置，以及冻结 baseline 参考配置。
- `tests/test_prototype_alignment.py`：原型构建、mask 和损失行为测试。
- `scripts/aggregate_seed_split_diagnostics.py`：将相同 split 上三个 seed 聚合为 15 个独立 inner split 的方法学收口脚本。
- `results/prototype_alignment_comparison_20260919.csv`：baseline、PA01、PA02 的总体比较。
- `results/baseline_methodology/`：Stage 2 baseline 的 15-split 聚合结果。
- `results/pa01/`、`results/pa02/`：两组候选的原始 fold diagnostics 与 15-split 聚合结果。
- `training_summaries/`：六次完整开发实验的配置、环境、实验计划、开发协议与汇总指标。
- `SHA256SUMS`：本归档内文件哈希。

## 训练产物来源

- `/home/zyt/deep_final/foundation_validation/outputs/invariant_representation/pa01_lambda001_seed{42,43,44}_20260919`
- `/home/zyt/deep_final/foundation_validation/outputs/invariant_representation/pa02_lambda005_seed{42,43,44}_20260919`
- `/home/zyt/deep_final/foundation_validation/outputs/invariant_representation/pa01_lambda001_diagnostics_20260919`
- `/home/zyt/deep_final/foundation_validation/outputs/invariant_representation/pa02_lambda005_diagnostics_20260919`

完整 checkpoint、逐 fold 日志及 embedding 仍保留在上述只读来源目录；本归档保存复核结论所需的代码、配置、汇总、诊断与 provenance，不复制大体积冗余张量。

## 决策

- PA01：拒绝。分类性能轻微下降，subject-specific 指标未下降。
- PA02：拒绝。disease probe AUROC 提升，但 subject-specific 信息没有稳定下降，shift 改善也未跨指标一致。
- 冻结结论：继续使用 V8-GN + 已冻结 training recipe；不继续搜索 λ，也不叠加新的不变表征模块。
