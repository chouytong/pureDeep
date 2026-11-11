# 主线目录整理验收（2026-11-11）

统一根目录为 `/home/zyt/deep_final`。当前主线仅为 PADS PD vs DD 的 STR-01 baseline 和 retained Frozen WSSL-STR；没有迁入其他数据集、其他论文复现工程或旧 MFAM 模型工厂。

- 可运行代码与配置：`mainline/`
- 原始/处理后 PADS 与冻结 SSL cache：`data/`
- 当前模型清单：[CURRENT_MODEL](../CURRENT_MODEL.json)
- 当前结论与历史报告入口：[结论索引](../README.md)
- 权重/预测等私有产物入口：`results/`（全部目标仍在 deep_final 内部）
- 整理前的根目录说明与发布脚本：`archives/layout_before_20261111/`

## 验收结果

数据共复制 17,956 文件、1,341,359,169 bytes；另复制 38,200,697-byte 冻结 HarNet10 cache。全部复制前后 SHA256 相同，processed manifests 的信号路径原本为相对路径，不需要改标签、内容或采样数据。

主线模型算子保留原实现和 checkpoint state keys；新配置只迁移数据/划分/output/provenance 路径。完整验证 STR/WSSL 各 45 个 checkpoint 的 development validation 概率、决定、标签和六指标；15 个 train-only normalization refit 全部与原权重一致。最大概率差 1.110e-16。原正式代码、配置、checkpoint、结果及 Patch 冻结哈希检查 PASS。

本次没有新训练、recipe 修改、模型选择、超参数搜索、outer 信号读取或 outer 结果分析。迁移验证为新入口兼容性检查，不构成新的性能证据或外部验证。

## 保留与边界

旧 `/home/zyt/MFAM` 及原数据未删除。它们已退出当前运行与发布来源；历史冻结文件中的旧路径只保留作 provenance。主线 CLI 和新数据目录独立，不需要从 MFAM 导入模型或读取训练信号。哈希对照工具可以读取原文件作迁移比对，这不属于模型运行依赖。

原 `foundation_validation/`、`source/`、`frozen_project/`、根 `configs/`、`models/` 和 V8 manifests 保留为阶段归档，不是当前入口。旧报告和图片仍在原位置，避免破坏冻结引用；统一 60 项报告/协议索引集中在 conclusions。

GitHub 发布新内容只来自 deep_final；已发布的 `MFAM/` 仅按旧 manifest 哈希保留，禁止再从旧工程增量同步。原始/处理后数据、private ledger、冻结 cache、checkpoint、个体预测、临床表、日志均不上传。认证和推送验收另见服务器外部同步 receipt，未核验远端前不声称完成。
