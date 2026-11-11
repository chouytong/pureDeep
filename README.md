# PADS PD vs DD：当前主线

**统一工作目录：`/home/zyt/deep_final`。** 2026-11-11 完成目录整理。当前保留模型为 **STR-01 + pretrained-frozen HarNet10 wrist residual（Frozen WSSL-STR）**；STR-01 是正式 matched 基线。此目录不是“V8 仍为最终模型”的旧状态。

从这里开始：

- [主线代码和运行说明](mainline/README.md)
- [当前结论与关键证据](conclusions/README.md)
- [完整报告索引](conclusions/EXPERIMENT_REPORT_INDEX.csv)
- [整理后 90 checkpoint 验证](conclusions/organization_20261111/VERIFICATION.json)
- [数据复制和源码来源审计](conclusions/organization_20261111/COPY_AUDIT.json)

```text
deep_final/
  mainline/       当前 STR/WSSL 代码、配置、固定划分、开发入口
  data/           PADS 原始数据、处理后信号、冻结 HarNet10 特征（仅服务器）
  conclusions/    当前结论、所有主线报告入口、整理验证证据
  results/        现有 STR/WSSL/Patch 产物入口；新训练输出位置（本次未训练）
  artifacts/      保留的各阶段正式协议、源码、报告、checkpoint 等原始证据
  archives/       整理前的说明和发布脚本备份
  tools/          整理工具和 GitHub 发布工具
```

| 模型 | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| STR-01（基线） | 0.730718 | 0.697189 | 0.717631 | 0.686068 | 0.777864 | 0.616515 |
| Frozen WSSL-STR（当前保留） | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |

指标是固定 15 个 development splits 内先平均 seeds 42/43/44，再平均 splits；共享受试者，不是独立外部验证。Patch-STR 和 Frequency-Prior 均 REJECT，activity scaling 为 INCONCLUSIVE / 不保留，当前不启动 P3、WSSL+Patch 或额外搜索。

新主线只从本目录的 `mainline/` 和 `data/` 运行，不依赖 `/home/zyt/MFAM`。MFAM 旧目录仍在原处保留，未删除、未导入当前模型工厂。现有历史 `foundation_validation/`、`source/`、`frozen_project/`、`configs/`、`models/`、`BUNDLE_MANIFEST.json`、`SHA256SUMS` 是原阶段归档，不能当作新的统一入口；其中旧路径和旧结论保留用于 provenance，不重写冻结记录。

代码、配置、报告、聚合结果可同步 GitHub；`data/`、private checkpoint、个体预测、临床个体表、缓存不上传。后续只从 `deep_final` 收集新内容；仓库既有 `MFAM/` 作为冻结历史保留，不再从旧目录同步。[同步规则](GIT_VERSION_CONTROL.md)

[当前模型机器可读清单](conclusions/CURRENT_MODEL.json) · [目录整理验收报告](conclusions/organization_20261111/ORGANIZATION_REPORT.md)
