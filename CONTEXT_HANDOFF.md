# 当前主线交接入口（2026-11-11）

唯一工作根目录：`/home/zyt/deep_final`。先读 `README.md`，再读 `mainline/README.md` 与 `conclusions/README.md`。

当前保留：Frozen WSSL-STR；基线：STR-01。主线代码在 `mainline/`，原始/处理后 PADS 及冻结 SSL cache 在 `data/`。不要继续在 MFAM 中开发或从其配置启动新训练。

整理已通过 90 个开发 checkpoint 的完整验证推理、15 次 train-only normalization 重拟合；模型和训练 recipe 不变。本次未训练，未访问 outer 信号或结果。完整证据在 `conclusions/organization_20261111/`。

所有历史负实验、条件停止规则和冻结边界继续有效。旧完整交接记录保存在 `archives/layout_before_20261111/CONTEXT_HANDOFF.before_20261111.md`；原正式实验产物不改名、不删除、不重写。
