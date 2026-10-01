# MFAM latest-v2 清理报告

执行时间：2026-08-21 22:49–23:02 CST  
活动项目：`/home/zyt/MFAM`  
可恢复隔离区：`/home/zyt/MFAM_cleanup_archive_20260821-224905`  

## 结果

- 清理完成，活动项目只保留 v2 subject-level 多活动主线；
- 没有永久删除文件，也没有使用递归删除命令；
- 旧 v1 数据、旧配置/脚本/报告、历史实验和缓存均移至隔离区；
- 活动项目从约 1.8 GB 降至约 1.4 GB；隔离区约 470 MB；
- 隔离区仍占用磁盘，只有用户手动永久删除后才会释放空间。

## 当前保留

- 唯一 active 配置：`configs/pads_multi_activity_v2.yaml`；
- 原始数据副本：`data/raw/pads`；
- v2 数据：`data/processed/pads_multi_activity/v2_l1_full_length`；
- 当前 subject split；
- SubjectMFAM 训练、评估、推理和 ensemble 分析代码；
- v2 正式 seeds 42/43/44 与正式 ensemble；
- v2 数据、训练及当前技术说明文档。

## 源码调整

- 移除独立 CNN1D baseline 注册和 active 源文件；
- `build_model` 与配置验证不再接受 CNN1D 名称；
- canonical v2 YAML 移除未使用的 `model.cnn`、`model.classifier`；
- canonical 正式参数统一为 epochs=50、train/eval batch size=8；
- `run_test_after_training=false`，使 test 与 validation 阈值选择分离；
- audit、split 和 static-check 脚本默认指向 v2 配置；
- README 与测试改为 latest-v2 语义。

没有修改 SubjectMFAM、WristMFAMEncoder、频带、卷积、Channel Attention、MIL、Top-K、bilateral fusion、activity attention 或最终分类头的计算。

## 验证

- 清理前：42 tests passed；
- 清理后：40 tests passed；减少的 2 个测试只属于已移除的 CNN1D baseline；
- static compilation：50 Python files passed；
- v2 audit：390 subjects、4,290 activity records、PD/DD activity records 3,036/1,254；
- split：264/47/79 subjects；
- normalization：`[11,2,6,1]`；
- 参数量：121,906；
- v2 audit、quality summary、split、三个 best checkpoints、ensemble test metrics 的 SHA-256 全部不变；
- 固定 subject 006 清理前后 probability 最大绝对差为 0；
- 固定 subject 006 清理前后 activity attention 最大绝对差为 0；
- 隔离区 10,020 个非缓存文件逐文件与清理前 SHA-256 匹配，无 mismatch。

## 审计文件

均位于 `/home/zyt/MFAM_cleanup_archive_20260821-224905/audit`：

- `inventory_before.tsv`
- `project_before.sha256`
- `critical_before.sha256`
- `inference_before.json`
- `inventory_after.tsv`
- `project_after.sha256`
- `quarantine_manifest.sha256`
- `quarantine_verification.json`
- `v2_audit_after.json`
- `inference_after.json`
- `cleanup_actions.jsonl`
- `verification_after.json`

恢复方法见隔离区根目录的 `RESTORE.md`。
