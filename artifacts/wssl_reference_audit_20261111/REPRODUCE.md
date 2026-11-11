# 复核与复现入口

仅服务器执行：`/home/zyt/envs/mfam_analysis/bin/python -B`；工作目录 `/home/zyt/deep_final/artifacts/wssl_reference_audit_20261111/scripts`。正式运行版本和GPU记录于analysis/runtime_environment.json；启动时Git SHA为404e83122d3c206328c5a84ceead7646313a9ed3。

已实际执行的顺序（无deep训练、无optimizer step）：bootstrap.py → typicality.py → extract_baseline.py（失败）→ extract_baseline_v2.py → oof_audit.py → loss_audit.py（失败）→ loss_audit_v2.py（失败）→ loss_audit_v3.py → modality_diagnostics.py → calibration_fusion.py → post_statistics.py → audit_bank_norms.py → finalize_reports.py → plot_aggregates.py → verify_final.py。A0、A1与校准部分曾并行等待，但所有候选协议先冻结。测试JSON是实际产生的机器日志，不伪造早期未保存的终端stdout。

冻结文件不能重复创建；下列命令用于已有产物的只读最终核验，不重跑模型、不覆盖freeze：

```bash
cd /home/zyt/deep_final/artifacts/wssl_reference_audit_20261111/scripts
/home/zyt/envs/mfam_analysis/bin/python -B verify_final.py
```

科学代码的入口和输出路径在common.py固定于独立目录；要从零重放，应先另建新的授权实验目录并显式调整HERE，保留原输出与freeze，不能在现有目录强行覆盖。最初bootstrap是一次性锁，不是日常复核命令。不得用通用nested-CV完整runner替代这些inner-only入口。

保护清单：configs/protected_assets.json为历史与规范源码/配置/权重/预测锁；configs/*_freeze.json为每次入口对应源码hash；analysis/source_output_manifest.json给本轮公开源码/结果hash，私有数据fit-ID及checkpoint provenance仍在restricted_manifests和原历史位置。新数据特征、校准/融合系数、逐人预测均不上传GitHub。

只用已安装环境。图形在独立进程引用已有 `/home/zyt/envs/patch_report_deps_20261008`；正式模型依赖未更改。无新的GPU显存峰值测量，不追填推测值。

已知偏差见AUDIT_REPORT/decision.json：历史综合报告带旧outer汇总；不能将“新fit-ID链合法”冒充“本轮完全无outer文字接触”。停止于当前审计结果，不因复现指南自动开启候选或发布。
