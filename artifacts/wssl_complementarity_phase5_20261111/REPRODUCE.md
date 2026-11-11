# Phase5 复核入口

所有科学分析在服务器执行。正式环境：`/home/zyt/envs/mfam_analysis/bin/python -B`；模型/数据根目录不改，产物只在 `/home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111`。起点Git SHA `38c89251a442bb245a77d4bffc539da050fcdeb1`。

实际执行顺序：bootstrap.py → audit_assets.py（语法失败，未执行）→ audit_assets_v2.py（错误假定历史test ID元数据为空，停止）→ audit_assets_v3.py（错误假定label_mapping是数值编码，停止）→ audit_assets_v4.py（按实际schema完整通过）→ analyze.py → cluster_statistics.py → audit_lineage_supplement.py（补齐保存的45个fusion fit-ID链核验，无方法变更）→ build_reports.py → plot_aggregates.py → verify_final.py。

这三次审计脚本修订没有改scientific protocol、数据允许范围或数值容差；原脚本、注册、失败日志均保留。active资产入口是v4。初版audit_assets.py为故意保留的失败快照，不是运行入口。

已有产物核验命令（读取固定产物，更新本轮核验记录；不重新拟合/推理）：

```bash
cd /home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111/scripts
/home/zyt/envs/mfam_analysis/bin/python -B verify_final.py
```

原bootstrap和统计脚本为一次性冻结执行，不要在现有目录重跑覆盖其正式输出。若要从零重放，应另建明确授权的新目录并修改HERE，保留原freeze；本任务不授权新模型或历史参考结果再生成。

实际运行与测试机器日志在logs/（服务器私有）；公开tests包含合成指标/ID/重放结果和修订记录。完整fit/query/excluded IDs与保护清单在restricted_manifests，仅服务器；不得上传。公开manifest只列本轮可发布源码/聚合产物的hash，不含保护资产/个体名单。历史225 checkpoint严格加载证明按相同hash复用，不能误写成本轮重新推理225模型。

绘图在独立进程引用已安装的 `/home/zyt/envs/patch_report_deps_20261008`，未安装/升级依赖，不改变正式数值环境。统计进程4线程。新深训练、轻量fit、PCA/Scaler fit、温度fit、神经推理及HarNet再提取全部0。

缺失资产和永久继承的outer接触偏差见ASSET_AUDIT/FINAL_REPORT；本轮不会因缺失而自动重新提取或训练。最终发布通过既有选择性工具与已授权GitHub连接；receipt放源目录外避免提交SHA自指。完成后停止。
