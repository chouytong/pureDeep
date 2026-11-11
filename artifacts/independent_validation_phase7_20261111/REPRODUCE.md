# 复核与运行入口

全部实验仅在服务器 `/home/zyt/deep_final/artifacts/independent_validation_phase7_20261111` 执行，Python `/home/zyt/envs/mfam_analysis/bin/python -B`，无新依赖。原正式运行顺序：bootstrap7.py → synthetic_tests.py → reference_audit.py(v1停止并保留) → reference_audit.py(active分类修复) → dataset_discovery.py → build_reports7.py → verify7.py。通过前后的日志和协议/源freeze分别保留，不能把重跑覆盖原冻结记录。

代码提供严格内存Stream接口，未实现或暗中调用外部数据下载。现有private模型与元数据授权是复核前提，public仓库不包含权重或数据。运行到新独立目录前必须显式适配HERE与原路径引用；不要原位覆盖本轮已发布输出，不更改mainline/历史源或checkpoint，不放松锁定hash和测试容差来通过检查。

synthetic_tests不是外部临床evaluation入口；未来真实clinical推理须授权/重叠/协议/设备/标签分离与实际格式验收。reference_audit不调用原Space.fit；运行旧typicality/audit_bank_norms以重fit明确不属本轮复核授权。
