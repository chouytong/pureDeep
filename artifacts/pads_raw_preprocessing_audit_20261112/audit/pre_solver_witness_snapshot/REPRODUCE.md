# 重现说明

在具相同私有PADS/历史权重与受保护来源的服务器隔离新目录中，使用/home/zyt/envs/mfam_analysis/bin/python -B；先验证起点与数据链。顺序startup9.py → metadata_schema9.py → tests9_v2.py → diagnostics9_schema_v2.py → statistics9.py / normalization9.py / resampling9.py → supplement9.py → plots9.py → release_privacy9.py → positions9.py → acc_semantics9.py → reports9_v2.py → finalize9_v3.py。各freeze拒绝覆盖，不能在已完成目录直接重复。initial diagnostics9.py为未运行schema v1；不作为正式入口；失败smokev1在logs保存。

plots9使用已有PYTHONPATH=/home/zyt/envs/patch_report_deps_20261008，不安装包。仅下载了当前项目代码用于本地审阅，所有数据诊断和计算都在服务器；私有统计/信号图未下载。公开仓库不含私有数据、权重、IDs/逐人表，因此只凭仓库不能完成原始数据审计，不能声称数据已发布。全390标签质量探索与历史偏差限制保留。
