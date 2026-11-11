# Correctness 与 provenance 最终审计

已通过的范围：225个既有模型的当前hash/训练query排除ID/标签/归一化来源；45个Frozen特征包保存概率与归档预测一致；45个既有fusion的输入/Scaler/系数线性重放，最大概率误差2.22e-16且全部原决策一致；所有模型六指标与归档一致（容差1e-12）；ID重复/allowlist负测试；canonical加权混淆矩阵/AUC的bootstrap恒等；15unit seed-first与subject-cluster点估计一致；4/36/3/4四个BH族大小保持；缺失表征数据为NA。

不能宣称全部研究能力通过：reference逐人OOF与完整Scaler/PCA/bank状态MISSING，无法独立重放完整参考拟合链；adapted成对表征及raw双logitsMISSING，不能计算cosine/L2/norm或raw DD logit drift。旧feature包缺创建时逐特征hash，当前依据来源代码、当前锁和概率一致性，未重新推理独立验证embedding。旧TF32梯度精度失败不因本轮纯分数审计而变成通过，CE仍NOT_RUN_GATE_FAIL。

失败修订如实保留：v1有一处多余括号，未进入执行；v2错误要求历史split里的test ID元数据为空，改为与canonical excluded名单完全相等；v3误把临床来源label_mapping当数值编码，v4同时核验class_names=[PD,DD]与临床映射。没有改数值容差、评分算法、协议或允许范围；三次修订在新Phase5统计前完成，原脚本/hash/日志保留。audit_assets.py是保留的失败快照，active入口为audit_assets_v4.py；不能声称所有历史脚本均syntax PASS。

静态检查active分析入口无fit/fit_transform、model build、DataLoader、train/eval epoch、backward/optimizer step或harnet调用。运行只读既有分数/参数，神经网络实例化0、训练0、推理0、新分类器/温度/Scaler/PCA拟合0、阈值/系数搜索0。该源代码检查不是系统级I/O隔离证明。

永久保留上轮outer汇总文字接触和整表/整数组读取偏差。本轮不解码全球缓存、不读原始IMU、不新建outer loader、不读outer performance。文件hash读取完整字节；CSV逐行filter也不等于物理零无关行读取。不把重复开发、seed/split波动和固定模型cluster CI冒充独立外部验证。

受保护资产清单和全部fit/query/excluded ID在restricted_manifests，禁止上传。公开仅聚合计数、审计状态和来源hash。当前mainline/历史正式产物起止hash不变，当前retained模型不变。最终发布receipt在/home/zyt/下独立保存，不修改源报告自指提交SHA。
