# PADS PD/DD 已执行实验总表（2026-11-16）

交付文件：[EXPERIMENT_RESULTS_ALL_20261116.csv](../../conclusions/EXPERIMENT_RESULTS_ALL_20261116.csv)。UTF-8 BOM、单行表头、比例数值，适合 Excel、Python 或 R 导入。

本次仅在服务器既有 Python 环境汇总保存结果：**0训练、0新模型推理、0阈值/normalization拟合**。统计计算、CSV生成和校验均在服务器执行。本机只编辑脚本及接收公开产物。历史数据、配置、模型权重、split、normalization和报告不改写；正式模型指针不变。

## 内容

- 38个阶段、257行，包括早期架构筛选、GroupNorm、训练配方、prototype alignment、双腕消融、STR/WSR、Phase2/3/4、外部SSL适配、EMA、activity scaling、频率先验、Patch、传统读出、参考评分、WSSL消融、Certified L1重建及AI pilot。
- 227行有完整 Accuracy、AUROC、BA、PD Recall、DD Recall、Macro-F1 均值和主SD。
- 19行诊断probe仅保存BA/AUROC，未保存的Accuracy/Recall/F1均值和SD标为`NA`，不为补表重拟合。对应RGD三种融合probe及DSG native/PCA共16种表示probe。
- 11行记录未运行、无可用独立队列、登记、数据审计或复核，不给虚构性能和零SD。它们也不计为已完成性能实验。
- 每行包含目的、问题、结果、历史判定、数据版本、统计单位、来源文件/报告/提交。重复基线/历史复用并非独立新增训练；不应按行数推导新实验或独立样本总数。

## 统计口径

`*_mean`为同一development split先平均seed的指标，再等权平均splits。**不是先平均概率再算指标**。标明ensemble的行使用历史实际ensemble结果，是另一种推理估计量。

`*_SD`通常是上述seed平均后各split值的样本标准差，`ddof=1`。单split三seed的Certified pilot只能给跨seed样本SD，行内`SD定义`明确标注，不能与15-split SD当成同一种不确定性。单seed但15-split筛选可计算split SD，不能计算seed SD。`*_seed_SD`单独记录各seed先跨split平均后的样本SD；无至少两个独立seed均值时为`NA`，包括seed索引0/−1的已汇总或确定性结果。没有将45个相关seed×split run作为45个独立样本。

PD Recall和DD Recall分列；此二分类的BA等于两者均值。所有指标为0–1比例，乘100才是百分比。SD是离散程度，不是置信区间、显著性或外部泛化证明。splits重复使用同一批受试者和训练池；早期多数方法只有seed42，未补跑。

PAG Accuracy由已保存的逐split/seed PD、DD recall和同一固定split支持人数恢复，每项正确数均验证为整数；未访问个体表。PRR仅在服务器读取既有私有corrected scores，沿用原`score>=0`规则计算六指标，逐run回核原AUROC/BA；没有训练、推理或阈值搜索。私有输入逐文件哈希保存在服务器`restricted_manifests/input_sources.json`，个体ID/标签/预测不导出。公开来源哈希见`analysis/source_manifest.json`。

## 解释边界

CSV范围是当前项目保留的development方法与相关诊断/未运行状态，不解封历史FOE-01、outer-final或旧最终测试，不导入其他论文复现，不重新同步退役MFAM。历史预处理与Certified L1分开标记，不能跨版本排名推导单一组件因果效果。Frozen WSSL-STR仍为Retained；AI-WSSL-STR仍为Experimental Candidate，只有3个Certified pilot run，DD Recall取舍仍在。工程登记不是性能实验，Certified正确性PASS也不是模型性能提升。历史reference审计的outer文字接触/整表I/O边界偏差保留在对应行。

## 复现与验证

服务器：`/home/zyt/envs/mfam_analysis/bin/python -B scripts/build_catalog.py`，再运行`verify_catalog.py`。构建默认拒绝覆盖已有产物；复现应使用单独的审核输出目录。源码中的显式输入清单与`analysis/source_manifest.json`提供来源追踪。

校验包括CSV读回、行唯一性、单位/缺失值、BA与两类召回恒等式、现有Frozen WSSL六指标精确回核、145个输入文件hash、109个保护文件与48个登记checkpoint、六组不同范围结果的独立NumPy mean/sample-SD检查。详见`analysis/validation.json`和`analysis/independent_verification.json`。

选择性发布继续使用原发布器。新增范围只允许本目录及该CSV；数据、处理信号、checkpoint、cache、个体预测和私有清单不上传。远端精确tree/镜像/服务器快照回读完成后，单独生成服务器发布回执；本README不是发布成功回执。
