# 独立数据兼容性审查（仅元数据，不是外部评估）

本轮没有确认可直接用于当前PD/DD模型的、已授权且输入兼容的独立队列。不能将“有公开PD数据”当作“有可用PD/DD外部test”。下表记录已核查来源和未知项，未下载任何外部sensor/个体表，未申请或接受新数据协议。

| dataset | access | overlap | compatibility | missing |
| --- | --- | --- | --- | --- |
| PADS1.0.0 | open subject to CC BY-NC-SA4.0; already authorized existing server cohort | same source as current390 PD/DD subjects, not independent | input compatible but cohort already used | no independent patients established |
| mPower | controlled research access; current individual grant/use conditions not verified or accepted | separate study design; person-level overlap not verified | not direct PD/DD dual-wrist11task test | authorization; exact device/protocol mapping; DD; overlap review |
| BEAT-PD CIS-PD/REAL-PD | research challenge portal; current access terms/grant not accepted or verified | separate studies; person-level overlap not verified | supports other severity questions, not direct PD/DD test | DD and task/device mapping; access approval; overlap |
| WATCH-PD | 3DT Stage2 members; others may submit proposal to steering committee; no user approval obtained | independent multicenter recruitment described; individual overlap not verified | no direct eligible PD/DD input protocol established | data grant; DD; exact task mapping; overlap review |
| new hospital cohort (not provided) | unknown; no authorization supplied | must be verified by source | NOT_READY | all source/permission/protocol fields |
| independent time/device cohort (not provided) | unknown; no authorization supplied | new time/device alone does not establish new participants | NOT_READY | all source/permission/protocol fields |


原始来源与证据边界：

- [PADS官方数据页](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)核验数据许可、设备/单位；同一PADS重打包不能成为独立队列。[原始论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770131/)支持其单中心及PD/DD/HC定义。本文不借其不同评估数字作性能目标。
- [mPower官方门户](https://www.synapse.org/Synapse:syn4993293/wiki/247860)支持handheld与PD/control任务信息；静态门户没有完整呈现当前个体授权条款，故不称已获访问。原论文/应用研究也不能替用户签署现行数据协议。
- [BEAT-PD官方门户](https://www.synapse.org/Synapse:syn20825169/wiki/600903)及[原研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10047543/)支持free-living严重度/用药任务；CIS/REAL-PD的设备/标签不同，不自动合并为PD/DD。当前数据授权未完成。
- [WATCH-PD原论文](https://www.nature.com/articles/s41531-024-00721-2)说明PD/control、设备与3DT/steering committee申请规则。申请途径存在不等于用户已经获准；没有联系作者或发邮件。

审查并未穷尽所有世界数据来源。没有DD或缺少固定11任务/双腕Gyro时，不能补零后计算原模型“外部准确率”。新医院/采集时间/设备队列仅为待准备项，不虚构医院、样本、传感器或授权。即使有不同研究名称，也须安全核查与本390人及相关训练来源的参与者重叠；外部SSL预训练成员无法核验时亦须披露。

浏览暴露记录：搜索服务曾返回PADS公开file_list个体元数据行摘要；未主动打开该个体文件、未下载或持久化其行、未用于任何本轮统计/拟合。不能把本次浏览过程声称为完全未接触任何个体信息。统计数据仅来自已授权服务器当前development产物，历史outer暴露永久保留。

当前准备结论：模型实例与评价协议已锁，数据/授权/匹配验证未就绪。外部全部metrics为NOT_EVALUATED，不提供伪造CI。具体入组、盲法、模型锁和停止条件见INDEPENDENT_VALIDATION_PROTOCOL.md。
