# 主线与边界审计

当前 retained single-model 仍为 Frozen WSSL-STR；A0候选未运行，B融合拒绝，校准只保留诊断sidecar。

边界偏差（必须保留）：bootstrap 历史审计完整读取的旧综合报告 `reports/PURE_DEEP_RESEARCH_FINAL_REPORT.md` 含旧 V8 outer-test 汇总，派生的初版 ledger/摘录也包含该内容。因此本轮“完全未接触历史 outer 指标”不成立。没有针对outer-test构建信号dataset、加载其独立预测或运行outer evaluation；但不能事后撤销已有的文字接触。该历史值未进入新拟合器或选择规则，所有新候选已停止，当前主线未改变。初版 `protected_assets.json` 的 `outer_access:false` 仅是当时声明，由本最终审计更正，不能作为合规证明。新分析只能作为有此偏差披露的 development 诊断。读取粒度也需明确：通用CrossArms manifest含全部390人的标签，labels()先读整表再索引合法ID；压缩NPZ先解压整数组再索引合法train/validation行。因此不能声称物理上完全未读任何outer行。当前split的outer标签/特征未进入拟合、查询评分或选择，不等于逐行I/O隔离；保护文件hash核验也读取文件字节。不能声称存在覆盖整个进程的强制系统级outer-access sentinel。

15 个 development splits 共享受试者和训练池，45 次已有 seed/run 不是 45 个独立样本。主统计先在同一 split 平均三个 seed 的指标，再汇总 15 splits；不是概率 ensemble。主 bootstrap seed=20261111、10,000 次；Wilcoxon 和 BH family 见各表。受试者分层 cluster bootstrap 将同一人的全部 appearance 一起重采样，仅描述固定已拟合模型下的条件性不确定性，不覆盖重训练、开发多重选择或外部泛化。5 个 context 汇总仅为 inner-development 分组，不是 outer-test。

完整历史文档157项、24实验组在启动时读入，来源清单与SHA256见 analysis/history_sources.csv；后续补充实际foundation inner stage和135 OOF资产。历史原文、checkpoint、split、归一化、预测与负结果均受hash锁保护，不覆盖历史报告。冻结入口与每次失败修订分别保存 configs/*_freeze.json；最终完整性见 analysis/final_integrity.json。

已确认：HarNet layer5、middle+final、输出adapter、gated fusion、train-only masked adaptation已做且未保留；deep-OOF阈值诊断已做，本轮未重复阈值网格。Activity scaling不是从未尝试，也不是证明有害。BioPM已测受限acc neural export，不是完整BioPM方法。历史固定配置失败不能泛化整个方法家族。

范围限制：本项目当前为PD/DD受试者分类诊断；尚不能当成临床手部连续严重程度回归结果。DD不是健康正常人群。无新的临床因果结论、独立外部验证或新最佳模型声明。
