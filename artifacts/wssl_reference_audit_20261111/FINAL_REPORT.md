# Frozen WSSL-STR 主线审计与双参考诊断：最终报告

原Frozen WSSL-STR继续保留；本轮新增deep training **0**，optimizer更新0，未运行outer evaluation。没有符合联合判别门槛的新主模型。

| 工作 | 状态 | 证据/原因 |
| --- | --- | --- |
| 历史审计 | 完成并披露边界偏差 | 24组+135 OOF资产 |
| 基线 | 复用+45精确推理 | 不重训 |
| A0 | 完成；CE候选未运行 | b/e闸门失败 |
| A1 | 完成/复用 | 90轻量probe；旧adapted45 |
| A2 | 完成；诊断保留 | 45温度；90描述斜率 |
| B | 完成；融合拒绝 | 9确定评分方法；45融合fit |
| 新深模型/outer | 未运行 | 0 / 0 |
| 论文全文/分区 | 部分待核验 | 见LITERATURE_VERIFICATION.md |


| priority | candidate | status | reason |
| --- | --- | --- | --- |
| 0 | 原Frozen WSSL-STR | RETAIN_CURRENT | 45推理精确；无新联合门槛赢家 |
| 1 | 正温度校准sidecar | DIAGNOSTIC_ONLY | NLL均值规则通过；Brier/ECE不确定；不部署替换主模型 |
| 2 | 双参考relative评分 | COMPLETED_EXPLORATORY | 优于单参考，低于WSSL；受历史outer文字接触偏差限制 |
| stop | 两分数融合 | REJECT | BA/F1/DD/seed guard失败 |
| stop | 固定分母CE | NOT_RUN_GATE_FAIL | 历史风险目标未确认；实际TF32精度失败 |
| future | 新内部适配/局部网络 | NOT_AUTHORIZED | 缺少足够稳定增量和机制证据，不起训练 |


| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |


边界偏差（必须保留）：bootstrap 历史审计完整读取的旧综合报告 `reports/PURE_DEEP_RESEARCH_FINAL_REPORT.md` 含旧 V8 outer-test 汇总，派生的初版 ledger/摘录也包含该内容。因此本轮“完全未接触历史 outer 指标”不成立。未加载 outer-test 原始信号/标签表、逐受试者预测，也未运行 outer evaluation；但不能事后撤销已有的文字接触。该历史值未进入新拟合器或选择规则，所有新候选已停止，当前主线未改变。初版 `protected_assets.json` 的 `outer_access:false` 仅是当时声明，由本最终审计更正，不能作为合规证明。新分析只能作为有此偏差披露的 development 诊断。另：压缩 NPZ 缓存按整数组解压后仅索引合法 train/validation 行，存储读取粒度不等于逐行物理隔离；未用 outer 行拟合或评分。不能声称存在覆盖整个进程的强制系统级 outer-access sentinel。

45次完整validation重推理概率误差最大1.11e-16；135历史deep-OOF checkpoint及预测确实保存并通过来源审计。A0正式weighted mean不是bug；历史全局目标未确认，seed44实际TF32梯度误差超预定容差，因此CE候选未运行。local probe的ΔBA+.000476、ΔAUROC+.000166，CI跨0；不能确认Gyro独有增量。

双参考relative AUC .735639，相对common PD-only +.067150（13/15，CI[.042404,.088420]，primary q=.000305）、DD-only +.236204（15/15，CI[.207170,.265064]，q=.000122）；但低于WSSL .759553。一次OOF两分数融合BA .627638，Δ-.092006（0/15胜，CI[-.110506,-.073277]，q=.000122）；DD recall降至.327707，拒绝。互补筛查不等于可部署融合收益。

正温度诊断NLL Δ-.039178（12/15改善，CI[-.064443,-.016595]，q=.016113），Brier/ECE的CI跨0；仅按预登记均值规则保留诊断sidecar，不改变AUROC/.5分类或当前原生模型。完整六指标、mean/median delta、wins/ties/losses、SD、p/q、effect、配对/subject cluster CI及5context敏感性见STATISTICAL_APPENDIX.md。距离无合法概率映射，校准指标NA。

已确认事实与机制推测区分：数值路径/资产/配对结果是本轮证据；“batch reduction导致DD错误”“融合失败由OOF尺度/先验迁移导致”均未证实；Gyro独有增量、cycle结构、论文详细subject协议/JCR分区仍待核验。15split重复开发不能替代独立验证，也不能用不相同CI跨0状态证明方法间显著不同。

服务器产物：/home/zyt/deep_final/artifacts/wssl_reference_audit_20261111。主线代码/mainline、数据/data与历史模型/artifacts均继续位于/home/zyt/deep_final。私有features/predictions/checkpoints/restricted_manifests仅服务器；公开代码/配置/报告/聚合表按既有工具发布。环境/资源见analysis/runtime_environment.json：RTX5090D v2，threads4，正式依赖未改，无依赖安装；推理65.3s、参考25.5s、校准融合16.6s分别为核心进程计时，不是GPU基准，峰值显存未测。

复现命令和执行顺序见REPRODUCE.md。停止点：完成最终hash核验和选择性GitHub提交后结束本轮；不后台扩展搜索，不重训baseline、OOF、adapter、Patch或full HarNet。完整性结论允许“保护资产hash未变”，不允许“所有合规测试通过”。


15 个 development splits 共享受试者和训练池，45 次已有 seed/run 不是 45 个独立样本。主统计先在同一 split 平均三个 seed 的指标，再汇总 15 splits；不是概率 ensemble。主 bootstrap seed=20261111、10,000 次；Wilcoxon 和 BH family 见各表。受试者分层 cluster bootstrap 将同一人的全部 appearance 一起重采样，仅描述固定已拟合模型下的条件性不确定性，不覆盖重训练、开发多重选择或外部泛化。5 个 context 汇总仅为 inner-development 分组，不是 outer-test。
