# 候选与停止点

| priority | candidate | status | reason |
| --- | --- | --- | --- |
| 0 | 原Frozen WSSL-STR | RETAIN_CURRENT | 45推理精确；无新联合门槛赢家 |
| 1 | 正温度校准sidecar | DIAGNOSTIC_ONLY | NLL均值规则通过；Brier/ECE不确定；不部署替换主模型 |
| 2 | 双参考relative评分 | COMPLETED_EXPLORATORY | 优于单参考，低于WSSL；受历史outer文字接触偏差限制 |
| stop | 两分数融合 | REJECT | BA/F1/DD/seed guard失败 |
| stop | 固定分母CE | NOT_RUN_GATE_FAIL | 历史风险目标未确认；实际TF32精度失败 |
| future | 新内部适配/局部网络 | NOT_AUTHORIZED | 缺少足够稳定增量和机制证据，不起训练 |


后续仅可先审阅明确的机制假设与周期/模态隔离证据；当前无足够稳定补救增量，不启动新内部适配或局部训练协议。若未来另获授权，应先隔离开发资料、重新冻结范围与真正独立验证方案。不得把本轮诊断重新标记为未接触outer结果。
