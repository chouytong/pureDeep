# 下一阶段候选计划（不执行）

保持正式Frozen WSSL-STR single-model和原0.5规则；不因本轮B的BA/AUC提升破例接受DD下降。当前优先级是真正独立队列和表征重复测量可靠性，须先完成患者不重叠、双腕11活动可比性、原始单位/采样率/设备、PD/DD诊断、伦理访问和模型单checkpoint锁定检查。没有满足条件的数据时明确阻塞，不用同一PADS重复划分冒充独立验证。

方法上本轮所测传统读出、B单层interaction、C单一PD-positive SupCon配置的固定研究闭合。不得按最好seed/阈值/新名字扩充失败配置。B留下的是“内容交互改变类间敏感性平衡”的描述假设，不是可自动保留模型；geometry不是临床机制。任何新的类别风险控制或损失设计都需要新的明确问题、独立验证/预登记预算和停止门槛，不能基于本轮validation错人名单训练。

本报告仅规划，Phase9未启动。


| direction | hypothesis | evidence | difference_from_failed_experiments | expected_contribution | training_cost | retain | stop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Independent PD/DD cohort | frozen representation and decision generalize to new patients | repeated PADS gains fail joint gate; Phase7 has no eligible independent cohort | new patients/cohort; no adapter/gate/threshold/recipe search | estimate transportability with clear clinical label and device limits | 0 new fits; budget one locked inference per eligible record after new authorization | prespecified endpoint and sample-size plan before outcomes; report BA/AUC/F1/class recalls with cluster CI | no nonoverlap proof, incompatible PD/DD labels/tasks, unresolved access/ethics, or failed preprocessing correctness |
| Repeated-measure representation reliability | locked embeddings and probabilities are reproducible across repeat sessions | seed stability is not measurement reliability; current geometry is repeated-development descriptive | within-patient repeated sessions, no new supervised auxiliary loss | separate acquisition/session variation from disease differences | 0 new fits; fixed inference only; requires genuine session repeats | prespecified ICC/agreement/embedding reliability endpoints and cohort-specific thresholds before inspection | no true repeats, incompatible sensors/tasks, inadequate precision or unresolved provenance |


这些为待明确授权及数据就绪的候选计划，尚未启动任何Phase9步骤。
