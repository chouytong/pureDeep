# 既有归一化来源（零重拟合）

FACT：45份normalization文件的fit IDs均恰好为对应inner-train，均值/SD为[11,2,6,1]，与各自checkpoint存储tensor逐点一致；同split的3seed数值相同。源码先按subject分组，再只向_activity_statistics传train_records，完整真实信号累计float64 sum/square/count，最后存float32；不遍历padding。没有按patient单独z-score，validation复用旧值。

最小存储std为0.011762863，高于epsilon1e−6，未见触发floor导致常数强行放大的证据。STR采用activity/wrist/channel归一化；HarNet raw g路径不使用此norm，进入projection前的LayerNorm是既有模型运算，不是新fit的输入scaler。

数学上固定非零mean/std的仿射变换可逆，不能说它把每个人幅值都消除了；跨activity/腕的相对尺度被重新参数化，下游GroupNorm可能改变特征利用方式，不能从可逆性推出没有任何可用信息损失。规范里全局、通道、activity、wrist、subject normalization的区别：全局一个统计混单位不合适；当前按activity/wrist/channel保留个体差异；subject-normalization会移除个体均值/幅度，但当前未使用。本轮不计算替代拟合值或重拟合旧值；真实性依赖现有metadata、源码、保护hash和历史重现链，不能冒称本轮又独立重算了训练统计。NO_ACTION：没有支持重开norm搜索的证据。


---
数据来源：服务器PADS v1.0.0的390人PD/DD任务子集，原始数据、既有v2信号及45 Frozen WSSL-STR checkpoint元数据；不是原库全部469人。统计单位为患者，同患者双腕先平均；原始记录/通道数只作描述性计数，不作为独立患者。15重叠development splits、seeds42/43/44用于来源核查，不是独立外部验证。缺失/不可计算值为NA，不补造；频谱超过记录Nyquist不外推；稀有交叉分层<10人的数值统计抑制，完整表仅私有。检验族132/440/880分别BH，患者bootstrap2000次；其余描述性表无额外显著性声明。接触全390人诊断标签及validation成员元数据用于质量关联，未读分类错误名单/分数，未构建outer loader、未解码outer性能/预测；历史outer接触永久披露。全队列探索会增加未来选择偏差，不能叫盲态独立验证。训练/分类器拟合/normalization重拟合均0，Level2未执行。


## 发布前数值正确性修订

本版结合[L1求解器证据](L1_SOLVER_CORRECTNESS_REPORT.md)。此前Phase9发布前草稿与全部源码/聚合输出保存在audit/pre_solver_witness_snapshot；旧CLOSURE_FREEZE按该版本命名空间核验，替代关系见configs/PREPUBLICATION_REVISION.json。正式Phase3–8、数据、模型及normalization均未改；原始质量关联表的p/q未改变。

**修订结论优先于此前逐记录独立假设**：既有L1有限迭代使用全batch共同停止，数值输出可随其他记录变化；真实输出保守目标值证据见新报告。此发现不改原始信号质量统计，不表示标签泄漏或已证明模型性能虚高。
