# 冻结基线复现

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 |


45个checkpoint全部独立重新推理；完整validation概率最大绝对差1.1102230246251565e-16，决策/指标一致。归档没有logits列，故未声称与归档logits比较；新logits保存在私有features。PD=0/DD=1，原argmax在tie选0；不是三seed概率ensemble。运行import根目录 /home/zyt/deep_final/mainline，非旧foundation副本。

实际local六通道Acc+Gyro temporal/learned moment wrist64D；外部frozen HarNet10 Acc1024D，经LayerNorm+Linear到64D后相加；bilateral258D、activity/subject path、11×16 ordered STR路径保持，最终logits相加。分类侧143,172可训练参数，外部10,457,408冻结；本轮不实例化更新外部encoder，以受保护缓存推理。完整zero-residual含投影bias，zero SSL input不能等价替代。独立显式接口避免deepcopy旧闭包wrapper。

tests/model_correctness.json覆盖三seed代表batch的真长度/padding、无效腕活动、ID重排、错误ID/重复ID/标签错位、reload、实例参数/cache/RNG隔离、分支logits分解。所有45概率核验不是只有代表batch测试。复用此前15个完全精确的train normalization refit证明（hash见baseline_reproduction.json）；OOF另重算45组，覆盖135模型。checkpoint来源/hash由protected_assets及OOF inventory锁定。

135历史deep-OOF状态和预测实际可用，均严格加载；训练/归一化排除query，全部来源排除当前target validation。固定epoch10、forced last，不以query指标选checkpoint。OOF较小训练池与full best-epoch分数尺度不等同。

初版失败：CPU softmax与CUDA归档差2.98e-8且train/val混合batch；v2分离loader并在CUDA softmax恢复原路径，未放宽1e-12 tolerance。旧脚本/冻结记录保留。
15 个 development splits 共享受试者和训练池，45 次已有 seed/run 不是 45 个独立样本。主统计先在同一 split 平均三个 seed 的指标，再汇总 15 splits；不是概率 ensemble。主 bootstrap seed=20261111、10,000 次；Wilcoxon 和 BH family 见各表。受试者分层 cluster bootstrap 将同一人的全部 appearance 一起重采样，仅描述固定已拟合模型下的条件性不确定性，不覆盖重训练、开发多重选择或外部泛化。5 个 context 汇总仅为 inner-development 分组，不是 outer-test。
