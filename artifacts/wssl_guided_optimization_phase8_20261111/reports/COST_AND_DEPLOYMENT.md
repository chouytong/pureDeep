# 参数与耗时边界

| condition | formal_fits | classifier_parameters | trainable_parameters | frozen_HarNet_parameters | train_validation_seconds_sum | mean_epochs | mean_best_epoch | optimizer_steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| activity_interaction | 45 | 210694 | 67522 | 10457408 | 102.484341 | 20.466667 | 8.466667 | 24170 |
| subject_supcon | 45 | 143172 | 143172 | 10457408 | 586.057875 | 22.733333 | 10.733333 | 26852 |


| measurement | mean_over45_models |
| --- | --- |
| batch_size | 8.000000 |
| baseline_resident_classifier_ms | 2.693165 |
| baseline_cached_bilateral_head_ms | 0.245328 |
| AI_cached_bilateral_head_ms | 0.590973 |
| SupCon_resident_classifier_ms | 2.692099 |
| AI_composed_classifier_estimate_ms | 3.038810 |
| baseline_classifier_parameters | 143172.000000 |
| AI_classifier_parameters | 210694.000000 |
| AI_new_trainable_parameters | 67522.000000 |
| HarNet_shared_frozen_parameters | 10457408.000000 |


B只训练新增参数，原143172分类侧全部冻结；C训练143172分类侧，10457408 HarNet固定。原模型部署共10600580参数；三seed ensemble共享同一官方HarNet时唯一参数数10886924，含三个独立143172分类侧；不共享时需三个HarNet实例。固定ensemble不是单模型。

training_validation_seconds含训练和既定每epoch验证，是45run的串行总和，不是一次推理时间；stage总耗时还包含setup/审计/geometry。原baseline训练属于历史成本，本轮新增深度训练0；A1新增forward0。A2共180finalclassifierfit与45train-onlyScaler，SVC内部概率估计还含内部拟合，不能把180当全部内部fit步数。readout_parameters.csv保留逐模型fit、CPU predict秒、数值参数计数和保存字节；CPU predict只对已标准化query、排除encoder/scaler，不能直接与GPU完整分类侧毫秒作速度排名。

新增计时统一使用各split/seed最前8个inner-train患者，禁用trainaugmentation；输入已在GPU，warmup3、repeat10、同步计时，RTX5090D v2、torch线程4。baseline/C测原始六轴经本地WSSL分类侧+已缓存HarNet external到logits；不含磁盘、预处理或HarNet编码。B单独测相同缓存bilateral→logits，与baseline缓存head对照；AI_composed_classifier_estimate为baseline全分类侧−baselinehead+AIhead，属于组合估计，未测端到端服务延迟。ensemble_cost_estimate是三个seed分类侧计时之和、HarNet可共享且本表不含其编码。所有时序是当前硬件微基准，无临床实时性结论。
