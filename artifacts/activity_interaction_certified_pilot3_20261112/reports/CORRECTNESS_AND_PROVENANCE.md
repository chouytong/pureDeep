# Correctness and provenance

固定context0/inner0，205名train、105名validation；3seed不是3个独立患者队列。基础WSSL及新增模块都在重复development框架用validation选epoch；train监督token为in-sample，不是无监督或end-to-end OOF。患者bootstrap条件于固定模型与本split，不能消除选checkpoint和多轮开发偏差。每seed配对的是同seed基础模型，三个实例同时含base和module seed变化。无seed显著性检验、无三模型ensemble、无10/15正式保留门槛。

Phase8 source4745549dabf34ccc89a740ad361a0410c6a93a4d；Certified sourcebeb57167a7b8b54792f1efed672927396fe8294d。用户固定split/seed在协议冻结，只有本split205 train/105 query允许访问，excluded只metadata；未读excluded数组或outer performance。必要资产hash保护与旧L2私有hash目录链核验，不重做全部历史审计。

原类直接复用，不新增独立head/bias/gate，不重复加baseline/STR。使用实际11活动mask，全部有效；人工masked NaN/Inf不影响输出，全无效明确拒绝。各seed零输出层时logits精确等于本seedbaseline，跨路径与归档精确；只有新增67522参数在优化器，base始终eval、无梯度、参数/buffer前后hash不变，2次synthetic smoke后正式重置seed。reload/repeat与baseline单次计算测试通过。

| seed | zero_residual_bitwise | repeated_eval_bitwise | token_vs_full_logit_max_error | probability_max_error | decisions_equal | subjects |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | True | True | 0.000000 | 0.000000 | True | 105 |
| 43 | True | True | 0.000000 | 0.000000 | True | 105 |
| 44 | True | True | 0.000000 | 0.000000 | True | 105 |


固定原Phase8B token DataLoader workers0、batch8、shuffleGenerator seed、balanced CE仅train标签、AdamW2e−4/WD1e−4/betas.9,.999/cosine50min1e−6/clip5/max50pat12/strictvalBA。AMPfalse，TF32matmulFalse/cuDNNTrue、deterministicwarn_only/cuDNNdetTruebenchmarkFalse，同环境原默认。完整版本见audit/environment.json。

DD p>.5、tie PD；AUROC为两概率列macro；NLL为unweighted raw-logit proper log loss；Brier是两列平方和；ECE为10等宽topconfidence。双方统一原Phase8B的float64 softmax；归档原概率float32差异单列baseline_metric_precision_comparison.csv，实际导出归档logits与float32概率均逐位一致。

私有tokens/checkpoint/逐人预测/IDs与完整来源存restricted_manifests，不发布。保护只读Phase8/L2/mainline/当前模型指针。最终integrity.json和私有哈希闭合，阶段失败及源码/日志按原样保留。
