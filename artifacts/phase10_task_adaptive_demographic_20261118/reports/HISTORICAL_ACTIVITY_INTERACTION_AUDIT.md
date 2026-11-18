# 历史来源补充核对

Phase8逐run baseline/AI预测90份文件均由既有哈希清单验证，query身份逐split核对。只读独立逐run预测，不解码旧整队列信号/embedding缓存。错误转移与margin单列LEGACY_L1_HISTORICAL_ONLY，不能进入Certified主比较或为本轮初始化。

历史Layer5实际45个config与epoch日志均核对：classifier初始LR2e-4；encoder2e-5来自实际optimizer hook的1/10参数组定义，旧日志仅记录首组LR，不冒称独立记录了encoder每epoch LR。BN running固定但affine原可训练。旧adapted的开发均值：

| variant | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- |
| adapt | 0.751827 | 0.724402 | 0.768908 | 0.712829 | 0.790493 | 0.65831 |


这些是旧L1结果，不是本轮FT-L1结果；本轮改用Certified/norm、fresh下游、encoder1e-5且BN affine也冻结，仍不是新的网络结构。
