# 数值正确性与记录独立性

**NUMERICAL_CORRECTNESS=PASS；RECORD_INDEPENDENCE=PASS。**

旧生产λ50/rho40/max5000/tol1e−4反例已复现，独立与混批trend最大差约0.008199391g，函数默认rho1不是生产参数。新solver58组合合成测试通过，覆盖常数/斜坡/已知KKT、多幅值/频率/混频/阶跃、1024/2048、非法输入、失败关闭和独立SciPy短QP参考。测试原始记录不依赖分类误例。

证书q满足|q|∞≤50；D=yᵀD2ᵀq−.5||D2ᵀq||²。保存前float64的原始gap不clip，门槛1e−8+1e−4E_aff，E_aff=.5||y−own affine||²；提前规定负值容忍，超过则失败。完整25740轴独立稀疏算子回读验证，最大gap/tolerance=0.999975888，最大stationarity=7.10542736e-15，舍入范围负gap计数=0。完整逐轴证书仅私有。

| test | max_abs_difference | tolerance | status |
| --- | --- | --- | --- |
| single_batch_reorder_repeat_restart | 0 | 1e-07 | PASS |
| real_pilot_all_orderings | 2.84217e-14 | 1e-07 | PASS |


1e−7g是相同输入不同分组/顺序/进程/cache输出差的独立性门槛，不是从gap推导的逐点最优解误差保证。小规模real固定hash前2腕/活动共22腕66轴，覆盖两长度，最大独立性差2.84217e−14g。float32 residual另检量化，最大绝对误差4.574461272000008e-07g，不宣称无损或float32结果具有原float64相同证书。


float32保守影响另见float32_quantization_impact_summary.csv。设e为保留段saved residual相对float64 residual的误差，前48点误差补零，仅用于定义完整目标的扰动参考，则P(xhat)−P(x)≤||raw−x||₂||e||₂+.5||e||₂²+50||D2e||₁。全轴实测最大保守上界=0.0036134371901117854。这是最坏情况上界，不是实际目标增量或另一个float32证书，也不把保留段重新当原完整目标。
