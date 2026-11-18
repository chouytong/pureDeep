# Phase10 最终报告

三个独立候选均未通过预登记pilot资源闸门。实际新增深度训练9 run（每方向context0/inner0×seeds42/43/44），人口学D0为15次train-only Logistic Regression拟合。没有新增42run矩阵、FT-L2或OOF训练。所有训练及统计在服务器既有环境完成。

| candidate | status | formal_runs | splits | seeds | complete45 | retained | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FT-L1 | PILOT_REJECT_STOP | 3 | 1 | 3 | False | False | BA_positive; BA_improved_at_least2of3; AUROC_non_decrease; DD_recall_drop_at_most01 |
| DEMO | PILOT_REJECT_STOP | 3 | 1 | 3 | False | False | BA_positive; BA_improved_at_least2of3; AUROC_non_decrease; DD_recall_drop_at_most01 |
| DD | PILOT_REJECT_STOP | 3 | 1 | 3 | False | False | NLL_increase_at_most05 |
| FT-L2 | NOT_RUN_NO_INDEPENDENT_DEPTH_EVIDENCE | 0 | 0 | 0 | False | False | FT-L1 rejected; no deeper-residual-stage bottleneck evidence |
| NEW_DEPLOYABLE_THRESHOLD | BLOCKED_NO_LEGAL_CERTIFIED_TRAIN_ONLY_OOF | 0 | 0 | 0 | False | False | Validation curves descriptive only |


九指标mean ± sample SD：完整基线为15split SD，候选和匹配pilot为3seed SD，两者不能直接当同一不确定性比较。

| condition | runs | splits | seeds | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0-DEMOGRAPHIC-ONLY | 15 | 15 | 0 | 0.602816 ± 0.047863 | 0.574716 ± 0.053490 | 0.607880 ± 0.045609 | 0.560924 ± 0.048774 | 0.642318 ± 0.053055 | 0.507114 ± 0.087519 | 0.679529 ± 0.020478 | 0.485406 ± 0.018845 | 0.095765 ± 0.048013 |
| DD | 3 | 1 | 3 | 0.711111 ± 0.054155 | 0.688826 ± 0.027810 | 0.726097 ± 0.025938 | 0.674391 ± 0.042093 | 0.743243 ± 0.094595 | 0.634409 ± 0.049275 | 0.895885 ± 0.132137 | 0.468835 ± 0.048151 | 0.198473 ± 0.008535 |
| DEMO | 3 | 1 | 3 | 0.695238 ± 0.034339 | 0.668192 ± 0.015117 | 0.711421 ± 0.006767 | 0.655223 ± 0.023577 | 0.734234 ± 0.066661 | 0.602151 ± 0.049275 | 0.768642 ± 0.177079 | 0.464446 ± 0.063881 | 0.154921 ± 0.068292 |
| FT-L1 | 3 | 1 | 3 | 0.695238 ± 0.016496 | 0.668192 ± 0.025662 | 0.704882 ± 0.016005 | 0.655298 ± 0.021794 | 0.734234 ± 0.007802 | 0.602151 ± 0.049275 | 0.612145 ± 0.022177 | 0.416861 ± 0.018103 | 0.058748 ± 0.018784 |
| STR-CERTIFIED-L1 | 45 | 15 | 3 | 0.717855 ± 0.023684 | 0.686585 ± 0.029236 | 0.712168 ± 0.030678 | 0.673253 ± 0.024932 | 0.761759 ± 0.040893 | 0.611410 ± 0.073724 | 0.685323 ± 0.058867 | 0.431873 ± 0.025152 | 0.134591 ± 0.030661 |
| WSSL-CERTIFIED-L1 | 45 | 15 | 3 | 0.741534 ± 0.052387 | 0.718900 ± 0.033726 | 0.759285 ± 0.042134 | 0.703561 ± 0.045305 | 0.773894 ± 0.091423 | 0.663905 ± 0.075380 | 0.623037 ± 0.075107 | 0.390936 ± 0.046687 | 0.113658 ± 0.019914 |
| WSSL-CERTIFIED-L1-MATCHED-PILOT | 3 | 1 | 3 | 0.695238 ± 0.034339 | 0.671316 ± 0.019097 | 0.711857 ± 0.006767 | 0.657144 ± 0.026596 | 0.729730 ± 0.058904 | 0.612903 ± 0.032258 | 0.770490 ± 0.176490 | 0.465705 ± 0.063156 | 0.150485 ± 0.076606 |


配对点估计：

| candidate | metric | mean_delta | median_delta |
| --- | --- | --- | --- |
| FT-L1 | accuracy | 7.40149e-17 | 0.00952381 |
| FT-L1 | ba | -0.00312409 | -0.00152572 |
| FT-L1 | auroc | -0.00697472 | -0.00697472 |
| FT-L1 | macro_f1 | -0.00184557 | -0.00225506 |
| FT-L1 | pd_recall | 0.0045045 | 0.0405405 |
| FT-L1 | dd_recall | -0.0107527 | -0.0322581 |
| FT-L1 | nll | -0.158344 | -0.180021 |
| FT-L1 | brier | -0.048844 | -0.0687612 |
| FT-L1 | ece | -0.0917368 | -0.116885 |
| DEMO | accuracy | 0 | 0 |
| DEMO | ba | -0.00312409 | 0 |
| DEMO | auroc | -0.00043592 | -0.00043592 |
| DEMO | macro_f1 | -0.00192021 | 0 |
| DEMO | pd_recall | 0.0045045 | 0 |
| DEMO | dd_recall | -0.0107527 | 0 |
| DEMO | nll | -0.00184754 | -0.00183274 |
| DEMO | brier | -0.00125844 | -0.00103946 |
| DEMO | ece | 0.00443689 | 0.000400611 |
| DD | accuracy | 0.015873 | 0.00952381 |
| DD | ba | 0.0175094 | 0.016129 |
| DD | auroc | 0.01424 | 0.00610288 |
| DD | macro_f1 | 0.0172476 | 0.0126021 |
| DD | pd_recall | 0.0135135 | 0 |
| DD | dd_recall | 0.0215054 | 0.0322581 |
| DD | nll | 0.125395 | -0.0293578 |
| DD | brier | 0.00313023 | -0.00098923 |
| DD | ece | 0.0479887 | 0.0032192 |


## 1. HarNet部分解冻是否优于完全冻结？

未建立。FT-L1单split三seed平均BA/AUROC均下降，未扩大；不能把小pilot概括为所有适配均无效。

## 2. 最后一与最后两block是否有差异？

NOT_TESTED。实际末stage layer5没有内部ResBlock，layer4含两个；FT-L2未运行，不能编造差异。历史Layer5位置相同，新版本Certified数据/norm、fresh下游、LR1e-5及冻结BN affine有区别，不宣称结构创新。

## 3. 部分解冻是否提高DD Recall？

本pilot均值下降0.010753，未支持。个别seed结果不能替代联合门槛。

## 4. 人口字段单独具有较强分类能力？

D0的15split BA约0.574716、AUROC约0.607880，存在有限判别信息，低于WSSL；不能称强或病理因果。

## 5. 人口融合是否提供额外增量？

未建立。DEMO pilot BA下降0.003124、AUROC下降0.000436、DD Recall下降0.010753，未扩大。

## 6. 人口增量是否受年龄/性别/设备混杂？

原队列存在分布和gender/device类关联，缺失率0；分层结果及设备×人口分布已保存。由于未出现稳定增量且小组样本有限，无法声称控制了混杂后的独立增量或因果。

## 7. DD错误主要来自某来源？

unique-subject consensus中Other Movement Disorders的recall约0.4833（60人），错误负担最大；ET28、Atypical15、MS11也有错误，不能仅归因单一极小来源。它们是粗来源类，consensus不是正式单模型成绩。

## 8. DD-aware loss是否改善平衡？

pilot BA/AUROC及两类Recall点估计上升，但NLL增加0.125395超过冻结0.05门槛，Brier/ECE也恶化；联合资源闸门REJECT，不搜索权重。

## 9. 阈值能否改善操作点？

描述曲线显示两类recall的操作点可互换，但不能据validation择优宣布收益。无合法Certified train-only OOF，部署阈值BLOCKED，保持0.5。阈值不改变AUROC或proper概率指标。

## 10. 哪个候选通过联合保留？

无。三个候选仅完成pilot且均拒绝扩展，没有45run保留判定。FT-L2未运行。

## 11. 哪些方向停止？

停止本轮FT-L1、当前人口融合及DD×1.10的扩展，不自动启动FT-L2或组合模型。历史Activity Interaction不扩大。

## 12. 当前最佳模型需要更新？

不更新。Certified WSSL仍为本轮研究基线，Frozen WSSL-STR正式Retained身份与CURRENT_MODEL指针原样保留。不存在本轮性能晋升。

## 统计、来源与交付边界

15个split有受试者重叠，三个seed也不是三个患者队列。新候选仅1 split×3 seed，全部未通过资源闸门，不能据此证明泛化劣于基线或发布15split显著性结论。历史数据与Certified数据分开，当前正式模型指针未变。详见统计附录、正确性与来源、成本报告及每阶段报告。受试者cluster CI是固定拟合条件性诊断，不用于绕过停止门槛。训练/推理时间和参数为实测，缓存训练与在线推理分别说明。原257行历史CSV逐值保留，本轮A/B/C/D追加9行，累计266行，新增NLL/Brier/ECE及参数/耗时字段，未知SD为NA，不伪造。

最终发布必须同时核对远端commit/tree、服务器公开source、发布镜像和PUBLICATION_MANIFEST。阶段A/PREB/B/C/D各自已有回读收据；最终收据保存在服务器项目之外，并在交付回复中给出实际提交与tree。私有数据、信号、权重、IDs和逐人预测不发布。完成回读后停止，不开展Phase11。
