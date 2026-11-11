# 统计补充

单位是15个配对development split，先各split内平均三seed的指标，不先混合所有appearance概率；ensemble单系统指标直接减基线seed均值。阈值DD>.5，tiePD。AUROC为原代码macro两列AUROC；NLL为未裁剪proper unweighted log loss，Brier为两类平方和，ECE为10等宽topconfidence。

Split bootstrap固定seed20261112、10000次抽15split；Wilcoxon含effect size/dz/rank-biserial及wins/ties/losses。原阶段主族2/10/2/2不变；最终全局主族16=8候选×BA/AUC，次族56=8×其余7指标；logit辅助独立9。缺失XGB与RF非有限NLL检验按p1占位参与调整、自身p/q为NA，不缩小family。BH不修复重叠split依赖。global p/q不是新增事后保留门槛。

患者cluster sensitivity按276PD/114DD分层、10000次多项抽样。同一人的所有split/seed出现共享一个权重；每个draw重新计算每model的九指标，seed内平均后15split等权，与原paired统计相同estimand，不改成pooled appearances。固定已训练模型，不含重新训练不确定性；有零类/非有限draw如实记录，<9500有效draw则NA。校验每个345预测包的单位权重weighted metric对原程序误差≤2e-12，cluster point estimate与原paired mean一致。

| condition | metric | mean_delta | ci_low | ci_high | finite_draws | invalid_draws | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probability_ensemble | ba | 0.000380 | -0.011964 | 0.012934 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| probability_ensemble | auroc | 0.022067 | 0.014831 | 0.029264 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| probability_ensemble | pd_recall | 0.027652 | 0.015878 | 0.039650 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| probability_ensemble | dd_recall | -0.026892 | -0.048482 | -0.005484 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logistic_regression | ba | -0.045297 | -0.062570 | -0.028366 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logistic_regression | auroc | -0.005475 | -0.018616 | 0.007473 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logistic_regression | pd_recall | 0.072414 | 0.055934 | 0.089386 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logistic_regression | dd_recall | -0.163008 | -0.193516 | -0.133050 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| linear_svm | ba | -0.058249 | -0.077286 | -0.039208 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| linear_svm | auroc | -0.012299 | -0.026900 | 0.002169 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| linear_svm | pd_recall | 0.092616 | 0.074888 | 0.110961 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| linear_svm | dd_recall | -0.209113 | -0.242887 | -0.176046 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| rbf_svm | ba | -0.043403 | -0.059789 | -0.027086 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| rbf_svm | auroc | -0.006028 | -0.016844 | 0.004994 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| rbf_svm | pd_recall | 0.085931 | 0.070266 | 0.102211 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| rbf_svm | dd_recall | -0.172737 | -0.201440 | -0.144205 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| random_forest | ba | -0.047045 | -0.063572 | -0.030669 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| random_forest | auroc | -0.018645 | -0.028699 | -0.008361 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| random_forest | pd_recall | 0.056370 | 0.042137 | 0.071203 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| random_forest | dd_recall | -0.150460 | -0.180263 | -0.121447 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| xgboost | ba | NA | NA | NA | NA | NA | NOT_RUN_DEPENDENCY_MISSING |
| xgboost | auroc | NA | NA | NA | NA | NA | NOT_RUN_DEPENDENCY_MISSING |
| xgboost | pd_recall | NA | NA | NA | NA | NA | NOT_RUN_DEPENDENCY_MISSING |
| xgboost | dd_recall | NA | NA | NA | NA | NA | NOT_RUN_DEPENDENCY_MISSING |
| activity_interaction | ba | 0.006161 | -0.003738 | 0.015992 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| activity_interaction | auroc | 0.006691 | 0.000348 | 0.012943 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| activity_interaction | pd_recall | 0.048517 | 0.038105 | 0.059377 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| activity_interaction | dd_recall | -0.036196 | -0.052537 | -0.019798 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| subject_supcon | ba | 0.003271 | -0.007503 | 0.014127 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| subject_supcon | auroc | 0.007708 | 0.000685 | 0.014880 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| subject_supcon | pd_recall | 0.016373 | 0.006794 | 0.026525 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| subject_supcon | dd_recall | -0.009831 | -0.028845 | 0.009575 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logit_ensemble | ba | -0.003905 | -0.016225 | 0.008650 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logit_ensemble | auroc | 0.018632 | 0.012132 | 0.025179 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logit_ensemble | pd_recall | 0.025826 | 0.013706 | 0.038024 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |
| logit_ensemble | dd_recall | -0.033635 | -0.055054 | -0.012285 | 10000.000000 | 0.000000 | SHARED_SUBJECT_CLUSTER_SENSITIVITY |


全部九指标详见subject_cluster_bootstrap.csv。seed_stability.csv同时区分mean within-split3seedSD、三个seed全局均值SD和between-splitSD。固定ensemble seedSD不可识别；between-splitSD是不同诊断。geometry无pair独立性检验。
