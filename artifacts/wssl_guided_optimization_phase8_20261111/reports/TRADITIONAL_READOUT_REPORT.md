# Phase8A2 固定监督表征的传统读出

唯一主要表示为原subject258+structured176=434D，原两个head恰为此空间的affine读出；没有验证挑层。只在各模型train fit Scaler/classifier，未更新任何encoder；SVC含train-only内部Platt，不将其说成encoder全链OOF。一个患者一行，不跨fold拼接。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_svm | 0.750068 | 0.661396 | 0.747254 | 0.669397 | 0.874960 | 0.447832 | 0.739386 | 0.388745 | 0.152168 |
| logistic_regression | 0.749238 | 0.674347 | 0.754078 | 0.679919 | 0.854757 | 0.493936 | 1.084555 | 0.423008 | 0.195916 |
| random_forest | 0.741508 | 0.672599 | 0.740908 | 0.675718 | 0.838714 | 0.506485 | inf | 0.368994 | 0.100759 |
| rbf_svm | 0.755990 | 0.676242 | 0.753525 | 0.684421 | 0.868275 | 0.484208 | 0.728147 | 0.382762 | 0.150615 |


| condition | metric | mean_delta | wins | ci_low | ci_high | primary_bh_q |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | accuracy | 0.003634 | 5.000000 | -0.008355 | 0.019258 | NA |
| logistic_regression | ba | -0.045297 | 0.000000 | -0.057889 | -0.034540 | 0.000122 |
| logistic_regression | auroc | -0.005475 | 6.000000 | -0.014974 | 0.003150 | 0.412827 |
| logistic_regression | macro_f1 | -0.026671 | 2.000000 | -0.039619 | -0.013698 | NA |
| logistic_regression | pd_recall | 0.072414 | 14.000000 | 0.041540 | 0.107117 | NA |
| logistic_regression | dd_recall | -0.163008 | 0.000000 | -0.210557 | -0.119998 | NA |
| logistic_regression | nll | 0.467835 | 15.000000 | 0.409240 | 0.529963 | NA |
| logistic_regression | brier | 0.036566 | 13.000000 | 0.019323 | 0.052866 | NA |
| logistic_regression | ece | 0.085583 | 15.000000 | 0.066708 | 0.103899 | NA |
| linear_svm | accuracy | 0.004464 | 6.000000 | -0.010695 | 0.023608 | NA |
| linear_svm | ba | -0.058249 | 0.000000 | -0.072308 | -0.045310 | 0.000122 |
| linear_svm | auroc | -0.012299 | 5.000000 | -0.024271 | -0.001480 | 0.137424 |
| linear_svm | macro_f1 | -0.037193 | 1.000000 | -0.053732 | -0.020288 | NA |
| linear_svm | pd_recall | 0.092616 | 15.000000 | 0.061010 | 0.129851 | NA |
| linear_svm | dd_recall | -0.209113 | 0.000000 | -0.254156 | -0.168912 | NA |
| linear_svm | nll | 0.122666 | 14.000000 | 0.088712 | 0.148338 | NA |
| linear_svm | brier | 0.002303 | 10.000000 | -0.016024 | 0.017926 | NA |
| linear_svm | ece | 0.041835 | 13.000000 | 0.026643 | 0.056412 | NA |
| rbf_svm | accuracy | 0.010386 | 7.000000 | -0.002889 | 0.025442 | NA |
| rbf_svm | ba | -0.043403 | 0.000000 | -0.056857 | -0.030671 | 0.000122 |
| rbf_svm | auroc | -0.006028 | 5.000000 | -0.010517 | -0.001195 | 0.058899 |
| rbf_svm | macro_f1 | -0.022168 | 4.000000 | -0.035622 | -0.008793 | NA |
| rbf_svm | pd_recall | 0.085931 | 15.000000 | 0.057904 | 0.118207 | NA |
| rbf_svm | dd_recall | -0.172737 | 0.000000 | -0.216464 | -0.132975 | NA |
| rbf_svm | nll | 0.111426 | 14.000000 | 0.077951 | 0.140463 | NA |
| rbf_svm | brier | -0.003679 | 7.000000 | -0.017661 | 0.007958 | NA |
| rbf_svm | ece | 0.040283 | 13.000000 | 0.024723 | 0.054515 | NA |
| random_forest | accuracy | -0.004096 | 5.000000 | -0.015631 | 0.008606 | NA |
| random_forest | ba | -0.047045 | 0.000000 | -0.058578 | -0.035788 | 0.000122 |
| random_forest | auroc | -0.018645 | 0.000000 | -0.023847 | -0.013442 | 0.000122 |
| random_forest | macro_f1 | -0.030872 | 1.000000 | -0.041538 | -0.019463 | NA |
| random_forest | pd_recall | 0.056370 | 12.000000 | 0.028701 | 0.086163 | NA |
| random_forest | dd_recall | -0.150460 | 0.000000 | -0.192277 | -0.111338 | NA |
| random_forest | nll | NA | NA | NA | NA | NA |
| random_forest | brier | -0.017447 | 5.000000 | -0.030992 | -0.005262 | NA |
| random_forest | ece | -0.009574 | 7.000000 | -0.023220 | 0.002622 | NA |
| xgboost | accuracy | NA | NA | NA | NA | NA |
| xgboost | ba | NA | NA | NA | NA | NA |
| xgboost | auroc | NA | NA | NA | NA | NA |
| xgboost | macro_f1 | NA | NA | NA | NA | NA |
| xgboost | pd_recall | NA | NA | NA | NA | NA |
| xgboost | dd_recall | NA | NA | NA | NA | NA |
| xgboost | nll | NA | NA | NA | NA | NA |
| xgboost | brier | NA | NA | NA | NA | NA |
| xgboost | ece | NA | NA | NA | NA | NA |


logistic_regression: **REJECT**。

linear_svm: **REJECT**。

rbf_svm: **REJECT**。

random_forest: **REJECT**。

xgboost: **NOT_RUN_DEPENDENCY_MISSING**。

本比较不能证明线性可分性的数学上限，也不能把固定核/概率拟合/正则化的差异全部归因非线性。RBF−Linear另有预先声明的直接敏感性；树模型即使改变分数也不是识别到具体活动因果关系。XGBoost如缺依赖不安装，主族仍10。共同9指标、原始参数和耗时、完整paired表及最后患者cluster敏感性均保留；不因单seed或单最高值选模型。A2阶段归档后才进入B。
