# D0人口统计Logistic Regression

固定四字段、C1、balanced class weights、lbfgs、train-only imputation/scaler，15确定性fit。没有validation fit、特征选择或missingness输入；没有复制成45独立fit。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
| --- | --- | --- | --- | --- | --- | --- |
| D0-DEMOGRAPHIC-ONLY | 0.602816 | 0.574716 | 0.60788 | 0.560924 | 0.642318 | 0.507114 |


D0是诊断baseline，不是正式候选；人口字段可信可用，10C可按原独立分支协议执行。

固定15个重叠development splits；同人重复出现。三seed不是独立患者队列。人口记录和逐人预测不公开，正式阈值/模型指针不改。历史outer文字接触和多轮development选择偏差保留；本轮未读取outer性能。
