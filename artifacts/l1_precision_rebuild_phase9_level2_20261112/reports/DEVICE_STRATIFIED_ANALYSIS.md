# 设备标记描述性分层

组间变化只作描述，未开展可识别硬件因果效应或正式异质性检验。各组方向、幅度和不确定性须共同读取下表，不能只挑有利组。

Series3 PD93/DD11；Series4 PD183/DD103，为metadata标记而非硬件日志。每人先平均全部合法query appearance及seed的correctness，再按设备×类别汇总recall及10000患者bootstrap不确定性；这个辅助estimand与主表15split等权不同。样本量为唯一患者，DD Series3仅11，区间较宽。split内缺少该类记NA，细分split表只私有，不公开可识别的小格。

| device | label | model | unique_subjects | legacy_recall | legacy_ci_low | legacy_ci_high | certified_recall | certified_ci_low | certified_ci_high | delta | delta_ci_low | delta_ci_high | estimand | draws | split_empty_class_groups | causal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple Watch Series 3 | DD | STR | 11 | 0.560606 | 0.340909 | 0.780303 | 0.522727 | 0.318182 | 0.727273 | -0.0378788 | -0.0984848 | 0.0227273 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |
| Apple Watch Series 3 | DD | WSSL | 11 | 0.643939 | 0.462121 | 0.810606 | 0.628788 | 0.439394 | 0.810606 | -0.0151515 | -0.0606061 | 0.030303 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |
| Apple Watch Series 3 | PD | STR | 93 | 0.811828 | 0.758961 | 0.862007 | 0.77957 | 0.726703 | 0.830645 | -0.0322581 | -0.0573477 | -0.00806452 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |
| Apple Watch Series 3 | PD | WSSL | 93 | 0.820789 | 0.763441 | 0.875448 | 0.801971 | 0.748208 | 0.853047 | -0.0188172 | -0.0385529 | 0.000896057 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |
| Apple Watch Series 4 | DD | STR | 103 | 0.622168 | 0.56068 | 0.683657 | 0.62055 | 0.56068 | 0.68123 | -0.00161812 | -0.0218447 | 0.0202265 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |
| Apple Watch Series 4 | DD | WSSL | 103 | 0.657767 | 0.590595 | 0.723301 | 0.666667 | 0.601133 | 0.729794 | 0.00889968 | -0.0105178 | 0.0283172 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |
| Apple Watch Series 4 | PD | STR | 183 | 0.760474 | 0.721311 | 0.797359 | 0.752732 | 0.714481 | 0.789162 | -0.00774135 | -0.0245902 | 0.00956284 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |
| Apple Watch Series 4 | PD | WSSL | 183 | 0.76275 | 0.721755 | 0.802368 | 0.759563 | 0.719945 | 0.797814 | -0.00318761 | -0.0154827 | 0.00910747 | patient mean of correctness over repeated legal validation appearances and seeds | 10000 | 0 | False |


新旧残差RMS、trend RMS、差异、边缘/内部差异与预固定频带0–3/3–12/12–15/15–Nyquist能量见device_signal_change_summary.csv；先双腕/活动均值再患者组分布，所有组最少11人。频率用原名义100Hz索引描述，未修补timestamp。差异不等于病理信息增益；没有设备输入、删组、lambda/阈值调整，也不解释成硬件必然因果。
