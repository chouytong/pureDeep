# 人口字段来源与混杂

实际patients JSON白名单age/gender/height/weight；age_at_diagnosis、disease_comment、药物、病程、问卷及设备不作为输入。所有390行四字段完整，年龄40–85年、身高152–203cm、体重45–181kg；gender仅原始male/female，不冒称核实的生物学sex。采集时自报问卷，与当次assessment关联；没有精确记录时间戳，不假定长期不变。

[官方数据字典](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)确认字段，[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770131/)Table3确认单位和自报来源。原字段和项目subset以本地文件哈希追踪，不将论文全样本数代替本地390人。

| field | group | units | n | mean | median | SD | IQR | q01 | q05 | q25 | q50 | q75 | q95 | q99 | category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | PD | years | 276 | 65.3587 | 67 | 9.63922 | 16 | 46 | 49 | 57 | 67 | 73 | 79 | 83.25 | NA |
| age | DD | years | 114 | 62.4298 | 63 | 11.5483 | 16 | 41 | 43 | 54 | 63 | 70 | 82 | 84.74 | NA |
| height | PD | cm | 276 | 175.71 | 176 | 9.77247 | 12.5 | 152.75 | 159.75 | 169.75 | 176 | 182.25 | 191 | 198 | NA |
| height | DD | cm | 114 | 173.044 | 173 | 8.73442 | 12 | 158 | 160 | 167 | 173 | 179 | 187 | 192.87 | NA |
| weight | PD | kg | 276 | 83.7754 | 84.5 | 18.0991 | 24.25 | 48.75 | 54 | 70.75 | 84.5 | 95 | 110 | 124.75 | NA |
| weight | DD | kg | 114 | 79.5439 | 77.5 | 17.8361 | 21.5 | 50.26 | 56 | 66.5 | 77.5 | 88 | 110.35 | 130.87 | NA |
| gender | PD | NA | 81 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | female |
| gender | PD | NA | 195 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | male |
| gender | DD | NA | 57 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | female |
| gender | DD | NA | 57 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | male |
| device | PD | NA | 93 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Apple Watch Series 3 |
| device | DD | NA | 11 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Apple Watch Series 3 |
| device | PD | NA | 183 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Apple Watch Series 4 |
| device | DD | NA | 103 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | Apple Watch Series 4 |
| age | Other Movement Disorders | NA | 60 | 61.3167 | 60 | 12.3226 | 17.5 | 41 | 42 | 51.75 | 60 | 69.25 | 82 | 83.82 | NA |
| height | Other Movement Disorders | NA | 60 | 173.067 | 173 | 7.6377 | 10 | 160 | 161.95 | 168 | 173 | 178 | 186.2 | 191.23 | NA |
| weight | Other Movement Disorders | NA | 60 | 76.5167 | 76 | 15.4476 | 18.75 | 51.18 | 58.85 | 64.75 | 76 | 83.5 | 104 | 118.61 | NA |
| age | Essential Tremor | NA | 28 | 66.1786 | 64 | 10.8391 | 13 | 46.08 | 49.35 | 60.5 | 64 | 73.5 | 82 | 84.19 | NA |
| height | Essential Tremor | NA | 28 | 173.357 | 172.5 | 9.60407 | 17.25 | 156.62 | 161.35 | 165 | 172.5 | 182.25 | 186.65 | 187 | NA |
| weight | Essential Tremor | NA | 28 | 84.6786 | 83 | 21.8768 | 28 | 49.35 | 54.05 | 69.5 | 83 | 97.5 | 123.35 | 137.3 | NA |
| age | Atypical Parkinsonism | NA | 15 | 66 | 66 | 7.96421 | 9.5 | 54 | 54 | 61 | 66 | 70.5 | 77.6 | 78.72 | NA |
| height | Atypical Parkinsonism | NA | 15 | 168.467 | 169 | 8.09644 | 14 | 158 | 158 | 161 | 169 | 175 | 180 | 180 | NA |
| weight | Atypical Parkinsonism | NA | 15 | 79.1333 | 79 | 18.6351 | 20.5 | 55.7 | 58.5 | 66.5 | 79 | 87 | 113.6 | 120.32 | NA |
| age | Multiple Sclerosis | NA | 11 | 54.0909 | 55 | 7.91776 | 9.5 | 40.3 | 41.5 | 50 | 55 | 59.5 | 63.5 | 66.3 | NA |
| height | Multiple Sclerosis | NA | 11 | 178.364 | 182 | 10.689 | 10.5 | 159 | 163 | 172.5 | 182 | 183 | 193.5 | 194.7 | NA |
| weight | Multiple Sclerosis | NA | 11 | 83.5455 | 85 | 16.1825 | 15.5 | 59.3 | 64.5 | 73 | 85 | 88.5 | 110 | 114 | NA |


类别/设备关联：

| field | chi2 | p_value | cramers_V | minimum_expected | interpretation |
| --- | --- | --- | --- | --- | --- |
| gender | 14.1601 | 0.000167895 | 0.190546 | 40.3385 | COHORT_ASSOCIATION_NOT_DISEASE_CAUSALITY |
| device | 22.6414 | 1.95228e-06 | 0.240946 | 30.4 | COHORT_ASSOCIATION_NOT_DISEASE_CAUSALITY |


年龄/性别可能反映队列构成；关联不是疾病因果，也不意味着跨中心可迁移。JSON完整解析后只保留白名单；不声称物理零读取其他字段。设备标记与论文统一Series4描述的差异按实际原始标记报告，不改写。

固定15个重叠development splits；同人重复出现。三seed不是独立患者队列。人口记录和逐人预测不公开，正式阈值/模型指针不改。历史outer文字接触和多轮development选择偏差保留；本轮未读取outer性能。
