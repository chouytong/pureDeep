# DD粗来源类错误

来源严格用实际condition四类，不读取diagnosis comment进行细分类。每run n<10性能NA；全人固定概率共识诊断单列Wilson区间，不等于逐run Recall。

| source_category | validation_count | recall | error_rate | ci_low | ci_high | mean_signed_margin |
| --- | --- | --- | --- | --- | --- | --- |
| Atypical Parkinsonism | 15 | 0.6 | 0.4 | 0.357468 | 0.801755 | 0.0328156 |
| Essential Tremor | 28 | 0.785714 | 0.214286 | 0.604614 | 0.897875 | 1.15511 |
| Multiple Sclerosis | 11 | 0.636364 | 0.363636 | 0.353801 | 0.848335 | 0.391396 |
| Other Movement Disorders | 60 | 0.483333 | 0.516667 | 0.36175 | 0.606922 | 0.253582 |


DD错误来源集中度与D启动检查：

| check | passed |
| --- | --- |
| dd_recall_lower_in_at_least10 | True |
| dd_signed_margin_lower_in_at_least10 | True |
| errors_in_at_least3_source_groups | True |
| not_single_small_source_concentration | True |
| certified_finite_complete_inputs | True |


固定15个重叠development splits；同人重复出现。三seed不是独立患者队列。人口记录和逐人预测不公开，正式阈值/模型指针不改。历史outer文字接触和多轮development选择偏差保留；本轮未读取outer性能。
