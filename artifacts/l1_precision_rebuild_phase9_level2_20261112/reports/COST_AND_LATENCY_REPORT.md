# 参数、CPU处理和GPU计时边界

STR分类器75524可训练参数；WSSL分类器143172可训练参数，外部HarNet10457408冻结参数，WSSL唯一部署总量10600580。Frozen不表示整个WSSL网络冻结。源码/实际模型state逐run核验。

| component | unit | observations | mean | median | p95 | maximum | throughput_items_per_CPU_second | peak_RSS_KiB | scope_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| axis_solver | CPU_seconds | 25740 | 0.00557661 | 0.00619071 | 0.00863506 | 0.0151133 | 179.32 | 98924 | measured user+system worker CPU; parent and QA counted separately in ledger |
| wrist_complete_export | CPU_seconds | 8580 | 0.0200974 | 0.0219826 | 0.0277846 | 0.0438917 | 49.7577 | 98924 | measured user+system worker CPU; parent and QA counted separately in ledger |
| subject_22wrist_complete_export | CPU_seconds | 390 | 0.442143 | 0.439193 | 0.495584 | 0.561252 | 2.26171 | 98924 | measured user+system worker CPU; parent and QA counted separately in ledger |
| str_classifier_forward_ms | classifier_forward_ms | 45 | 3.49569 | 3.49866 | 3.52004 | 3.53433 | NA | NA | batch8, warmup3, repeat10, synchronized; no IO/L1/normfit/HarNet encoding |
| str_subjects_per_second | subjects_per_second | 45 | 2288.61 | 2286.59 | 2318.15 | 2323.76 | NA | NA | batch8, warmup3, repeat10, synchronized; no IO/L1/normfit/HarNet encoding |
| str_peak_GPU_allocated_bytes | peak_GPU_allocated_bytes | 45 | 1.50294e+08 | 1.50294e+08 | 1.50294e+08 | 1.50294e+08 | NA | NA | batch8, warmup3, repeat10, synchronized; no IO/L1/normfit/HarNet encoding |
| wssl_classifier_forward_ms | classifier_forward_ms | 45 | 3.90275 | 3.90841 | 3.93089 | 3.95678 | NA | NA | batch8, warmup3, repeat10, synchronized; no IO/L1/normfit/HarNet encoding |
| wssl_subjects_per_second | subjects_per_second | 45 | 2049.92 | 2046.87 | 2075.29 | 2086.81 | NA | NA | batch8, warmup3, repeat10, synchronized; no IO/L1/normfit/HarNet encoding |
| wssl_peak_GPU_allocated_bytes | peak_GPU_allocated_bytes | 45 | 1.50835e+08 | 1.50835e+08 | 1.50835e+08 | 1.50835e+08 | NA | NA | batch8, warmup3, repeat10, synchronized; no IO/L1/normfit/HarNet encoding |
| train-wssl_total | CPU_seconds_parent_and_children | 1 | 2136.75 | NA | NA | NA | NA | NA | all attempts total user+system CPU including workers; cumulative wall_seconds=629.1755307391286 |
| train-str_total | CPU_seconds_parent_and_children | 1 | 2464.42 | NA | NA | NA | NA | NA | all attempts total user+system CPU including workers; cumulative wall_seconds=690.2612413410097 |
| fit-normalization_total | CPU_seconds_parent_and_children | 1 | 16.0948 | NA | NA | NA | NA | NA | all attempts total user+system CPU including workers; cumulative wall_seconds=13.515636823140085 |
| solver_QA_pilot_export_and_readbacks_total | CPU_seconds_parent_and_children | 1 | 214.953 | NA | NA | NA | NA | NA | all charged attempts; limit21600 seconds |


solver QA/pilot/导出/独立回读累计214.952797 CPU秒，上限21600，按进程与子进程user+system而非墙钟。累计明细和失败/重复尝试存私有ledger，各stage日志保留。每轴CPU仅求解；腕CPU包含raw读取/solve/证书压缩/写入；人CPU为22腕累加，parent调度和QA另在总账。RSS是worker进程连同依赖的峰值，不冒称纯单轴内存。

RTX5090Dv2，同float32/AMPfalse、batch8、warmup3/repeat10，CUDA同步，90个正式最佳模型在各自inner-train固定前8人的resident新归一化输入计时，模型state前后hash一致。WSSL采用冻结HarNet缓存。forward不含磁盘IO、L1、norm应用、HarNet原始编码或数据传输；吞吐为该边界。训练cost CSV中的peak GPU是该训练进程的累计高水位，统一模型forward计时则逐模型reset后测峰值，不能混为同一内存口径。**在线raw→预测端到端=NOT_MEASURED**，重新提取全部HarNet不在本轮授权范围，不把缓存forward当端到端。
