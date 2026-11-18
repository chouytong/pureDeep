# Phase10 参数、成本与部署

GPU：NVIDIA GeForce RTX 5090 D v2, 24455 MiB。既有Python/Torch环境，无软件安装或权重下载。

| condition | total_parameters | trainable_parameters | frozen_parameters |
| --- | --- | --- | --- |
| WSSL-CERTIFIED-L1 | 10600580 | 143172 | 10457408 |
| FT-L1 | 10600580 | 2764612 | 7835968 |
| DEMO | 10600796 | 216 | 10600580 |
| DD | 10600580 | 143172 | 10457408 |


正式训练实测（每run为50上限、patience12、best BA；没有用推算45run替代实测）：

| condition | seed | context | inner | total_parameters | trainable_parameters | frozen_parameters | epochs | best_epoch | optimizer_steps | wall_seconds | parent_CPU_seconds | peak_GPU_allocated_bytes | cached_path_validation_inference_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT-L1 | 42 | 0 | 0 | 10600580 | 2764612 | 7835968 | 18 | 6 | 468 | 13.9523 | 14.2136 | 1020548608 | 0.160159 |
| FT-L1 | 43 | 0 | 0 | 10600580 | 2764612 | 7835968 | 17 | 5 | 442 | 12.2957 | 12.6352 | 1020548608 | 0.148371 |
| FT-L1 | 44 | 0 | 0 | 10600580 | 2764612 | 7835968 | 15 | 3 | 390 | 10.5983 | 11.0495 | 1020548608 | 0.150823 |
| DEMO | 42 | 0 | 0 | 10600796 | 216 | 10600580 | 13 | 1 | 338 | 3.31304 | 3.18206 | 69372928 | 0.0155434 |
| DEMO | 43 | 0 | 0 | 10600796 | 216 | 10600580 | 15 | 3 | 390 | 2.65457 | 2.33684 | 69372928 | 0.0154347 |
| DEMO | 44 | 0 | 0 | 10600796 | 216 | 10600580 | 13 | 1 | 338 | 2.64009 | 2.18846 | 69372928 | 0.0148781 |
| DD | 42 | 0 | 0 | 10600580 | 143172 | 10457408 | 26 | 14 | 676 | 18.2878 | 17.8103 | 946549248 | 0.149167 |
| DD | 43 | 0 | 0 | 10600580 | 143172 | 10457408 | 46 | 34 | 1196 | 29.8159 | 28.6698 | 946549248 | 0.15756 |
| DD | 44 | 0 | 0 | 10600580 | 143172 | 10457408 | 26 | 14 | 676 | 16.9497 | 16.4425 | 946549248 | 0.154539 |


同一固定8人、1warmup+3次在线推理实测：

| condition | subjects | repetitions | online_seconds_mean | online_seconds_SD | seconds_per_subject | peak_allocated_bytes | max_probability_drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DD | 8 | 3 | 0.151565 | 0.000189222 | 0.0189456 | 198880768 | 3.77581e-06 |
| DEMO | 8 | 3 | 0.15195 | 0.000447775 | 0.0189938 | 198775808 | 2.26498e-06 |
| FT-L1 | 8 | 3 | 0.15211 | 0.000468289 | 0.0190137 | 241591296 | 0.000129133 |
| WSSL-CERTIFIED-L1 | 8 | 3 | 0.152355 | 0.00170065 | 0.0190444 | 195402752 | 2.26498e-06 |


训练wall包含各run初始化/加载和正式循环；DEMO还包含新导出Certified frozen train/validation logits的成本。parent_CPU_seconds仅主进程CPU，不含DataLoader worker CPU总量。cached_path_validation_inference_seconds只计缓存后的validation路径，不能代表完整部署延迟；冻结HarNet训练不在GPU计算完整encoder，峰值也不代表在线总资源。

在线benchmark包含固定8人、11活动、双腕的原Raw Acc文件读取、trim48/clip±3g/first-last1000/3:10 resampling、已有Certified STR文件读取/原norm/transform、CPU→GPU、HarNet及classifier前向、概率回CPU。模型和官方权重预加载不计，原Certified detrending生产阶段不计；磁盘受系统缓存影响，不是穿戴设备实时功耗或单人p95。canonical缓存的全局batch布局与本次局部8人重算HarNet batch不同，GPU算术可有微小漂移，标量最大差公开；本次只计时，不替换正式预测、不重新择epoch、不用漂移结果改善指标。

FT-L1总参数包含完整官方HarNet，即使训练只缓存冻结prefix；DEMO/DD加一次10,457,408参数encoder，不能重复计入classifier。DEMO只有216新增可训练参数，部署仍需原双腕IMU和四字段；所有候选均拒绝扩展，无新模型替换当前指针。推理脚本与checkpoint/optimizer仅用于重现研究，不把私有权重上传GitHub。

