# Parameters, training and token-readout timing

| seed | context | inner | best_epoch | training_epochs | optimizer_steps | training_validation_wall_seconds | total_classifier_parameters | new_trainable_parameters | original_frozen_classifier_parameters | frozen_HarNet_parameters | unique_deployment_parameters | base_state_unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0 | 0 | 17 | 29 | 754 | 4.048085 | 210694 | 67522 | 143172 | 10457408 | 10668102 | True |
| 43 | 0 | 0 | 11 | 23 | 598 | 2.988992 | 210694 | 67522 | 143172 | 10457408 | 10668102 | True |
| 44 | 0 | 0 | 9 | 21 | 546 | 2.702103 | 210694 | 67522 | 143172 | 10457408 | 10668102 | True |


原WSSL唯一部署10600580=143172 classifier+10457408 HarNet；AI部署10668102=210694 classifier侧(67522 trainable、143172 frozen)+10457408外部frozen。不是仅报告新增参数。

| seed | condition | batch_size | warmup | repeats | batch_ms | amortized_ms_per_subject | subjects_per_second | peak_GPU_allocated_bytes | train_only | resident_tokens_float32 | not_end_to_end | not_standalone_patient_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | baseline_readout | 8 | 3 | 10 | 0.438345 | 0.054793 | 18250.447749 | 35761664 | True | True | True | True |
| 42 | baseline_plus_interaction | 8 | 3 | 10 | 0.991824 | 0.123978 | 8065.950296 | 35906560 | True | True | True | True |
| 43 | baseline_readout | 8 | 3 | 10 | 0.424673 | 0.053084 | 18838.030286 | 35761664 | True | True | True | True |
| 43 | baseline_plus_interaction | 8 | 3 | 10 | 0.996313 | 0.124539 | 8029.602436 | 35906560 | True | True | True | True |
| 44 | baseline_readout | 8 | 3 | 10 | 0.424711 | 0.053089 | 18836.344888 | 35761664 | True | True | True | True |
| 44 | baseline_plus_interaction | 8 | 3 | 10 | 0.989466 | 0.123683 | 8085.168621 | 35906560 | True | True | True | True |


同inner-train前8人、相同resident token/mask/float32与硬件，CUDA同步，warmup3/repeat10。两系统均驻留内存，原head与原head+AI以同一基础权重比较；参数及buffer指纹不变。batch/8是摊销，不是batch1独立时延。排除原编码/L1/norm/HarNet/IO；在线raw端到端未测。训练耗时含validation及保存，CPU阶段记账见stage_compute_summary.csv。token_export_summary.csv记录6包/930 full-signal subject forward，train205+validation105各三seed，共10230 activity token；每实例另做一次head重建，无深度训练。
