# 外部输入适配与合成correctness

状态PASS_SYNTHETIC_ONLY；真实外部adapter验收尚未完成。所有执行在服务器已有mfam_analysis环境，没有安装或升级依赖；3个程序生成、无临床身份的fixture，固定种子20261111。主线/原资产未修改。

## 已实现接口

external_adapter.py接收内存中的11活动→left/right→Stream结构，不扫描文件系统或读取外部个体文件。输入不含诊断标签；要求有限[T,6]、明确通道映射、独立设备/腕侧来源、已确认的原PADS轴向与wrist placement、规律时间戳和单位。未知映射、不规则时间、缺活动/腕/通道/单位拒绝；没有复制对侧或伪造任务。许可/重叠等真实clinical gate的布尔检查仅是应用接口，不能替代数据方核查和用户批准，不能称系统强制隔离。

确定性重命名、g与m/s²、rad/s与deg/s、ms与s转换，固定polyphase到100Hz。原local预处理trim48、Acc减固定L1趋势、Gyro原样和既有norm；external rawAcc按原clip/窗口/resample路径。两支共享实际活动/腕结构和真实长度；不足1000的edge padding仅为原HarNet合法窗口处理，不是补缺失活动。batch末端容器padding不改变真实长度。只接受调用者核实的raw device stream，预滤波/Euler角/特征表不具有等价性。

所有允许的单位/采样转换仍须记录源设备带宽、硬件滤波、轴向与安装差异；upsampling不会恢复丢失频带，ms→s不能修复时钟错位。代码当前不擅自插值不规则时钟或学习设备变换。源格式未知时不能凭合成内存结构宣称.mat/TXT真实解析器验收完成。

## 实际测试结果

| test | status |
| --- | --- |
| channel_order | PASS |
| unit_and_timestamp_conversion | PASS |
| anti_alias_resampling_50_to_100 | PASS |
| missing_activity | PASS_EXPECTED_REJECTION |
| missing_wrist | PASS_EXPECTED_REJECTION |
| wrong_shape | PASS_EXPECTED_REJECTION |
| unknown_unit | PASS_EXPECTED_REJECTION |
| wrong_frame | PASS_EXPECTED_REJECTION |
| wrong_placement | PASS_EXPECTED_REJECTION |
| irregular_time | PASS_EXPECTED_REJECTION |
| nonfinite | PASS_EXPECTED_REJECTION |
| duplicate_channels | PASS_EXPECTED_REJECTION |
| real_clinical_gate_closed | PASS_EXPECTED_REJECTION |
| adapter_input_unchanged_and_real_lengths | PASS |
| dimension_norm_padding_and_true_lengths | PASS |
| last_partial_encoder_and_classifier_batches | PASS |
| eval_mode | PASS |
| all_weights_and_BN_buffers_unchanged | PASS |
| deterministic_inference_bitwise | PASS |
| model_inputs_unchanged | PASS |
| finite_logits_and_probabilities | PASS |
| argmax_tie_PD | PASS |
| long_first_last_external_windows | PASS |
| padded_tail_ignored_by_model | PASS |
| actual_SHA_config_lock | PASS |


重复两次完整合成推理最大logit差=0.0（bitwise相同），两次的classifier batches均[2,1]，encoder末批每人22窗，满足64上限。另测试长记录首末双窗44的窗口映射；该长窗测试不冒称做过额外完整临床推理。padding perturbation对照两次forward一致，输入未变、logits有限、状态及BN未变。合成运行记录elapsed=1.099883s（不含启动资产哈希成本，未测峰值内存）。

这25项涵盖用户要求14类检查及多个拒绝路径，不是25个临床验证病例。所有真实clinical sensor读取0、外部下载0、新fit/optimizer更新0。可进一步验证实际file parser、放置/时间同步和真实设备信号，但须先取得具体数据授权与兼容性证据；本轮不实施。
