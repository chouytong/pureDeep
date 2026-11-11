# 历史与真实forward审计

启动前读取服务器当前25文档及45mainline文件；GitHub origin/main与服务器mirror均为35b0a7c，源码逐文件一致。证据hash见audit/history_sources.csv、mainline_sources.csv、frozen45_integrity.csv。full-text扫描历史文件，展示摘录抑制旧outer数值；未重新读取旧outer predictions或信号。

| experiment | historical_status | source_relative_to_deep_final | constraint_or_qualification |
| --- | --- | --- | --- |
| V8-GN | HISTORICAL_BACKBONE_SUPERSEDED | mainline/README.md | 不重启V8 recipe搜索 |
| STR-01 | RETAINED_MATCHED_PURE_DEEP_BASELINE | artifacts/phase2_str_training_20260926/PHASE2_REPORT.md | 原recipe精确重现；不是当前WSSL模型 |
| WSR-01 | REJECT | artifacts/wsr01_wrist_structured_residual_20260922/WSR01_REPORT.md | 可恢复性改善未转换为稳定分类收益 |
| Recipe / Focal / Label smoothing / sampling | 0_OF_16_RETAINED | artifacts/phase2_str_training_20260926/PHASE2_REPORT.md | STR阶段结果；不声称证明所有WSSL recipe无效 |
| SAM and V8-only recipe variants | TASK_CONSTRAINT_NOT_TESTED_IN_PHASE8 | configs/USER_REQUEST.md | 任务列为不重复范围；本次Phase2报告未证实SAM执行，不能虚构其性能负结果；V8 recipe不能代替WSSL对照 |
| Frozen HarNet10 WSSL-STR | OFFICIAL_RETAINED | artifacts/phase3b_external_ssl_20260928/PHASE3B_REPORT.md | 唯一正式最佳single-model；概率集成历史已出现 |
| HarNet Layer5 / multi-layer adaptation | NOT_RETAINED | artifacts/phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md | AUROC改善不足以通过联合目标 |
| Ordinary deep-OOF threshold | CLOSED_DIAGNOSTIC | artifacts/phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md | 已有135模型OOF；不再搜阈值 |
| Output adapter / learned gate / masked domain adaptation | REJECT | artifacts/phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md | 不改名重开固定失败设计 |
| Activity gate E3 | NOT_RUN_GATE_FAIL | artifacts/phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md | 条件未运行与性能负结果分开 |
| HarNet5 / Bio-PM bounded export | NOT_RETAINED | artifacts/phase3e_external_priors_20261001/PHASE3E_REPORT.md | 不扩大为全家族否定 |
| Window mean/delta / Raw Acc / DD auxiliary | NOT_RETAINED | artifacts/phase4_window_dynamics_20261002/FINAL_REPORT.md | 不新增此类分支 |
| WSSL EMA | NOT_RETAINED | artifacts/wssl_ema_20261002/FINAL_REPORT.md | 不重复 |
| Frequency prior STR | NOT_RETAINED | artifacts/frequency_prior_str_20261007/FINAL_REPORT.md | 不重复 |
| Activity scalar scaling | INCONCLUSIVE | artifacts/wssl_activity_scaling_20261008/FINAL_REPORT.md | 不误标为已证实有害 |
| Patch STR P1/P2 | REJECT | artifacts/patch_str_20261008/PATCH_STR_REPORT.md | P3未获自动继续授权 |
| Dual-reference fusion | REJECT | artifacts/wssl_complementarity_phase5_20261111/FINAL_REPORT.md | BA/F1/DD/稳定性未联合通过 |
| Fixed-denominator CE | NOT_RUN_GATE_FAIL | artifacts/wssl_reference_audit_20261111/NEGATIVE_RESULTS_DO_NOT_REPEAT.md | 不是已训练性能负结果 |
| Representation / independent cohort audit | AUDIT_ONLY_NO_NEW_RETAINED_MODEL | artifacts/independent_validation_phase7_20261111/FINAL_REPORT.md | 不把重复PADS患者称独立验证 |


原local wrist64加Frozen HarNet1024经LN+Linear→64；bilateral258是left/right/mean/absdiff四个64及2mask。11活动各自加ID、LN、独立MLP打分，softmax竞争后weighted pooling成subject258；Dropout+Linear→2。STR逐token258→16/GELU，固定activity顺序展平176→2；两路logit相加。最终读出是共同434D空间的仿射函数，但活动打分与STR前级仍非线性。原模型存在跨活动softmax权重竞争，不能称完全没有跨活动关系；缺的是pooling前内容依赖token-to-token mixing。B固定新增1层4head/64D模块，不声称全球创新。真实长度分组和截断先于wrist encoder，mask保持原语义。

归一化直接复用匹配当前inner-train的旧参数；预处理及45旧checkpoint未改。历史deep-OOF135模型仅核验inventory，不重训、不用于新阈值。全部新候选只属于development evidence。
