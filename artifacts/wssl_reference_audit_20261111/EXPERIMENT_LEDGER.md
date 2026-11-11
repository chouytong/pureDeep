# 历史实验总账

初版只数 artifact/runs，遗漏 foundation_validation/outputs，已按真实 inner stage 元数据补齐。初版保留于私有 restricted_manifests。来源路径/hash 见 CSV、history_sources 和 foundation_inner_stage_inventory。阶段日期无法核实时写 unknown。

| experiment_id | expected_runs | actual_completed_inner_stages | status | conclusion |
| --- | --- | --- | --- | --- |
| V8 | 45 | 45 | retained | retained historical baseline; not current |
| V8-GN | 45 | 45 | retained | retained historical reference |
| STR-01 | 45 | 45 | retained | retained matched baseline |
| WSR-01 | 45 | 45 | rejected | rejected |
| prototype | 90 | 90 | rejected | rejected; not identical to frozen reference classifier |
| strong-backbone | 135 | 135 | rejected | rejected / not retained |
| DSG-01 | 0 | 0 | completed-diagnostic; conditional neural candidate not-run | completed; conditional consistency training not-run |
| RGD-01 | 0 | 0 | completed-diagnostic; conditional neural candidate not-run | completed; new neural training not-run |
| PRR-01 | 0 | 0 | completed-diagnostic; conditional neural candidate not-run | completed; new readout training not-run |
| PAG-01 | 0 | 0 | completed-diagnostic; conditional neural candidate not-run | completed; new training gate failed |
| phase2 | 765 | 765 | rejected | all16 alternatives not retained |
| phase3a | 180 | 180 | rejected | CASE D; no candidate retained |
| phase3b | 135 | 135 | retained | b_str_pretrained retained; other controls not retained |
| phase3c | 360 | 360 | completed | CASE D; retain frozen WSSL; threshold diagnostic already done |
| phase3d | 135 | 135 | rejected | none retained; conditional activity gate not-run |
| phase3e | 180 | 180 | rejected | none retained; dual prior not-run |
| phase4 | 180 | 180 | rejected | all four fixed configurations rejected |
| wssl-ema | 90 | 90 | rejected | EMA not retained |
| wssl-review | 45 | 45 | rejected | rejected; original LR retained |
| frequency | 90 | 90 | rejected | rejected |
| wssl-contribution | 0 | 0 | completed | completed; phase B qualification only, no phase C loss audit |
| activity-scaling | 45 | 45 | inconclusive | INCONCLUSIVE / DO NOT RETAIN; phase C not started |
| patch-str | 90 | 90 | rejected | REJECT; P3 not-run |
| patch-recheck | 0 | 0 | completed | reused exact historical experiment; no new training |
| phase3c-deep-OOF | 135 | 135 | completed-diagnostic | 135 actual checkpoints and prediction sets reused |


计数口径：prototype=两种 loss×45=90；strong-backbone=ResNet45+InceptionTime45+GyroFFT45=135。Phase2=baseline重训45+16候选×45=765。Phase3C=360监督训练+135独立 deep-OOF，p100 frozen/STR复用。EMA表内90是ordinary45+EMA45模型评估状态，但只有45条optimizer训练轨迹，EMA同一次训练携带shadow，不是90次独立训练。Phase3D表内135监督run；train-only masked pretext另有历史报告记录45，未把它混入监督run，也未重新加载其权重。smoke不计入正式run。所有这些均非本轮新训练。

Phase3D条件activity gate当时NOT_RUN；后续11-scalar activity scaling确已做45次，INCONCLUSIVE/DO NOT RETAIN，不能改写有害。Patch P1/P2在10月8日训练，11月11日仅复核。Phase C当时未启动独立loss数值审计；本轮才补做。Full HarNet fine-tuning、Patch P3、双prior均不因未做而自动获准。prototype alignment训练loss与当前冻结参考距离分类不是同一实验。
