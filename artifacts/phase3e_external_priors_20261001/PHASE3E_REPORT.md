# Phase-3E: External Wearable SSL Prior Diversification

Status: complete and audited, 2026-10-01 Asia/Shanghai. Development only; PURE-DEEP.

## Scope and frozen protocol
This user-authorized phase tests different external wearable SSL priors after Phase-3D HarNet10 adaptation failed.15 fixed subject-level development splits (5 contexts x3inner) x seeds42/43/44; original STR architecture, bilateral/activity order/structured path, balanced CE, AdamW LR2e-4 WD1e-4 batch8, cosine50/min1e-6, max50 BA early stopping patience12, train-only normalization, DD probability>0.5. PHASE3E_PROTOCOL.md and official asset/source SHA records preceded formal candidate evaluation. No H1, handcrafted-feature branch, teacher/student, outer performance, STR/loss/sampling/augmentation search, threshold/weight tuning. Fold-local manifests limit each run to innertrain/validation. FOE-01 remains sealed and does not independently validate this phase. Old formal outputs and production forward unchanged.

E0 fixes0.5/0.5 mixing of frozen and lastblock probabilities. E1 tests official HarNet5 with fixed first/last5s windows. E2 tests official default50mr BioPM neural acc encoder only, standalone feasibility and unchanged STR64D wrist residual. E3 random/E4 complementarity/E5 dual are conditional under the frozen gate.

## Official implementation and limits
HarNet5 official source commit150550ea5d41800229c95e36f88f5bf0d2e7cf04; checkpoint SHA74ffaefbafb467b7253d121ef71fe734ee03bc802ba33c33a6aa1aef7f91de5e. InputNx3x150 at30Hz. Raw PADS g Acc, trim48,clip+-3g,resample_poly3/10, exactly first500/last500 raw samples per activity/wrist,mean features.17160contexts. Official final512D/4,227,904encoder parameters, versus10s1024D/10,457,408. Initial1024 assumption was corrected after official forward and before formal training. Only projection input changes;64D residual/downstream unchanged. This official-family comparison cannot isolate duration from capacity/window coverage.

BioPM official commit41979e6c36decbd794e8e79550d98796efc47bdc (2026-08-06),50mr SHA d7c37cf5d54c7249ef97a0b162bef631a7e440836e70db3c13dd604536c01312, strict complete acc-encoder loading,1,418,464parameters. Default1023 includes384 pooled neural-token dims and639 unencoded gravity waveform dims. To honor pure-deep, use official384D per-axis mean/std of neural tokens only; exclude raw gravity and optional gravityCNN (not pretrained in release). This acc-encoder-only result cannot falsify complete BioPM HAR performance.

Official downstream preprocessing: linear100to30Hz, sixth-order0.5-12Hz bandpass, spline zero-crossings, official50ms separation/low-amplitude merge,32sample movement patches, fractional position/axis/duration metadata,192token cap. Input[B,192,32],position[B,192],metadata[B,192,5];unmasked64D neural tokens,official384D pooling. Raw g Acc only,trim48;filter full record then fixed10s first/last contexts;short contexts edge padded. No raw gyro/gravity feature input.10920contexts,6240padded,0empty,3-192tokens,8.14%at token cap. No label-based context selection or token-cap adjustment. Frozen extraction has no data-fitted transformation; fitting/normalization remain train-only and cache retrieval uses subject IDs within active innertrain/validation.

Server DNS blocked direct GitHub access; official sources/weights downloaded locally and transferred. Missing h5py3.11 installed offline in this phase vendor/deps only,original environments unchanged. No skill/plugin installed. Official sources: https://github.com/Prithvitarale/biopm and https://github.com/OxWearables/ssl-wearables .

## Smoke, logits consistency and integrity
Initial STR+HarNet5 and STR+BioPM logits exactly equal original STR (zero residual),maxdifference0. Loading a frozen WSSL checkpoint into the generalized1024 interface gives maxdifference0 against original Phase3B at identical input/features. HarNet5/BioPM-only/BioPM-residual/random-control one-batch training/checkpoint smoke passed. E3 control clarification preceded complete BioPM residual statistics and broadens only the random-control trigger to any positive primary mean; retention unchanged. Strict encoder weights,finite cached features,complete subject/activity/wrist mapping passed. ANALYSIS_IMPLEMENTATION_NOTE.md records exclusion of incomplete candidates from aggregation/gates; no action used partial-seed statistics. Full-candidate formulas/training/protocol unchanged.

FINAL_AUDIT.json:

```json
{
  "status": "PASS",
  "new_completed_runs": 180,
  "variant_counts": {
    "harnet5": 45,
    "biopm_only": 45,
    "biopm": 45,
    "biopm_random": 45
  },
  "fixed_split_sha256": "b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e",
  "subject_id_label_alignment": true,
  "train_only_normalization_matches": true,
  "no_outer_loader_or_performance": true,
  "protocol_and_source_hashes_match": true,
  "zero_initialized_residual_projection_learned_all_residual_runs": true,
  "ssl_only_projection_nonzero_at_checkpoint": true
}
```

## Complete seed-first development metrics
Three seeds averaged within split,then15splits. Within-seed SD is the mean15split three-seed sample SD. All six metrics,global-seed SD,split SD,loss/early-epoch summaries are in CSV.

| variant | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | ba_within_seed_sd | auroc_within_seed_sd | score_spearman | threshold_disagreement |
|---|---|---|---|---|---|---|---|---|---|---|
| biopm | 0.7238 | 0.6924 | 0.7181 | 0.6806 | 0.7680 | 0.6168 | 0.0212 | 0.0341 | 0.6022 | 0.2344 |
| biopm_only | 0.6963 | 0.6438 | 0.6659 | 0.6347 | 0.7705 | 0.5171 | 0.0284 | 0.0314 | 0.7410 | 0.1943 |
| biopm_random | 0.7316 | 0.6932 | 0.7205 | 0.6846 | 0.7855 | 0.6009 | 0.0289 | 0.0330 | 0.6185 | 0.2201 |
| frozen | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 | 0.0216 | 0.0217 | 0.7734 | 0.1780 |
| harnet5 | 0.7236 | 0.6967 | 0.7238 | 0.6811 | 0.7612 | 0.6323 | 0.0349 | 0.0290 | 0.6016 | 0.2556 |
| lastblock | 0.7518 | 0.7244 | 0.7689 | 0.7128 | 0.7905 | 0.6583 | 0.0202 | 0.0222 | 0.7940 | 0.1786 |
| str | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | 0.0300 | 0.0323 | 0.6000 | 0.2372 |

## Paired primary differences
10000 paired15split bootstrap,paired dz,Wilcoxon. Planned primary family10 tests,unexecuted tests conservatively p1;secondary family20. BioPM-only feasibility comparison descriptive,not retention primary.

| candidate | reference | metric | mean_delta | positive_splits | ci95_low | ci95_high | cohen_dz | q_primary_bh |
|---|---|---|---|---|---|---|---|---|
| harnet5 | frozen | ba | -0.0229 | 4 | -0.0377 | -0.0077 | -0.7410 | 0.0754 |
| harnet5 | frozen | auroc | -0.0358 | 1 | -0.0525 | -0.0186 | -1.0454 | 0.0062 |
| biopm_only | str | ba | -0.0534 | 2 | -0.0736 | -0.0325 | -1.2518 | nan |
| biopm_only | str | auroc | -0.0517 | 2 | -0.0769 | -0.0268 | -1.0022 | nan |
| biopm | str | ba | -0.0048 | 5 | -0.0099 | 0.0004 | -0.4515 | 0.1892 |
| biopm | str | auroc | 0.0005 | 8 | -0.0064 | 0.0072 | 0.0347 | 1.0000 |
| biopm | frozen | ba | -0.0272 | 1 | -0.0396 | -0.0155 | -1.1204 | 0.0062 |
| biopm | frozen | auroc | -0.0414 | 1 | -0.0551 | -0.0279 | -1.4973 | 0.0012 |
| biopm | biopm_random | ba | -0.0008 | 7 | -0.0062 | 0.0053 | -0.0682 | 1.0000 |
| biopm | biopm_random | auroc | -0.0024 | 8 | -0.0109 | 0.0062 | -0.1353 | 1.0000 |

Retention gate ledger:

```json
{
  "harnet5": {
    "ba_stable": false,
    "auroc_stable": false,
    "macro_f1_non_decrease": false,
    "dd_drop_at_most_001": false,
    "pd_drop_at_most_001": false,
    "ba_seed_sd_not_materially_worse": false,
    "auroc_seed_sd_not_materially_worse": false,
    "retain": false
  },
  "biopm": {
    "ba_stable": false,
    "auroc_stable": false,
    "macro_f1_non_decrease": false,
    "dd_drop_at_most_001": false,
    "pd_drop_at_most_001": false,
    "ba_seed_sd_not_materially_worse": true,
    "auroc_seed_sd_not_materially_worse": false,
    "retain": false
  },
  "execute_random_control": true,
  "biopm_valuable_vs_str": false
}
```

## E0 fixed inference references
Single models use seed-first metric averages;3/6model ensembles evaluate seed-averaged probabilities, a different estimand. Both per-seed equal mix and6model ensemble use exactly one0.5/0.5 rule,no weight search.

| model | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall |
|---|---|---|---|---|---|---|
| equal_6model_ensemble | 0.7541 | 0.7150 | 0.7870 | 0.7099 | 0.8090 | 0.6210 |
| equal_mix_per_seed | 0.7463 | 0.7145 | 0.7684 | 0.7048 | 0.7911 | 0.6380 |
| frozen_3seed_ensemble | 0.7573 | 0.7200 | 0.7816 | 0.7141 | 0.8100 | 0.6301 |
| frozen_single | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 |
| lastblock_3seed_ensemble | 0.7567 | 0.7176 | 0.7854 | 0.7125 | 0.8117 | 0.6235 |
| lastblock_single | 0.7518 | 0.7244 | 0.7689 | 0.7128 | 0.7905 | 0.6583 |

Frozen/lastblock mean Spearman0.9312,threshold disagreement0.1061;frozen-only/lastblock-only correct counts5.20/5.84 per validation unit (descriptive).

| metric | delta | wins | ci_low | ci_high | dz | bh_q |
|---|---|---|---|---|---|---|
| accuracy | -0.0033 | 5 | -0.0149 | 0.0083 | -0.1370 | 0.8319 |
| ba | -0.0050 | 4 | -0.0164 | 0.0070 | -0.2079 | 0.8319 |
| auroc | 0.0054 | 14 | 0.0028 | 0.0074 | 1.1267 | 0.0427 |
| macro_f1 | -0.0043 | 5 | -0.0162 | 0.0073 | -0.1766 | 0.8319 |
| pd_recall | -0.0010 | 6 | -0.0192 | 0.0206 | -0.0244 | 0.8659 |
| dd_recall | -0.0090 | 5 | -0.0401 | 0.0173 | -0.1564 | 0.8761 |

Six-model equal mix raises AUROC over frozen3seed inference but does not jointly improve BA/MacroF1/DD Recall. Inference reference only,no calibration or single-model retention.

## Parameter counts
Frozen encoders included in total,excluded from classifier trainable count. Conditional random capacity row does not imply execution.

| model | frozen_encoder_parameters | trainable_classifier_parameters | total_parameters |
|---|---|---|---|
| STR-01 | 0 | 75524 | 75524 |
| HarNet10 WSSL | 10457408 | 143172 | 10600580 |
| HarNet5 WSSL | 4227904 | 109380 | 4337284 |
| BioPM-only acc | 1418464 | 29906 | 1448370 |
| BioPM WSSL acc | 1418464 | 100932 | 1519396 |
| Random BioPM WSSL acc (conditional) | 1418464 | 100932 | 1519396 |

## Statistical/evidence limits
The15fixed splits reuse/overlap subjects. Split-bootstrap/Wilcoxon/BH describe repeated-development variability,not independent cohort replication. Seeds,subjects,activities,windows,tokens and PD-DDpairs are not statistical units. Sequential-phase selection optimism remains;no Phase3E method has new independent outer validation. HarNet5 differs in official capacity/window coverage as well as duration. BioPM is restricted to pure-neural acc output and fixed integration/classifier recipe. Neither failure establishes global HarNet10 optimality,refutes full BioPM HAR pipeline,or stops every external SSL prior. A single fixed random encoder realization,if used,also limits control generality. OldFOE01 must not be reused for tuning or validation.

## Reproduction ledger
PHASE3E_PROTOCOL.md,OFFICIAL_ASSET_AUDIT.md,manifest.json;official vendor source/checkpoints;formal source SHA ledgers;independent scripts;feature extraction audits/caches;smoke logs/consistency.json;new runs split/seed artifacts;analysis model/e0/e4 CSV/JSON;FINAL_AUDIT.json/final_integrity.csv. No Phase1-3D formal output overwritten.

## Final answers and phase decision

1. **5s prior 没有优于 10s prior。** HarNet5 BA 0.6967、AUROC 0.7238；相对 HarNet10 分别 −0.0229 / −0.0358，只有 4/15 / 1/15 splits 改善，seed 波动增大。停止本次 temporal-scale 路线。官方 family 的维度和参数量也不同，不能将差异仅归因于时间尺度。
2. **本轮 Bio-PM acc-only 未建立有用的增量迁移收益。** BioPM-only BA 0.6438 / AUROC 0.6659；STR+BioPM 为 0.6924 / 0.7181。相对原 STR 的 BA −0.0048，AUROC +0.0005（95% CI [−0.0064,+0.0072]，BH q=1.0）；相对 HarNet10 的两项指标均明显较低。本轮只使用官方 384D 神经 acc-token pooling，不包含默认输出中的未编码 gravity；不能据此否定完整 Bio-PM HAR pipeline。
3. **未证实 Bio-PM pretrained 优于 random-frozen control。** BA 差 −0.0008，7/15 改善，CI [−0.0062,+0.0053]，dz −0.068；AUROC 差 −0.0024，8/15 改善，CI [−0.0109,+0.0062]，dz −0.135；两项 BH q=1.0。DD Recall 的正向点估计伴随 PD Recall 下降，相关 CI 均跨零，不能归因于稳定的预训练收益。
4. **HarNet/BioPM disease complementarity 尚未检验。** E4 的 Bio-PM 价值门槛失败，因此按条件未执行；未测量不等于证明不存在互补信息。
5. **Dual-prior 未训练。** Bio-PM 未稳定接近 HarNet10，也没有满足 E4 证据要求；不存在可报告的 dual-prior 增益。
6. **最佳 retained single-model 仍是 frozen WSSL-STR。** STR-01 + pretrained-frozen HarNet10 final1024 wrist residual，原 recipe 和 0.5 decision rule。Accuracy 0.7456、BA 0.7196、AUROC 0.7596、Macro-F1 0.7066、PD Recall 0.7823、DD Recall 0.6569。本轮没有保留新 single-model candidate。
7. **最高预注册 inference AUROC 为六模型等权 ensemble：0.7870。** Accuracy 0.7541、BA 0.7150、Macro-F1 0.7099、PD Recall 0.8090、DD Recall 0.6210。相对 frozen 三 seed ensemble（AUROC 0.7816 / BA 0.7200），AUROC +0.0054，14/15 改善，CI [+0.0028,+0.0074]，dz 1.127，E0 BH q=0.0427；BA 与 DD Recall 点估计更低。只作为 inference reference，不替代 single-model。
8. **当前不支持新机制或 dual-prior training。** 保留现有模型，停止本次 5s / Bio-PM acc-only integration 变体。External-prior 路线整体未被否定，后续只考虑有官方资产和明确依据的单一 prior 验证，并规划真正独立的验证数据。不得重开已停止的 HarNet adapter/gate/domain/layer 或 STR architecture/loss/sampling/augmentation 搜索；不会自动启动另一个 encoder。

Optional plot rendering was unavailable because the plotting runtime lacks matplotlib; no extra plotting dependency was installed. All requested numerical tables, CIs and audits are complete.
