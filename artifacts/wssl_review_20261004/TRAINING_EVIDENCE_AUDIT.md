# WSSL-STR training evidence audit

Date: 2026-10-04. Scope: read-only review of the local copies explicitly listed below, including the current authoritative server source/report snapshot in `current_source/`. No training, outer access, checkpoint selection, architecture change or historical-result modification. **Status: historical runner/manifest and aggregate coverage audit complete; original weights/logs were not re-read by this subtask.**

## 1. Evidence hierarchy and current retained method

The latest local WSSL EMA and Phase-4 reports agree on retained single-model WSSL-STR: frozen official HarNet10 final 1024D wrist features added through the learned 1024→64 residual projection to STR-01, with its original bilateral/activity/structured path. Seed metrics average within each of 15 development splits, then splits average equally. Accuracy .745604, BA .719644, AUROC .759553, Macro-F1 .706589, PD Recall .782344, DD Recall .656945. Classifier-side trainable parameters 143172; frozen HarNet parameters 10457408; total 10600580.

Evidence: `../wssl_ema_20261002/FINAL_REPORT.md`, `../phase4_window_dynamics_20261002/FINAL_REPORT.md`, and the ordinary rows of `../wssl_ema_20261002/analysis/ema_seed_split_metrics.csv`. Reports are supported by aggregate rows, but this local audit does not re-read the original server weights or every epoch JSONL.

## 2. What the actual WSSL trajectories establish

I read the existing diagnosis implementation and its CSV outputs, rather than relying only on prose:

- `../wssl_ema_20261002/scripts/diagnose_trajectory.py` reads all archived ordinary logs, constructs datasets from each original split ID list and actual checkpoint normalization, and evaluates only real best/last checkpoints in `eval()` / `no_grad()`.
- `analysis/trajectory_45runs.csv` has 45 unique (seed, context, inner) records; all stop exactly 12 epochs after the BA-selected best. Best epochs 2–24, stopping epochs 14–36, **zero runs reach the 50-epoch cap**.
- `analysis/eval_mode_best_last.csv` has 180 rows: 45 runs × best/last × train/validation. This represents 90 actual checkpoints, not 180 independent models.

| Actual checkpoint | Train eval BA | Validation eval BA | Train eval AUROC | Validation eval AUROC | Train main CE | Validation main CE |
|---|---:|---:|---:|---:|---:|---:|
| best | .913406 | .719644 | .964805 | .759553 | .239857 | .679387 |
| last | .999546 | .658764 | .999988 | .745135 | .013802 | 1.099845 |

The last-minus-best validation BA is −.060880 and AUROC −.014419. Best train CE evaluated with dropout off is .239857, versus .291046 in online dropout-active train logs. Online train AUROC was not recorded. These facts support a real train/validation gap and later overfitting under the existing recipe; they do not support insufficient training caused by the 50-epoch cap.

The mean BA peak at offset zero in `analysis/trajectory_best_neighborhood.csv` is conditioned on selecting by BA. It is not an independent demonstration that BA is universally the best stopping metric. Only best and last weights exist: intermediate epoch metrics can be read from logs, but intermediate train-eval predictions and retrospective full EMA cannot be reconstructed.

## 3. Reproduction and EMA coverage

The independence correction uses explicit optional SSL forward input and instance-local feature storage, retaining old state keys. The actual independent class contains no captured instance closure; its copy helper preserves Python/NumPy/CPU/CUDA RNG and creates fresh parameter storage.

Evidence: `../wssl_ema_20261002/scripts/independent_wssl.py`, `scripts/test_independence.py`, `analysis/independence_45checkpoints.csv`, `analysis/independence_test.json`, `ITEM01_INDEPENDENCE.md`. The 45 real eight-subject batches have bit-identical old/new logits; archived probability max difference 1.11e-16. The tested historical construction paths are Phase-3B and Phase-4, **not every historical experiment**.

`scripts/audit_completed.py` explicitly checks model-state tensor equality, epoch equality, normalization equality, expected update counts, source/cache/split hashes and archived prediction hashes. `analysis/item04_45run_audit.csv` contains 45 unique runs with all ordinary-weight, ordinary-trajectory and checkpoint-epoch checks true. The corresponding audit additionally records ordinary validation predictions exactly equal. This is strong evidence for reproduction of this locked WSSL configuration.

Exactly one EMA design was tested: initialization from ordinary starting parameters, per-optimizer-step updates from step one, one-epoch half-life, S=26/27, alpha=.9736927206974342/.9746546091224311. Ordinary BA chooses the same epoch for ordinary/EMA; EMA never chooses its own epoch. All 45 ordinary runs reproduced; EMA BA decreased in 15/15 splits, AUROC in 11/15; DD Recall fell .101568. The registered stop decision applies to coefficient/start-time/combination/stopping-rule EMA search. It does not imply every imaginable averaging method or stopping rule has been proved globally optimal.

## 4. Historical optimization coverage: avoid overstating scope

| Area | Local evidence actually available | What is established | Remaining boundary |
|---|---|---|---|
| LR, WD, batch, temporal dropout, scheduler | `../MFAM-pure-deep/training_recipe_staging/TRAINING_RECIPE_CONFIRMATION_REPORT.md` plus candidate YAML | V8-GN: LR1e-4/3e-4, batch16 at LR1e-4, WD1e-3, dropout.2, constant LR were screened at seed42; LR1e-4 alone received three-seed confirmation and was rejected | This is **V8-GN**, not a WSSL-specific optimization study; report-only historical metrics locally, no original per-run logs/weights rechecked |
| Optimizer | Same V8 report explicitly states no optimizer comparison | AdamW recipe is retained | AdamW cannot be claimed globally optimal from absence of a comparison |
| STR Phase-2 optimization/loss/augmentation | Current server snapshot: Phase-2 manifest, runner/hooks, protocol/report and 765 aggregate run rows | Sixteen candidates plus baseline each cover 45 unique seed/split units; all are STR-01, not WSSL | WSSL classifier hyperparameters were not retested by this matrix; the loss/augmentation route has an explicit stop decision |
| SSL tuning/gates/domain/layers/threshold | Direct Phase-3C/3D manifests, protocols, runners/hooks, reports and aggregate CSVs | Last-block, multilevel, adapter, gate and domain adaptation have full 45-unit rows; train-OOF thresholds are separate diagnostics; activity gate was conditionally not run | No current frozen-WSSL classifier-only LR/WD/batch/dropout/scheduler/optimizer comparison; SSL encoder LR adaptation is a different variable |
| Alternative SSL priors | `../phase3e_external_priors_20261001/PHASE3E_REPORT.md`, protocol, `analysis/model_completion.csv`, audit | HarNet5, BioPM-only/BioPM-residual/random control were tested, all rejected; dual-prior branch not run | Different official dimensions/capacity and extraction rules prevent assigning all effect to temporal scale; skipped conditions are not failed experiments |
| SSL window/Raw Acc/DD auxiliary | `../phase4_window_dynamics_20261002/FINAL_REPORT.md` and numbered reports | Four 45-run candidate matrices rejected; stopping-rule review condition explicitly skipped | No new best method and no automatic follow-on search; no general proof that every representation design is inferior |
| Longer training/early stopping | WSSL archived best/last eval and all epoch summaries | Later observed checkpoints overfit; 0/45 hit max50 | No independent alternative stopping-rule comparison; cannot select a retrospectively best rule on these validation trajectories |
| EMA | Full local protocol/report/aggregate audits | Specified single control loses BA/AUROC/DD Recall; stop EMA route | Only this fixed design was measured, but explicit stop is binding |

V8's rejected LR1e-4 final study had BA+.0043, 11/15 wins, but missed its preregistered +.005 gate, increased seed SD and lowered DD Recall. It is not valid to retain the seed42 improvement or call it a WSSL result.

## 5. Fixed-architecture checks that are supported by these records

1. Audit original/current code/config identity, numerical class-weight construction, train-only normalization, scheduler step/order, ID retrieval, threshold direction and frozen HarNet status against actual artifacts. These are correctness and reproduction checks, not new performance candidates.
2. Verify reported WSSL split/seed means and seed variability from original predictions, if needed; no threshold/epoch/probe selection.
3. Use existing actual best/last checkpoints and logs to test undertraining/overfitting claims. Avoid repeating 45 training runs merely to re-establish the reproduction already demonstrated unless a concrete current-code discrepancy remains.
4. Enumerate direct Phase-2/3C/3D coverage before proposing any current fixed-architecture controlled test. A true evidence gap does not by itself cancel an explicit prior stop decision. New testing should answer a defined correctness or coverage question, preserve frozen formal outputs, use only development units, and not silently reopen rejected augmentation/loss/EMA/module routes.

## 6. Interpretation limits

Fifteen fixed splits share subjects and training contexts; their paired bootstrap intervals describe repeated development robustness. The 45 seed runs, windows, activities, pairs and optimizer updates are not independent inferential samples. Multiple sequential phases reused the same development set, so the retained configuration is the best supported **within the tested, stopped search history**, not an independently confirmed global optimum. None of this review reads FOE-01 or validates WSSL on outer data.

## 7. Direct server snapshot: exact candidate-to-code mapping

Source prefix below: `current_source/artifacts/`. The current config is `current_source/foundation_validation/configs/str01_seed42.yaml`, inheriting `pads_multi_activity_v3_nested_cv.yaml`; seeds43/44 only change experiment seed/name/output. The effective STR recipe is AdamW LR2e-4, WD1e-4, betas.9/.999, batch8, cosine T_max50/minLR1e-6, clip5, FP32, balanced CE without smoothing, dropout encoder/activity/classifier .1/.1/.2, strict greater-than ordinary validation BA selection, patience12/max50.

### Phase-2: STR only, 16 single-factor controls

`phase2_str_training_20260926/scripts/run_inner.py` loads `str01_seed{seed}.yaml`, applies only manifest `overrides`, then activates `phase2_hooks`; there is no HarNet or SSL cache/model factory. The declared candidate maps and executed hooks agree:

| Candidate | Actual change relative to STR-01 | Implementation evidence |
|---|---|---|
| lr_1e4 | `training.learning_rate=1e-4` | manifest override, applied by runner |
| wd_1e3 | `training.weight_decay=1e-3` | manifest override, applied by runner |
| smooth_005 | `loss.label_smoothing=.05` | manifest override, applied by runner |
| ce_unweighted | `loss.class_weights=null` | manifest override, applied by runner |
| ce_sqrtweighted | square-root train-only inverse-frequency CE weights, normalized mean1 | hook custom loss reads current-fold train counts |
| focal_balanced | balanced CE per item ×(1−p_target)^2, then mean | hook FocalLoss, gamma2; not a new backbone |
| ce_effective099 | effective-number train-only CE weights beta.99, normalized mean1 | hook custom loss |
| balanced_sampler | train-only inverse-frequency `WeightedRandomSampler`, replacement, len(train) draws; balanced CE unchanged | hook training loader only |
| jitter_equal / jitter_dd_only | normalized noise SD.01, probability.5 both classes /1 DD and0 PD | AugDataset accesses only training base |
| scaling_equal / scaling_dd_only | raw-space factor N(1,.03), clipped[.94,1.06], restandardized with same train normalization | AugDataset uses original per-activity/wrist mean/std |
| temporal_equal / temporal_dd_only | smooth sine displacement max1% of valid duration, endpoints retained | training dataset only |
| mixup_equal / mixup_dd_only | same-label input mixup lambda[.9,1], requires same mask/length | training dataset only; unchanged labels and validation |

The aggregate `variant_seed_split_metrics.csv` has **765 rows =17×45**. I independently checked each variant has exactly 45 unique (seed, outer, inner) tuples, seeds42/43/44, 15 splits and finite six metrics. This proves aggregate matrix completeness, not a fresh checkpoint audit. `HISTORICAL_MATRIX_AUDIT.csv/json` preserves this check. `variant_completion.csv` agrees. All sixteen are rejected by the direct Phase-2 report; no augmentation combination was run.

The operational numeric Phase-2 gate was disclosed as specified after baseline reproduction and an interim check of already-completed negative LR1e4/equal-jitter results. It should not be called fully preregistered before every candidate result. Preserve that timing limitation; it does not justify re-selecting a candidate from the recorded metrics.

### Phase-3A: also STR only

The runner again loads STR config, not WSSL. The new manifest contains balanced sampler + unweighted CE, SAM rho.05 with AdamW, and activity auxiliary CE weights.1/.3; the balanced-sampler/balanced-CE control is reused from Phase-2. `phase3_hooks.py` implements exactly those modifications. The aggregate includes six complete45-unit matrices (270 rows, including original/reused controls); this must not be described as 270 new WSSL runs. No WSSL LR coverage arises from this phase.

### Phase-3B: introduce frozen WSSL, keep the inherited recipe

`phase3b_external_ssl_20260928/manifest.json` has SSL-only pretrained, STR residual pretrained, and STR residual random. The runner loads original STR config and chooses only `model_kind` / `embedding_kind`. `transfer_hooks.py` replaces the model factory and supplies subject-ID-indexed SSL features to the unchanged training/eval routines. No LR/WD/batch/dropout/scheduler/optimizer override exists. The four aggregate matrices (baseline plus three candidates) each have45 units,180 rows. This establishes the pretrained-frozen residual benefit and a fixed random control, not classifier recipe optimality.

### Phase-3C: SSL-block/layer/threshold/sample-size diagnosis, not classifier optimization

The manifest specifies last-block LR ratio.1, OOF3folds/fixed10epochs, middle+final feature, and train-count fractions25/50/75/100. For adapt, `phase3c_hooks.py` creates two optimizer groups: **base+projection2e-4**, **HarNet layer5 2e-5**, WD1e-4 both, preserving cosine LR ratio. Layer5 BN statistics are frozen. Frozen and multi use the ordinary original optimizer; no frozen WSSL classifier LR1e-4 run appears. The standard model aggregate has180 rows =STR/frozen/adapt/multi ×45; each complete.

OOF fitting uses smaller training subsets and exactly epoch10 with cosine horizon still50. It tests transfer of a train-derived threshold onto the already frozen validation predictions. It is not a matched early-stopping-rule or current full-train fixed-epoch optimization study. The corrected threshold analysis note transparently records the wrong expected count405→135 and an ensemble aggregation indentation fix5→15split rows; training/predictions/threshold rule were unchanged. Final direct report keeps .5 because thresholding worsened BA/Macro-F1.

### Phase-3D: adaptations and gate, classifier recipe stays inherited

The adapter/gate runner loads original STR config and patches only models and cache handling. Adapter uses the same base LR2e-4/WD1e-4; no new optimizer group. Gate changes only fusion. The domain pretext phase separately updates layer5 LR1e-5 and a temporary decoder LR1e-3 for5epochs; afterwards the unchanged supervised WSSL classifier runs with original recipe. **These encoder-side/pretext LRs are not a classifier-only LR comparison.**

The model aggregate has270 rows: STR, frozen, lastblock read-only references plus adapter/gate/domain ×45. Completion table records activity_gate0/45 because the gate prerequisite failed. The activity gate must remain “not run”, not a measured rejection. Direct report rejects adapter/gate/domain and explicitly stops adaptation/layer/gate routes.

### Reused references are exact copies of metrics, not independent evidence

By (seed, outer, inner) keys, all six metrics in Phase-3B baseline, Phase-3C str and Phase-3D str equal Phase-2 baseline exactly (maxdifference0). Phase-3C frozen and Phase-3D frozen equal Phase-3B b_str_pretrained exactly (maxdifference0). These are appropriately reused reference matrices; their presence in several analyses does not supply multiple independent reproductions or extra WSSL hyperparameter tests. The separate EMA run is the actual later45-unit ordinary reproduction.

## 8. Real remaining fixed-architecture control variable

**An inherited frozen-WSSL classifier-side learning rate is genuinely untested as a single-factor comparison.** A bounded contrast of original2e-4 versus one predetermined1e-4 would change no model module, feature, HarNet parameter/statistic, bilateral/activity/structured path, loss, batch, threshold or selection metric. Phase-2 tested1e-4 on STR-01 alone; that negative result cannot logically establish its effect after adding the learned SSL residual projection. Phase-3C's2e-5 applies to an unfrozen HarNet block, not the present frozen WSSL classifier. Phase-3D/3E/Phase-4/EMA retain classifier2e-4.

This identifies an evidence gap; it is **not a claim that lower LR will improve**, nor a recommendation to reopen a grid, loss, sampling, augmentation, EMA or structure search. If the current user-authorized review proceeds to testing, this one control can be specified before results, use the same45 units and seed-first15split paired gate, and retain the original formal baseline unless the fixed joint endpoint succeeds. Do not use partial seed results to select LR, add a third LR after a failure, tune patience to rescue it, or use outer information. It will remain another development-only comparison with selection-optimism limits. I have not launched it.

## 9. Remaining verification scope

- The root audit must still verify current canonical code/config/asset identity against actual server checkpoints and normalization; local CSV rows alone are insufficient for those binary/input claims.
- A new fixed-architecture correctness or LR test must supply actual smoke/run/output evidence before being described as complete.
- The strongest supported statement today is reproducible retained WSSL within the examined candidates; **global or near-global recipe optimality is unproven**.

Operational documentation distinction: the generic base YAML includes `model_selection.threshold.source=inner_oof_only` and AMP=true. STR overrides AMP=false; actual primary inference uses argmax/.5, and only the separate rejected Phase-3C E2 diagnostic selects OOF thresholds. Review expanded runtime configs and executed runner behavior; do not assign unused generic declarations to the retained method.
