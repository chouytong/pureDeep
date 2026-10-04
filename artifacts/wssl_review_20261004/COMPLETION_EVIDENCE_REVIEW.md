# Completion evidence review — 2026-10-04

Status: technical review and the single existing-recipe test have supporting evidence; final closure documentation/publication remains **PENDING** at this review. No new model, training, code change, deletion or experiment was performed for this review.

## Scope and sources

The active objective is to review the current model across preprocessing, hyperparameter selection, model design and feature fusion, then test a justified existing method. Earlier restrictions require one performance candidate at a time, development-only selection, frozen formal assets and Git publication of completed work. The authorized control is the existing frozen WSSL classifier at initial LR1e-4 versus its locked LR2e-4 reference. It is not a new architecture.

This independent check read current local reports, executed audit JSON, aggregate CSV, protocol, frozen scripts and lock. It verified all seven relative-file lock hashes and recomputed the completeness, seed-first means, split means, paired deltas, improvement/tie/worse counts and paired Cohen dz from the aggregate tables. It did not independently open server-only datasets, checkpoints or individual prediction tables; their stronger per-stage checks are implemented in the SHA-locked runner/analyzer and were executed by the parent task on the server. This note does not substitute aggregate tables for that live artifact verification.

## Requirement-to-evidence coverage

| Requirement | Current evidence | Supported conclusion / boundary |
|---|---|---|
| Restore current retained model and formal baseline | `AUDIT_REPORT.md`, `assets_normalization_audit.json`, 45-row asset/recipe CSV | Frozen WSSL = STR-01 + frozen HarNet10 residual; 143,172 classifier parameters, 10,457,408 frozen encoder parameters; original metrics unchanged |
| Review preprocessing and leakage | `PREPROCESSING_AUDIT.md`, `preprocessing_contract.json`, 15 normalization refits | 390 subjects / 4,290 records / 8,580 wrists reviewed; raw Gyro trim and fixed SSL inputs/counters/ID order verified; all 15 train-only normalization refits agree with all 45 checkpoint statistics |
| Review physical/input assumptions | Units, timestamps, clip/pad/resampling contract and historical solver evidence | Exposes uniform-index resampling, four large timestamp gaps and wrist clock offsets; does not establish per-device calibration or clinical phenotype preservation; no correction/exclusion was made |
| Review current design and fusion | `FUSION_CODE_AUDIT.md`, current foundation source, `fusion_properties.json` | Existing GN wrist encoder and unchanged residual/full bilateral/activity/structured paths agree; concrete closure-copy issue was repaired in the earlier independent interface; no new retained-path bug demonstrated |
| Correctness/reload/isolation verification | 45 sampled-checkpoint batches, first-checkpoint instance/cache/RNG test, two-step training-only scratch | Old/new/reloaded logits exact; explicit features take priority; ID permutation and finite mask invariance pass; scratch proves gradient connectivity, not a new model's performance |
| Reproduce original training | Prior completed EMA ordinary controls plus current formal-asset SHA/ID recheck | All 45 ordinary trajectories, selected weights and complete validation predictions were already exactly reproduced; this round correctly reuses that evidence rather than claiming another 45-run baseline reproduction |
| Review training sufficiency | Existing real best/last eval and per-epoch logs | Best epochs2–24, all stop best+12, no max50 cap hits; later train improvement with validation deterioration supports overfitting, not an epoch-cap explanation; no fictitious middle checkpoint/retrospective EMA |
| Review hyperparameter coverage | `TRAINING_EVIDENCE_AUDIT.md`, historical matrix, actual checkpoint recipe | Separates STR-only Phase2 searches from current WSSL; generic AMP/threshold metadata are scoped to actual runtime; inherited WSSL LR was a genuine target-specific gap |
| Fixed single-method test | `LR_CONTROL_PROTOCOL.md`, `lr_control_lock.json`, `SMOKE_REPORT.md`, initialization check | Initial state/logits exact and smoke PASS before formal results; only initial LR changes; all other recipe/architecture/decision rules fixed; no EMA or encoder adaptation |
| Complete formal candidate matrix | `lr_execution_state.json`, `lr_seed_split_metrics.csv`, `lr_decision.json`, frozen analyzer | 45 candidate runs complete, all seeds42/43/44 and all 15 fixed splits; analysis checks eight per-stage artifact hashes and ID/label/normalization/decision alignment; no partial-seed selection |
| Proper paired inference and errors | 15-split seed-first, paired, seed-SD/agreement and error tables | Metrics first average the three seeds inside each split; all metric means/counts/effect sizes independently recomputed here; bootstrap/BH code inspected; overlapping splits remain descriptive development units |
| Choose/stop based on predefined joint rule | `lr_decision.json` plus re-derived paired table | LR1e-4 **REJECT**; original LR2e-4 retained; no LR/WD/patience rescue or additional candidate follows from this outcome |
| Final documentation/Git delivery | Existing audit/protocol/smoke milestones published; final closure still pending | Requires final report, latest status/README/handoff and final verified commit/push/tag before claiming fully delivered |

## Result check: positive and negative evidence both matter

| Metric | Original LR2e-4 | LR1e-4 | Paired mean difference | Split improvements |
|---|---:|---:|---:|---:|
| Accuracy | 0.745604 | 0.741614 | −0.003990 | 6/15 |
| BA | 0.719644 | 0.725133 | +0.005489 | 9/15 |
| AUROC | 0.759553 | 0.758559 | −0.000994 | 6/15 |
| Macro-F1 | 0.706589 | 0.706159 | −0.000431 | 8/15 |
| PD Recall | 0.782344 | 0.764914 | −0.017430 | 5/15 |
| DD Recall | 0.656945 | 0.685352 | +0.028408 | 11/15 (one tie) |

BA bootstrap CI is [0.000352, 0.011512] and DD Recall CI [0.006280, 0.049196]; these positive descriptive effects must not be erased merely because the candidate is rejected. BA improves only 9/15 splits, below the fixed ten-split requirement. AUROC mean falls, Macro-F1 mean falls and PD Recall mean falls by more than the allowed0.01. Thus the joint reject decision is correct without modifying any gate after results.

The positive BA bootstrap interval is not a claim of universally significant superiority: paired Wilcoxon BA p=0.094604, primary-family BH q=0.189209. AUROC CI crosses zero. DD Recall exploratory six-endpoint BH q=0.166889. These methods summarize different aspects of the same repeated-development design; report them as recorded.

Mean within-split seed SD rises slightly for BA0.021603→0.022919 and AUROC0.021741→0.022825, although both remain below the protocol's1.25× cap. Do not summarize this as variance unchanged. New control has median best/stop epoch12/24 versus10/22, and no max50 cap hits. Candidate training loss remains online Dropout-active; only the earlier original-model best/last eval establishes an eval-mode training gap.

## Remaining closure items

1. Produce final result report with both the BA/DD benefit and AUROC/F1/PD trade-off, rejection and retained original recipe. Include paired intervals/counts/effect sizes, seed variability/agreement and errors; do not call LR1e-4 wholly ineffective or original recipe globally optimal.
2. Append latest terminal state to `CURRENT_STATUS.md`; its most recent visible section still states the45-run matrix is running/incomplete. Earlier blocked/pre-smoke sections may remain as history if clearly superseded. `SMOKE_REPORT.md` is an entry milestone and should remain scoped to that date, not be used as the final status.
3. Supersede planned language in the audit/fusion summaries with the completed control result or explicitly point readers to the final report. Preserve historical protocols/source/results.
4. Append README/handoff with the terminal outcome and development/outer boundary; verify the final server-source/checkpoint/cache/prediction integrity closure already checked by runner/analyzer remains valid.
5. Commit/push/tag the final code/reports/aggregate outputs, verify repository main and tag against the intended commit. Dataset, weights, caches and per-subject label/prediction tables remain excluded.

## Claim boundaries for final response

- The supported selection is **best retained configuration among tested candidates**, not a global optimum in preprocessing, loss, hyperparameters or fusion.
- The current audit reviews existing development data/contract across390 subjects; use precise wording that no outer predictions/performance or test loader informed fitting/selection. Avoid a broader literal claim that no raw file belonging to any context's test subject was ever inspected, because subjects recur across development contexts and the contract audit spans the pool.
- Correctness PASS is not clinical semantic preservation, independent HarNet re-extraction or a re-run of every real Acc ADMM solve.
- Failed historical candidate families stay closed. No new architecture/augmentation/loss or follow-on LR search is needed to finish this bounded audit and single existing-method contrast.
- No new external validation was performed. Original FOE-01 results remain frozen and cannot validate this later development selection independently.

After these documentation/publication items are verified, the requested bounded current-model review and single justified existing-recipe test have no remaining implementation/test acceptance gap shown by the inspected evidence. At this review point, full delivery remains unproven because final publication/closure is pending.
