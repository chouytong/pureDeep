# Frozen-WSSL learning-rate control: independent protocol review

Date: 2026-10-04. **Status: review recommendation before candidate training/results, not an executed experiment or a finalized protocol.** This review changes no code, trains no model and reads no outer information. The root task must freeze the actual protocol, manifest and source hashes before formal candidate validation results are inspected.

## 1. Is this contrast justified and within scope?

**Yes, as one bounded test of the existing model's inherited training recipe.** It is not a new model, external encoder or fusion design. The question is whether the present frozen-WSSL classifier-side LR2e-4 remains preferable to one predetermined lower LR1e-4 under the same complete development protocol.

Direct evidence is in `TRAINING_EVIDENCE_AUDIT.md` sections7–8:

- Phase-2 `manifest.json` and `scripts/run_inner.py` test LR1e-4 on STR-01 only; no SSL model or feature hook exists.
- Phase-3B uses the original STR config with frozen SSL insertion and no training override.
- Phase-3C changes only the unfrozen HarNet layer5 LR to2e-5; original STR/projection LR remains2e-4. That is not a classifier-LR test for the retained frozen model.
- Phase-3D adapter/gate/domain, Phase-3E, Phase-4 and EMA retain the same supervised classifier-side recipe.
- The EMA ordinary control reproduces all45 baseline trajectories, selected weights and predictions exactly; redundant retraining of another45 baseline runs is unnecessary if the current technical audit confirms unchanged inputs/source compatibility.

The existing WSSL trains to near-perfect training performance at its actual last checkpoints while validation declines. A lower LR could alter the trajectory and selected-checkpoint generalization; **that is a hypothesis, not evidence that it works**. Current data do not establish undertraining, global hyperparameter optimality or fusion failure. The control closes one specific coverage gap without reopening a grid or failed architecture, loss, sampling, augmentation, encoder-adaptation, threshold or EMA route.

## 2. Exact comparison and frozen invariants

| Field | Reference | Single candidate |
|---|---|---|
| Model | Retained frozen WSSL-STR | Identical class and modules |
| Classifier-side learning rate | 2e-4 | **1e-4 only** |
| HarNet | Official frozen HarNet10 final1024 cached wrist features | Same weight/cache hashes; no encoder gradients or statistics updates |
| Trainable parameters | 143172, all current classifier-side parameters | Same143172; no differential LR group |
| Optimizer / WD / betas | AdamW /1e-4 /(.9,.999) | Identical |
| Batch / scheduler | 8; cosine T_max50/minLR1e-6 | Identical absolute minimum1e-6 |
| Loss / dropout / clipping | Train-balanced CE; smoothing0; .1/.1/.2; clip5 | Identical |
| Inputs and normalization | Original processed STR six channels + original frozen SSL cache; activity/wrist order; train-only normalization | Identical |
| Subject splits / seeds | 15 fixed development splits /42,43,44 | Identical45 units |
| Early stopping | Ordinary validation BA, strict increase, minimum_delta0; patience12/max50 | Same rule |
| Decision | DD probability>.5; tie remains PD | Same rule |
| Inference | Single best checkpoint per run, no ensemble candidate | Same |

Use the corrected independent model implementation, already proven equivalent to archived WSSL, for this candidate. Changing LR means all classifier-side trainable parameters, including STR backbone, SSL LayerNorm/projection, activity aggregation, structured residual and classifier. Calling it “SSL-only LR” would be wrong. HarNet is not part of the optimizer.

Keep the absolute scheduler minimum1e-6 and weight_decay1e-4 unchanged to preserve a one-config-field intervention. Their relative/effective effects nevertheless depend on LR: cosine minimum relative to initial LR becomes.01 versus.005; AdamW shrinkage per update contains the LR. **The experiment estimates the effect of changing initial LR under this fixed recipe, not an isolated universal optimization-speed mechanism.** Do not silently scale minimum LR or WD to compensate.

## 3. Technical entry gate

Formal45-unit candidate training may start only after these checks pass and their evidence is recorded:

1. Current baseline audit confirms architecture/state keys, classifier parameter count, source compatibility, fixed split SHA, frozen HarNet and cache SHA, original checkpoint/prediction integrity, subject-ID/label alignment, normalization and original six-metric means.
2. Same old checkpoint and same real development batch give matching original/corrected logits; independent parameter/cache storage and fresh checkpoint reload pass. No test loader is created.
3. Expanded candidate config differs from expanded baseline only in `training.learning_rate`, new output/name/provenance and explicit diagnostic annotations. Canonical inputs/order/loss/threshold/selection rule do not differ.
4. Fresh starting model parameter tensors and first batch subject IDs/input hashes match the reference under each seed. No constructor/evaluation consumes extra training RNG; deterministic settings and loader generators stay fixed. Before the first optimizer update, loss/logits should agree. After updates they are expected to differ because LR differs.
5. One-batch smoke verifies finite loss/gradients, optimizer LR exactly1e-4, unchanged balanced CE and normalization, best/last checkpoint reload, absence of HarNet adaptation and correct provenance. Smoke results are for technical correctness only; do not use their validation metrics to accept/reject or revise the LR.

If the baseline entry audit fails, diagnose correctness first and preserve original formal artifacts. A corrected implementation may require reproduction verification before performance testing. Do not interpret a source/config/data mismatch as an LR effect.

## 4. Execution and checkpoint selection

- Run all45 candidate units without seed/split screening. Never keep only favorable runs, replace a seed, adjust LR after partial metrics or stop the scientific comparison early for poor mean performance.
- Within each run, keep original ordinary-BA checkpoint selection and early stopping. **Each LR selects its own epoch by the same pre-existing rule.** This differs from EMA's same-training-trajectory snapshot requirement: two LR values produce different training trajectories. Forcing candidate selection to the archived baseline epoch would instead answer a different, potentially unfavorable question.
- Report best epoch, actual stopping epoch, total optimizer updates, cap hits, train/validation main CE and six selected-checkpoint metrics. First-common-epoch initialization/data-order checks can support coupling; numerical trajectory equality after LR changes is neither expected nor a valid acceptance test.
- Only actual best/last checkpoints may be re-evaluated in train `eval()` mode. Intermediate epochs have logged metrics only unless new checkpoints were prospectively saved. No retrospective checkpoint reconstruction or full historical EMA claim.
- If low LR hits max50 or seems delayed, report that as a limitation of this fixed-rule contrast. **Do not extend epochs/patience, change scheduler or choose a different stopping metric to rescue it.** No same-epoch secondary result may overturn the primary BA-selected result.
- A technical interruption may be repaired and rerun with the same frozen config. Since current checkpoint resume does not fully restore all RNG/loader states, do not silently resume a partially trained run and claim an exact original seeded trajectory. Restart that specific interrupted unit from seed initialization, preserve the failed attempt record and distinguish successful from failed attempts.

## 5. Proposed retention gate: carry forward the existing bounded rule

To avoid another result-dependent definition of “better,” use the already recorded Phase-4/EMA operational gate **unchanged**, fixed before this candidate's formal results:

1. Seed-first mean ΔBA>0, at least10/15 positive paired splits, paired bootstrap95% lower bound>0.
2. Mean ΔAUROC>0 and at least10/15 positive paired splits.
3. Mean ΔMacro-F1≥0.
4. Mean ΔDD Recall≥−.01 and mean ΔPD Recall≥−.01.
5. Mean within-split three-seed SD for BA and AUROC each≤1.25×reference, using the previously fixed.001 floor.

All five conditions are required. No Accuracy-only, AUROC-only, DD-only or agreement-only retention. AUROC CI and effect size must be reported even though the inherited gate does not require its lower bound>0; the result must not be described as a confirmed AUROC improvement if that interval includes zero. Report whether any passing numerical gain is small in absolute terms; the gate is an operational development selection rule, not a clinical usefulness threshold.

This is a recommendation to reuse an existing gate, **not an assertion that the root has already frozen it**. If the root adopts a different stricter gate, that choice must be explicit and frozen before candidate results; never change it in response to the observed45-unit table.

## 6. Analysis specification

- Pair by exact (context, inner, seed), with subject-ID/label validation-prediction matching. For each metric average the three seed metrics within split, then compare15 split values. Do not use score-averaged ensemble performance as the primary single-model estimand.
- Primary BA/AUROC; secondary Accuracy/Macro-F1/PD Recall/DD Recall. Report both means, differences, positive/negative/tied split counts, paired10,000-draw split-bootstrap95% CI, Cohen dz, Wilcoxon p and BH over the two primary comparisons. Fix bootstrap random seed and CI method in the executable analysis before results. Do not increase the multiplicity family or switch tests to obtain significance.
- Report within-split seed SD (average of15 sample SDs across3 seeds), seed-mean SD, split SD and prediction agreement separately. Seeds are variability checks, not45 independent inferential samples.
- Descriptive error changes use same ID-aligned validation subjects: corrected versus newly wrong, separately for PD/DD, first average counts over seeds within split. Do not select subgroups, thresholds or training policies using those errors. Existing stable-error grouping may be sensitivity only, not a new candidate design step.
- CIs/p-values describe this repeated, overlapping development protocol. Multiple historical phases used the same cohort; the result is not independent external validation and cannot revise old locked outer conclusions.

## 7. Frozen stopping and reporting decisions

| Outcome | Decision |
|---|---|
| Technical entry/integrity fails | Stop interpretation; repair and verify the same authorized control, do not tune for a desired outcome |
| Complete45 units and all gates pass | Retain WSSL architecture with LR1e-4 as a development-only recipe candidate; preserve original baseline and report all six outcomes/limits |
| Complete45 units and any gate fails | Retain original LR2e-4; stop this LR control, with no additional coefficient, differential group, WD, scheduler, patience or epoch rescue sweep |
| Candidate cap hits or stopping timing shifts | Report observed limitation; do not relabel a failed gate as success or rerun a modified schedule |
| Advantage only on Accuracy/AUROC or subset | No retention; publish the trade-off and full table |

Even a positive result does not automatically authorize combining the lower LR with rejected EMA, adapter, gate, window, raw-Acc, auxiliary, loss or augmentation methods. A negative result does not prove every possible WSSL recipe is optimal; it completes this one previously missing contrast. Both outcomes must be appended to README/handoff, archived separately, committed/pushed/tagged with code/protocol/aggregate reports only. Data, weights, feature caches, raw logs and per-subject records remain outside public Git selection. No outer artifacts are accessed.

## Review verdict

**The single LR control is scientifically defensible under the current review-and-test objective, provided the correctness audit passes and its protocol is frozen before results.** It measures a real gap in current-model training coverage and stays inside the current architecture. Its final claim must be “best supported among the examined fixed configurations,” never a guaranteed/global optimum. No other candidate or failed route should start alongside it.
