# Existing frozen-WSSL single learning-rate control

Frozen before any candidate validation output, 2026-10-04. This is a controlled recipe test of the existing model, not a new network. Scope derives from the current user-authorized model review followed by testing. Root audit PASS is required first.

## Question and two configurations

Does classifier-side initial LR1e-4 improve the retained frozen WSSL configuration relative to inherited LR2e-4? Phase2's STR-only result does not establish this effect in WSSL. No expected positive outcome is assumed.

- Reference: all45 archived Phase3B `b_str_pretrained` stages, already exactly reproduced in the completed ordinary EMA controls. Reuse SHA-locked original predictions; no extra reference training needed.
- Only new performance candidate: the same frozen WSSL with `training.learning_rate=1e-4`. Exactly15 fixed subject-level development splits × seeds42/43/44. Candidate has no EMA.
- STR architecture, SSL residual mapping, HarNet official weights/statistics/cache, activity and wrist order, normalization, loss and class weights, dropout, AdamW betas/WD1e-4, batch8, clip5, FP32, cosine horizon50/minLR1e-6, max50, strict ordinary BA selection and patience12 stay fixed. DD probability>.5 meansDD; tiePD. No threshold fitting.
- LR applies to all143172 classifier-side parameters, including learned SSL normalization/projection; all10457408 HarNet parameters remain frozen and outside the optimizer. Total parameter count unchanged.
- Each LR uses the same ordinary BA best-epoch rule independently. Do not force the new LR onto the reference epoch; do not search a better endpoint from several epochs/metrics.
- Changing initial LR changes the cosine path, relative floor and effective AdamW step shrinkage. This contrast is the effect of initial LR under fixedWD/floor, not an isolated mechanistic proof about one of those quantities.

## Correctness and execution

Use the independent WSSL interface already compatible with original state keys and the original engine, with a train|validation-only dataset builder and subject-ID SSL loader. No engine training/evaluation forward computation changes. Recompute train-only normalization exactly. Config diff, original source/cache/split hashes, class-weight/ID identity and no-outer invariants must be checked.

Run a disposable train-only gradient/property test (already PASS) and original engine smoke for one candidate split before the full matrix. Smoke never selects LR/config or contributes to final metrics. Compare initial model state/forward at both learning-rate configs, because LR must not change initialization or inference. Existing45 ordinary reproduction supplies the reference trajectory guarantee; lowLR trajectories need not be numerically identical after updates.

Only complete, independently verified units can be reused. No overwrite or incomplete checkpoint resume: historical engine does not save complete RNG/loader state. If a unit interrupts, preserve its failed directory/log and document any from-seed restart in a new attempt directory. No concurrent candidate training or partial-seed selection. If all inputs remain verified, complete the full45matrix regardless of early scores. After folds, shut down persistent worker iterators to avoid descriptor accumulation; this cleanup cannot affect within-fold numerical training.

Write outputs only to `artifacts/wssl_review_20261004`, keep original artifacts unchanged. Runtime weights/cache/predictions stay on server. Git includes scripts/protocols/reports/aggregate tables only.

## Analysis, before results

Primary BA/AUROC; secondary Accuracy/Macro-F1/PDRecall/DDRecall. For each run use its original fixed threshold. Average the three seed metrics within each split, then average15splits; do not compute metrics of an ensemble as the primary endpoint.

Report paired deltas, improvement/tie/worse counts, paired split bootstrap95%CI (10000 draws, RNG20261004), Cohen dz, two-sided paired Wilcoxon and BH over the two primary endpoints. Also disclose BH over all6 endpoints as exploratory, within-split seedSD, SD of per-seed mean metrics, splitSD, seed prediction agreement, selected/stop epochs, losses and ID-aligned corrected/new errors. No subjects/pairs/45runs are independent units. Overlapping split uncertainty is descriptive development robustness, not independent population validation.

Fixed joint retention gate, inherited from previous Phase4/EMA protocols:

1. BA delta>0, at least10/15 improving splits, BA CI lower bound>0.
2. AUROC delta>0 and at least10/15 improving splits.
3. Mean Macro-F1 does not decrease.
4. Mean PDRecall and DDRecall each decrease no more than .01 absolute.
5. Mean within-split seedSD for BA and AUROC each ≤1.25×reference (floor .001 on reference SD).

All criteria must pass. AUROC CI/BH are reported even though the inherited gate does not separately require their significance; do not label a gate pass as proof of population superiority. No further criterion may be added after seeing candidate results.

If rejected, retain original2e-4 and stop this LR control. Do not add LR values, WD compensation, extraepochs/patience, another optimizer/scheduler, EMA or a combination. If accepted, call1e-4 the best supported tested recipe for this development history; no new outer evaluation or architecture is automatically authorized. If it hits epoch50, report the fixed-budget limitation without rescue training.

## Boundary

No FOE01 or historical quarantined outer outcomes, H1/handcrafted features, diagnostic probes, fusion corrections or outer errors are read for selection. Prior failed architecture/readout/gate/adapter/domain/augmentation/loss/EMA routes remain stopped. Multiple sequential phases reuse development; this adds selection optimism and cannot establish global/near-global optimality or external generalization.
