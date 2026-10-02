# WSSL window dynamics: locked development protocol

Registered 2026-10-02 before any candidate training or inspection. Execute items 1–8 in order; publish and tag each completed item before starting the next. Original archived artifacts remain unchanged.

## Baseline and boundary

Retained baseline is Phase-3B `b_str_pretrained`: STR-01 + frozen official HarNet10 final wrist features. Reuse only after all 45 archived development stages pass checkpoint, prediction, split, normalization and configuration audits. Latest formal Phase-3B–E records override historical V8/STR-only README/handoff entries. No historical outer performance, predictions, error groups or diagnostic models enter this study.

Fifteen existing inner-development splits, seeds 42/43/44; subject IDs are explicit matching keys. Class order PD=0/DD=1. DD probability >0.5 predicts DD (tie predicts PD, matching original argmax). Train-only activity/wrist/channel normalization, bilateral wrist order left/right and original 11-activity order remain fixed. AdamW lr 2e-4, weight decay 1e-4, batch 8, cosine minimum lr 1e-6, max 50 epochs, validation-BA early stopping patience 12, clip 5, no AMP, balanced train-fold CE, zero smoothing. No threshold or hyperparameter search. New runners supply empty test-subject lists and never create outer loaders.

## Ordered experiments

1. Audit and lock baseline, hashes, aggregate metrics and new directory.
2. Extract frozen HarNet10 front/back features and valid-window masks. Preserve raw trim 48, clip ±3g, short edge pad 12+12, first/last 1000 samples for long records, 100→30 Hz resample and original extraction batch ordering/size 64. Short records have mask [true,false], back features zero. Reconstructed mean must numerically match the archived cache; report exactness/tolerance. Baseline logits must remain unchanged.
3. A1 adds a 1024→8→GELU→64 residual branch at the original wrist fusion location, fed valid two-window mean. Parameter-free LayerNorm(1024, eps=1e-5), last Linear weight/bias zero. Branch active only when both windows and the wrist/activity are valid. Original residual path remains unchanged. Train from the original STR/WSSL initialization recipe; do not initialize from any other candidate.
4. A2 independently uses identical branch, normalization, mask, initialization seed and training recipe; input is back minus front. Compare A2–baseline and A2–A1. No dimension search.
5. B1 independently changes only the local STR Acc input to raw post-trim Acc; Gyro, lengths, SSL features and architecture fixed. Refit normalization using original training subjects only. No combinations.
6. Aggregate current frozen WSSL validation predictions by independent subject. Audit DD subtype labels and data quality; no exclusion or validation-error-based training sampling. Groups and recalls are descriptive repeated-development evidence.
7. C1 is conditional: require verified source subtype labels, at least 10 independent DD subjects per included subtype, and a subtype recall difference supported by subject-level bootstrap plus BH-adjusted exploratory test. If labels unreliable or evidence absent, skip. If eligible, register exact label set and one auxiliary-loss weight 0.1 before training. DD-only training loss, PD/DD primary output unchanged, no inference labels. No loss-weight search.
8. Review any retained candidate's epoch/split sensitivity. No stopping-rule change without both baseline and candidate rerun. Repeated subjects are not external validation.

## Tests and retention criteria (fixed before results)

Each implementation passes shape, mask, gradients, checkpoint reload and one-batch training checks before 45-run evaluation. Zero-initialized branches must preserve loaded baseline logits; hidden-layer gradient is initially zero as expected and becomes nonzero after output-layer update. No smoke performance selection.

Primary BA retention: positive mean paired delta, at least 10/15 improved splits and split-bootstrap 95% lower bound >0. AUROC must also have positive mean delta and at least 10/15 improved splits; its CI is reported, not an additional significance requirement. Macro-F1 cannot decrease on average. Neither PD nor DD mean recall may decrease by more than 0.01 absolute. Within-split seed SD of BA/AUROC cannot exceed baseline by more than 25%; use floor 0.001 for near-zero SD. All conditions required to retain; otherwise reject or label insufficient where the full matrix is incomplete. A2 must additionally beat A1 in mean BA and AUROC to attribute a gain to dynamics rather than capacity.

Average the three seed metrics within each split first. Paired differences use 15 aligned splits; report improvement count, 10,000 paired split-bootstrap percentile CI, Cohen dz, and descriptive split/seed variance. Wilcoxon and BH-FDR are supplementary across the two primary metrics per completed comparison. Splits overlap, so bootstrap CIs/p-values quantify repeated development robustness and are not independent population inference. Subjects, pairs and seed runs do not inflate N. Also report per-seed mean SD and within-split prediction agreement.

Publish source/configuration/reports/small aggregate results only. Datasets, per-subject predictions, features/caches and model weights remain on the server. Each item ends with files/tests/metrics/deltas/decision and a Git tag.
