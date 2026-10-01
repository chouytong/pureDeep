# Phase-3D: Advanced SSL Adaptation

Status: complete and audited, 2026-10-01 Asia/Shanghai. PURE-DEEP; DEVELOPMENT-ONLY; NO OUTER-BASED MODEL SELECTION.

## Scope, protocol and evidence boundary

The user explicitly authorized a new bounded Phase-3D after Phase-3C CASE D. The frozen `PHASE3D_PROTOCOL.md` specifies the same 15 fixed subject-level inner-development splits × seeds 42/43/44, original STR architecture, activity/wrist order, bilateral and structured residual paths, balanced CE, train-only normalization, original AdamW/cosine/BA early stopping and 0.5 threshold. The only Phase-3D supervised variants are one 64D bottleneck adapter before the existing SSL projection (E1), one 128→32→64 gated wrist fusion (E2), and an activity-specific scalar gate (E3) only if E2 passes the retention gate. E4 is one fixed train-only PADS-domain masked HarNet layer5 adaptation, conditional on no E1–E3 retention. H1, handcrafted features, teacher/student, outer performance, backbone replacement and ordinary augmentation/loss/sampling searches were excluded. No formal STR, Phase-3B/C or FOE-01 artifact was overwritten. Reusing the same development cohort for a new sequential phase limits confirmatory strength even with preregistration and split-level inference.

The references are frozen Phase-3B WSSL-STR and read-only Phase-3C last-block adaptation. WSSL encoder cache, manifest and fixed split hashes were verified before training. New models and outputs are isolated under this directory. The historical locked FOE-01 outer was not used for selection or validation in this phase.

## Smoke and implementation checks

E1/E2 one-batch smoke runs and checkpoint reload passed. With the same STR backbone/input, initial logits of frozen WSSL, adapter, gate and hypothetical activity gate were bitwise identical (maximum difference 0). Loading a Phase-3B WSSL checkpoint into the zero-initialized adapter also gave exactly matching logits. The frozen HarNet feature cache required no gradient. New trainable parameter increments over the Phase-3B WSSL classifier (143,172) were adapter +132,160 (total 275,332), gate +6,240 (total 149,412), potential activity gate +11; the external frozen HarNet extractor has 10,457,408 parameters and is not counted in these classifier numbers. The E1/E2 training source hashes still match `formal_e1_e2_source_sha256.txt`.

E4's initial smoke strict cache-equality assertion stopped before adaptation. Fold-local window batching causes rare floating convolution differences from the Phase-3B global batching: among 5,740 train windows, Phase-3C layer4 comparison had mean absolute difference 5.14e-7, p99 0 and max 0.002111; final1024 subject features had mean 4.56e-7, p99 0 and max 0.001776. The label-independent check now bounds all three summaries; preprocessing and subject mapping were unchanged. The rerun one-batch masked adaptation and one-batch unchanged WSSL-STR training passed with deterministic CUDA setup. See `DOMAIN_SMOKE_NOTE.md`; formal E4 source is frozen in `formal_domain_source_sha256.txt`.

## E1/E2 complete supervised comparisons

All 45 adapter and 45 gate inner runs completed; checkpoint SHA, matching validation ID/label, matching train-subject and train-only normalization hashes, and no outer test loader passed. The added adapter/gate weights changed from initialization in all 90 runs (`analysis/parameter_learning_audit.csv`). Three seeds were averaged within each of the 15 splits for the main comparison.

| Model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Mean within-split BA/AUROC seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original STR-01 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | 0.0300 / 0.0323 |
| Phase-3B frozen WSSL-STR | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 | 0.0216 / 0.0217 |
| Phase-3C last-block adapted | 0.7518 | 0.7244 | 0.7689 | 0.7128 | 0.7905 | 0.6583 | 0.0202 / 0.0222 |
| E1 64D adapter | 0.7452 | 0.7195 | 0.7643 | 0.7064 | 0.7815 | 0.6576 | 0.0164 / 0.0268 |
| E2 gated fusion | 0.7520 | 0.7178 | 0.7581 | 0.7092 | 0.8004 | 0.6351 | 0.0201 / 0.0281 |

E1 adapter−frozen BA was −0.0001 (9/15 positive; 95% split-bootstrap CI [−0.0053,+0.0044]; paired dz −0.009; six-primary-test BH q=0.9780). AUROC was +0.0048 (10/15; CI [−0.0011,+0.0100]; dz 0.422; q=0.3616). Macro-F1 changed −0.0001, DD Recall +0.0007 and PD Recall −0.0009; each secondary interval crosses zero. Mean BA seed SD fell 0.0216→0.0164, but AUROC seed SD rose 0.0217→0.0268. Mean three-seed probability Spearman changed 0.7734→0.7697 and 0.5-threshold disagreement 0.1780→0.1842. Adapter did not pass the prespecified BA/AUROC/seed gate. Versus Phase-3C **last-block** adaptation, adapter BA was −0.0049 (CI [−0.0107,+0.0010]) and AUROC −0.0046 (CI [−0.0112,+0.0017]); neither difference is stable. Full HarNet fine-tuning was neither requested nor run, so adapter cannot be claimed superior or inferior to *full* fine-tuning.

E2 gate−frozen BA was −0.0019 (4/15 positive; CI [−0.0055,+0.0024]; dz −0.227; q=0.5048), AUROC −0.0015 (7/15; CI [−0.0077,+0.0039]; dz −0.122; q=0.9780). Accuracy rose +0.0064, but DD Recall fell −0.0218 (CI [−0.0408,−0.0020]); Macro-F1 +0.0026 had a CI crossing zero. Mean AUROC seed SD rose 0.0217→0.0281; probability Spearman fell 0.7734→0.7555 and threshold disagreement rose 0.1780→0.1853. The gate learned nonzero weights but did not show beneficial SSL contribution control. E2 failed the joint gate, so **E3 activity-specific gate was not run** under its pre-registered condition. It has no measured positive or negative outcome.

## E4 train-only masked PADS-domain SSL adaptation

E1 and E2 both failed the preset retention gate, so E3 was skipped and E4's pre-registered condition was met. For each of the 15 fixed splits × three seeds, the same official pretrained HarNet10 was adapted on *only that split's inner-training raw Acc windows* with 15% contiguous temporal masking and a temporary 1024→900 reconstruction decoder. Layer1–4 were frozen; layer5 trained at 1e-5 for exactly five epochs; BN running statistics were fixed. The decoder was discarded. Only after adaptation were unmasked features inferred for inner-training and validation subjects, and the unchanged WSSL-STR classifier trained with its original supervised recipe. The raw subject access audit shows 0 outer raw subjects and 0 validation windows in the adaptation loss in all 45 units (`analysis/domain_integrity.csv`). Feature/checkpoint SHA, frozen BN, subject IDs, train-only normalization and complete 45 supervised outputs all passed. `formal_domain_source_sha256.txt` verified all formal E4 training source. E4's transient trainable parameters were 2,623,488 layer5 + 922,500 decoder = 3,545,988; at supervised fitting, only the original 143,172 classifier-side parameters trained.

Masked reconstruction loss fell in 45/45 units, from mean epoch-1 0.02661 to epoch-5 0.01882; layer5 parameters changed with mean L2 norm 1.469 (`analysis/domain_pretext_summary.json`). Thus the pretext adaptation did learn its own objective. It did **not** improve downstream disease classification:

| Model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Within-split BA/AUROC seed SD | Seed-pair score Spearman | 0.5 prediction disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen WSSL-STR | 0.7456 | 0.7196 | 0.7596 | 0.7066 | 0.7823 | 0.6569 | 0.0216 / 0.0217 | 0.7734 | 0.1780 |
| E4 masked-domain WSSL-STR | 0.7362 | 0.7127 | 0.7572 | 0.6971 | 0.7696 | 0.6557 | 0.0208 / 0.0271 | 0.7499 | 0.2205 |

E4−frozen BA was −0.0070 (5/15 split improvements; 95% split-bootstrap CI [−0.0149,+0.0003]; paired dz −0.450; six-primary-test BH q=0.3616), AUROC −0.0023 (8/15; [−0.0080,+0.0028]; dz −0.208; q=0.9780), Macro-F1 −0.0095 (CI [−0.0204,−0.0007]), Accuracy −0.0094, PD Recall −0.0127 and DD Recall −0.0012. AUROC seed SD and threshold disagreement worsened. E4 fails BA, AUROC, Macro-F1, PD Recall and AUROC-seed-stability gates. The lower masked reconstruction loss is not evidence of better PD/DD transfer. One bounded layer5/decoder objective does not rule out every possible domain-adaptation method, but it does not support expanding this search on the same cohort.

## Final decision: retain frozen WSSL-STR

**NO PHASE-3D candidate is retained.** E1 adapter did not improve BA and had a higher AUROC seed SD; E2 gated fusion traded DD recognition for PD recognition and failed both primary metrics; E3 activity-conditioned fusion was **not run** because its prerequisite E2 failed; E4 masked train-only domain adaptation improved its reconstruction objective but not classification. Within these registered candidates, frozen SSL transfer appears difficult to improve and remains the best *retained* single-model pure-deep development configuration: STR-01 + pretrained-**frozen** HarNet10 final 1024D wrist residual, original balanced CE/training recipe, fixed 0.5 decision rule. Its seed-first 15-split metrics are Accuracy **0.7456**, BA **0.7196**, AUROC **0.7596**, Macro-F1 **0.7066**, PD Recall **0.7823**, DD Recall **0.6569**. The full system has 10,600,580 parameters including the 10,457,408-parameter frozen encoder; 143,172 classifier-side parameters are trained. The three-seed ensemble AUROC 0.7816 remains an inference-only reference, not the primary single model.

This phase does **not** establish that frozen SSL is globally optimal. It also cannot answer whether adapter tuning beats *full* HarNet fine-tuning: full fine-tuning was not run, and the measured comparison is only against Phase-3C **last-block** adaptation. Adapter−last-block BA −0.0049 and AUROC −0.0046 both had CIs crossing zero. The gated result provides no evidence that fixed residual addition is the current limiting factor; activity-conditioned fusion has no measured result because the gate prerequisite failed. The current masked PADS-domain adaptation did not further improve BA/AUROC. Stop adapter-width, gate, activity-gate and masked-domain-SSL searches on these repeated development splits. Consolidate the paper evidence and plan a genuinely independent cohort/context for validation; do not use the old locked FOE-01 outer set to tune or claim independent validation of Phase-3B/D.

The fixed 15 inner splits reuse subjects and contexts; 10,000 split-bootstrap CIs, Wilcoxon p and BH q summarize this repeated development design, not independent external-cohort replication. Three seeds were averaged inside each split; subjects, wrists, activities, raw windows, masks and seed-runs were not counted as independent sample units. The Phase-3D hypothesis follows several earlier rounds on the same cohort, so residual selection optimism is possible even though this phase's configurations and gates were frozen before validation inspection. No Phase-3D candidate was evaluated on an outer set.

## Reproduction ledger

- Frozen design and source: `PHASE3D_PROTOCOL.md`, `manifest.json`, `formal_e1_e2_source_sha256.txt`, `formal_domain_source_sha256.txt`.
- E1/E2 implementation/smoke: `scripts/phase3d_hooks.py`, `scripts/run_inner.py`, `scripts/test_consistency.py`, `analysis/consistency.json`, `smoke_adapter.log`, `smoke_gate.log`, `analysis/parameter_learning_audit.csv`.
- E1/E2/E4 matched statistics, seed agreement and gates: `scripts/analyze_models.py`, `analysis/model_summary.csv`, `analysis/model_paired.csv`, `analysis/model_seed_prediction_pairs.csv`, `analysis/model_gates.json`, `final_model_analysis.log`.
- E4 raw/pretext/classifier pipeline and integrity: `scripts/prepare_domain_raw.py`, `scripts/run_domain.py`, `scripts/audit_domain.py`, `DOMAIN_SMOKE_NOTE.md`, `analysis/domain_integrity.csv`, `analysis/domain_pretext_summary.json`, `domain_audit.log`.
- Parameter counts, including frozen encoder and temporary E4 decoder: `analysis/parameter_counts.json`.
