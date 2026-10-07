# Frequency-Prior STR — F0 reuse audit

Date: 2026-10-07 (Asia/Shanghai). **PASS; reuse archived STR-01, no F0 retraining.**

Original STR-01 has 75,524 trainable parameters. Its formal 45 best checkpoints, source configuration, preprocessing contract files, manifests, source files, metric implementation, predictions and histories were hashed into the new independent lock. All 45 full validation forwards reproduce archived probabilities (maximum difference 1.11e-16) and predictions by subject ID. Every label, split and seed was verified. Fifteen actual inner-train normalization refits match the frozen normalization tensors in all three seeds exactly. Train-fold balanced CE weights agree.

Existing Phase-2 baseline retraining was independently rechecked: all 45 model-state tensors, best epochs and predictions match formal STR exactly. Full checkpoint container bytes differ because run/provenance fields differ; this is not a model-weight difference. We record current source hashes and exact behavioral/state compatibility; we do not invent a missing historical source hash or claim all present source bytes existed unchanged on 2026-09-21.

Seed-first means: Accuracy .730718; BA .697189; AUROC .717631; Macro-F1 .686068; PD Recall .777864; DD Recall .616515.

## Context and historical differences

The early handoff header calls STR the current overall best, but latest reports retain frozen WSSL-STR as the stronger overall development configuration. **This user-authorized experiment specifically selects original STR-01 as F0.** No WSSL/HarNet component is used. The original gyro-only rFFT experiment on V8-GN used log magnitudes, 128 interpolated bins and concat fusion; it was rejected (BA/AUROC −.0146/−.0163). It is not rerun. The new question concerns five fixed reconstructed time-domain bands in all six STR input channels with zero-initialized per-wrist residual injection and an explicit capacity control.

STR input is current processed/normalized Acc+Gyro, not untreated raw acceleration: full-length Acc L1 trend removal, trim48, original Gyro after trim, nominal100Hz, actual lengths976/2000. Bilateral left/right and all11activity identities remain unchanged. Source/model/config review found no current discrepancy requiring a formal-model edit. Unused inherited YAML frequency/encoder/MIL and OOF-threshold fields are not the STR forward/inner-decision implementation: STR spectral stream is disabled and development prediction uses fixed probability argmax, DD probability>.5, tie PD.

## Boundary

Only restricted inner-train/validation datasets were instantiated. No outer-final/test signal, outcome, prediction or performance was read; canonical partition membership and existing checkpoint provenance counts are protocol metadata. Original formal files were not modified. All fresh artifacts are in `artifacts/frequency_prior_str_20261007`. Dataset/checkpoint/cache/individual predictions stay server-only. Historical stopped routes remain stopped beyond this explicitly authorized bounded test.
