# PADS V3 Nested Cross-Validation Protocol

## Scope

V3 keeps the V2 data representation and SubjectMFAM architecture unchanged. It
only adds frozen subject splits, fold-specific normalization, provenance, and a
dry-run-only orchestration entry point. Labels remain `PD=0`, `DD=1`; Healthy is
excluded.

## Frozen split

The canonical split is:

`splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json`

It is a deterministic 5-fold outer by 3-fold inner split stratified by PD,
Other Movement Disorders, Essential Tremor, Atypical Parkinsonism, and Multiple
Sclerosis. The five strata are used only for splitting; the learning task stays
binary.

Regeneration refuses to overwrite a non-empty target directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/zyt/envs/mfam/bin/python \
  scripts/generate_nested_cv_splits.py
```

## Leakage boundary

For an inner fold, normalization is fitted only on the explicit inner-train
subject IDs. Inner validation and outer test are passed as distinct ID sets and
cannot overlap. For an outer-final fold, normalization is fitted on all
outer-train subjects. The outer-test loader is not created by dry-run.

Threshold selection is defined as DD-probability optimization on pooled inner
out-of-fold validation predictions with balanced accuracy fixed in the config.
The V2 threshold is not reusable. Final outer training uses the deterministic
median of the three one-based inner best epochs and does not early-stop on the
outer test.

## Dry run

The current runner intentionally requires `--dry-run`; it contains no training
loop and no outer-test evaluation:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/zyt/envs/mfam/bin/python \
  scripts/run_nested_cv.py --dry-run --outer-fold 0 --inner-fold 0 \
  --mode inner --forward-smoke
```

The output records subject IDs, diagnosis counts, DataLoader sizes,
normalization values and hash, split hash, runtime provenance, and planned
checkpoint/prediction directories.

## Checkpoint compatibility

V3 checkpoints embed strict provenance and receive a post-save
`.provenance.json` sidecar containing the checkpoint SHA-256. Evaluation and
inference fail fast on any config, split, manifest, preprocessing, source,
normalization, label, activity, wrist, sensor, or channel mismatch.

Historical V2 checkpoints are accepted only with both explicit options:

```text
--allow-legacy-checkpoint \
--legacy-provenance outputs/pads_classification/v2_frozen_provenance/provenance.json
```

The sidecar must contain the exact checkpoint SHA-256 and match the runtime V2
identity. Existing V2 checkpoint bytes are never rewritten.
