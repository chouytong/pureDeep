# Certified independent L1 rebuild: Phase 9 Level 2

All signal processing, normalization, training, evaluation and timing run on the authorized server. Scientific outputs are under `/home/zyt/deep_final/artifacts/l1_precision_rebuild_phase9_level2_20261112`. Start with [the final report](reports/FINAL_REPORT.md), [protocol](PROTOCOL.md) and [amendment](PROTOCOL_AMENDMENT.md). Legacy mainline and the current-model pointer are preserved. Frozen WSSL-STR freezes HarNet10; its classifier side remains trainable.

## Environment and execution

Existing `/home/zyt/envs/mfam_analysis/bin/python`, Python 3.10, torch 2.8.0+cu128, NumPy2.2.6/SciPy1.15.3, RTX5090Dv2; no installation or global environment modification. Commands below are actual stage interfaces, in mandatory order. The original baseline commit is `4da80813a98586acf23a89507359e4ca93689289`. Scripts import read-only project mainline, not a separately installed model approximation.

```bash
cd /home/zyt/deep_final/artifacts/l1_precision_rebuild_phase9_level2_20261112
PY=/home/zyt/envs/mfam_analysis/bin/python
L2=/home/zyt/deep_final/artifacts/l1_precision_rebuild_phase9_level2_20261112
DATA_VERSION=pads_l1_lambda50_certified_v1
DATA_MANIFEST="$L2/restricted_manifests/manifests/$DATA_VERSION.json"
NORM_INDEX="$L2/restricted_manifests/normalization/$DATA_VERSION/index.json"

"$PY" -B scripts/run.py audit
"$PY" -B scripts/run.py solver-test
"$PY" -B scripts/run.py pilot
"$PY" -B scripts/run.py build-data
"$PY" -B scripts/run.py solver-test --verify-data
"$PY" -B scripts/run.py solver-test --quantization-impact
"$PY" -B scripts/run.py fit-normalization --data-version "$DATA_VERSION" --manifest "$DATA_MANIFEST" --output-dir "$L2/restricted_manifests/normalization/$DATA_VERSION"
"$PY" -B scripts/run.py train-wssl --data-version "$DATA_VERSION" --manifest "$DATA_MANIFEST" --normalization "$NORM_INDEX" --split all --seed all --output-dir "$L2/restricted_manifests/checkpoints/wssl"
"$PY" -B scripts/predictions_l2.py wssl
"$PY" -B scripts/run.py train-str --data-version "$DATA_VERSION" --manifest "$DATA_MANIFEST" --normalization "$NORM_INDEX" --split all --seed all --output-dir "$L2/restricted_manifests/checkpoints/str"
"$PY" -B scripts/run.py analyze
"$PY" -B scripts/run.py integrity-check
"$PY" -B scripts/run.py publish-dry-run
```

Training stages inject explicit `--model-type wssl` or `--model-type str` into `train_l2.py`; the direct training script also requires all seven explicit arguments. `--split 0:0` selects a canonical context:inner, `--seed 42` selects one registered seed. The formal execution uses `all/all`, exactly 45 WSSL then 45 STR; no extra fits are needed to reproduce the published analyses. A synthetic-only optimizer step per model is separately recorded and is not a formal performance candidate. The WSSL historical paired comparison report is archived by `predictions_l2.py wssl` before STR begins.

## Gates, resume and accounting

Do not run stages concurrently against the shared CPU ledger. Upstream certificate, version, source, normalization and split gates fail closed. The CPU wrapper charges solver-test (including independent readback), pilot and build-data parent/child user+system CPU; limit21600 seconds. Training/norm/analysis CPU and wall time are separately logged.

`build-data` atomically commits one complete hash-certified wrist at a time. A partial export resumes only records whose raw/source/config/certificate/output hashes match. A completed version is hash-verified and reused without rewriting signals. Orphaned uncommitted files cause BLOCKED rather than overwrite or deletion. Normalization reuses only valid matching split/version files. Training resumes completed runs only after source/data/norm/checkpoint/prediction hash checks; an unfinished formal run is preserved and explicitly BLOCKED, avoiding an unrecorded retry or excess budget. This is completed-run recovery, not a claim of exact mid-epoch RNG restoration. Stage attempts and failures are retained. Locked-source audit/solver freeze files are evidence from their execution time; their historical `results_seen=false` fields do not describe the later completed experiment.

Report-only recovery, if the analysis log shows the completed statistics and90 timings followed by a missing matplotlib import:

```bash
"$PY" -B scripts/run.py analyze --resume-reports
```

This recovery uses the already installed `/home/zyt/envs/patch_report_deps_20261008` only in that reporting process. It does not install or modify packages, repeat model fitting, or re-run completed statistical analyses. The initial failed attempt and source copies are preserved; see `audit/report_recovery.json`.

## Outputs and privacy

Public: scripts/configs/tests, protocol/amendment, 12 reports, safe aggregated CSV/JSON and one aggregate figure. Full axis certificates, raw/new signals, patient IDs, prediction arrays, fitted normalization and checkpoints live under `restricted_manifests/`, which is explicitly excluded by the unchanged selective publisher. This directory implements the private area recommended in the task. No private arrays are needed in GitHub.

`analysis/final_integrity.json` closes numerical/data/training correctness independently of `PERFORMANCE_PROMOTION`. The public source manifest hashes the public scientific snapshot; a private manifest hashes new scientific assets. The mutable operational CPU ledger is excluded from that private frozen hash catalog because the wrapper records the integrity/publication stages after their child exits.

Publication uses the original `/home/zyt/deep_final/tools/git_publish/publish_to_github.py` collection, size/privacy/tracked-file checks and commit workflow. `scripts/publication_transport.py --prepare` intercepts only transport after its audit and local commit; the connected repository account publishes the exact immutable Git tree by non-force compare-and-swap. `scripts/postpublication_verify.py REMOTE_SHA` verifies GitHub main/tree/parent and server/source/mirror hashes, preserving the transport-local commit in an archive ref. The final receipt is outside this scientific snapshot: `/home/zyt/pureDeep-l1level2-publication-receipt-20261112.json`. Dry-run is not publication, and pending publication is not described as complete.

No extra model, loss, threshold, lambda, HarNet boundary, gravity/coordinate/timestamp change or outer evaluation is started after this task.
