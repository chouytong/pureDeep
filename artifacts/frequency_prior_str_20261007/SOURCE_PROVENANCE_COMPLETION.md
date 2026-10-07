# F0 whole-tree source provenance: correction and resolution

2026-10-07. **PASS, all45 exact-current-source matched F0 runs available; no redundant retraining.**

The initial F0 audit checked current per-file hashes, actual outputs, configurations, normalization and previously reproduced weights. It omitted the direct comparison of the checkpoint's historical **whole-tree source hash**. It is incorrect to describe historical whole-tree source provenance as missing: the hash is saved in `development_protocol.json` and checkpoint provenance.

- Original2026-09-21 formal STR source tree: `4fd74c01a05dbb743dea943264fbe9d52ced140379e0f23c1bb945452fefbc4c`.
- Current source tree: `fae112dcf6d05de5f94bdbe0c7e2b3748d79d343200f441762f4dd8db82f6617`.
- All45 Phase2 STR baseline-reproduction checkpoint source hashes: **the exact current `fae112...` hash**.

The source-tree manifest covers source/config/scripts/tests and selected documentation/environment files. A tree-hash change alone does not establish a changed active STR computation. We do not infer which historical files caused the difference without an archived file-level manifest.

The user requested a matched F0 if strict reuse failed. Before launching redundant training, we found and audited the existing complete matched F0 from Phase2: all45current-source hashes, model/data/training/evaluation/loss/nested/self-supervised config sections, selected model-state tensors, best epochs, normalization tensors, validation IDs/labels/decisions/probabilities match the original formal STR exactly. Thus this current-source matched F0 is a valid reusable baseline. The F0 numerical values in the frozen analysis are exactly its values; using the identical original cached predictions does not change any comparison.

This check occurred during final closure, after the first F1/F2 analysis. We disclose this sequencing omission rather than claiming it was part of the initial preformal source check. No condition, dimension, band, seed, threshold, stopping/retention criterion or statistical method changed after results. No additional model or retraining was started. F0 formal and Phase2 matched checkpoints remain unchanged. Source consistency does not by itself prove clinical or outer generalization.

Evidence: `analysis/matched_f0_provenance_audit.json`, `analysis/matched_f0_provenance_45run_audit.csv`, and reproducible `scripts/audit_matched_f0.py`. This note supersedes the initial source-provenance wording in F0_AUDIT_REPORT.md, not its verified45state/output/15normalization results.
