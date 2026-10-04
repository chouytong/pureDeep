# Single existing-WSSL LR control — entry gate PASS

2026-10-04. No formal candidate training/result exists at this milestone. Original frozen WSSL stays retained. No outer performance/data loader, EMA, encoder adaptation or architecture change.

## Actual execution

- Existing current-model audit:15normalization refits/45frozen artifacts;390metadata/8580wrists/10920SSL input windows;45sampled checkpoint forward/mask/reload checks PASS. Scope and clinical/normalization boundaries remain in AUDIT_REPORT.md.
- Initialization check: seed42, original2e-4 constructor versus new independent-interface1e-4 config, all initial state tensors exact; logits exact on eight inner-training subjects, zero optimizer updates. Python/NumPy/CPU/CUDA RNG restored.
- Original engine candidate smoke: one optimizer batch, one validation batch, oneepoch, context0/inner0/seed42. Checkpoint load, fixed argmax, split/labels/normalization, onlyLR config diff, checkpoint hash/metrics and BA selection all PASS. Smoke performance is not evidence for retaining/rejecting1e-4 and is not included in the formal matrix.
- First smoke also passed; read-only code review identified a generic single-class-batch AUROC None comparison edge. Audit now treats two None values as consistent, and mismatched availability as failure. Prior smoke preserved at `smoke_lr_preformal_v1/...`, no deletion. The revised identical smoke passed; no forward, recipe, result-selection or gate changes.
- Independent statistical review confirms full45matrix/15seed-first bootstrap/BH/gate/error aggregation. Before formal results, freeze was broadened to all current foundation source (including actual metrics/loss modules); analysis asserts all eight candidate artifact hashes and adds best/last classification losses. No endpoint/gate/architecture changes.

## Formal fixed test

Protocol `LR_CONTROL_PROTOCOL.md`, hashed inputs/scripts `analysis/lr_control_lock.json`. One new LR1e-4 candidate, all45stages, source architecture/HarNet/normalization/optimizer/loss/batch/scheduler/threshold/stopping rules retained. Each configuration uses original ordinary BA selection independently. Reference uses45locked Phase3B outputs already exact-reproduced by ordinary EMA runs. Sequential launcher cannot choose a configuration or add a candidate.

Code/report/protocol/small aggregate audits are committed/pushed/tagged before formal execution. Checkpoints, raw logs, caches, signals and per-subject prediction/label tables are excluded. Results are development-only. Prior FOE01 and quarantined outer information are not used.

## Limit

Only one smoke unit was run; complete45candidate execution and performance analysis remain required. Local syntax/configuration/helper tests are supplemental, not substitutes for server entry or full-matrix evidence. No global/near-global optimum claim.
