# Original-engine smoke and final pre-result freeze

2026-10-07. Both F1/F2 PASS one original inner-training epoch with one train and one validation batch, seed42/context0/inner0. Original AdamW/balancedCE/normalization/scheduler/FP32/clip/BA-selection/checkpoint writer and final reload used. Audits verify total78,005/77,998params, correct train/validation IDs and labels, normalization exact to original STR, saved predictions and metrics consistent, fixed.5/argmax decision, no outer loader, no EMA/WSSL. Smoke scores are not scientific results or a basis for changing configuration. No smoke checkpoint is reused for formal initialization.

Protocol, all14independent Python scripts (including analysis), F0/source lock and preformal correctness results frozen in analysis/study_lock.json. F0 audit/filter/model tests and known-answer metric/BH/import/syntax checks PASS. A preliminary manual AUROC testcase expected.5 incorrectly; it was corrected to.75 (3of4correctpairs) and repeated before freezing. No production metric/model/recipe changed.

Formal execution requires this protocol/source/smoke release pushed to GitHub before results, then all45F1+45F2runs. No formal candidate has completed at the time of this document. Stats/design are fixed in PROTOCOL.md; no band/hidden/recipe/threshold search or outcome-adaptive rule.
