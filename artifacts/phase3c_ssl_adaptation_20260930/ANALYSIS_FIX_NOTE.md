# Phase-3C E2 post-training analysis fixes

The E2 crossfit training code and artifacts were unchanged. After all 135 crossfit models had completed, two analysis-only defects were corrected before interpreting E2 results:

1. `audit_oof.py` asserted 405 rather than 135 crossfit models (15 target inner splits x 3 model seeds x 3 OOF folds). Its preceding subject-level checks had passed. The corrected audit passed 135/135 and wrote `analysis/e2_oof_integrity.csv`.
2. `analyze_e2_threshold.py` had the ensemble score aggregation block one indent too shallow, so it emitted 5 rather than 15 ensemble split rows. The corrected script aggregates inside every inner fold; it now emits 15 split rows per threshold rule and passed pair matching.

The pre-fix sources are preserved with `_pre_fix.py` suffix. `remaining_source_sha256.txt` remains the frozen source ledger for the training process; its pre-fix analysis entry can be matched to `analyze_e2_threshold_pre_fix.py`. Final corrected analysis source hashes are recorded separately. Neither correction changes model training, OOF probabilities, Phase-3B validation predictions, or selection rules.
