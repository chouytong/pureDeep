# Item 08 — conditional candidate review

Decision: **SKIP: no candidate passed the preregistered retention gate**. Items03/04/05/07 each completed 45 runs and were rejected. Existing frozen WSSL remains retained.

Files: scripts/finish_study.py, analysis/study_final_summary.csv, study_completion.json, FINAL_REPORT.md and this report. Checks: all four matrices complete; all decision files match 45 required runs; original split/cache/source/checkpoint/prediction/normalization hashes unchanged. No new model or inference runs required for this conditional resolution.

Selected epochs, split SD, seed SD, prediction agreement and paired differences were already reported for every tested variant. There is no retained candidate to freeze for an additional stopping-epoch/split sensitivity study. Original early-stopping rule remains unchanged. No train-internal rule replacement, baseline refit, extra checkpoint selection or hyperparameter search was performed. The condition is explicitly resolved as skipped, not presented as a successful candidate review. Same-subject repeated development results are not external validation.

The final metric/delta table is in FINAL_REPORT.md and analysis/study_final_summary.csv. All new candidates rejected; keep original WSSL. No follow-on experiment is automatically authorized.
