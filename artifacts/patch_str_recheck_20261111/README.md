# Patch-STR 2026-11-11 verification archive

This archive records verification of the completed 2026-10-08 fixed 200/100 Patch-STR study. It introduces no candidate model, training run, hyperparameter search or retention-rule change. The result remains **REJECT**.

## Server paths

- Main project: `/home/zyt/deep_final`
- STR production source: `/home/zyt/deep_final/foundation_validation`
- Frozen Patch experiment: `/home/zyt/deep_final/artifacts/patch_str_20261008`
- Patch implementation: `/home/zyt/deep_final/artifacts/patch_str_20261008/scripts`
- Formal report: `/home/zyt/deep_final/artifacts/patch_str_20261008/PATCH_STR_REPORT.md`
- Private candidate checkpoints and predictions: `/home/zyt/deep_final/artifacts/patch_str_20261008/runs`
- This verification archive: `/home/zyt/deep_final/artifacts/patch_str_recheck_20261111`
- Independent publication mirror: `/home/zyt/pureDeep-github-upload`

## Reproduce verification on the server

Use the existing environment; no installation is needed. Run the phases sequentially with the same command, replacing `integrity` with `baseline`, `correctness`, `validation`, then `analysis`:

```bash
/home/zyt/envs/mfam_analysis/bin/python -B /home/zyt/deep_final/artifacts/patch_str_recheck_20261111/remote_recheck.py integrity
```

The verifier compares historical report outputs in memory instead of overwriting them. Baseline and validation use only development folds. Correctness includes fresh two-step gradient checks and audits the historical one-epoch smoke; it does not repeat formal training. The `run_recheck.py` file is the original local SSH dispatcher and requires an already-authenticated control socket; the command above runs directly on the server.

`VERIFICATION.json` records the original five-phase verification and its historical Git remote unavailability. It is preserved rather than retroactively relabeled. The later user-authorized GitHub sync is verified separately against remote `main`; its receipt is stored outside the repository at `/home/zyt/pureDeep-patch-str-recheck-sync-receipt-20261111.json` to avoid a self-referential commit hash.

Published content is limited to verification code, this report, README and aggregate verification evidence. Full verification logs stay in the server `logs/` directory and are excluded from Git. No raw data, clinical individual tables, private checkpoints, individual predictions, feature caches or credentials are included. The original frozen study is preserved.
