# PADS PD/DD maintenance audit — 2026-11-16

This maintenance pass performs **zero new experiments or training**. It checks the server against the last verified public snapshot, preserves user changes, verifies stored provenance and protected assets, and publishes only this audit's scripts, report and aggregate checks.

Execution is restricted to `/home/zyt/deep_final` on the server using `/home/zyt/envs/mfam_analysis/bin/python`. No local model execution, new seeds/splits, normalization refits, signal processing, threshold searches or optimizer updates are authorized by this maintenance pass.

- [Maintenance report](reports/MAINTENANCE_REPORT.md)
- [Executed aggregate checks](analysis/maintenance_checks.json)
- [Audit script](scripts/audit.py), [unchanged-publisher transport wrapper](scripts/publish.py), [remote readback verifier](scripts/verify.py)

The separate server receipt `/home/zyt/pureDeep-maintenance-audit-publication-receipt-20261116.json` is created only after successful GitHub commit/tree readback, source/manifest/mirror equality and a clean mirror check. It remains outside the committed snapshot to avoid a self-referential commit receipt.
