# PADS PD/DD maintenance report — 2026-11-16

## Scope and result

This is a stored-artifact maintenance audit, with **zero training runs, optimizer updates or new performance experiments**. The server and GitHub initially resolve to `8720b45fe82f0037b10ba779588c1bd6281be911`, tree `748bec911c629ae09a473caaceaf4f6967d578ee`. The upload mirror is clean. The unchanged publisher preview selects 25,461 files; all match the existing server source, mirror and publication manifest. No intervening public-source changes were found or overwritten.

The server instructions, project/mainline/conclusions READMEs and model index were read before maintenance. Work remains in `/home/zyt/deep_final`; retired `/home/zyt/MFAM` is not developed or resynchronized. Existing legacy Git files are verified in place through the publisher.

The executed server audit checks the registration's 109 protected historical files, all 48 registered checkpoint hashes, the registration's private-asset and public-artifact manifests, current-model pointer, registry lifecycle roles, historical publication manifest and complete selected source/mirror equality. Counts and results are recorded in `../analysis/maintenance_checks.json`; failures stop publication. These are hash/provenance checks, not a model-performance rerun or an independent statistical validation.

## Model and evidence boundaries

Frozen WSSL-STR remains **Retained Model**; STR-01 remains **Matched Baseline**; AI-WSSL-STR remains **Experimental Candidate**. `conclusions/CURRENT_MODEL.json` is preserved byte-for-byte, SHA-256 `a30c1262c7e37793050a73bfd1c0ba6f8f68c0cc020a21cda1434a0f5f05a820`. Its historical metrics are not relabeled as Certified L1 metrics.

AI registration links 45 old-L1 checkpoint identities and only3 Certified L1 identities (context0/inner0, seeds42/43/44). The historical45 runs cannot stand in for the incomplete Certified15-split comparison. All3 Certified pilot DD Recall deltas are below −0.01; the observed PD/DD tradeoff remains a limitation. No remaining42 runs, extra seeds, splits, thresholds, hyperparameters or model promotion are started. Certified rebuilding did not establish improvement over historical WSSL; the old/new difference is not proof of label leakage or a quantified inflation effect.

## Connectivity and publication

SSH password authentication succeeded without writing credentials to code, reports or Git. The server's existing Git proxy targeted an inactive session-local loopback port. A temporary SSH remote dynamic forward restored anonymous GitHub fetch/readback using the existing proxy configuration; no persistent Git/proxy configuration was changed. Direct server HTTPS timed out; authenticated server HTTPS push dry-run returned an invalid-token error. No credential value is recorded here.

The existing `/home/zyt/deep_final/tools/git_publish/publish_to_github.py` is unchanged and owns selection, filtering, manifest generation and tracked-file audits. The new wrapper permits only additive files under this maintenance artifact and the regenerated publication manifest. It stops before push to publish the identical audited tree through the authorized `chouytong` GitHub connector with a non-force expected-head check. Remote changes invalidate that expected head and require reconciliation.

Only scripts, Markdown and safe aggregate JSON checks are added. Data, signals, weights, HarNet caches, IDs, individual predictions and private certificates are not newly uploaded. Private existing assets remain under `restricted_manifests/` and are checked on the server. No files are deleted or historical reports overwritten.

Publication is complete only when the separate final receipt confirms the remote parent/tree, all selected source/mirror/manifest hashes, unchanged protected assets, mirror HEAD equal to remote main and a clean worktree. This report describes the audited snapshot; its existence alone is not a publication receipt.
