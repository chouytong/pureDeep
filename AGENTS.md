# Server mainline instructions

## Canonical root (user instruction, 2026-11-11)
- Develop the PADS PD-vs-DD mainline only in /home/zyt/deep_final/mainline. Keep its data under /home/zyt/deep_final/data and current conclusions under /home/zyt/deep_final/conclusions.
- /home/zyt/MFAM is retired historical material. Do not develop there or resynchronize its contents. Do not import unrelated paper replications into the active mainline.
- Current retained model: frozen WSSL-STR; matched baseline: STR-01. Read README.md and conclusions/README.md first. Do not treat old V8 root manifests as current-model selection.
- Preserve original frozen source/config/checkpoint/result paths, provenance hashes, negative experiments and outer boundaries. Migration is not permission for new training/search/outer evaluation.
- No automatic deletion, bulk deletion or force push. Preserve old MFAM files and original data; cleanup is a separate user action.

## Git publication
- Preview and publish using /home/zyt/deep_final/tools/git_publish/publish_to_github.py; retain all filtering and audits. New source is deep_final only. Already-published MFAM/ files remain an immutable legacy snapshot without resync or deletion.
- Use /home/zyt/pureDeep-github-upload as the independent mirror for https://github.com/chouytong/pureDeep.git. Do not initialize Git over the source tree or replace the old MFAM remote.
- Verify code, commit meaningfully and publish only code/config/report/aggregate evidence; exclude data, weights/checkpoints, individual clinical/prediction tables, features/caches, logs and credentials.
- If server HTTPS authentication fails, the already-authorized GitHub connector may publish the identical audited tree; preserve an unpublished local commit before aligning mirror HEAD. Never force-update remote main. Verify local HEAD equals remote main and the worktree is clean.
- Inspect and reconcile remote changes before publishing. Preserve commits and report remaining work if no authorized authentication is available.
