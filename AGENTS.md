# Server research project instructions

## Git publication (user instruction, 2026-10-01)
- Continue development in /home/zyt/deep_final and /home/zyt/MFAM.
- After each authorized code change, verify it, record a meaningful Git commit and push to https://github.com/useanything0520-ux/pureDeep.git. Verify local HEAD equals remote main; do not report upload complete before this succeeds.
- Use /home/zyt/pureDeep-github-upload as the independent publishing mirror. Do not replace the original MFAM remote, initialize Git over the source tree, or force-push.
- Preview and publish using /home/zyt/deep_final/tools/git_publish/publish_to_github.py. See /home/zyt/deep_final/GIT_VERSION_CONTROL.md.
- Include code, configuration, reports and small aggregate experimental results. Exclude datasets, checkpoints, weights, features, caches, per-subject prediction tables, credentials and large files. Augmented samples are not independent subjects.
- No automatic deletion. Missing previously published files require inspection and explicit handling of each path. Never use bulk deletion.
- If remote history changed, reconcile source and mirror before publishing. If network/authentication fails, preserve commits and report the exact remaining push step.
- Preserve frozen formal results and existing development/outer evidence boundaries. Repository publication does not authorize any new training or experimental search.
