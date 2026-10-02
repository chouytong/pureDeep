# Git version control and GitHub publication

Established: 2026-10-01 (Asia/Shanghai), explicitly requested by the project owner.

## Repository mapping

- GitHub: https://github.com/chouytong/pureDeep.git, branch main.
- Development/archive source: /home/zyt/deep_final, published at repository root.
- Original source: /home/zyt/MFAM, published under MFAM/; its existing Git remote is preserved.
- Independent Git publishing mirror: /home/zyt/pureDeep-github-upload.
- Publisher: /home/zyt/deep_final/tools/git_publish/publish_to_github.py.

## After every code change

Complete relevant checks, then run on the server:

```bash
python3 /home/zyt/deep_final/tools/git_publish/publish_to_github.py
python3 /home/zyt/deep_final/tools/git_publish/publish_to_github.py --apply --message "Describe the verified change"
git -C /home/zyt/pureDeep-github-upload status --short
git -C /home/zyt/pureDeep-github-upload rev-parse HEAD
git -C /home/zyt/pureDeep-github-upload ls-remote origin refs/heads/main
```

The publisher discovers new eligible files, verifies SHA-256 copies, fetches remote history, stages a manifest, checks file size/content policy, commits, performs a push dry run, pushes and verifies matching HEADs. No automatic deletion or force push. A dirty mirror, unincorporated remote changes, missing previously tracked files or policy violations stop publication for inspection. No-change runs still verify/push existing commits.

## Published content and exclusions

Code, configs, Markdown reports, aggregate metrics, small diagnostic tables/figures and canonical split protocol metadata are included. PUBLICATION_MANIFEST.json records source paths, selected file hashes and exclusion counts. Historical/rejected model code is retained as an archive; its presence does not reopen experiments.

Raw/processed datasets, fitted normalization files containing subject membership, checkpoints/model weights, feature arrays, individual prediction/clinical tables, training logs, caches, vendored dependencies, quarantined historical outer artifacts and credentials are excluded. Maximum published file size is 5 MiB. Formal readable reports and aggregate outer conclusions remain archived; outer data/predictions are not published or reused for development.

The repository is an archival code/report snapshot, not a complete sensor dataset or model-weight distribution. Existing manifests may reference omitted server artifacts; those references document provenance and do not promise that all files are in Git. Reproduction requires separately obtaining the permitted dataset/external pretrained weights and installing the documented dependencies. Server-specific paths are preserved to avoid changing formal experiment behavior.

## Connection and credential boundary

Git runs on the server through the existing Mac SSH reverse SOCKS tunnel at socks5h://127.0.0.1:18743. The Mac must remain connected. Git uses the existing external askpass credential store; secrets are never embedded in repository URLs, scripts, reports or commits. Do not terminate unrelated tunnels.

If authentication expires or permission is unavailable, retain the verified local commit and restore authentication before pushing. If GitHub contains new commits, inspect and merge/reconcile them first, then align the source tree with the agreed content before rerunning the publisher. Do not overwrite remote edits with an unchecked source sync.

Mirror-local pre-commit/pre-push policy hooks are installed separately and are not automatically installed by cloning. Run the publisher audit explicitly on a new mirror.

## Account migration — 2026-10-02

At the owner's request, the active public repository is now https://github.com/chouytong/pureDeep. Existing Git history is retained; the previous-account remote remains an archival reference. New-account credentials are stored separately under /home/zyt/github-puredeep-chouytong-credentials, outside source/mirror. Publication filtering and frozen experiment results are unchanged. Verify unauthenticated access after pushing.
