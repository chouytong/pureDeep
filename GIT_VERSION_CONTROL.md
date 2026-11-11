# GitHub publication after mainline consolidation

Canonical source is `/home/zyt/deep_final` only; active code is `mainline/`, private data is `data/`, conclusions are `conclusions/`. `/home/zyt/MFAM` is not a publishing source anymore.

The independent mirror is `/home/zyt/pureDeep-github-upload`; repository is https://github.com/chouytong/pureDeep, branch main. Use `tools/git_publish/publish_to_github.py` to preview, filter, checksum, audit and prepare/commit the snapshot. Never use force push or remove files automatically.

The publisher retains already-published `MFAM/` files directly from the mirror only if their hashes match the previous publication manifest. This preserves history without importing new old-project code or requiring deletion. It must not read MFAM as an active source. Historical source paths inside reports/configs remain provenance, not current runtime dependencies.

Datasets, private raw/processed signals, checkpoint/model weights, individual predictions/clinical tables, fitted normalization with subject membership, caches/features and logs stay on the server. Data and result symlinks are excluded. Code, configs, reports, aggregate metrics and hashes are published.

The server HTTPS token was rejected in the 2026-11-11 sync. Existing authorized GitHub connector access for chouytong can publish the exact audited Git tree. Preserve any unpublished server commit on a local archive ref, then align the mirror only after proving identical trees and remote main. Connected-account access does not authorize changing data, recipe or experiments.

Network access can use a temporary Mac SSH reverse SOCKS tunnel at 127.0.0.1:18743; do not terminate unrelated tunnels or change permanent proxy/credential configuration. Passwords/tokens never belong in source, URLs, reports or commits.
