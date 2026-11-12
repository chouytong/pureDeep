# Certified L1 Activity Interaction: fixed split, three seeds

Server root: `/home/zyt/deep_final/artifacts/activity_interaction_certified_pilot3_20261112`.

Read [the final report](reports/FINAL_REPORT.md), [protocol](PROTOCOL.md), [correctness](reports/CORRECTNESS_AND_PROVENANCE.md) and [cost boundaries](reports/COST_AND_LATENCY.md). Phase8 interaction class is directly imported unchanged; each module uses its matching seed's Certified WSSL. This is one selected development split, not the full15×3 and not independent external validation. Baseline signal→token export is inference; only three new interaction modules are trained. All original WSSL classifier parameters and HarNet are frozen.

Actual execution, using the existing server environment without installations:

```bash
cd /home/zyt/deep_final/artifacts/activity_interaction_certified_pilot3_20261112
PY=/home/zyt/envs/mfam_analysis/bin/python
"$PY" -B scripts/run.py audit
"$PY" -B scripts/run.py export
"$PY" -B scripts/run.py correctness-test
"$PY" -B scripts/run.py train3
"$PY" -B scripts/run.py analyze
"$PY" -B scripts/run.py integrity
"$PY" -B scripts/run.py publish-dry-run
```

Stages freeze their source/protocol inputs and refuse silent overwrites. Stage attempts, failures and CPU/wall time persist under the private operational ledger/logs. Existing completed formal runs are not retrained by this entry; an attempted rerun blocks rather than exceeding the three-run budget. There is no claim of exact mid-epoch restart. Completed scientific outputs are hash-bound, and these three runs can be reused as the matching split of a separately approved full comparison if code/config/provenance remain identical.

`audit` checks only necessary frozen sources, the prescribed split and selected input hashes against the existing Certified chain. It does not repeat the whole project audit or decode excluded arrays. `export` reads only205 train and105 validation subjects: six role/seed packages,930 full-network subject instances,10230 actual activity tokens; it also checks token readout for every instance. No L1 or normalization fitting, no base training, no HarNet extraction, no legacy supervised tokens. Each query package matches the archived Certified probabilities/logits exactly. Cross-path tolerances are fixed before export; no relaxation based on results.

`correctness-test` uses exactly two synthetic optimizer updates separate from formal training, then formal models reset seed/initialization. It tests invalid mask rejection, masked nonfinite exclusion, frozen eval/no gradients/state hashes, upstream gradients after the zero-output first step, single baseline/STR/classifier computation, no extra classifier bias, exact zero residual, repeat and checkpoint restore.

`train3` preserves Phase8B batch8/AdamW/balanced CE/cosine50/clip5/patience12/strictvalidationBA and token DataLoader workers0. No full-network end-to-end training, no ensemble, no15split gate and no seed significance testing. `analyze` reports all nine metrics, both confusion matrices, sample SD and paired delta for every seed. It includes fixed-fit within-split shared-patient bootstrap sensitivity and fair resident-token readout timing. Timing excludes raw encoding, L1, normalization, HarNet and IO; batch/8 is amortized cost, not standalone patient latency.

Private tokens, checkpoints, per-person predictions, subject maps and hash catalogs use `restricted_manifests/`, explicitly excluded by the unchanged selective publisher. Public CSVs contain aggregates only. The mainline, old Phase8/Certified results and official model pointer remain unchanged. No remaining42 runs are started.

Publication uses the original `/home/zyt/deep_final/tools/git_publish/publish_to_github.py` collection, filtering and checks. `scripts/publication_transport.py --prepare` prepares its audited local commit and intercepts transport only. The authorized connected GitHub account publishes the identical tree with a non-force compare-and-swap from `beb57167a7b8b54792f1efed672927396fe8294d`. `scripts/postpublication_verify.py REMOTE_SHA` verifies remote tree/parent and server/source/mirror hashes, preserving the transport-local commit in an archive ref. The final receipt is outside this scientific snapshot: `/home/zyt/pureDeep-aipilot3-publication-receipt-20261112.json`. A dry-run is not completed publication.
