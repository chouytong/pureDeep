# AI-WSSL-STR independent model registration — 2026-11-16

This is an engineering registration, with **zero new training, optimizer updates or performance experiments**. AI-WSSL-STR is an **Experimental Candidate**, not the Retained Model. Frozen WSSL-STR remains the current selection. Start with [the independent modification report](reports/MODEL_REGISTRATION_REPORT.md), [test evidence](analysis/model_load_forward_restore_tests.json) and [change manifest](analysis/CHANGE_MANIFEST.csv).

The new model is callable through `mainline/src/models/registry.py` and the named `mainline/scripts/model_cli.py`. Its core is the Phase8 ActivityInteraction class, AST-identical except for class name, with explicit adapters for token and already-normalized signal input. The original STR/WSSL low-level factories and source/config/weights remain unchanged. No global dependencies are installed or modified. The four active README/report-index documents were explicitly updated; their pre-edit snapshots are private, and old report-index entries retain their exact byte prefix.

Server root: `/home/zyt/deep_final/artifacts/ai_wssl_str_registration_20261116`.

Existing environment: `/home/zyt/envs/mfam_analysis/bin/python`.

Actual registration workflow was: preserve source/weight/document baselines → generate new independent configs and checkpoint catalogs → install new source → load/forward/round-trip and CLI tests → metadata-only config closure with model/loss/training equality proof → document/index updates → final integrity/report → unchanged selective publisher → exact remote-tree readback. No training function was invoked. Training entry tests used `--validate-only` exclusively.

Named calls:

```bash
cd /home/zyt/deep_final/mainline
PY=/home/zyt/envs/mfam_analysis/bin/python
"$PY" -B scripts/model_cli.py --list
"$PY" -B scripts/model_cli.py --model AI-WSSL-STR --checkpoint-id AI-WSSL-STR:certified-l1:seed42:context0:inner0 --device cuda
"$PY" -B scripts/train_ai_wssl_str.py --help
```

Full API/inference/training-validation examples, lifecycle status, evidence scopes and Mermaid architecture diagram are in `/home/zyt/deep_final/mainline/docs/models/AI_WSSL_STR.md`. Stable identities cover45 historical legacy-L1 checkpoints and3 Certified pilot checkpoints; the latter are only context0/inner0, not a completed Certified15-split comparison. No historical checkpoint was renamed or rewritten.

`scripts/test_registration.py` is the executed server-only registration test. It strict-loads all three pilot identities, compares train-only token logits to the original implementation, verifies repeated and round-trip inference bitwise, exercises synthetic full-signal/padding behavior, checks wrong-model/hash/seed/architecture rejection, and tests the new CLI. Its fixed private outputs are preserved; rerunning against the same output directory intentionally refuses overwrites. Saved round-trip files are registration test copies with no optimizer state or training updates, not new experiments. No individual predictions or weights are public.

The dedicated future training entry retains the original fixed recipe and requires explicit token/base/split binding. Its implementation exists for independent model management, but future experiments still need their own authorization. This registration does not launch the remaining42 runs or select a new retained model.

Publication retains `/home/zyt/deep_final/tools/git_publish/publish_to_github.py` collection/filter/audits. `scripts/publication_transport.py --prepare` permits only the declared active-document updates, new mainline model files, new model index and this registration directory. It intercepts transport after the unchanged publisher's audited local commit. The GitHub connector publishes the identical immutable tree with a non-force compare-and-swap from `50a319441e2391ce662646cbd3e690ea64d0e1aa`; `scripts/postpublication_verify.py REMOTE_SHA` confirms remote parent/tree, source/mirror hashes and old/private integrity. The external final receipt is `/home/zyt/pureDeep-ai-registration-publication-receipt-20261116.json`.

`source_output_manifest.json` hashes the public artifact snapshot. Self-referential rows in `CHANGE_MANIFEST.csv` defer to the outer `PUBLICATION_MANIFEST.json`. Private original-document/config snapshots, round-trip weights, CLI outputs and protected catalogs remain under `restricted_manifests/` and are excluded by the existing publisher.
