# Project model index

The registry contains three parallel model identities with distinct lifecycle roles. Current retained selection remains [CURRENT_MODEL.json](CURRENT_MODEL.json), unchanged by AI registration.

| Model | Canonical ID | Lifecycle | Architecture / configuration | Evidence |
|---|---|---|---|---|
| STR-01 | `str01` | Matched Baseline | [pure_deep.py](../mainline/src/models/pure_deep.py), existing str01 seed configs | [STR report](../artifacts/str01_structured_token_residual_20260921/STR01_REPORT.md) |
| Frozen WSSL-STR | `frozen_wssl` | **Retained Model** | [wssl.py](../mainline/src/models/wssl.py), existing backbone configs and frozen HarNet path | [Phase3B report](../artifacts/phase3b_external_ssl_20260928/PHASE3B_REPORT.md) |
| **AI-WSSL-STR** (Activity Interaction WSSL-STR) | `ai_wssl_str` | **Experimental Candidate** | [independent architecture](../mainline/src/models/ai_wssl_str.py), [seed42 config](../mainline/configs/ai_wssl_str_seed42.yaml), seed43/44 | [model card and diagram](../mainline/docs/models/AI_WSSL_STR.md), [checkpoint catalog](../mainline/models/ai_wssl_str_checkpoints.json) |

[Machine-readable registry](../mainline/models/registry.json) · [Named load/inference CLI](../mainline/scripts/model_cli.py) · [AI training entry](../mainline/scripts/train_ai_wssl_str.py) · [Registration change report](../artifacts/ai_wssl_str_registration_20261116/reports/MODEL_REGISTRATION_REPORT.md)

AI-WSSL-STR evidence is kept in two distinct scopes: [old-L1 Phase8 15×3](../artifacts/wssl_guided_optimization_phase8_20261111/reports/ACTIVITY_INTERACTION_REPORT.md) and [Certified L1 one-split three-seed pilot](../artifacts/activity_interaction_certified_pilot3_20261112/reports/FINAL_REPORT.md). The latter has not completed full15-split validation and shows DD Recall declines in all three seeds. This index does not relabel it Retained, overwrite historical results or authorize more experiments.
