"""Explicitly authorized README/index updates; preserve historical entries and snapshots."""
from pathlib import Path
import json,hashlib,csv,io
ROOT=Path('/home/zyt/deep_final');HERE=ROOT/'artifacts/ai_wssl_str_registration_20261116'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 before=json.loads((HERE/'audit/before_state.json').read_text())['authorized_existing_file_changes']
 for rel,h in before.items():assert sha(ROOT/rel)==h,'Concurrent edit: '+rel
 p=ROOT/'README.md';s=p.read_text();s+='''

## Registered model identities (2026-11-16)

| Model | Lifecycle status | Entry |
|---|---|---|
| STR-01 | Matched Baseline | existing STR development entry |
| Frozen WSSL-STR | **Retained Model** | existing frozen WSSL entry |
| **AI-WSSL-STR** (Activity Interaction WSSL-STR) | **Experimental Candidate** | independent named model/config/training/checkpoint interface |

[Model index](conclusions/MODEL_INDEX.md) · [AI model card and architecture diagram](mainline/docs/models/AI_WSSL_STR.md) · [Machine-readable registry](mainline/models/registry.json) · [Registration change report](artifacts/ai_wssl_str_registration_20261116/reports/MODEL_REGISTRATION_REPORT.md).

AI-WSSL-STR adds cross-activity Self-Attention and residual logit correction to Frozen WSSL-STR with its original modules frozen. Existing Phase8 old-L1 results and Certified L1 three-seed pilot are linked separately. Full Certified L1 15-split validation is incomplete and DD Recall declines remain a limitation. Registration is not promotion: the current Frozen WSSL-STR pointer, STR/WSSL historical files, architecture/loss/recipe and existing checkpoints remain unchanged. No training was started for registration.
''';p.write_text(s)
 p=ROOT/'mainline/README.md';s=p.read_text();first=s.split('\n\n',2);assert len(first)==3;first[1]='主线并列管理 STR-01（Matched Baseline）、Frozen WSSL-STR（Retained Model）和 AI-WSSL-STR（Experimental Candidate）。AI 通过新增统一注册 API 和独立配置/训练/推理入口调用；原 STR/WSSL 底层工厂、实现及历史入口保持兼容，不导入 MFAM 等其他论文复现。正式模型指针仍为 Frozen WSSL-STR。';s='\n\n'.join(first);s+='''

## Independent model registry

[Project model index](../conclusions/MODEL_INDEX.md) · [Registry](models/registry.json) · [AI-WSSL-STR architecture/config/provenance](docs/models/AI_WSSL_STR.md).

```python
from src.models.registry import build_registered_model, list_models
from src.models.registered_checkpoint import load_registered_checkpoint
model = build_registered_model("AI-WSSL-STR", seed=42)
model, provenance = load_registered_checkpoint(
    "AI-WSSL-STR",
    checkpoint_id="AI-WSSL-STR:certified-l1:seed42:context0:inner0",
    device="cuda",
)
# No training: named checkpoint forward on matching cached bilateral tokens.
result = model.forward_tokens(bilateral_tokens, activity_mask)
```

`src.models.registry.build_registered_model` accepts all three model names. The original `src.models.build_model` remains the unchanged low-level STR-backbone factory used by historical STR/WSSL code; it is not silently repurposed. AI owns `src/models/ai_wssl_str.py`, three independent `configs/ai_wssl_str_seed*.yaml`, `scripts/train_ai_wssl_str.py`, the named `scripts/model_cli.py`, and a48-checkpoint catalog. New save format records model ID/status/config; historical checkpoints strict-load unchanged.

```bash
/home/zyt/envs/mfam_analysis/bin/python -B scripts/model_cli.py --list
/home/zyt/envs/mfam_analysis/bin/python -B scripts/model_cli.py --model AI-WSSL-STR --checkpoint-id AI-WSSL-STR:certified-l1:seed42:context0:inner0 --device cuda
/home/zyt/envs/mfam_analysis/bin/python -B scripts/train_ai_wssl_str.py --help
```

AI training remains the original Phase8B token recipe (batch8/workers0/train-balanced CE/AdamW/cosine50/patience12); only the added67,522 parameters train. The training CLI offers `--validate-only`, which was used for registration tests. No new training, L1 processing, normalization fit, HarNet extraction, threshold fitting or full15-split comparison was run. The candidate's current Certified pilot has DD Recall tradeoffs and is **not Retained**.
''';p.write_text(s)
 p=ROOT/'conclusions/README.md';s=p.read_text();s+='''

## Model registration update: AI-WSSL-STR

[MODEL_INDEX](MODEL_INDEX.md) now lists STR-01, Frozen WSSL-STR and **AI-WSSL-STR (Activity Interaction WSSL-STR)** in parallel. AI is an **Experimental Candidate** with independent source/config/training/checkpoint identity. It is not the current retained model. [Model card and architecture diagram](../mainline/docs/models/AI_WSSL_STR.md) links the old-L1 Phase8 45-run result and Certified L1 one-split three-seed pilot, with their distinct provenance and DD Recall limitation. Full Certified 15-split paired validation remains incomplete.

[Independent registration change report](../artifacts/ai_wssl_str_registration_20261116/reports/MODEL_REGISTRATION_REPORT.md). This update does not modify CURRENT_MODEL.json or replace STR/WSSL results; registration testing performs loading/forward/save-restore only, with zero optimizer updates.
''';p.write_text(s)
 p=ROOT/'conclusions/EXPERIMENT_REPORT_INDEX.csv';raw=p.read_bytes();existing={r['path'] for r in csv.DictReader(io.StringIO(raw.decode()))};records=[('AI_WSSL_STR_registration_20261116','MODEL_REGISTRATION_REPORT.md','artifacts/ai_wssl_str_registration_20261116/reports/MODEL_REGISTRATION_REPORT.md'),('AI_WSSL_STR_legacy_Phase8','ACTIVITY_INTERACTION_REPORT.md','artifacts/wssl_guided_optimization_phase8_20261111/reports/ACTIVITY_INTERACTION_REPORT.md'),('AI_WSSL_STR_certified_pilot3','FINAL_REPORT.md','artifacts/activity_interaction_certified_pilot3_20261112/reports/FINAL_REPORT.md')];out=io.StringIO();writer=csv.writer(out,lineterminator='\r\n' if b'\r\n' in raw else '\n');added=[]
 for row in records:
  if row[2] not in existing:writer.writerow(row);added.append(row[2])
 suffix=out.getvalue().encode();assert raw.endswith(b'\n');p.write_bytes(raw+suffix);assert p.read_bytes().startswith(raw)
 (HERE/'audit/document_changes.json').write_text(json.dumps(dict(changed_existing_files=[dict(path=r,before_sha256=h,after_sha256=sha(ROOT/r)) for r,h in before.items()],historical_report_index_entries_preserved=True,index_paths_added=added),ensure_ascii=False,indent=2)+'\n');print('DOCUMENTS_AND_INDEX_UPDATED')
if __name__=='__main__':main()
