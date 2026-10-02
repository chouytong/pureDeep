"""Record completed item07, including DD-only loss and unchanged frozen assets."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent.parent
decision=json.loads((HERE/'analysis/c1_dd_aux_decision.json').read_text());assert decision['status']=='complete' and decision['completed_runs']==45
test=json.loads((HERE/'analysis/c1_dd_aux_implementation_test.json').read_text());assert test['status']=='PASS'
eligibility=json.loads((HERE/'analysis/wssl_error_analysis_decision.json').read_text());assert eligibility['c1_eligible']
lock=json.loads((HERE/'analysis/baseline_lock.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in lock['source_sha256'].items():assert sha(p)==h
assert sha(lock['cache_path'])==lock['cache_sha256']
for s in lock['stages']:
    suffix=f"seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}"
    original=Path(lock['baseline_root'])/suffix
    for name,k in [('checkpoints/best.pt','checkpoint_sha256'),('predictions/validation.csv','prediction_sha256'),('normalization.json','normalization_file_sha256')]:assert sha(original/name)==s[k]
    stage=HERE/'runs/c1_dd_aux'/suffix;status=json.loads((stage/'stage_status.json').read_text());assert status['status']=='complete' and status['outer_test_loader_created'] is False
    assert sha(stage/'checkpoints/best.pt')==status['summary']['checkpoint_sha256']
    assert status['summary']['validation_metrics']['dd_aux_loss']==0
    hist=[json.loads(l) for l in (stage/'logs/epochs.jsonl').read_text().splitlines()]
    assert all(r['validation']['dd_aux_loss']==0 for r in hist)
    ck=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False);state=ck['model_state']
    assert sum(v.numel() for v in state.values())==144208 and not any(k.startswith('window_') for k in state)
    assert state['dd_aux_head.weight'].shape==(4,258)
    assert all(torch.isfinite(v).all() for v in state.values());del ck
summary=pd.read_csv(HERE/'analysis/c1_dd_aux_summary.csv').set_index('variant');paired=pd.read_csv(HERE/'analysis/c1_dd_aux_paired.csv');agreement=pd.read_csv(HERE/'analysis/c1_dd_aux_prediction_agreement.csv')
report=f'# Item 07 — conditional DD source-category auxiliary supervision\n\nDecision: **{decision["decision"]}** under the original preregistered gate; 45/45 complete. Original baseline hashes unchanged. No outer evaluation.\n\n'
report+='Files: C1_PROTOCOL.md, scripts/dd_aux_hooks.py, run_aux_inner.py, test_dd_aux.py, finalize_aux_item.py; run_matrix.py/analyze_variant.py extensions; analysis/c1_dd_aux_* aggregate results. Per-subject labels/predictions/checkpoints stay on server and are excluded from Git.\n\n'
report+='Eligibility: item06 Other–ET diagnosis-category consensus recall difference passed bootstrap and BH criteria; all source labels consistent and category N≥10. Categories are broad source diagnoses; no fine-grained clinical subtype labels invented. Use all four categories, one fixed auxiliary weight .1; no category, weight, sampling or head search.\n\n'
report+='Original WSSL paths/inputs/normalization and balanced CE unchanged. Training auxiliary Linear(258,4) taps original subject embedding: 1,036 added parameters, total 144,208 trainable. Loss is original main CE + .1×unweighted subtype CE on DD training samples in each batch. PD-only batch has no auxiliary loss, gradient or optimizer update. Validation and inference have no subtype targets/output/loss; the original PD/DD decision is unchanged. Frozen HarNet cache unchanged. No A1/A2/B1 combination.\n\n'
report+='Tests: exact original inference for all 45 frozen baseline checkpoints; real [8,11,2,6,2000] input and [8,4] training auxiliary logits; DD auxiliary/local gradients finite nonzero, PD-only auxiliary gradient absent and weights unchanged, invalid masks isolated; single-batch update; exact checkpoint reload with empty training-ID/label sets; zero validation auxiliary loss. Formal smoke passed before full training. All final checkpoints finite, correctly dimensioned, and every validation epoch has auxiliary loss 0.\n\n'
metrics=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
report+='| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in ('baseline','c1_dd_aux'):report+='| '+name+' | '+' | '.join(f'{summary.loc[name,m]:.6f}' for m in metrics)+' |\n'
report+='\n| Metric | Delta vs baseline | Improved /15 | 95% split-bootstrap CI | Cohen dz | Primary BH q |\n|---|---:|---:|---|---:|---:|\n'
for r in paired.itertuples():
    q=f'{r.bh_q_primary:.4f}' if np.isfinite(r.bh_q_primary) else '—'
    report+=f'| {r.metric} | {r.mean_delta:+.6f} | {r.improved_splits} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] | {r.cohen_dz:+.3f} | {q} |\n'
report+='\n| Configuration | BA seed-mean SD | AUROC seed-mean SD | BA within-split seed SD | AUROC within-split seed SD | Pair prediction agreement | Median best epoch |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in summary.index:
    r=summary.loc[name];a=agreement[agreement.variant==name].mean_pair_agreement.mean()
    report+=f'| {name} | {r.ba_seed_sd:.6f} | {r.auroc_seed_sd:.6f} | {r.ba_within_split_seed_sd:.6f} | {r.auroc_within_split_seed_sd:.6f} | {a:.6f} | {r.selected_epoch_median:.0f} |\n'
report+='\nGate: '+', '.join(f'{k}={v}' for k,v in decision['gate'].items())+'.\n\nThree seeds aggregate within each of 15 overlapping splits; statistical intervals/BH/effect sizes are repeated-development evidence, not external confirmation. All permitted average recall drops stay .01. No hyperparameter/threshold adjustment after results. Item08 applies only if a candidate passes the existing retention gate.\n'
(HERE/'ITEM07_REPORT.md').write_text(report)
readme=Path('/home/zyt/deep_final/README.md');marker='## 2026-10-02 — WSSL window dynamics, item 07'
if marker not in readme.read_text():
    with readme.open('a') as f:f.write('\n\n'+marker+'\n\n'+report.split('\n',1)[1]+'\n')
print(json.dumps({'item':'07','decision':decision['decision'],'report':str(HERE/'ITEM07_REPORT.md')},indent=2))
