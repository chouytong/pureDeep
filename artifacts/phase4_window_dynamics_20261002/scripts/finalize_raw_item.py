"""Finish item05 only after all B1 runs and fixed-gate analysis complete."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent.parent
decision=json.loads((HERE/'analysis/b1_raw_decision.json').read_text());assert decision['status']=='complete' and decision['completed_runs']==45
test=json.loads((HERE/'analysis/b1_raw_implementation_test.json').read_text());assert test['status']=='PASS'
lock=json.loads((HERE/'analysis/baseline_lock.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in lock['source_sha256'].items():assert sha(p)==h
assert sha(lock['cache_path'])==lock['cache_sha256']
for s in lock['stages']:
    suffix=f"seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}"
    original=Path(lock['baseline_root'])/suffix
    for name,k in [('checkpoints/best.pt','checkpoint_sha256'),('predictions/validation.csv','prediction_sha256'),('normalization.json','normalization_file_sha256')]:assert sha(original/name)==s[k]
    stage=HERE/'runs/b1_raw'/suffix;status=json.loads((stage/'stage_status.json').read_text());assert status['status']=='complete' and status['outer_test_loader_created'] is False
    assert sha(stage/'checkpoints/best.pt')==status['summary']['checkpoint_sha256']
    ck=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False);state=ck['model_state']
    assert sum(v.numel() for v in state.values())==143172 and not any(k.startswith('window_') for k in state)
    assert all(torch.isfinite(v).all() for v in state.values());del ck
summary=pd.read_csv(HERE/'analysis/b1_raw_summary.csv').set_index('variant');paired=pd.read_csv(HERE/'analysis/b1_raw_paired.csv');agreement=pd.read_csv(HERE/'analysis/b1_raw_prediction_agreement.csv')
report=f'# Item 05 — local raw Acc input control\n\nDecision: **{decision["decision"]}** under the original preregistered gate; 45/45 complete. Frozen baseline hashes remain unchanged. No outer evaluation.\n\n'
report+='Files: scripts/prepare_raw_acc.py, raw_acc_hooks.py, run_raw_inner.py, test_raw_acc.py, finalize_raw_item.py; extensions to run_matrix.py/analyze_variant.py; analysis/b1_raw_* aggregate results. Server-only raw arrays are under features/raw_acc and excluded from Git, along with checkpoints/predictions.\n\n'
report+='Input: only local Acc changes from L1-detrended to original raw post-trim48 samples; original processed Gyro stays bit-identical. All 8,580 wrist records preserve 976/2000 lengths. HarNet mean cache, classifier architecture (143,172 trainable parameters), bilateral/ordered structure, balanced CE, optimizer/scheduler, seeds and 0.5 threshold stay fixed. No A1/A2 branch is present. Raw local Acc has no added clipping/augmentation.\n\n'
report+='Tests: shape [8,11,2,6,2000], actual activity lengths, invalid activity/wrist mask isolation, finite nonzero local encoder/SSL projection gradients, one-batch training, exact checkpoint reload, frozen SSL features; train-only normalization refitted, same train-ID hashes. Gyro normalization remains exactly equal to baseline in every run; Acc normalization changes. All formal checkpoints verified finite and correctly dimensioned. Formal one-batch smoke passed before full training; smoke metrics did not enter selection.\n\n'
metrics=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
report+='| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in ('baseline','b1_raw'):report+='| '+name+' | '+' | '.join(f'{summary.loc[name,m]:.6f}' for m in metrics)+' |\n'
report+='\n| Metric | Delta vs baseline | Improved /15 | 95% split-bootstrap CI | Cohen dz | Primary BH q |\n|---|---:|---:|---|---:|---:|\n'
for r in paired.itertuples():
    q=f'{r.bh_q_primary:.4f}' if np.isfinite(r.bh_q_primary) else '—'
    report+=f'| {r.metric} | {r.mean_delta:+.6f} | {r.improved_splits} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] | {r.cohen_dz:+.3f} | {q} |\n'
report+='\n| Configuration | BA seed-mean SD | AUROC seed-mean SD | BA within-split seed SD | AUROC within-split seed SD | Pair prediction agreement | Median best epoch |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in summary.index:
    r=summary.loc[name];a=agreement[agreement.variant==name].mean_pair_agreement.mean()
    report+=f'| {name} | {r.ba_seed_sd:.6f} | {r.auroc_seed_sd:.6f} | {r.ba_within_split_seed_sd:.6f} | {r.auroc_within_split_seed_sd:.6f} | {a:.6f} | {r.selected_epoch_median:.0f} |\n'
report+='\nGate: '+', '.join(f'{k}={v}' for k,v in decision['gate'].items())+'.\n\nThree seeds aggregate within each of 15 overlapping splits. Bootstrap intervals/BH/effect sizes are repeated-development robustness, not independent external evidence. Recall allowance 0.01 unchanged. No input/loss/hyperparameter combinations or filter search. Next numbered item is current frozen WSSL error analysis, using independent subjects and verified subtype metadata.\n'
(HERE/'ITEM05_REPORT.md').write_text(report)
readme=Path('/home/zyt/deep_final/README.md');marker='## 2026-10-02 — WSSL window dynamics, item 05'
if marker not in readme.read_text():
    with readme.open('a') as f:f.write('\n\n'+marker+'\n\n'+report.split('\n',1)[1]+'\n')
print(json.dumps({'item':'05','decision':decision['decision'],'report':str(HERE/'ITEM05_REPORT.md')},indent=2))
