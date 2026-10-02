"""Record a completed full-matrix numbered item; never starts another experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['a1_mean','a2_delta'],required=True);args=ap.parse_args()
variant=args.variant; item='03' if variant=='a1_mean' else '04'
decision=json.loads((HERE/f'analysis/{variant}_decision.json').read_text())
assert decision['status']=='complete' and decision['completed_runs']==45
test=json.loads((HERE/f'analysis/{variant}_implementation_test.json').read_text());assert test['status']=='PASS'
lock=json.loads((HERE/'analysis/baseline_lock.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in lock['source_sha256'].items():assert sha(p)==h,('frozen source modified',p)
for s in lock['stages']:
    original=Path(lock['baseline_root'])/f"seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}"
    assert sha(original/'checkpoints/best.pt')==s['checkpoint_sha256']
    assert sha(original/'predictions/validation.csv')==s['prediction_sha256']
    assert sha(original/'normalization.json')==s['normalization_file_sha256']
    stage=HERE/f"runs/{variant}/seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}"
    status=json.loads((stage/'stage_status.json').read_text());assert status['status']=='complete' and status['outer_test_loader_created'] is False
    assert sha(stage/'checkpoints/best.pt')==status['summary']['checkpoint_sha256']
    ck=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    weights=ck['model_state'];assert sum(w.numel() for w in weights.values())==151948
    assert weights['window_branch.0.weight'].shape==(8,1024)
    assert weights['window_branch.2.weight'].shape==(64,8)
    assert all(torch.isfinite(w).all() for w in weights.values())
    assert weights['window_branch.2.weight'].abs().sum()>0
    del ck
summary=pd.read_csv(HERE/f'analysis/{variant}_summary.csv').set_index('variant')
paired=pd.read_csv(HERE/f'analysis/{variant}_paired.csv')
agreement=pd.read_csv(HERE/f'analysis/{variant}_prediction_agreement.csv')
metrics=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
title='A1 capacity-matched mean residual' if item=='03' else 'A2 front/back dynamics residual'
report=f'# Item {item} — {title}\n\nDecision: **{decision["decision"]}** under the preregistered gate. All 45 development stages completed; baseline source/checkpoint/prediction/normalization hashes remain unchanged. No outer evaluation.\n\n'
report+='Files: scripts/window_hooks.py, run_inner.py, run_matrix.py, test_branch.py, analyze_variant.py and finalize_item.py; analysis/'+variant+'_* aggregate CSV/JSON and this report. Historical model code stays unchanged. Checkpoints, caches and per-subject predictions are server-only.\n\n'
report+='Tests: all 45 baseline checkpoints retain exactly unchanged logits at zero initialization; real inner-train batch [8,11,2,6,2000]; invalid-window/wrist/activity isolation; finite nonzero output gradients; hidden gradient zero initially and nonzero after output update; two single-batch updates; exact checkpoint reload. Formal one-epoch/one-batch smoke passed with original train-only normalization SHA. All final checkpoint weights are finite and learned output branches are nonzero. Smoke performance did not enter retention.\n\n'
report+='Original WSSL classifier: 143,172 trainable parameters. Candidate: 151,948 (+8,776). HarNet10 stays frozen (10,457,408). Parameter-free LayerNorm, 1024→8→GELU→64, zero output initialization; branch enabled only for actual double windows and valid wrist/activity.\n\n'
report+='| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in ['baseline']+(['a1_mean'] if variant=='a2_delta' else [])+[variant]:
    report+='| '+name+' | '+' | '.join(f'{summary.loc[name,m]:.6f}' for m in metrics)+' |\n'
report+='\n| Reference | Metric | Mean delta | Improved /15 | 95% split-bootstrap CI | Cohen dz | Primary BH q |\n|---|---|---:|---:|---|---:|---:|\n'
for r in paired.itertuples():
    q=f'{r.bh_q_primary:.4f}' if np.isfinite(r.bh_q_primary) else '—'
    report+=f'| {r.reference} | {r.metric} | {r.mean_delta:+.6f} | {r.improved_splits} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] | {r.cohen_dz:+.3f} | {q} |\n'
report+='\n| Configuration | BA seed-mean SD | AUROC seed-mean SD | BA within-split seed SD | AUROC within-split seed SD | Mean pair prediction agreement | Median best epoch |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in summary.index:
    r=summary.loc[name]; a=agreement[agreement.variant==name].mean_pair_agreement.mean()
    report+=f'| {name} | {r.ba_seed_sd:.6f} | {r.auroc_seed_sd:.6f} | {r.ba_within_split_seed_sd:.6f} | {r.auroc_within_split_seed_sd:.6f} | {a:.6f} | {r.selected_epoch_median:.0f} |\n'
report+='\nGate results: '+', '.join(f'{k}={v}' for k,v in decision['gate'].items())+'.\n\n'
if variant=='a2_delta':report+=f'Dynamics-specific positive mean BA and AUROC advantage over A1: **{decision["dynamic_specific_mean_advantage"]}**. A numerical advantage alone does not establish robust dynamics-specific benefit; use paired CIs/counts above.\n\n'
report+='Three seed metrics are averaged within each split before paired analysis. The 15 splits share subjects; bootstrap intervals, Cohen dz, Wilcoxon and BH results describe repeated development robustness, not 15 independent population samples or external validation. No 45-run or subject-pair inflation. Allowed average PD/DD recall drop remains 0.01, fixed before results. No threshold/recipe/dimension search or candidate combinations.\n'
(HERE/f'ITEM{item}_REPORT.md').write_text(report)
readme=Path('/home/zyt/deep_final/README.md');marker=f'## 2026-10-02 — WSSL window dynamics, item {item}'
if marker not in readme.read_text():
    with readme.open('a') as f:f.write('\n\n'+marker+'\n\n'+report.split('\n',1)[1]+'\n')
print(json.dumps({'item':item,'decision':decision['decision'],'checks':'PASS','report':str(HERE/f'ITEM{item}_REPORT.md')},indent=2))
