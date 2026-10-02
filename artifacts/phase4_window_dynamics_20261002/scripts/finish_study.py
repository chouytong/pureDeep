"""Resolve conditional item08 and archive study; no new training or selection."""
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent.parent
VARIANTS=['a1_mean','a2_delta','b1_raw','c1_dd_aux']
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
decisions={v:json.loads((HERE/f'analysis/{v}_decision.json').read_text()) for v in VARIANTS}
assert all(d['status']=='complete' and d['completed_runs']==45 for d in decisions.values())
retained=[v for v,d in decisions.items() if d['decision']=='RETAIN']
assert not retained, 'A retained candidate requires actual item08 epoch/split sensitivity review'
lock=json.loads((HERE/'analysis/baseline_lock.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in lock['source_sha256'].items():assert sha(p)==h
assert sha(lock['cache_path'])==lock['cache_sha256'] and sha(lock['split_path'])==lock['split_sha256']
for s in lock['stages']:
    stage=Path(lock['baseline_root'])/f"seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}"
    for path,k in [('checkpoints/best.pt','checkpoint_sha256'),('predictions/validation.csv','prediction_sha256'),('normalization.json','normalization_file_sha256')]:assert sha(stage/path)==s[k]
rows=[]
baseline=pd.read_csv(HERE/'analysis/baseline_summary.csv').iloc[0]
rows.append(dict(variant='baseline',decision='RETAIN existing frozen WSSL',**{m:baseline[m] for m in M},parameters=143172,ba_within_split_seed_sd=baseline.ba_within_split_seed_sd,auroc_within_split_seed_sd=baseline.auroc_within_split_seed_sd,delta_ba=0.,delta_auroc=0.,best_epoch_median=baseline.best_epoch_median))
for v in VARIANTS:
    assert json.loads((HERE/f'{v}_matrix_progress.json').read_text())['status']=='complete'
    summary=pd.read_csv(HERE/f'analysis/{v}_summary.csv').set_index('variant').loc[v]
    pairs=pd.read_csv(HERE/f'analysis/{v}_paired.csv');pairs=pairs[pairs.reference=='baseline'].set_index('metric')
    rows.append(dict(variant=v,decision=decisions[v]['decision'],**{m:summary[m] for m in M},parameters=decisions[v]['parameters'],ba_within_split_seed_sd=summary.ba_within_split_seed_sd,auroc_within_split_seed_sd=summary.auroc_within_split_seed_sd,delta_ba=pairs.loc['ba','mean_delta'],delta_auroc=pairs.loc['auroc','mean_delta'],best_epoch_median=summary.selected_epoch_median))
summary=pd.DataFrame(rows);summary.to_csv(HERE/'analysis/study_final_summary.csv',index=False)
item8='''# Item 08 — conditional candidate review

Decision: **SKIP: no candidate passed the preregistered retention gate**. Items03/04/05/07 each completed 45 runs and were rejected. Existing frozen WSSL remains retained.

Files: scripts/finish_study.py, analysis/study_final_summary.csv, study_completion.json, FINAL_REPORT.md and this report. Checks: all four matrices complete; all decision files match 45 required runs; original split/cache/source/checkpoint/prediction/normalization hashes unchanged. No new model or inference runs required for this conditional resolution.

Selected epochs, split SD, seed SD, prediction agreement and paired differences were already reported for every tested variant. There is no retained candidate to freeze for an additional stopping-epoch/split sensitivity study. Original early-stopping rule remains unchanged. No train-internal rule replacement, baseline refit, extra checkpoint selection or hyperparameter search was performed. The condition is explicitly resolved as skipped, not presented as a successful candidate review. Same-subject repeated development results are not external validation.

The final metric/delta table is in FINAL_REPORT.md and analysis/study_final_summary.csv. All new candidates rejected; keep original WSSL. No follow-on experiment is automatically authorized.
'''
(HERE/'ITEM08_REPORT.md').write_text(item8)
report='# WSSL ordered experiment final record — items01–08\n\n**Retained configuration: original frozen WSSL-STR.** Verified/reused 45 baseline stages; four independent 45-run studies (180 newly trained runs), all rejected by registered gates. Item06 updated current WSSL errors and enabled the one fixed C1 test; item08 skipped because no candidate was retained. No candidate stacking, dimensions/weights/threshold/recipe search or outer information use.\n\n'
report+='| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | ΔBA | ΔAUROC | Decision |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n'
for r in summary.itertuples():report+='| '+r.variant+' | '+' | '.join(f'{getattr(r,m):.6f}' for m in M)+f' | {r.delta_ba:+.6f} | {r.delta_auroc:+.6f} | {r.decision} |\n'
report+='''
## Decisions and checks

1. Baseline asset/config/split/seed/normalization/metric audit passed; reuse confirmed.
2. True front/back windows and masks preserved. Mean cache and same-input logits exactly equal for all 45 frozen checkpoints; no fake second window for short records.
3. A1 mean branch: small BA/AUROC mean gain, BA improvement only 8/15, BA CI crosses zero; reject.
4. A2 delta branch: BA and Macro-F1 decrease; no simultaneous BA/AUROC advantage over matched A1; reject. Both use same initialization/norm/masks; only Entrainment/Relaxed/RelaxedTask actually have double windows.
5. Raw local Acc: Accuracy/PD recall increase, BA/AUROC decrease, DD recall decreases .026286 (CI −.049631 to −.003442), above allowed .01 drop; reject. Gyro/SSL/lengths unchanged.
6. Fresh WSSL independent-subject summary: primary stable errors PD14/DD24. Verified source categories Other60/ET28/Atypical15/MS11. Other–ET exploratory consensus recall difference passes six-pair BH q=.029499 and subject-bootstrap CI. Strict seed unanimity is sensitivity only. No errors removed or validation-informed sampling.
7. C1 source-category auxiliary weight .1: BA/F1 decrease; AUROC increase lacks stable support/required count. DD-only loss, no PD auxiliary update, no validation/inference subtype labels; reject. Broad source categories are not granular adjudicated clinical subtype labels.
8. Conditional review skipped: no retained new candidate; stopping rule unchanged.

All implementations passed applicable shape/mask/gradient/checkpoint-reload/single-batch tests and formal smoke before full runs. Classifier-side trainable parameters: baseline 143,172; A1/A2 151,948; raw Acc 143,172; C1 144,208 (auxiliary head unused at inference). Official HarNet10 stays frozen at 10,457,408. Selected epoch/seed/split SD/agreement, paired bootstrap intervals/counts/effect sizes/BH are in each numbered report and aggregate CSV.

Original source/checkpoint/cache/normalization/prediction/split hashes verified unchanged at closure. Public Git contains source/configs/reports/aggregate metrics; datasets, raw arrays, feature caches, weights, per-subject predictions and labels stay on server. Each numbered item is individually committed/pushed/tagged; item08 final closure publishing follows this record.

## Evidence boundary

Three seeds average within 15 fixed overlapping development splits. The 180 executions are not independent statistical samples; split-bootstrap and exploratory BH describe repeated development robustness. Current WSSL errors use one record per independent subject after seed/context aggregation, with model-training overlap acknowledged. Quality measures are descriptive and do not establish bad recordings or justify deletion. Original FOE-01 outer results were not used in this study, and these results provide no new external validation. No new best configuration or automatic follow-on search is supported.
'''
(HERE/'FINAL_REPORT.md').write_text(report)
completion=dict(status='analysis_complete',publication_verification='separate Git receipt after final commit and tag push',items={'01':'baseline reused, verified','02':'windows and exact consistency PASS','03':'A1 REJECT','04':'A2 REJECT','05':'B1 REJECT','06':'WSSL errors complete, C1 eligible','07':'C1 REJECT','08':'conditional SKIP, no retained candidate'},new_training_runs=180,baseline_stages_reused=45,retained_configuration='original phase3b b_str_pretrained frozen WSSL-STR',outer_results_used=False,early_stopping_rule_changed=False,candidate_combinations=False,formal_baseline_hashes_unchanged=True)
(HERE/'analysis/study_completion.json').write_text(json.dumps(completion,indent=2)+'\n')
root=Path('/home/zyt/deep_final');readme=root/'README.md';marker='## 2026-10-02 — WSSL window dynamics, item 08 and final closure'
if marker not in readme.read_text():
    with readme.open('a') as f:f.write('\n\n'+marker+'\n\n'+item8.split('\n',1)[1]+'\n\n'+report.split('\n',1)[1]+'\n')
handoff=root/'CONTEXT_HANDOFF.md';marker='## 2026-10-02 final ordered WSSL study — all items resolved'
if marker not in handoff.read_text():
    with handoff.open('a') as f:f.write('\n\n'+marker+'\n\nLatest formal record: `artifacts/phase4_window_dynamics_20261002/FINAL_REPORT.md`, reports ITEM01–ITEM08. All180 new training runs complete. A1 mean, A2 delta, B1 raw Acc and fixed-weight C1 DD source-category auxiliary all rejected. Retain original frozen WSSL-STR: Acc .745604, BA .719644, AUROC .759553, F1 .706589, PD .782344, DD .656945. Item08 skipped due no retained candidate; no stopping-rule change or extra refit. Updated WSSL primary stable-error groups PD14/DD24; strict all-seed unanimity remains separate sensitivity. No outer information used, no active experiment and no automatically recommended additional model search. All numbered items committed/pushed/tagged on chouytong/pureDeep; final item08 push/tag follows this closure record.\n')
print(summary.round(6).to_string(index=False));print(json.dumps(completion,indent=2))
