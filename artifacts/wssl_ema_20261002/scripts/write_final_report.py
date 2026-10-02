"""Report fixed primary analysis; no parameter/epoch selection."""
import json
from pathlib import Path
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent.parent;ROOT=Path('/home/zyt/deep_final')
M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall'];names=dict(accuracy='Accuracy',ba='BA',auroc='AUROC',macro_f1='Macro-F1',pd_recall='PD Recall',dd_recall='DD Recall')
s=pd.read_csv(H/'analysis/ema_summary.csv').set_index('variant');p=pd.read_csv(H/'analysis/ema_paired.csv').set_index('metric');decision=json.loads((H/'analysis/ema_decision.json').read_text());assert decision['decision']=='REJECT'
rows=[]
for m in M:
 r=p.loc[m];rows.append(f"|{names[m]}|{s.loc['baseline',m]:.6f}|{s.loc['ema',m]:.6f}|{r.mean_delta:+.6f}|[{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]|{int(r.improved_splits)}/15|{r.cohen_dz:.3f}|")
seedrows=[]
for m in M:seedrows.append(f"|{names[m]}|{s.loc['baseline',m+'_within_split_seed_sd']:.6f}|{s.loc['ema',m+'_within_split_seed_sd']:.6f}|{s.loc['baseline',m+'_seed_sd']:.6f}|{s.loc['ema',m+'_seed_sd']:.6f}|")
a=pd.read_csv(H/'analysis/ema_prediction_agreement.csv').groupby('variant')[['prediction_agreement','mean_pair_agreement','mean_score_spearman']].mean()
report='''# WSSL single EMA control — final report

Started 2026-10-02; completed 2026-10-03 Asia/Shanghai. **REJECT EMA; retained best remains original frozen WSSL-STR.** Exactly one new performance candidate. No additional training, stopping-rule sensitivity, decay/start or combination sweep is authorized by these results.

## Execution and integrity

1. Isolated independent_wssl.py removes captured model closures, preserves state keys and accepts explicit SSL forward input; unchanged engine uses an instance-local feature bridge. 45 old checkpoints: logits bit-identical; old archived probability max difference1.11e-16. Separate parameters/caches, fresh-instance and deepcopy isolation, checkpoint reload, RNG preservation and gradient smoke PASS. Reviewed independent construction paths are the archived Phase3B baseline and Phase4 variants; no blanket audit of every historical experiment is claimed.
2. Existing 45 logs and 90 actual best/last checkpoints evaluated in eval mode. Best/stop epoch medians10/22; no intermediate weights/full historical EMA. Best train/validation BA=.913406/.719644, last=.999546/.658764. Last-minus-best validation AUROC=-.014419, main CE=+.420458. Online dropout-active train metrics distinguished from checkpoint eval. BA-best versus neighboring/later BA comparisons are conditioned on the selection rule; not an independent proof of a universally optimal stopping rule. See ITEM02_TRAJECTORY.md and ITEM02_NEIGHBORHOOD.md.
3. 45 same-trajectory ordinary+EMA controls: 15 splits×seeds42/43/44; 26722 updates. S26/27 gives alpha .9736927206974342/.9746546091224311. EMA initialized from ordinary initial weights, updated after each optimizer step from step1. HarNet weights/statistics remain frozen cached features; no adaptation. Original optimizer/loss/scheduler/threshold/input order/normalization unchanged. Parameters unchanged:143172 classifier trainable +10457408 frozen HarNet =10600580 total; shadow adds no deployed model parameters.
4. Ordinary BA selects the same epoch for both parameter sets. No EMA epoch search. EMA inference only after ordinary training. All 45 ordinary checkpoint weights, numerical epoch metrics, best/stop epochs and validation probabilities exactly match archive; normalization and IDs align. All 45 EMA selected checkpoints pass strict provenance checks. Pretraining-fixed core code/protocol and original source/split/cache/checkpoints/predictions unchanged. Inherited EMA-output best_metric denotes ordinary selection BA; copied ordinary logs are selection provenance, not EMA loss trajectory.
5. Seed-first paired comparison and descriptive error changes; no outer predictions/performance, test loaders or outer-based decisions enter this analysis. Original formal artifacts remain immutable; new artifacts are in a separate directory.

## Fixed-point performance

Metrics are mean of three seed metrics within each split, then equal mean of 15 splits. Differences=EMA−ordinary WSSL.

|Metric|Ordinary WSSL|EMA|Paired difference|Bootstrap 95% CI|Improved splits|Cohen dz|
|---|---:|---:|---:|---|---:|---:|
'''+ '\n'.join(rows)+f'''

Primary Wilcoxon BH q (BA/AUROC family): BA={p.loc['ba','bh_q_primary']:.6g}, AUROC={p.loc['auroc','bh_q_primary']:.6g}. All six metric CIs are descriptive marginal bootstrap intervals, not simultaneous familywise intervals. BA and DD Recall decrease in every split. PD Recall increases while DD recognition worsens; Accuracy does not rescue this trade-off. Retention gates fail on BA, AUROC, Macro-F1, DD Recall and BA seed variability. Do not retain on agreement or any single metric.

## Seed variability and agreement

Within-split seed SD is SD of the three seed metrics in each split, averaged across15 splits. Seed-mean SD is SD of each seed's 15-split mean; reported as variability, not inference with45 independent observations.

|Metric|Ordinary within-split seed SD|EMA within-split seed SD|Ordinary seed-mean SD|EMA seed-mean SD|
|---|---:|---:|---:|---:|
'''+ '\n'.join(seedrows)+f'''

Within-split BA seed SD increases .021603→.028945 (about34.0%), failing <=1.25x gate. AUROC seed SD .021741→.022657. Three-seed prediction unanimity increases {a.loc['baseline','prediction_agreement']:.6f}→{a.loc['ema','prediction_agreement']:.6f}; pairwise seed agreement {a.loc['baseline','mean_pair_agreement']:.6f}→{a.loc['ema','mean_pair_agreement']:.6f}. Greater agreement does not imply better BA or DD recognition.

## Error changes

Per split, first average counts across seeds, then average15 splits:

|Class|Ordinary error corrected by EMA|Ordinary correct newly wrong under EMA|Net corrections|Ordinary/EMA prediction agreement|
|---|---:|---:|---:|---:|
|PD|3.577778|1.377778|+2.200000|.932696|
|DD|.377778|3.466667|-3.088889|.873486|

Separate descriptive sensitivity uses seed-mean decisions per validation context (not formal seed-mean metric performance). Original primary stable-error definition: wrong in all4 validation contexts after three-seed score averaging; sensitivity only: wrong in all12 seed/context predictions. Baseline groups: stable-correct228, stable-error38 (PD14/DD24), unstable124. Within stable-error, only3 PD and1 DD validation appearances recover; stable-correct gains0 and suffers3 PD/10 DD new wrong appearances; unstable PD gains40 net appearances, unstable DD loses22 net. These overlapping appearances are not independent samples or mechanistic evidence. No phenotype/representation diagnosis or new threshold redesign follows this summary.

## Interpretation and stop decision

Late ordinary weights increasingly fit training while selected-checkpoint generalization is better than actual last weights. The chosen one-epoch-half-life EMA nevertheless **does not improve** the original ordinary-BA-selected operating point; it trades higher PD Recall for much lower DD Recall and worsens both BA and AUROC. This is evidence against this specified EMA control, not a claim that every conceivable EMA setting is ineffective. The user-fixed stop rule is applied: **STOP EMA coefficient, start-time, combination and stopping-rule searches; no new candidate.** Model-copy independence fix is retained as a correctness improvement, without changing the formal model architecture or archived metrics.

Retained best remains original WSSL-STR: Accuracy .745604, BA .719644, AUROC .759553, Macro-F1 .706589, PD Recall .782344, DD Recall .656945.

## Statistical boundary

Only15 fixed overlapping development splits are paired units, after seed averaging; bootstrap10000. Subjects, PD–DD pairs,45 runs and26722 optimizer steps are not independent inferential samples. Shared subjects/training contexts and prior development selection limit generalization of CIs/p-values; no external confirmation or outer mechanism claim. No actual intermediate checkpoints or reconstructed historical EMA. EMA has no full validation trajectory by design. Negative results and the stop decision are preserved.

## Files and version control

Code: independent_wssl.py, test_independence.py, diagnose_trajectory.py, ema_hooks.py, run_ema.py, test_ema_recurrence.py, audit_completed.py, analyze_ema.py, analyze_errors.py, write_final_report.py. Protocol and source/alpha lock, ITEM01–04 reports, fixed-point aggregate metrics/paired CI/seed agreement/error summaries, and FINAL_REPORT.md in artifacts/wssl_ema_20261002. Root README and CONTEXT_HANDOFF updated. Normalization values, per-subject predictions, raw logs, datasets/features and weights stay outside public Git selection. Git commits/pushes and annotated tags for all five items; see GIT_RELEASES.md.
'''
(H/'FINAL_REPORT.md').write_text(report)
with (ROOT/'README.md').open('a') as f:f.write('\n\n## WSSL single EMA — final negative result (2026-10-03)\n\n**REJECT EMA; original frozen WSSL-STR remains retained best.** Model-copy independence corrected; all45 old checkpoints exact logits. Existing logs/90 best-last checkpoint eval diagnose late fitting. 45 same-trajectory EMA controls, one-epoch half-life, ordinary BA selects matched epoch, all ordinary numerical trajectories/weights/predictions exactly reproduce archive. No outer use.\n\n|Metric|Ordinary WSSL|EMA|Delta EMA−ordinary|\n|---|---:|---:|---:|\n'+'\n'.join(f"|{names[m]}|{s.loc['baseline',m]:.6f}|{s.loc['ema',m]:.6f}|{p.loc[m,'mean_delta']:+.6f}|" for m in M)+'\n\nBA delta -.035820, 95%CI[-.044770,-.027454],0/15 improved; AUROC -.004642,CI[-.007912,-.001495],4/15 improved; DD Recall -.101568,CI[-.127289,-.076559],0/15 improved. PD Recall+.029927 trades against DD. BA within-split seed SD .021603→.028945; unanimity improves but performance does not.15 overlapping splits are descriptive paired units after seed averaging, not45 independent runs. **Stop EMA decay/start/combination/stopping-rule sensitivity searches; no other candidate.** Correctness interface retained, formal model/recipe/threshold and archived artifacts unchanged. Development-only result; no outer conclusion. [Full report](artifacts/wssl_ema_20261002/FINAL_REPORT.md); [paired results](artifacts/wssl_ema_20261002/analysis/ema_paired.csv).\n')
with (ROOT/'CONTEXT_HANDOFF.md').open('a') as f:f.write('\n\n## 2026-10-03 Latest verified record: WSSL single EMA COMPLETE / REJECT\n\nAll5 requested items complete. Isolated copy-independent WSSL interface passes45 old-checkpoint exact logits/caches/reload/RNG; architecture unchanged.45 same-trajectory EMA controls with fixed one-epoch half-life alpha2**(-1/S),S26/27. Ordinary BA chooses common checkpoint epoch; all ordinary weights/epoch metrics/predictions exact archive. EMA BA .683824 vs .719644,AUROC .754912 vs .759553,DDRecall .555377 vs .656945;BA/DD lower15/15;BA seed SD worsens.**REJECT; STOP EMA coefficient/start/combination and stop-rule search.** No further training running or recommended automatically. Retained best remains original frozen WSSL-STR (Acc.745604,BA.719644,AUC.759553,F1.706589,PD.782344,DD.656945). No outer use and no new mechanism/module search. Correctness interface is available at artifacts/wssl_ema_20261002/scripts/independent_wssl.py; report FINAL_REPORT.md. Prior in-progress sections are superseded by this complete record.\n')
print('FINAL_REPORT written; README/handoff appended; decision REJECT')
