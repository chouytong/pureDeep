# WSSL single EMA control — final report

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
|Accuracy|0.745604|0.737090|-0.008513|[-0.019181, +0.001142]|5/15|-0.409|
|BA|0.719644|0.683824|-0.035820|[-0.044770, -0.027454]|0/15|-2.042|
|AUROC|0.759553|0.754912|-0.004642|[-0.007912, -0.001495]|4/15|-0.708|
|Macro-F1|0.706589|0.682623|-0.023966|[-0.033693, -0.015234]|1/15|-1.266|
|PD Recall|0.782344|0.812271|+0.029927|[+0.009136, +0.050381]|10/15|0.713|
|DD Recall|0.656945|0.555377|-0.101568|[-0.127289, -0.076559]|0/15|-1.923|

Primary Wilcoxon BH q (BA/AUROC family): BA=0.00012207, AUROC=0.0255737. All six metric CIs are descriptive marginal bootstrap intervals, not simultaneous familywise intervals. BA and DD Recall decrease in every split. PD Recall increases while DD recognition worsens; Accuracy does not rescue this trade-off. Retention gates fail on BA, AUROC, Macro-F1, DD Recall and BA seed variability. Do not retain on agreement or any single metric.

## Seed variability and agreement

Within-split seed SD is SD of the three seed metrics in each split, averaged across15 splits. Seed-mean SD is SD of each seed's 15-split mean; reported as variability, not inference with45 independent observations.

|Metric|Ordinary within-split seed SD|EMA within-split seed SD|Ordinary seed-mean SD|EMA seed-mean SD|
|---|---:|---:|---:|---:|
|Accuracy|0.031205|0.029744|0.008650|0.010788|
|BA|0.021603|0.028945|0.006714|0.012922|
|AUROC|0.021741|0.022657|0.001812|0.001141|
|Macro-F1|0.025605|0.029354|0.004947|0.012858|
|PD Recall|0.065659|0.047199|0.025591|0.009858|
|DD Recall|0.078542|0.061886|0.035930|0.020241|

Within-split BA seed SD increases .021603→.028945 (about34.0%), failing <=1.25x gate. AUROC seed SD .021741→.022657. Three-seed prediction unanimity increases 0.732973→0.750617; pairwise seed agreement 0.821982→0.833745. Greater agreement does not imply better BA or DD recognition.

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

Publication supplement: analysis/error_group_counts_public.csv contains only six group/class aggregate rows. The count column is named subject_count to distinguish it from ID-list headers; no per-subject table is published. Export helper export_group_counts.py changes no metric, result, protocol or frozen core code.
