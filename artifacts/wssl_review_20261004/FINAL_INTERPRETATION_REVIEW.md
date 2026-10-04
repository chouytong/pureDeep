# Independent final interpretation review

Date: 2026-10-04. **Verdict: REJECT the single LR1e-4 control; retain frozen WSSL-STR with LR2e-4.** This is a read-only independent review of the completed aggregate results and frozen protocol. No data, script, gate or historical output was changed; no new model or outer analysis was run.

## 1. Evidence and verification scope

Reviewed `LR_CONTROL_PROTOCOL.md`, current `scripts/analyze_lr_control.py`, `analysis/lr_control_lock.json`, `lr_decision.json`, `lr_seed_split_metrics.csv`, `lr_15split_seedfirst.csv`, `lr_summary.csv`, `lr_paired.csv`, `lr_prediction_agreement.csv`, and all three aggregate error tables.

Independent checks from those CSVs passed:

- 90 unique method×seed×split rows:45 reference and45 candidate; exactly seeds42/43/44 and15 splits per method.
- 30 unique method×split rows, each averaging exactly its three seed metrics. Six metric means, within-split seed SD, SD of per-seed means and epoch medians independently recomputed and matched the tables within1e-14.
- All paired means and positive/tied counts independently recomputed from15 split rows and matched. Primary-two and exploratory-all-six BH values independently recomputed from recorded Wilcoxon p-values and matched.
- Error-transition arithmetic and seed-first/split-average count tables matched; mean split-level net-correction rates equal the corresponding PD/DD Recall differences within1e-14.
- The actual analysis SHA and protocol SHA equal both their frozen lock entries and the decision provenance.

This subtask did **not** reread45 server checkpoints or raw prediction files. The root run's strict entry/per-stage audit supplies those ID/hash/config/no-outer checks. Bootstrap intervals and Wilcoxon p-values are reviewed against the fixed implementation and reported output here; this subtask did not rerun formal analysis or change its random seed/test method.

## 2. Complete fixed-protocol results

Three seed metrics average within each split;15 splits are the paired comparison units. Difference is LR1e-4 minus original LR2e-4.

| Metric | Original2e-4 | LR1e-4 | Paired difference | Percentile bootstrap95% CI | Improve / tie / worse |
|---|---:|---:|---:|---|---|
| Accuracy | .745604 | .741614 | −.003990 | [−.012139,+.004521] | 6 /1 /8 |
| BA | .719644 | .725133 | +.005489 | [+.000352,+.011512] | 9 /0 /6 |
| AUROC | .759553 | .758559 | −.000994 | [−.007181,+.005459] | 6 /0 /9 |
| Macro-F1 | .706589 | .706159 | −.000431 | [−.006884,+.006345] | 8 /0 /7 |
| PD Recall | .782344 | .764914 | −.017430 | [−.034942,+.001872] | 5 /0 /10 |
| DD Recall | .656945 | .685352 | +.028408 | [+.006280,+.049196] | 11 /1 /3 |

This contrast has a real favorable BA/DD point estimate, with a corresponding PD-recall trade-off. It lacks the frozen joint benefit. It is incorrect to say “no metric improved,” or to discard the positive BA/DD results. It is equally incorrect to select it solely for those two metrics.

## 3. CI, Wilcoxon and BH must remain distinct

- BA mean bootstrap CI excludes zero narrowly, but its two-sided Wilcoxon p=.094604 and two-primary BH q=.189209 do not pass. These are different summaries/test statistics; one cannot substitute for the other or claim FDR-supported BA superiority. The frozen gate uses the BA CI rather than BH, and that criterion **passes**.
- DD Recall CI excludes zero and Wilcoxon p=.027815, but exploratory six-endpoint BH q=.166889 does not pass. Describe a positive development estimate and class-sensitivity trade-off, not a multiplicity-confirmed independent DD-recognition improvement.
- AUROC has a slightly negative mean, CI includes zero, primary BH q=.599487. The candidate fails the operational positive-AUROC gate; the evidence does **not** show statistically established AUROC deterioration.
- Macro-F1 and PD Recall CIs also include zero. Their point estimates nevertheless fail the prespecified nondecrease /−.01 operational limits. Gate failure is not the same as proof of a population-level decrease.

All intervals are marginal descriptive split-bootstrap intervals, not simultaneous confidence regions, external-cohort confirmation or an absence-of-selection-bias guarantee.

## 4. Frozen gate recomputation

| Fixed requirement | Observed | Pass? |
|---|---|---|
| Mean BA>0 | +.005489 | Yes |
| BA improves in≥10/15 | 9/15 | **No** |
| BA CI lower>0 | +.000352 | Yes |
| Mean AUROC>0 | −.000994 | **No** |
| AUROC improves in≥10/15 | 6/15 | **No** |
| Mean Macro-F1≥reference | −.000431 | **No** |
| Mean PD Recall drop≤.01 | Drop .017430 | **No** |
| Mean DD Recall drop≤.01 | Increase .028408 | Yes |
| BA within-split seed SD≤1.25×reference | Ratio1.060953 | Yes |
| AUROC within-split seed SD≤1.25×reference | Ratio1.049820 | Yes |

Several independent components of the registered joint rule fail. The final `REJECT`, retained LR.0002 and stop decision are correct. Do not retrospectively waive the count, AUROC, F1 or PD-recall criteria because BA CI is positive. No additional BH criterion is needed or added to reach this decision.

## 5. Seed variability and prediction agreement

| Metric | Mean within-split seed SD, baseline→LR1e-4 | SD of three per-seed15split means, baseline→LR1e-4 |
|---|---|---|
| BA | .021603→.022919 | .006714→.007247 |
| AUROC | .021741→.022825 | .001812→.004762 |
| Accuracy | .031205→.042347 | .008650→.027812 |
| Macro-F1 | .025605→.033723 | .004947→.020700 |
| PD Recall | .065659→.085058 | .025591→.056650 |
| DD Recall | .078542→.090096 | .035930→.042459 |

The fixed gate uses **mean within-split SD for BA/AUROC only**, both increasing about6.1%/5.0% and staying below1.25×. Do not reject by substituting larger global seed-mean SD increases after results. Conversely, do not claim seed variability universally improved: the full six-metric table increases in both definitions.

Mean three-seed classification unanimity falls .732973→.699774; mean pairwise prediction agreement falls .821982→.799849. Mean pairwise score Spearman rises .773440→.782181. Higher rank correlation and lower classification agreement can coexist; neither is a rescue metric or proof of independent generalization.

## 6. Error-change units and interpretation

These are mean counts **per validation split after averaging three seeds**, then averaging15 splits. Fractional counts are expected. They are not numbers of unique independent people recovered or harmed across the cohort.

| Class | Mean validation count | Original error corrected | Original correct newly wrong | Net corrections |
|---|---:|---:|---:|---:|
| PD | 73.6 | 3.844444 | 5.133333 | −1.288889 |
| DD | 30.4 | 2.400000 | 1.533333 | +.866667 |

For each split, dividing the class net-correction count by that split's class count gives its Recall difference; averaging the15 rates exactly matches PD−.017430 and DD+.028408. Dividing the global mean count by the global mean class count is only an approximation because class counts differ by split. There is no phenotype, stable-error, activity or subject-level causal inference in these tables.

## 7. Training adequacy and loss scope

Median selected/stopping epochs shift10/22→12/24; neither method has a50-epoch cap hit. Both use their own original strict-BA selection rule. Do not call the shift an extra epoch search or compare them as an EMA same-epoch experiment.

| Logged loss | Original2e-4 | LR1e-4 |
|---|---:|---:|
| Best online train CE | .291046 | .358802 |
| Best validation CE | .679387 | .631339 |
| Last online train CE | .019575 | .102103 |
| Last validation CE | 1.099845 | .858135 |

Train values are online, dropout-active; they are not checkpoint train-eval performance. At the candidate's actual last point, train loss falls relative to best in45/45 runs, while validation loss rises in43/45. Reduced loss divergence under lower LR does not establish a better joint classifier; its selected validation CE improves while AUROC does not. Zero cap hits offer no basis to rescue this candidate with more epochs or patience. No intermediate checkpoint or retrospective full EMA is claimed.

## 8. Final mechanism and stopping boundary

Supported statement: **changing the existing frozen-WSSL classifier LR shifts its PD/DD operating behavior and reduces some recorded validation loss, but does not meet the pre-frozen joint retention rule.** It closes the specific LR coverage gap identified in the review.

Unsupported statements: LR1e-4 is a globally better model; all WSSL hyperparameters are now optimal; imbalance/fusion is proved the sole ceiling; lower LR significantly harms AUROC; the BA/DD point estimates are externally confirmed; or the result permits another model/threshold/loss/EMA experiment.

Retain original frozen WSSL-STR, classifier LR2e-4, all prior recipe/architecture/decision rules. Apply the registered stop: no extra LR value, WD compensation, new optimizer/scheduler, epoch/patience extension, differential group, EMA or combination to rescue this result. Prior failed structure/loss/sampling/augmentation/encoder-adaptation routes remain stopped. Repeated development subjects/splits and previous sequential selection limit inference; old FOE-01 information was not consulted and cannot validate this later WSSL recipe contrast.
