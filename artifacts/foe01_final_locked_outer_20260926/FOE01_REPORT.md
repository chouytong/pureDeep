# FOE-01: Final Locked Outer Evaluation

Date: 2026-09-26. Dataset: PADS PD (0) versus DD (1), 390 unique subjects, five subject-disjoint outer test folds (80/78/78/77/77; total PD 276, DD 114). The five outer folds are the primary paired analysis units. Three deep seeds (42/43/44) are averaged **within each fold** before model comparisons. H1 is deterministic with one fit per fold. These outer folds are held out from development/model selection, but they are from the same source cohort; this is not an independent external cohort.

## Chronology, frozen protocol, and evidence boundary

Before the first FOE outer outcome was read, the complete protocol and all 30 deep model/seed/fold epoch counts and thresholds were written to `LOCKED_BEFORE_OUTER.json` at 2026-09-26 11:49:33 UTC. Its SHA-256 is `124d091122153ddceced6fb90d59b51611fa9b32ac2c90f5d2087501415eb64b`; the file is read-only and has a SHA sidecar. The prelock audit verified 15 fixed inner development splits, all 90 formal inner prediction/checkpoint records, and outer-train coverage without reading outer outcomes. The split SHA-256 is `b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e`.

STR-01 (75,524 parameters) was locked as the primary model; V8-GN / NR01 (71,026 parameters) as the matched deep reference; H1 as the handcrafted statistical reference. STR and V8 use the unchanged full bilateral 11-activity architecture/recipe, train-only normalization, balanced cross-entropy, AdamW learning rate 2e-4 and weight decay 1e-4, batch size 8, cosine schedule, and seeds 42/43/44. Inner fits had maximum 50 epochs with validation BA early stopping/patience 12. For each outer fold and seed, final refit on **outer-training subjects only** ran the median of that model/seed/fold's three already completed inner best epochs (locked counts span 6–22). The DD decision threshold was chosen solely from the corresponding inner out-of-fold DD probabilities to maximize BA, then fixed before outer testing. AUROC uses continuous DD probability and is threshold-independent. H1 is the frozen handcrafted feature pipeline (imputation, variance filter, scaling, C=1 liblinear logistic regression), fit only on outer-training subjects, with DD threshold 0.5. Primary outer metrics were locked as BA, with accuracy, AUROC, macro-F1, PD recall, and DD recall reported. No diagnostic probe, score correction, fusion, or ambiguity rule was an outer candidate.

The historical accidentally generated outer artifact remained isolated and was never consulted for protocol design, selection, or this report. **outer information was not used for development/model selection and was accessed only after the final protocol was frozen.** Outer outcomes were not used to change architecture, hyperparameters, normalization, epochs, thresholds, training recipe, candidate list, or development hypotheses.

## Execution and verification

The frozen outer-final routine produced 30 deep refit checkpoints and exactly one outer test prediction for each STR/V8 seed/fold unit. H1 was independently refit and evaluated once in each of the five folds. The postrun asset audit passed: all 30 deep units and five H1 folds present; all 390 subjects appear exactly once per seed across the five outer folds; labels and subject IDs align across models; train-only normalization subject IDs match outer train; no train/test overlap; one outer evaluation per unit; protocol SHA unchanged. The metrics script independently re-read prediction files, recomputed all six metrics and confusion matrices, and verified the saved thresholded predictions against the frozen thresholds. No new model search, model correction, or second outer test was performed.

The primary report below uses the **locked inner-OOF thresholds** for deep model thresholded metrics. Development formal metrics used the default 0.5 decision rule, so a separate 0.5 outer view is provided for like-for-like development comparison. H1 uses 0.5 in both.

## Primary outer results

Equal-weight mean over the five outer folds, after averaging the three seeds within each deep fold:

| Model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---|---:|---:|---:|---:|---:|---:|
| STR-01 | 0.7317 | 0.6498 | 0.7218 | 0.6482 | 0.8456 | 0.4539 |
| V8-GN | 0.6905 | 0.6267 | 0.6801 | 0.6163 | 0.7792 | 0.4741 |
| H1 | 0.7281 | 0.6841 | 0.7659 | 0.6804 | 0.7899 | 0.5784 |

Confusion matrices are `[[PD correctly called PD, PD called DD], [DD called PD, DD correctly called DD]]`, summed across the 390 outer test subjects. Deep matrices average the three seed-specific counts, hence fractional entries; they are **not** a seed ensemble or a set of 1,170 independent subjects:

| Model | PD→PD | PD→DD | DD→PD | DD→DD |
|---|---:|---:|---:|---:|
| STR-01 | 233.33 | 42.67 | 62.00 | 52.00 |
| V8-GN | 215.00 | 61.00 | 59.67 | 54.33 |
| H1 | 218 | 58 | 48 | 66 |

Each fold/context, still seed-first for deep models:

| Outer fold | n (PD/DD) | Model | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 80 (56/24) | STR-01 | 0.7208 | 0.6736 | 0.7245 | 0.6653 | 0.7917 | 0.5556 |
| 0 | 80 (56/24) | V8-GN | 0.6875 | 0.6696 | 0.7272 | 0.6536 | 0.7143 | 0.6250 |
| 0 | 80 (56/24) | H1 | 0.7250 | 0.6964 | 0.7768 | 0.6866 | 0.7679 | 0.6250 |
| 1 | 78 (55/23) | STR-01 | 0.7179 | 0.6819 | 0.7212 | 0.6722 | 0.7697 | 0.5942 |
| 1 | 78 (55/23) | V8-GN | 0.6966 | 0.6246 | 0.6630 | 0.6144 | 0.8000 | 0.4493 |
| 1 | 78 (55/23) | H1 | 0.6667 | 0.6119 | 0.7059 | 0.6088 | 0.7455 | 0.4783 |
| 2 | 78 (55/23) | STR-01 | 0.7650 | 0.6478 | 0.7178 | 0.6573 | 0.9333 | 0.3623 |
| 2 | 78 (55/23) | V8-GN | 0.7308 | 0.6573 | 0.7209 | 0.6572 | 0.8364 | 0.4783 |
| 2 | 78 (55/23) | H1 | 0.8333 | 0.7806 | 0.8482 | 0.7913 | 0.9091 | 0.6522 |
| 3 | 77 (55/22) | STR-01 | 0.7186 | 0.6121 | 0.7124 | 0.6151 | 0.8606 | 0.3636 |
| 3 | 77 (55/22) | V8-GN | 0.6277 | 0.5939 | 0.6623 | 0.5683 | 0.6727 | 0.5152 |
| 3 | 77 (55/22) | H1 | 0.6753 | 0.6500 | 0.7446 | 0.6335 | 0.7091 | 0.5909 |
| 4 | 77 (55/22) | STR-01 | 0.7359 | 0.6333 | 0.7331 | 0.6312 | 0.8727 | 0.3939 |
| 4 | 77 (55/22) | V8-GN | 0.7100 | 0.5879 | 0.6270 | 0.5878 | 0.8727 | 0.3030 |
| 4 | 77 (55/22) | H1 | 0.7403 | 0.6818 | 0.7537 | 0.6818 | 0.8182 | 0.5455 |

## Paired comparisons and fold heterogeneity

Differences are STR minus reference. Bootstrap 95% CIs resample the five paired outer folds (10,000 draws, fixed analysis RNG seed); the confidence limits are descriptive because there are only five folds from one source cohort. Improvement count is out of five. All six metrics and the individual fold differences are archived in `analysis/outer_paired_differences_five_fold.csv`.

| Comparison | Metric | Mean paired Δ | Fold-bootstrap 95% CI | Improved folds |
|---|---|---:|---:|---:|
| STR − V8 | Accuracy | +0.0412 | [+0.0256,+0.0664] | 5 |
| STR − V8 | BA | +0.0231 | [+0.0014,+0.0447] | 4 |
| STR − V8 | AUROC | +0.0417 | [+0.0077,+0.0757] | 3 |
| STR − V8 | Macro-F1 | +0.0319 | [+0.0118,+0.0505] | 5 |
| STR − V8 | PD Recall | +0.0664 | [+0.0012,+0.1321] | 3 |
| STR − V8 | DD Recall | −0.0202 | [−0.1187,+0.0819] | 2 |
| STR − H1 | Accuracy | +0.0035 | [−0.0332,+0.0370] | 2 |
| STR − H1 | BA | −0.0344 | [−0.0939,+0.0269] | 1 |
| STR − H1 | AUROC | −0.0441 | [−0.0912,−0.0077] | 1 |
| STR − H1 | Macro-F1 | −0.0322 | [−0.0890,+0.0242] | 1 |
| STR − H1 | PD Recall | +0.0557 | [+0.0241,+0.1066] | 5 |
| STR − H1 | DD Recall | −0.1244 | [−0.2372,+0.0090] | 1 |

STR−V8 AUROC differs by fold: −0.0027, +0.0582, −0.0032, +0.0501, +0.1061. The mean advantage therefore does not mean every context improved; two slight negative contexts and one large positive context matter. STR DD Recall ranges 0.3623–0.5942, and H1 AUROC ranges 0.7059–0.8482. Fold 2 is especially unfavorable to STR DD recognition and ranking versus H1. No outer phenotype or representation mechanism search was performed to explain these patterns.

## Development versus outer

For the **same 0.5 threshold** used in formal development reporting, the 15 inner development splits (deep seeds first aggregated within split) and five outer folds are:

| Model | Development BA | Outer BA at 0.5 | Δ BA | Development AUROC | Outer AUROC | Δ AUROC | Development DD Recall | Outer DD Recall at 0.5 | Δ DD Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STR-01 | 0.6972 | 0.6527 | −0.0445 | 0.7176 | 0.7218 | +0.0042 | 0.6165 | 0.4597 | −0.1568 |
| V8-GN | 0.6707 | 0.6305 | −0.0401 | 0.6800 | 0.6801 | +0.0001 | 0.5631 | 0.4831 | −0.0800 |
| H1 | 0.6918 | 0.6841 | −0.0076 | 0.7515 | 0.7659 | +0.0143 | 0.5900 | 0.5784 | −0.0116 |

The prelocked outer deep thresholds give BA 0.6498 STR and 0.6267 V8, compared with outer 0.5 BA 0.6527 and 0.6305. Thus changing only the prelocked threshold does not explain STR's DD recall loss. At matched 0.5, development STR−V8 DD Recall was +0.0534, but outer STR−V8 DD Recall is −0.0234. Development STR−V8 BA/AUROC gains were +0.0265/+0.0377; outer locked gains are +0.0231/+0.0417. H1−STR AUROC gap is +0.0339 in development and +0.0441 outer.

These development/outer differences are descriptive. The inner validation models were trained on smaller training partitions with early stopping, whereas outer models were final-refit on larger outer-training partitions for preselected fixed epoch counts. They are not a pure estimate of population transport or distribution shift. All comparisons are within the same PADS cohort and five outer partitions.

## Final answers and inference boundary

1. **STR improvement partly holds.** Its outer BA and AUROC exceed V8 by +0.0231 and +0.0417. Accuracy and macro-F1 also favor STR. BA improves in 4/5 folds, AUROC in 3/5; fold variation is material.
2. **Evidence is external to the development model-selection process**, because all 390 outer subjects were held out, the full rule was frozen before access, and only the three prespecified models were evaluated. This is not external-dataset validation. Five-fold CIs are fragile and folds share a source cohort.
3. **DD recognition improvement does not hold.** Outer STR DD Recall is 0.4539 versus V8 0.4741 (paired Δ −0.0202, CI spanning zero), while PD Recall rises to 0.8456 versus 0.7792. Thus the development claim of DD improvement without clear PD sacrifice is not supported on outer; the outer BA benefit has a different class-recall composition.
4. **H1–STR ranking gap remains.** H1 AUROC 0.7659 versus STR 0.7218, paired STR−H1 −0.0441 (1/5 favorable to STR). H1 also has higher point-estimate BA and DD Recall, whereas STR has higher PD Recall.
5. **A generalization limitation is visible in thresholded class balance.** STR BA falls from 0.6972 development to 0.6527 outer at matched 0.5, driven by DD Recall falling from 0.6165 to 0.4597; AUROC is approximately stable (0.7176 to 0.7218). V8 BA also falls, while H1 is relatively stable. This does not identify a specific causal source of the gap.
6. **Paper claim boundary.** The locked outer comparison supports STR as an improved deep model over V8 on mean BA/AUROC within PADS, with fold heterogeneity. It does **not** support a robust DD-recognition improvement, superiority to H1 in AUROC, an external-cohort generalization claim, or a validated information-compression/phenotype mechanism.
7. **Development-only mechanisms.** STR's disease separability, pair-rescue burden, residual ranking complementarity, activity-raw incremental information, ambiguity, and local representation-stage attenuation from DSG/RGD/PRR/PAG remain development-only diagnostic evidence. They were not re-tested as new hypotheses or models on outer. PAG's `ALLOW_NEW_TRAINING = NO` remains unchanged.

No outer-informed re-training, threshold adjustment, model search, representation diagnosis, or reuse of these outer subjects for a revised STR model is allowed. Negative and mixed outer findings are part of the final result.

## Artifact map

- `LOCKED_BEFORE_OUTER.json`, `.json.sha256`, `prelock.log`, `prelock_selection_audit.csv`: immutable protocol and chronology.
- `scripts/freeze_protocol.py`, `run_deep_outer.py`, `run_h1_outer.py`, `audit_outer.py`, `analyze_outer.py`: standalone orchestration and read-only analysis; frozen production forward code was not changed.
- `model_runs/STR-01/`, `model_runs/V8-GN/`: 30 final checkpoints, normalization records, and outer prediction/metric archives.
- `model_runs/H1/`: five fit pipelines and prediction/metric archives.
- `outer_asset_audit.csv`, `outer_audit.log`: postrun boundary and completeness checks.
- `analysis/outer_seed_fold_metrics.csv`, `outer_five_fold_seed_aggregated.csv`, `outer_metric_summary_fold_unit.csv`, `outer_paired_differences_five_fold.csv`, `outer_pooled_390_descriptive.csv`, `outer_prediction_records.csv.gz`, `development_0p5_split_metrics.csv`, `development_15split_seedfirst_mean.csv`, `development_outer_comparison.csv`: reproducible tables and predictions. The 390-subject pooled view is descriptive, not the inferential unit.
