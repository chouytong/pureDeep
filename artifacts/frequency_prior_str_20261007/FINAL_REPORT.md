# Fixed Frequency-Prior STR: final controlled development result

Date: 2026-10-07 (Asia/Shanghai). **COMPLETE / REJECT F2.** Retain original STR-01 for this comparison. The overall retained WSSL-STR configuration is unchanged; it was not a model in this experiment.

## 1. Question and completed work

Does a fixed, activity-preserving, bilateral, five-band learned wrist residual provide stable PD/DD classification value beyond original STR and a near-capacity residual control?

Completed original-model/data/config/history review, all45F0full-validation/15normalization audits, synthetic filter-bank tests, F1 then F2 implementation, all45frozen checkpoints×two candidates90zero-init/reload cases, fresh3seed/RNG and scratch gradient/mask/independence tests, both original-engine smoke runs, pre-result Git freeze, full45F1+45F2training and frozen15split seed-first analysis. No other performance candidate, WSSL combination, pretrained adaptation, auxiliary loss, sampler, augmentation, threshold or recipe search.

The90formal candidate runs ended normally, UTC2026-10-07T14:11:25.912562 to14:35:08.590697 (about23minutes43seconds). F0 reused45existing matched STR runs; **no new F0 training was performed**. All artifacts are independent of original formal outputs.

## 2. F0 identity and source-provenance correction

F0=original STR-01,75,524trainable parameters; fixed processed/normalized Acc+Gyro, nominal100Hz, true lengths976/2000,11activities and Left/Right order, Full bilateral258, original activity-ID attention and ordered11×16structured residual, final base+structured logits. Acc already underwent original full-length L1 detrending and trim48; this experiment does not operate on untreated raw Acc.

Initial audit checked all45frozen weights/configs/ID-aligned predictions/metric implementation,15actual train-only normalization refits, and existing45Phase2exact reproduction states. Full-validation re-inference matched original archived probabilities to maximum1.11e-16. All selected states, best epochs and predictions matched the existing reproduced STR.

**One audit sequencing omission must be disclosed:** the initial check did not directly compare the checkpoint's whole-tree source hash. Final provenance review found original formal hash `4fd74c01a05dbb743dea943264fbe9d52ced140379e0f23c1bb945452fefbc4c`, versus current `fae112dcf6d05de5f94bdbe0c7e2b3748d79d343200f441762f4dd8db82f6617`. Historical whole-tree hashes are available; earlier wording implying missing historical source provenance is corrected.

All45already completed Phase2baseline checkpoints have the **exact current whole-tree source hash**, and their model/data/training/evaluation/loss/nested/self-supervised config sections, selected model-state tensors, normalization, best epochs and validation IDs/labels/decisions/probabilities match original STR exactly. They therefore provide a reusable, strictly current-source matched F0. The cached original F0 predictions in frozen arithmetic are numerically identical to this matched baseline; all comparisons and the REJECT result are unchanged. No redundant45run training was launched.

This direct source comparison was supplemented **after** the initial F1/F2 analysis, not before it. It is a disclosed procedural timing limitation, not a post-hoc configuration/statistical change. The tree manifest also includes configs/scripts/tests and selected documents; a changed whole-tree hash does not establish a changed active forward calculation. Without a historical file-level manifest we do not assign the change to particular files. See [source correction and45run evidence](SOURCE_PROVENANCE_COMPLETION.md).

## 3. Implemented conditions and capacity

|Condition|Design|Total/trainable params|Added params|
|---|---|---:|---:|
|F0|Original STR-01|75,524|0|
|F1|Wrist64→Linear64to17→GELU→Linear17to16→GELU→zero-initLinear16to64→wrist residual addition|78,005|2,481|
|F2|Current6channel wrist→fixed5bands→each depthwise6/kernel15 + pointwise6to8 + GELU + temporal average pooling→concatenate40→Linear40to16+GELU→zero-initLinear16to64→wrist residual addition|77,998|2,474|

Added capacity differs by7parameters, about.28% of branch capacity. No dimension scan. All original STR parameters remain trainable. Candidates start independently from the same original random initialization at each seed, as explicitly confirmed by the user; trained checkpoints are consistency references, not warm starts. New branch initialization uses local CPU `fork_rng`, preserving original parameter initialization and subsequent RNG; new branches have no stochastic modules.

F2 bands are [.5,3),[3,4),[4,6),[6,8),[8,12]Hz. Each true-length signal is even-extended `[x,reverse(x)]`, transformed at2L, masked with disjoint fixed bins, inverse-transformed and cropped. No bandpower/PSD peak/entropy/statistical feature input or fixed band-to-disease rule. This returns learned time-domain band representations, rather than rerunning the rejected gyro log-magnitude/interpolated-bin rFFT branch.

Both wrist and activity identities are retained. Injection precedes original bilateral fusion; no Left/Right or activity averaging occurs beforehand. Activity conditioning is supplied by the unchanged downstream activity-ID and ordered STR paths, with a shared per-wrist branch. No extra activity classifier, asymmetry statistic or attention/gate is added.

**Control limitation:** F1 matches capacity, small16D embedding, nonlinear activation, zero-init residual and injection site, but consumes the original learned wrist vector rather than raw filtered time signals. It does not isolate frequency masking from every consequence of adding another raw-signal path. Neither control performance nor this experiment establishes general equivalence of residual architectures.

## 4. Correctness, smoke and frozen protocol

- Synthetic filter tests at976/2000: shapes[N,5,6,L], bin membership/edges, DC/out-of-band rejection, exact even-extension cosine modes, finite nonzero autograd and NaN/Inf rejection PASS. Resolution.0512295/.025Hz; bins perband49/20/39/39/78 and100/40/80/80/161. Fixed filter has no learned parameters.
- All45frozen checkpoints×F1/F2: each8validation subjects, **zero-init logits exactly equal original**, state save/reload exact. This is90sampled cases, not every possible input; F0full validation itself was checked completely.
- Fresh original state and post-factory CPU RNG exact for3seeds. Independent model copies have different parameter storage; mutating one does not alter another output.
- Two scratch inner-training updates percondition: first upstream gradients are exactly zero behind zero-init residual; after first projection update upstream gradients are finite/nonzero. All6F1/19F2branch state tensors update. Scratch models are not performance candidates or saved results.
- After updates, finite ignored padding/absent-wrist and NaN absent-activity perturbations leave logits exactly unchanged. Valid nonfinite input is rejected. The original check still requires finite placeholders for an absent wrist within a valid activity; this contract was not silently weakened.
- F1/F2 original-engine one-epoch/one-batch smoke PASS, selected checkpoint reload, normalization/ID/metric/recipe audits PASS. Smoke scores are not used for design or scientific performance. A preliminary manual metric assertion expected AUROC.5 incorrectly; the known-answer3of4pair case was corrected to.75 before freezing, with production metrics unchanged.
- All90formal run audits PASS: original recipe/loss, train-only normalization, seed/split/labels, parameter count, strict ordinaryBA-selected earliest best, cap/patience, checkpoint SHA/last epoch, saved metric and fixed argmax/tiePD consistency. Final source/checkpoint/prediction/normalization/manifest/preprocessing/protocol guards PASS; no GPU compute process remained at closure.

Original recipe: balanced train-fold CE/no smoothing, AdamW2e-4/WD1e-4/betas.9,.999, batch8, FP32, clip5, cosine50floor1e-6, max50, BAearly stopping/patience12/minimum_delta0, seeds42/43/44. Threshold.5, exact tiePD. No EMA or pretrained model. Each condition selects its own checkpoint under this unchanged ordinaryBA rule.

Runtime: torch2.8.0+cu128, numpy2.2.6, pandas2.2.3, scipy1.15.3, scikit-learn1.6.1. No dependency installation or version change was made.

## 5. Formal performance: three seeds averaged within each split

|Condition|Accuracy|BA|AUROC|Macro-F1|PD Recall|DD Recall|
|---|---:|---:|---:|---:|---:|---:|
|F0|.730718|.697189|.717631|.686068|.777864|.616515|
|F1|.729412|.692987|.718699|.683594|.780863|.605112|
|F2|.724912|.694125|.717512|.682089|.768485|.619764|

### Primary BA paired comparisons

|Comparison|MeanΔBA|Win/tie/loss|Bootstrap95%CI|Paired dz|Rank-biserial|Wilcoxon p|BH q(3BA tests)|
|---|---:|---|---|---:|---:|---:|---:|
|F2−F0|−.003065|4/2/9|[−.006134,−.000270]|−.512|−.560|.074735|.224206|
|F2−F1|+.001137|10/0/5|[−.001803,+.004778]|+.168|+.083|.803955|.803955|
|F1−F0|−.004202|6/0/9|[−.008646,−.000257]|−.488|−.433|.151428|.227142|

F2 does not improve F0. Ten wins againstF1 alone do not establish stable extra value: effect magnitude is small, CI crosses0 and primaryBH does not pass. F1 does not provide BA benefit overF0 either; the results do not support the pattern “extra capacity improves both models.” They also do not constitute a statistical equivalence test between F1 and F2.

F2−F0 BA/F1 and F1−F0BA bootstrap intervals are negative, but Wilcoxon/BH results do not confirm FDR-significant deterioration. We report both; a bootstrap interval is not interchangeable with a Wilcoxon/BH test. **REJECT is the prespecified retention decision, not a claim of proven harmfulness of all frequency information.**

### Secondary comparisons

|Comparison|Metric|Mean delta|Win/tie/loss|Bootstrap95%CI|Wilcoxon p|BH q(18 tests)|
|---|---|---:|---|---|---:|---:|
|F2−F0|Accuracy|−.005807|3/2/10|[−.012236,+.000976]|.132721|.389387|
|F2−F0|AUROC|−.000118|9/0/6|[−.005462,+.004714]|.638672|.766406|
|F2−F0|Macro-F1|−.003978|4/1/10|[−.007002,−.000720]|.047990|.336310|
|F2−F0|PD Recall|−.009379|4/3/8|[−.026867,+.008927]|.310897|.587669|
|F2−F0|DD Recall|+.003250|6/3/6|[−.018591,+.024755]|.694887|.781747|
|F2−F1|Accuracy|−.004500|3/3/9|[−.009189,+.000191]|.070972|.336310|
|F2−F1|AUROC|−.001187|7/0/8|[−.005229,+.002777]|.599487|.766406|
|F2−F1|Macro-F1|−.001505|6/0/9|[−.004763,+.001789]|.359131|.587669|
|F2−F1|PD Recall|−.012378|3/3/9|[−.024732,−.000312]|.059739|.336310|
|F2−F1|DD Recall|+.014652|8/2/5|[−.000569,+.031605]|.132010|.389387|
|F1−F0|Accuracy|−.001307|6/1/8|[−.007701,+.004465]|.777351|.803955|
|F1−F0|AUROC|+.001068|10/0/5|[−.002918,+.004941]|.513571|.766406|
|F1−F0|Macro-F1|−.002473|7/0/8|[−.006608,+.001486]|.330261|.587669|
|F1−F0|PD Recall|+.002999|7/3/5|[−.012925,+.018010]|.582629|.766406|
|F1−F0|DD Recall|−.011403|6/1/8|[−.031565,+.008884]|.314817|.587669|

All18exploratoryBH q≥.3363. F1's small AUROC increase and F2's small DD increase are positive point estimates, not stable improvements. Full paired dz/rank-biserial for all18comparisons are in [paired_comparisons.csv](analysis/paired_comparisons.csv).

## 6. Seed variation, agreement and trajectories

SD below is across the3seed means, each mean first taken across15splits:

|Condition|Accuracy SD|BA SD|AUROC SD|Macro-F1 SD|PD Recall SD|DD Recall SD|
|---|---:|---:|---:|---:|---:|---:|
|F0|.015091|.011699|.011980|.010743|.032733|.040803|
|F1|.014869|.013642|.015552|.012128|.031803|.043573|
|F2|.011381|.015099|.015991|.012249|.023774|.043092|

|Condition|BA mean within-split seedSD|AUROC mean within-split seedSD|BA splitSD|AUROC splitSD|Seed unanimity|Mean seed-pair agreement|Mean seed score Spearman|
|---|---:|---:|---:|---:|---:|---:|---:|
|F0|.030011|.032350|.031579|.039340|.644221|.762814|.600027|
|F1|.027343|.032809|.030771|.039269|.655686|.770458|.606008|
|F2|.030468|.029809|.029164|.037046|.638377|.758918|.594668|

F2 passes the predeclared1.25ratio guard using **mean within-split seedSD**. Its SD across3seed means is higher for BA/AUROC; these different dispersion definitions must not be mixed or substituted post-hoc. Agreement/correlation also do not show a stable F2 advantage. Remaining metric splitSDs and all3seed values are in `summary.csv`/`seed_means.csv`.

F2 seed42BA/AUROC .710452/.733150 exceed matched F0 .708392/.727649 slightly; seed43and44both have lower BA/AUROC. F2 Macro-F1 is lower in all3seed means. Selecting seed42would give a misleading conclusion; no seed was selected.

|Condition|Best epoch range/median|Stop epoch range/median|Epoch50 cap hits|Best online train CE|Best validation CE|Last online train CE|Last validation CE|
|---|---|---|---:|---:|---:|---:|---:|
|F0|6–31/12|18–43/24|0|.338780|.714679|.041499|1.086545|
|F1|4–33/12|16–45/24|0|.382016|.700528|.045922|1.136639|
|F2|4–42/12|16–50/24|1|.343236|.710403|.045715|1.075925|

These train CE values are online with Dropout active, not eval-mode train performance. Slightly lower selected validation CE does not establish improved BA or calibration. One F2run reaches the existing cap; this does not authorize changing max epochs/patience or retraining. All selections/stops follow the original rule.

## 7. PD/DD trade-off and error changes

F2−F0 DDRecall +.003250 (about+.325percentage points) is small and unstable, alongside PDRecall−.009379. Relative toF1, DDRecall+.014652 accompanies PDRecall−.012378, exceeding the frozen1ppPD-drop guard. This is a class-recognition trade-off, not reliable DD-recognition improvement. No subtype was selected/reweighted.

Seed-average within each split, then average15splits; counts are per-validation-unit means, **not independent people**:

|Comparison|Class|Reference errors|Candidate errors|Corrected|Harmed|Net corrected|
|---|---|---:|---:|---:|---:|---:|
|F2−F0|PD|16.356|17.044|1.600|2.289|−.689|
|F2−F0|DD|11.667|11.578|.822|.733|+.089|
|F2−F1|PD|16.133|17.044|1.022|1.933|−.911|
|F2−F1|DD|12.022|11.578|.933|.489|+.444|
|F1−F0|PD|16.356|16.133|1.956|1.733|+.222|
|F1−F0|DD|11.667|12.022|.667|1.022|−.356|

## 8. One-time decision and conditional analyses

**REJECT F2 under the unchanged pre-result rule.** F2−F0 fails positiveBA,≥10splitBAimprovements, positiveBA bootstrap lower bound, primaryBH and Macro-F1 nondecrease. F2−F1 has10wins but fails positiveCI/primaryBH, Macro-F1 nondecrease and thePDRecall-drop guard. AUROC and the fixed within-split seedSD margins pass; they do not rescue failed primary/recall criteria.

The prespecified positive-trend explanation gate is false. Consequently no activity/band importance, bilateral residual phenotype or DD subtype search was performed. We did not find a favored activity/band or use error patterns to redesign the branch. Aggregate class error counts above are the required core performance comparison.

This fixed five-band implementation does **not** establish additional stable/generalizable PD/DD decision value beyond STR and the matched capacity control. It does not prove frequency information useless or absent, identify a4–6Hz causal reason, or show F1andF2equivalent. Redundancy, optimization effects and representational limitations remain unseparated; no new mechanism training is inferred from them.

**STOP this configuration. No band/hidden/filter/loss/recipe/threshold rescue search. No WSSL+Frequency combination.** Original STR and overall retained frozen WSSL remain unchanged. The proposed next-stage pretrained-plus-frequency comparison has not met its prerequisite.

## 9. Evidence and statistical boundaries

15fixedsplits are the primary comparison units after averaging3seeds; they overlap in subjects/training pools and contexts. The45runs percondition, reusedsubject appearances, activities/windows and pairs are not independent sample-size multipliers. Bootstrap10,000paired draws(seed20261007), Wilcoxon/BH/effect sizes describe repeated development robustness, not independent external inference. Multi-stage development reuse introduces selection optimism and may understate uncertainty.

Only restricted inner-train/validation signals, predictions and fitted statistics enter this experiment. Canonical partition membership/original manifest metadata and checkpoint provenance were used for protocol/identity checks; excluded context test records do not enter normalization, training, inference or metrics. No outer-final/test signal loader, outcome prediction or performance was accessed, and no historical outer conclusion was used to choose this design.

No clinicalphenotype-preservation, causal real-time applicability or new cohort/clinical-severity claim follows. Nominal100Hz, finite grids, ideal noncausal masks/ringing, full even reflection and the original Acc preprocessing are specific implementation assumptions. Branch amplitudes would not be causal importance even if analyzed.

|Boundary|Actual status|
|---|---|
|Outer signal/prediction/performance used|NO|
|Original training recipe/loss/sampler/augmentation changed|NO|
|Threshold changed/calibrated|NO|
|Warm-start trained STR for candidates|NO|
|WSSL/HarNet/H1/handcrafted/teacher used|NO|
|Post-hoc hyperparameter/statistical selection|NO|
|Historical source-tree direct audit before formal training|NO; completed during final closure, existing exact-current-source matchedF0verified45/45, comparison unchanged|
|F2 retained|NO|
|WSSL+frequency prerequisite met|NO|

## 10. Files, preservation and Git

Independent scripts: `common.py`, `audit_f0.py`, `filter_bank.py`, `test_filter_bank.py`, `capacity_control.py`, `prior_model.py`, `test_f1.py`, `frequency_branch.py`, `test_models.py`, `run_condition.py`, `launch_matrix.py`, `analyze_results.py`, `freeze_study.py`, `test_analysis_contract.py`; post-result supplementary provenance audit `audit_matched_f0.py`. Production foundation source/model/config files were not changed. No original checkpoint/prediction/result was overwritten. New checkpoints/private predictions/logs stay in the server artifact directory.

Reports: F0_AUDIT_REPORT, FILTER_BANK_REPORT, F1_IMPLEMENTATION_REPORT, IMPLEMENTATION_TEST_REPORT, SMOKE_REPORT, PROTOCOL, SOURCE_PROVENANCE_COMPLETION, DESIGN_REVIEW and this FINAL_REPORT. Aggregate analysis CSV/JSON includes135metric rows(45eachF0/F1/F2),45seed-first condition/split rows,18paired comparisons,90candidate run audits, seed/variance/agreement/error tables and source/F0/final integrity checks.

Milestones already pushed/tagged before final analysis:

|Stage|Commit|Tag|
|---|---|---|
|F0 audit/protocol|093a2ab1d5fb3b26641a22c4400bf7210fa49639|frequency-prior-f0-audit-20261007|
|Filter numerics|2b5d0efe696e178161fced548507676efb871c53|frequency-prior-filter-pass-20261007|
|F1 implementation|05bd912785fd80a4e3c63f9cb0c649f5c22cb2fe|frequency-prior-f1-implementation-20261007|
|F2/model tests|bc65630a649532f54e6292019209ba9e80e83fe3|frequency-prior-f2-tests-pass-20261007|
|Preformal code/analysis/smoke freeze|ddfae6fca6d06fcb103e19240c04a22f7c547591|frequency-prior-protocol-smoke-20261007|

Final report/all positive and negative findings/source correction/statistical limits/stop decision are appended to README and handoff and published with a final tag. Final SHA/tag verification is recorded in an external publication receipt to avoid self-referential commit hashes. Datasets/checkpoints/weights/features/caches/individual prediction/clinical tables are excluded.
