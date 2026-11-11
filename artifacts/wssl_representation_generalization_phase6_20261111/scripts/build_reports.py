"""Compile reports from completed aggregate analyses; no inference or fitting."""
from common import *
import shutil

def read(name):return pd.read_csv(HERE/'analysis'/name)
def save(name,body): (HERE/name).write_text(body.strip()+'\n')

def main():
 guard()
 # Preserve the gate produced before any forward; enrich the required final table.
 source=HERE/'analysis/paired_representation_asset_audit.csv'
 if not source.exists():shutil.copy2(HERE/'analysis/paired_representation_integrity.csv',source)
 asset=pd.read_csv(source);identity=read('paired_inference_identity.csv')
 merged=asset.merge(identity,on=['context','inner','seed','query_count'],validate='one_to_one')
 assert len(merged)==45
 merged.to_csv(HERE/'analysis/paired_representation_integrity.csv',index=False)
 layers=read('representation_layer_summary.csv');tests=read('representation_tests_planned_BH_sensitivity.csv')
 primary=tests[tests.family=='A_primary'];cluster=read('representation_subject_cluster.csv')
 groups=read('representation_group_subject_cluster.csv');rank=read('representation_seed_rank_summary.csv')
 stability=read('representation_seed_split_stability.csv');btests=read('coverage_interaction_tests.csv');bcluster=read('coverage_interaction_subject_cluster.csv')
 gm=read('coverage_group_metrics.csv');groupmeans=gm.groupby(['label','coverage_group'])[['error_rate','correct_rate','absolute_margin','true_signed_margin','relative_score']].mean().reset_index()
 groupmeans.to_csv(HERE/'analysis/coverage_class_group_summary.csv',index=False)
 hist='''Historical outer performance exposure and past whole-table/whole-array reading deviations remain permanently disclosed. This round does not restore an untouched outer test. No new outer performance was read, no outer loader was built, and no outer signal was supplied to a target model. Subjects rotate roles across development splits: the union of legitimate validation queries spans 390 people, but every model is restricted to its own validation allowlist. Full metadata rows are parsed for filtering and window-position/length metadata is used for historical numerical batch layout; hashing reads file bytes. These are application-level controls, not OS-enforced physical isolation. The 15 splits share patients and are repeated development evidence, never 15 independent cohorts.'''
 save('ADAPTATION_REPRESENTATION_REPORT.md',f'''
# Phase 6A: paired representation audit

All45 pairs passed the asset gate before inference. Fresh inference recovered both archived validation prediction tables with maximum absolute probability error 1.1102230246251565e-16 (fixed tolerance1e-12), all decisions unchanged. 90 validation evaluations, 89.659460s; no training or new fitting. This fills the previous missing paired-feature evidence. It does not establish disease-specific mechanisms.

## Inputs and extraction

Checkpoint, provenance, canonical split, labels, per-split normalization and raw/processed input hashes were checked, rather than inferring pairs from filenames. Activities follow the canonical11-name order, wrists left/right, local branch AccXYZ/GyroXYZ at nominal100Hz. External HarNet uses Acc only: trim48, clip±3g, short edge pad to1000, otherwise first/last1000 windows, resample3/10 to3×300. Local preprocessing and train-only normalization use the protected mainline implementation.

Legacy full-cohort feature/sensor NPZ arrays were never decoded. Raw signals were opened only after per-target validation ID allowlisting. To reproduce original CUDA numerical shapes, metadata fixes64-window batch positions and final-batch shape; other slots contain synthetic zeros, with no excluded subject signal or label. Eval mode prevents cross-row BN updates. Each raw-subject external export is first created only while that person is a legal query; later models fetch only their own query IDs.

Layer4 window maps and means (512), layer5 means (1024), projected residual and final wrist (64 each) preserve subject×11activities×2wrists, window counts and lengths. Final wrist equals local plus projected residual under the explicit saved architecture. Layer1–4 are the same fixed official encoder in both models; layer4 is deliberately a shared zero-drift control. Adapted layer5 is loaded into a plain explicit official layer5 module; no closure/deepcopy substitution. Projected/final changes include independently learned projection/local/head parameters and cannot be attributed solely to HarNet last-block changes.

![Aggregate representation changes and patient-cluster association](figures/representation_summary.png)

## Drift magnitude

Metrics are computed per valid activity/wrist vector and averaged equally across the22 vectors to obtain one patient scalar. Means below average seeds within each split, then15 splits equally; std columns in the CSV are split SD, not independent-patient uncertainty.

{table(layers[layers.label=='all'][['layer','cosine_distance_mean','L2_mean','relative_displacement_mean','frozen_norm_mean','adapted_norm_mean']])}

Dcos=1−cos, L2=Euclidean displacement, Drel=L2/(Frozen norm+1e-12). Zero norms are explicitly handled in code. Raw L2 must not rank changes across different dimensionalities/scales. Layer4 zero is a design check. Layer5 changes are measurable and include norm contraction; neither cosine nor Drel identifies disease information.

## Relationship with prediction change

Primary margin is true-label signed log-odds `(2y−1)*(zDD−zPD)`; change is adapted minus frozen. Predictions, probabilities, margins, flips and correct→wrong/wrong→correct events are matched by subject/split/seed. Do not confuse these appearance-level events with Phase5's patient grouping, which averages3 seed probabilities per appearance and correctness over4 appearances.

Primary3 tests: within each seed/split, Spearman(Drel layer5, signed log-odds margin change), then mean over all3 seeds and all15 units. Split-bootstrap intervals and Wilcoxon/BH are descriptive for correlated development splits.

{table(primary[['test','mean_delta','ci_low','ci_high','wilcoxon_p','bh_q_planned_family_sensitivity']])}

No primary association is supported. Subject-cluster sensitivity changes the estimand to Spearman of each subject's12-appearance scalar means, conditional on fixed fits:

{table(cluster[(cluster.layer=='layer5')&(cluster.outcome=='margin_change')][['label','subject_count','spearman','ci_low','ci_high']])}

The old43 improved/46 worsened/301 unchanged patient grouping was exactly recovered. Improved minus worsened mean layer5 Drel is−0.000088, patient-cluster95%CI[−0.017923,0.017407]. Corresponding projected and final-wrist intervals also cross0. Appearance-group split tests are NA at14/15 complete units; no unit was silently dropped.

DD−PD layer5 Drel averages+0.008108 across15 units,15/15positive, splitCI[0.005299,0.011131], conservative planned-family BH6 q=0.000366. The patient-mean class contrast is+0.008386, clusterCI[−0.001045,0.017585], a different estimand. DD Frozen norms are lower (9.688120 vs PD9.964711), so a denominator contribution cannot be excluded. A disease-specific adaptation mechanism is not established.

Exploratory projected/final-wrist drift–margin correlations are positive for PD (0.159208/0.137567), and weaker/negative for DD. These are exploratory associations, not a direct between-class interaction test or causal localization. The patient-cluster projected PD correlation is0.112451[0.001800,0.223609], whereas the overall interval crosses0. Final-wrist flip correlation is0.105816[0.009043,0.201151] at patient-mean level; a flip can harm or help. These unadjusted exploratory cluster intervals cannot override the primary null.

## Seed versus split stability

`representation_seed_split_stability.csv` separates mean within-split seed SD, between-split SD and SD of3 seed means. Layer5 overall Drel=.396967, mean within-split seedSD=.071072, splitSD=.043329. Magnitudes vary even though patient rankings recur.

An additional post-result descriptive audit computes within-split matched-patient Drel rank correlations between seed pairs42/43,42/44,43/44 (no new hypothesis tests or selection):

{table(rank[rank.label=='all'])}

Layer5 rank correlation averages.982454 (range.970365–.995812), projected.417576, final-wrist.349275. Rank repeatability does not prove stable benefit, anatomical relevance, external transport or absolute magnitude agreement. Layer4 rank is NA because all displacements are0.

## Limits and provenance

{hist}

Features, IDs, raw lineage and per-patient drift/predictions remain server-private. All495 exported feature NPZs have content hashes linked to checkpoint, source, preprocessing, normalization, split/seed/context and fit/query/excluded IDs. No gate was trained from improved patients. See statistical appendix for undefined tests and the explicitly post-result conservative BH sensitivity.
''')
 save('CLASS_CONDITIONAL_COVERAGE_REPORT.md',f'''
# Phase 6B: class-conditional coverage reliability

The direct disease-class×high-reference-distance error interaction is directionally stable in these development records. Its clinical cause is unresolved. High distance is not a clinical anomaly label or a safe universal rejection rule.

## Frozen score definitions

Only Phase5 stored legal query scores are used: u=min(dPD,dDD), existing relative score, and saved split q25/q75 low/high groups. Stored values and group labels were checked unchanged. Neither reference bank, scaler, PCA nor cutoff was fitted/recomputed. All three seeds remain present. Values below are equal split/seed means, not pooled patient estimates.

{table(groupmeans)}

Coverage margin is the original probability margin: abs(pDD−.5), and (2y−1)*(pDD−.5) for true signed margin. This differs from Phase6A log-odds margin and must not be mixed numerically.

## Direct interaction

Error interaction=(PD high−low error)−(DD high−low error). Margin interactions use DD minus PD, as fixed in the protocol. The estimand first averages3 seed contrasts within a split, then15 splits. No claim is based on comparing two separate class p-values.

{table(btests[['test','mean_delta','ci_low','ci_high','wins','wilcoxon_p','bh_q']])}

Error interaction=.439291 (43.9291 percentage points),15/15positive, BH3q=.000183. Split intervals do not make overlapping splits independent.

Patient-cluster bootstrap resamples390 subjects stratified by class, retaining all4 appearances×3seeds per sampled patient; its estimand is pooled exposure high−low contrasts, conditional on fixed fits/cutoffs:

{table(bcluster[['metric','direction','interaction','PD_high_minus_low','DD_high_minus_low','ci_low','ci_high']])}

Pooled error interaction=.422650,95%CI[.237459,.603781]; this corroborates direction while estimating a different weighting from.439291. No retraining/bootstrap refit uncertainty or new-hospital uncertainty is included. Absolute-margin clusterCI crosses0 despite its split result; do not promote it as robust confidence separation.

![Pooled patient-cluster error interaction](figures/coverage_interaction.png)

## Confounders and unresolved explanations

All11 activities and both wrists are present and finite for these390 validation patients, with lengths/window counts retained. Thus binary availability and nominal activity composition do not vary here; they cannot identify effects of completion quality, movement severity, duration differences, timing or device quality. An observed complete task is not proof of identical clinical performance.

All5 context-mean error interactions are positive (.467370,.417666,.542186,.365891,.403344); these use development context labels, not outer performance. Reference banks are balanced PD/DD within split (59–62/class); the within-split class contrast cannot be explained solely by unequal class bank count. Geometry, nearest-neighbor coverage, subtype heterogeneity and fit-sample quality remain unresolved. Bank count varies with context/fit composition and cannot be causally isolated across15 overlapping units.

Every person has4 validation appearances×3seeds, and patient-cluster weighting preserves those repetitions. DD subtype information does not support an individual query-subtype interaction: the public source describes heterogeneous DD, but no new clinical subtype linkage was created. Phase5 did not save per-activity/wrist reference distances, so task/wrist decomposition is BLOCKED_NOT_SAVED. No new scoring, bank or PCA fit was performed to fill that gap.

A plausible interpretation is that this pooled reference-distance geometry has class-dependent meaning: proximity may align differently with PD-like or heterogeneous DD features. This is a hypothesis, not a verified clinical explanation. Correlated training composition, subject difficulty and denominator/aggregation choices could also contribute. An independent compatible PD/DD cohort and pre-specified subgroup metadata are needed to distinguish explanations.

{hist}

No anomaly gate, reject option, threshold search or reference fusion was created. Phase5 fusion remains REJECT, strict WSSL-error relative ranking .416706 remains weak, and overall reference net rescue remains negative.
''')
 save('STATISTICAL_APPENDIX.md',f'''
# Statistical appendix

## Units and estimands

There are45 seed/split models,15 overlapping development splits and390 unique patients (276PD/114DD). Each appears in4 splits and3 seeds. Activities, wrists, windows and repeated appearances are not independent patients. All feature metrics start as22 matched activity/wrist values and reduce to one scalar per patient/model.

Seed stability:3 seeds within a fixed split. Split stability:15 seed-first unit estimates. Subject-cluster uncertainty:10000 class-stratified patient resamples, bootstrap seed20261111, fixed historical fits. External generalization: not estimated. Neither resampling method cures prior development exposure or estimates new training-fit uncertainty. The split-bootstrap CI and Wilcoxon p-values assume a unit structure that overlaps; they are descriptive stability summaries, not independent-cohort confirmation. BH addresses comparison multiplicity, not split dependence or repeated research reuse.

Phase6A primary = equal15-unit mean of3-seed mean within-seed Spearman correlations. Subject sensitivity = correlation of each patient's12 scalar means. Group sensitivity uses the pre-existing4-appearance correctness-difference patient groups. Do not average vectors across different learned coordinates. Phase6B split estimand =15 equal unit contrasts after seed averaging; cluster estimand = pooled exposure ratio contrasts with all appearances retained under patient resampling. These are deliberately distinct, with no opportunistic choice of interval.

## Predefined families and undefined outcomes

The original PROTOCOL and source freeze preceded representation statistics: A_primary3 (layer5×all/PD/DD margin correlations); A_exploratory18 (3variable layers×2outcomes×3classes); A_group6 (3variable layers×improved−worsened/DD−PD); B_interaction3 (error/true margin/absolute margin).

{table(pd.DataFrame(load(HERE/'analysis/undefined_test_accounting.json')['counts']))}

An implementation limitation was found during report verification: original BH adjusted only finite p-values, so exploratory m=15 and group m=3 despite planned18/6. The original outputs are retained in `representation_paired_tests.csv`. A clearly post-result conservative sensitivity in `representation_tests_planned_BH_sensitivity.csv` includes unavailable comparisons as p=1 for adjustment only; their own p/q remainNA, raw p/CI are unchanged. Reports use primaryq=.389404 and conservative planned6q=.000366 for the DD−PD layer5 contrast. This correction is disclosed, not represented as pre-result implementation. B and A_primary were complete from the start.

All3 DD flip correlations lack2 complete units because flip is constant; NA13/15. All3 appearance improved−worsened contrasts lack1 complete unit; NA14/15. No unit, negative result or class was removed to manufacture a full test. No inference is supplied for layer4's constant displacement. Patient-level cluster intervals for exploratory associations are unadjusted, descriptive sensitivities; do not read them as new family-wise discoveries.

## Full split tests with conservative sensitivity

{table(tests[['test','family','status','units','mean_delta','ci_low','ci_high','wilcoxon_p','bh_q','bh_q_planned_family_sensitivity']])}

## Coverage interaction family

{table(btests)}

{hist}
''')
 save('CORRECTNESS_AND_PROVENANCE.md',f'''
# Correctness and provenance

Starting Git SHA161b904462e754a892c3d3d2e4f9ea68b33d18c0. All outputs are new under this Phase6 artifact directory. Mainline, historical formal artifacts, split/norm/checkpoint files and current input files are guarded by21942 SHA256 entries before/after stages. Protected inventories and fit/query/excluded IDs remain in server-only restricted_manifests.

`pre_inference_asset_gate.json`:45/45 pairs pass,0 forward calls at gate. `paired_representation_asset_audit.csv` preserves the pre-inference table; the required `paired_representation_integrity.csv` adds feature hashes and probability identity. No pair was inferred from filename alone. Raw and migrated per-subject processed input hashes were checked. No protected files were overwritten.

`ANALYSIS_FREEZE.json` records the protocol, exact extraction/common/explicit model source and analysis budgets before inference. `STATISTICS_SOURCE_FREEZE.json` precedes drift/coverage statistics but follows successful inference identity. `SUPPLEMENT_SOURCE_RECORD.json` honestly follows statistics: planned group summaries, dataset inventory, post-result conservative BH accounting. `stability_check.py` adds post-result descriptive matched-seed rank repeatability, not a new inferential family. Final manifest hashes every public source/report/aggregate.

Paired correctness:45 explicit-forward/parts checks on first batches,45 input-not-mutated checks and45 forbidden-ID I/O rejection checks. Both complete prediction tables per pair match within fixed1e-12 (observed maximum1.11e-16), decisions identical. All model parameter/buffer fingerprints and official encoder/BN buffers remain unchanged; gradients disabled, optimizer steps0. Identical/scaled synthetic drift hand checks pass. Phase5 score/cutoff groups are unchanged, and fresh inference recovers43/46/301 patient grouping. No new mask perturbation experiment was run; do not infer one from old protected tests.

495 feature NPZs comprise390 legal-query raw external subject exports,15 shared layer4 structural exports,45 paired full exports and45 mapped drift exports. Private feature SHA/source hashes plus raw source paths, preprocessing, normalization, split/seed/context and exact fit/query/excluded IDs are in the restricted lineage. Patient IDs and all arrays stay private. Source/protocol and aggregate scalar tables only are publication candidates.

No dependency installation, package upgrade, new model fitting, PCA/scaler fitting, optimizer/backward step, new calibration or threshold was performed. Inference90 evaluations,89.659460s; drift10.414970s; coverage3.968466s. GPU RTX5090D v2; Python and package versions in runtime_environment.json. Peak memory was not instrumented. No claim of cost measurement beyond these stage wall times. Figures use aggregate outputs and an already-installed isolated matplotlib directory, without changing the formal environment.

The initial final-verifier static check flagged `other.train(model.training)` in the protected explicit-model file's unused copy helper. This was a verifier overreach: extraction instantiates explicit eval models directly and never invokes that helper. The verifier was narrowed to check this exact unused mode-copy call, absence of helper invocations, and absence of fit/backward/optimizer-step call sites. The science source, model weights, inference and original numerical results were unchanged. Final verification was then rerun; this audit-script failure is not hidden as a clean first pass.

## Data access boundaries and disclosures

{hist}

The former full-cohort compressed sensor/feature cache was not decoded this round. Full-cohort ID/order/window-count metadata was used solely to reproduce original CUDA batch positions, with zeros in nonquery slots. Parsing complete metadata CSV rows and reading bytes for hashes is not physical isolation. The data access controls reject an outside query before opening its raw signal. These facts distinguish allowed metadata/provenance access from inference signal input.

External source investigation used official dataset pages/papers, with no data download, credential use, author contact or approval request. Broad web search inadvertently displayed public PADS individual metadata CSV snippets; no active individual-file open/download, persistence or experimental use followed. This new information exposure is disclosed in DATASET_COMPATIBILITY_AUDIT and does not erase historical outer exposure. Future work must preserve both disclosures.

No eligible independent PD/DD dataset has been authorized and verified. The locked single checkpoint is an existing seed42/context0/inner0 model chosen deterministically without performance selection, not a newly fitted all-patient model or45-model ensemble. External validation metrics remain null. See final_integrity.json for the final guard and publication manifest for exact public scope.
''')
 save('REPRODUCE.md','''
# Reproduction on the authorized server

Canonical root `/home/zyt/deep_final`; this artifact directory is `/home/zyt/deep_final/artifacts/wssl_representation_generalization_phase6_20261111`. No data, pretrained weights or packages are downloaded by these scripts. Existing private assets and access authorization are required; the public repository alone cannot reproduce private features.

Use `/home/zyt/envs/mfam_analysis/bin/python -B` and four CPU threads. Run from this artifact directory, with `PYTHONPATH=scripts`. The original completed run order was bootstrap_audit.py (gate/source lock), paired_inference.py, drift_analysis.py, coverage_analysis.py, supplement_checks.py, external_inventory.py, stability_check.py, build_reports.py, verify_final.py. Statistical source freeze JSONs were created before their indicated stages and must not be silently replaced or backdated. The saved USER_REQUEST/PROTOCOL, source hashes and final manifests are the reproducibility record. The descriptive stability/report/final verification scripts intentionally follow results.

Do not rerun into the finalized directory: outputs would be overwritten and the publication snapshot would diverge. Create a separate new artifact directory and explicitly adapt HERE/root constants and locks in a new authorized audit if reproduction is needed. Do not alter historical source/model/norm/split assets or use another fold/seed's feature arrays. Failed identity/allowlist/provenance checks must stop extraction, not loosen tolerance. No training is part of this procedure.

Scientific outputs are aggregate CSV/JSON and Markdown. features/ and restricted_manifests/ are server-private, include identifiers and fitted/derived arrays, and must never enter Git. The existing selective publisher is `/home/zyt/deep_final/tools/git_publish/publish_to_github.py`; its privacy, size, stale-file and protected-history checks stay active. A connected GitHub transport may publish its exact audited tree if server HTTPS credentials are unavailable; tree SHA and parent must match before updating the remote ref.
''')
 save('NEXT_STAGE_CANDIDATES.md','''
# Candidate work only; no Phase7 started

| Priority | Candidate | Rationale | Cost | Prerequisite / stop condition |
| --- | --- | --- | --- | --- |
| 1 | Authorized independent clinical PD/DD cohort and blinded metadata inventory | Main missing evidence is transport, not more modules | Clinical/data coordination; no GPU training | Stop if permissions, overlap audit, DD diagnosis or task/device comparability cannot be verified |
| 2 | Validate format-only input adapter and locked inference against nonclinical synthetic fixtures | Single checkpoint/protocol already locked | Small engineering/inference budget; training0 | Stop on channel/unit/rate/task mismatch; never impute invented activities to claim comparability |
| 3 | Independent pre-specified class-conditional reliability replication | Direct interaction now documented, mechanism unresolved | Existing-model inference and patient-cluster statistics | Eligible cohort and fixed old algorithm only; no new best cutoff/gate |
| 4 | If separately authorized, recover old per-activity distance decomposition from traceable existing algorithm | Current Phase5 distances were only aggregated | Storage/provenance audit first; training0 | Stop if old metric cannot be exactly reconstructed without refit or identity loss; no work initiated here |
| Deferred | Mechanism-discriminating study of norm contraction, DD subtype and movement quality | Current drift magnitude does not explain benefit | New labels/design; training not justified yet | Must specify falsifiable prediction and independent evaluation before proposing any new training |

Stop reopening failed Adapter/gated fusion/masked adaptation, Patch-STR or dual-reference fusion under new names. No patient-error gate, anomaly/reject option, validation-selected layer/rank/threshold or CE search. CE fixed-denominator remains NOT_RUN_GATE_FAIL rather than a measured negative. The present findings do not justify starting a new training run. This phase ends with publication and stops.
''')
 save('FINAL_REPORT.md',f'''
# Phase 6 final report: representation mechanism and generalization audit

Frozen WSSL-STR remains the retained single model. The paired representation gap is closed:45 pairs/90 validation evaluations passed, max probability error1.11e-16. Layer5 representation changes are reproducible, but drift magnitude does not reliably explain which patients improve. Class-conditional reference reliability has a direct, stable development interaction; its clinical cause and external validity remain unresolved. No new training or fitting was performed; no Phase7 is started.

Formal retained development metrics are unchanged: Accuracy.745604, BA.719644, AUROC.759553, Macro-F1.706589, PD Recall.782344, DD Recall.656945. These are historical retained-model summaries, not an external cohort result or the new locked single-checkpoint performance.

## Complete experiment state

| Work | Status | Evidence / boundary |
| --- | --- | --- |
| Historical reports and current source audit | COMPLETE | history_sources.csv;21942 protected hashes |
| 45 paired asset gate | PASS45/45 | Exact provenance/split/norm/raw input identity before forward |
| Paired layer4/layer5/projected/final wrist exports | PASS45/45 |90 evaluations; full11×2 structure, private features |
| Drift / margin / flip / group statistics | COMPLETE with declared NAs | Primary3 complete;3 DD flip and3 appearance group tests undefined |
| Direct high-distance×class interaction | COMPLETE | Existing Phase5 scores/cutoffs only; split and patient-cluster estimands |
| Activity/wrist reference-distance decomposition | BLOCKED_NOT_SAVED | No per-activity reference scores; no new fitting/rescoring |
| Task completion quality / clinical DD subtype mechanism | UNRESOLVED | Complete inputs do not identify clinical quality or heterogeneous diagnosis effects |
| Independent dataset investigation | COMPLETE inventory | No eligible authorized compatible independent PD/DD cohort verified |
| Frozen independent validation protocol | LOCKED_PROTOCOL_ONLY | Existing seed42/context0/inner0 checkpoint, norm, preprocessing,0.5 threshold, no calibrator |
| Actual independent cohort evaluation | NOT_RUN_NO_ELIGIBLE_DATA | All external performance fields null; adapter not yet validated |
| New deep training / new fitting |0 /0 | Runtime/checkpoint/BN/source checks |

## Hypothesis evidence

| Hypothesis | Evidence | Judgment |
| --- | --- | --- |
| H1 adaptation changes middle representation | Layer5 cosine.085248,Drel.396967,norm9.883820→7.942572; layer4 shared0 | Supported for numerical representation change; disease-specific information gain unproven |
| H1 drift predicts benefit | Layer5 drift–margin r=.021579,splitCI[−.005612,.048844],BH3q=.389404; patient r=−.012639,CI[−.109757,.085344] | Not supported; null evidence is not proof of exact0 |
| H1 stable patient changes | Layer5 matched-seed Drel rank r=.982454 mean,post-result descriptive | Rank repeatability supported; benefit repeatability not established |
| H1 PD/DD difference | DD−PD Drel+.008108 split; patientCI[−.001045,.017585] and norm-denominator concern | Limited descriptive difference, no causal class-specific mechanism |
| H2 class-conditional reliability | Error interaction+.439291,15/15positive,q=.000183; pooled patient-cluster+.422650[.237459,.603781] | Stable development interaction; not independent confirmation |
| H2 why PD/DD directions oppose | Nominal11tasks/bothwrists complete; bank count balanced59–62/class; quality/geometry/subtypes unknown | Cause unresolved; high u cannot be called clinical anomaly |
| H3 independent generalization | Current15splits overlap390 subjects; external sources not eligible/authorized/compatible | No independent generalization claim |

## Answers to the eight research questions

1. Adaptation changes layer5 and downstream vectors, while shared frozen layer4 is unchanged by design. Calling the changed coordinates disease-related information would exceed the evidence: nuisance change, norm contraction and head/projection differences remain possible.
2. Greater layer5 drift has no supported primary relationship with margin improvement. Improved43 versus worsened46 patients have nearly identical layer5 Drel (.401333 vs.401421); cluster difference−.000088[−.017923,.017407]. The301 unchanged patients and all negative/undefined results remain included.
3. DD's average relative layer5 drift is slightly higher; lower baseline norms may contribute. Projected/final correlations show exploratory class patterns, but separate within-class significance is not a between-class effect test and cannot explain a disease mechanism.
4. The PD/DD high-u error relationship is directly different: pooled PD high−low+.148441 versus DD−.274209. Why this occurs is not identified. Distance measures similarity to these particular fitted reference geometries, not an ordered clinical abnormality axis.
5. Activity/wrist availability and unequal within-split class-bank counts cannot alone explain the pattern because they do not vary in the required way. Movement quality, duration, fit composition/coverage geometry and DD heterogeneity remain plausible; per-activity distance decomposition is blocked by absent saved scores. Subject repetition is handled by cluster weighting but does not create an external cohort.
6. Input alignment, exact predictions, numerical drift and direct interaction are verified. Disease-specific causal localization, task/wrist reference decomposition and independent external performance remain unproven or BLOCKED as specified above. Do not carry forward the old blanket feature-extraction block: paired inference succeeded this round.
7. The evidence motivates a falsifiable reliability/norm/clinical-quality study, but provides no sufficient basis for another training search. Priority is independent data and adapter validation. No new training candidate is launched and all failed fusion/module searches remain closed.
8. The validation protocol and deterministic single existing checkpoint are locked, but necessary independent authorized compatible data and a validated external format adapter are missing. Dataset descriptions are not access grants; protocol readiness is not validation completion.

## New findings, limitations and next actions

The new engineering finding is legal raw-file reconstruction of all90 prediction tables without decoding the legacy full-cohort feature arrays. The new research findings are actual layer5 change with little primary drift–benefit association, stable layer5 patient drift rankings, and a directly tested class-conditional coverage interaction. None resolves individual clinical failure mechanisms or external transport. See [adaptation report](ADAPTATION_REPRESENTATION_REPORT.md), [coverage report](CLASS_CONDITIONAL_COVERAGE_REPORT.md), [statistics](STATISTICAL_APPENDIX.md) and [correctness/provenance](CORRECTNESS_AND_PROVENANCE.md).

Independent data inventory: current PADS is the same source cohort; its79 healthy controls must not be relabeled DD. mPower and BEAT-PD do not establish the required11task bilateral PD/DD protocol. WATCH-PD is PD/control, with proposal/consortium access not granted here. New-hospital/time/device cohorts remain unknown. Official source links and grant/compatibility fields are in [dataset audit](DATASET_COMPATIBILITY_AUDIT.md). No external dataset was downloaded or used for scoring.

{hist}

The independent lock uses checkpoint SHA60b6083bcef9fcf5e7a468c03ed01a383eed4368e0fa261d6a7a3b1a72582d73, fixed saved train normalization, preprocessing,0.5 threshold/tiePD and calibratorNONE. No external-label calibration is permitted. See [independent protocol](INDEPENDENT_VALIDATION_PROTOCOL.md) and external_validation_lock.json for all hashes and9 metrics/CI/subgroup definitions.

[Candidate work and explicit stop conditions](NEXT_STAGE_CANDIDATES.md) prioritize cohort authorization, compatible input handling and independent reliability replication, with no new training. Source/protocol/reports/aggregate statistics are published through the existing selective Git workflow; features/checkpoints/raw data/private IDs/predictions/restricted manifests stay on server. Final verification checks protected assets and public scope. This phase stops after publication.
''')
 write_json(HERE/'decision.json',dict(retained_model='Frozen WSSL-STR',model_changed=False,new_deep_training=0,new_fits=0,paired_pairs=45,inference_evaluations=90,representation_change_verified=True,disease_specific_mechanism_proven=False,primary_drift_benefit_supported=False,class_coverage_interaction_development_supported=True,clinical_causal_explanation_proven=False,activity_wrist_reference_decomposition='BLOCKED_NOT_SAVED',independent_validation='PROTOCOL_ONLY_NO_ELIGIBLE_AUTHORIZED_COMPATIBLE_DATA',start_phase7=False,historical_outer_exposure_permanent=True))
 guard();print('REPORTS_BUILT')

if __name__=='__main__':main()
