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

Historical outer performance exposure and past whole-table/whole-array reading deviations remain permanently disclosed. This round does not restore an untouched outer test. No new outer performance was read, no outer loader was built, and no outer signal was supplied to a target model. Subjects rotate roles across development splits: the union of legitimate validation queries spans 390 people, but every model is restricted to its own validation allowlist. Full metadata rows are parsed for filtering and window-position/length metadata is used for historical numerical batch layout; hashing reads file bytes. These are application-level controls, not OS-enforced physical isolation. The 15 splits share patients and are repeated development evidence, never 15 independent cohorts.

The independent lock uses checkpoint SHA60b6083bcef9fcf5e7a468c03ed01a383eed4368e0fa261d6a7a3b1a72582d73, fixed saved train normalization, preprocessing,0.5 threshold/tiePD and calibratorNONE. No external-label calibration is permitted. See [independent protocol](INDEPENDENT_VALIDATION_PROTOCOL.md) and external_validation_lock.json for all hashes and9 metrics/CI/subgroup definitions.

[Candidate work and explicit stop conditions](NEXT_STAGE_CANDIDATES.md) prioritize cohort authorization, compatible input handling and independent reliability replication, with no new training. Source/protocol/reports/aggregate statistics are published through the existing selective Git workflow; features/checkpoints/raw data/private IDs/predictions/restricted manifests stay on server. Final verification checks protected assets and public scope. This phase stops after publication.
