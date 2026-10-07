# Frequency-Prior experiment: interpretation boundaries

2026-10-07. Written after correctness/smoke and before complete formal performance results.

## Confirmed design facts

- Original STR is the requested F0, despite the stronger retained WSSL configuration elsewhere in development history. WSSL is excluded from this comparison.
- The five bands and16D branch are fixed. Clinical frequency descriptions are motivating priors, not deterministic disease classifiers or guarantees of specificity.
- F1/F2 share original initial STR parameter tensors and post-factory RNG. They are independent full training runs under the same original recipe; trained STR checkpoints are not warm starts. No extra pretraining duration is given to candidates.
- Current normalized Acc has already undergone the original L1 trend-removal/trim processing; frequency input is not untreated raw Acc. Both branches use the unchanged data protocol.
- Activity conditioning is through the existing activity identity and ordered STR decision path. There are no new per-activity classifiers or activity/band selection.
- Left/right frequency representations remain separate until original Full bilateral fusion.
- F1/F2 extra parameters differ by only7, but compute, raw-signal access and representation sources differ. This is a near-capacity control, not proof that all differences are caused only by frequency boundaries.

## What results could support

Stable paired F2 benefit over both F0 and F1 would support this complete fixed frequency-prior residual design under repeated development. It would not show a single causal4–6Hz component, general benefit of every frequency method, clinical specificity, independent cohort generalization or superiority over WSSL. Exact causal isolation of masking versus a raw-time residual would require another specifically authorized experiment; none is added here.

Failure would reject or leave inconclusive this implementation, not demonstrate that frequency information is absent/useless. It does not authorize scanning boundaries/dimensions, adding losses, replacing the backbone or combining with WSSL.

## Statistical and numerical limits

The15fixed splits overlap in subjects and training pools. Seed-first split bootstrap/p/BH/effect sizes are development robustness descriptions.45seed-runs or pair counts are not independent sample-size multipliers. Multi-stage reuse introduces selection optimism.

Nominal100Hz, finite-grid masks, full even reflection and ideal noncausal reconstruction are numerical design assumptions. They do not establish actual timestamp interpolation, causal real-time applicability, freedom from ringing or phenotype preservation. All masks/padding are tested under the original finite-placeholder contract.

New train losses are online with Dropout active. They must not be called eval-mode train performance or used alone to quantify a clinical/generalization gap. Correctness/smoke metrics are not performance-selection evidence.

No outer data/performance is used. No claim depends on historical outer outcomes, diagnostic probes, H1 or subtype-informed tuning.
