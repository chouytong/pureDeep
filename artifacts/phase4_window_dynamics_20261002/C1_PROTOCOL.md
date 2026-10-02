# Item 07: eligible DD source-category auxiliary supervision

Registered before C1 training. Eligibility passed item06: verified consistent source categories, N≥10, Other–ET consensus recall difference bootstrap CI excludes zero and six-pair BH q=.029499. Categories are broad PADS source diagnoses, not granular clinical subtype labels.

Independently start from the original frozen WSSL initialization/recipe. Use original detrended local Acc, original frozen HarNet mean cache and all original PD/DD paths. Add only a training auxiliary Linear(258,4) on the original subject embedding. Standard PyTorch Linear initialization; 1,036 extra trainable parameters. PD/DD output calculation and .5 decision rule unchanged. No auxiliary output required at inference.

Category order: Other Movement Disorders / Essential Tremor / Atypical Parkinsonism / Multiple Sclerosis. No category exclusion or label invention. Main original train-balanced CE + 0.1 × unweighted 4-class CE, averaged over DD examples in the current training batch. No DD in batch → no auxiliary contribution/gradient. Only training DD labels are attached. Validation/test/inference auxiliary loss zero and no subtype targets or labels required. No loss-weight, head or sampler search.

Same 15 splits, seeds42/43/44, train-only normalization and original training/early-stop settings. Technical tests and formal one-batch smoke precede 45-run training. Retention gates exactly those in PROTOCOL.md. Comparison only C1 vs original WSSL; no candidate combinations. Item08 review is conditional on a retained candidate.
