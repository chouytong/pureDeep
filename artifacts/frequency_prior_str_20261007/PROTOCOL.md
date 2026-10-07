# Fixed five-band Frequency-Prior STR — preregistered development protocol

Date: 2026-10-07 (Asia/Shanghai). F0 is original STR-01; F1 and F2 are the only new training conditions. User explicitly confirmed **same random initialization training**, not warm-start fine-tuning. Frozen trained weights are used for output consistency tests only. This document and executable source will be locked and Git-published before formal results.

## Conditions

- F0: reuse audited original75,524-param STR, original45predictions/checkpoints.
- F1: original wrist64→Linear64→17→GELU→Linear17→16→GELU→zero-initLinear16→64→residual add. Additional2,481parameters.
- F2: normalized valid6channel wrist signal→fixed five-band time reconstruction→for each band DepthwiseConv1d6(kernel15,pad7,biasFalse)→PointwiseConv6→8(biasTrue)→GELU→temporal average pooling1; concatenate5×8=40→Linear40→16→GELU→zero-initLinear16→64→residual add. Additional2,474parameters. Difference7parameters(.28% of added capacity), fixed16D final branch embedding. No dropout or fitted normalization is added in either branch.

Five fixed bands: [.5,3),[3,4),[4,6),[6,8),[8,12]Hz. Fixed nominal100Hz; no band/width/hidden search. Band decomposition uses rFFT of `[x,reverse(x)]` at exactly2×true length, disjoint Boolean masks, irFFT and crop back to true length. Full even reflection avoids an endpoint value discontinuity; it is a deterministic boundary assumption, not physiological continuation. Resolution .0512295Hz at976 and .025Hz at2000. Ideal masks are noncausal and can ring; do not claim causal online deployment or clinical phenotype preservation. No bandpower/PSD/entropy or class-specific band weights are computed as model input.

Shared encoder/branch processes activity×wrist separately. No activity/wrist averaging precedes injection. Original length grouping slices padding before either path; existing wrist-valid scatter, Full bilateral258, activity-ID attention, ordered11×16 structured residual and classifier are retained. Activity conditioning occurs downstream through unchanged activity identity/ordered STR, not through new per-activity modules. No HarNet, WSSL, H1, handcrafted input, augmentation, loss, sampler or threshold change.

F1 is a clear near-capacity-matched learned-wrist-feature control, not an exactly computationally matched raw-time branch. A benefit over F1 supports the registered frequency-prior-path design relative to this control; it cannot isolate frequency masking from every consequence of a second signal path or prove that4–6Hz alone causes benefit.

## Initialization and frozen recipe

Construct original STR completely first with the same seed, preserving all original initial parameters. Construct the new branch under a local CPU RNG save/restore so branch construction does not alter the base-model or DataLoader/Dropout RNG sequence. No new stochastic module. F1/F2 start independently from the same original initial STR parameters; original model remains fully trainable and branch is trainable, same optimizer LR for all. Zero-init behavior is tested with actual frozen weights and fresh initialization; source output-equivalence and checkpoint reload are required.

Canonical split SHA b3c52317cb12b73c66046bbd7c50c94a7e707a64a38d3ad4f0fd4bc0c6ba6b3e. 15fixeddevelopment splits(5context×3inner), seeds42/43/44. Restrict record loader to current inner-train∪validation; test subjects are never supplied to fold construction. Train-only activity/wrist/channel mean/std and class-balanced CE, AdamW2e-4/WD1e-4/betas.9,.999, batch8, cosine50floor1e-6, FP32, clip5, max50, strict ordinary validationBA best, patience12. DD probability>.5; tiePD. No refit/outer-final function. Each condition has its own best epoch under the original ordinaryBA rule, no post-hoc alternate checkpoint or EMA.

Complete all45F1+45F2 even if partial scores are unfavorable. No interim performance selection or retuning. Smoke is one engine epoch/batch for software verification, independently archived. Do not resume incomplete training lacking RNG-state checkpoints; preserve failure and diagnose implementation first.

## Execution gates and analysis

1. F0 asset/config/metric/full-validation/state/normalization audit PASS.
2. Synthetic filter shape/actual lengths/edges/finite/autograd PASS; no real performance selection.
3. F1 then F2 implementation; zero-init baseline equivalence on all45 frozen checkpoint sampled batches, fresh base initial state and CPU-RNG preservation; forward/backward, two updates (upstream branch gradients are zero at exact zero-init, must appear after first projection update), reload, padding/wrist/activity mask isolation, no branch mixing.
4. Original engine smoke PASS for both, frozen source/config/asset guards; publish protocol/source/test milestone before full training.
5. Analyze F2−F0, F2−F1, F1−F0, averaging seed metrics first within15splits. Metrics Accuracy/BA/AUROC/Macro-F1/PD/DDRecall. Report wins/ties/losses,10,000paired bootstrap95%CI(seed20261007), paired dz/rank-biserial, Wilcoxon, BH across3primaryBA comparisons and separately all18descriptive comparison×metric tests. Report seed-meanSD, mean within-split seedSD, splitSD, prediction agreement, best/stop epochs, train/validation mainloss and error changes.

**Pre-result decision rule:** RETAIN requires F2 positive meanBA,≥10/15BA improvements, positive paired-bootstrap lower bound and primaryBHq<.05 versus BOTH F0 and F1; AUROC mean delta≥−.005, Macro-F1 nondecrease and neither class-recall drop>.01 versus BOTH; F2 BA/AUROC mean within-split seedSD≤1.25×each reference(max(referenceSD,.001)). These fixed operational margins instantiate the requested majority/stability/no-serious-trade-off criteria; they are design choices, not known-optimal margins. F2 meanBA≤F0, or clear failure of auxiliary trade-off guard, yields REJECT; otherwise failure to establish robust extra benefit yields INCONCLUSIVE. Both stop this configuration with no rescue search. F1 is an interpretive control, not a third optimization candidate to retain post-hoc.

Only if F2 meanBA>F0 AND F1, without recall trade-off guard failure, is descriptive activity/band/wrist residual analysis allowed. All11activities and all5bands are reported, no selection; no additional training. DD subtype description only if DDRecall improves, sample-size-limited and non-tuning. Aggregate explanation is not a causal band ablation.

15splits overlap in subjects/contexts; statistics describe repeated development robustness, not independent external generalization.45seed-runs, windows, subjects reused acrosssplits and PD–DDpairs do not enlarge the primary sample size. Existing multi-stage selection optimism remains. No outer outcome/diagnostic access. No post-hoc frequency dimension/band/loss/recipe/threshold changes. WSSL+frequency is not part of this run, regardless of outcome; a separate next-stage authorization/protocol would be needed.
