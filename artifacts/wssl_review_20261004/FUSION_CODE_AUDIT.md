# WSSL-STR fusion and reproducibility review — 2026-10-04

Status: current server property tests executed and PASS; see section below. Earlier coverage tables describe archived tests, not the expanded scope of this execution. This note does not report a new performance experiment. Historical source, checkpoints, reports and predictions are unchanged.

## Executed current-state verification

`analysis/fusion_properties.json` and `fusion_properties_45checkpoints.csv` record actual server PASS on all45 frozen checkpoints, one real eight-validation-subject batch per checkpoint. Old/new logits and fresh reload outputs are bit-identical; probability error versus archive max1.11e-16; complete-subject input/cache reordering logit error0 (prespecified tolerance1e-5). Explicit SSL beats stale NaN bridge storage; missing/wrong-shaped inputs are rejected. Large finite masked-wrist/activity feature perturbations do not affect output; retained valid SSL features have an observable nonzero effect in every checkpoint. Cache is finite and original source/cache/checkpoint/prediction guards pass.

First checkpoint copy test directly verifies distinct parameter/cache storage, mutation isolation and Python/NumPy/CPU/CUDA RNG preservation. A fresh two-step, training-only scratch check confirms finite/nonzero backbone, projection and structured gradients, zero LayerNorm gradient at exact zero projection then positive gradient after one update; frozen cache has no gradients. No performance or trained checkpoint is retained from scratch. These tests do not cover all validation subjects or prove clinical/generalization optimality. Companion preprocessing and15 normalization refit tests also PASS; no outer dataset/results accessed.

## Authoritative evidence and stale local assets

- `code/wssl_ema_20261002/scripts/independent_wssl.py` is the new state-compatible WSSL interface audited in the completed EMA study.
- The archived independence test covers 45 frozen best checkpoints, each evaluated on one real development batch of eight subjects. It does **not** by itself cover every validation record of every checkpoint.
- The completed EMA runner supplies stronger end-to-end reproducibility evidence: all 45 ordinary runs match the archived per-epoch numeric records after excluding time/memory fields, selected/stopping epochs, all validation IDs/targets/probabilities, and best-weight tensors. The local audit JSON records are archived claims until the server files are rechecked.
- `code/MFAM-pure-deep/foundation_validation/src/models/pure_deep.py` is an older local copy. It lacks `_encode_activities_with_wrists`, `structured_token_residual` and `bilateral_fusion_mode` required by the retained WSSL wrapper. It must not be used to infer the current architecture or reproduce current checkpoints. The current snapshot under `current_source/foundation_validation/` contains the required implementation; server source remains authoritative and the live test guards its SHA against the archived source lock.

## Current fusion calculation visible in the wrapper

1. Build the unchanged STR backbone, with 11 ordered activities and 64-dimensional wrist embeddings.
2. Encode raw local input through the backbone wrist encoder. The initial bilateral activity representation returned here is discarded; the wrist embeddings are retained.
3. Apply learned LayerNorm(1024) then learned Linear(1024,64) to independently frozen HarNet10 features.
4. Multiply the SSL residual by the valid-wrist and valid-activity mask, and add it to the corresponding STR wrist embedding.
5. Recompute unchanged full bilateral fusion, yielding 258-dimensional activity representations.
6. Apply the original activity aggregator and 258-dimensional subject classifier.
7. Add the original structured-token residual logits. A wrist-structured residual path is implemented conditionally in the general foundation code; retained STR-01 configuration does not enable it. The live test additionally requires the instantiated module to be `None`.

The SSL projection is zero initialized. Thus the initial frozen feature input cannot alter STR outputs; learned SSL use develops through projection gradients. This is a fixed residual fusion, not a post-hoc score ensemble. HarNet itself is represented by a fixed feature cache, not a train-mode module in this classifier.

## Foundation snapshot: architecture details confirmed

- The STR input encoder uses a shared six-channel temporal stem with feature dimension 64, learned convolutional temporal blocks (kernel 7, dilations 1/2/4), GroupNorm with eight groups, and temporal attention/mean/std pooling. A separate learned normalization-free moment branch uses kernels 1/15/63 and 32 channels; learned fusion returns a 64-dimensional wrist embedding. The spectral and relative-energy additions are not enabled in the retained configuration.
- `BilateralFusionHead.fusion_features` concatenates left/right embeddings, their valid-wrist mean, masked absolute left-right difference, and two validity indicators: `64 + 64 + 64 + 64 + 2 = 258`. It requires at least one valid wrist for every selected valid activity. This function contains no trainable operation, Dropout or running state; recomputing it after SSL addition introduces no extra random draw or statistic update.
- Activity attention first computes `LayerNorm(activity_features + learned activity position embedding)`, then a 258→64→1 scorer with Tanh and training Dropout 0.1. Invalid activities receive zero final attention. The resulting weighted sum is the 258-dimensional subject embedding.
- The subject classifier is Dropout 0.2 followed by Linear(258,2). The structured branch independently computes Linear(258,16)→GELU per activity, masks invalid tokens, flattens the fixed 11-activity order to 176 dimensions, then Linear(176,2). Its final linear layer is zero initialized. Final logits are exactly the sum of these two branches.
- At eval, Dropout is inactive; concatenating subject embedding and structured ordered representation would describe a 434-dimensional decision input whose two linear readouts sum. This is a description of the existing computation, not a proposed new classifier.
- The original backbone forward and the independent wrapper follow the same aggregation/classifier/residual order. Returned `activity_embeddings` are the enriched activity context representation, not the raw 258-dimensional bilateral features before aggregation. The current interface does not return those raw features explicitly.
- The wrapper operates on true activity lengths before wrist encoding, so padding is excluded from the local temporal features. The supplied frozen SSL feature cache has its own fixed, previously archived preprocessing and window averaging; it is not recomputed from normalized local tensors in the current forward.

The source exposes several historical optional architectures, but their existence in a general builder does not mean they are active. Only the checkpoint-compatible retained configuration is used for this review.

## Confirmed historical copy hazard and its repair

The archived `ResidualSSLSubject` installs a closure that captures both the original backbone bound method and the original wrapper. `copy.deepcopy` of this already wrapped object preserves those Python closure references. A copied wrapper can therefore use original parameters or feature storage even when its registered tensors have been copied. The independence test explicitly checks the captured original wrapper.

`IndependentWSSL` has a normal explicit `forward` and no captured-model method closures. Its `independent_copy` constructs a fresh instance, loads state into independent parameter storage, clones the ephemeral feature tensor and restores Python, NumPy, CPU and CUDA RNG states. The EMA copy is then put in eval mode and made non-trainable. No classifier buffers are present in the archived inspected configuration.

This hazard does **not** automatically invalidate earlier reported experiments. Phase-3E's consistency test deep-copies an unwrapped pure STR backbone, not an already wrapped WSSL model. Phase-4 candidate paths instantiate new wrappers around independently built backbones. A blanket claim that every historical copy path was checked would exceed the archived test scope.

## Existing test coverage and limits

| Property | Existing evidence | Limit |
|---|---|---|
| Old/new forward equality | 45 old checkpoints, one real eight-subject batch each, bit-identical logits | Not all masks, lengths or batch sizes |
| Archived probability equality | Maximum recorded difference 1.11e-16 for those batches | Same selected records only |
| Fresh-instance parameter isolation | Pointer inequality and mutation isolation | One checkpoint/batch |
| Feature-cache isolation | Cloned feature storage; changing copied cache leaves original output unchanged | One checkpoint/batch |
| New-wrapper deepcopy output | Parameter pointers differ and output agrees | Does not separately mutate deepcopy cache |
| Checkpoint reload | Fresh instance reproduces logits exactly | One checkpoint/batch |
| RNG preservation | CPU/CUDA states directly asserted; Python/NumPy restored by implementation | No direct Python/NumPy assertion |
| Finite, nonzero SSL-projection gradient | One real batch and AdamW step | Does not check all trainable groups or zero-init gradient progression |
| Invalid-wrist/activity invariance | Phase-4 legacy wrapper, finite large SSL perturbations | Not rerun on independent wrapper |
| All ordinary training trajectories | Completed EMA run and audit compare all 45 ordinary histories, predictions and best weights | Requires live artifact/lock recheck |
| HarNet frozen | Cache reuse, source/cache SHA checks, features non-trainable | Does not infer full encoder state from classifier state alone |

## Remaining useful checks on the current model

These checks were executed in `scripts/test_fusion_properties.py` for existing frozen checkpoints plus a disposable two-step inner-training instance; actual PASS coverage is stated in the first section. They do not select a new model.

- Explicit SSL input must take precedence over stale instance-local bridge features. The compatibility bridge intentionally remains for the existing loader; direct callers must supply features or attach them for **every** batch.
- Permuting subjects, their local tensors and ID-looked-up SSL features together must permute outputs correspondingly, within an appropriately fixed numeric tolerance. Cache identity must be based on subject ID, not array position.
- Large finite changes to SSL features of invalid wrists/activities must leave logits unchanged; perturbing valid features of a learned nonzero projection must affect the fusion path.
- Confirm original activity/wrist order and configuration against the current server source; checkpoint keys/counts alone do not prove order semantics. The new script checks the cache/config ordered activity lists and labeled records by subject ID; the executed companion preprocessing audit checks full raw metadata/channel/wrist ordering.
- Inspect projection, normalization and backbone gradients in a scratch instance. With exactly zero projection initialization, the projection should receive a gradient while LayerNorm initially receives zero gradient; after a projection update, normalization can receive a gradient.
- Confirm copied instances do not share parameter/cache storage, preserve RNG states, and reload the same checkpoint with the current wrapper/config.

## Boundary conditions, not demonstrated current-data bugs

- The forward interface checks SSL shape but cannot infer whether an equal-shaped feature tensor belongs to the correct subjects. Subject identity remains a loader/API contract. A wrong same-shaped or stale bridge tensor is accepted by design; direct explicit input reduces this risk.
- `projection(norm(ssl)) * mask` does not protect against NaN/Inf in masked SSL features because NaN multiplied by zero remains NaN. The current finite-cache invariant must therefore be checked; finite masked perturbation tests do not prove nonfinite masking safety. No evidence yet shows nonfinite values in the retained cache.
- `independent_copy` recreates the architecture and applies the root training flag to every child; it does not preserve arbitrary mixed submodule training modes or individual `requires_grad` flags. This is adequate for the recorded all-trainable classifier/EMA usage and is not a generic replacement for every model-copy use case.
- The wrapper repeats the fusion portion after wrist encoding. Current source confirms the retained full fusion is deterministic concatenation/mask arithmetic with no state or randomness. This is not evidence that arbitrary future fusion modules would be side-effect free.

## Review conclusion so far

Current source/cache/property checks PASS within their stated scope, with no newly demonstrated retained data-path bug. The already repaired closure-copy defect is concrete. A separately frozen single classifier-LR contrast addresses inherited recipe coverage; it changes no fusion/encoder computation. Failed Phase-3E/Phase-4/EMA variants remain rejected; no gating, attention, SSL encoder, fine-tuning or augmentation search is reopened.
