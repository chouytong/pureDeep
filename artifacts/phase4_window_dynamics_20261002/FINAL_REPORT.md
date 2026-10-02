# WSSL ordered experiment final record — items01–08

**Retained configuration: original frozen WSSL-STR.** Verified/reused 45 baseline stages; four independent 45-run studies (180 newly trained runs), all rejected by registered gates. Item06 updated current WSSL errors and enabled the one fixed C1 test; item08 skipped because no candidate was retained. No candidate stacking, dimensions/weights/threshold/recipe search or outer information use.

| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | ΔBA | ΔAUROC | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 | +0.000000 | +0.000000 | RETAIN existing frozen WSSL |
| a1_mean | 0.744948 | 0.720714 | 0.761608 | 0.706817 | 0.779316 | 0.662111 | +0.001069 | +0.002054 | REJECT |
| a2_delta | 0.741305 | 0.718099 | 0.762503 | 0.703521 | 0.774137 | 0.662062 | -0.001545 | +0.002950 | REJECT |
| b1_raw | 0.753274 | 0.717325 | 0.756623 | 0.709342 | 0.803990 | 0.630659 | -0.002320 | -0.002930 | REJECT |
| c1_dd_aux | 0.742770 | 0.718603 | 0.762054 | 0.704038 | 0.777107 | 0.660099 | -0.001041 | +0.002500 | REJECT |

## Decisions and checks

1. Baseline asset/config/split/seed/normalization/metric audit passed; reuse confirmed.
2. True front/back windows and masks preserved. Mean cache and same-input logits exactly equal for all 45 frozen checkpoints; no fake second window for short records.
3. A1 mean branch: small BA/AUROC mean gain, BA improvement only 8/15, BA CI crosses zero; reject.
4. A2 delta branch: BA and Macro-F1 decrease; no simultaneous BA/AUROC advantage over matched A1; reject. Both use same initialization/norm/masks; only Entrainment/Relaxed/RelaxedTask actually have double windows.
5. Raw local Acc: Accuracy/PD recall increase, BA/AUROC decrease, DD recall decreases .026286 (CI −.049631 to −.003442), above allowed .01 drop; reject. Gyro/SSL/lengths unchanged.
6. Fresh WSSL independent-subject summary: primary stable errors PD14/DD24. Verified source categories Other60/ET28/Atypical15/MS11. Other–ET exploratory consensus recall difference passes six-pair BH q=.029499 and subject-bootstrap CI. Strict seed unanimity is sensitivity only. No errors removed or validation-informed sampling.
7. C1 source-category auxiliary weight .1: BA/F1 decrease; AUROC increase lacks stable support/required count. DD-only loss, no PD auxiliary update, no validation/inference subtype labels; reject. Broad source categories are not granular adjudicated clinical subtype labels.
8. Conditional review skipped: no retained new candidate; stopping rule unchanged.

All implementations passed applicable shape/mask/gradient/checkpoint-reload/single-batch tests and formal smoke before full runs. Classifier-side trainable parameters: baseline 143,172; A1/A2 151,948; raw Acc 143,172; C1 144,208 (auxiliary head unused at inference). Official HarNet10 stays frozen at 10,457,408. Selected epoch/seed/split SD/agreement, paired bootstrap intervals/counts/effect sizes/BH are in each numbered report and aggregate CSV.

Original source/checkpoint/cache/normalization/prediction/split hashes verified unchanged at closure. Public Git contains source/configs/reports/aggregate metrics; datasets, raw arrays, feature caches, weights, per-subject predictions and labels stay on server. Each numbered item is individually committed/pushed/tagged; item08 final closure publishing follows this record.

## Evidence boundary

Three seeds average within 15 fixed overlapping development splits. The 180 executions are not independent statistical samples; split-bootstrap and exploratory BH describe repeated development robustness. Current WSSL errors use one record per independent subject after seed/context aggregation, with model-training overlap acknowledged. Quality measures are descriptive and do not establish bad recordings or justify deletion. Original FOE-01 outer results were not used in this study, and these results provide no new external validation. No new best configuration or automatic follow-on search is supported.
