# Phase-2 entry: development-only stage summary

This summary precedes Phase-2 experiments. It uses prior formal development records, not FOE-01 outer outcomes. Current task: subject-level PADS PD versus DD classification, 390 subjects (276 PD, 114 DD), fixed 5 outer development contexts × 3 inner folds, three seeds, train-only normalization. The 15 inner validation splits are the primary paired units after seed aggregation.

| Prior formal model | Params | Development Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0 fixed-band MFAM | — | — | 0.5897 | 0.6131 | — | — | — | historical reference |
| V8-GN | 71,026 | 0.7154 | 0.6707 | 0.6800 | 0.6631 | 0.7782 | 0.5631 | matched deep reference |
| ResNet1D | 279,137 | 0.7093 | 0.6470 | 0.6540 | 0.6380 | 0.7968 | 0.4972 | rejected |
| InceptionTime | 484,449 | 0.7078 | 0.6692 | 0.6819 | 0.6579 | 0.7625 | 0.5760 | not retained |
| STR-01 | 75,524 | 0.7307 | 0.6972 | 0.7176 | 0.6861 | 0.7779 | 0.6165 | **best formal pure-deep; Phase-2 baseline** |
| WSR-01 | 75,962 | 0.7234 | 0.6965 | 0.7216 | 0.6824 | 0.7615 | 0.6315 | rejected; no robust joint gain |
| H1 handcrafted logistic | — | 0.7341 | 0.6918 | 0.7515 | 0.6858 | 0.7936 | 0.5900 | statistical reference, not pure deep |

The current STR-01 has a modest development BA/AUROC advantage over V8-GN (+0.0265/+0.0377). Its DD recall varies more by seed (SD 0.0408 versus V8 0.0134). H1 has a higher AUROC than STR by 0.0339 in development. STR does not resolve persistent hard-subject errors. PAG-01 found frozen activity-raw incremental disease information but no stable monotone information-loss chain; it ruled out new information-preservation training. DD subjects are fewer, but imbalance alone has not been shown to be the causal bottleneck. The original baseline already uses training-fold balanced CE; simply identifying a 276:114 dataset ratio cannot establish that more reweighting helps.

Previously rejected/stopped: generic larger backbones, long-range dilation, new aggregation/readout/gating/attention, Gyro FFT frequency branch, prototype alignment, DANN/MMD/invariance, contrastive, decision-head and structured-readout extension, activity search, and Phase-1 information-preservation training. The earlier V8-GN local recipe search rejected LR 1e-4, batch 16, WD 1e-3, temporal dropout 0.2, and constant LR as V8-GN replacements. Phase-2 does not repeat V8-GN studies; any new training study is STR-specific and must be bounded.

Remaining worthwhile Phase-2 development questions: whether the 50-epoch / patience-12 STR run is genuinely undertrained or instead overfits; whether small STR-specific recipe changes improve paired 15-split BA/AUROC and DD recall; whether balanced CE is already adequate versus carefully controlled alternative loss weighting; and whether mild train-only sensor/noise/time perturbation improves held-out validation without changing movement semantics. All findings stay development-only. This phase does not alter the formal STR-01 model or make a new outer claim.
