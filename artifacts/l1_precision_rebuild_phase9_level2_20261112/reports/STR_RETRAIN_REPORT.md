# STR certified vs legacy L1

45 legacy run predictions paired with 45 new runs. Each split first averages 3 seeds, then 15 splits have equal weight. New models use original initialization, architecture, CE and strict validation BA selection. HarNet is frozen, classifier side trained; no old supervised warm start.

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STR-CERTIFIED-L1 | 0.717855 | 0.686585 | 0.712168 | 0.673253 | 0.761759 | 0.61141 | 0.685323 | 0.431873 | 0.134591 |
| STR-LEGACY-L1 | 0.730718 | 0.697189 | 0.717631 | 0.686068 | 0.777864 | 0.616515 | 0.649321 | 0.415655 | 0.125107 |


| condition | metric | status | units | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | split_SD | paired_dz | rank_biserial | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STRnew-old | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.730718 | 0.717855 | -0.0128629 | -0.00634921 | 3 | 2 | 10 | -0.0233859 | -0.00383424 | 0.020086 | -0.640393 | -0.681319 | 0.0279306 |
| STRnew-old | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.697189 | 0.686585 | -0.0106048 | -0.0103501 | 3 | 0 | 12 | -0.0183759 | -0.00258507 | 0.016159 | -0.656276 | -0.633333 | 0.0301514 |
| STRnew-old | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.717631 | 0.712168 | -0.00546254 | -0.0106545 | 5 | 0 | 10 | -0.0133413 | 0.00292126 | 0.0165912 | -0.329244 | -0.333333 | 0.276855 |
| STRnew-old | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.686068 | 0.673253 | -0.0128145 | -0.0157964 | 3 | 0 | 12 | -0.0197083 | -0.00549029 | 0.0146329 | -0.875732 | -0.783333 | 0.00537109 |
| STRnew-old | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.777864 | 0.761759 | -0.0161051 | 0.018018 | 8 | 0 | 7 | -0.0443254 | 0.00959881 | 0.0553461 | -0.29099 | -0.191667 | 0.513401 |
| STRnew-old | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.616515 | 0.61141 | -0.00510444 | -0.0111111 | 7 | 0 | 8 | -0.0419989 | 0.0327689 | 0.0766866 | -0.0665623 | -0.1 | 0.733168 |
| STRnew-old | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.649321 | 0.685323 | 0.0360024 | 0.0432339 | 10 | 0 | 5 | 0.00112268 | 0.0700028 | 0.0691607 | 0.520561 | 0.55 | 0.0637207 |
| STRnew-old | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.415655 | 0.431873 | 0.0162181 | 0.0119764 | 9 | 0 | 6 | 0.00172888 | 0.033149 | 0.0318898 | 0.508566 | 0.483333 | 0.106995 |
| STRnew-old | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.125107 | 0.134591 | 0.0094843 | 0.0136581 | 10 | 0 | 5 | -0.00509823 | 0.0263526 | 0.0319283 | 0.29705 | 0.316667 | 0.302795 |


NLL is unweighted proper log loss, Brier sums both class columns, ECE uses ten equal width top-confidence bins. DD > .5; tie PD. STR legacy NLL uses stored probabilities without clipping; WSSL legacy and new models use stored raw logits. Repeated development splits are not external validation. Full fixed-family and subject-cluster analysis follows both stages.
