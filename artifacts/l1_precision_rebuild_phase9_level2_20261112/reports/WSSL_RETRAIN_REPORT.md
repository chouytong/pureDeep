# WSSL certified vs legacy L1

45 legacy run predictions paired with 45 new runs. Each split first averages 3 seeds, then 15 splits have equal weight. New models use original initialization, architecture, CE and strict validation BA selection. HarNet is frozen, classifier side trained; no old supervised warm start.

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WSSL-CERTIFIED-L1 | 0.741534 | 0.7189 | 0.759285 | 0.703561 | 0.773894 | 0.663905 | 0.623037 | 0.390936 | 0.113658 |
| WSSL-LEGACY-L1 | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 | 0.61672 | 0.386441 | 0.110333 |


| condition | metric | status | units | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | split_SD | paired_dz | rank_biserial | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WSSLnew-old | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.745604 | 0.741534 | -0.00407004 | 0 | 7 | 3 | 5 | -0.0169096 | 0.00735846 | 0.0248775 | -0.163603 | -0.0512821 | 0.875329 |
| WSSLnew-old | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.719644 | 0.7189 | -0.000744774 | 0 | 7 | 1 | 7 | -0.00677689 | 0.00481179 | 0.0118807 | -0.0626877 | -0.0285714 | 0.924978 |
| WSSLnew-old | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.759553 | 0.759285 | -0.000268235 | 0.00319675 | 9 | 0 | 6 | -0.00816897 | 0.0063136 | 0.0146731 | -0.0182807 | 0.116667 | 0.690886 |
| WSSLnew-old | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.706589 | 0.703561 | -0.00302843 | -0.00131859 | 7 | 0 | 8 | -0.0128209 | 0.00542951 | 0.018728 | -0.161706 | -0.0666667 | 0.846924 |
| WSSLnew-old | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.782344 | 0.773894 | -0.00844955 | 0 | 7 | 2 | 6 | -0.0366613 | 0.0165578 | 0.0550542 | -0.153477 | -0.043956 | 0.888773 |
| WSSLnew-old | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.656945 | 0.663905 | 0.00696 | 0 | 6 | 2 | 7 | -0.0191876 | 0.0354362 | 0.05681 | 0.122513 | 0.0879121 | 0.779632 |
| WSSLnew-old | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.61672 | 0.623037 | 0.0063168 | 0.00951231 | 9 | 0 | 6 | -0.0329493 | 0.0452018 | 0.0792154 | 0.0797421 | 0.133333 | 0.678772 |
| WSSLnew-old | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.386441 | 0.390936 | 0.00449457 | -0.000883312 | 7 | 0 | 8 | -0.00914233 | 0.0190348 | 0.0287871 | 0.156132 | 0.116667 | 0.719727 |
| WSSLnew-old | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.110333 | 0.113658 | 0.00332514 | -0.000662514 | 7 | 0 | 8 | -0.0133161 | 0.0197155 | 0.0334974 | 0.0992658 | 0.1 | 0.761536 |


NLL is unweighted proper log loss, Brier sums both class columns, ECE uses ten equal width top-confidence bins. DD > .5; tie PD. STR legacy NLL uses stored probabilities without clipping; WSSL legacy and new models use stored raw logits. Repeated development splits are not external validation. Full fixed-family and subject-cluster analysis follows both stages.
