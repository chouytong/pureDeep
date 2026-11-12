# 四格模型与SSL增量

同一Certified L1下WSSL−STR：ΔBA=0.032315，paired95%CI[0.018431,0.045738]，13/15 split提高，BA的BH4 q=0.00231934；ΔAUROC=0.047117。共享患者cluster的BA95%CI[0.005118,0.059187]。结果支持这一重复development设置下的SSL增量。这不是外部或临床有效性证据。

| condition | accuracy | ba | auroc | macro_f1 | pd_recall | dd_recall | nll | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STR-LEGACY-L1 | 0.730718 | 0.697189 | 0.717631 | 0.686068 | 0.777864 | 0.616515 | 0.649321 | 0.415655 | 0.125107 |
| WSSL-LEGACY-L1 | 0.745604 | 0.719644 | 0.759553 | 0.706589 | 0.782344 | 0.656945 | 0.61672 | 0.386441 | 0.110333 |
| STR-CERTIFIED-L1 | 0.717855 | 0.686585 | 0.712168 | 0.673253 | 0.761759 | 0.61141 | 0.685323 | 0.431873 | 0.134591 |
| WSSL-CERTIFIED-L1 | 0.741534 | 0.7189 | 0.759285 | 0.703561 | 0.773894 | 0.663905 | 0.623037 | 0.390936 | 0.113658 |


DD为正类，pDD>.5，恰好tie判PD。AUROC是两概率列的macro AUROC；NLL为未加权proper log loss，新模型/WSSL历史用原logits float64 logsumexp，STR历史只有归档概率所以用−log(p_true)，不clip，有限精度差异如实保留。Brier为两类平方误差之和，ECE是10等宽top-confidence分箱。balanced CE仅训练使用，绝不把加权CE冒充NLL。沿用历史paired函数的wins/ties/losses表示差值正/零/负，损失指标越低越好，最终CSV另提供better_direction、improved_splits、worsened_splits以免误读。

| condition | metric | status | units | reference_mean | candidate_mean | mean_delta | median_delta | wins | ties | losses | ci_low | ci_high | split_SD | paired_dz | rank_biserial | wilcoxon_p | family | BH_q | better_direction | improved_splits | worsened_splits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_WSSL_new-old | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.745604 | 0.741534 | -0.00407004 | 0 | 7 | 3 | 5 | -0.0169096 | 0.00735846 | 0.0248775 | -0.163603 | -0.0641026 | 0.84446 | C1_secondary_7 | 0.861304 | higher | 7 | 5 |
| C1_WSSL_new-old | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.719644 | 0.7189 | -0.000744774 | 0 | 7 | 1 | 7 | -0.00677689 | 0.00481179 | 0.0118807 | -0.0626877 | -0.0285714 | 0.924978 | C1_primary_2 | 0.924978 | higher | 7 | 7 |
| C1_WSSL_new-old | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.759553 | 0.759285 | -0.000268235 | 0.00319675 | 9 | 0 | 6 | -0.00816897 | 0.0063136 | 0.0146731 | -0.0182807 | 0.116667 | 0.719727 | C1_primary_2 | 0.924978 | higher | 9 | 6 |
| C1_WSSL_new-old | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.706589 | 0.703561 | -0.00302843 | -0.00131859 | 7 | 0 | 8 | -0.0128209 | 0.00542951 | 0.018728 | -0.161706 | -0.0666667 | 0.846924 | C1_secondary_7 | 0.861304 | higher | 7 | 8 |
| C1_WSSL_new-old | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.782344 | 0.773894 | -0.00844955 | 0 | 7 | 2 | 6 | -0.0366613 | 0.0165578 | 0.0550542 | -0.153477 | -0.0549451 | 0.861304 | C1_secondary_7 | 0.861304 | higher | 7 | 6 |
| C1_WSSL_new-old | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.656945 | 0.663905 | 0.00696 | 0 | 6 | 2 | 7 | -0.0191876 | 0.0354362 | 0.05681 | 0.122513 | 0.0879121 | 0.801637 | C1_secondary_7 | 0.861304 | higher | 6 | 7 |
| C1_WSSL_new-old | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.61672 | 0.623037 | 0.0063168 | 0.00951231 | 9 | 0 | 6 | -0.0329493 | 0.0452018 | 0.0792154 | 0.0797421 | 0.133333 | 0.678772 | C1_secondary_7 | 0.861304 | lower | 6 | 9 |
| C1_WSSL_new-old | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.386441 | 0.390936 | 0.00449457 | -0.000883312 | 7 | 0 | 8 | -0.00914233 | 0.0190348 | 0.0287871 | 0.156132 | 0.116667 | 0.719727 | C1_secondary_7 | 0.861304 | lower | 8 | 7 |
| C1_WSSL_new-old | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.110333 | 0.113658 | 0.00332514 | -0.000662514 | 7 | 0 | 8 | -0.0133161 | 0.0197155 | 0.0334974 | 0.0992658 | 0.1 | 0.761536 | C1_secondary_7 | 0.861304 | lower | 8 | 7 |
| C2_STR_new-old | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.730718 | 0.717855 | -0.0128629 | -0.00634921 | 3 | 2 | 10 | -0.0233859 | -0.00383424 | 0.020086 | -0.640393 | -0.67033 | 0.0302866 | C2_secondary_14 | 0.0706687 | higher | 3 | 10 |
| C2_STR_new-old | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.697189 | 0.686585 | -0.0106048 | -0.0103501 | 3 | 0 | 12 | -0.0183759 | -0.00258507 | 0.016159 | -0.656276 | -0.633333 | 0.0301514 | C2_primary_4 | 0.0402018 | higher | 3 | 12 |
| C2_STR_new-old | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.717631 | 0.712168 | -0.00546254 | -0.0106545 | 5 | 0 | 10 | -0.0133413 | 0.00292126 | 0.0165912 | -0.329244 | -0.333333 | 0.276855 | C2_primary_4 | 0.276855 | higher | 5 | 10 |
| C2_STR_new-old | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.686068 | 0.673253 | -0.0128145 | -0.0157964 | 3 | 0 | 12 | -0.0197083 | -0.00549029 | 0.0146329 | -0.875732 | -0.783333 | 0.00537109 | C2_secondary_14 | 0.0375977 | higher | 3 | 12 |
| C2_STR_new-old | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.777864 | 0.761759 | -0.0161051 | 0.018018 | 8 | 0 | 7 | -0.0443254 | 0.00959881 | 0.0553461 | -0.29099 | -0.191667 | 0.513486 | C2_secondary_14 | 0.552985 | higher | 8 | 7 |
| C2_STR_new-old | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.616515 | 0.61141 | -0.00510444 | -0.0111111 | 7 | 0 | 8 | -0.0419989 | 0.0327689 | 0.0766866 | -0.0665623 | -0.075 | 0.798193 | C2_secondary_14 | 0.798193 | higher | 7 | 8 |
| C2_STR_new-old | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.649321 | 0.685323 | 0.0360024 | 0.0432339 | 10 | 0 | 5 | 0.00112268 | 0.0700028 | 0.0691607 | 0.520561 | 0.55 | 0.0637207 | C2_secondary_14 | 0.111511 | lower | 5 | 10 |
| C2_STR_new-old | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.415655 | 0.431873 | 0.0162181 | 0.0119764 | 9 | 0 | 6 | 0.00172888 | 0.033149 | 0.0318898 | 0.508566 | 0.483333 | 0.106995 | C2_secondary_14 | 0.149792 | lower | 6 | 9 |
| C2_STR_new-old | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.125107 | 0.134591 | 0.0094843 | 0.0136581 | 10 | 0 | 5 | -0.00509823 | 0.0263526 | 0.0319283 | 0.29705 | 0.316667 | 0.302795 | C2_secondary_14 | 0.353261 | lower | 5 | 10 |
| C2_WSSLnew-STRnew | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.717855 | 0.741534 | 0.0236782 | 0.0380952 | 12 | 0 | 3 | -0.00473568 | 0.0448954 | 0.0516474 | 0.45846 | 0.65 | 0.0255737 | C2_secondary_14 | 0.0706687 | higher | 12 | 3 |
| C2_WSSLnew-STRnew | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.686585 | 0.7189 | 0.032315 | 0.0403952 | 13 | 0 | 2 | 0.0184308 | 0.045738 | 0.0280451 | 1.15225 | 0.883333 | 0.00115967 | C2_primary_4 | 0.00231934 | higher | 13 | 2 |
| C2_WSSLnew-STRnew | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.712168 | 0.759285 | 0.0471167 | 0.0499855 | 15 | 0 | 0 | 0.0322943 | 0.0622572 | 0.0304713 | 1.54626 | 1 | 6.10352e-05 | C2_primary_4 | 0.000244141 | higher | 15 | 0 |
| C2_WSSLnew-STRnew | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.673253 | 0.703561 | 0.0303076 | 0.0405496 | 12 | 0 | 3 | 0.00751422 | 0.0480524 | 0.0418784 | 0.723704 | 0.7 | 0.0150757 | C2_secondary_14 | 0.0527649 | higher | 12 | 3 |
| C2_WSSLnew-STRnew | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.761759 | 0.773894 | 0.0121354 | 0.027027 | 10 | 0 | 5 | -0.0421865 | 0.0551773 | 0.102069 | 0.118894 | 0.383333 | 0.191178 | C2_secondary_14 | 0.243317 | higher | 10 | 5 |
| C2_WSSLnew-STRnew | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.61141 | 0.663905 | 0.0524945 | 0.0555556 | 13 | 0 | 2 | 0.00563667 | 0.0972209 | 0.0947024 | 0.554311 | 0.583333 | 0.0467378 | C2_secondary_14 | 0.0934756 | higher | 13 | 2 |
| C2_WSSLnew-STRnew | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.685323 | 0.623037 | -0.0622861 | -0.0849851 | 3 | 0 | 12 | -0.104796 | -0.018569 | 0.0885779 | -0.703179 | -0.7 | 0.0150757 | C2_secondary_14 | 0.0527649 | lower | 12 | 3 |
| C2_WSSLnew-STRnew | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.431873 | 0.390936 | -0.0409369 | -0.0522883 | 3 | 0 | 12 | -0.0611148 | -0.019122 | 0.0433701 | -0.943897 | -0.816667 | 0.00335693 | C2_secondary_14 | 0.0375977 | lower | 12 | 3 |
| C2_WSSLnew-STRnew | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0.134591 | 0.113658 | -0.0209337 | -0.0158261 | 6 | 0 | 9 | -0.0397952 | -0.00289769 | 0.0379654 | -0.55139 | -0.533333 | 0.072998 | C2_secondary_14 | 0.113553 | lower | 9 | 6 |
| DID_exploratory | accuracy | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.00879287 | 0.00879287 | 0.00647249 | 10 | 0 | 5 | -0.00573159 | 0.0230117 | 0.0298521 | 0.294548 | 0.383333 | 0.207764 | DID_secondary_7 | 0.645996 | higher | 10 | 5 |
| DID_exploratory | ba | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.00986002 | 0.00986002 | 0.00943683 | 9 | 0 | 6 | -0.00154628 | 0.021707 | 0.0238791 | 0.412913 | 0.4 | 0.187622 | DID_primary_2 | 0.25238 | higher | 9 | 6 |
| DID_exploratory | auroc | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.00519431 | 0.00519431 | 0.00857309 | 9 | 0 | 6 | -0.002483 | 0.0129675 | 0.015692 | 0.331017 | 0.35 | 0.25238 | DID_primary_2 | 0.25238 | higher | 9 | 6 |
| DID_exploratory | macro_f1 | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.00978604 | 0.00978604 | 0.0189165 | 11 | 0 | 4 | -0.00381089 | 0.0222407 | 0.0268592 | 0.364346 | 0.416667 | 0.168823 | DID_secondary_7 | 0.645996 | higher | 11 | 4 |
| DID_exploratory | pd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.0076556 | 0.0076556 | -0.00913242 | 6 | 0 | 9 | -0.0246881 | 0.0418367 | 0.0686415 | 0.11153 | 0.0333333 | 0.909543 | DID_secondary_7 | 0.909543 | higher | 6 | 9 |
| DID_exploratory | dd_recall | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | 0.0120644 | 0.0120644 | 0.0322581 | 8 | 1 | 6 | -0.0322844 | 0.0548368 | 0.0891588 | 0.135314 | 0.219048 | 0.470228 | DID_secondary_7 | 0.65832 | higher | 8 | 6 |
| DID_exploratory | nll | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | -0.0296856 | -0.0296856 | -0.0581217 | 5 | 0 | 10 | -0.0704808 | 0.0170641 | 0.0895729 | -0.331413 | -0.333333 | 0.276855 | DID_secondary_7 | 0.645996 | lower | 10 | 5 |
| DID_exploratory | brier | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | -0.0117235 | -0.0117235 | -0.0067428 | 6 | 0 | 9 | -0.0317541 | 0.00769079 | 0.0406606 | -0.288326 | -0.266667 | 0.389404 | DID_secondary_7 | 0.65832 | lower | 9 | 6 |
| DID_exploratory | ece | REPEATED_DEVELOPMENT_DESCRIPTIVE | 15 | 0 | -0.00615916 | -0.00615916 | 0.000337814 | 8 | 0 | 7 | -0.0261683 | 0.013097 | 0.0399467 | -0.154185 | -0.133333 | 0.678772 | DID_secondary_7 | 0.791901 | lower | 7 | 8 |


| comparison | metric | mean_delta | ci_low | ci_high | draws | finite_draws | PD_unique_subjects | DD_unique_subjects | estimand | status | external_validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_WSSL_new-old | accuracy | -0.00407004 | -0.0131663 | 0.0050641 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | ba | -0.000744774 | -0.0108519 | 0.0093952 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | auroc | -0.000268235 | -0.00655809 | 0.00575698 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | macro_f1 | -0.00302843 | -0.0131609 | 0.00697698 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | pd_recall | -0.00844955 | -0.0187559 | 0.00168106 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | dd_recall | 0.00696 | -0.010712 | 0.024326 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | nll | 0.0063168 | -0.00890719 | 0.021872 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | brier | 0.00449457 | -0.0032054 | 0.0122713 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C1_WSSL_new-old | ece | 0.00332514 | -0.00793102 | 0.0122509 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | accuracy | -0.0128629 | -0.0246194 | -0.00132753 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | ba | -0.0106048 | -0.0224695 | 0.00123531 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | auroc | -0.00546254 | -0.0152208 | 0.00459133 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | macro_f1 | -0.0128145 | -0.0250799 | -0.00054614 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | pd_recall | -0.0161051 | -0.0294666 | -0.00238182 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | dd_recall | -0.00510444 | -0.024298 | 0.014265 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | nll | 0.0360024 | 0.0147421 | 0.0567318 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | brier | 0.0162181 | 0.00470145 | 0.0274145 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_STR_new-old | ece | 0.0094843 | -0.00261804 | 0.0234383 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | accuracy | 0.0236782 | 0.000102927 | 0.046826 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | ba | 0.032315 | 0.00511845 | 0.059187 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | auroc | 0.0471167 | 0.0119487 | 0.0823213 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | macro_f1 | 0.0303076 | 0.00399706 | 0.0565508 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | pd_recall | 0.0121354 | -0.0150044 | 0.038504 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | dd_recall | 0.0524945 | 0.0044107 | 0.0993265 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | nll | -0.0622861 | -0.111632 | -0.0129939 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | brier | -0.0409369 | -0.0695162 | -0.0123573 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| C2_WSSLnew-STRnew | ece | -0.0209337 | -0.0362888 | -0.00281926 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | accuracy | 0.00879287 | -0.00546767 | 0.0229879 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | ba | 0.00986002 | -0.00559497 | 0.025681 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | auroc | 0.00519431 | -0.00566246 | 0.0158587 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | macro_f1 | 0.00978604 | -0.00577667 | 0.025369 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | pd_recall | 0.0076556 | -0.00858719 | 0.0240177 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | dd_recall | 0.0120644 | -0.0146172 | 0.0389747 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | nll | -0.0296856 | -0.0530608 | -0.00599874 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | brier | -0.0117235 | -0.0242649 | 0.000931828 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |
| DID_exploratory | ece | -0.00615916 | -0.0248733 | 0.00827081 | 10000 | 10000 | 276 | 114 | equal15split mean of within-split3seed metric contrast | FIXED_FITS_CONDITIONAL_SENSITIVITY | False |


WSSLnew−old回答修复对WSSL的变化；STRnew−old回答本地STR变化；WSSLnew−STRnew回答相同Certified L1下冻结外部表示的增量。差中差单列探索族，不能隔离成泄漏或临床因果效应。BH完整族C1 2/7、C2 4/14、DID2/7，缺项自身NA、校正时p1占位，无事后缩族。

重复的15个development split共享390人，每split先平均3个seed的单模型指标，再对15split等权；不是概率ensemble，也不是390人的独立外部验证。历史outer接触与多轮development的选择偏差保留，本次PASS不能消除。subject-cluster条件于固定已训练模型，所有同一人的重复appearance/seed共用同一个权重。
