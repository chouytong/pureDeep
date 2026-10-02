# Item 06: current WSSL error analysis rules

Registered after items 01–05, before inspecting current WSSL subtype recalls/error groups. Use original frozen WSSL validation predictions only; exclude previous V8 error groups, candidate and outer predictions.

Match explicit subject ID, label and split. Average three seeds within each validation context, then average four available validation-context scores for each independent subject. Consensus DD prediction uses probability >0.5. This descriptive consensus performance is separate from formal seed-first 15-split metrics.

Primary stable-error means wrong in all four context-level seed-mean decisions; stable-correct means correct in all four; others unstable. Strict all-12-seed-run unanimity is sensitivity only and is reported separately. Neither rule alters data selection or training sampling.

DD labels use verified `source_condition` from frozen manifests: Other Movement Disorders, Essential Tremor, Atypical Parkinsonism, Multiple Sclerosis. Check consistency across 11 activities and binary labels. These are source diagnosis categories; Other is a broad heterogeneous category. Do not invent more granular clinical subtypes.

For each category report independent N, consensus recall, mean context recall and stable-error counts. Pairwise consensus-recall differences use Fisher exact tests across six DD category pairs, BH-FDR, plus 10,000 subject bootstraps within categories (95% percentile CI). These describe repeated development evidence; training overlap and small categories limit population inference. C1 eligibility requires source-label checks pass, each category N≥10, and at least one pair with BH q<0.05 and bootstrap difference interval excluding zero. If eligible use all four source categories, one auxiliary weight 0.1, DD training labels only; otherwise skip. No category/weight search.

Quality checks: all 11 bilateral activities present, consistent 100Hz metadata, finite original/processed signals, fixed lengths; describe per-subject maximum wrist timestamp offset and raw-Acc fraction outside ±3g. Offset >0.01s means larger than one sampling interval, not proof of a bad recording. Extreme Acc is descriptive and may represent physiology. Compare primary error groups descriptively; no error subject deletion, alignment/filter tuning or validation-informed sampling.
