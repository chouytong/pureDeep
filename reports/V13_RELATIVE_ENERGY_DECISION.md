# V13 learnable relative-energy decision

Date: 2026-09-16

V13 adds one small pure-neural branch to frozen V8: eight shared temporal filters
are optimized end to end, constrained to zero mean and unit norm, and summarized
by within-channel relative log energy. No predefined band energy, RMS, dominant
frequency or other handcrafted input is used.

The seed-42 discovery run used the same 15 development folds and downstream
training protocol as V8. Its preregistered replication gate required a BA gain of
at least 0.01, at least 9/15 paired-fold wins, no material AUROC regression and no
numerical instability.

| Metric | V8 seed 42 | V13 seed 42 | Delta | V13 wins |
|---|---:|---:|---:|---:|
| Accuracy | 0.7359 | 0.7218 | -0.0141 | 7/15 (2 ties) |
| Balanced accuracy | 0.6682 | 0.6684 | +0.0002 | 8/15 |
| Macro-F1 | 0.6710 | 0.6633 | -0.0077 | 5/15 |
| AUROC | 0.6649 | 0.6677 | +0.0028 | 6/15 |
| DD recall | 0.5049 | 0.5398 | +0.0349 | 6/15 (1 tie) |
| NLL | 0.6242 | 0.6416 | +0.0174 | worse |
| Brier | 0.4196 | 0.4411 | +0.0215 | worse |

Outer-context BA deltas are -0.0021, +0.0075, +0.0013, -0.0109 and +0.0053.
The BA fold standard deviation rises from 0.0279 to 0.0344.

Decision: reject V13 and stop this direction. Seeds 43/44 are not run, and filter
count, kernel size and initialization frequencies are not tuned on the development
folds. V8 remains frozen for the one-time final nested outer evaluation.
