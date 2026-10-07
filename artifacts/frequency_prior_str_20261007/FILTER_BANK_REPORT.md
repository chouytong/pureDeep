# Fixed filter bank numerical verification

2026-10-07. PASS; synthetic inputs only, no performance selection. Five fixed disjoint bands at nominal100Hz. True lengths976/2000 are transformed after full even reflection to1952/4000 samples, reconstructed and cropped. Frequency-bin counts respectively49/20/39/39/78 and100/40/80/80/161. Shape[N,5,6,L], exact DCT cosine mode mapping, DC/out-of-band rejection, half-open boundaries(last upper inclusive), nonzero finite autograd, NaN/Inf rejection all pass. No learned/fitted filter parameters.

Padding is excluded by the original STR true-length grouping; model-level wrist/activity mask tests follow after F1/F2 implementation. Ideal filtering has noncausal/ringing/boundary-assumption limitations. This test does not prove phenotype preservation or predictive value. See analysis/filter_bank_tests.json.
