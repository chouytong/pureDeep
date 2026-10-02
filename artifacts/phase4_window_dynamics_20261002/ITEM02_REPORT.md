# Item 02 — frozen HarNet10 window preservation

Decision: **PASS, preserve window features**. No candidate training or outer evaluation.

Files: scripts/extract_windows.py, scripts/test_window_consistency.py; analysis/window_extraction_audit.json, window_consistency_test.json, window_logit_consistency.csv. Server-only features/window_features.npz and smoke/window_reload.pt are excluded from Git.

| Check | Result |
|---|---|
| Wrist records / windows | 8,580 / 10,920 |
| Single / double window records | 6,240 / 2,340 |
| Window feature shape | 390×11×2×2×1024 |
| Effective mask | 390×11×2×2; short [true,false], back=0 |
| Same resampled input SHA | 7f77341dafe0225910cc736d80898b33fe8f098e8b9649693e1eaa78e353b0fe |
| New reconstructed mean vs frozen cache | Exact equality, max difference 0 |
| Frozen checkpoint logits, new vs old mean | Exact equality for all 45 checkpoints |
| Archived probability reproduction | Max difference 1.1102e-16 (45 real batches, batch 8) |
| Invalid wrist/activity SSL perturbation | Logits unchanged |
| Projection gradient / single-batch update | Finite nonzero / passed |
| Checkpoint reload | Exact logits |
| HarNet10 trainable parameters | 0 |

Preprocessing, weights, window locations and extraction order/batch size 64 remain identical. Subject lookup uses explicit IDs, including a reordered lookup test. All cache means are exactly equal; no second window is synthesized for short records. Global development subjects are the union of inner-training/validation memberships. No outer results or sensor test partition are used for tuning.

A first archived-probability check used batch 2 and omitted original deterministic settings; it differed by approximately 1e-5. The test was corrected to the frozen execution settings (deterministic, batch 8), not by relaxing tolerance; final difference 1.1e-16. New/old means and same-input logits remain exact. Training smoke updates only an in-memory copy and a new smoke checkpoint, never original checkpoints.

Performance stays at baseline: Accuracy .745604, BA .719644, AUROC .759553, Macro-F1 .706589, PD Recall .782344, DD Recall .656945; delta 0. Next: item 03, A1 capacity control.
