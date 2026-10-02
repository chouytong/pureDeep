# Item 02 — archived WSSL trajectory diagnosis (PASS)

All 45 archived logs inspected. Best epoch median 10 (range 2–24), actual stop median 22 (range 14–36). Original patience=12. Validation after the selected best: last minus best BA=-0.0608805, AUROC=-0.0144186, main classification loss=+0.4204585. Neighborhood ±1/±2 logs are retained in trajectory_best_neighborhood.csv, including actual available row counts; online train logs do not record train AUROC.

|Actual checkpoint / eval role|BA|AUROC|Macro-F1|Main CE loss|
|---|---:|---:|---:|---:|
|best / train eval|0.913406|0.964805|0.896661|0.239857|
|best / validation eval|0.719644|0.759553|0.706589|0.679387|
|last / train eval|0.999546|0.999988|0.999232|0.013802|
|last / validation eval|0.658764|0.745135|0.663840|1.099845|

90 actual best/last checkpoints re-evaluated under model.eval() and no_grad. Validation BA reproduces its archived epoch. Best-checkpoint train CE 0.239857 differs from its online dropout-active train loss; therefore online train-versus-validation differences must not be interpreted as checkpoint generalization gap. Even with dropout disabled, last checkpoints fit training nearly perfectly while validation BA/AUROC/loss deteriorate. This supports existing BA early stopping, and provides a bounded rationale for the single EMA control; it does not establish EMA efficacy.

Only best and last weights exist. Middle epochs can be described from logs only; no intermediate checkpoint or retrospective full EMA is claimed. No training, new selection or outer access in this item. Diagnostic repeated eval initially hit DataLoader temporary-resource exhaustion; disk was available. The diagnosis was rerun completely with num_workers=0 and pin_memory=False for eval only (training recipe unchanged) and passed all 90 checkpoints. Complete logs summarized in trajectory_45runs.csv; eval_mode_best_last.csv.
