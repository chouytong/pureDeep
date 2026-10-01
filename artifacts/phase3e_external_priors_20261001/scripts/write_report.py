from pathlib import Path
import pandas as pd,json
B=Path(__file__).resolve().parents[1];M=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall'];s=pd.read_csv(B/'analysis/model_summary.csv');p=pd.read_csv(B/'analysis/model_paired.csv');e=pd.read_csv(B/'analysis/e0_summary.csv');ep=pd.read_csv(B/'analysis/e0_paired.csv');g=json.loads((B/'analysis/model_gates.json').read_text());au=json.loads((B/'FINAL_AUDIT.json').read_text());tech=pd.read_csv(B/'analysis/parameter_counts.csv');ec=pd.read_csv(B/'analysis/e0_complementarity.csv')
def table(f,cols):
 out=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for _,r in f.iterrows():
  out.append('| '+' | '.join(f'{r[c]:.4f}' if isinstance(r[c],float) else str(r[c]) for c in cols)+' |')
 return '\n'.join(out)+'\n'
t='''# Phase-3E: External Wearable SSL Prior Diversification

Status: complete and audited, 2026-10-01 Asia/Shanghai. Development only; PURE-DEEP.

## Scope and frozen protocol
This user-authorized phase tests different external wearable SSL priors after Phase-3D HarNet10 adaptation failed.15 fixed subject-level development splits (5 contexts x3inner) x seeds42/43/44; original STR architecture, bilateral/activity order/structured path, balanced CE, AdamW LR2e-4 WD1e-4 batch8, cosine50/min1e-6, max50 BA early stopping patience12, train-only normalization, DD probability>0.5. PHASE3E_PROTOCOL.md and official asset/source SHA records preceded formal candidate evaluation. No H1, handcrafted-feature branch, teacher/student, outer performance, STR/loss/sampling/augmentation search, threshold/weight tuning. Fold-local manifests limit each run to innertrain/validation. FOE-01 remains sealed and does not independently validate this phase. Old formal outputs and production forward unchanged.

E0 fixes0.5/0.5 mixing of frozen and lastblock probabilities. E1 tests official HarNet5 with fixed first/last5s windows. E2 tests official default50mr BioPM neural acc encoder only, standalone feasibility and unchanged STR64D wrist residual. E3 random/E4 complementarity/E5 dual are conditional under the frozen gate.

## Official implementation and limits
HarNet5 official source commit150550ea5d41800229c95e36f88f5bf0d2e7cf04; checkpoint SHA74ffaefbafb467b7253d121ef71fe734ee03bc802ba33c33a6aa1aef7f91de5e. InputNx3x150 at30Hz. Raw PADS g Acc, trim48,clip+-3g,resample_poly3/10, exactly first500/last500 raw samples per activity/wrist,mean features.17160contexts. Official final512D/4,227,904encoder parameters, versus10s1024D/10,457,408. Initial1024 assumption was corrected after official forward and before formal training. Only projection input changes;64D residual/downstream unchanged. This official-family comparison cannot isolate duration from capacity/window coverage.

BioPM official commit41979e6c36decbd794e8e79550d98796efc47bdc (2026-08-06),50mr SHA d7c37cf5d54c7249ef97a0b162bef631a7e440836e70db3c13dd604536c01312, strict complete acc-encoder loading,1,418,464parameters. Default1023 includes384 pooled neural-token dims and639 unencoded gravity waveform dims. To honor pure-deep, use official384D per-axis mean/std of neural tokens only; exclude raw gravity and optional gravityCNN (not pretrained in release). This acc-encoder-only result cannot falsify complete BioPM HAR performance.

Official downstream preprocessing: linear100to30Hz, sixth-order0.5-12Hz bandpass, spline zero-crossings, official50ms separation/low-amplitude merge,32sample movement patches, fractional position/axis/duration metadata,192token cap. Input[B,192,32],position[B,192],metadata[B,192,5];unmasked64D neural tokens,official384D pooling. Raw g Acc only,trim48;filter full record then fixed10s first/last contexts;short contexts edge padded. No raw gyro/gravity feature input.10920contexts,6240padded,0empty,3-192tokens,8.14%at token cap. No label-based context selection or token-cap adjustment. Frozen extraction has no data-fitted transformation; fitting/normalization remain train-only and cache retrieval uses subject IDs within active innertrain/validation.

Server DNS blocked direct GitHub access; official sources/weights downloaded locally and transferred. Missing h5py3.11 installed offline in this phase vendor/deps only,original environments unchanged. No skill/plugin installed. Official sources: https://github.com/Prithvitarale/biopm and https://github.com/OxWearables/ssl-wearables .
'''
t+='''
## Smoke, logits consistency and integrity
Initial STR+HarNet5 and STR+BioPM logits exactly equal original STR (zero residual),maxdifference0. Loading a frozen WSSL checkpoint into the generalized1024 interface gives maxdifference0 against original Phase3B at identical input/features. HarNet5/BioPM-only/BioPM-residual/random-control one-batch training/checkpoint smoke passed. E3 control clarification preceded complete BioPM residual statistics and broadens only the random-control trigger to any positive primary mean; retention unchanged. Strict encoder weights,finite cached features,complete subject/activity/wrist mapping passed. ANALYSIS_IMPLEMENTATION_NOTE.md records exclusion of incomplete candidates from aggregation/gates; no action used partial-seed statistics. Full-candidate formulas/training/protocol unchanged.

FINAL_AUDIT.json:

```json
'''+json.dumps(au,indent=2)+'''
```

## Complete seed-first development metrics
Three seeds averaged within split,then15splits. Within-seed SD is the mean15split three-seed sample SD. All six metrics,global-seed SD,split SD,loss/early-epoch summaries are in CSV.

'''+table(s,['variant']+M+['ba_within_seed_sd','auroc_within_seed_sd','score_spearman','threshold_disagreement'])+'''
## Paired primary differences
10000 paired15split bootstrap,paired dz,Wilcoxon. Planned primary family10 tests,unexecuted tests conservatively p1;secondary family20. BioPM-only feasibility comparison descriptive,not retention primary.

'''+table(p[p.metric.isin(['ba','auroc'])],['candidate','reference','metric','mean_delta','positive_splits','ci95_low','ci95_high','cohen_dz','q_primary_bh'])+'''
Retention gate ledger:

```json
'''+json.dumps(g,indent=2)+'''
```

## E0 fixed inference references
Single models use seed-first metric averages;3/6model ensembles evaluate seed-averaged probabilities, a different estimand. Both per-seed equal mix and6model ensemble use exactly one0.5/0.5 rule,no weight search.

'''+table(e,['model']+M)+f'''
Frozen/lastblock mean Spearman{ec.spearman.mean():.4f},threshold disagreement{ec.disagreement.mean():.4f};frozen-only/lastblock-only correct counts{ec.frozen_only_correct.mean():.2f}/{ec.lastblock_only_correct.mean():.2f} per validation unit (descriptive).

'''+table(ep[(ep.candidate=='equal_6model_ensemble')&(ep.reference=='frozen_3seed_ensemble')],['metric','delta','wins','ci_low','ci_high','dz','bh_q'])+'''
Six-model equal mix raises AUROC over frozen3seed inference but does not jointly improve BA/MacroF1/DD Recall. Inference reference only,no calibration or single-model retention.

## Parameter counts
Frozen encoders included in total,excluded from classifier trainable count. Conditional random capacity row does not imply execution.

'''+table(tech,['model','frozen_encoder_parameters','trainable_classifier_parameters','total_parameters'])+'''
## Statistical/evidence limits
The15fixed splits reuse/overlap subjects. Split-bootstrap/Wilcoxon/BH describe repeated-development variability,not independent cohort replication. Seeds,subjects,activities,windows,tokens and PD-DDpairs are not statistical units. Sequential-phase selection optimism remains;no Phase3E method has new independent outer validation. HarNet5 differs in official capacity/window coverage as well as duration. BioPM is restricted to pure-neural acc output and fixed integration/classifier recipe. Neither failure establishes global HarNet10 optimality,refutes full BioPM HAR pipeline,or stops every external SSL prior. A single fixed random encoder realization,if used,also limits control generality. OldFOE01 must not be reused for tuning or validation.

## Reproduction ledger
PHASE3E_PROTOCOL.md,OFFICIAL_ASSET_AUDIT.md,manifest.json;official vendor source/checkpoints;formal source SHA ledgers;independent scripts;feature extraction audits/caches;smoke logs/consistency.json;new runs split/seed artifacts;analysis model/e0/e4 CSV/JSON;FINAL_AUDIT.json/final_integrity.csv;static plots. No Phase1-3D formal output overwritten.
'''
(B/'PHASE3E_REPORT.md').write_text(t)
