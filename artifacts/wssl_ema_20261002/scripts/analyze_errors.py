"""Subject-ID aligned errors; split-seed-first counts and descriptive subject groups."""
import json
from pathlib import Path
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent.parent;B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained')
rows=[];counts=[]
for seed in (42,43,44):
 for oi in range(5):
  for ii in range(3):
   frames=[]
   for name,root in [('baseline',B),('ema',H/'runs/ema')]:
    f=pd.read_csv(root/f'seed{seed}/outer_{oi}/inner_{ii}/predictions/validation.csv',dtype={'subject_id':str})[['subject_id','target','probability_dd']].rename(columns={'probability_dd':name+'_score'})
    frames.append(f)
   f=frames[0].merge(frames[1],on=['subject_id','target'],validate='one_to_one');assert len(f)==len(frames[0])==len(frames[1]);f['seed']=seed;f['context']=oi;f['inner']=ii
   f['baseline_wrong']=(f.baseline_score>.5)!=f.target;f['ema_wrong']=(f.ema_score>.5)!=f.target
   rows.append(f)
   for label in (0,1):
    c=f[f.target==label];counts.append(dict(seed=seed,context=oi,inner=ii,label=label,n=len(c),recovered=int((c.baseline_wrong&~c.ema_wrong).sum()),harmed=int((~c.baseline_wrong&c.ema_wrong).sum()),both_wrong=int((c.baseline_wrong&c.ema_wrong).sum()),agreement=float((c.baseline_wrong==c.ema_wrong).mean())))
raw=pd.concat(rows);raw.to_csv(H/'predictions/error_transitions_private.csv',index=False)
c=pd.DataFrame(counts);c.to_csv(H/'analysis/error_counts_45runs.csv',index=False);fold=c.groupby(['context','inner','label'],as_index=False)[['n','recovered','harmed','both_wrong','agreement']].mean();fold['net_recovery']=fold.recovered-fold.harmed;fold.to_csv(H/'analysis/error_counts_15split_seedfirst.csv',index=False)
# Original primary stable error: seed-mean decision wrong in all four validation contexts.
contexts=raw.groupby(['subject_id','target','context'],as_index=False)[['baseline_score','ema_score']].mean();contexts['baseline_wrong']=(contexts.baseline_score>.5)!=contexts.target;contexts['ema_wrong']=(contexts.ema_score>.5)!=contexts.target
subjects=[]
for (sid,label),g in contexts.groupby(['subject_id','target']):
 assert len(g)==4
 bw=int(g.baseline_wrong.sum());ew=int(g.ema_wrong.sum());group='stable_error' if bw==4 else 'stable_correct' if bw==0 else 'unstable'
 seedg=raw[raw.subject_id==sid];assert len(seedg)==12
 subjects.append(dict(subject_id=sid,label=label,baseline_group=group,baseline_errors=bw,ema_errors=ew,contexts=4,all12_baseline_error=bool(seedg.baseline_wrong.all()),recovered=int((g.baseline_wrong&~g.ema_wrong).sum()),harmed=int((~g.baseline_wrong&g.ema_wrong).sum())))
s=pd.DataFrame(subjects);s.to_csv(H/'predictions/subject_error_private.csv',index=False)
g=s.groupby(['baseline_group','label']).agg(subjects=('subject_id','count'),baseline_errors=('baseline_errors','sum'),ema_errors=('ema_errors','sum'),recovered=('recovered','sum'),harmed=('harmed','sum')).reset_index();g['net_recovery']=g.recovered-g.harmed;g.to_csv(H/'analysis/error_group_summary.csv',index=False)
result=dict(primary_definition='All 4 validation contexts wrong after averaging 3 seed scores within context',sensitivity='All 12 individual seed/context predictions wrong',groups_not_selection_rules=True,subject_inference=False,descriptive_only=True,validation_appearances=1560,unique_subjects=390,outer_access=False)
(H/'analysis/error_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(g.to_string(index=False));print(fold.groupby('label')[['recovered','harmed','net_recovery','agreement']].mean().to_string())
