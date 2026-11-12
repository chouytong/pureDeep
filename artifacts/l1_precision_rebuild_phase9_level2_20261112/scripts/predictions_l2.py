"""Actual per-run historical probabilities, never replicated overall means."""
from common_l2 import *
from metrics_l2 import pred_metrics,paired

def old_prediction(kind,seed,c,i,query):
 p=P8/'predictions'/f'baseline_seed{seed}_context{c}_inner{i}.npz' if kind=='wssl' else baseline(kind,seed,c,i)/'predictions/validation.csv'
 protected=load(PRIVATE/'manifests/protected_assets.json')['sha256'];assert str(p) in protected and sha(p)==protected[str(p)]
 if kind=='wssl':
  with np.load(p,allow_pickle=False) as a:ids=a['subject_ids'].astype(str);y=a['target'].astype(int);prob=a['probabilities'].copy();z=a['logits'].copy()
 else:
  f=pd.read_csv(p,dtype={'subject_id':str});ids=f.subject_id.to_numpy();y=f.target.to_numpy(int);prob=f[['probability_pd','probability_dd']].to_numpy(float);z=np.empty((0,2))
 assert len(ids)==len(set(ids)) and set(ids)==set(query)
 return p,ids,y,prob,z

def catalog(kinds=('wssl','str'),include_new=True):
 rows=[];out=[];lineage=[]
 for kind in kinds:
  new=load(PRIVATE/'manifests'/f'{kind}_lineage.json')['records'] if include_new else []
  lookup={(r['seed'],r['context'],r['inner']):r for r in new}
  for c,i,sp,ex in splits():
   for seed in [42,43,44]:
    path,ids,y,p,z=old_prediction(kind,seed,c,i,sp['validation_subjects'])
    conditions=[(f'{kind.upper()}-LEGACY-L1',path,ids,y,p,z)]
    if include_new:
     r=lookup[seed,c,i];path=Path(r['prediction']);assert sha(path)==r['prediction_sha256']
     with np.load(path,allow_pickle=False) as v:ni=v['subject_ids'].astype(str);ny=v['target'].astype(int);np_=v['probabilities'].copy();nz=v['logits'].copy()
     assert len(ni)==len(set(ni)) and set(ni)==set(sp['validation_subjects']);idx={s:j for j,s in enumerate(ni)};assert np.array_equal(y,ny[[idx[s] for s in ids]])
     conditions.append((f'{kind.upper()}-CERTIFIED-L1',path,ni,ny,np_,nz))
    for condition,path,ids,y,p,z in conditions:
     metric=dict(condition=condition,seed=seed,context=c,inner=i,**pred_metrics(y,p,z if len(z) else None));rows.append(metric);out.append((condition,seed,c,i,ids,y,p,z));lineage.append(dict(condition=condition,seed=seed,context=c,inner=i,path=str(path),sha256=sha(path),NLL_source='float64 logsumexp of stored logits' if len(z) else 'minus log archived probability; no clipping; no historical raw-logit receipt'))
 return pd.DataFrame(rows),out,lineage

def stage_report(kind):
 f,_,lineage=catalog((kind,));ref=f[f.condition==f'{kind.upper()}-LEGACY-L1'];new=f[f.condition==f'{kind.upper()}-CERTIFIED-L1'];stat=paired(new,ref,f'{kind.upper()}new-old');path=HERE/'reports'/f'{kind.upper()}_RETRAIN_REPORT.md';prior=PRIVATE/'manifests'/f'{kind}_initial_completion_report.md'
 if not prior.exists():prior.write_bytes(path.read_bytes())
 text=f'# {kind.upper()} certified vs legacy L1\n\n45 legacy run predictions paired with 45 new runs. Each split first averages 3 seeds, then 15 splits have equal weight. New models use original initialization, architecture, CE and strict validation BA selection. HarNet is frozen, classifier side trained; no old supervised warm start.\n\n'+table(f.groupby('condition')[METRICS].mean().reset_index())+'\n\n'+table(stat)+'\n\nNLL is unweighted proper log loss, Brier sums both class columns, ECE uses ten equal width top-confidence bins. DD > .5; tie PD. STR legacy NLL uses stored probabilities without clipping; WSSL legacy and new models use stored raw logits. Repeated development splits are not external validation. Full fixed-family and subject-cluster analysis follows both stages.\n'
 path.write_text(text);f.to_csv(HERE/'analysis'/f'{kind}_stage_comparison.csv',index=False);stat.to_csv(HERE/'analysis'/f'{kind}_stage_paired.csv',index=False);write(PRIVATE/'manifests'/f'{kind}_historical_prediction_sources.json',lineage);write(HERE/'analysis'/f'{kind}_comparison_report_archived.json',dict(status='PASS_ARCHIVED_BEFORE_NEXT_STAGE',legacy_runs=45,new_runs=45,report_sha256=sha(path),original_completion_report_sha256=sha(prior),next_stage_not_started=not (HERE/'analysis/str_training_integrity.json').exists() if kind=='wssl' else None))
if __name__=='__main__':stage_report(sys.argv[1])
