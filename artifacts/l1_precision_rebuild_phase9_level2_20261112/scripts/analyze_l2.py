"""Fixed four-cell comparison, shared patient-cluster sensitivity, device descriptions."""
from common_l2 import *
from metrics_l2 import pred_metrics,paired,bh_planned,gate
from predictions_l2 import catalog,stage_report
from weighted_l2 import weighted
from threadpoolctl import threadpool_limits
from scipy.special import logsumexp
import argparse
CONDITIONS=['STR-LEGACY-L1','WSSL-LEGACY-L1','STR-CERTIFIED-L1','WSSL-CERTIFIED-L1']
COMPARISONS={'C1_WSSL_new-old':{'WSSL-CERTIFIED-L1':1,'WSSL-LEGACY-L1':-1},'C2_STR_new-old':{'STR-CERTIFIED-L1':1,'STR-LEGACY-L1':-1},'C2_WSSLnew-STRnew':{'WSSL-CERTIFIED-L1':1,'STR-CERTIFIED-L1':-1},'DID_exploratory':{'WSSL-CERTIFIED-L1':1,'STR-CERTIFIED-L1':-1,'WSSL-LEGACY-L1':-1,'STR-LEGACY-L1':1}}

def statistics(frame,cat):
 stats=[]
 for name,terms in COMPARISONS.items():
  candidate=None;zero=None
  for cond,coef in terms.items():
   f=frame[frame.condition==cond].sort_values(['context','inner','seed']).reset_index(drop=True);assert len(f)==45
   if candidate is None:candidate=f.copy();candidate[METRICS]=0;zero=candidate.copy()
   candidate[METRICS]+=coef*f[METRICS]
  # Difference versus zero retains the exact paired contrast including DID.
  stats.append(paired(candidate,zero,name))
 stat=pd.concat(stats,ignore_index=True);stat['family']='';stat['BH_q']=np.nan
 for prefix in ['C1','C2','DID']:
  for main,n in [(True,2 if prefix!='C2' else 4),(False,7 if prefix!='C2' else 14)]:
   mask=stat.condition.str.startswith(prefix)&(stat.metric.isin(['ba','auroc'])==main);assert int(mask.sum())==n;stat.loc[mask,'family']=prefix+('_primary_' if main else '_secondary_')+str(n);stat.loc[mask,'BH_q']=bh_planned(stat.loc[mask,'wilcoxon_p'])
 # Add interpretable actual group means for the three ordinary contrasts.
 for name,terms in COMPARISONS.items():
  if len(terms)!=2:continue
  positive=next(k for k,v in terms.items() if v==1);negative=next(k for k,v in terms.items() if v==-1)
  for m in METRICS:
   ix=(stat.condition==name)&(stat.metric==m);stat.loc[ix,'candidate_mean']=frame.loc[frame.condition==positive,m].mean();stat.loc[ix,'reference_mean']=frame.loc[frame.condition==negative,m].mean()
 lower=stat.metric.isin(['nll','brier','ece']);stat['better_direction']=np.where(lower,'lower','higher');stat['improved_splits']=np.where(lower,stat.losses,stat.wins);stat['worsened_splits']=np.where(lower,stat.wins,stat.losses)
 stat.to_csv(HERE/'analysis/paired_statistics.csv',index=False);stability=[];state=[]
 for cond,f in frame.groupby('condition'):
  unit=f.groupby(['context','inner'])[METRICS].mean();state.append(dict(condition=cond,status='LEGACY_REFERENCE' if 'LEGACY' in cond else 'COMPLETE_45_RUNS',runs=len(f),splits=len(unit),**unit.mean().to_dict()))
  for m in METRICS:stability.append(dict(condition=cond,metric=m,mean_within_split_seed_SD=f.groupby(['context','inner'])[m].std(ddof=1).mean(),SD_across_three_seed_means=f.groupby('seed')[m].mean().std(ddof=1),between_split_SD=unit[m].std(ddof=1)))
 pd.DataFrame(stability).to_csv(HERE/'analysis/seed_stability.csv',index=False);pd.DataFrame(state).set_index('condition').loc[CONDITIONS].reset_index().to_csv(HERE/'analysis/experiment_state.csv',index=False)
 base=frame[frame.condition=='WSSL-LEGACY-L1'];new=frame[frame.condition=='WSSL-CERTIFIED-L1'];g=gate(new,stat[stat.condition=='C1_WSSL_new-old'],base,'WSSL-CERTIFIED-L1');g['status']='PASS' if all(g['checks'].values()) else 'NOT_ESTABLISHED';write(HERE/'analysis/performance_decision.json',g)
 label={}
 for condition,seed,c,i,ids,y,p,z in cat:
  assert np.allclose(weighted(y,p,z,np.ones((1,len(y))))[0],list(pred_metrics(y,p,z if len(z) else None).values()),atol=2e-12,rtol=0,equal_nan=True)
  for sid,t in zip(ids,y):
   if sid in label:assert label[sid]==t
   label[sid]=int(t)
 ids=sorted(label);index={s:j for j,s in enumerate(ids)};labels=np.array([label[s] for s in ids]);assert np.bincount(labels).tolist()==[276,114];rng=np.random.default_rng(SEED);weights=np.zeros((10000,len(ids)),np.int16)
 for k in [0,1]:
  ix=np.flatnonzero(labels==k);weights[:,ix]=rng.multinomial(len(ix),np.repeat(1/len(ix),len(ix)),10000)
 sums={};points={};counts={}
 with threadpool_limits(limits=4):
  for no,(cond,seed,c,i,local,y,p,z) in enumerate(cat):
   v=weighted(y,p,z,weights[:,[index[s] for s in local]]);key=cond,c,i
   if key not in sums:sums[key]=np.zeros_like(v);points[key]=np.zeros(9);counts[key]=0
   sums[key]+=v;points[key]+=weighted(y,p,z,np.ones((1,len(y))))[0];counts[key]+=1
   if (no+1)%45==0:print(json.dumps(dict(stage='subject_cluster',prediction_files=no+1)),flush=True)
 cluster=[];maxerr=0
 for name,terms in COMPARISONS.items():
  values=[];point=[]
  for c,i,sp,ex in splits():
   values.append(sum(coef*sums[cond,c,i]/counts[cond,c,i] for cond,coef in terms.items()));point.append(sum(coef*points[cond,c,i]/counts[cond,c,i] for cond,coef in terms.items()))
  v=np.mean(values,axis=0);estimate=np.mean(point,axis=0)
  for j,m in enumerate(METRICS):
   expected=stat[(stat.condition==name)&(stat.metric==m)].iloc[0].mean_delta;err=abs(estimate[j]-expected);maxerr=max(maxerr,err);assert err<2e-12;valid=np.isfinite(v[:,j]);ok=valid.sum()>=9500;ci=np.quantile(v[valid,j],[.025,.975]) if ok else [np.nan,np.nan]
   cluster.append(dict(comparison=name,metric=m,mean_delta=estimate[j],ci_low=ci[0],ci_high=ci[1],draws=10000,finite_draws=int(valid.sum()),PD_unique_subjects=276,DD_unique_subjects=114,estimand='equal15split mean of within-split3seed metric contrast',status='FIXED_FITS_CONDITIONAL_SENSITIVITY' if ok else 'NA_INSUFFICIENT_FINITE_DRAWS',external_validation=False))
 pd.DataFrame(cluster).to_csv(HERE/'analysis/subject_cluster_summary.csv',index=False);write(HERE/'tests/statistics_checks.json',dict(status='PASS',actual_prediction_files=180,weighted_ones_matches_canonical=True,cluster_point_max_error=maxerr,shared_subject_weight_all_appearances_seeds=True,split_seed_first=True,draws=10000,C1_families=[2,7],C2_families=[4,14],DID_families=[2,7],no_seed_ensemble=True))
 return g

def devices(cat):
 meta=pd.DataFrame(records()).drop_duplicates('subject_id').set_index('subject_id');assert len(meta)==390
 appearances=[];private_split=[]
 for cond,seed,c,i,ids,y,p,z in cat:
  decision=(p[:,1]>.5).astype(int)
  for s,t,d in zip(ids,y,decision):appearances.append(dict(subject_id=s,condition=cond,label='PD' if t==0 else 'DD',device=meta.loc[s,'device'],correct=float(t==d)))
  for device in meta.device.unique():
   for k in [0,1]:
    ix=np.array([(meta.loc[s,'device']==device and t==k) for s,t in zip(ids,y)]);private_split.append(dict(condition=cond,seed=seed,context=c,inner=i,device=device,label='PD' if k==0 else 'DD',count=int(ix.sum()),recall=float(np.mean(decision[ix]==y[ix])) if ix.any() else np.nan,status='NA_EMPTY_CLASS' if not ix.any() else 'PRIVATE_SPLIT_GROUP'))
 a=pd.DataFrame(appearances);patient=a.groupby(['subject_id','condition','label','device']).correct.mean().reset_index();private_path=PRIVATE/'subject_diagnostics/device_patient_accuracy.csv';private_path.parent.mkdir(exist_ok=True,parents=True);patient.to_csv(private_path,index=False);pd.DataFrame(private_split).to_csv(PRIVATE/'subject_diagnostics/device_split_metrics.csv',index=False);rng=np.random.default_rng(SEED);rows=[]
 for (device,label),group in patient.groupby(['device','label']):
  pivot=group.pivot(index='subject_id',columns='condition',values='correct');n=len(pivot);assert n>=10;ix=rng.integers(0,n,(10000,n));draw={cond:pivot[cond].to_numpy()[ix].mean(1) for cond in CONDITIONS}
  for model in ['STR','WSSL']:
   old=model+'-LEGACY-L1';new=model+'-CERTIFIED-L1';d=draw[new]-draw[old];lo,hi=np.quantile(d,[.025,.975]);rows.append(dict(device=device,label=label,model=model,unique_subjects=n,legacy_recall=pivot[old].mean(),legacy_ci_low=np.quantile(draw[old],.025),legacy_ci_high=np.quantile(draw[old],.975),certified_recall=pivot[new].mean(),certified_ci_low=np.quantile(draw[new],.025),certified_ci_high=np.quantile(draw[new],.975),delta=pivot[new].mean()-pivot[old].mean(),delta_ci_low=lo,delta_ci_high=hi,estimand='patient mean of correctness over repeated legal validation appearances and seeds',draws=10000,split_empty_class_groups=sum(r['count']==0 for r in private_split if r['device']==device and r['label']==label and r['condition']==new),causal=False))
 pd.DataFrame(rows).to_csv(HERE/'analysis/device_stratified_summary.csv',index=False)

def signals():
 data=load(PRIVATE/'manifests'/f'{VERSION}.json');old={(r['subject_id'],r['activity'],r['wrist']):r for r in records()};rows=[]
 for r in data['records']:
  sid,a,w=r['subject_id'],r['activity'],r['wrist'];o=old[sid,a,w];new=np.load(r['output_path']).astype(float)[:3];legacy=np.load(o['processed_path']).astype(float)[:3];assert new.shape==legacy.shape;delta=new-legacy
  with np.load(r['certificate_arrays_path'],allow_pickle=False) as z:trend=z['trend'][:,48:].copy()
  value=dict(subject_id=sid,activity=a,wrist=w,device=o['device'],label=r['label'],new_residual_RMS=np.sqrt(np.mean(new**2)),legacy_residual_RMS=np.sqrt(np.mean(legacy**2)),new_trend_RMS=np.sqrt(np.mean(trend**2)),difference_RMS=np.sqrt(np.mean(delta**2)),edge_difference_RMS=np.sqrt(np.mean(np.concatenate([delta[:,:64],delta[:,-64:]],1)**2)),interior_difference_RMS=np.sqrt(np.mean(delta[:,64:-64]**2)))
  for name,array in [('new',new),('legacy',legacy)]:
   freq=np.fft.rfftfreq(array.shape[1],.01);power=abs(np.fft.rfft(array,axis=1))**2/array.shape[1]**2;power[:,1:-1]*=2
   for lo,hi in [(0,3),(3,12),(12,15),(15,50.00001)]:value[f'{name}_energy_{lo}_{int(hi)}Hz']=np.sum(power[:,(freq>=lo)&(freq<hi)],1).mean()
  rows.append(value)
 f=pd.DataFrame(rows);f.to_csv(PRIVATE/'subject_diagnostics/signal_change_records.csv',index=False);cols=[c for c in f.columns if c not in ['subject_id','activity','wrist','device','label']];patients=f.groupby(['subject_id','device','label'])[cols].mean().reset_index();summary=[]
 for (device,label),g in patients.groupby(['device','label']):
  for metric in cols:summary.append(dict(device=device,label=label,unique_subjects=len(g),metric=metric,mean=g[metric].mean(),median=g[metric].median(),p25=g[metric].quantile(.25),p75=g[metric].quantile(.75),patient_first=True,processing_difference_not_disease_information=True))
 pd.DataFrame(summary).to_csv(HERE/'analysis/device_signal_change_summary.csv',index=False);print('SIGNAL_DESCRIPTIONS_COMPLETE',flush=True)

def main():
 for kind in ['wssl','str']:assert load(HERE/'analysis'/f'{kind}_training_integrity.json')['status']=='PASS_COMPLETE_45_RUNS'
 from statistic_tests_l2 import main as synthetic_metric_checks
 synthetic_metric_checks()
 guard();freeze('ANALYSIS',[Path(__file__),HERE/'scripts/weighted_l2.py',HERE/'scripts/predictions_l2.py'],additional_model_fits=0,subject_draws=10000,performance_rule_unchanged=True);stage_report('str');frame,cat,lineage=catalog();assert len(frame)==180;frame.to_csv(HERE/'analysis/metrics_by_split_seed.csv',index=False);write(PRIVATE/'manifests/all_prediction_sources.json',lineage)
 # Historical summary is a check only, never the source of paired observations.
 expected={'WSSL-LEGACY-L1':[.745604,.719644,.759553,.706589,.782344,.656945],'STR-LEGACY-L1':[.730718,.697189,.717631,.686068,.777864,.616515]}
 for cond,v in expected.items():assert np.allclose(frame[frame.condition==cond][METRICS[:6]].mean(),v,atol=1e-6,rtol=0)
 g=statistics(frame,cat);devices(cat);signals();write(HERE/'analysis/analysis_integrity.json',dict(status='PASS_FOUR_CELL_FIXED_ANALYSIS',actual_prediction_files=180,new_fits=0,unique_subjects=390,subject_cluster_draws=10000,paired_split_draws=10000,performance_promotion=g['status'],legacy_summary_checks=True,protected_files_unchanged=guard()));print(json.dumps(dict(stage='ANALYSIS_COMPLETE',performance=g['status'])),flush=True)
if __name__=='__main__':
 main()
 from cost_l2 import main as cost_main
 cost_main()
 from reports_l2 import main as reports_main
 reports_main()
