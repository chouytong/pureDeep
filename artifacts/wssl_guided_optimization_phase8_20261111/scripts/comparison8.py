"""Final fixed-family comparisons and shared subject-cluster sensitivity."""
from common8 import *
from threadpoolctl import threadpool_limits

CANDIDATES=['probability_ensemble','logistic_regression','linear_svm','rbf_svm','random_forest','xgboost','activity_interaction','subject_supcon']

def weighted(y,p,z,w):
 y=np.asarray(y,int);p=np.asarray(p,float);w=np.asarray(w,float);decision=p[:,1]>.5;n=w.sum(1);a=w[:,y==0].sum(1);b=w[:,y==1].sum(1)
 tn=w@((y==0)&~decision);tp=w@((y==1)&decision);fp=a-tn;fn=b-tp
 with np.errstate(invalid='ignore',divide='ignore'):
  result=np.stack([(tn+tp)/n,.5*(tn/a+tp/b),np.zeros(len(w)),.5*(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn)),tn/a,tp/b,np.zeros(len(w)),np.zeros(len(w)),np.zeros(len(w))],axis=1)
  # Match the project's macro two-probability-column AUROC, including ties.
  pos=y==1;neg=~pos
  def compare(x,v):return (x[:,None]>v[None,:]).astype(float)+.5*(x[:,None]==v[None,:])
  kernel=.5*(compare(p[pos,1],p[neg,1])+compare(p[neg,0],p[pos,0]).T)
  result[:,2]=np.sum((w[:,pos]@kernel)*w[:,neg],axis=1)/(a*b)
  loss=logsumexp(z.astype(float),1)-z[np.arange(len(y)),y] if len(z) else -np.log(p[np.arange(len(y)),y])
  finite=np.isfinite(loss);result[:,6]=(w[:,finite]@loss[finite])/n
  result[(w[:,~finite].sum(1)>0),6]=np.inf
  result[:,7]=(w@np.sum((p-np.eye(2)[y])**2,axis=1))/n
  confidence=p.max(1);binid=np.minimum((confidence*10).astype(int),9);correct=decision==y
  result[:,8]=sum(np.abs(w[:,binid==j]@(correct[binid==j]-confidence[binid==j])) for j in range(10))/n
 result[(a==0)|(b==0)]=np.nan
 return result

def main():
 assert load(HERE/'analysis/supcon_integrity.json')['status']=='PASS_COMPLETE_C';guard()
 freeze('D',[HERE/'PROTOCOL.md',HERE/'scripts/common8.py',Path(__file__)],B_C_archived=True,bootstrap_draws=10000,primary_family_size=16,secondary_family_size=56,missing_tests_adjustment_p=1)
 frames=[pd.read_csv(HERE/'analysis'/n) for n in ['ensemble_metrics.csv','readout_metrics.csv','activity_interaction_metrics.csv','supcon_metrics.csv']]
 frame=pd.concat(frames,ignore_index=True);frame.to_csv(HERE/'analysis/all_model_metrics.csv',index=False);base=frame[frame.condition=='baseline'];stats=[];decisions=[]
 for name in CANDIDATES+['logit_ensemble']:
  f=frame[frame.condition==name]
  if len(f)==0:
   stats.extend(dict(condition=name,metric=m,status='NOT_RUN_DEPENDENCY_MISSING',wilcoxon_p=np.nan) for m in METRICS);decisions.append(dict(condition=name,status='NOT_RUN_DEPENDENCY_MISSING'));continue
  stat=paired(f,base,name);stats.extend(stat.to_dict('records'));decisions.append(gate(f,stat,base,name,ensemble=name.endswith('_ensemble')))
 stat=pd.DataFrame(stats);primary=stat.condition.isin(CANDIDATES)&stat.metric.isin(['ba','auroc']);secondary=stat.condition.isin(CANDIDATES)&~stat.metric.isin(['ba','auroc']);sensitivity=stat.condition=='logit_ensemble'
 assert primary.sum()==16 and secondary.sum()==56 and sensitivity.sum()==9
 stat['global_primary_BH16_q']=np.nan;stat.loc[primary,'global_primary_BH16_q']=bh_planned(stat.loc[primary,'wilcoxon_p']);stat['global_secondary_BH56_q']=np.nan;stat.loc[secondary,'global_secondary_BH56_q']=bh_planned(stat.loc[secondary,'wilcoxon_p']);stat['logit_sensitivity_BH9_q']=np.nan;stat.loc[sensitivity,'logit_sensitivity_BH9_q']=bh_planned(stat.loc[sensitivity,'wilcoxon_p']);stat.to_csv(HERE/'analysis/final_paired_statistics.csv',index=False)
 summary=[];stability=[]
 for condition,f in frame.groupby('condition'):
  unit=f.groupby(['context','inner'])[METRICS].mean();seedmeans=f.groupby('seed')[METRICS].mean()
  summary.append(dict(condition=condition,status='OFFICIAL_RETAINED_SINGLE_MODEL' if condition=='baseline' else next(d['status'] for d in decisions if d['condition']==condition),paired_split_count=len(unit),metric_row_count=len(f),**unit.mean().to_dict()))
  for m in METRICS:stability.append(dict(condition=condition,metric=m,within_split_seed_SD_mean=f.groupby(['context','inner'])[m].std(ddof=1).mean() if f.seed.nunique()==3 else np.nan,SD_across_three_seed_means=seedmeans[m].std(ddof=1) if f.seed.nunique()==3 else np.nan,between_split_SD=unit[m].std(ddof=1),fixed_ensemble_seed_variance_identifiable=False if condition.endswith('_ensemble') else True))
 summary.append(dict(condition='xgboost',status='NOT_RUN_DEPENDENCY_MISSING',paired_split_count=0,metric_row_count=0));pd.DataFrame(summary).to_csv(HERE/'analysis/final_experiment_state.csv',index=False);pd.DataFrame(stability).to_csv(HERE/'analysis/seed_stability.csv',index=False)
 write(HERE/'analysis/final_decisions.json',dict(decisions=decisions,baseline='OFFICIAL_RETAINED_SINGLE_MODEL',primary_family_size=16,secondary_family_size=56,logit_sensitivity_size=9,unavailable_p_for_adjustment=1,unavailable_own_p_q=None,positive_class='DD',threshold=.5,tie='PD'))
 # Only private Phase8 validation predictions, identity checked against each split.
 files=sorted((HERE/'predictions').glob('*.npz'));assert len(files)==345
 label={};catalog=[]
 for path in files:
  stem=path.stem;condition,part=stem.rsplit('_seed',1);seedpart,cp=part.split('_context');cpart,ip=cp.split('_inner');seed,c,i=int(seedpart),int(cpart),int(ip)
  sp=next(s for cc,ii,s,e in splits() if (cc,ii)==(c,i))
  with np.load(path,allow_pickle=False) as z:
   ids=ids_ok(z['subject_ids'],sp['validation_subjects']);y=z['target'].astype(int);p=z['probabilities'].astype(float);raw=z['logits'].astype(float)
  for sid,target in zip(ids,y):
   if sid in label:assert label[sid]==target
   label[sid]=int(target)
  ones=weighted(y,p,raw,np.ones((1,len(ids))))[0];canonical=np.array(list(pred_metrics(y,p,raw if len(raw) else None).values()));assert np.allclose(ones,canonical,atol=2e-12,rtol=0,equal_nan=True)
  catalog.append((condition,seed,c,i,path,ids,y,p,raw))
 ids=sorted(label);index={s:i for i,s in enumerate(ids)};labels=np.array([label[s] for s in ids]);assert np.bincount(labels).tolist()==[276,114]
 rng=np.random.default_rng(SEED);weights=np.zeros((10000,len(ids)),np.int16)
 for klass in [0,1]:
  keep=np.flatnonzero(labels==klass);draw=rng.multinomial(len(keep),np.repeat(1/len(keep),len(keep)),size=10000);weights[:,keep]=draw
 # Every occurrence of a patient receives its one common cluster weight.
 sums={};counts={};point={};maxerr=0.;start=time.perf_counter()
 with threadpool_limits(limits=4):
  for k,(condition,seed,c,i,path,localids,y,p,z) in enumerate(catalog):
   w=weights[:,[index[s] for s in localids]];values=weighted(y,p,z,w);key=(condition,c,i)
   if key not in sums:sums[key]=np.zeros_like(values);counts[key]=0;point[key]=np.zeros(9)
   sums[key]+=values;counts[key]+=1;point[key]+=weighted(y,p,z,np.ones((1,len(y))))[0]
   if (k+1)%45==0:print(json.dumps(dict(stage='D_cluster',predictions=k+1,elapsed_seconds=time.perf_counter()-start)),flush=True)
 cluster=[]
 for condition in CANDIDATES+['logit_ensemble']:
  if condition=='xgboost':
   cluster.extend(dict(condition=condition,metric=m,status='NOT_RUN_DEPENDENCY_MISSING',bootstrap_draws=10000) for m in METRICS);continue
  differences=[];points=[]
  for c,i,sp,ex in splits():
   a=(condition,c,i);b=('baseline',c,i)
   with np.errstate(invalid='ignore'):differences.append(sums[a]/counts[a]-sums[b]/counts[b]);points.append(point[a]/counts[a]-point[b]/counts[b])
  values=np.mean(differences,axis=0);estimate=np.mean(points,axis=0)
  for j,m in enumerate(METRICS):
   valid=np.isfinite(values[:,j]);original=stat[(stat.condition==condition)&(stat.metric==m)].iloc[0];expected=original.get('mean_delta',np.nan)
   if np.isfinite(expected):maxerr=max(maxerr,abs(estimate[j]-expected));assert abs(estimate[j]-expected)<2e-12
   good=valid.sum()>=9500 and np.isfinite(estimate[j]);ci=np.quantile(values[valid,j],[.025,.975]) if good else [np.nan,np.nan]
   cluster.append(dict(condition=condition,metric=m,status='SHARED_SUBJECT_CLUSTER_SENSITIVITY' if good else 'NA_NONFINITE_OR_INSUFFICIENT_DRAWS',mean_delta=estimate[j],ci_low=ci[0],ci_high=ci[1],bootstrap_draws=10000,finite_draws=int(valid.sum()),invalid_draws=int((~valid).sum()),PD_cluster_count=276,DD_cluster_count=114,paired_split_count=15,estimand='equal15 split mean of within-split seedmean metric differences',independent_external_evidence=False))
 pd.DataFrame(cluster).to_csv(HERE/'analysis/subject_cluster_bootstrap.csv',index=False)
 write(HERE/'tests/comparison_metrics.json',dict(status='PASS',prediction_files_verified=345,weighted_ones_matches_canonical=True,cluster_point_max_error=maxerr,shared_patient_weights_all_appearances=True,AUROC_uses_both_probability_columns=True,seed_first_estimand=True,primary_tests=16,secondary_tests=56,logit_tests=9,absent_XGB_tests_retained=True,RF_infinite_NLL_unclipped_and_inferential_NA=True))
 write(HERE/'analysis/comparison_integrity.json',dict(status='PASS_COMPLETE_D_COMPARISON',new_fits=0,bootstrap_draws=10000,elapsed_seconds=time.perf_counter()-start,protected_files_unchanged=guard()))

if __name__=='__main__':main()
