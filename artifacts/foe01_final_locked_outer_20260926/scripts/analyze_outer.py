#!/usr/bin/env python3
"""Locked FOE summary: five paired outer folds, seed-first deep aggregation."""
from pathlib import Path
import csv,json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score,confusion_matrix
B=Path('/home/zyt/deep_final/artifacts/foe01_final_locked_outer_20260926');O=B/'analysis';O.mkdir(exist_ok=True);LOCK=json.loads((B/'LOCKED_BEFORE_OUTER.json').read_text());rng=np.random.default_rng(20260926)
METRICS=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
def metrics(y,p,threshold):
 pred=(p>=threshold).astype(int);cm=confusion_matrix(y,pred,labels=[0,1]);return dict(accuracy=float(accuracy_score(y,pred)),ba=float(balanced_accuracy_score(y,pred)),auroc=float(roc_auc_score(y,p)),macro_f1=float(f1_score(y,pred,average='macro')),pd_recall=float(recall_score(y,pred,pos_label=0)),dd_recall=float(recall_score(y,pred,pos_label=1)),tn=int(cm[0,0]),fp=int(cm[0,1]),fn=int(cm[1,0]),tp=int(cm[1,1]),n=len(y),pd_n=int(sum(y==0)),dd_n=int(sum(y==1)))
def boot(x,n=10000):
 x=np.asarray(x,float);return np.quantile(x[rng.integers(0,len(x),(n,len(x)))].mean(axis=1),[.025,.975]).tolist()
rows=[];records=[]
for model in ('STR-01','V8-GN'):
 for seed in (42,43,44):
  for outer in range(5):
   path=B/'model_runs'/model/f'seed{seed}'/f'outer_{outer}/outer_final/outer_test/predictions.csv';frame=pd.read_csv(path,dtype={'subject_id':str});y=frame.target.to_numpy(int);p=frame.probability_dd.to_numpy(float);threshold=LOCK['deep_models'][model][str(seed)]['selection'][outer]['threshold']
   assert len(frame)==len(set(frame.subject_id)) and np.array_equal((p>=threshold).astype(int),frame.prediction_thresholded.to_numpy(int))
   for rule,t in [('locked_inner_oof_threshold',threshold),('default_0p5',.5)]:rows.append(dict(model=model,seed=seed,outer_fold=outer,rule=rule,threshold=t,**metrics(y,p,t)))
   records.extend(dict(model=model,seed=seed,outer_fold=outer,subject_id=str(r.subject_id),target=int(r.target),probability_dd=float(r.probability_dd),locked_threshold=threshold,locked_prediction=int(r.prediction_thresholded),default_prediction=int(r.prediction)) for r in frame.itertuples())
for outer in range(5):
 frame=pd.read_csv(B/'model_runs/H1'/f'outer_{outer}/predictions.csv',dtype={'subject_id':str});y=frame.target.to_numpy(int);p=frame.probability_dd.to_numpy(float)
 rows.append(dict(model='H1',seed=np.nan,outer_fold=outer,rule='locked_0p5',threshold=.5,**metrics(y,p,.5)))
 records.extend(dict(model='H1',seed=np.nan,outer_fold=outer,subject_id=str(r.subject_id),target=int(r.target),probability_dd=float(r.probability_dd),locked_threshold=.5,locked_prediction=int(r.prediction),default_prediction=int(r.prediction)) for r in frame.itertuples())
seedfold=pd.DataFrame(rows);seedfold.to_csv(O/'outer_seed_fold_metrics.csv',index=False);pd.DataFrame(records).to_csv(O/'outer_prediction_records.csv.gz',index=False,compression='gzip')
fold=seedfold.groupby(['model','outer_fold','rule'],as_index=False).mean(numeric_only=True);fold.to_csv(O/'outer_five_fold_seed_aggregated.csv',index=False)
summary=[]
for (model,rule),g in fold.groupby(['model','rule']):
 for m in METRICS:
  v=g[m].to_numpy();ci=boot(v);summary.append(dict(model=model,rule=rule,metric=m,mean=v.mean(),sd=v.std(ddof=1),median=np.median(v),min=v.min(),max=v.max(),ci95_low=ci[0],ci95_high=ci[1],outer_folds=5))
pd.DataFrame(summary).to_csv(O/'outer_metric_summary_fold_unit.csv',index=False)
comp=[]
for rule in ('locked_inner_oof_threshold','default_0p5'):
 a=fold[(fold.model=='STR-01')&(fold.rule==rule)].set_index('outer_fold')
 for other,orule in [('V8-GN',rule),('H1','locked_0p5')]:
  b=fold[(fold.model==other)&(fold.rule==orule)].set_index('outer_fold')
  for m in METRICS:
   d=(a[m]-b[m]).to_numpy();ci=boot(d);comp.append(dict(comparison=f'STR-01_minus_{other}',str_rule=rule,reference_rule=orule,metric=m,mean_difference=d.mean(),sd_difference=d.std(ddof=1),median_difference=np.median(d),ci95_low=ci[0],ci95_high=ci[1],improved_folds=int(sum(d>0)),worsened_folds=int(sum(d<0)),p_wilcoxon=float(wilcoxon(d).pvalue) if np.any(d) else 1.0,fold_differences=json.dumps(d.tolist())))
pd.DataFrame(comp).to_csv(O/'outer_paired_differences_five_fold.csv',index=False)
# Subject-disjoint pooled 390 per seed, descriptive only; no probability ensemble.
pooled=[]
for (model,seed),g in pd.DataFrame(records).groupby(['model','seed'],dropna=False):
 if model=='H1':
  assert len(g)==390 and g.subject_id.nunique()==390;pooled.append(dict(model=model,seed=np.nan,rule='locked_0p5',**metrics(g.target.to_numpy(int),g.probability_dd.to_numpy(float),.5)))
 else:
  assert len(g)==390 and g.subject_id.nunique()==390
  for rule in ('locked_inner_oof_threshold','default_0p5'):
   pred=g.locked_prediction.to_numpy(int) if rule=='locked_inner_oof_threshold' else g.default_prediction.to_numpy(int);y=g.target.to_numpy(int);p=g.probability_dd.to_numpy(float);cm=confusion_matrix(y,pred,labels=[0,1]);pooled.append(dict(model=model,seed=int(seed),rule=rule,accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1),tn=int(cm[0,0]),fp=int(cm[0,1]),fn=int(cm[1,0]),tp=int(cm[1,1]),n=390,pd_n=int(sum(y==0)),dd_n=int(sum(y==1))))
pd.DataFrame(pooled).to_csv(O/'outer_pooled_390_descriptive.csv',index=False)
# Development 0.5 rule from already frozen 15 inner splits; no outer reuse in development selection.
dev=[]
for model in ('STR-01','V8-GN'):
 for seed in (42,43,44):
  root=Path(LOCK['deep_models'][model][str(seed)]['development_inner_root'])
  for outer in range(5):
   for inner in range(3):
    f=pd.read_csv(root/f'outer_{outer}/inner_{inner}/predictions/validation.csv');y=f.target.to_numpy(int);p=f.probability_dd.to_numpy(float);dev.append(dict(model=model,seed=seed,outer_fold=outer,inner_fold=inner,**metrics(y,p,.5)))
h1path=Path('/home/zyt/MFAM/outputs/pads_classification/v3_analysis_and_baselines/analysis_and_baselines_20260828/baselines/H1_handcrafted_logistic/development_predictions_all.csv');h=pd.read_csv(h1path)
for (outer,inner),g in h.groupby(['outer_context','inner_fold']):dev.append(dict(model='H1',seed=np.nan,outer_fold=int(outer),inner_fold=int(inner),**metrics(g.binary_label.to_numpy(int),g.probability_dd.to_numpy(float),.5)))
devframe=pd.DataFrame(dev);devframe.to_csv(O/'development_0p5_split_metrics.csv',index=False)
agg=devframe.groupby(['model','outer_fold','inner_fold'],as_index=False).mean(numeric_only=True);means=agg.groupby('model')[METRICS].mean().reset_index();means.to_csv(O/'development_15split_seedfirst_mean.csv',index=False)
outer05=fold[((fold.model=='H1')&(fold.rule=='locked_0p5'))|((fold.model!='H1')&(fold.rule=='default_0p5'))].groupby('model')[METRICS].mean().reset_index();outerlocked=fold[((fold.model=='H1')&(fold.rule=='locked_0p5'))|((fold.model!='H1')&(fold.rule=='locked_inner_oof_threshold'))].groupby('model')[METRICS].mean().reset_index()
gap=means.merge(outer05,on='model',suffixes=('_development_0p5','_outer_0p5')).merge(outerlocked,on='model');gap=gap.rename(columns={m:f'{m}_outer_locked' for m in METRICS})
for m in METRICS:gap[f'{m}_outer0p5_minus_development']=gap[f'{m}_outer_0p5']-gap[f'{m}_development_0p5']
gap.to_csv(O/'development_outer_comparison.csv',index=False)
print('outer_primary_fold_mean');print(fold[((fold.model=='H1')|(fold.rule=='locked_inner_oof_threshold'))].groupby('model')[METRICS+['tn','fp','fn','tp']].mean().round(4).to_string());print('paired_primary');print(pd.DataFrame(comp).query("str_rule=='locked_inner_oof_threshold' and metric in ['ba','auroc','dd_recall','pd_recall']")[['comparison','metric','mean_difference','ci95_low','ci95_high','improved_folds']].round(4).to_string(index=False));print('development_outer');print(gap[['model','ba_development_0p5','ba_outer_0p5','ba_outer_locked','auroc_development_0p5','auroc_outer_0p5']].round(4).to_string(index=False))
