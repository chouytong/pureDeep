#!/usr/bin/env python3
import json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,recall_score
B=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926');out=B/'analysis';out.mkdir(exist_ok=True)
formal=Path('/home/zyt/deep_final/foundation_validation/outputs/structured_token_residual')
metrics=[];comparisons=[]
for seed in (42,43,44):
 new=B/'runs/baseline'/f'seed{seed}';old=formal/f'str01_seed{seed}_20260921'
 for outer in range(5):
  for inner in range(3):
   rel=Path(f'outer_{outer}/inner_{inner}');a=new/rel;b=old/rel
   assert json.loads((a/'stage_status.json').read_text())['status']=='complete'
   x=pd.read_csv(a/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True);y=pd.read_csv(b/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   assert x.subject_id.tolist()==y.subject_id.tolist() and np.array_equal(x.target,y.target)
   err=float(np.max(abs(x.probability_dd.to_numpy()-y.probability_dd.to_numpy())))
   sn=json.loads((a/'stage_status.json').read_text())['summary'];so=json.loads((b/'stage_status.json').read_text())['summary']
   hn=[json.loads(z) for z in (a/'logs/epochs.jsonl').read_text().splitlines() if z];ho=[json.loads(z) for z in (b/'logs/epochs.jsonl').read_text().splitlines() if z]
   assert sn['train_subject_ids_sha256']==so['train_subject_ids_sha256'] and sn['normalization_sha256']==so['normalization_sha256']
   yy=x.target.to_numpy(int);p=x.probability_dd.to_numpy(float);pred=x.prediction.to_numpy(int)
   d={'accuracy':accuracy_score(yy,pred),'ba':balanced_accuracy_score(yy,pred),'auroc':roc_auc_score(yy,p),'macro_f1':f1_score(yy,pred,average='macro'),'pd_recall':recall_score(yy,pred,pos_label=0),'dd_recall':recall_score(yy,pred,pos_label=1)}
   metrics.append(dict(seed=seed,outer=outer,inner=inner,n=len(x),best_epoch=sn['best_epoch'],epochs_run=len(hn),last_train_loss=hn[-1]['train']['loss'],last_validation_loss=hn[-1]['validation']['loss'],best_epoch_train_loss=hn[sn['best_epoch']-1]['train']['loss'],best_epoch_validation_loss=hn[sn['best_epoch']-1]['validation']['loss'],**d))
   comparisons.append(dict(seed=seed,outer=outer,inner=inner,max_prob_diff=err,best_epoch_new=sn['best_epoch'],best_epoch_formal=so['best_epoch'],epochs_run_new=len(hn),epochs_run_formal=len(ho),checkpoint_sha_match=sn['checkpoint_sha256']==so['checkpoint_sha256'],normalization_sha_match=True,ba_diff=d['ba']-so['validation_metrics']['balanced_accuracy'],auroc_diff=d['auroc']-so['validation_metrics']['macro_auroc']))
m=pd.DataFrame(metrics);c=pd.DataFrame(comparisons);m.to_csv(out/'baseline_seed_split.csv',index=False);c.to_csv(out/'baseline_reproduction_audit.csv',index=False)
agg=m.groupby(['outer','inner'],as_index=False)[['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']].mean();agg.to_csv(out/'baseline_15split_seedfirst.csv',index=False)
seedmean=m.groupby('seed')[['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']].mean();seedmean.to_csv(out/'baseline_seed_mean.csv')
summary={'status':'PASS' if c.max_prob_diff.max()<1e-7 and c.ba_diff.abs().max()<1e-8 and (c.best_epoch_new==c.best_epoch_formal).all() else 'MISMATCH','units':len(c),'max_prob_diff':c.max_prob_diff.max(),'max_ba_diff':c.ba_diff.abs().max(),'max_auroc_diff':c.auroc_diff.abs().max(),'same_best_epoch':bool((c.best_epoch_new==c.best_epoch_formal).all()),'same_checkpoint_sha':bool(c.checkpoint_sha_match.all()),'mean_metrics':agg[['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']].mean().to_dict(),'seed_sd':seedmean.std(ddof=1).to_dict(),'split_sd':agg[['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']].std(ddof=1).to_dict(),'best_epoch_median':float(m.best_epoch.median()),'best_epoch_range':[int(m.best_epoch.min()),int(m.best_epoch.max())],'epochs_run_median':float(m.epochs_run.median()),'at_best_train_loss_mean':float(m.best_epoch_train_loss.mean()),'at_best_validation_loss_mean':float(m.best_epoch_validation_loss.mean()),'last_train_loss_mean':float(m.last_train_loss.mean()),'last_validation_loss_mean':float(m.last_validation_loss.mean())}
(out/'baseline_reproduction_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));sys.exit(0 if summary['status']=='PASS' else 2)
