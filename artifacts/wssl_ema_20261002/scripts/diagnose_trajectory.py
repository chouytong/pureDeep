"""Read archived logs; evaluate only actual best/last weights in eval mode."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
H=Path(__file__).resolve().parent.parent;F=Path('/home/zyt/deep_final/foundation_validation');B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path[:0]=[str(F),str(B/'scripts')]
from transfer_hooks import FrozenEmbeddingCache,SSLLoader
from independent_wssl import IndependentWSSL
from src.datasets.builders import load_configured_records
from src.datasets.subject_activity import SubjectActivityDataset
from src.engine import nested_training as nt
from src.utils.config import load_config
from src.utils.seed import seed_everything

def main():
 assert json.loads((H/'analysis/independence_test.json').read_text())['status']=='PASS'
 seed_everything(42,True);torch.set_num_threads(4);cache=FrozenEmbeddingCache();cfg=load_config(str(F/'configs/str01_seed42.yaml'));records=load_configured_records(cfg['data'],['PD','DD']);rows=[];around=[];evalrows=[]
 for seed in (42,43,44):
  for oi in range(5):
   for ii in range(3):
    path=B/f'runs/b_str_pretrained/seed{seed}/outer_{oi}/inner_{ii}';hist=[json.loads(l) for l in (path/'logs/epochs.jsonl').read_text().splitlines()];best=json.loads((path/'stage_status.json').read_text())['summary']['best_epoch'];sel=hist[best-1];last=hist[-1]
    row=dict(seed=seed,context=oi,inner=ii,best_epoch=best,stop_epoch=len(hist),epochs_after_best=len(hist)-best)
    for name,r in [('best',sel),('last',last)]:
     for role in ('train','validation'):
      for key in ('balanced_accuracy','macro_auroc','classification_loss','loss'):
       row[f'{name}_{role}_{key}']=r[role].get(key,np.nan)
    rows.append(row)
    for offset in (-2,-1,0,1,2):
     ix=best-1+offset
     if 0<=ix<len(hist):around.append(dict(seed=seed,context=oi,inner=ii,offset=offset,epoch=ix+1,**{role+'_'+k:hist[ix][role].get(k,np.nan) for role in ('train','validation') for k in ('balanced_accuracy','macro_auroc','classification_loss')}))
    split=json.loads((path/'split.json').read_text())
    for which in ('best','last'):
     ck=torch.load(path/f'checkpoints/{which}.pt',weights_only=False,map_location='cpu');model=IndependentWSSL(ck['config']).cuda().eval();model.load_state_dict(ck['model_state']);criterion=nt.build_loss(ck['config']).cuda();ck['config']['data']['num_workers']=0;ck['config']['data']['pin_memory']=False
     for role in ('train','validation'):
      ids=set(split[role+'_subjects']);ds=SubjectActivityDataset([r for r in records if r.subject_id in ids],cache.activities,cfg['data'],ck['normalization']['mean'],ck['normalization']['std'],role)
      loader=nt._loader(ds,ck['config'],train=False,seed_offset=23 if role=='train' else 1)
      metrics,preds=nt.evaluate_epoch(model,SSLLoader(loader,model,cache,'pretrained'),criterion,torch.device('cuda'))
      if role=='validation':assert abs(metrics['balanced_accuracy']-hist[ck['epoch']]['validation']['balanced_accuracy'])<1e-12
      evalrows.append(dict(seed=seed,context=oi,inner=ii,checkpoint=which,role=role,epoch=ck['epoch']+1,n=metrics['sample_count'],**{k:metrics.get(k,np.nan) for k in ('accuracy','balanced_accuracy','macro_auroc','macro_f1','classification_loss','loss')}))
     del model,ck
    print(f'diagnosed {seed}/{oi}/{ii}',flush=True)
 pd.DataFrame(rows).to_csv(H/'analysis/trajectory_45runs.csv',index=False);pd.DataFrame(around).to_csv(H/'analysis/trajectory_best_neighborhood.csv',index=False);ev=pd.DataFrame(evalrows);ev.to_csv(H/'analysis/eval_mode_best_last.csv',index=False)
 summary=ev.groupby(['checkpoint','role'])[['accuracy','balanced_accuracy','macro_auroc','macro_f1','classification_loss']].mean().reset_index();summary.to_csv(H/'analysis/eval_mode_summary.csv',index=False)
 t=pd.DataFrame(rows);result=dict(status='PASS',archived_runs=45,evaluated_actual_checkpoints=90,intermediate_checkpoints_available=False,retrospective_ema_available=False,median_best_epoch=float(t.best_epoch.median()),median_stop_epoch=float(t.stop_epoch.median()),best_range=[int(t.best_epoch.min()),int(t.best_epoch.max())],stop_range=[int(t.stop_epoch.min()),int(t.stop_epoch.max())],last_minus_best_validation_ba=float((t.last_validation_balanced_accuracy-t.best_validation_balanced_accuracy).mean()),last_minus_best_validation_auroc=float((t.last_validation_macro_auroc-t.best_validation_macro_auroc).mean()),last_minus_best_validation_classification_loss=float((t.last_validation_classification_loss-t.best_validation_classification_loss).mean()),outer_access=False)
 (H/'analysis/trajectory_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(summary.to_string(index=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
