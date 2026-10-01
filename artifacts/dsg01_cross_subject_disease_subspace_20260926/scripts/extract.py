#!/usr/bin/env python3
"""DSG-01: frozen, inner-development-only representation extraction."""
from __future__ import annotations
import argparse,csv,gzip,json,re,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/zyt/deep_final')
FV=ROOT/'foundation_validation'
sys.path.insert(0,str(FV));sys.path.insert(0,str(FV/'scripts'))
from src.datasets.builders import make_loader
from src.models import build_model
from src.utils.config import load_config
from extract_str01_gain_mechanism import datasets
BASE=ROOT/'artifacts/dsg01_cross_subject_disease_subspace_20260926'
OLD=ROOT/'artifacts/str01_gain_mechanism_diagnosis_20260923/representation_extract/embeddings'
RUNS={
 'v8':{s:FV/f'outputs/foundation_validation/groupnorm_nr01_seed{s}_20260917' for s in (42,43,44)},
 'str':{s:FV/f'outputs/structured_token_residual/str01_seed{s}_20260921' for s in (42,43,44)},
}
TYPES={'v8':['activity_raw','activity_context','subject_embedding','subject_logits','final_logits'],
 'str':['activity_raw','activity_context','subject_embedding','structured_tokens','structured_embedding','subject_logits','structured_logits','final_logits']}
ACTIVITY_TYPES={'activity_raw','activity_context','structured_tokens'}
def stage_path(model,seed,outer,inner):return RUNS[model][seed]/f'outer_{outer}'/f'inner_{inner}'
def audit_assets():
 report=[];expected=None
 for model in ('v8','str'):
  for seed in (42,43,44):
   stages=sorted(RUNS[model][seed].glob('outer_*/inner_*'))
   assert len(stages)==15,(model,seed,len(stages))
   for stage in stages:
    assert re.fullmatch(r'outer_[0-4]',stage.parent.name) and re.fullmatch(r'inner_[0-2]',stage.name)
    for filename in ('config.yaml','split.json','normalization.json','checkpoints/best.pt'):
     assert (stage/filename).is_file(),stage/filename
    cfg=load_config(stage/'config.yaml');split=json.loads((stage/'split.json').read_text());norm=json.loads((stage/'normalization.json').read_text())
    tr=set(map(str,split['train_subjects']));va=set(map(str,split['validation_subjects']))
    assert not tr&va and set(map(str,norm['subject_ids']))==tr
    assert norm['fitted_on']=='explicit_training_subjects_only'
    activities=cfg['data']['activities'];assert len(activities)==11 and len(set(activities))==11
    assert cfg['data']['wrist_mode']=='bilateral'
    if expected is None:expected=activities
    assert activities==expected
    report.append({'model':model,'seed':seed,'outer':int(stage.parent.name[-1]),'inner':int(stage.name[-1]),'train_n':len(tr),'validation_n':len(va),'normalization_train_only':True,'checkpoint':str(stage/'checkpoints/best.pt')})
 return report,expected
@torch.inference_mode()
def extract(model,dataset,cfg,device,structured):
 loader=make_loader(dataset,batch_size=int(cfg['evaluation']['batch_size']),shuffle=False,num_workers=int(cfg['data'].get('num_workers',0)),pin_memory=False,seed=cfg['experiment']['seed']+2601)
 keys=TYPES['str' if structured else 'v8'];parts={k:[] for k in keys};parts.update(subject_id=[],label=[],activity_mask=[])
 model.eval()
 for batch in loader:
  captured=[]
  hook=model.activity_aggregator.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
  output=model(batch['x'].to(device),batch['wrist_mask'].to(device),batch['activity_mask'].to(device),batch['activity_lengths'].to(device));hook.remove()
  assert len(captured)==1
  tensors={'activity_raw':captured[0],'activity_context':output['activity_embeddings'],'subject_embedding':output['bag_embedding'],'subject_logits':output['base_logits'],'final_logits':output['logits']}
  if structured:tensors.update(structured_tokens=output['structured_activity_tokens'],structured_embedding=output['structured_residual_embedding'],structured_logits=output['structured_residual_logits'])
  for k in keys:parts[k].append(tensors[k].cpu().numpy())
  parts['label'].append(batch['y'].numpy());parts['activity_mask'].append(batch['activity_mask'].numpy());parts['subject_id'].extend(str(x) for x in batch['subject_id'])
 payload={k:np.concatenate(v,axis=0) if k!='subject_id' else np.asarray(v) for k,v in parts.items()}
 assert payload['activity_raw'].shape[1:]==(11,258)
 assert payload['subject_embedding'].shape[1:]==(258,)
 if structured:
  assert payload['structured_tokens'].shape[1:]==(11,16) and payload['structured_embedding'].shape[1:]==(176,)
  assert np.max(np.abs(payload['final_logits']-payload['subject_logits']-payload['structured_logits']))<1e-5
 else:assert np.array_equal(payload['final_logits'],payload['subject_logits'])
 return payload

def write_records(writer,name,seed,outer,inner,role,payload,activities):
 for i,(sid,label) in enumerate(zip(payload['subject_id'],payload['label'])):
  for typ in TYPES[name]:
   if typ in ACTIVITY_TYPES:
    for aid,activity in enumerate(activities):
     if payload['activity_mask'][i,aid]:writer.writerow([name,seed,outer,inner,role,sid,int(label),activity,typ,i,aid])
   else:writer.writerow([name,seed,outer,inner,role,sid,int(label),'ALL',typ,i,''])

def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 audit,activities=audit_assets();BASE.mkdir(exist_ok=True)
 (BASE/'asset_audit.json').write_text(json.dumps({'rows':audit,'activities':activities,'wrist_order':['left','right'],'outer_information_used':False},indent=2))
 out=BASE/('smoke' if args.smoke else 'extraction');out.mkdir(exist_ok=True)
 em=out/'embeddings';em.mkdir(exist_ok=True)
 checks=[]
 with gzip.open(out/'records.csv.gz','wt',newline='') as f:
  writer=csv.writer(f);writer.writerow(['model','seed','outer','inner','role','subject_id','label','activity_id','representation_type','row_index','activity_index'])
  jobs=[(s,o,i) for s in (42,43,44) for o in range(5) for i in range(3)]
  if args.limit:jobs=jobs[:args.limit]
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  for seed,outer,inner in jobs:
   old=OLD/f'seed{seed}_outer{outer}_inner{inner}.npz';assert old.is_file()
   with np.load(old,allow_pickle=False) as archive:
    for name in ('v8','str'):
     stage=stage_path(name,seed,outer,inner);cfg=load_config(stage/'config.yaml');model=build_model(cfg).to(device).eval();ckpt=torch.load(stage/'checkpoints/best.pt',map_location=device,weights_only=False);model.load_state_dict(ckpt['model_state'],strict=True)
     train,val=datasets(stage,cfg)
     for role,dataset in (('train',train),('validation',val)):
      payload=extract(model,dataset,cfg,device,name=='str')
      prefix=f'{name}_{role}_'
      assert np.array_equal(payload['subject_id'],archive[prefix+'subject_id']) and np.array_equal(payload['label'],archive[prefix+'label'])
      err=float(np.max(np.abs(payload['final_logits']-archive[prefix+'final_logits'])))
      assert err<1e-5,(name,seed,outer,inner,role,err)
      np.savez_compressed(em/f'{name}_seed{seed}_outer{outer}_inner{inner}_{role}.npz',**payload)
      write_records(writer,name,seed,outer,inner,role,payload,activities)
      checks.append({'model':name,'seed':seed,'outer':outer,'inner':inner,'role':role,'subjects':len(payload['subject_id']),'old_logit_max_delta':err})
     del model;torch.cuda.empty_cache()
   print('done',seed,outer,inner,flush=True)
 with (out/'extraction_checks.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=checks[0]);writer.writeheader();writer.writerows(checks)
 (out/'protocol.json').write_text(json.dumps({'scope':'fixed inner-development train/validation only','outer_information_used':False,'models_trained':False,'forward_modified':False,'seeds':[42,43,44],'jobs':len(jobs),'activity_order':activities,'wrist_order':['left','right'],'representation_records':'records.csv.gz links each vector to archive row and activity index'},indent=2))
if __name__=='__main__':main()
