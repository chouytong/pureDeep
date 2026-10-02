"""Fixed ordinary-selected EMA comparison, with mandatory baseline trajectory audit."""
import argparse,hashlib,json,sys,gc
from pathlib import Path
import numpy as np,pandas as pd,torch
H=Path(__file__).resolve().parent.parent;F=Path('/home/zyt/deep_final/foundation_validation');B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');P4=Path('/home/zyt/deep_final/artifacts/phase4_window_dynamics_20261002')
sys.path[:0]=[str(F),str(B/'scripts')]
from transfer_hooks import FrozenEmbeddingCache,CACHE_PATH
from independent_wssl import IndependentWSSL
from ema_hooks import activate
from src.engine import nested_training as nt
from src.utils.config import load_config
from src.utils.device import select_device

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('duration_seconds','timestamp','gpu_peak_memory_bytes')}
 if isinstance(x,list):return [clean(v) for v in x]
 return x

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=[42,43,44],required=True);ap.add_argument('--context',type=int);ap.add_argument('--inner',type=int);ap.add_argument('--smoke',action='store_true');args=ap.parse_args();torch.set_num_threads(4)
 assert json.loads((H/'analysis/independence_test.json').read_text())['status']=='PASS'
 assert json.loads((H/'analysis/trajectory_diagnosis.json').read_text())['status']=='PASS'
 lock=json.loads((P4/'analysis/baseline_lock.json').read_text())
 assert sha(CACHE_PATH)==lock['cache_sha256']
 for p,digest in lock['source_sha256'].items():assert sha(p)==digest
 cfg=load_config(str(F/f'configs/str01_seed{args.seed}.yaml'))
 cfg['experiment']['name']=f'wssl_single_ema_seed{args.seed}';cfg['experiment']['output_root']=str(H/('smoke' if args.smoke else 'runs')/'ordinary'/f'seed{args.seed}')
 cfg['development']=dict(single_ema=True,protocol_sha256=sha(H/'PROTOCOL.md'),no_outer_access=True)
 loaders=[];original_loader=nt._loader
 def tracked_loader(*a,**k):
  loader=original_loader(*a,**k)
  if loader is not None:loaders.append(loader)
  return loader
 nt._loader=tracked_loader
 cache=FrozenEmbeddingCache();states=activate(cache);split,_,_,digest=nt._load_frozen_split(cfg);assert digest==lock['split_sha256'];device=select_device('cuda')
 for context in split['outer']:
  oi=int(context['outer_fold'])
  if args.context is not None and oi!=args.context:continue
  for inner in context['inner_folds']:
   ii=int(inner['inner_fold'])
   if args.inner is not None and ii!=args.inner:continue
   stage=Path(cfg['experiment']['output_root'])/f'outer_{oi}/inner_{ii}'
   if (stage/'stage_status.json').exists():raise RuntimeError('No incomplete resume or overwrite; inspect existing stage')
   summary=nt._run_inner_fold(cfg,{'outer_fold':oi,'test_subjects':[]},inner,stage,device,resume=False,smoke=args.smoke)
   old=B/f'runs/b_str_pretrained/seed{args.seed}/outer_{oi}/inner_{ii}'
   hist=[json.loads(l) for l in (stage/'logs/epochs.jsonl').read_text().splitlines()];meta=json.loads((stage/'checkpoints/ema_trajectory.json').read_text());sel=json.loads((stage/'checkpoints/ema_selection.json').read_text())
   assert meta['updates']==len(hist)*(1 if args.smoke else meta['steps_per_epoch'])
   assert sel['best_updates']==summary['best_epoch']*(1 if args.smoke else meta['steps_per_epoch'])
   audit=dict(seed=args.seed,context=oi,inner=ii,smoke=args.smoke,best_epoch=summary['best_epoch'],stop_epoch=len(hist),ema_epoch_matched=True,**meta)
   if not args.smoke:
    oldsum=json.loads((old/'stage_status.json').read_text())['summary'];oldhist=[json.loads(l) for l in (old/'logs/epochs.jsonl').read_text().splitlines()]
    assert len(hist)==len(oldhist),'Ordinary stopping epoch differs'
    assert [clean(r) for r in hist]==[clean(r) for r in oldhist],'Ordinary numeric trajectory differs'
    assert summary['best_epoch']==oldsum['best_epoch']
    for k in ('normalization_sha256','train_subject_ids_sha256'):assert summary[k]==oldsum[k]
    a=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id');b=pd.read_csv(old/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id')
    assert a[['subject_id','target']].equals(b[['subject_id','target']]);assert np.array_equal(a.probability_dd.to_numpy(),b.probability_dd.to_numpy()),'Ordinary predictions differ'
    audit['ordinary_full_trajectory_exact']=True;audit['ordinary_predictions_exact']=True
   (stage/'ordinary_trajectory_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
   ck=torch.load(stage/'checkpoints/ema_at_ordinary_best.pt',map_location='cpu',weights_only=False);assert ck['epoch']+1==summary['best_epoch']
   ema=IndependentWSSL(ck['config']).to(device).eval();ema.load_state_dict(ck['model_state'],strict=True)
   bundle=nt.build_subject_fold_datasets(ck['config'],train_subject_ids=inner['train_subjects'],validation_subject_ids=inner['validation_subjects'],test_subject_ids=[],fold_id=f'outer_{oi}/inner_{ii}')
   metrics,preds=nt.evaluate_epoch(ema,nt._loader(bundle.validation,ck['config'],train=False,seed_offset=1),nt.build_loss(ck['config']).to(device),device,max_batches=1 if args.smoke else 0)
   dest=H/('smoke' if args.smoke else 'runs')/'ema'/f'seed{args.seed}/outer_{oi}/inner_{ii}';dest.mkdir(parents=True)
   nt._write_predictions(dest/'predictions/validation.csv',preds,bundle.class_names)
   (dest/'stage_status.json').write_text(json.dumps(dict(status='complete',phase='inner',outer_test_loader_created=False,summary=dict(**summary,validation_metrics=nt._named_metrics(metrics),checkpoint_sha256=sha(stage/'checkpoints/ema_at_ordinary_best.pt'),ordinary_selected_epoch=True)),indent=2)+'\n')
   (dest/'logs').mkdir();(dest/'logs/epochs.jsonl').write_text((stage/'logs/epochs.jsonl').read_text())  # Ordinary selection history, NOT EMA history.
   print(json.dumps(dict(status='PASS',**audit)),flush=True)
   for st in states.values():st.handle.remove()
   states.clear();del ema,ck,bundle
   for loader in loaders:
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:iterator._shutdown_workers();loader._iterator=None
   loaders.clear();gc.collect()
if __name__=='__main__':main()
