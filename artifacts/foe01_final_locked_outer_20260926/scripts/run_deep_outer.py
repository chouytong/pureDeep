#!/usr/bin/env python3
"""Execute only locked outer final refits using the already frozen engine."""
from pathlib import Path
import argparse,csv,json,sys
R=Path('/home/zyt/deep_final');FV=R/'foundation_validation';B=R/'artifacts/foe01_final_locked_outer_20260926';sys.path.insert(0,str(FV))
from src.utils.config import load_config
from src.utils.provenance import sha256_file
from src.engine.nested_training import _run_outer_final
from src.utils.device import select_device

def read_lock():
 p=B/'LOCKED_BEFORE_OUTER.json';actual=sha256_file(p);expected=(B/'LOCKED_BEFORE_OUTER.json.sha256').read_text().split()[0];assert actual==expected
 lock=json.loads(p.read_text());assert lock['status']=='LOCKED_BEFORE_OUTER_ACCESS' and lock['outer_information_accessed_for_this_protocol'] is False
 for name,h in lock['source_and_asset_sha256'].items():
  if name.endswith('handcrafted_features.npz') or name.endswith('feature_schema.json'):continue
  assert sha256_file(Path(name))==h,name
 return lock

def one(lock,model,seed,o,payload,device):
 detail=lock['deep_models'][model][str(seed)];sel=detail['selection'][o];assert sel['outer_fold']==o
 cfgpath=Path(detail['config']);assert sha256_file(cfgpath)==detail['config_sha256'];cfg=load_config(str(cfgpath))
 outer=payload['outer'][o];assert int(outer['outer_fold'])==o and len(outer['train_subjects'])==sel['outer_train_n'] and len(outer['test_subjects'])==sel['outer_test_n']
 root=B/'model_runs'/('STR-01' if model=='STR-01' else 'V8-GN')/f'seed{seed}';outer_dir=root/f'outer_{o}';selection_dir=outer_dir/'selection';selection_dir.mkdir(parents=True,exist_ok=True)
 rows=[]
 for item in sel['inner_oof_prediction_files']:
  i=item['inner'];p=Path(detail['development_inner_root'])/f'outer_{o}/inner_{i}/predictions/validation.csv';assert sha256_file(p)==item['prediction_sha256'];rows.extend(csv.DictReader(p.open()))
 oof=selection_dir/'inner_oof_predictions.csv';fields=('subject_id','target','probability_pd','probability_dd')
 if not oof.exists():
  with oof.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows([{k:r[k] for k in fields} for r in rows])
 ids=[r['subject_id'] for r in rows];assert len(ids)==len(set(ids))==sel['inner_oof_n'] and set(ids)==set(map(str,outer['train_subjects']))
 threshold={'source':'inner_oof_only','positive_class':'DD','decision_rule':'predict DD when p(DD) >= threshold','metric':'balanced_accuracy','threshold':sel['threshold'],'balanced_accuracy':sel['inner_oof_ba_at_selected_threshold'],'sample_count':sel['inner_oof_n']}
 epoch={'strategy':'median_inner_best_epoch','inner_best_epochs':sel['inner_epochs'],'selected_epoch_count':sel['selected_epoch_count']}
 frozen_selection=selection_dir/'selection_locked.json'
 if not frozen_selection.exists():frozen_selection.write_text(json.dumps({'model':model,'seed':seed,'outer_fold':o,'threshold':threshold,'final_epoch':epoch,'lock_sha256':sha256_file(B/'LOCKED_BEFORE_OUTER.json')},indent=2))
 result=_run_outer_final(cfg,outer,outer_dir,device,threshold=threshold,epoch_selection=epoch,oof_path=oof,resume=True)
 assert result['outer_test_evaluation_count']==1 and result['fixed_epoch_count']==sel['selected_epoch_count'] and result['threshold']==sel['threshold']
 print(json.dumps({'status':'complete','model':model,'seed':seed,'outer_fold':o,'selected_epochs':sel['selected_epoch_count'],'threshold':sel['threshold'],'checkpoint_sha256':result['checkpoint_sha256'],'normalization_sha256':result['normalization_sha256']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['STR-01','V8-GN']);ap.add_argument('--seed',type=int);ap.add_argument('--outer',type=int);args=ap.parse_args();lock=read_lock();payload=json.loads((FV/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json').read_text());device=select_device('auto');assert device.type=='cuda'
 models=[args.model] if args.model else ['STR-01','V8-GN'];seeds=[args.seed] if args.seed is not None else [42,43,44];outers=[args.outer] if args.outer is not None else list(range(5))
 for model in models:
  for seed in seeds:
   for o in outers:one(lock,model,seed,o,payload,device)
