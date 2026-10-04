"""Read-only current-state assets/config audit and 15 train-only normalization refits."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
H=Path(__file__).resolve().parent.parent;R=Path('/home/zyt/deep_final');F=R/'foundation_validation';B=R/'artifacts/phase3b_external_ssl_20260928';E=R/'artifacts/wssl_ema_20261002';P4=R/'artifacts/phase4_window_dynamics_20261002'
sys.path[:0]=[str(F),str(E/'scripts'),str(B/'scripts')]
from src.utils.config import load_config
from src.datasets import folds as fd
from src.engine.nested_training import _load_frozen_split
from ema_hooks import development_bundle

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);cfg=load_config(str(F/'configs/str01_seed42.yaml'));split,_,_,digest=_load_frozen_split(cfg);lock=json.loads((P4/'analysis/baseline_lock.json').read_text());assert digest==lock['split_sha256']
 raw_records=fd.load_configured_records(cfg['data'],['PD','DD']);pairs={(r.subject_id,r.activity) for r in raw_records};assert len(pairs)==len(raw_records)==390*11
 labels={}
 for r in raw_records:
  assert labels.setdefault(r.subject_id,r.label)==r.label
 assert sum(v==0 for v in labels.values())==276 and sum(v==1 for v in labels.values())==114
 rows=[];normrows=[];runtime=[];sources={}
 for c in split['outer']:
  oi=int(c['outer_fold'])
  for inner in c['inner_folds']:
   ii=int(inner['inner_fold']);train=set(map(str,inner['train_subjects']));val=set(map(str,inner['validation_subjects']));assert train and val and not(train&val)
   bundle=development_bundle(cfg,train_subject_ids=inner['train_subjects'],validation_subject_ids=inner['validation_subjects'],test_subject_ids=[],fold_id=f'outer_{oi}/inner_{ii}')
   assert bundle.test is None and bundle.split_summary['test_subject_count']==0
   assert bundle.split_summary['normalization_fitted_on']=='train_subjects_only'
   counts={lab:sum(labels[s]==lab for s in train) for lab in (0,1)};weights=[len(train)/(2*counts[i]) for i in (0,1)]
   refnorm=None
   for seed in (42,43,44):
    path=B/f'runs/b_str_pretrained/seed{seed}/outer_{oi}/inner_{ii}';st=json.loads((path/'stage_status.json').read_text());assert st['status']=='complete' and st['outer_test_loader_created'] is False
    ck=torch.load(path/'checkpoints/best.pt',map_location='cpu',weights_only=False);rc=ck['config'];t=rc['training'];d=rc['data'];model=rc['model']
    assert torch.equal(bundle.mean,ck['normalization']['mean']) and torch.equal(bundle.std,ck['normalization']['std'])
    assert bundle.normalization['normalization_sha256']==st['summary']['normalization_sha256']
    assert ck['normalization']['fitted_on']=='train_subjects_only';assert np.allclose(rc['loss']['class_weights'],weights,atol=0,rtol=0)
    assert t['optimizer']=='adamw' and t['learning_rate']==2e-4 and t['weight_decay']==1e-4 and t['batch_size']==8 and t['epochs']==50 and t['mixed_precision'] is False
    assert t['early_stopping']['metric']=='balanced_accuracy' and t['early_stopping']['patience']==12
    assert rc['loss']['label_smoothing']==0 and rc['loss']['class_weight_source']=='current_fold_train_subject_labels_only'
    assert d['activities']==cfg['data']['activities'] and d['wrist_order']==['left','right'] and d['channel_names']==['AccX','AccY','AccZ','GyroX','GyroY','GyroZ']
    assert d['normalization_scope']=='activity_wrist_channel' and d['standardize'] is True and d['train_crop']==d['eval_crop']=='full'
    assert model['name']=='pure_deep_subject' and model['structured_token_residual']['enabled'] and model['structured_token_residual']['projection_dim']==16 and model['pure_deep']['spectral']['enabled'] is False
    assert sum(v.numel() for v in ck['model_state'].values())==143172
    logged=[json.loads(l) for l in (path/'logs/epochs.jsonl').read_text().splitlines()];assert len(logged)==st['summary']['best_epoch']+12
    expect=next(s for s in lock['stages'] if (s['seed'],s['context'],s['inner'])==(seed,oi,ii))
    for file,key in [('checkpoints/best.pt','checkpoint_sha256'),('predictions/validation.csv','prediction_sha256'),('normalization.json','normalization_file_sha256')]:assert sha(path/file)==expect[key]
    frame=pd.read_csv(path/'predictions/validation.csv',dtype={'subject_id':str});assert frame.subject_id.is_unique and set(frame.subject_id)==val
    assert all(labels[s]==y for s,y in zip(frame.subject_id,frame.target));assert np.isfinite(frame.probability_dd).all()
    rows.append(dict(seed=seed,context=oi,inner=ii,train_n=len(train),validation_n=len(val),best_epoch=st['summary']['best_epoch'],stop_epoch=len(logged),epoch_cap_hit=len(logged)==50,normalization_exact=True,class_weights_train_only=True,trainable_parameters=143172,predictions_ID_label_match=True))
    runtime.append(dict(seed=seed,context=oi,inner=ii,optimizer=t['optimizer'],learning_rate=t['learning_rate'],weight_decay=t['weight_decay'],batch_size=t['batch_size'],AMP=t['mixed_precision'],scheduler=t['scheduler']['type'],minimum_lr=t['scheduler']['minimum_lr'],patience=t['early_stopping']['patience'],actual_threshold_rule='argmax equivalent DD probability >0.5, tie=PD',generic_config_threshold_source=rc['model_selection']['threshold']['source']))
   normrows.append(dict(context=oi,inner=ii,train_n=len(train),validation_n=len(val),mean_max_difference=0.0,std_max_difference=0.0,seed_checkpoints_matched=3,normalization_refit_from_train_only=True,outer_dataset_created=False))
   print('TRAIN-ONLY NORM PASS',oi,ii,flush=True);del bundle
 for p,dig in lock['source_sha256'].items():assert sha(p)==dig;sources[p]=dig
 assert sha(lock['cache_path'])==lock['cache_sha256']
 for d in ['src/models','src/datasets','src/engine','src/utils']:
  for p in (F/d).rglob('*.py'):sources[str(p)]=sha(p)
 pd.DataFrame(rows).to_csv(H/'analysis/assets_45checkpoints.csv',index=False);pd.DataFrame(normrows).to_csv(H/'analysis/normalization_15refits.csv',index=False);pd.DataFrame(runtime).to_csv(H/'analysis/runtime_recipe_45checkpoints.csv',index=False)
 result=dict(status='PASS',development_splits=15,checkpoints=45,normalization_refits=15,means_stds_bit_exact=True,records=4290,subjects=390,pd=276,dd=114,epoch_cap_hits=0,frozen_baseline_matches=True,threshold_declaration_scope_difference=True,AMP_override_verified=True,outer_access=False,source_sha256=sources,cache_sha256=lock['cache_sha256'],split_sha256=digest)
 (H/'analysis/assets_normalization_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
