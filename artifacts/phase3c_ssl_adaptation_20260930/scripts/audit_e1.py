import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930');P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
manifest=json.loads((BASE/'manifest.json').read_text());stage4=BASE/'analysis/stage4_windows.npz'
assert hashlib.sha256(stage4.read_bytes()).hexdigest()==manifest['stage4_cache_sha256']
source={line.strip().split('  ')[1]:line.strip().split('  ')[0] for line in (BASE/'e1_source_sha256.txt').read_text().splitlines()}
for file,sha in source.items():
 path=BASE/file
 if file=='scripts/phase3c_hooks.py':path=BASE/'scripts/phase3c_hooks_e1_frozen.py'
 if file=='scripts/run_inner.py':path=BASE/'scripts/run_inner_e1_frozen.py'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,(file,str(path))
rows=[];reference_bn=None
for seed in (42,43,44):
 for outer in range(5):
  for inner in range(3):
   a=BASE/f'runs/adapt/p100/seed{seed}/outer_{outer}/inner_{inner}'
   b=P3B/f'runs/b_str_pretrained/seed{seed}/outer_{outer}/inner_{inner}'
   sa=json.loads((a/'stage_status.json').read_text());sb=json.loads((b/'stage_status.json').read_text())
   assert sa['status']=='complete' and sa['outer_test_loader_created'] is False
   assert sa['summary']['train_subject_ids_sha256']==sb['summary']['train_subject_ids_sha256']
   assert sa['summary']['normalization_sha256']==sb['summary']['normalization_sha256']
   fa=pd.read_csv(a/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   fb=pd.read_csv(b/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   assert fa.subject_id.is_unique and fa[['subject_id','target']].equals(fb[['subject_id','target']])
   path=a/'checkpoints/best.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==sa['summary']['checkpoint_sha256']
   c=torch.load(path,map_location='cpu',weights_only=False);state=c['model_state']
   bn=(state['layer5.1.running_mean'],state['layer5.1.running_var'])
   if reference_bn is None:reference_bn=bn
   else:assert all(torch.equal(x,y) for x,y in zip(bn,reference_bn))
   groups=c['optimizer_state']['param_groups'];assert len(groups)==2 and abs(groups[1]['lr']/groups[0]['lr']-.1)<1e-8
   rows.append(dict(seed=seed,outer=outer,inner=inner,n=len(fa),best_epoch=sa['summary']['best_epoch'],normalization_sha256=sa['summary']['normalization_sha256'],checkpoint_sha256=sa['summary']['checkpoint_sha256'],encoder_lr_ratio=groups[1]['lr']/groups[0]['lr'],outer_test_loader_created=False))
assert len(rows)==45
pd.DataFrame(rows).to_csv(BASE/'analysis/e1_integrity.csv',index=False)
print(json.dumps({'runs_verified':45,'validation_id_label_match':True,'train_only_normalization_match':True,'checkpoint_sha_match':True,'encoder_lr_ratio_exact':True,'bn_running_stats_unchanged_across_runs':True,'e1_original_source_hash_match':True,'outer_test_loader_created':False},indent=2))
