"""Read-only E3 integrity audit after full completion."""
import hashlib,json
from pathlib import Path
import pandas as pd
BASE=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930')
P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
rows=[]
for seed in (42,43,44):
 for outer in range(5):
  for inner in range(3):
   a=BASE/f'runs/multi/p100/seed{seed}/outer_{outer}/inner_{inner}'
   b=P3B/f'runs/b_str_pretrained/seed{seed}/outer_{outer}/inner_{inner}'
   sa=json.loads((a/'stage_status.json').read_text());sb=json.loads((b/'stage_status.json').read_text())
   assert sa['status']=='complete' and sa['outer_test_loader_created'] is False
   assert sa['summary']['train_subject_ids_sha256']==sb['summary']['train_subject_ids_sha256']
   assert sa['summary']['normalization_sha256']==sb['summary']['normalization_sha256']
   fa=pd.read_csv(a/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   fb=pd.read_csv(b/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   assert fa.subject_id.is_unique and fa[['subject_id','target']].equals(fb[['subject_id','target']])
   cp=a/'checkpoints/best.pt'
   assert hashlib.sha256(cp.read_bytes()).hexdigest()==sa['summary']['checkpoint_sha256']
   rows.append(dict(seed=seed,outer=outer,inner=inner,n=len(fa),best_epoch=sa['summary']['best_epoch'],normalization_sha256=sa['summary']['normalization_sha256'],checkpoint_sha256=sa['summary']['checkpoint_sha256'],outer_test_loader_created=False))
assert len(rows)==45
pd.DataFrame(rows).to_csv(BASE/'analysis/e3_integrity.csv',index=False)
print(json.dumps({'runs_verified':45,'validation_id_label_match':True,'train_only_normalization_match':True,'checkpoint_sha_match':True,'outer_test_loader_created':False},indent=2))
