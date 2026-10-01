"""Verify inner-only Phase-3A run assets against exact reproduced STR baseline."""
import hashlib,json,yaml
from pathlib import Path
import pandas as pd
B=Path('/home/zyt/deep_final/artifacts/phase3a_puredeep_20260928')
P=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926/runs/baseline')
variants=list(json.loads((B/'manifest.json').read_text())['variants'])
checks={k:0 for k in ['complete','same_train_ids','same_validation_ids','same_validation_labels','same_normalization','train_only_normalization','no_outer_loader','unique_validation_predictions','registered_config']}
issues=[]
for v in variants:
 for seed in (42,43,44):
  for o in range(5):
   for i in range(3):
    p=B/'runs'/v/f'seed{seed}'/f'outer_{o}'/f'inner_{i}';q=P/f'seed{seed}'/f'outer_{o}'/f'inner_{i}';key=f'{v}/seed{seed}/outer_{o}/inner_{i}'
    try:
     a=json.loads((p/'stage_status.json').read_text());b=json.loads((q/'stage_status.json').read_text());sa=json.loads((p/'split.json').read_text());sb=json.loads((q/'split.json').read_text());fa=pd.read_csv(p/'predictions/validation.csv',dtype={'subject_id':str});fb=pd.read_csv(q/'predictions/validation.csv',dtype={'subject_id':str});cfg=yaml.safe_load((p/'config.yaml').read_text())
     cond={'complete':a['status']=='complete','same_train_ids':sa['train_subjects']==sb['train_subjects'],'same_validation_ids':sa['validation_subjects']==sb['validation_subjects'],'same_validation_labels':fa[['subject_id','target']].sort_values('subject_id').reset_index(drop=True).equals(fb[['subject_id','target']].sort_values('subject_id').reset_index(drop=True)),'same_normalization':a['summary']['normalization_sha256']==b['summary']['normalization_sha256'],'train_only_normalization':sa['normalization_fitted_on']=='train_subjects_only','no_outer_loader':a['outer_test_loader_created'] is False,'unique_validation_predictions':fa.subject_id.is_unique and set(fa.subject_id)==set(sa['validation_subjects']),'registered_config':cfg['development']['phase3a_variant']==v}
     for k,value in cond.items():
      checks[k]+=int(value)
      if not value:issues.append([key,k])
    except Exception as e:issues.append([key,str(e)])
hashes=[]
for line in (B/'SOURCE_SHA256SUMS').read_text().splitlines():
 expected,path=line.split(maxsplit=1);actual=hashlib.sha256(Path(path).read_bytes()).hexdigest();hashes.append({'path':path,'match':expected==actual})
result={'expected_units':len(variants)*45,'checks':checks,'source_hashes':hashes,'issues':issues}
(B/'analysis/phase3_integrity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'expected_units':result['expected_units'],'checks':checks,'source_hashes_all_match':all(h['match'] for h in hashes),'issue_count':len(issues)},indent=2))
