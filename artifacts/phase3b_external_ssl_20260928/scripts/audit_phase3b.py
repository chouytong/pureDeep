"""Audit frozen-feature provenance, matched validation IDs, and train-only normalization."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
BASE=Path('/home/zyt/deep_final/artifacts/phase2_str_training_20260926/runs/baseline')
V=['a_ssl_only_pretrained','b_str_pretrained','c_str_random']
cache=np.load(B/'analysis/ssl_embeddings_all.npz',allow_pickle=False)
ids=set(map(str,cache['subject_ids']))
assert len(ids)==390 and cache['pretrained'].shape==cache['random'].shape==(390,11,2,1024)
assert np.isfinite(cache['pretrained']).all() and np.isfinite(cache['random']).all()
manifest=json.loads((B/'manifest.json').read_text())
assert hashlib.sha256((B/'analysis/ssl_embeddings_all.npz').read_bytes()).hexdigest()==manifest['embedding_sha256']
rows=[];corr=[]
for outer in range(5):
 for inner in range(3):
  for seed in (42,43,44):
   base=BASE/f'seed{seed}'/f'outer_{outer}'/f'inner_{inner}'
   bf=pd.read_csv(base/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
   bs=json.loads((base/'stage_status.json').read_text())['summary']
   assert set(bf.subject_id).issubset(ids)
   frames={'baseline':bf}
   for v in V:
    stage=B/'runs'/v/f'seed{seed}'/f'outer_{outer}'/f'inner_{inner}'
    status=json.loads((stage/'stage_status.json').read_text());s=status['summary']
    assert status['status']=='complete' and status['outer_test_loader_created'] is False
    assert s['normalization_sha256']==bs['normalization_sha256']
    assert s['train_subject_ids_sha256']==bs['train_subject_ids_sha256']
    assert (stage/'checkpoints/best.pt').is_file()
    assert hashlib.sha256((stage/'checkpoints/best.pt').read_bytes()).hexdigest()==s['checkpoint_sha256']
    f=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
    assert f[['subject_id','target']].equals(bf[['subject_id','target']])
    assert f.subject_id.is_unique
    frames[v]=f
    rows.append(dict(variant=v,seed=seed,outer=outer,inner=inner,n=len(f),normalization_sha256=s['normalization_sha256'],checkpoint_sha256=s['checkpoint_sha256'],outer_test_loader_created=False))
   for v in V:
    for u in ('baseline',)+tuple(x for x in V if x<v):
     x=frames[v].probability_dd.to_numpy(float);y=frames[u].probability_dd.to_numpy(float)
     corr.append(dict(candidate=v,reference=u,seed=seed,outer=outer,inner=inner,score_spearman=spearmanr(x,y).statistic,threshold_disagreement=np.mean((x>.5)!=(y>.5))))
pd.DataFrame(rows).to_csv(B/'analysis/phase3b_integrity.csv',index=False)
c=pd.DataFrame(corr);c.to_csv(B/'analysis/phase3b_crossmodel_disagreement.csv',index=False)
c.groupby(['candidate','reference'],as_index=False)[['score_spearman','threshold_disagreement']].mean().to_csv(B/'analysis/phase3b_crossmodel_disagreement_summary.csv',index=False)
print(json.dumps({'candidate_runs_verified':len(rows),'matched_baseline_runs':45,'cache_subjects':390,'all_normalization_sha_match':True,'all_subject_ids_labels_match':True,'outer_test_loader_created':False},indent=2))
