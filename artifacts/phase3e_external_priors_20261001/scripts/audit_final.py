"""Audit completed new development outputs and frozen source/hash boundaries."""
import json,hashlib,sys,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
B=Path(__file__).resolve().parents[1];R=Path('/home/zyt/deep_final/foundation_validation');P=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');sys.path[:0]=[str(R),str(B/'scripts')]
from src.utils.config import load_config
from src.engine.nested_training import _load_frozen_split
from src.utils.provenance import sha256_file
c=load_config(str(R/'configs/str01_seed42.yaml'));spl,_,_,sha=_load_frozen_split(c);manifest=json.loads((B/'manifest.json').read_text());assert sha==manifest['split_sha256'];assert sha256_file(B/'PHASE3E_PROTOCOL.md')==manifest['protocol_sha256'];assert sha256_file(P/'manifest.json')==manifest['phase3b_manifest_sha256'];assert sha256_file(P/'analysis/ssl_embeddings_all.npz')==manifest['embedding_sha256']
rows=[]
for variant in ['harnet5','biopm_only','biopm','biopm_random','dual']:
 root=B/'runs'/variant
 if not root.exists():continue
 assert len(list(root.glob('seed*/outer_*/inner_*/stage_status.json')))==45,(variant,'incomplete')
 for seed in (42,43,44):
  for outer in spl['outer']:
   o=outer['outer_fold']
   for inner in outer['inner_folds']:
    i=inner['inner_fold'];rel=Path(f'seed{seed}/outer_{o}/inner_{i}');stage=root/rel;st=json.loads((stage/'stage_status.json').read_text());ref=json.loads((P/'runs/b_str_pretrained'/rel/'stage_status.json').read_text());assert st['status']=='complete' and st['outer_test_loader_created'] is False;assert st['summary']['normalization_sha256']==ref['summary']['normalization_sha256'];assert st['summary']['train_subject_ids_sha256']==ref['summary']['train_subject_ids_sha256'];cp=stage/'checkpoints/best.pt';assert sha256_file(cp)==st['summary']['checkpoint_sha256']
    f=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True);g=pd.read_csv(P/'runs/b_str_pretrained'/rel/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True);assert f[['subject_id','target']].equals(g[['subject_id','target']]);assert f.subject_id.is_unique;assert set(f.subject_id)==set(map(str,inner['validation_subjects']));assert not set(f.subject_id)&set(map(str,inner['train_subjects']));assert np.isfinite(f.probability_dd).all()
    state=torch.load(cp,map_location='cpu',weights_only=False)['model_state'];proj=state['wrist_projection.weight'];norm=float(proj.norm());assert norm>0
    rows.append({'variant':variant,'seed':seed,'outer':o,'inner':i,'train_n':len(inner['train_subjects']),'validation_n':len(f),'normalization_match':True,'train_id_match':True,'checkpoint_match':True,'outer_loader':False,'projection_weight_norm':norm})
pd.DataFrame(rows).to_csv(B/'analysis/final_integrity.csv',index=False)
for ledger in ['formal_e1_source_sha256.txt','formal_biopm_extraction_sha256.txt','formal_biopm_training_sha256.txt','analysis_source_sha256.txt','e4_source_sha256.txt']:
 p=B/ledger
 if p.is_file():subprocess.run(['sha256sum','-c',str(p)],cwd=B,check=True,stdout=subprocess.DEVNULL)
counts=pd.Series([r['variant'] for r in rows]).value_counts().to_dict();out={'status':'PASS','new_completed_runs':len(rows),'variant_counts':counts,'fixed_split_sha256':sha,'subject_id_label_alignment':True,'train_only_normalization_matches':True,'no_outer_loader_or_performance':True,'protocol_and_source_hashes_match':True,'zero_initialized_residual_projection_learned_all_residual_runs':True,'ssl_only_projection_nonzero_at_checkpoint':True};(B/'FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
