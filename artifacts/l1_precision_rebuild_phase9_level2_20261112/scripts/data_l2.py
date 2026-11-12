"""Explicit versioned data/norm binding; no implicit old-data fallback."""
from common_l2 import *
sys.path.insert(0,str(MAIN))
from src.datasets.manifest import ManifestRecord,ManifestTimeSeriesDataset

def data_manifest(version,manifest):
 if version!=VERSION:raise ValueError('DATA_VERSION_MISMATCH')
 ready=load(PRIVATE/'manifests/data_ready.json');m=Path(manifest)
 if sha(m)!=ready['manifest_sha256'] or load(m)['version']!=version:raise ValueError('DATA_MANIFEST_MISMATCH')
 verify=load(HERE/'analysis/data_certificate_readback.json')
 if verify['status']!='PASS_INDEPENDENT_SPARSE_READBACK' or verify['manifest_sha256']!=sha(m):raise ValueError('MISSING_CERTIFIED_DATA_GATE')
 return load(m)
def configured_records(data):
 lookup={(r['subject_id'],r['activity'],r['wrist']):r for r in data['records']};ids=sorted({r['subject_id'] for r in data['records']});result=[]
 for a in ACTS:
  for sid in ids:
   l=lookup[(sid,a,'left')];r=lookup[(sid,a,'right')];assert l['label']==r['label'] and l['output_length']==r['output_length']
   result.append(ManifestRecord(left_path=Path(l['output_path']),right_path=Path(r['output_path']),subject_id=sid,label=0 if l['label']=='PD' else 1,activity=a,pair_id=f'{sid}_{a}_certified',fold=None,record_name=a,source_condition=l['source_condition'],left_length=l['output_length'],right_length=r['output_length'],left_sampling_rate=100,right_sampling_rate=100))
 return result
def norm_binding(version,manifest,normpath,c,i):
 data=data_manifest(version,manifest);norm=load(normpath);sp=next(s for cc,ii,s,e in splits() if (cc,ii)==(c,i))
 if norm.get('version')!=version or norm.get('manifest_sha256')!=sha(manifest) or norm.get('context')!=c or norm.get('inner')!=i or set(norm.get('fit_subject_ids',[]))!=set(sp['train_subjects']):raise ValueError('DATA_NORMALIZATION_VERSION_OR_SPLIT_MISMATCH')
 return data,norm
