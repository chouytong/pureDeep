"""Fold-local label-free raw Acc windows for conditional E4; no outer raw files opened."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import resample_poly
ROOT=Path('/home/zyt/deep_final/foundation_validation')
BASE=Path('/home/zyt/deep_final/artifacts/phase3d_ssl_adaptation_20261001')
DATA=Path('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests')
RAW=Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries')
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config
from src.engine.nested_training import _load_frozen_split

def prepare(outer,inner):
 dest=BASE/'domain_raw'/f'outer_{outer}'/f'inner_{inner}';auditfile=dest/'audit.json'
 if auditfile.is_file():
  audit=json.loads(auditfile.read_text());assert audit['outer']==outer and audit['inner']==inner and audit['outer_raw_subjects_opened']==[]
  for role in ('train','validation'):
   assert hashlib.sha256((dest/f'{role}.npz').read_bytes()).hexdigest()==audit[f'{role}_sha256']
  return dest
 c=load_config(str(ROOT/'configs/str01_seed42.yaml'));split,_,_,sha=_load_frozen_split(c)
 assert sha==json.loads((BASE/'manifest.json').read_text())['split_sha256']
 fold=split['outer'][outer]['inner_folds'][inner]
 train=sorted(map(str,fold['train_subjects']));val=sorted(map(str,fold['validation_subjects']));allowed=set(train)|set(val)
 assert not set(train)&set(val)
 acts=[str(a) for a in c['data']['activities']]
 entries={'train':{'x':[],'key':[]},'validation':{'x':[],'key':[]}}
 opened=set();windowsha=hashlib.sha256()
 for ai,activity in enumerate(acts):
  table=pd.read_csv(DATA/f'{activity}.csv',dtype={'subject_id':str})
  table=table[table.subject_id.isin(allowed)]
  assert len(table)==len(allowed) and table.subject_id.is_unique
  for row in table.itertuples():
   role='train' if row.subject_id in train else 'validation'
   ids=train if role=='train' else val;si=ids.index(row.subject_id)
   for wi,wn in enumerate(('LeftWrist','RightWrist')):
    path=RAW/f'{row.subject_id}_{row.record_name}_{wn}.txt'
    raw=np.loadtxt(path,delimiter=',',dtype=np.float32)
    opened.add(str(row.subject_id))
    acc=raw[48:,1:4];expected=row.left_length if wi==0 else row.right_length
    assert raw.shape[1]==7 and len(acc)==expected and np.isfinite(acc).all()
    acc=np.clip(acc,-3,3)
    if len(acc)<1000:
     before=(1000-len(acc))//2;after=1000-len(acc)-before
     segments=[np.pad(acc,((before,after),(0,0)),mode='edge')]
    else:segments=[acc[:1000],acc[-1000:]]
    for seg in segments:
     x=resample_poly(seg,3,10,axis=0).T.astype(np.float32)
     assert x.shape==(3,300)
     entries[role]['x'].append(x);entries[role]['key'].append((si,ai,wi));windowsha.update(x.tobytes())
 assert opened==allowed
 dest.mkdir(parents=True,exist_ok=True)
 audit={'outer':outer,'inner':inner,'train_subject_ids':train,'validation_subject_ids':val,'raw_subjects_opened':sorted(opened),'outer_raw_subjects_opened':[],'activities':acts,'window_sha256':windowsha.hexdigest(),'preprocessing':'Phase3B: raw Acc XYZ in g, 48-sample trim, clip ±3g, first/last 1000 or edge-pad, resample_poly 3/10 to [3,300]'}
 for role,ids in (('train',train),('validation',val)):
  x=np.stack(entries[role]['x']);key=np.asarray(entries[role]['key'],dtype=np.int16)
  np.savez_compressed(dest/f'{role}.npz',x=x,key=key,subject_ids=np.asarray(ids),activities=np.asarray(acts))
  audit[f'{role}_windows']=int(len(x));audit[f'{role}_sha256']=hashlib.sha256((dest/f'{role}.npz').read_bytes()).hexdigest()
 auditfile.write_text(json.dumps(audit,indent=2)+'\n')
 return dest

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--outer',type=int,required=True);ap.add_argument('--inner',type=int,required=True);args=ap.parse_args()
 print(prepare(args.outer,args.inner))
