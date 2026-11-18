"""Phase10 explicit Certified-L1 and subject-role bindings; server execution only."""
from pathlib import Path
import os,sys,json,hashlib,math,datetime,time,copy
from collections import defaultdict
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,pandas as pd
ROOT=Path('/home/zyt/deep_final');HERE=ROOT/'artifacts/phase10_task_adaptive_demographic_20261118'
PRIVATE=HERE/'restricted_manifests';MAIN=ROOT/'mainline'
P9=ROOT/'artifacts/l1_precision_rebuild_phase9_level2_20261112'
P6=ROOT/'artifacts/wssl_representation_generalization_phase6_20261111'
BASE='99db920b97b81004327a575796c70476f8381090';VERSION='pads_l1_lambda50_certified_v1'
sys.path[:0]=[str(MAIN),str(P9/'scripts')]
from common_l2 import ACTS,SIDES,METRICS
from metrics_l2 import pred_metrics,paired,bh_planned
from train_l2 import cfg,model_for,ids_ok,external
from data_l2 import norm_binding,configured_records
from common_l2 import splits

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def write(p,d,new=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x' if new else 'w') as f:json.dump(clean(d),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvout(name,rows):
 p=HERE/'analysis'/name;assert not p.exists(),str(p)
 pd.DataFrame(rows).to_csv(p,index=False)
 return p
def report(name,text):
 p=HERE/'reports'/name
 with p.open('x') as f:f.write(text+'\n')
 return p
def table(frame):
 if not isinstance(frame,pd.DataFrame):frame=pd.DataFrame(frame)
 def fmt(x):return 'NA' if pd.isna(x) else f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(map(fmt,r))+' |\n' for r in frame.itertuples(index=False,name=None))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def role(c,i):return next((sp,set(map(str,ex))) for cc,ii,sp,ex in splits() if (cc,ii)==(c,i))
def lineage(kind):return load(P9/'restricted_manifests/manifests'/f'{kind}_lineage.json')['records']
def line(kind,seed,c,i):return next(x for x in lineage(kind) if (x['seed'],x['context'],x['inner'])==(seed,c,i))
def norm(c,i):
 ix=load(P9/'restricted_manifests/normalization'/VERSION/'index.json')
 entry=next(x for x in ix['normalizations'] if (x['context'],x['inner'])==(c,i))
 assert sha(entry['path'])==entry['sha256']
 ready=load(P9/'restricted_manifests/manifests/data_ready.json')
 data,n=norm_binding(VERSION,ready['manifest_path'],entry['path'],c,i)
 return data,n,entry
def certified(kind,seed,c,i):
 r=line(kind,seed,c,i);sp,ex=role(c,i);allowed=set(map(str,sp['validation_subjects']))
 assert sha(r['checkpoint'])==r['checkpoint_sha256'] and sha(r['prediction'])==r['prediction_sha256']
 assert r['manifest_sha256']==load(P9/'restricted_manifests/manifests/data_ready.json')['manifest_sha256']
 assert sha(r['normalization'])==r['normalization_sha256']
 with np.load(r['prediction'],allow_pickle=False) as z:
  ids=z['subject_ids'].astype(str);y=z['target'].astype(int);p=z['probabilities'].copy();logits=z['logits'].copy()
 ids_ok(ids,allowed);assert not set(ids)&ex
 return ids,y,p,logits
def baseline_frame(kind='wssl'):
 rows=[]
 for c,i,sp,ex in splits():
  for seed in [42,43,44]:
   ids,y,p,z=certified(kind,seed,c,i)
   rows.append(dict(condition=kind.upper()+'-CERTIFIED-L1',seed=seed,context=c,inner=i,**pred_metrics(y,p,z)))
 return pd.DataFrame(rows)
def guard():
 for path,digest in load(PRIVATE/'protected_assets.json')['sha256'].items():
  assert sha(path)==digest,'Protected asset changed: '+path
 return len(load(PRIVATE/'protected_assets.json')['sha256'])
def freeze(name,paths,**extra):
 write(HERE/'configs'/f'{name}_FREEZE.json',dict(timestamp_utc=utc(),files={str(Path(p).relative_to(HERE)):sha(p) for p in paths},**extra),new=True)
def checkfreeze(name):
 for p,h in load(HERE/'configs'/f'{name}_FREEZE.json')['files'].items():assert sha(HERE/p)==h
def summary(f):
 rows=[]
 for condition,g in f.groupby('condition'):
  units=g.groupby(['context','inner'])[METRICS].mean();seeds=g[g.seed>=0].groupby('seed')[METRICS].mean()
  row=dict(condition=condition,runs=len(g),splits=len(units),seeds=len(seeds))
  for m in METRICS:
   row[m]=units[m].mean();row[m+'_SD']=units[m].std(ddof=1) if len(units)>1 else seeds[m].std(ddof=1)
   row[m+'_seed_SD']=seeds[m].std(ddof=1) if len(seeds)>1 else np.nan
  rows.append(row)
 return pd.DataFrame(rows)
def wilson(correct,n):
 if n<10:return (np.nan,np.nan)
 z=1.959963984540054;phat=correct/n;den=1+z*z/n
 middle=(phat+z*z/(2*n))/den;half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
 return middle-half,middle+half
def metadata():
 rows={};labelmap={"Parkinson's":0,'Other Movement Disorders':1,'Essential Tremor':1,'Atypical Parkinsonism':1,'Multiple Sclerosis':1}
 # Full JSON parsing is necessary; only this explicit whitelist is retained.
 for p in (ROOT/'data/raw/pads/patients').glob('*.json'):
  d=load(p);sid=str(d['id']);assert d['condition'] in labelmap
  rows[sid]=dict(age=d.get('age'),gender=d.get('gender'),height=d.get('height'),weight=d.get('weight'),
                 condition=d['condition'],label=labelmap[d['condition']],path=str(p))
 assert len(rows)==390
 return rows
def demographic_matrix(meta,ids):
 # Fixed mapping, never learned from validation frequencies. No missingness inputs.
 out=[]
 for sid in ids:
  d=meta[str(sid)];sex={'female':0.,'male':1.}.get(d['gender'],np.nan)
  out.append([d['age'],sex,d['height'],d['weight']])
 return np.asarray(out,float)

