from pathlib import Path
import hashlib,json,sys,math
import numpy as np
import pandas as pd
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline'
P6=ROOT/'artifacts/wssl_representation_generalization_phase6_20261111'
P5=ROOT/'artifacts/wssl_complementarity_phase5_20261111'
REF=ROOT/'artifacts/wssl_reference_audit_20261111'
HERE=ROOT/'artifacts/independent_validation_phase7_20261111'
MIRROR=Path('/home/zyt/pureDeep-github-upload')
START='6de6480fce87920a899181fa7c8586c01ce616f9'
ACTS=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose']
CHANNELS=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ']
sys.path.insert(0,str(MAIN))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def write(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def guard():
 x=load(HERE/'restricted_manifests/protected_assets.json')
 for p,d in x['sha256'].items():assert sha(p)==d,'Protected asset changed: '+p
 return len(x['sha256'])
def table(f):
 def v(x):return 'NA' if pd.isna(x) else (f'{x:.6f}' if isinstance(x,float) else str(x).replace('|','/').replace('\n',' '))
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(map(v,row))+' |\n' for row in f.itertuples(index=False,name=None))
