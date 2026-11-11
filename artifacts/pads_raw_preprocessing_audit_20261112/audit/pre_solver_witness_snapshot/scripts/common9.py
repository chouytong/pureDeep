"""Read-only Phase9 diagnostics; never import a training entry or fitted global cache."""
from pathlib import Path
import sys,json,hashlib,math,time
import numpy as np
import pandas as pd
from scipy import signal,stats
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline';P8=ROOT/'artifacts/wssl_guided_optimization_phase8_20261111';HERE=ROOT/'artifacts/pads_raw_preprocessing_audit_20261112'
RAW=ROOT/'data/raw/pads';PROC=ROOT/'data/processed/pads_multi_activity/v2_l1_full_length';START='4745549dabf34ccc89a740ad361a0410c6a93a4d';SEED=20261112
ACTS=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose'];SIDES=['left','right'];CHANNELS=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csv(frame,name,private=False):
 p=HERE/('restricted_manifests/analysis' if private else 'analysis')/name;p.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(p,index=False)
def bh(p):
 a=np.asarray(p,float);valid=np.isfinite(a);v=np.where(valid,a,1.);ix=np.argsort(v);q=np.empty(len(v));q[ix]=np.minimum(1,np.minimum.accumulate((len(v)*v[ix]/np.arange(1,len(v)+1))[::-1])[::-1]);q[~valid]=np.nan;return q
def table(frame):
 def f(x):return 'NA' if pd.isna(x) else f'{x:.6g}' if isinstance(x,float) else str(x).replace('|','/')
 return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join('| '+' | '.join(map(f,row))+' |\n' for row in frame.itertuples(index=False,name=None))
def guard():
 d=load(HERE/'restricted_manifests/protected_assets.json')['sha256']
 for p,h in d.items():assert sha(p)==h,'Protected changed '+p
 return len(d)
def freeze(name,paths,**extra):
 p=HERE/'configs'/f'{name}_FREEZE.json';assert not p.exists();write(p,dict(files={str(p.relative_to(HERE)):sha(p) for p in paths},**extra))
def records():return load(HERE/'restricted_manifests/records.json')['records']
def read_raw(r):
 p=Path(r['raw_path']);assert p.is_relative_to(RAW);a=np.loadtxt(p,delimiter=',',dtype=float)
 if a.ndim!=2 or a.shape[1]!=7:raise ValueError('not time+6 channels')
 return a[:,0],a[:,1:]
def psd(x,fs):
 x=np.asarray(x,float)
 if len(x)<4:return np.array([]),np.array([])
 f,p=signal.welch(x,fs=fs,nperseg=min(256,len(x)),detrend='constant',axis=0,scaling='density');return f,p
def spectral(x,fs):
 f,p=psd(x,fs)
 if len(f)==0:return {k:np.nan for k in ['spectral_energy','dominant_frequency','centroid','bandwidth','spectral_entropy','band_0_3','band_3_12','band_12_15','band_15_50','fraction_3_12','high_to_low_proxy']}
 if p.ndim==2:p=p.sum(1)
 df=f[1]-f[0];total=float(p.sum()*df);den=p.sum();prob=p/den if den>0 else np.zeros_like(p);centroid=float((prob*f).sum());bands={k:float(p[(f>=lo)&(f<hi)].sum()*df) for k,lo,hi in [('band_0_3',0,3),('band_3_12',3,12),('band_12_15',12,15),('band_15_50',15,fs/2+1e-9)]}
 return dict(spectral_energy=total,dominant_frequency=float(f[1+np.argmax(p[1:])]) if len(p)>1 and den>0 else np.nan,centroid=centroid,bandwidth=float(np.sqrt((prob*(f-centroid)**2).sum())),spectral_entropy=float(-np.sum(prob[prob>0]*np.log(prob[prob>0]))/np.log(len(prob))),**bands,fraction_3_12=bands['band_3_12']/total if total>0 else np.nan,high_to_low_proxy=(bands['band_0_3']+bands['band_3_12'])/bands['band_15_50'] if bands['band_15_50']>0 else np.nan)
