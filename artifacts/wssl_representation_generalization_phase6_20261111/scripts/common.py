from pathlib import Path
import hashlib,json,csv,sys,math
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata,wilcoxon,spearmanr,pearsonr
ROOT=Path('/home/zyt/deep_final');P5=ROOT/'artifacts/wssl_complementarity_phase5_20261111';OLD=ROOT/'artifacts/wssl_reference_audit_20261111'
HERE=ROOT/'artifacts/wssl_representation_generalization_phase6_20261111';MAIN=ROOT/'mainline';SEED=20261111
ACTS=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose']
VENDOR=ROOT/'artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables'
OFFICIAL=VENDOR/'model_check_point/mtl_best.mdl'
sys.path.insert(0,str(MAIN))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def write_json(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def guard():
 x=load(HERE/'restricted_manifests/protected_assets.json')
 for p,d in x['sha256'].items():assert sha(p)==d,'Protected file changed: '+p
 return x
def splits():
 x=load(MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json')
 return [(int(o['outer_fold']),int(i['inner_fold']),i,list(map(str,o['test_subjects']))) for o in x['outer'] for i in o['inner_folds']]
def cfg(seed=42):return yaml.safe_load((MAIN/f'configs/str01_seed{seed}.yaml').read_text())
def stage(kind,seed,c,i):
 if kind=='frozen':return ROOT/f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}'
 if kind=='adapted':return ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/runs/adapt/p100/seed{seed}/outer_{c}/inner_{i}'
 raise ValueError(kind)
def check_ids(ids,allowed,exact=False):
 ids=list(map(str,ids));assert len(ids)==len(set(ids)),'Duplicate IDs';assert set(ids)<=set(allowed),'Outside allowlist'
 if exact:assert set(ids)==set(allowed),'Incomplete/wrong IDs'
 return ids
def prediction(path,allowed):
 f=pd.read_csv(path,dtype={'subject_id':str}).set_index('subject_id').sort_index();check_ids(f.index,allowed,True)
 p=f[['probability_pd','probability_dd']].to_numpy();assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6,rtol=0) and np.array_equal(p.argmax(1),f.prediction)
 return f
def canonical_path(value,root):
 p=Path(value)
 if p.is_absolute() and str(p).startswith('/home/zyt/MFAM/data/'):p=ROOT/'data'/p.relative_to('/home/zyt/MFAM/data')
 elif not p.is_absolute():p=Path(root)/p
 p=p.resolve();assert p.is_relative_to(ROOT/'data') and p.is_file();return p
def records(allowed):
 from src.datasets.manifest import ManifestRecord,_parse_label,_optional_int,_optional_float
 config=cfg()['data'];out=[];seen=set()
 for a in ACTS:
  with Path(config['activity_manifests'][a]).open(encoding='utf-8-sig',newline='') as f:
   for row in csv.DictReader(f):
    sid=row['subject_id'].strip()
    if sid not in allowed:continue
    assert row['activity']==a and (sid,a) not in seen;seen.add((sid,a))
    out.append(ManifestRecord(left_path=canonical_path(row['left_path'],config['root']),right_path=canonical_path(row['right_path'],config['root']),subject_id=sid,label=_parse_label(row['label'],['PD','DD']),activity=a,pair_id=row['pair_id'],fold=_optional_int(row.get('fold')),session_index=_optional_int(row.get('session_index')),record_name=row['record_name'],source_condition=row['source_condition'],left_length=_optional_int(row.get('left_length')),right_length=_optional_int(row.get('right_length')),left_sampling_rate=_optional_float(row.get('left_sampling_rate')),right_sampling_rate=_optional_float(row.get('right_sampling_rate')),max_time_offset_seconds=_optional_float(row.get('max_time_offset_seconds'))))
 assert {r.subject_id for r in out}==set(allowed) and len(out)==len(allowed)*11
 return out
def raw_path(record,wrist):return ROOT/'data/raw/pads/movement/timeseries'/f'{record.subject_id}_{record.record_name}_{["LeftWrist","RightWrist"][wrist]}.txt'
def layout():
 # Only ID/order/length metadata, never other subjects' signals or targets.
 keys=[];counts={}
 for a,name in enumerate(ACTS):
  with Path(cfg()['data']['activity_manifests'][name]).open(encoding='utf-8-sig',newline='') as f:
   for row in csv.DictReader(f):
    sid=row['subject_id'];assert row['activity']==name
    for w,col in enumerate(['left_length','right_length']):
     n=1 if int(row[col])<1000 else 2;counts[sid,a,w]=n
     keys.extend((sid,a,w,k) for k in range(n))
 ids=sorted({s for s,a,w in counts});flat=[(s,a,w,k) for s in ids for a in range(11) for w in range(2) for k in range(counts[s,a,w])]
 assert len(ids)==390 and len(keys)==len(flat)==10920
 return keys,flat,counts
def bh(p):
 a=np.asarray(p,float);out=np.full(len(a),np.nan);mask=np.isfinite(a);v=a[mask];idx=np.argsort(v);q=np.empty(len(v));q[idx]=np.minimum(1,np.minimum.accumulate((v[idx]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1]);out[mask]=q;return out
def unit_test(v,name,family):
 v=np.asarray(v,float);ok=np.isfinite(v)
 if len(v)!=15 or not ok.all():return dict(test=name,family=family,status='NA_INCOMPLETE_15_UNITS',units=int(ok.sum()),mean_delta=np.nan,wilcoxon_p=np.nan)
 idx=np.random.default_rng(SEED).integers(0,15,(10000,15));ci=np.quantile(v[idx].mean(1),[.025,.975]);nz=v[abs(v)>1e-12];r=rankdata(abs(nz));sd=v.std(ddof=1)
 return dict(test=name,family=family,status='REPEATED_DEVELOPMENT_DESCRIPTIVE',units=15,mean_delta=v.mean(),median_delta=np.median(v),split_SD=sd,wins=int(sum(v>1e-12)),ties=int(sum(abs(v)<=1e-12)),losses=int(sum(v< -1e-12)),ci_low=ci[0],ci_high=ci[1],dz=v.mean()/sd if sd else np.nan,rank_biserial=sum(r*np.sign(nz))/sum(r) if len(r) else 0.,wilcoxon_p=wilcoxon(v,zero_method='wilcox').pvalue if len(nz) else 1.)
def corr(x,y,method='spearman'):
 x=np.asarray(x);y=np.asarray(y)
 if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
 return float((spearmanr(x,y) if method=='spearman' else pearsonr(x,y)).statistic)
def table(f,digits=6):
 def fmt(x):
  if pd.isna(x):return 'NA'
  if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}'
  return str(x).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(fmt(x) for x in row)+' |\n' for row in f.itertuples(index=False,name=None))
