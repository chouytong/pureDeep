from pathlib import Path
import hashlib,json,sys,math,time
import numpy as np
import pandas as pd
import torch,yaml
from scipy.special import logsumexp,expit
from scipy.stats import wilcoxon,rankdata
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline';OLD=ROOT/'artifacts/wssl_reference_audit_20261111'
P6=ROOT/'artifacts/wssl_representation_generalization_phase6_20261111';P7=ROOT/'artifacts/independent_validation_phase7_20261111'
HERE=ROOT/'artifacts/wssl_guided_optimization_phase8_20261111';START='35b0a7cb78b28f29385724cf98f4f1dbdde8b2b2'
ACTS=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose']
METRICS=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall','nll','brier','ece'];CLASS_METRICS=METRICS[:6]
SEED=20261112;sys.path.insert(0,str(MAIN))
from src.metrics.classification import classification_metrics
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
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def guard():
 d=load(HERE/'restricted_manifests/protected_assets.json')['sha256']
 for p,h in d.items():assert sha(p)==h,'Protected changed: '+p
 return len(d)
def splits():
 d=load(MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json')
 return [(int(o['outer_fold']),int(i['inner_fold']),i,list(map(str,o['test_subjects']))) for o in d['outer'] for i in o['inner_folds']]
def cfg(seed=42):return yaml.safe_load((MAIN/f'configs/str01_seed{seed}.yaml').read_text())
def baseline(seed,c,i):return ROOT/f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}'
def featurepath(seed,c,i):return OLD/'features'/f'seed{seed}_context{c}_inner{i}.npz'
def ids_ok(ids,allowed,exact=True):
 ids=list(map(str,ids));assert len(ids)==len(set(ids)) and set(ids)<=set(map(str,allowed))
 if exact:assert set(ids)==set(map(str,allowed))
 return ids
def features(seed,c,i):
 sp=next(sp for cc,ii,sp,ex in splits() if (cc,ii)==(c,i));allowed=list(map(str,sp['train_subjects']))+list(map(str,sp['validation_subjects']))
 p=featurepath(seed,c,i);expected=load(HERE/'restricted_manifests/protected_assets.json')['sha256'][str(p)];assert sha(p)==expected
 with np.load(p,allow_pickle=False) as z:
  ids=ids_ok(z['subject_ids'],allowed);val=z['is_validation'].astype(bool);assert {s for s,v in zip(ids,val) if v}==set(map(str,sp['validation_subjects']))
  result={k:z[k].copy() for k in ['target','bilateral','subject','structured','logits','probabilities']}
 result.update(subject_ids=np.asarray(ids),is_validation=val)
 return result
def pred_metrics(y,p,z=None):
 y=np.asarray(y,int);p=np.asarray(p,np.float64);assert p.shape==(len(y),2) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-7,rtol=0)
 decision=(p[:,1]>.5).astype(int);m=classification_metrics(y,decision,2,probabilities=p)
 if z is None:
  with np.errstate(divide='ignore'):loss=-np.log(p[np.arange(len(y)),y])
 else:loss=logsumexp(np.asarray(z,np.float64),1)-np.asarray(z,np.float64)[np.arange(len(y)),y]
 confidence=p.max(1);binid=np.minimum((confidence*10).astype(int),9);ece=sum(np.mean(binid==b)*abs(np.mean(decision[binid==b]==y[binid==b])-confidence[binid==b].mean()) for b in range(10) if np.any(binid==b))
 return dict(accuracy=m['accuracy'],ba=m['balanced_accuracy'],auroc=m['macro_auroc'],macro_f1=m['macro_f1'],pd_recall=m['per_class_recall'][0],dd_recall=m['per_class_recall'][1],nll=float(loss.mean()),brier=float(np.mean(np.sum((p-np.eye(2)[y])**2,1))),ece=ece)
def save_prediction(condition,seed,c,i,ids,y,p,z=None,**extra):
 path=HERE/'predictions'/f'{condition}_seed{seed}_context{c}_inner{i}.npz';assert not path.exists()
 np.savez_compressed(path,subject_ids=np.asarray(ids),target=y,probabilities=p,logits=np.empty((0,2)) if z is None else z,**extra)
 return dict(condition=condition,seed=seed,context=c,inner=i,**pred_metrics(y,p,z))
def bh_planned(p):
 a=np.array(p,float);good=np.isfinite(a);v=np.where(good,a,1.);idx=np.argsort(v);q=np.empty(len(a));q[idx]=np.minimum(1,np.minimum.accumulate((v[idx]*len(a)/np.arange(1,len(a)+1))[::-1])[::-1]);q[~good]=np.nan;return q
def paired(candidate,reference,condition):
 a=candidate.groupby(['context','inner'])[METRICS].mean().sort_index();b=reference.groupby(['context','inner'])[METRICS].mean().sort_index();assert a.index.equals(b.index) and len(a)==15
 rng=np.random.default_rng(SEED);ix=rng.integers(0,15,(10000,15));rows=[]
 for m in METRICS:
  d=(a[m]-b[m]).to_numpy();finite=np.isfinite(d)
  if not finite.all():rows.append(dict(condition=condition,metric=m,status='NA_NONFINITE_OR_MISSING',units=int(finite.sum())));continue
  nz=d[abs(d)>1e-12];r=rankdata(abs(nz));ci=np.quantile(d[ix].mean(1),[.025,.975]);sd=d.std(ddof=1)
  rows.append(dict(condition=condition,metric=m,status='REPEATED_DEVELOPMENT_DESCRIPTIVE',units=15,reference_mean=b[m].mean(),candidate_mean=a[m].mean(),mean_delta=d.mean(),median_delta=np.median(d),wins=int(sum(d>1e-12)),ties=int(sum(abs(d)<=1e-12)),losses=int(sum(d< -1e-12)),ci_low=ci[0],ci_high=ci[1],split_SD=sd,paired_dz=d.mean()/sd if sd else np.nan,rank_biserial=sum(r*np.sign(nz))/sum(r) if len(r) else 0.,wilcoxon_p=wilcoxon(d,zero_method='wilcox').pvalue if len(nz) else 1.))
 return pd.DataFrame(rows)
def gate(metrics,stats,reference,condition,ensemble=False):
 a=metrics.groupby(['context','inner'])[METRICS].mean();b=reference.groupby(['context','inner'])[METRICS].mean();d=a.mean()-b.mean();s=stats.set_index('metric');checks=dict(BA_positive=bool(d.ba>0),BA_wins_at_least10=bool(s.loc['ba','wins']>=10),BA_CI_low_positive=bool(s.loc['ba','ci_low']>0),AUROC_non_decrease=bool(d.auroc>=-1e-12),MacroF1_non_decrease=bool(d.macro_f1>=-1e-12),PD_recall_drop_at_most01=bool(d.pd_recall>=-.01-1e-12),DD_recall_drop_at_most01=bool(d.dd_recall>=-.01-1e-12),correctness_and_provenance=True)
 if ensemble:ratios={m:a[m].std(ddof=1)/b[m].std(ddof=1) for m in ['ba','auroc']};stability='between-split SD diagnostic for fixed ensemble; seedSD not identifiable'
 else:ratios={m:metrics.groupby(['context','inner'])[m].std(ddof=1).mean()/reference.groupby(['context','inner'])[m].std(ddof=1).mean() for m in ['ba','auroc']};stability='mean within-split3seed SD BA+AUROC'
 checks['stability_at_most1_25']=all(np.isfinite(v) and v<=1.25+1e-12 for v in ratios.values())
 return dict(condition=condition,status='RETAIN_DEVELOPMENT_ONLY' if all(checks.values()) else 'REJECT',checks=checks,stability_ratios=ratios,stability_estimand=stability,mean_deltas=d.to_dict(),independent_validation=False)
def table(f):
 def fmt(v):return 'NA' if pd.isna(v) else f'{v:.6f}' if isinstance(v,float) else str(v).replace('|','/')
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(map(fmt,row))+' |\n' for row in f.itertuples(index=False,name=None))
def freeze(name,paths,**extra):
 path=HERE/'configs'/f'{name}_FREEZE.json';assert not path.exists();write(path,dict(files={str(p.relative_to(HERE)):sha(p) for p in paths},**extra))
