from pathlib import Path
import hashlib,json,csv,sys,math
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata,spearmanr,pearsonr

ROOT=Path('/home/zyt/deep_final')
OLD=ROOT/'artifacts/wssl_reference_audit_20261111'
HERE=ROOT/'artifacts/wssl_complementarity_phase5_20261111'
SEED=20261111
METRICS=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
sys.path.insert(0,str(ROOT/'mainline'))
from src.metrics.classification import classification_metrics

def metrics(y,pred,ddscore,probs=None):
    m=classification_metrics(np.asarray(y,int),np.asarray(pred,int),2,probabilities=probs)
    if probs is None:
        from sklearn.metrics import roc_auc_score
        auc=roc_auc_score(y,ddscore)
    else:auc=m['macro_auroc']
    return dict(accuracy=m['accuracy'],ba=m['balanced_accuracy'],auroc=auc,macro_f1=m['macro_f1'],pd_recall=m['per_class_recall'][0],dd_recall=m['per_class_recall'][1])

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def write_json(p,v):Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())

def guard():
    lock=load(HERE/'restricted_manifests/protected_assets_phase5.json')
    for p,digest in lock['sha256'].items():assert sha(p)==digest,'Protected asset changed: '+p
    return lock

def splits():
    obj=load(ROOT/'mainline/splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json')
    out=[]
    for outer in obj['outer']:
      for inner in outer['inner_folds']:
        out.append((int(outer['outer_fold']),int(inner['inner_fold']),inner,list(map(str,outer['test_subjects']))))
    return out

def stage(kind,seed,c,i):
    if kind=='baseline':return ROOT/f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}'
    if kind=='adapted':return ROOT/f'artifacts/phase3c_ssl_adaptation_20260930/runs/adapt/p100/seed{seed}/outer_{c}/inner_{i}'
    raise ValueError(kind)

def require_ids(ids,allowed,expected=None):
    ids=list(map(str,ids));assert len(ids)==len(set(ids)),'Duplicate ID'
    assert set(ids)<=set(map(str,allowed)),'Out-of-allowlist ID'
    if expected is not None:assert set(ids)==set(map(str,expected)),'Missing/wrong query ID'
    return ids

def prediction(path,allowed):
    f=pd.read_csv(path,dtype={'subject_id':str}).set_index('subject_id').sort_index()
    require_ids(f.index,allowed,allowed)
    p=f[['probability_pd','probability_dd']].to_numpy();assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert np.allclose(p.sum(1),1,rtol=0,atol=1e-6)
    assert np.array_equal(p.argmax(1),f.prediction) and set(f.target.unique())=={0,1}
    return f

def fusion_rows(c,i,seed,allowed):
    rows=[]
    with (OLD/'predictions/B_fusion_predictions.csv').open() as f:
      for row in csv.DictReader(f):
        if (int(row['context']),int(row['inner']),int(row['seed']))==(c,i,seed):
            assert row['subject_id'] in allowed
            rows.append(row)
    f=pd.DataFrame(rows).set_index('subject_id').sort_index()
    require_ids(f.index,allowed,allowed)
    for k in ['target','prediction']:f[k]=f[k].astype(int)
    for k in ['probability_pd','probability_dd']:f[k]=f[k].astype(float)
    return f

def feature_scores(seed,c,i,sp):
    p=OLD/'features'/f'seed{seed}_context{c}_inner{i}.npz'
    with np.load(p,allow_pickle=False) as z:
        ids=require_ids(z['subject_ids'],set(sp['train_subjects'])|set(sp['validation_subjects']))
        mask=z['is_validation'].copy()
        assert set(np.asarray(ids)[mask])==set(sp['validation_subjects']) and set(np.asarray(ids)[~mask])==set(sp['train_subjects'])
        y=z['target'].copy();logits=z['logits'].astype(np.float64);prob=z['probabilities'].astype(np.float64)
    assert logits.shape==(len(ids),2) and np.isfinite(logits).all()
    f=pd.DataFrame(dict(subject_id=ids,target=y,is_validation=mask,raw_dd_logit=logits[:,1],wssl_logit=logits[:,1]-logits[:,0],stored_probability_pd=prob[:,0],stored_probability_dd=prob[:,1])).set_index('subject_id')
    return f

def oof_scores(seed,c,i,sp):
    p=OLD/'features'/f'OOF_seed{seed}_context{c}_inner{i}.npz'
    with np.load(p,allow_pickle=False) as z:
        ids=require_ids(z['subject_ids'],sp['train_subjects'],sp['train_subjects']);y=z['target'].copy();s=z['logit'].copy();fold=z['crossfit'].copy()
    assert np.isfinite(s).all() and set(y)=={0,1}
    return pd.DataFrame(dict(subject_id=ids,target=y,wssl_logit=s,crossfit=fold)).set_index('subject_id')

def bh(p):
    a=np.asarray(p,float);out=np.full(len(a),np.nan);keep=np.isfinite(a);a=a[keep]
    idx=np.argsort(a);q=np.empty(len(a));q[idx]=np.minimum(1,np.minimum.accumulate((a[idx]*len(a)/np.arange(1,len(a)+1))[::-1])[::-1]);out[keep]=q
    return out

def test_units(values,name,family):
    d=np.asarray(values,float);d=d[np.isfinite(d)]
    if len(d)!=15:return dict(test=name,family=family,status='BLOCKED_INCOMPLETE_UNITS',units=len(d),mean_delta=np.nan,wilcoxon_p=np.nan)
    idx=np.random.default_rng(SEED).integers(0,15,(10000,15));ci=np.quantile(d[idx].mean(1),[.025,.975]);nonzero=d[abs(d)>1e-12];r=rankdata(abs(nonzero));sd=d.std(ddof=1)
    return dict(test=name,family=family,status='DESCRIPTIVE_REPEATED_DEVELOPMENT',units=15,mean_delta=float(d.mean()),median_delta=float(np.median(d)),split_delta_sd=float(sd),wins=int(sum(d>1e-12)),ties=int(sum(abs(d)<=1e-12)),losses=int(sum(d< -1e-12)),ci_low=ci[0],ci_high=ci[1],cohen_dz=float(d.mean()/sd) if sd else np.nan,rank_biserial=float(sum(r*np.sign(nonzero))/sum(r)) if len(r) else 0.,wilcoxon_p=float(wilcoxon(d,zero_method='wilcox').pvalue) if len(nonzero) else 1.)

def save_tests(rows,name):
    f=pd.DataFrame(rows);f['bh_q']=np.nan
    for family,idx in f.groupby('family').groups.items():f.loc[idx,'bh_q']=bh(f.loc[idx,'wilcoxon_p'])
    f.to_csv(HERE/'analysis'/name,index=False);return f

def descript(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return {k:np.nan for k in ['n','mean','sd','q05','q25','q50','q75','q95','minimum','maximum']}
    q=np.quantile(x,[.05,.25,.5,.75,.95])
    return dict(n=len(x),mean=x.mean(),sd=x.std(ddof=1) if len(x)>1 else np.nan,q05=q[0],q25=q[1],q50=q[2],q75=q[3],q95=q[4],minimum=x.min(),maximum=x.max())

def correlation(x,y,kind='spearman'):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
    return float((spearmanr(x,y) if kind=='spearman' else pearsonr(x,y)).statistic)

def table(f,digits=6):
    def fmt(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.{digits}f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in f.itertuples(index=False,name=None))
