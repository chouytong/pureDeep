from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path('/home/zyt/deep_final')
HERE = ROOT / 'artifacts/wssl_reference_audit_20261111'
MAIN = ROOT / 'mainline'
sys.path.insert(0, str(MAIN))
from src.metrics.classification import classification_metrics
METRICS = ['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
SEED = 20261111

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def guard():
    lock=json.loads((HERE/'configs/protected_assets.json').read_text())
    for path,digest in lock['sha256'].items():
        if sha(path)!=digest:raise AssertionError('Protected asset changed: '+path)
    if 'private_data_ledger' in lock:
        record=lock['private_data_ledger'];assert sha(record['path'])==record['sha256']
        for item in json.loads(Path(record['path']).read_text())['files']:
            assert sha(ROOT/'data'/item['relative_path'])==item['sha256'],'Protected signal/metadata changed'
    return lock

def splits():
    p=MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json'
    value=json.loads(p.read_text())
    return [(int(o['outer_fold']),int(i['inner_fold']),i) for o in value['outer'] for i in o['inner_folds']]

def baseline(seed,c,i):
    return ROOT/f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}'

def pred(path):
    f=pd.read_csv(path,dtype={'subject_id':str}).set_index('subject_id').sort_index()
    assert f.index.is_unique and set(f.target.unique())=={0,1}
    p=f[['probability_pd','probability_dd']].to_numpy()
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    assert np.array_equal(p.argmax(1),f.prediction)
    return f

def metric(y,score,decision=None,prob=None):
    y=np.asarray(y,int);score=np.asarray(score,float)
    decision=(score>.5).astype(int) if decision is None else np.asarray(decision,int)
    m=classification_metrics(y,decision,2,probabilities=prob)
    if prob is None:
        from sklearn.metrics import roc_auc_score
        auc=float(roc_auc_score(y,score))
    else:auc=m['macro_auroc']
    return dict(accuracy=m['accuracy'],ba=m['balanced_accuracy'],auroc=auc,macro_f1=m['macro_f1'],pd_recall=m['per_class_recall'][0],dd_recall=m['per_class_recall'][1])

def labels(ids):
    f=pd.read_csv(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/manifests/CrossArms.csv',dtype={'subject_id':str})
    d=dict(zip(f.subject_id,f.label));return np.array([int(d[str(s)]=='DD') for s in ids])

def cache(ids):
    ids=list(map(str,ids));assert len(ids)==len(set(ids))
    path=ROOT/'data/ssl/harnet10_embeddings.npz'
    with np.load(path,allow_pickle=False) as z:
        all_ids=list(map(str,z['subject_ids']));assert len(all_ids)==len(set(all_ids))==390
        expected=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose']
        assert list(map(str,z['activities']))==expected
        lookup={s:i for i,s in enumerate(all_ids)}
        if not set(ids)<=set(all_ids):raise ValueError('Unknown subject ID')
        take=[lookup[s] for s in ids]
        x=z['pretrained'][take].copy();n=z['window_counts'][take].copy()
    assert x.shape==(len(ids),11,2,1024) and np.isfinite(x).all() and np.all(n>=1)
    return x

def bh(p):
    p=np.asarray(p,float);idx=np.argsort(p);out=np.empty(len(p))
    out[idx]=np.minimum(1,np.minimum.accumulate((p[idx]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);return out

def paired(a,b,metrics=METRICS):
    from scipy.stats import wilcoxon,rankdata
    keys=['context','inner'];a=a.groupby(keys)[metrics].mean().sort_index();b=b.groupby(keys)[metrics].mean().sort_index()
    assert a.index.equals(b.index) and len(a)==15
    idx=np.random.default_rng(SEED).integers(0,15,(10000,15));rows=[]
    for m in metrics:
        d=(a[m]-b[m]).to_numpy();ci=np.quantile(d[idx].mean(1),[.025,.975]);nz=d[abs(d)>1e-12];sd=d.std(ddof=1)
        p=1. if not len(nz) else float(wilcoxon(d,zero_method='wilcox').pvalue)
        ranks=rankdata(abs(nz));rb=float(np.sum(ranks*np.sign(nz))/sum(ranks)) if len(nz) else 0.
        rows.append(dict(metric=m,reference_mean=float(b[m].mean()),candidate_mean=float(a[m].mean()),mean_delta=float(d.mean()),median_delta=float(np.median(d)),wins=int(sum(d>1e-12)),ties=int(sum(abs(d)<=1e-12)),losses=int(sum(d< -1e-12)),ci_low=float(ci[0]),ci_high=float(ci[1]),paired_dz=float(d.mean()/sd) if sd else 0.,rank_biserial=rb,wilcoxon_p=p,split_delta_sd=float(sd)))
    f=pd.DataFrame(rows);f['bh_q']=bh(f.wilcoxon_p);return f

def fit_ids(name,fit,query,allowed,**extra):
    fit=list(map(str,fit));query=list(map(str,query));allowed=set(map(str,allowed))
    assert len(fit)==len(set(fit)) and len(query)==len(set(query))
    assert set(fit)<=allowed and not set(fit)&set(query)
    write_json(HERE/'restricted_manifests'/f'{name}.json',dict(train_subject_ids=fit,query_subject_ids=query,**extra))

def freeze(name,files):
    p=HERE/'configs'/f'{name}_freeze.json';assert not p.exists()
    write_json(p,dict(date='2026-11-11',files={str(Path(f).relative_to(HERE)):sha(f) for f in files},new_validation_performance_seen=False))
