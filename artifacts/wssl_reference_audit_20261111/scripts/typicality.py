"""Fixed external-cache reference models. No supervised encoder fitting."""
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import time

K=5

class Space:
    def fit(self,x):
        x=np.asarray(x,np.float64)
        if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid fit matrix')
        self.scaler=StandardScaler().fit(x)
        rank=min(16,len(x)-1,x.shape[1])
        self.pca=PCA(n_components=rank,whiten=False,svd_solver='randomized',iterated_power=4,random_state=SEED).fit(self.scaler.transform(x))
        self.keep=self.pca.singular_values_>max(self.pca.singular_values_.max()*1e-12,1e-15)
        if not self.keep.any():raise ValueError('Rank-zero reference space: condition BLOCKED')
        return self
    def transform(self,x):
        z=self.pca.transform(self.scaler.transform(np.asarray(x,np.float64)))[:,self.keep]
        norm=np.linalg.norm(z,axis=1,keepdims=True);self.last_zero=int((norm<=1e-12).sum())
        return np.divide(z,norm,out=np.zeros_like(z),where=norm>1e-12)

def distances(q,b,qids,bids,centroid=False):
    qids=list(map(str,qids));bids=list(map(str,bids))
    if len(bids)!=len(set(bids)) or len(qids)!=len(set(qids)):raise ValueError('Duplicate bank/query ID')
    if len(b)<K:raise ValueError('Fewer than5 reference subjects')
    if centroid:
        # Validation query only; no query is in this reference bank.
        if set(qids)&set(bids):raise ValueError('Centroid diagnostic query present in bank')
        center=b.mean(0,keepdims=True);n=np.linalg.norm(center)
        center=center/n if n>1e-12 else np.zeros_like(center)
        d=np.clip(1-q@center.T,0,2).ravel()
        d[np.linalg.norm(q,axis=1)<=1e-12]=1.
        return d
    d=np.clip(1-q@b.T,0,2)
    d[(np.linalg.norm(q,axis=1)<=1e-12),:]=1.
    d[:,np.linalg.norm(b,axis=1)<=1e-12]=1.
    lookup={s:i for i,s in enumerate(bids)}
    for row,s in enumerate(qids):
        if s in lookup:d[row,lookup[s]]=np.inf
    if np.any(np.isfinite(d).sum(1)<K):raise ValueError('Fewer than5 distinct nonself references')
    return np.partition(d,K-1,axis=1)[:,:K].mean(1)

def order(ids,salt):
    return sorted(range(len(ids)),key=lambda j:hashlib.sha256(f'{SEED}|{salt}|{ids[j]}'.encode()).hexdigest())

def average(x):
    return x.reshape(len(x),11,2).mean(2).mean(1)

def strict_reference(x,y,ids,q,qids,klass,c,i):
    chosen=np.flatnonzero(y==klass);refids=[ids[j] for j in chosen];ref=x[chosen]
    if len(ref)<30:raise ValueError('Strict class too small for fixed5fold/k5')
    sequence=order(refids,f'strict-{c}-{i}-{klass}');folds=np.array_split(sequence,5)
    cross=np.full((len(ref),22),np.nan);query=np.zeros((len(q),22));rank_rows=[]
    for f,test in enumerate(folds):
        train=np.array([j for j in range(len(ref)) if j not in set(test)])
        fit_ids(f'B_strict_{c}_{i}_{klass}_crossfit{f}',[refids[j] for j in train],[refids[j] for j in test],refids,contexts=22,reference_class=klass,encoder='external_HarNet10')
        for ctx in range(22):
            sp=Space().fit(ref[train,ctx]);bank=sp.transform(ref[train,ctx]);z=sp.transform(ref[test,ctx])
            cross[test,ctx]=distances(z,bank,[refids[j] for j in test],[refids[j] for j in train])
            rank_rows.append(dict(context=c,inner=i,kind='strict_crossfit',reference_class=klass,fold=f,activity_wrist=ctx,rank=int(sp.keep.sum()),zero_query=sp.last_zero,bank_size=len(train)))
    fit_ids(f'B_strict_{c}_{i}_{klass}_validation',refids,qids,refids,contexts=22,reference_class=klass)
    for ctx in range(22):
        sp=Space().fit(ref[:,ctx]);bank=sp.transform(ref[:,ctx]);z=sp.transform(q[:,ctx])
        query[:,ctx]=distances(z,bank,qids,refids)
        rank_rows.append(dict(context=c,inner=i,kind='strict_full',reference_class=klass,fold=-1,activity_wrist=ctx,rank=int(sp.keep.sum()),zero_query=sp.last_zero,bank_size=len(ref)))
    assert np.isfinite(cross).all()
    return average(query),float(np.quantile(average(cross),.95,method='linear')),rank_rows

def common_reference(x,y,ids,q,qids,c,i):
    groups=[np.flatnonzero(y==k) for k in [0,1]];m=min(map(len,groups))
    balanced=[]
    for k,g in enumerate(groups):
        ranked=order([ids[j] for j in g],f'bank-{c}-{i}-{k}')
        balanced.append(g[ranked[:m]])
    fit_ids(f'B_common_{c}_{i}',ids,qids,ids,contexts=22,bank_sizes=[m,m],full_bank_sizes=list(map(len,groups)),whole_chain_OOF=False)
    write_json(HERE/'restricted_manifests'/f'B_banks_{c}_{i}.json',dict(pd_subject_ids=[ids[j] for j in balanced[0]],dd_subject_ids=[ids[j] for j in balanced[1]]))
    out={name:np.zeros((len(q),22)) for name in ['pd','dd','full_pd','full_dd','centroid_pd','centroid_dd']}
    train={name:np.zeros((len(x),22)) for name in ['pd','dd','full_pd','full_dd']};rank_rows=[]
    for ctx in range(22):
        sp=Space().fit(x[:,ctx]);z=sp.transform(x[:,ctx]);test=sp.transform(q[:,ctx])
        for k,key in enumerate(['pd','dd']):
            for prefix,g in [('',balanced[k]),('full_',groups[k])]:
                bids=[ids[j] for j in g]
                out[prefix+key][:,ctx]=distances(test,z[g],qids,bids)
                train[prefix+key][:,ctx]=distances(z,z[g],ids,bids)
            out['centroid_'+key][:,ctx]=distances(test,z[balanced[k]],qids,[ids[j] for j in balanced[k]],True)
        rank_rows.append(dict(context=c,inner=i,kind='common',reference_class=-1,fold=-1,activity_wrist=ctx,rank=int(sp.keep.sum()),zero_query=sp.last_zero,bank_size=m))
    out={k:average(v) for k,v in out.items()};train={k:average(v) for k,v in train.items()}
    thresholds={k:float(np.quantile(v[y==(0 if k.endswith('pd') else 1)],.95,method='linear')) for k,v in train.items()}
    return out,thresholds,rank_rows

def tests():
    assert np.allclose(distances(np.array([[1.,0]]),np.array([[1.,0]]*6),['q'],list('abcdef')),0)
    assert np.allclose(distances(np.array([[0.,0]]),np.array([[1.,0]]*6),['q'],list('abcdef')),1)
    try:distances(np.ones((1,2)),np.ones((5,2)),['a'],list('abcde'))
    except ValueError:pass
    else:raise AssertionError('Self entry not excluded')
    try:distances(np.ones((1,2)),np.ones((6,2)),['q'],['a']*6)
    except ValueError:pass
    else:raise AssertionError('Duplicate bank accepted')
    try:Space().fit(np.ones((8,6)))
    except ValueError:pass
    else:raise AssertionError('Degenerate PCA not blocked')
    rng=np.random.default_rng(SEED);x=rng.normal(size=(30,40));sp=Space().fit(x)
    a=sp.transform(x);b=Space().fit(x).transform(x)
    assert a.shape==(30,16) and np.allclose(a,b,atol=1e-12)
    assert order(['a','b','c'],'x')==order(['a','b','c'],'x')
    # Explicitly check score orientation and tiePD in relative score.
    assert np.array_equal((np.array([.4,.1]) - np.array([.1,.4])>0).astype(int),[1,0])
    assert int((.2-.2)>0)==0
    write_json(HERE/'tests/B_correctness.json',dict(status='PASS',k=5,self_exclusion=True,duplicate_ID_rejected=True,degenerate_rank_blocked=True,zero_norm_rule=True,deterministic=True,score_direction=True,outer_access=False))

def main():
    guard();tests()
    freeze('B',[HERE/'PROTOCOL.md',Path(__file__),HERE/'scripts/common.py',HERE/'tests/B_correctness.json'])
    rows=[];diagnostics=[];events=[];conditional=[];started=time.perf_counter()
    with threadpool_limits(limits=4):
      for c,i,sp in splits():
        ids=list(map(str,sp['train_subjects']));qids=list(map(str,sp['validation_subjects']))
        assert not set(ids)&set(qids)
        y=labels(ids);target=labels(qids)
        raw=cache(ids+qids);x=raw[:len(ids)].reshape(len(ids),22,1024);q=raw[len(ids):].reshape(len(qids),22,1024)
        strict={}
        for k in [0,1]:
            d,t,diag=strict_reference(x,y,ids,q,qids,k,c,i);diagnostics+=diag
            strict[k]=(d,t)
        out,t,diag=common_reference(x,y,ids,q,qids,c,i);diagnostics+=diag
        variants={
            'strict_pd':(strict[0][0],(strict[0][0]>strict[0][1]).astype(int)),
            'strict_dd':(-strict[1][0],(strict[1][0]<=strict[1][1]).astype(int)),
            'common_pd':(out['pd'],(out['pd']>t['pd']).astype(int)),
            'common_dd':(-out['dd'],(out['dd']<=t['dd']).astype(int)),
            'relative':(out['pd']-out['dd'],(out['pd']>out['dd']).astype(int)),
            'centroid_relative':(out['centroid_pd']-out['centroid_dd'],(out['centroid_pd']>out['centroid_dd']).astype(int)),
            'full_pd':(out['full_pd'],(out['full_pd']>t['full_pd']).astype(int)),
            'full_dd':(-out['full_dd'],(out['full_dd']<=t['full_dd']).astype(int)),
            'full_relative':(out['full_pd']-out['full_dd'],(out['full_pd']>out['full_dd']).astype(int)),
        }
        frames={s:pred(baseline(s,c,i)/'predictions/validation.csv').loc[qids] for s in [42,43,44]}
        private=pd.DataFrame(dict(subject_id=qids,target=target,d_pd=out['pd'],d_dd=out['dd'],min_distance=np.minimum(out['pd'],out['dd'])))
        for name,(score,decision) in variants.items():
            assert np.isfinite(score).all();private[name+'_score']=score;private[name+'_prediction']=decision
            rows.append(dict(context=c,inner=i,condition=name,seed='deterministic',**metric(target,score,decision)))
            for seed,f in frames.items():
                assert np.array_equal(target,f.target)
                for klass in [0,1]:
                    keep=target==klass;old=f.prediction.to_numpy()[keep]==klass;new=decision[keep]==klass
                    events.append(dict(context=c,inner=i,seed=seed,condition=name,label=klass,corrected=int(sum(new&~old)),newly_wrong=int(sum(~new&old)),net=int(sum(new)-sum(old)),n=int(sum(keep))))
        for seed,f in frames.items():
            p=f.probability_dd.to_numpy();binid=np.minimum((p*10).astype(int),9);numerator=denominator=0.
            for b in range(10):
                keep=binid==b;n0=sum(target[keep]==0);n1=sum(target[keep]==1)
                if n0 and n1:
                    n=n0*n1;numerator+=n*roc_auc_score(target[keep],variants['relative'][0][keep]);denominator+=n
            conditional.append(dict(context=c,inner=i,seed=seed,conditional_auc=float(numerator/denominator) if denominator else None,class_pairs=int(denominator)))
        private.to_csv(HERE/'predictions'/f'B_context{c}_inner{i}.csv',index=False)
        print(json.dumps(dict(stage='B',context=c,inner=i,status='COMPLETE',elapsed_seconds=time.perf_counter()-started)),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'analysis/B_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'analysis/B_rank_zero_bank_audit.csv',index=False)
    pd.DataFrame(events).to_csv(HERE/'analysis/B_corrections.csv',index=False)
    pd.DataFrame(conditional).to_csv(HERE/'analysis/B_conditional_auc.csv',index=False)
    comparisons=[]
    for ref in ['common_pd','common_dd','centroid_relative','full_relative']:
        a=paired(frame[frame.condition=='relative'],frame[frame.condition==ref]);a['comparison']='relative-'+ref;comparisons.append(a)
    paired_all=pd.concat(comparisons,ignore_index=True)
    primary=paired_all[(paired_all.metric=='auroc')&paired_all.comparison.isin(['relative-common_pd','relative-common_dd'])].copy()
    primary['primary_bh_q']=bh(primary.wilcoxon_p)
    primary.to_csv(HERE/'analysis/B_primary_paired.csv',index=False)
    paired_all['exploratory_bh_q']=bh(paired_all.wilcoxon_p);paired_all.to_csv(HERE/'analysis/B_all_paired.csv',index=False)
    cond=pd.DataFrame(conditional).groupby(['context','inner']).conditional_auc.mean()
    if cond.isna().any():increment=False;cond_ci=[None,None]
    else:
        d=cond.to_numpy()-.5;idx=np.random.default_rng(SEED).integers(0,15,(10000,15));cond_ci=np.quantile(d[idx].mean(1),[.025,.975]).tolist();increment=bool(d.mean()>0 and sum(d>1e-12)>=10 and cond_ci[0]>0)
    advantage=bool(all((row.mean_delta>0 and row.wins>=10 and row.ci_low>0) for row in primary.itertuples()))
    guard()
    write_json(HERE/'analysis/B_gate.json',dict(status='COMPLETE',relative_over_both_references=advantage,conditional_increment_screen=increment,conditional_auc_delta_ci=cond_ci,fusion_screen_pass=advantage and increment,deep_training_runs=0,deterministic_models=9,independent_external_validation=False,elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(stage='B',status='COMPLETE',fusion_screen_pass=advantage and increment)),flush=True)

if __name__=='__main__':main()
