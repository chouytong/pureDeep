"""Unchanged Phase8 metric/statistic functions, isolated output namespace."""
from common_l2 import *
from scipy.special import logsumexp
from scipy.stats import wilcoxon,rankdata
sys.path.insert(0,str(MAIN))
from src.metrics.classification import classification_metrics

def pred_metrics(y,p,z=None):
 y=np.asarray(y,int);p=np.asarray(p,np.float64);assert p.shape==(len(y),2) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-7,rtol=0)
 decision=(p[:,1]>.5).astype(int);m=classification_metrics(y,decision,2,probabilities=p)
 if z is None:
  with np.errstate(divide='ignore'):loss=-np.log(p[np.arange(len(y)),y])
 else:loss=logsumexp(np.asarray(z,np.float64),1)-np.asarray(z,np.float64)[np.arange(len(y)),y]
 confidence=p.max(1);binid=np.minimum((confidence*10).astype(int),9);ece=sum(np.mean(binid==b)*abs(np.mean(decision[binid==b]==y[binid==b])-confidence[binid==b].mean()) for b in range(10) if np.any(binid==b))
 return dict(accuracy=m['accuracy'],ba=m['balanced_accuracy'],auroc=m['macro_auroc'],macro_f1=m['macro_f1'],pd_recall=m['per_class_recall'][0],dd_recall=m['per_class_recall'][1],nll=float(loss.mean()),brier=float(np.mean(np.sum((p-np.eye(2)[y])**2,1))),ece=ece)

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
