"""Full-matrix seed-first analysis with fixed retention gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, recall_score

HERE = Path(__file__).resolve().parent.parent
OLD = Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
M = ['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']


def metrics(y,p):
    pred=(p>.5).astype(int)
    return dict(accuracy=accuracy_score(y,pred),ba=balanced_accuracy_score(y,pred),auroc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),pd_recall=recall_score(y,pred,pos_label=0),dd_recall=recall_score(y,pred,pos_label=1))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['ema'],default='ema');args=ap.parse_args()
    references=['baseline','a1_mean'] if args.variant=='a2_delta' else ['baseline']
    roots={'baseline':OLD/'runs/b_str_pretrained',args.variant:HERE/'runs'/args.variant}
    if args.variant=='a2_delta':roots['a1_mean']=HERE/'runs/a1_mean'
    raw,agreement,aligned=[],[],{}
    for seed in (42,43,44):
        for oi in range(5):
            for ii in range(3):
                audit=json.loads((HERE/f'runs/ordinary/seed{seed}/outer_{oi}/inner_{ii}/ordinary_trajectory_audit.json').read_text())
                assert audit['ordinary_full_trajectory_exact'] and audit['ordinary_predictions_exact'] and not audit['smoke']
    for name,root in roots.items():
        for oi in range(5):
            for ii in range(3):
                probs=[]
                for seed in (42,43,44):
                    stage=root/f'seed{seed}/outer_{oi}/inner_{ii}'
                    status=json.loads((stage/'stage_status.json').read_text())
                    assert status['status']=='complete' and status['outer_test_loader_created'] is False and status['summary']['smoke'] is False
                    frame=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').reset_index(drop=True)
                    assert frame.subject_id.is_unique
                    key=(oi,ii,seed)
                    idlabels=frame[['subject_id','target']]
                    if key in aligned:assert idlabels.equals(aligned[key])
                    else:aligned[key]=idlabels
                    p=frame.probability_dd.to_numpy(float);y=frame.target.to_numpy(int)
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    probs.append(p)
                    summary=status['summary']
                    hist=[json.loads(l) for l in (stage/'logs/epochs.jsonl').read_text().splitlines()]
                    best=hist[summary['best_epoch']-1]
                    if name!='baseline':
                        old=json.loads((roots['baseline']/f'seed{seed}/outer_{oi}/inner_{ii}/stage_status.json').read_text())['summary']
                        if args.variant=='b1_raw':
                            assert summary['normalization_sha256']!=old['normalization_sha256']
                            an=json.loads((stage/'normalization.json').read_text());bn=json.loads((roots['baseline']/f'seed{seed}/outer_{oi}/inner_{ii}/normalization.json').read_text())
                            for k in ('mean','std'):assert np.array_equal(np.asarray(an[k])[:,:,3:],np.asarray(bn[k])[:,:,3:])
                        else:
                            assert summary['normalization_sha256']==old['normalization_sha256']
                        assert summary['train_subject_ids_sha256']==old['train_subject_ids_sha256']
                    raw.append(dict(variant=name,context=oi,inner=ii,seed=seed,n=len(y),best_epoch=summary['best_epoch'],epochs_run=len(hist),train_loss=best['train']['loss'],validation_loss=best['validation']['loss'],**metrics(y,p)))
                probs=np.stack(probs)
                agreement.append(dict(variant=name,context=oi,inner=ii,prediction_agreement=np.mean(np.all((probs>.5)==(probs[0]>.5),axis=0)),mean_pair_agreement=np.mean([np.mean((probs[a]>.5)==(probs[b]>.5)) for a,b in ((0,1),(0,2),(1,2))]),mean_score_spearman=np.mean([spearmanr(probs[a],probs[b]).statistic for a,b in ((0,1),(0,2),(1,2))])))
    raw=pd.DataFrame(raw); fold=raw.groupby(['variant','context','inner'],as_index=False)[M].mean()
    summary=[]
    for name,g in fold.groupby('variant'):
        r=raw[raw.variant==name];seedmeans=r.groupby('seed')[M].mean();within=r.groupby(['context','inner'])[M].std(ddof=1).mean()
        summary.append(dict(variant=name,**{m:g[m].mean() for m in M},**{m+'_split_sd':g[m].std(ddof=1) for m in M},**{m+'_seed_sd':seedmeans[m].std(ddof=1) for m in M},**{m+'_within_split_seed_sd':within[m] for m in M},selected_epoch_median=r.best_epoch.median(),epochs_run_median=r.epochs_run.median(),ordinary_selection_train_loss=r.train_loss.mean(),ordinary_selection_validation_loss=r.validation_loss.mean()))
    summary=pd.DataFrame(summary)
    rng=np.random.default_rng(20261002); pairs=[]
    for ref in references:
        for m in M:
            a=fold[fold.variant==args.variant].set_index(['context','inner'])[m].sort_index()
            b=fold[fold.variant==ref].set_index(['context','inner'])[m].sort_index()
            assert len(a)==len(b)==15 and a.index.equals(b.index)
            d=(a-b).to_numpy();ci=np.quantile(d[rng.integers(0,15,size=(10000,15))].mean(axis=1),[.025,.975]);sd=d.std(ddof=1)
            pairs.append(dict(candidate=args.variant,reference=ref,metric=m,mean_delta=d.mean(),improved_splits=int((d>0).sum()),worse_splits=int((d<0).sum()),ci95_low=ci[0],ci95_high=ci[1],cohen_dz=d.mean()/sd if sd else 0,wilcoxon_p=wilcoxon(d).pvalue if np.any(d) else 1))
    pairs=pd.DataFrame(pairs);pairs['bh_q_primary']=np.nan
    primary=pairs.metric.isin(['ba','auroc']);pv=pairs.loc[primary,'wilcoxon_p'].to_numpy();idx=np.argsort(pv);q=np.minimum.accumulate((pv[idx]*len(pv)/(np.arange(len(pv))+1))[::-1])[::-1];restored=np.empty_like(q);restored[idx]=np.minimum(q,1);pairs.loc[primary,'bh_q_primary']=restored
    s=summary.set_index('variant');b=s.loc['baseline'];c=s.loc[args.variant]
    pb=pairs[pairs.reference=='baseline'].set_index('metric')
    gate=dict(ba_mean_positive=bool(pb.loc['ba','mean_delta']>0),ba_improved_10_of_15=bool(pb.loc['ba','improved_splits']>=10),ba_ci_low_positive=bool(pb.loc['ba','ci95_low']>0),auroc_mean_positive=bool(pb.loc['auroc','mean_delta']>0),auroc_improved_10_of_15=bool(pb.loc['auroc','improved_splits']>=10),macro_f1_non_decrease=bool(c.macro_f1>=b.macro_f1),pd_recall_drop_at_most_001=bool(c.pd_recall>=b.pd_recall-.01),dd_recall_drop_at_most_001=bool(c.dd_recall>=b.dd_recall-.01))
    for m in ('ba','auroc'):gate[m+'_within_split_seed_sd_at_most_125x']=bool(c[m+'_within_split_seed_sd']<=1.25*max(b[m+'_within_split_seed_sd'],.001))
    retained=all(gate.values());decision='RETAIN' if retained else 'REJECT'
    params=143172
    result=dict(ema_loss_history_note='EMA epoch log copy is ordinary history for fixed epoch provenance, not EMA loss trajectory',variant=args.variant,status='complete',required_runs=45,completed_runs=45,statistical_units=15,overlapping_split_inference='repeated-development descriptive robustness only',gate=gate,decision=decision,parameters=params,additional_parameters=params-143172)
    if args.variant=='a2_delta':
        pa=pairs[pairs.reference=='a1_mean'].set_index('metric')
        result['dynamic_specific_mean_advantage']=bool(pa.loc['ba','mean_delta']>0 and pa.loc['auroc','mean_delta']>0)
    out=HERE/'analysis';prefix=args.variant
    raw.to_csv(out/f'{prefix}_seed_split_metrics.csv',index=False)
    fold.to_csv(out/f'{prefix}_15split_seedfirst.csv',index=False)
    summary.to_csv(out/f'{prefix}_summary.csv',index=False)
    pairs.to_csv(out/f'{prefix}_paired.csv',index=False)
    pd.DataFrame(agreement).to_csv(out/f'{prefix}_prediction_agreement.csv',index=False)
    (out/f'{prefix}_decision.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary[['variant']+M].round(6).to_string(index=False));print(pairs.round(6).to_string(index=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
