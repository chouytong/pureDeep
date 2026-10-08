"""Frozen three-comparison, seed-first15split analysis; no model selection alternatives."""
import itertools
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon,rankdata,spearmanr
from common import *
from run_condition import audit_stage,study_guard
from src.metrics.classification import classification_metrics

METRICS=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
PAIRS=[('P2','P0'),('P2','P1'),('P1','P0')]


def bh(values):
    values=np.asarray(values,float);order=np.argsort(values);q=np.empty(len(values))
    q[order]=np.minimum(1,np.minimum.accumulate((values[order]*len(values)/np.arange(1,len(values)+1))[::-1])[::-1])
    return q


def main():
    torch.set_num_threads(4)
    state=json.loads((HERE/'analysis/execution_state.json').read_text())
    require(state['status']=='complete' and state['completed_runs']==90,'Full matrix not complete; no interim analysis')
    lock=json.loads((HERE/'analysis/f0_lock.json').read_text());study=json.loads((HERE/'analysis/study_lock.json').read_text())
    guard(lock);study_guard(study);split=json.loads(SPLIT.read_text())
    rows=[];frames={};audits=[];error_rows=[];trajectory_rows=[];cross_agreement=[]
    for condition in ['P0','P1','P2']:
        for seed in [42,43,44]:
            for outer in split['outer']:
                c=int(outer['outer_fold'])
                for inner in outer['inner_folds']:
                    i=int(inner['inner_fold'])
                    stage=reference(seed,c,i) if condition=='P0' else HERE/f'runs/{condition}/seed{seed}/outer_{c}/inner_{i}'
                    if condition!='P0':
                        audit=audit_stage(stage,condition,seed,c,i,inner,lock,study,False)
                        require(audit==json.loads((stage/'patch_audit.json').read_text()),'Saved run audit changed')
                        audits.append(dict(condition=condition,seed=seed,context=c,inner=i,status='PASS',
                                           best_epoch=audit['best_epoch'],stop_epoch=audit['stop_epoch']))
                    frame=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id').sort_index()
                    require(set(frame.index)==set(inner['validation_subjects']),'Missing validation subjects')
                    frames[(condition,seed,c,i)]=frame
                    if condition!='P0':
                        ref=frames[('P0',seed,c,i)]
                        require(frame.index.equals(ref.index) and np.array_equal(frame.target,ref.target),'Unmatched predictions')
                    probability=frame[['probability_pd','probability_dd']].to_numpy()
                    metric=classification_metrics(frame.target.to_numpy(),frame.prediction.to_numpy(),2,probabilities=probability)
                    history=[json.loads(l) for l in (stage/'logs/epochs.jsonl').read_text().splitlines()]
                    status=json.loads((stage/'stage_status.json').read_text());best_epoch=status['summary']['best_epoch']
                    best=history[best_epoch-1];last=history[-1]
                    for epoch in history:
                        trajectory_rows.append(dict(condition=condition,seed=seed,context=c,inner=i,epoch=epoch['epoch']+1,
                            selected_best=epoch['epoch']+1==best_epoch,
                            online_train_ce=epoch['train']['classification_loss'],validation_ce=epoch['validation']['classification_loss'],
                            online_train_ba=epoch['train']['balanced_accuracy'],validation_ba=epoch['validation']['balanced_accuracy'],
                            validation_auroc=epoch['validation']['macro_auroc'],validation_f1=epoch['validation']['macro_f1'],
                            validation_pd_recall=epoch['validation']['per_class_recall'][0],validation_dd_recall=epoch['validation']['per_class_recall'][1]))
                    rows.append(dict(condition=condition,seed=seed,context=c,inner=i,subject_count=len(frame),
                        accuracy=metric['accuracy'],ba=metric['balanced_accuracy'],auroc=metric['macro_auroc'],
                        macro_f1=metric['macro_f1'],pd_recall=metric['per_class_recall'][0],dd_recall=metric['per_class_recall'][1],
                        best_epoch=best_epoch,stop_epoch=len(history),
                        training_and_prediction_seconds=json.loads((stage/'runtime.json').read_text())['training_and_prediction_seconds'] if condition!='P0' else np.nan,
                        best_validation_ba=best['validation']['balanced_accuracy'],last_validation_ba=last['validation']['balanced_accuracy'],
                        best_validation_auroc=best['validation']['macro_auroc'],last_validation_auroc=last['validation']['macro_auroc'],
                        best_online_train_ce=best['train']['classification_loss'],best_validation_ce=best['validation']['classification_loss'],
                        last_online_train_ce=last['train']['classification_loss'],last_validation_ce=last['validation']['classification_loss']))
    runs=pd.DataFrame(rows);require(len(runs)==135,'Incomplete P0/P1/P2 metric matrix')
    units=runs.groupby(['condition','context','inner'],as_index=False)[METRICS].mean()
    require(len(units)==45,'Incorrect split aggregation')
    seed_means=runs.groupby(['condition','seed'],as_index=False)[METRICS].mean()
    summary=[]
    for condition in ['P0','P1','P2']:
        for metric in METRICS:
            group=runs[runs.condition==condition];unit=units[units.condition==condition]
            summary.append(dict(condition=condition,metric=metric,mean=unit[metric].mean(),
                 split_sd=unit[metric].std(ddof=1),seed_mean_sd=seed_means[seed_means.condition==condition][metric].std(ddof=1),
                 mean_within_split_seed_sd=group.groupby(['context','inner'])[metric].std(ddof=1).mean()))
    summary=pd.DataFrame(summary)
    rng=np.random.default_rng(20261008);boot_idx=rng.integers(0,15,size=(10000,15))
    comparisons=[]
    for candidate,reference_name in PAIRS:
        a=units[units.condition==candidate].set_index(['context','inner']).sort_index()
        b=units[units.condition==reference_name].set_index(['context','inner']).sort_index()
        require(a.index.equals(b.index) and len(a)==15,'Mismatched comparison units')
        for metric in METRICS:
            delta=(a[metric]-b[metric]).to_numpy();ci=np.quantile(delta[boot_idx].mean(1),[.025,.975])
            nonzero=delta[np.abs(delta)>1e-12]
            p=1. if not len(nonzero) else float(wilcoxon(delta,zero_method='wilcox',alternative='two-sided').pvalue)
            if len(nonzero):
                ranks=rankdata(abs(nonzero));rb=float((ranks[nonzero>0].sum()-ranks[nonzero<0].sum())/ranks.sum())
            else:rb=0.
            sd=delta.std(ddof=1)
            comparisons.append(dict(comparison=f'{candidate}-{reference_name}',metric=metric,
                mean_delta=delta.mean(),improve_count=int((delta>1e-12).sum()),tie_count=int((abs(delta)<=1e-12).sum()),
                worse_count=int((delta< -1e-12).sum()),ci_low=ci[0],ci_high=ci[1],
                paired_dz=float(delta.mean()/sd) if sd>0 else 0.,rank_biserial=rb,wilcoxon_p=p))
    paired=pd.DataFrame(comparisons);paired['bh_q_18']=bh(paired.wilcoxon_p)
    primary=paired.metric=='ba';paired['bh_q_primary_ba3']=np.nan
    paired.loc[primary,'bh_q_primary_ba3']=bh(paired.loc[primary,'wilcoxon_p'])
    def result(comparison,metric):return paired[(paired.comparison==comparison)&(paired.metric==metric)].iloc[0]
    gate={}
    for ref in ['P0','P1']:
        comparison='P2-'+ref;r=result(comparison,'ba')
        gate[comparison]=dict(ba_mean_positive=bool(r.mean_delta>0),ba_improvement_at_least_10=bool(r.improve_count>=10),
            ba_ci_low_positive=bool(r.ci_low>0),ba_primary_bh_pass=bool(r.bh_q_primary_ba3<.05),
            auroc_drop_at_most_005=bool(result(comparison,'auroc').mean_delta>=-.005),
            f1_non_decrease=bool(result(comparison,'macro_f1').mean_delta>=0),
            pd_recall_drop_at_most_01=bool(result(comparison,'pd_recall').mean_delta>=-.01),
            dd_recall_drop_at_most_01=bool(result(comparison,'dd_recall').mean_delta>=-.01))
        for metric in ['ba','auroc']:
            s=summary[(summary.condition=='P2')&(summary.metric==metric)].iloc[0].mean_within_split_seed_sd
            r_sd=summary[(summary.condition==ref)&(summary.metric==metric)].iloc[0].mean_within_split_seed_sd
            gate[comparison][metric+'_within_seed_sd_at_most_125']=bool(s<=1.25*max(r_sd,.001))
    all_pass=all(all(g.values()) for g in gate.values())
    aux_failure=any(not g[k] for g in gate.values() for k in ['auroc_drop_at_most_005','f1_non_decrease',
                            'pd_recall_drop_at_most_01','dd_recall_drop_at_most_01'])
    decision='RETAIN' if all_pass else ('REJECT' if result('P2-P0','ba').mean_delta<=0 or aux_failure else 'INCONCLUSIVE')
    explain=bool(result('P2-P0','ba').mean_delta>0 and result('P2-P1','ba').mean_delta>0 and not aux_failure)
    agreement=[]
    for condition in ['P0','P1','P2']:
        for c in range(5):
            for i in range(3):
                values=[frames[(condition,s,c,i)] for s in [42,43,44]]
                prediction=np.stack([v.prediction.to_numpy() for v in values])
                probability=np.stack([v.probability_dd.to_numpy() for v in values])
                agreement.append(dict(condition=condition,context=c,inner=i,
                    unanimity=float((prediction==prediction[0]).all(0).mean()),
                    mean_pair_agreement=float(np.mean([(prediction[a]==prediction[b]).mean() for a,b in itertools.combinations(range(3),2)])),
                    mean_pair_score_spearman=float(np.mean([spearmanr(probability[a],probability[b]).statistic for a,b in itertools.combinations(range(3),2)]))))
    for candidate,ref in PAIRS:
        for seed in [42,43,44]:
            for c in range(5):
                for i in range(3):
                    a=frames[(candidate,seed,c,i)];b=frames[(ref,seed,c,i)]
                    cross_agreement.append(dict(comparison=candidate+'-'+ref,seed=seed,context=c,inner=i,
                        prediction_agreement=float((a.prediction.to_numpy()==b.prediction.to_numpy()).mean()),
                        score_spearman=float(spearmanr(a.probability_dd,b.probability_dd).statistic)))
                    for target,label in [(0,'PD'),(1,'DD')]:
                        mask=a.target.to_numpy()==target
                        ac=a.prediction.to_numpy()[mask]==target;bc=b.prediction.to_numpy()[mask]==target
                        error_rows.append(dict(comparison=candidate+'-'+ref,seed=seed,context=c,inner=i,label=label,
                            subject_count=int(mask.sum()),reference_errors=int((~bc).sum()),candidate_errors=int((~ac).sum()),
                            corrected=int((ac&~bc).sum()),harmed=int((~ac&bc).sum()),
                            net_corrected=int((ac&~bc).sum()-(~ac&bc).sum())))
    errors=pd.DataFrame(error_rows)
    errors15=errors.groupby(['comparison','context','inner','label'],as_index=False)[
        ['subject_count','reference_errors','candidate_errors','corrected','harmed','net_corrected']].mean()
    for name,frame in [('seed_split_metrics',runs),('epoch_trajectories',pd.DataFrame(trajectory_rows)),
            ('method_agreement_45',pd.DataFrame(cross_agreement)),
            ('method_agreement_15',pd.DataFrame(cross_agreement).groupby(['comparison','context','inner'],as_index=False).mean(numeric_only=True)),('15split_seedfirst',units),('seed_means',seed_means),
            ('summary',summary),('paired_comparisons',paired),('prediction_agreement',pd.DataFrame(agreement)),
            ('run_integrity_90',pd.DataFrame(audits)),('error_counts_90',errors),('error_counts_15split',errors15),
            ('error_summary',errors15.groupby(['comparison','label'],as_index=False).mean(numeric_only=True))]:
        frame.to_csv(HERE/f'analysis/{name}.csv',index=False)
    write_json(HERE/'analysis/decision.json',dict(status='complete',decision=decision,gate=gate,
        full_candidate_runs=90,reused_f0_runs=45,primary_units=15,seeds_averaged_first=True,
        explanation_allowed=explain,
        p1_utilization_allowed=bool(result('P1-P0','ba').mean_delta>0 and result('P1-P0','auroc').mean_delta>=0 and result('P1-P0','macro_f1').mean_delta>=0 and min(result('P1-P0','pd_recall').mean_delta,result('P1-P0','dd_recall').mean_delta)>=-.01),
        p2_order_shuffle_allowed=explain,dd_subtype_analysis_allowed=decision=='RETAIN',p3_allowed=decision=='RETAIN',retained_formal_str_changed=False,outer_access=False,
        threshold_changed=False,recipe_changed=False,posthoc_tuning=False,source_guard_pass=True,
        statistical_boundary='Overlapping fixed development splits, not independent external inference',
        control_boundary='P2 adds321parameters, depthwise order convolution and attention pooling; not isolated order or capacity control',
        protocol_sha256=study['files']['PROTOCOL.md']))
    guard(lock);study_guard(study)
    print(summary[['condition','metric','mean','seed_mean_sd','mean_within_split_seed_sd']].to_string(index=False))
    print(paired.to_string(index=False));print(json.dumps({'decision':decision,'explanation_allowed':explain,'gate':gate}),flush=True)


if __name__=='__main__':main()
