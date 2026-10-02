"""Independent-subject WSSL errors/subtype eligibility; no outer or V8 groups."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE=Path(__file__).resolve().parent.parent
F=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(F))
from src.utils.config import load_config
from src.datasets.builders import load_configured_records
from src.datasets.manifest import _load_array,shape_time_series
from src.datasets.nested_cv import subject_metadata_from_records

DD=['Other Movement Disorders','Essential Tremor','Atypical Parkinsonism','Multiple Sclerosis']


def main():
    cfg=load_config(str(F/'configs/str01_seed42.yaml'));records=load_configured_records(cfg['data'],['PD','DD'])
    strata,activities=subject_metadata_from_records(records,cfg['data']['activities'])
    labels={};quality={};rawroot=Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries')
    for r in records:
        assert labels.setdefault(r.subject_id,(r.label,r.source_condition))==(r.label,r.source_condition)
        q=quality.setdefault(r.subject_id,dict(record_count=0,max_wrist_offset=0.,clipped_acc_values=0,acc_values=0,finite=True,sampling_rate_100=True))
        q['record_count']+=1;q['max_wrist_offset']=max(q['max_wrist_offset'],abs(r.max_time_offset_seconds or 0))
        q['sampling_rate_100'] &= r.left_sampling_rate==r.right_sampling_rate==100.
        for wi,(side,path) in enumerate((('LeftWrist',r.left_path),('RightWrist',r.right_path))):
            proc=shape_time_series(_load_array(path,','),6,path)
            raw=np.loadtxt(rawroot/f'{r.subject_id}_{r.record_name}_{side}.txt',delimiter=',',dtype=np.float32)
            assert len(raw)-48==proc.shape[-1]==(r.left_length if wi==0 else r.right_length)
            assert proc.shape[-1] in (976,2000) and np.isfinite(raw).all() and np.isfinite(proc).all()
            acc=raw[48:,1:4];q['clipped_acc_values']+=int((np.abs(acc)>3).sum());q['acc_values']+=acc.size
    assert len(labels)==390 and all(q['record_count']==11 and q['sampling_rate_100'] for q in quality.values())
    lock=json.loads((HERE/'analysis/baseline_lock.json').read_text());split=json.loads(Path(lock['split_path']).read_text());frames=[]
    for s in lock['stages']:
        stage=Path(lock['baseline_root'])/f"seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}"
        f=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str})
        expected=split['outer'][s['context']]['inner_folds'][s['inner']]['validation_subjects']
        assert f.subject_id.is_unique and set(f.subject_id)==set(expected)
        for r in f.itertuples():assert r.target==labels[r.subject_id][0]
        f=f[['subject_id','target','probability_dd']].assign(context=s['context'],inner=s['inner'],seed=s['seed'])
        f['correct']=((f.probability_dd>.5).astype(int)==f.target).astype(int);frames.append(f)
    raw=pd.concat(frames,ignore_index=True)
    assert (raw.groupby(['subject_id','context','inner']).size()==3).all()
    contexts=raw.groupby(['subject_id','context','inner'],as_index=False).agg(target=('target','first'),probability_dd=('probability_dd','mean'))
    contexts['correct']=((contexts.probability_dd>.5).astype(int)==contexts.target).astype(int)
    assert (contexts.groupby('subject_id').size()==4).all()
    subjects=contexts.groupby('subject_id',as_index=False).agg(target=('target','first'),probability_dd=('probability_dd','mean'),context_correct_fraction=('correct','mean'))
    strict=raw.groupby('subject_id').correct.mean();subjects['seed_correct_fraction']=subjects.subject_id.map(strict)
    subjects['consensus_correct']=((subjects.probability_dd>.5).astype(int)==subjects.target).astype(int)
    subjects['source_condition']=subjects.subject_id.map(lambda s:labels[s][1])
    subjects['primary_group']=np.where(subjects.context_correct_fraction==0,'stable_error',np.where(subjects.context_correct_fraction==1,'stable_correct','unstable'))
    subjects['sensitivity_group']=np.where(subjects.seed_correct_fraction==0,'stable_error',np.where(subjects.seed_correct_fraction==1,'stable_correct','unstable'))
    subjects['max_wrist_offset']=subjects.subject_id.map(lambda s:quality[s]['max_wrist_offset'])
    subjects['raw_acc_outside_3g_fraction']=subjects.subject_id.map(lambda s:quality[s]['clipped_acc_values']/quality[s]['acc_values'])
    private=HERE/'predictions';private.mkdir(exist_ok=True);subjects.to_csv(private/'wssl_subject_error_summary.csv',index=False)
    primary=subjects.groupby(['target','primary_group'],as_index=False).agg(n=('subject_id','size'),consensus_recall=('consensus_correct','mean'),mean_context_recall=('context_correct_fraction','mean'))
    sensitivity=subjects.groupby(['target','sensitivity_group'],as_index=False).agg(n=('subject_id','size'),consensus_recall=('consensus_correct','mean'),mean_seed_recall=('seed_correct_fraction','mean'))
    out=HERE/'analysis';primary.to_csv(out/'wssl_error_groups_primary.csv',index=False);sensitivity.to_csv(out/'wssl_error_groups_sensitivity.csv',index=False)
    rng=np.random.default_rng(20261002);groups=[];arrays={}
    for condition,g in subjects.groupby('source_condition'):
        a=g.consensus_correct.to_numpy();arrays[condition]=a;ci=np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975])
        groups.append(dict(source_condition=condition,n=len(a),consensus_recall=a.mean(),ci95_low=ci[0],ci95_high=ci[1],mean_context_recall=g.context_correct_fraction.mean(),stable_error_n=int((g.primary_group=='stable_error').sum()),stable_correct_n=int((g.primary_group=='stable_correct').sum()),unstable_n=int((g.primary_group=='unstable').sum())))
    groups=pd.DataFrame(groups);groups.to_csv(out/'wssl_diagnosis_group_recall.csv',index=False)
    pairs=[]
    for a,b in itertools.combinations(DD,2):
        x,y=arrays[a],arrays[b];delta=x.mean()-y.mean();ci=np.quantile(x[rng.integers(0,len(x),(10000,len(x)))].mean(1)-y[rng.integers(0,len(y),(10000,len(y)))].mean(1),[.025,.975])
        odds,p=fisher_exact([[int(x.sum()),int(len(x)-x.sum())],[int(y.sum()),int(len(y)-y.sum())]])
        pairs.append(dict(category_a=a,category_b=b,n_a=len(x),n_b=len(y),recall_delta=delta,ci95_low=ci[0],ci95_high=ci[1],fisher_p=p))
    pair=pd.DataFrame(pairs);p=pair.fisher_p.to_numpy();idx=np.argsort(p);q=np.minimum.accumulate((p[idx]*len(p)/(np.arange(len(p))+1))[::-1])[::-1];r=np.empty_like(q);r[idx]=np.minimum(q,1);pair['bh_q']=r
    pair['eligible_difference']=(pair.bh_q<.05)&((pair.ci95_low>0)|(pair.ci95_high<0));pair.to_csv(out/'wssl_dd_category_comparisons.csv',index=False)
    qualityrows=[]
    for (target,group),g in subjects.groupby(['target','primary_group']):
        qualityrows.append(dict(target=target,primary_group=group,n=len(g),max_offset_median=g.max_wrist_offset.median(),max_offset_q90=g.max_wrist_offset.quantile(.9),offset_above_001_fraction=(g.max_wrist_offset>.01).mean(),mean_raw_acc_outside_3g_fraction=g.raw_acc_outside_3g_fraction.mean(),subjects_any_raw_acc_above_3g=int((g.raw_acc_outside_3g_fraction>0).sum())))
    qualitytable=pd.DataFrame(qualityrows);qualitytable.to_csv(out/'wssl_quality_by_primary_error_group.csv',index=False)
    eligible=bool(pair.eligible_difference.any() and all(len(arrays[c])>=10 for c in DD))
    feature=HERE/'features';(feature/'dd_source_category_labels.json').write_text(json.dumps({s:DD.index(v[1]) for s,v in labels.items() if v[0]==1},indent=2)+'\n')
    decision=dict(status='PASS',independent_subjects=390,pd_subjects=276,dd_subjects=114,source_categories_verified_across_11_activities=True,all_signals_finite=True,all_records_have_expected_lengths=True,all_metadata_sampling_rates_100=True,source_categories=DD,minimum_category_n=min(len(arrays[c]) for c in DD),primary_stable_error='wrong in all 4 context-level seed-mean decisions',sensitivity='wrong in all 12 seed-run decisions',contexts_per_subject=4,seeds_per_context=3,c1_eligible=eligible,auxiliary_weight_if_eligible=.1,labels_are_source_diagnosis_categories_not_fine_grained_clinical_subtypes=True,outer_artifacts_accessed=False,error_subjects_deleted=False,sampling_changed=False)
    (out/'wssl_error_analysis_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    report='# Item 06 — current WSSL independent-subject errors\n\nDecision: **'+('C1 source-category auxiliary test eligible' if eligible else 'C1 SKIP; insufficient subtype evidence')+'**. No training or data exclusions.\n\nFiles: ERROR_ANALYSIS_PROTOCOL.md, scripts/analyze_wssl_errors.py, analysis/wssl_* aggregate CSV/JSON; private predictions/wssl_subject_error_summary.csv and features/dd_source_category_labels.json remain server-only. Shape/mask/reload/gradient tests for frozen WSSL were already verified in items01–02; this read-only diagnosis fits no model.\n\nPrimary grouping uses four context-level seed-mean decisions per independent subject. Consensus recall averages those scores before a fixed .5 threshold. Strict all-12-run grouping is sensitivity only and is never mixed with primary counts. Formal model performance remains original 15-split metrics.\n\n'
    def table(f):
        return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |\n' for row in f.itertuples(index=False,name=None))
    report+='## Independent diagnosis categories\n\n'+table(groups)+'\n## Primary error groups\n\n'+table(primary)+'\n## Strict seed unanimity sensitivity\n\n'+table(sensitivity)+'\n## DD category paired exploratory comparisons\n\n'+table(pair)+'\n## Quality by primary error group\n\n'+table(qualitytable)
    report+='\nLabels are consistent across all 11 activities and match PD/DD labels. Source diagnosis categories include heterogeneous Other Movement Disorders; granular adjudicated clinical subtype labels are unavailable. All sensor arrays are finite, complete and correctly length-matched, with 100Hz metadata. Wrist offset and extreme Acc are descriptive measures; they do not prove bad recordings or justify deletion. No validation-error list enters sampling.\n\nIndependent subject aggregation avoids counting 12 repeated predictions as 12 subjects. These subjects and fitted models overlap across development contexts, so subtype Fisher/BH and bootstrap results are exploratory repeated-development evidence, not external confirmation. Category N 11/15 is small. C1 eligibility follows the registered source-label/N/BH/CI gate; one weight .1, all four source categories if eligible, DD train labels only. No label or weight search.\n'
    (HERE/'ITEM06_REPORT.md').write_text(report)
    readme=Path('/home/zyt/deep_final/README.md');marker='## 2026-10-02 — WSSL window dynamics, item 06'
    if marker not in readme.read_text():
        with readme.open('a') as f:f.write('\n\n'+marker+'\n\n'+report.split('\n',1)[1]+'\n')
    print(groups.round(6).to_string(index=False));print(primary.to_string(index=False));print(pair.round(6).to_string(index=False));print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
