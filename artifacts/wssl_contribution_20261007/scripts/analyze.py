"""Seed-first descriptive contribution profiles, fixed case rules, private subjects."""
import itertools
from common import *
from scipy.stats import spearmanr


def rho(a,b):
    if np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(spearmanr(a,b).statistic)


def summary_delta(delta, indices):
    d=np.asarray(delta,float);ci=np.quantile(d[indices].mean(1),[.025,.975]);sd=d.std(ddof=1)
    return dict(mean_delta=float(d.mean()),median_delta=float(np.median(d)),positive_split_count=int((d>1e-12).sum()),
                zero_split_count=int((abs(d)<=1e-12).sum()),negative_split_count=int((d< -1e-12).sum()),
                split_sd=float(sd),ci_low=float(ci[0]),ci_high=float(ci[1]),paired_dz=float(d.mean()/sd) if sd>0 else 0.)


def group_name(errors, strict=False):
    rate=np.asarray(errors,float)
    return np.where(rate>=(1 if strict else .75),'stable_error',np.where(rate<=(0 if strict else .25),'stable_correct','unstable'))


def main():
    guard();state=json.loads((HERE/'analysis/execution_state.json').read_text())
    require(state['status']=='complete' and state['completed_checkpoints']==45,'Complete inference matrix required')
    require(not (HERE/'analysis/decision.json').exists(),'Do not overwrite completed analysis')
    runs=pd.read_csv(HERE/'analysis/metrics_45runs.csv',keep_default_na=False)
    require(len(runs)==45*37,'Condition coverage')
    cols=[m+'_'+s for m in METRICS for s in ['full','masked','delta']]
    units=runs.groupby(['condition','kind','activity','wrist','context','inner'],as_index=False)[cols].mean()
    require(len(units)==15*37,'Seed-first unit coverage')
    seed_profile=runs.groupby(['condition','kind','activity','wrist','seed'],as_index=False)[cols].mean()
    rng=np.random.default_rng(20261007);indices=rng.integers(0,15,(10000,15));rows=[]
    for (name,kind,a,w),g in units.groupby(['condition','kind','activity','wrist'],sort=False):
        g=g.sort_values(['context','inner']);require(len(g)==15,'Missing split')
        for m in METRICS:
            raw=runs[runs.condition==name]
            rows.append(dict(condition=name,kind=kind,activity=a,wrist=w,metric=m,
              metric_full=float(g[m+'_full'].mean()),metric_masked=float(g[m+'_masked'].mean()),
              delta_seed_mean_sd=float(seed_profile[seed_profile.condition==name][m+'_delta'].std(ddof=1)),
              mean_within_split_delta_seed_sd=float(raw.groupby(['context','inner'])[m+'_delta'].std(ddof=1).mean()),
              **summary_delta(g[m+'_delta'],indices)))
    summary=pd.DataFrame(rows)
    wide=[]
    for name,kind,a,w,_ in conditions():
        group=summary[summary.condition==name].set_index('metric');ba=group.loc['ba']
        row=dict(condition=name,kind=kind,activity=a,wrist=w)
        row.update({m+'_delta':group.loc[m,'mean_delta'] for m in METRICS})
        row.update({k:ba[k] for k in ['median_delta','positive_split_count','zero_split_count','negative_split_count','split_sd','ci_low','ci_high']})
        wide.append(row)
    wide=pd.DataFrame(wide)
    activity_units=units[units.kind=='activity'].pivot(index=['context','inner'],columns='activity',values='ba_delta').reindex(columns=ACTIVITIES).sort_index()
    matrix=activity_units.to_numpy();profile=matrix.mean(0)
    spread=float(np.ptp(profile));spreads=np.ptp(matrix[indices].mean(1),axis=1);spread_ci=np.quantile(spreads,[.025,.975])
    seed_matrix=seed_profile[seed_profile.kind=='activity'].pivot(index='seed',columns='activity',values='ba_delta').reindex(columns=ACTIVITIES).sort_index()
    seed_correlations=[dict(seed_a=a,seed_b=b,spearman=rho(seed_matrix.loc[a],seed_matrix.loc[b])) for a,b in itertools.combinations([42,43,44],2)]
    loso=[dict(context=int(c),inner=int(i),profile_spearman=rho(matrix[j],np.delete(matrix,j,0).mean(0))) for j,(c,i) in enumerate(activity_units.index)]
    cor=np.array([x['spearman'] for x in seed_correlations]);lr=np.array([x['profile_spearman'] for x in loso]);pos=(matrix>1e-12).sum(0)
    gate=dict(mean_BA_spread_at_least_005=bool(spread>=.005),spread_bootstrap_lower_above_0025=bool(spread_ci[0]>.0025),
       seed_profile_all_positive_median_at_least_05=bool(np.isfinite(cor).all() and (cor>0).all() and np.median(cor)>=.5),
       LOSO_median_at_least_03_and_10_positive=bool(np.isfinite(lr).all() and np.median(lr)>=.3 and (lr>0).sum()>=10),
       at_least_two_activities_positive_10splits=bool((pos>=10).sum()>=2))
    case='A' if all(gate.values()) else ('B' if spread_ci[1]<.005 else 'C')
    action='PROCEED' if case=='A' else ('STOP' if case=='B' else 'INCONCLUSIVE')
    wrist_rows=[]
    left=units[units.condition=='left_off'].set_index(['context','inner']).sort_index()
    right=units[units.condition=='right_off'].set_index(['context','inner']).sort_index()
    for m in METRICS:
        wrist_rows.append(dict(comparison='left_contribution_minus_right_contribution',metric=m,
                              **summary_delta(left[m+'_delta']-right[m+'_delta'],indices)))
    # Strictly aligned formal STR link; individual predictions stay in predictions/.
    manifest=json.loads((HERE/'predictions/manifest.json').read_text())
    require(sha(HERE/'predictions/manifest.json')==state['private_prediction_manifest_sha256'],'Private manifest changed')
    full_rows=[];long_rows=[];gain_rows=[];correlation_rows=[]
    for item in manifest:
        require(sha(item['path'])==item['sha256'],'Private prediction changed')
        seed,c,i=item['seed'],item['context'],item['inner'];f=pd.read_csv(item['path'],dtype={'subject_id':str},keep_default_na=False)
        full=f[f.condition=='full'].set_index('subject_id').sort_index();s=predictions(str_path(seed,c,i)/'predictions/validation.csv')
        require(full.index.equals(s.index) and np.array_equal(full.target,s.target),'STR gain link alignment')
        fm=metrics(full.target,full[['p_pd','p_dd']]);sm=metrics(s.target,s[['probability_pd','probability_dd']])
        gain_rows.append(dict(seed=seed,context=c,inner=i,**{m+'_full':fm[m] for m in METRICS},
                 **{m+'_str':sm[m] for m in METRICS},**{m+'_delta':fm[m]-sm[m] for m in METRICS}))
        full=full.reset_index();full['p_str']=s.probability_dd.to_numpy();full['prediction_str']=s.prediction.to_numpy()
        full_rows.append(full)
        f=f.merge(s[['probability_dd','prediction']].rename(columns={'probability_dd':'p_str','prediction':'prediction_str'}),on='subject_id',validate='many_to_one')
        f['signed_contribution']=(2*f.target-1)*f.delta_probability
        f['absolute_shift']=abs(f.delta_probability)
        f['signed_WSSL_STR_gain']=(2*f.target-1)*(f.p_full-f.p_str)
        long_rows.append(f)
        for name,kind,a,w,_ in conditions():
            if kind!='activity':continue
            g=f[f.condition==name]
            correlation_rows.append(dict(seed=seed,context=c,inner=i,activity=a,
              raw_probability_delta_vs_WSSL_STR_gain_spearman=rho(g.delta_probability,g.p_full-g.p_str),
              signed_delta_vs_signed_gain_spearman=rho(g.signed_contribution,g.signed_WSSL_STR_gain),
              full_masked_probability_spearman=rho(g.p_full,g.p_dd)))
    full=pd.concat(full_rows,ignore_index=True);long=pd.concat(long_rows,ignore_index=True)
    require(len(full)==390*12,'Each development subject must have12seed-context appearances')
    contexts=full.groupby(['subject_id','target','context','inner'],as_index=False)[['p_full','p_str']].mean()
    require(contexts.groupby('subject_id').size().eq(4).all(),'Each subject must have4context appearances')
    contexts['error_str']=(contexts.p_str>.5)!=contexts.target;contexts['error_wssl']=(contexts.p_full>.5)!=contexts.target
    subject=contexts.groupby(['subject_id','target'],as_index=False)[['error_str','error_wssl']].mean()
    for model in ['str','wssl']:
        subject[model+'_primary_group']=group_name(subject['error_'+model])
        subject[model+'_strict4_group']=group_name(subject['error_'+model],strict=True)
        raw_error=full.assign(err=((full['p_str' if model=='str' else 'p_full']>.5)!=full.target)).groupby('subject_id').err.mean()
        subject[model+'_all12_group']=group_name(subject.subject_id.map(raw_error),strict=True)
    expected=subject.str_primary_group.value_counts().to_dict()
    require(expected=={'stable_correct':284,'stable_error':74,'unstable':32},'Historic primary STR groups do not reproduce')
    counts=[]
    for model in ['str','wssl']:
        for definition in ['primary','strict4','all12']:
            for (group,target),g in subject.groupby([model+'_'+definition+'_group','target']):
                counts.append(dict(group_basis=model,definition=definition,group=group,label='PD' if target==0 else 'DD',subject_count=len(g)))
    linked=contexts.merge(subject,on=['subject_id','target'],validate='many_to_one',suffixes=('','_subject'))
    primary_rows=[]
    for (group,target),g in linked.groupby(['str_primary_group','target']):
        primary_rows.append(dict(group_basis='STR_DSG_primary',group=group,label='PD' if target==0 else 'DD',
          subject_count=g.subject_id.nunique(),development_appearance_count=len(g),
          str_error_fraction=g.error_str.mean(),wssl_error_fraction=g.error_wssl.mean(),
          WSSL_corrected_appearances=int((g.error_str&~g.error_wssl).sum()),WSSL_harmed_appearances=int((~g.error_str&g.error_wssl).sum()),
          net_corrected_appearances=int((g.error_str&~g.error_wssl).sum()-(~g.error_str&g.error_wssl).sum()),
          mean_signed_probability_gain=float(((2*g.target-1)*(g.p_full-g.p_str)).mean())))
    # Same group basis, all conditions: unique-subject probability averages and four seed-mean decision appearances.
    long=long.merge(subject.drop(columns=['error_str','error_wssl']),on=['subject_id','target'],validate='many_to_one')
    subject_effect=long.groupby(['subject_id','target','condition','kind','activity','wrist','str_primary_group','wssl_primary_group'],as_index=False)[
             ['delta_probability','signed_contribution','absolute_shift','signed_WSSL_STR_gain']].mean()
    masked_contexts=long.groupby(['subject_id','target','context','inner','condition','str_primary_group','wssl_primary_group'],as_index=False)[['p_full','p_dd']].mean()
    masked_contexts['full_wrong']=(masked_contexts.p_full>.5)!=masked_contexts.target
    masked_contexts['masked_wrong']=(masked_contexts.p_dd>.5)!=masked_contexts.target
    group_rows=[]
    for basis in ['str','wssl']:
        key=basis+'_primary_group'
        for (condition,group,target),g in masked_contexts.groupby(['condition',key,'target']):
            e=subject_effect[(subject_effect.condition==condition)&(subject_effect[key]==group)&(subject_effect.target==target)]
            group_rows.append(dict(group_basis=basis+'_primary',condition=condition,group=group,label='PD' if target==0 else 'DD',
              subject_count=len(e),development_appearance_count=len(g),full_error_fraction=g.full_wrong.mean(),masked_error_fraction=g.masked_wrong.mean(),
              residual_corrected_appearances=int((~g.full_wrong&g.masked_wrong).sum()),residual_harmed_appearances=int((g.full_wrong&~g.masked_wrong).sum()),
              net_residual_corrected_appearances=int((~g.full_wrong&g.masked_wrong).sum()-(g.full_wrong&~g.masked_wrong).sum()),
              mean_raw_DD_probability_delta=e.delta_probability.mean(),mean_signed_true_label_delta=e.signed_contribution.mean(),mean_absolute_probability_shift=e.absolute_shift.mean()))
    concentrations=[]
    for name,kind,a,w,_ in conditions():
        e=subject_effect[subject_effect.condition==name]
        value=e.absolute_shift.to_numpy();top=max(1,int(np.ceil(.1*len(value))));total=value.sum()
        raw=long[long.condition==name];flip=raw.prediction!=raw.prediction_full
        concentrations.append(dict(condition=name,kind=kind,activity=a,wrist=w,unique_subject_count=len(e),
          top10pct_n=top,top10pct_absolute_shift_share=float(np.sort(value)[-top:].sum()/total) if total>0 else 0.,
          median_absolute_subject_shift=float(np.median(value)),mean_absolute_subject_shift=float(value.mean()),
          changed_prediction_fraction=float(flip.mean()),raw_probability_delta_mean=float(raw.delta_probability.mean()),
          raw_probability_delta_sd=float(raw.delta_probability.std(ddof=1)),
          median_full_boundary_distance_flipped=float(abs(raw.loc[flip,'p_full']-.5).median()) if flip.any() else np.nan,
          median_full_boundary_distance_unchanged=float(abs(raw.loc[~flip,'p_full']-.5).median())))
    gains=pd.DataFrame(gain_rows);gain_units=gains.groupby(['context','inner'],as_index=False).mean(numeric_only=True);gain_summary=[]
    for m in METRICS:gain_summary.append(dict(metric=m,metric_full=gain_units[m+'_full'].mean(),metric_str=gain_units[m+'_str'].mean(),**summary_delta(gain_units[m+'_delta'],indices)))
    for name,frame in [('metrics_15split_seedfirst',units),('contribution_summary_long',summary),('activity_contribution',wide[wide.kind=='activity']),
       ('wrist_contribution',wide[wide.kind=='wrist']),('activity_wrist_contribution',wide[wide.kind=='activity_wrist']),('all_ssl_off_anchor',wide[wide.kind=='anchor']),
       ('seed_profiles',seed_profile),('wrist_direct_difference',pd.DataFrame(wrist_rows)),('profile_seed_correlations',pd.DataFrame(seed_correlations)),
       ('profile_LOSO_correlations',pd.DataFrame(loso)),('subject_group_counts',pd.DataFrame(counts)),('subject_group_WSSL_STR_gain',pd.DataFrame(primary_rows)),
       ('subject_group_contribution',pd.DataFrame(group_rows)),('subject_shift_concentration',pd.DataFrame(concentrations)),
       ('STR_WSSL_gain_45runs',gains),('STR_WSSL_gain_15split',gain_units),('STR_WSSL_gain_summary',pd.DataFrame(gain_summary)),
       ('activity_gain_link_descriptive_correlations',pd.DataFrame(correlation_rows))]:frame.to_csv(HERE/'analysis'/f'{name}.csv',index=False)
    subject.to_csv(HERE/'predictions/subject_groups_private.csv',index=False)
    subject_effect.to_csv(HERE/'predictions/subject_effects_private.csv',index=False)
    long.to_csv(HERE/'predictions/all_subject_counterfactual_private.csv',index=False)
    decision=dict(status='complete',case=case,decision=action,phase_B_eligibility=case=='A',phase_B_started=False,phase_C_started=False,
       diagnostic_gate=gate,BA_activity_profile_spread=spread,spread_bootstrap_CI=list(map(float,spread_ci)),
       seed_profile_spearman_median=float(np.nanmedian(cor)),LOSO_profile_spearman_median=float(np.nanmedian(lr)),
       LOSO_profile_positive_count=int((lr>0).sum()),activities_positive_at_least10splits=int((pos>=10).sum()),
       primary_group_definition='DSG/RGD/PRR/PAG: four seed-mean validation appearances, error>=.75 / <=.25; STR-defined primary groups',
       historical_EMA_strict4_definition_is_sensitivity=True,primary_STR_group_counts=expected,
       formal_logits_archived=False,probabilities_reproduced=True,training=False,optimizer_steps=0,outer_access=False,
       threshold_changed=False,activity_or_wrist_selected=False,no_multiple_testing_selection=True,
       diagnostic_CI_not_external_inference=True,causal_WSSL_gain_decomposition_claimed=False,
       interpretation='Counterfactual reliance in a co-adapted fixed model; not additive attribution of independently trained STR-to-WSSL gain',
       protocol_sha256=sha(HERE/'PROTOCOL.md'),diagnostic_lock_sha256=sha(HERE/'analysis/diagnostic_lock.json'))
    guard();write_json(HERE/'analysis/decision.json',decision)
    print(wide[wide.kind=='activity'].to_string(index=False));print(pd.DataFrame(gain_summary).to_string(index=False))
    print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
