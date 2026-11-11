"""Paired inference feature statistics; no fitting or model selection."""
from common import *
from threadpoolctl import threadpool_limits
import time
LAYERS=['layer4','layer5','projected','final_wrist'];VARIABLE=LAYERS[1:]
MEASURES=['cosine_distance','L2','relative_displacement','frozen_norm','adapted_norm']

def drift(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1);l2=np.linalg.norm(b-a,axis=-1);den=na*nb
 co=np.divide(np.sum(a*b,axis=-1),den,out=np.zeros_like(den),where=den>1e-24);d=1-np.clip(co,-1,1);both=(na<=1e-12)&(nb<=1e-12);d[both]=0.
 same=np.all(a==b,axis=-1);d[same]=0.
 return dict(cosine_distance=d,L2=l2,relative_displacement=l2/(na+1e-12),frozen_norm=na,adapted_norm=nb)

def subject_boot_correlation(x,y,indices):
 if np.ptp(x)==0 or np.ptp(y)==0:return np.full(len(indices),np.nan)
 a=rankdata(x[indices],axis=1);b=rankdata(y[indices],axis=1);a-=a.mean(1)[:,None];b-=b.mean(1)[:,None];den=np.sqrt((a*a).sum(1)*(b*b).sum(1));return np.divide((a*b).sum(1),den,out=np.full(len(indices),np.nan),where=den>0)

def main():
 guard();assert load(HERE/'analysis/phase6a_extraction.json')['status']=='PASS_ALL45';started=time.perf_counter()
 x=np.array([[[[1.,2.],[0.,0.]]]])
 assert all(np.max(abs(drift(x,x)[m]))==0 for m in ['cosine_distance','L2','relative_displacement'])
 assert abs(drift(np.ones((3,4)),2*np.ones((3,4)))['relative_displacement'].mean()-1)<1e-12
 rows=[];spatial=[];assoc=[];people=[];groupstats=[]
 for c,i,sp,excluded in splits():
  allowed=set(sp['validation_subjects']);l4=HERE/'features'/f'layer4_context{c}_inner{i}.npz'
  with np.load(l4,allow_pickle=False) as z:ids4=check_ids(z['subject_ids'],allowed,True);shared=z['layer4_mean'].copy()
  for seed in [42,43,44]:
   path=HERE/'features'/f'paired_seed{seed}_context{c}_inner{i}.npz'
   with np.load(path,allow_pickle=False) as z:
    ids=check_ids(z['subject_ids'],allowed,True);assert ids==ids4 and z['activities'].tolist()==ACTS and z['wrist_order'].tolist()==['left','right']
    y=z['target'].copy();frozenlog=z['frozen_logits'].astype(np.float64);adaptlog=z['adapted_logits'].astype(np.float64);fp=z['frozen_probabilities'].astype(np.float64);ap=z['adapted_probabilities'].astype(np.float64);mask=z['wrist_mask'].astype(bool)&z['activity_mask'].astype(bool)[...,None]
    fpr=frozenlog.argmax(1);apr=adaptlog.argmax(1);event=np.where((fpr!=y)&(apr==y),'improved',np.where((fpr==y)&(apr!=y),'worsened','unchanged'))
    fm=(2*y-1)*np.diff(frozenlog,axis=1).ravel();am=(2*y-1)*np.diff(adaptlog,axis=1).ravel();dm=am-fm;flip=(fpr!=apr).astype(float);dp=ap[np.arange(len(y)),y]-fp[np.arange(len(y)),y]
    packed={};unit_subject=[]
    for layer in LAYERS:
     a=shared if layer=='layer4' else z['frozen_'+layer];b=shared if layer=='layer4' else z['adapted_'+layer]
     assert a.shape==b.shape and a.shape[:3]==(len(ids),11,2);d=drift(a,b);assert all(np.isfinite(v).all() for v in d.values());packed.update({layer+'_'+k:v for k,v in d.items()})
     means={k:(v*mask).sum((1,2))/mask.sum((1,2)) for k,v in d.items()}
     f=pd.DataFrame(dict(subject_id=ids,target=y,event=event,context=c,inner=i,seed=seed,layer=layer,frozen_prediction=fpr,adapted_prediction=apr,frozen_signed_margin=fm,adapted_signed_margin=am,margin_change=dm,prediction_flip=flip,true_label_probability_change=dp,**means))
     people.append(f);unit_subject.append(f)
     for label in [-1,0,1]:
      keep=np.ones(len(y),bool) if label==-1 else y==label;lab='all' if label==-1 else ('PD' if label==0 else 'DD')
      for group in ['all','improved','worsened','unchanged']:
       m=keep if group=='all' else keep&(event==group);n=int(sum(m));row=dict(context=c,inner=i,seed=seed,layer=layer,dimension=a.shape[-1],label=lab,event=group,subject_count=n,zero_frozen_norm_records=int(sum((d['frozen_norm'][m]<=1e-12).ravel())),zero_adapted_norm_records=int(sum((d['adapted_norm'][m]<=1e-12).ravel())))
       for name,v in means.items():row[name]=v[m].mean() if n else np.nan
       row['mean_margin_change']=dm[m].mean() if n else np.nan;row['flip_rate']=flip[m].mean() if n else np.nan;row['true_label_probability_change']=dp[m].mean() if n else np.nan;rows.append(row)
      for aa,name in enumerate(ACTS):
       for ww in range(2):
        m=keep&mask[:,aa,ww];spatial.append(dict(context=c,inner=i,seed=seed,layer=layer,label=lab,activity=name,wrist=['left','right'][ww],subject_count=int(sum(m)),**{key:float(value[m,aa,ww].mean()) if m.any() else np.nan for key,value in d.items()}))
      for outcome,vector in [('margin_change',dm),('prediction_flip',flip)]:
       assoc.append(dict(context=c,inner=i,seed=seed,layer=layer,label=lab,outcome=outcome,subject_count=int(sum(keep)),spearman=corr(means['relative_displacement'][keep],vector[keep]),pearson=corr(means['relative_displacement'][keep],vector[keep],'pearson'),layer4_constant=(layer=='layer4')))
     base=f[['target','event','relative_displacement']]
     improve=base[base.event=='improved'].relative_displacement;worse=base[base.event=='worsened'].relative_displacement
     groupstats.append(dict(context=c,inner=i,seed=seed,layer=layer,comparison='improved_minus_worsened',difference=improve.mean()-worse.mean() if len(improve) and len(worse) else np.nan))
     groupstats.append(dict(context=c,inner=i,seed=seed,layer=layer,comparison='DD_minus_PD',difference=base[base.target==1].relative_displacement.mean()-base[base.target==0].relative_displacement.mean()))
    metricfile=HERE/'features'/f'drift_seed{seed}_context{c}_inner{i}.npz';assert not metricfile.exists();np.savez_compressed(metricfile,subject_ids=np.asarray(ids),target=y,**packed)
   print(json.dumps(dict(stage='representation_statistics',context=c,inner=i,seed=seed,status='COMPLETE')),flush=True)
 pd.DataFrame(rows).to_csv(HERE/'analysis/representation_drift.csv',index=False);pd.DataFrame(spatial).to_csv(HERE/'analysis/activity_wrist_drift.csv',index=False);af=pd.DataFrame(assoc);af.to_csv(HERE/'analysis/representation_prediction_association.csv',index=False)
 f=pd.concat(people,ignore_index=True);f.to_csv(HERE/'restricted_manifests/subject_drift_private.csv',index=False)
 summary=pd.DataFrame(rows);summary=summary[(summary.event=='all')].groupby(['layer','label','context','inner'])[MEASURES].mean().groupby(['layer','label']).agg(['mean','std']).reset_index();summary.columns=['_'.join([p for p in c if p]) if isinstance(c,tuple) else c for c in summary.columns];summary.to_csv(HERE/'analysis/representation_layer_summary.csv',index=False)
 tests=[]
 for key,t in af[af.layer!='layer4'].groupby(['layer','label','outcome']):
  values=t.groupby(['context','inner']).spearman.agg(lambda v:v.mean() if len(v)==3 and np.isfinite(v).all() else np.nan).reindex(pd.MultiIndex.from_tuples([(c,i) for c,i,_,_ in splits()]))
  row=unit_test(values,'|'.join(key),'A_exploratory');tests.append(row)
  if key[0]=='layer5' and key[2]=='margin_change':tests.append(unit_test(values,'|'.join(key),'A_primary'))
 gs=pd.DataFrame(groupstats)
 for key,t in gs[gs.layer!='layer4'].groupby(['layer','comparison']):
  v=t.groupby(['context','inner']).difference.agg(lambda a:a.mean() if len(a)==3 and np.isfinite(a).all() else np.nan);tests.append(unit_test(v,'|'.join(key),'A_group'))
 tf=pd.DataFrame(tests);tf['bh_q']=np.nan
 for family,ix in tf.groupby('family').groups.items():tf.loc[ix,'bh_q']=bh(tf.loc[ix,'wilcoxon_p'])
 tf.to_csv(HERE/'analysis/representation_paired_tests.csv',index=False)
 # A different, explicitly subject-mean estimand; preserve every person's12 appearances.
 counts=f[f.layer=='layer5'].groupby('subject_id').size();assert len(counts)==390 and (counts==12).all()
 subject=f.groupby(['subject_id','layer']).agg(target=('target','first'),relative_displacement=('relative_displacement','mean'),cosine_distance=('cosine_distance','mean'),margin_change=('margin_change','mean'),prediction_flip=('prediction_flip','mean')).reset_index();subject.to_csv(HERE/'restricted_manifests/subject_mean_drift_private.csv',index=False)
 clusters=[];rng=np.random.default_rng(SEED)
 with threadpool_limits(limits=4):
  for layer in VARIABLE:
   s=subject[subject.layer==layer].sort_values('subject_id');p0=np.flatnonzero(s.target.to_numpy()==0);p1=np.flatnonzero(s.target.to_numpy()==1)
   idx=np.column_stack([rng.choice(p0,(10000,len(p0)),replace=True),rng.choice(p1,(10000,len(p1)),replace=True)])
   for lab,keep,take in [('all',np.ones(len(s),bool),idx),('PD',s.target.to_numpy()==0,idx[:,:len(p0)]),('DD',s.target.to_numpy()==1,idx[:,len(p0):])]:
    for out in ['margin_change','prediction_flip']:
     x=s.relative_displacement.to_numpy();y=s[out].to_numpy();draw=subject_boot_correlation(x,y,take);good=np.isfinite(draw);ci=np.quantile(draw[good],[.025,.975]) if good.any() else [np.nan,np.nan]
     clusters.append(dict(layer=layer,label=lab,outcome=out,subject_count=int(sum(keep)),spearman=corr(x[keep],y[keep]),ci_low=ci[0],ci_high=ci[1],valid_draws=int(sum(good)),estimand='Spearman of per-subject12-appearance/seed scalar means; not mean15split correlation',conditional_on_fitted_models=True))
 pd.DataFrame(clusters).to_csv(HERE/'analysis/representation_subject_cluster.csv',index=False)
 write_json(HERE/'tests/drift_correctness.json',dict(status='PASS',identical_tensor_drift_zero=True,relative_displacement_scale_hand_check=True,layer4_zero_by_shared_frozen_encoder_design=True,subject_count=390,appearances_per_subject_per_layer=12,activities_wrists_not_independent_samples=True))
 guard();write_json(HERE/'analysis/phase6a_integrity.json',dict(status='PASS_ALL45_PAIRED_EXTRACTION_AND_ANALYSIS',pairs=45,subjects=390,layers=LAYERS,raw_global_feature_arrays_decoded=False,layer4_shared_control=True,statistical_family_sizes=tf.groupby('family').size().to_dict(),new_deep_training=0,new_fit=0,checkpoint_weights_unchanged=True,correlation_not_causation=True,elapsed_seconds=time.perf_counter()-started))
 print('REPRESENTATION_DRIFT_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
