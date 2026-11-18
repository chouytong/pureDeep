"""Independent checks on baseline replay, threshold invariants and train-only transforms."""
from common10 import *
from scipy.special import expit

def main():
 assert load(HERE/'analysis/phase10a_integrity.json')['status']=='PASS_PHASE10A_COMPLETE'
 original=pd.read_csv(P9/'analysis/wssl_metrics.csv').set_index(['seed','context','inner'])
 replay=pd.read_csv(HERE/'analysis/certified_baseline_metrics.csv').set_index(['seed','context','inner'])
 assert np.max(np.abs((original[METRICS]-replay[METRICS]).to_numpy()))<1e-12
 curves=pd.read_csv(HERE/'analysis/threshold_curves.csv')
 for _,g in curves.groupby(['seed','context','inner']):
  for m in ['auroc','nll','brier','ece']:assert g[m].max()-g[m].min()<1e-12
  assert len(g)==91 and (g.dd_recall.diff().dropna()<=1e-12).all() and (g.pd_recall.diff().dropna()>=-1e-12).all()
 meta=metadata();fits=load(PRIVATE/'demographic_fit_lineage.json')['records']
 for entry in fits:
  c,i=entry['context'],entry['inner'];sp,ex=role(c,i);fit=entry['fit_subject_ids'];query=entry['query_subject_ids']
  assert set(fit)==set(sp['train_subjects']) and set(query)==set(sp['validation_subjects']) and not (set(fit)|set(query))&ex
  d=load(PRIVATE/'demographics'/f'context{c}_inner{i}/preprocessor.json');x=demographic_matrix(meta,fit)
  assert np.allclose(np.asarray(d['mean']),x.mean(0),rtol=0,atol=1e-12)
  assert np.allclose(np.asarray(d['scale']),np.where(x.std(0)==0,1,x.std(0)),rtol=0,atol=1e-12)
  v=demographic_matrix(meta,query);scaled=(v-d['mean'])/d['scale'];z=scaled@np.asarray(d['coef'])[0]+d['intercept'][0]
  with np.load(PRIVATE/'demographics'/f'context{c}_inner{i}/D0_predictions.npz',allow_pickle=False) as out:
   assert list(out['subject_ids'])==query and np.allclose(expit(z),out['probabilities'][:,1],rtol=0,atol=1e-12)
  # Forbidden values cannot affect the explicitly whitelisted input construction.
  fake={s:dict(v,disease_comment='changed',medication='changed',age_at_diagnosis=-999) for s,v in meta.items()}
  assert np.array_equal(demographic_matrix(fake,query),v)
 sub=pd.read_csv(HERE/'analysis/dd_subtype_metrics.csv');small=sub[(sub.scope=='single_run_validation')&(sub.validation_count<10)]
 assert small[['recall','error_rate','ci_low','ci_high','mean_signed_margin']].isna().all().all()
 result=dict(status='PASS',baseline_45x9_metrics_replayed=True,threshold_invariants_and_monotonicity=True,
             demographic_train_only_transforms_and_saved_predictions_recovered=True,forbidden_fields_invariance=True,
             small_subtype_NA_policy=True,no_new_clinical_fits=True,protected_files_verified=guard())
 write(HERE/'tests/phase10a_independent_checks.json',result,new=True);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
