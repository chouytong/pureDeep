"""Independent numerical replay and closed provenance before final publication."""
from common10 import *
import csv,subprocess,torch
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,recall_score,roc_auc_score

def independent(y,p,z):
 p=p.astype(float);z=z.astype(float);d=(p[:,1]>.5).astype(int);rec=recall_score(y,d,labels=[0,1],average=None,zero_division=0)
 confidence=p.max(1);idx=np.minimum(np.floor(confidence*10).astype(int),9);correct=(d==y).astype(float);ece=0.
 for b in range(10):
  sel=idx==b
  if sel.any():ece+=abs(np.sum(correct[sel]-confidence[sel]))/len(y)
 return dict(accuracy=accuracy_score(y,d),ba=balanced_accuracy_score(y,d),auroc=(roc_auc_score(y==0,p[:,0])+roc_auc_score(y==1,p[:,1]))/2,
  macro_f1=f1_score(y,d,average='macro',zero_division=0),pd_recall=rec[0],dd_recall=rec[1],nll=np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]),
  brier=np.mean((p[:,0]-(y==0))**2+(p[:,1]-(y==1))**2),ece=ece)

def main():
 import train10
 for phase in ['A','B','C','D']:checkfreeze('PHASE10'+phase)
 rows=[];maxerr=0.;optimizer_steps=0;inventory=[]
 for kind,stage in [('FT-L1','B'),('DEMO','C'),('DD','D')]:
  f=pd.read_csv(HERE/'analysis'/f'{kind}_pilot_metrics.csv');assert len(f)==3
  gate=load(HERE/'analysis'/f'{kind}_pilot_gate.json');assert not gate['expand'] and gate['status']=='PILOT_REJECT_STOP'
  for r in f.itertuples():
   seed,c,i=int(r.seed),int(r.context),int(r.inner);directory=PRIVATE/'runs'/kind/f'seed{seed}_context{c}_inner{i}'
   done=load(directory/'complete.json');started=load(directory/'started.json');sp,excluded=role(c,i)
   assert set(done['actual_fit_subject_ids'])==set(map(str,sp['train_subjects']))
   assert set(done['actual_query_subject_ids'])==set(map(str,sp['validation_subjects']))
   assert set(started['excluded_subject_ids'])==excluded
   assert not set(done['actual_fit_subject_ids'])&set(done['actual_query_subject_ids'])
   assert not (set(done['actual_fit_subject_ids'])|set(done['actual_query_subject_ids']))&excluded
   for path,h in done['outputs'].items():assert sha(path)==h
   assert done['source_sha256']==sha(HERE/'scripts/train10.py') and done['models_source_sha256']==sha(HERE/'scripts/models10.py')
   cfgpath=HERE/'configs'/f'{kind}_seed{seed}_context{c}_inner{i}.json';cfg_=load(cfgpath)
   assert sha(cfgpath)==done['config_sha256']
   with np.load(directory/'predictions.npz',allow_pickle=False) as z:
    ids=z['subject_ids'].astype(str);y=z['target'];p=z['probabilities'];logit=z['logits']
   ids_ok(ids,set(done['actual_query_subject_ids']));assert np.array_equal(y,np.array([metadata()[s]['label'] for s in ids]))
   independent_metrics=independent(y,p,logit)
   for m in METRICS:
    err=abs(independent_metrics[m]-getattr(r,m));maxerr=max(maxerr,err);assert err<1e-12,(kind,seed,m,err)
   ndata,norm_,entry=norm(c,i);ck=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
   assert ck['version']==VERSION and ck['normalization_sha256']==entry['sha256'] and ck['config_sha256']==sha(cfgpath)
   assert ck['manifest_sha256']==load(P9/'restricted_manifests/manifests/data_ready.json')['manifest_sha256']
   assert ck['source_sha256']==sha(HERE/'scripts/train10.py') and ck['source_commit']==cfg_['source_commit']==done['source_commit']
   assert ck['scheduler_position']=='after_completed_epoch_step' and ck['epoch']==done['cost']['best_epoch']
   assert all(k in ck for k in ['optimizer_state','scheduler_state','torch_rng_state','cuda_rng_state','train_loader_rng_state'])
   assert ck['optimizer_state']['state'] and ck['scheduler_state']['last_epoch']==ck['epoch']
   for state in ck['optimizer_state']['state'].values():
    for k,v in state.items():
     if torch.is_tensor(v):assert torch.isfinite(v).all()
   optimizer_steps+=done['cost']['optimizer_steps']
   inventory.append(dict(candidate=kind,seed=seed,context=c,inner=i,source_commit=done['source_commit'],code_sha256=done['source_sha256'],
    models_code_sha256=done['models_source_sha256'],config_sha256=done['config_sha256'],checkpoint_sha256=done['checkpoint_sha256'],
    normalization_sha256=done['normalization_sha256'],best_epoch=ck['epoch'],optimizer_state_verified=True,query_count=len(ids)))
   assert done['frozen_state_unchanged'] and done['optimizer_state_saved']
   rows.append(dict(candidate=kind,seed=seed,metric_replay=True,role_provenance=True,checkpoint_optimizer_scheduler=True,frozen_state=True))
   del ck
 # Existing 90 predictions are re-read with original source/norm/role gates.
 for kind in ['wssl','str']:
  for c,i,sp,ex in splits():
   for seed in [42,43,44]:
    ids,y,p,z=certified(kind,seed,c,i);actual=independent(y,p,z);expected=pred_metrics(y,p,z)
    assert all(abs(actual[m]-expected[m])<1e-12 for m in METRICS)
 cat=ROOT/'conclusions/EXPERIMENT_RESULTS_ALL_20261118.csv';old=ROOT/'conclusions/EXPERIMENT_RESULTS_ALL_20261116.csv'
 with old.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);oldcols=r.fieldnames;oldrows=list(r)
 with cat.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);cols=r.fieldnames;newrows=list(r)
 assert len(oldrows)==257 and len(newrows)==266
 assert [{k:x[k] for k in oldcols} for x in newrows[:257]]==oldrows
 for metric in ['Accuracy','AUROC','BA','PD_Recall','DD_Recall','Macro_F1','NLL','Brier','ECE']:
  assert metric+'_mean' in cols and metric+'_SD' in cols
 states=pd.read_csv(HERE/'analysis/experiment_state.csv');assert not states.retained.any() and int(states.formal_runs.sum())==9
 summary_=pd.read_csv(HERE/'analysis/final_metrics.csv');assert set(summary_.condition)>={'FT-L1','DEMO','DD','WSSL-CERTIFIED-L1','STR-CERTIFIED-L1'}
 for kind in ['FT-L1','DEMO','DD']:
  f=pd.read_csv(HERE/'analysis'/f'{kind}_pilot_metrics.csv');s=summary_[summary_.condition==kind].iloc[0]
  for m in METRICS:assert abs(s[m]-f[m].mean())<1e-12 and abs(s[m+'_SD']-f[m].std(ddof=1))<1e-12
 required_reports=['FINAL_REPORT.md','PHASE10A_DIAGNOSTIC_REPORT.md','PHASE10B_UNFREEZING_REPORT.md','PHASE10C_DEMOGRAPHIC_REPORT.md','PHASE10D_DD_LOSS_REPORT.md',
  'THRESHOLD_DIAGNOSTIC_REPORT.md','STATISTICAL_APPENDIX.md','CORRECTNESS_AND_PROVENANCE.md','COST_AND_DEPLOYMENT_REPORT.md','NEXT_STAGE_RESEARCH_PLAN.md']
 for p in required_reports:assert (HERE/'reports'/p).is_file()
 for p in ['phase10a_independent_checks.json','unfreezing_correctness.json','demographic_correctness.json']:
  test=load(HERE/'tests'/p);assert str(test['status']).startswith('PASS'),(p,test['status'])
 assert load(HERE/'tests/dd_cost_correctness.json')['status']=='PASS'
 assert load(HERE/'tests/inference_benchmark_integrity.json')['status']=='PASS_TIMING_ONLY'
 devices=pd.read_csv(HERE/'analysis/demographics_by_device.csv');assert int(devices[devices.field!='gender'].n.min())>=10
 csvout('run_provenance_aggregate.csv',inventory)
 # Close all private artifacts, including failed engineering attempts, without
 # copying their paths/IDs/contents into public reports.
 manifest={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in PRIVATE.rglob('*') if p.is_file()}
 write(PRIVATE/'final_private_asset_manifest.json',dict(files=manifest,source_sha256=sha(Path(__file__))),new=True)
 protected=guard();assert sha(ROOT/'conclusions/CURRENT_MODEL.json')=='a30c1262c7e37793050a73bfd1c0ba6f8f68c0cc020a21cda1434a0f5f05a820'
 test=dict(status='PASS',clinical_runs=9,clinical_optimizer_updates=optimizer_steps,existing_certified_predictions_replayed=90,
  new_prediction_metric_max_error=maxerr,tests=rows,protected_files_verified=protected,old_257_catalog_rows_unchanged=True,catalog_rows=266,
  private_files_closed=len(manifest),private_manifest_sha256=sha(PRIVATE/'final_private_asset_manifest.json'),
  all_freezes_unchanged=True,final_report_files_checked=len(required_reports),source_sha256=sha(Path(__file__)),timestamp_utc=utc())
 write(HERE/'tests/final_verification.json',test,new=True)
 write(HERE/'analysis/phase10_integrity.json',dict(status='PASS_PHASE10_COMPLETE_NO_PROMOTION',new_deep_runs=9,demographic_LR_fits=15,
  candidates_expanded=0,FT_L2_runs=0,threshold_training_runs=0,formal_threshold=.5,outer_test_performance_read=False,
  data_version=VERSION,normalization_count=15,protected_files_verified=protected,current_model_pointer_unchanged=True,
  catalog_rows=266,original_catalog_rows_preserved=257,final_verification_sha256=sha(HERE/'tests/final_verification.json'),
  final_aggregation_sha256=sha(HERE/'scripts/final10.py'),no_phase11_started=True,final_remote_readback_receipt_outside_snapshot=True,timestamp_utc=utc()),new=True)
 print(json.dumps({k:v for k,v in test.items() if k!='tests'},indent=2),flush=True)

if __name__=='__main__':main()
