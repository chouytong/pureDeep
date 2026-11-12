"""Train-only resident classifier timings; raw online HarNet explicitly unmeasured."""
from common_l2 import *
from data_l2 import configured_records
from train_l2 import cfg,model_for,external
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
import torch,gc
from src.utils.seed import seed_everything

def fingerprint(model):return {k:__import__('hashlib').sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
def main():
 assert load(HERE/'analysis/analysis_integrity.json')['status']=='PASS_FOUR_CELL_FIXED_ANALYSIS';torch.set_num_threads(4);seed_everything(42,True);freeze('COST',[Path(__file__)],batch_size=8,warmup=3,repeats=10,precision='float32 AMPfalse; audited runtime defaults',train_only=True,new_fits=0)
 lockpath=P6/'analysis/external_validation_lock.json';protected=load(PRIVATE/'manifests/protected_assets.json')['sha256'];assert sha(lockpath)==protected[str(lockpath)];lock=load(lockpath);assert lock['external_encoder_parameter_count']==10457408 and lock['classifier_trainable_parameter_count']==143172
 write(HERE/'audit/parameter_count_provenance.json',dict(external_parameter_count=lock['external_encoder_parameter_count'],external_source=str(lockpath),external_source_sha256=sha(lockpath),official_HarNet10_checkpoint_sha256=lock['official_HarNet10_checkpoint_sha256'],classifier_counts_measured_in_all_90_new_instances=True,external_encoder_reinstantiated=False,HarNet_cache_same_frozen_encoder=True))
 m=load(PRIVATE/'manifests'/f'{VERSION}.json');records_=configured_records(m);normindex=load(load(HERE/'analysis/normalization_integrity.json')['normalization_index']);norms={(r['context'],r['inner']):load(r['path']) for r in normindex['normalizations']};lineages={k:{(r['seed'],r['context'],r['inner']):r for r in load(PRIVATE/'manifests'/f'{k}_lineage.json')['records']} for k in ['wssl','str']};timings=[]
 for c,i,sp,ex in splits():
  fit=set(sp['train_subjects']);r=[v for v in records_ if v.subject_id in fit];n=norms[c,i];ds=SubjectActivityDataset(r,ACTS,cfg(42)['data'],torch.tensor(n['mean']),torch.tensor(n['std']),'validation');batch=collate_subject_activities([ds[j] for j in range(8)]);assert set(batch['subject_id'])<=fit;x=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']];ext=external(set(batch['subject_id']));ssl=torch.from_numpy(np.stack([ext[s] for s in batch['subject_id']])).cuda()
  for seed in [42,43,44]:
   for kind in ['wssl','str']:
    record=lineages[kind][seed,c,i];assert sha(record['checkpoint'])==record['checkpoint_sha256'];model=model_for(kind,cfg(seed)).cuda().eval();ck=torch.load(record['checkpoint'],map_location='cuda',weights_only=False);model.load_state_dict(ck['model_state'])
    if kind=='wssl':model.ssl_features=ssl
    prior=fingerprint(model);torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
     for _ in range(3):model(*x)
     torch.cuda.synchronize();beg=time.perf_counter()
     for _ in range(10):out=model(*x)['logits']
     torch.cuda.synchronize();ms=(time.perf_counter()-beg)*1000/10
    assert torch.isfinite(out).all() and fingerprint(model)==prior;nparams=sum(p.numel() for p in model.parameters());frozen=10457408 if kind=='wssl' else 0;timings.append(dict(model=kind,seed=seed,context=c,inner=i,batch_size=8,warmup=3,repeats=10,classifier_forward_ms=ms,subjects_per_second=8000/ms,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),classifier_parameters=nparams,trainable_parameters=nparams,frozen_external_parameters=frozen,total_deployment_parameters=nparams+frozen,scenario='resident normalized full-length input and frozen HarNet cache; classifier only',end_to_end=False,train_only=True));del model,ck;gc.collect();torch.cuda.empty_cache()
  print(json.dumps(dict(stage='timing',completed_model_timings=len(timings))),flush=True);del ds,batch,x,ext,ssl;gc.collect();torch.cuda.empty_cache()
 pd.DataFrame(timings).to_csv(HERE/'analysis/inference_timings.csv',index=False);wrist=pd.read_csv(PRIVATE/'subject_diagnostics/build_record_costs.csv',dtype={'subject_id':str});axis=np.array([a['cpu_seconds'] for r in m['records'] for a in r['axis_certificates']]);person=wrist.groupby('subject_id').record_CPU_seconds.sum().to_numpy();rows=[]
 for scope,values in [('axis_solver',axis),('wrist_complete_export',wrist.record_CPU_seconds.to_numpy()),('subject_22wrist_complete_export',person)]:
  rows.append(dict(component=scope,unit='CPU_seconds',observations=len(values),mean=values.mean(),median=np.median(values),p95=np.quantile(values,.95),maximum=values.max(),throughput_items_per_CPU_second=1/values.mean(),peak_RSS_KiB=wrist.worker_peak_RSS_KiB.max(),scope_note='measured user+system worker CPU; parent and QA counted separately in ledger'))
 for kind,f in pd.DataFrame(timings).groupby('model'):
  for col in ['classifier_forward_ms','subjects_per_second','peak_GPU_allocated_bytes']:
   v=f[col].to_numpy();rows.append(dict(component=kind+'_'+col,unit=col,observations=len(v),mean=v.mean(),median=np.median(v),p95=np.quantile(v,.95),maximum=v.max(),scope_note='batch8, warmup3, repeat10, synchronized; no IO/L1/normfit/HarNet encoding'))
 ledger=load(PRIVATE/'manifests/cpu_ledger.json');pd.DataFrame([{k:r.get(k) for k in ['stage','attempt','status','exit_code','CPU_seconds','charged_CPU_seconds','wall_seconds']} for r in ledger['attempts']]).to_csv(HERE/'analysis/stage_compute_summary.csv',index=False)
 for stage in ['train-wssl','train-str','fit-normalization']:
  attempts=[r for r in ledger['attempts'] if r['stage']==stage];total=sum(r['CPU_seconds'] for r in attempts);wall=sum(r.get('wall_seconds',0) for r in attempts);rows.append(dict(component=stage+'_total',unit='CPU_seconds_parent_and_children',observations=len(attempts),mean=total,scope_note='all attempts total user+system CPU including workers; cumulative wall_seconds='+str(wall)))
 rows.append(dict(component='solver_QA_pilot_export_and_readbacks_total',unit='CPU_seconds_parent_and_children',observations=1,mean=ledger['total_solver_CPU_seconds'],scope_note='all charged attempts; limit21600 seconds'))
 pd.DataFrame(rows).to_csv(HERE/'analysis/cost_summary.csv',index=False);write(HERE/'analysis/cost_integrity.json',dict(status='PASS_MEASURED_BOUNDARIES',timings=90,batch_size=8,warmup=3,repeats=10,new_fits=0,train_only=True,models_unchanged=True,hardware=torch.cuda.get_device_name(),cached_HarNet=True,AMP=False,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,online_raw_to_prediction='NOT_MEASURED_OUTSIDE_AUTHORIZED_HARNET_REEXTRACTION_SCOPE',CPU_solver_ledger=str(PRIVATE/'manifests/cpu_ledger.json'),GPU_forward_excludes_preprocessing_IO_HarNet=True))
if __name__=='__main__':main()
