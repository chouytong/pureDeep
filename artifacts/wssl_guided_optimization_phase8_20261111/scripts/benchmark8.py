"""Train-only resident-input timing; never a validation rerun or new fit."""
from common8 import *
from interaction8 import explicit_model,ActivityInteraction,fingerprint
from supcon8 import make_data,inputs
from src.datasets.subject_activity import collate_subject_activities
import gc

def measure(fn):
 with torch.inference_mode():
  for _ in range(3):fn()
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(10):out=fn()
  torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000/10
 assert torch.isfinite(out).all();return elapsed

def main():
 assert load(HERE/'analysis/comparison_integrity.json')['status']=='PASS_COMPLETE_D_COMPARISON';guard();torch.set_num_threads(4)
 freeze('TIMING',[Path(__file__)],new_fits=0,train_only=True,batch_size=8,warmup=3,repeats=10,scope='resident GPU input; cached frozen HarNet external; no preprocessing/dataIO/HarNet encoding')
 rows=[]
 for c,i,sp,ex in splits():
  for seed in [42,43,44]:
   ds,ext,old,paths=make_data(seed,c,i,sp,ex);ds[0].activity_dropout=0.;ds[0].rotation_degrees=0.;batch=collate_subject_activities([ds[0][j] for j in range(8)]);ids,x,ssl,y=inputs(batch,set(sp['train_subjects']),ext)
   base=explicit_model(cfg(seed)).cuda().eval();base.load_state_dict(old['model_state']);prior=fingerprint(base)
   f=features(seed,c,i);ix=[list(f['subject_ids']).index(s) for s in ids];h=torch.from_numpy(f['bilateral'][ix]).cuda();mask=torch.ones(h.shape[:2],dtype=torch.bool,device='cuda')
   def head():
    b=base.backbone;return b.classifier(b.activity_aggregator(h,mask)['subject_embedding'])+b.structured_token_residual(h,mask)['structured_residual_logits']
   rawms=measure(lambda:base(*x,ssl_features=ssl)['logits']);headms=measure(head)
   ai=ActivityInteraction(explicit_model(cfg(seed))).cuda();ai.load_state_dict(torch.load(HERE/'checkpoints'/f'ai_seed{seed}_context{c}_inner{i}/best.pt',map_location='cuda',weights_only=False)['model_state']);ai.eval();aiprior=fingerprint(ai);aims=measure(lambda:ai(h,mask)['logits'])
   sup=explicit_model(cfg(seed)).cuda();sup.load_state_dict(torch.load(HERE/'checkpoints'/f'supcon_seed{seed}_context{c}_inner{i}/best.pt',map_location='cuda',weights_only=False)['model_state']);sup.eval();supprior=fingerprint(sup);supms=measure(lambda:sup(*x,ssl_features=ssl)['logits'])
   assert fingerprint(base)==prior and fingerprint(ai)==aiprior and fingerprint(sup)==supprior
   rows.append(dict(context=c,inner=i,seed=seed,batch_size=8,baseline_resident_classifier_ms=rawms,baseline_cached_bilateral_head_ms=headms,AI_cached_bilateral_head_ms=aims,SupCon_resident_classifier_ms=supms,AI_composed_classifier_estimate_ms=rawms-headms+aims,train_only=True,preprocessing_IO_HarNet_time_included=False,composition_is_measured_end_to_end=False,baseline_classifier_parameters=sum(p.numel() for p in base.parameters()),AI_classifier_parameters=sum(p.numel() for p in ai.parameters()),AI_new_trainable_parameters=sum(p.numel() for p in ai.parameters() if p.requires_grad),HarNet_shared_frozen_parameters=10457408))
   print(json.dumps(dict(stage='timing',completed=len(rows))),flush=True);del ds,ext,base,ai,sup,x,ssl,h,batch;gc.collect();torch.cuda.empty_cache()
 frame=pd.DataFrame(rows);frame.to_csv(HERE/'analysis/inference_timing.csv',index=False)
 ensemble=frame.groupby(['context','inner']).agg(three_seed_classifier_sum_ms=('baseline_resident_classifier_ms','sum')).reset_index();ensemble['batch_size']=8;ensemble['shared_HarNet_parameters']=10457408;ensemble['three_classifiers_parameters']=3*143172;ensemble['deployment_unique_parameters']=10457408+3*143172;ensemble['latency_is_sequential_sum_estimate']=True;ensemble['HarNet_and_IO_excluded']=True;ensemble.to_csv(HERE/'analysis/ensemble_cost_estimate.csv',index=False)
 write(HERE/'analysis/timing_integrity.json',dict(status='PASS',models=45,model_states_unchanged=True,train_only=True,new_fits=0,warmup=3,repeats=10,batch_size=8,hardware=torch.cuda.get_device_name(),torch_threads=4,all_inputs_resident=True,HarNet_encoder_preprocessing_IO_excluded=True,AI_composed_estimate_not_end_to_end_measurement=True,readout_CPU_times_in_readout_parameters_csv=True,protected_files_unchanged=guard()))

if __name__=='__main__':main()
