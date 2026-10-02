import copy,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
H=Path(__file__).resolve().parent.parent;F=Path('/home/zyt/deep_final/foundation_validation');B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path[:0]=[str(F),str(B/'scripts')]
from transfer_hooks import ResidualSSLSubject,FrozenEmbeddingCache
from independent_wssl import IndependentWSSL,independent_copy
from src.datasets.builders import load_configured_records
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import seed_everything

def main():
 seed_everything(42,True);torch.set_num_threads(4);cache=FrozenEmbeddingCache();cfg=load_config(str(F/'configs/str01_seed42.yaml'));records=load_configured_records(cfg['data'],['PD','DD'])
 lock=json.loads(Path('/home/zyt/deep_final/artifacts/phase4_window_dynamics_20261002/analysis/baseline_lock.json').read_text());rows=[];tests={}
 for s in lock['stages']:
  path=B/f"runs/b_str_pretrained/seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}";frame=pd.read_csv(path/'predictions/validation.csv',dtype={'subject_id':str}).sort_values('subject_id').iloc[:8]
  ck=torch.load(path/'checkpoints/best.pt',map_location='cpu',weights_only=False);ids=set(frame.subject_id)
  ds=SubjectActivityDataset([r for r in records if r.subject_id in ids],cache.activities,cfg['data'],ck['normalization']['mean'],ck['normalization']['std'],'validation')
  batch=collate_subject_activities([ds[i] for i in range(len(ds))]);args=[batch[k].cuda() for k in ('x','wrist_mask','activity_mask','activity_lengths')];ssl=cache.batch(batch['subject_id'],'pretrained','cuda')
  old=ResidualSSLSubject(build_model(cfg)).cuda().eval();new=IndependentWSSL(cfg).cuda().eval()
  old.load_state_dict(ck['model_state']);new.load_state_dict(ck['model_state']);old.ssl_features=ssl
  with torch.no_grad():a=old(*args)['logits'];b=new(*args,ssl_features=ssl)['logits']
  assert torch.equal(a,b);p=a.softmax(-1)[:,1].cpu().numpy();assert np.allclose(p,frame.probability_dd,atol=1e-6,rtol=0)
  rows.append(dict(**s,exact=True,archived_probability_max_difference=float(abs(p-frame.probability_dd.to_numpy()).max())))
  if not tests:
   new.ssl_features=ssl.clone();cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all();other=independent_copy(new,cfg)
   assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
   assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(new.parameters(),other.parameters()))
   assert other.ssl_features.data_ptr()!=new.ssl_features.data_ptr()
   with torch.no_grad():
    other.wrist_projection.weight.add_(1);other.ssl_features.zero_();after=new(*args)['logits']
   assert torch.equal(after,b)
   shallow_hazard=copy.deepcopy(old)
   captured=shallow_hazard.backbone._encode_activities_with_wrists.__func__.__closure__
   assert any(c.cell_contents is old for c in captured)
   copied=copy.deepcopy(new);assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(new.parameters(),copied.parameters()))
   with torch.no_grad():assert torch.equal(copied(*args)['logits'],b)
   (H/'smoke').mkdir(exist_ok=True);torch.save(new.state_dict(),H/'smoke/reload.pt');reload=IndependentWSSL(cfg).cuda().eval();reload.load_state_dict(torch.load(H/'smoke/reload.pt',weights_only=True));
   with torch.no_grad():assert torch.equal(reload(*args,ssl_features=ssl)['logits'],b)
   new.train();optim=torch.optim.AdamW(new.parameters(),lr=2e-4,weight_decay=1e-4);optim.zero_grad();loss=torch.nn.functional.cross_entropy(new(*args)['logits'],batch['y'].cuda());loss.backward();assert torch.isfinite(new.wrist_projection.weight.grad).all() and new.wrist_projection.weight.grad.abs().sum()>0;optim.step()
   tests=dict(fresh_instance_parameter_isolation=True,cache_isolation=True,old_deepcopy_captured_original_confirmed=True,new_deepcopy_safe=True,reload_exact=True,copy_rng_preserved=True,gradient_step_finite=True,buffer_names=list(dict(new.named_buffers())),trainable_parameters=sum(p.numel() for p in new.parameters()))
  del old,new,ck
 pd.DataFrame([{k:v for k,v in r.items() if k in ('seed','context','inner','exact','archived_probability_max_difference')} for r in rows]).to_csv(H/'analysis/independence_45checkpoints.csv',index=False)
 result=dict(status='PASS',checkpoints=45,all_logits_bit_identical=True,max_archived_probability_difference=max(r['archived_probability_max_difference'] for r in rows),tests=tests,outer_access=False)
 (H/'analysis/independence_test.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
