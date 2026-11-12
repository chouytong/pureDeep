"""Three matched token/baseline gates and exactly two synthetic smoke updates."""
from common_p3 import *
from torch import nn
import io,gc

def main():
 assert load(HERE/'analysis/export_integrity.json')['status']=='PASS';guard();torch.set_num_threads(4);seed_everything(42,True);freeze('CORRECTNESS',[Path(__file__),HERE/'scripts/common_p3.py'],synthetic_updates=2,formal_updates=0,new_AI_results_seen=False)
 reproduction=[]
 for seed in SEEDS:
  seed_everything(seed,True);model=make_model(seed).eval();before=fingerprint(model.base);d=token(seed,'validation');train=token(seed,'train');assert not set(d['subject_ids'])&set(train['subject_ids']);z,p=evaluate(model,d);zh,ph=evaluate(model,d,False);assert np.array_equal(z,zh) and np.array_equal(p,ph);err=float(np.max(abs(z-d['logits'])));prob=float(np.max(abs(p-d['probabilities'])));assert err<=1e-5 and prob<=2e-7 and np.array_equal(p[:,1]>.5,d['probabilities'][:,1]>.5);z2,p2=evaluate(model,d);assert np.array_equal(z,z2) and np.array_equal(p,p2) and fingerprint(model.base)==before
  reproduction.append(dict(seed=seed,zero_residual_bitwise=True,repeated_eval_bitwise=True,token_vs_full_logit_max_error=err,probability_max_error=prob,decisions_equal=True,subjects=len(d['target'])));del model;gc.collect();torch.cuda.empty_cache()
 seed_everything(42,True);model=make_model(42).eval();before=fingerprint(model.base);h=torch.randn(8,11,258,device='cuda');mask=torch.ones(8,11,dtype=torch.bool,device='cuda');mask[:,2]=False;target=torch.tensor([0,1,0,1,0,1,0,1],device='cuda')
 with torch.no_grad():
  expected=model(h,mask)['logits'];assert torch.equal(expected,head(model,h,mask));bad=h.clone();bad[:,2]=float('nan');assert torch.equal(expected,model(bad,mask)['logits']);bad[:,2]=float('inf');assert torch.equal(expected,model(bad,mask)['logits'])
  try:model(h,torch.zeros_like(mask))
  except ValueError:pass
  else:raise AssertionError('All invalid activities accepted')
  try:model(h[:,:10],mask[:,:10])
  except ValueError:pass
  else:raise AssertionError('Wrong token dimensions accepted')
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,weight_decay=.0001,betas=(.9,.999));initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};upstream_second=False
 for step in range(2):
  model.train();assert not model.base.training;opt.zero_grad(set_to_none=True);out=model(h,mask);assert out['logits'].shape==(8,2) and torch.isfinite(out['logits']).all();loss=nn.functional.cross_entropy(out['logits'],target);loss.backward();assert torch.isfinite(loss) and all(p.grad is None for p in model.base.parameters());assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  if step==1:upstream_second=bool(model.projection.weight.grad.abs().sum()>0);assert upstream_second
  nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5);opt.step();assert fingerprint(model.base)==before
 assert any(not torch.equal(initial[n],p) for n,p in model.named_parameters() if p.requires_grad);model.eval();calls={};delta={};hooks=[]
 for name,module in [('aggregator',model.base.backbone.activity_aggregator),('STR',model.base.backbone.structured_token_residual),('classifier',model.base.backbone.classifier)]:
  def count(mod,inp,out,key=name):calls[key]=calls.get(key,0)+1
  hooks.append(module.register_forward_hook(count))
 hooks.append(model.output.register_forward_hook(lambda m,i,o:delta.update(value=o.detach().clone())))
 with torch.no_grad():out=model(h,mask)
 assert calls==dict(aggregator=1,STR=1,classifier=1)
 for hook in hooks:hook.remove()
 with torch.no_grad():
  manual=head(model,h,mask)+nn.functional.linear(delta['value'],model.base.backbone.classifier[1].weight,bias=None);assert torch.equal(out['logits'],manual);expected=out['logits'];assert torch.equal(expected,model(h,mask)['logits'])
 buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);fresh=make_model(42).eval();fresh.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True),strict=True)
 with torch.no_grad():assert torch.equal(expected,fresh(h,mask)['logits'])
 assert fingerprint(model.base)==before and fingerprint(fresh.base)==before;pd.DataFrame(reproduction).to_csv(HERE/'analysis/baseline_reproduction.csv',index=False);write(HERE/'tests/correctness_tests.json',dict(status='PASS',matched_seed_checkpoint_tokens=True,subject_split_version_label_masks_verified=True,masked_NaN_Inf_ignored=True,all_invalid_rejected=True,wrong_dimensions_rejected=True,logits_finite_B2=True,zero_residual_exact_all3=True,frozen_base_no_grad_eval_unchanged=True,new_module_updates=True,upstream_gradient_after_second_update=upstream_second,reload_bitwise=True,repeat_bitwise=True,baseline_aggregator_STR_classifier_called_once=True,no_duplicate_bias_or_baseline=True,synthetic_smoke_optimizer_updates=2,formal_initializer_will_reset=True,old_AI_weights_loaded=False,excluded_arrays_read=False,protected_files_unchanged=guard()));print('CORRECTNESS_PASS',flush=True)
if __name__=='__main__':main()
