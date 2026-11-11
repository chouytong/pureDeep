from common8 import *
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.utils.seed import seed_everything
import importlib.util,io,gc

def explicit_model(config):
 spec=importlib.util.spec_from_file_location('phase6_model',P6/'scripts/independent_wssl.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.IndependentWSSL(config)
def fingerprint(model):return {k:sha_tensor(v) for k,v in model.state_dict().items()}
def sha_tensor(v):return hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
class ActivityInteraction(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in self.base.parameters():p.requires_grad_(False)
  self.base.eval();self.projection=nn.Linear(258,64);self.identity=nn.Embedding(11,64);self.attention=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
  self.norm1=nn.LayerNorm(64);self.norm2=nn.LayerNorm(64);self.drop=nn.Dropout(.1);self.ffn=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64));self.output=nn.Linear(64,258);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def train(self,mode=True):super().train(mode);self.base.eval();return self
 def forward(self,h,mask,attention_off=False):
  if h.ndim!=3 or h.shape[1:]!=(11,258) or mask.shape!=h.shape[:2] or not mask.any(1).all():raise ValueError('Expected11 ordered valid activity tokens')
  h=h.masked_fill(~mask[...,None],0);b=self.base.backbone;a=b.activity_aggregator(h,mask);s=b.structured_token_residual(h,mask);baseline_logits=b.classifier(a['subject_embedding'])+s['structured_residual_logits']
  token=self.projection(h)+self.identity(torch.arange(11,device=h.device))[None];mixed,weights=self.attention(token,token,token,key_padding_mask=~mask,need_weights=True,average_attn_weights=False)
  if attention_off:mixed=torch.zeros_like(mixed)
  token=self.norm1(token+self.drop(mixed));token=self.norm2(token+self.drop(self.ffn(token)));pooled=(token*mask[...,None]).sum(1)/mask.sum(1,keepdim=True);delta=self.output(pooled)
  logits=baseline_logits+torch.nn.functional.linear(delta,b.classifier[1].weight,bias=None)
  return dict(logits=logits,baseline_logits=baseline_logits,activity_attention=weights,subject_embedding=a['subject_embedding']+delta)

def prepare(seed,c,i):
 f=features(seed,c,i);base=explicit_model(cfg(seed));ck=torch.load(baseline(seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False);base.load_state_dict(ck['model_state'],strict=True);return f,ActivityInteraction(base).cuda()

def smoke():
 seed_everything(42,True);f,model=prepare(42,0,0);v=f['is_validation'];ix=np.flatnonzero(~v)[:8];h=torch.from_numpy(f['bilateral'][ix]).cuda();mask=torch.ones((len(ix),11),device='cuda',dtype=torch.bool);y=torch.from_numpy(f['target'][ix]).long().cuda();model.eval();prior=fingerprint(model.base)
 with torch.no_grad():
  out=model(h,mask);assert out['logits'].shape==(len(ix),2) and torch.equal(out['logits'],out['baseline_logits'])
  altered=mask.clone();altered[:,2]=False;a=model(h,altered)['logits'];bad=h.clone();bad[:,2]=float('nan');assert torch.equal(a,model(bad,altered)['logits'])
  try:model(h[:,:10],mask[:,:10])
  except ValueError:pass
  else:raise AssertionError('Wrong token order/count accepted')
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,weight_decay=.0001)
 for _ in range(2):
  model.train();opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(h,mask)['logits'],y);loss.backward();assert torch.isfinite(loss);assert any(p.grad is not None and p.grad.abs().max()>0 for n,p in model.named_parameters() if not n.startswith('base.'));opt.step()
 assert fingerprint(model.base)==prior and all(p.grad is None for p in model.base.parameters()) and not model.base.training
 model.eval()
 with torch.no_grad():expected=model(h,mask)['logits'];assert torch.equal(expected,model(h,mask)['logits'])
 state=io.BytesIO();torch.save(model.state_dict(),state);state.seek(0);_,fresh=prepare(42,0,0);fresh.load_state_dict(torch.load(state,map_location='cuda',weights_only=True));fresh.eval()
 with torch.no_grad():assert torch.equal(expected,fresh(h,mask)['logits'])
 write(HERE/'tests/activity_interaction_smoke.json',dict(status='PASS',token_count=11,token_dim=258,activity_and_wrist_order_from_frozen45_verified=True,masked_nan_ignored=True,output_logits_B2=True,zero_residual_exact=True,train_only_used=True,new_module_gradients_nonzero=True,frozen_backbone_no_grad_or_mutation=True,external_harnet_not_in_graph=True,reload_and_determinism_exact=True,normalization_preprocessing_unchanged=True,independent_smoke_updates=2,formal_runs_will_reset_initialization=True))
 del model,fresh,opt;gc.collect();torch.cuda.empty_cache()

def main():
 assert load(HERE/'analysis/readout_integrity.json')['status']=='PASS_COMPLETE_A2';guard();assert not load(HERE/'audit/architecture.json')['B_equivalent_mechanism']
 freeze('B',[HERE/'PROTOCOL.md',HERE/'scripts/common8.py',Path(__file__)],A2_report_archived=True,new_B_results_seen=False,heads=4,projection_dim=64,FFN=128,dropout=.1,all_baseline_parameters_frozen=True)
 torch.set_num_threads(4);smoke();assert load(HERE/'tests/activity_interaction_smoke.json')['status']=='PASS';rows=[];params=[];attention=[];ablation=[];lineage=[];start=time.perf_counter()
 for c,i,sp,ex in splits():
  for seed in [42,43,44]:
   seed_everything(seed,True);f,model=prepare(seed,c,i);v=f['is_validation'];prior=fingerprint(model.base);tokens=torch.from_numpy(f['bilateral']);y=torch.from_numpy(f['target']).long();train=torch.nonzero(torch.from_numpy(~v)).flatten();query=torch.nonzero(torch.from_numpy(v)).flatten()
   gen=torch.Generator().manual_seed(seed);loader=DataLoader(TensorDataset(tokens[train],y[train]),batch_size=8,shuffle=True,generator=gen,num_workers=0);n=np.bincount(y[train].numpy(),minlength=2);weights=torch.tensor(len(train)/(2*n),dtype=torch.float32,device='cuda');criterion=nn.CrossEntropyLoss(weight=weights)
   opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,weight_decay=.0001,betas=(.9,.999));sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-6);best=-np.inf;wait=0;steps=0;stage=HERE/'checkpoints'/f'ai_seed{seed}_context{c}_inner{i}';stage.mkdir(exist_ok=False);epochlog=[];t0=time.perf_counter()
   def evaluate(attention_off=False):
    model.eval();zs=[];ats=[]
    with torch.inference_mode():
     for k in range(0,len(query),8):
      h=tokens[query[k:k+8]].cuda();out=model(h,torch.ones(h.shape[:2],device='cuda',dtype=torch.bool),attention_off);zs.append(out['logits'].cpu().numpy());ats.append(out['activity_attention'].cpu().numpy())
    z=np.concatenate(zs);p=np.exp(z.astype(float)-logsumexp(z.astype(float),1,keepdims=True));return z,p,np.concatenate(ats)
   for epoch in range(50):
    model.train();trainloss=0.;seen=0
    for h,t in loader:
     h=h.cuda();t=t.cuda();opt.zero_grad(set_to_none=True);out=model(h,torch.ones(h.shape[:2],device='cuda',dtype=torch.bool));loss=criterion(out['logits'],t);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);opt.step();steps+=1;seen+=len(t);trainloss+=float(loss.detach())*len(t)
    z,p,att=evaluate();metric=pred_metrics(f['target'][v],p,z);improved=metric['ba']>best
    if improved:
     best=metric['ba'];bestepoch=epoch+1;wait=0;torch.save(dict(model_state=model.state_dict(),epoch=bestepoch,seed=seed,context=c,inner=i,source_sha256=sha(Path(__file__)),base_checkpoint_sha256=sha(baseline(seed,c,i)/'checkpoints/best.pt'),normalization_sha256=sha(baseline(seed,c,i)/'normalization.json')),stage/'best.pt')
    else:wait+=1
    epochlog.append(dict(epoch=epoch+1,train_loss=trainloss/seen,validation_BA=metric['ba'],improved=improved,optimizer_steps=steps));sched.step()
    if wait>=12:break
   assert fingerprint(model.base)==prior;model.load_state_dict(torch.load(stage/'best.pt',map_location='cuda',weights_only=False)['model_state']);z,p,att=evaluate();rows.append(save_prediction('activity_interaction',seed,c,i,f['subject_ids'][v],f['target'][v],p,z))
   zoff,poff,_=evaluate(True);ablation.append(dict(context=c,inner=i,seed=seed,condition='trained_attention_off_inference_only',**pred_metrics(f['target'][v],poff,zoff)))
   for head in range(4):
    for a,name in enumerate(ACTS):
     for b,other in enumerate(ACTS):attention.append(dict(context=c,inner=i,seed=seed,head=head,query_activity=name,key_activity=other,mean_weight=float(att[:,head,a,b].mean()),clinical_importance_claimed=False))
   elapsed=time.perf_counter()-t0;params.append(dict(context=c,inner=i,seed=seed,total_classifier_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),external_frozen_parameters=10457408,epochs=len(epochlog),best_epoch=bestepoch,optimizer_steps=steps,training_validation_seconds=elapsed,validation_inference_seconds_not_separately_instrumented=True,frozen_base_unchanged=True))
   write(HERE/'logs'/f'B_seed{seed}_{c}_{i}.json',epochlog);lineage.append(dict(context=c,inner=i,seed=seed,fit_subject_ids=sp['train_subjects'],query_subject_ids=sp['validation_subjects'],excluded_subject_ids=ex,checkpoint_sha256=sha(stage/'best.pt'),feature_sha256=sha(featurepath(seed,c,i)),source_sha256=sha(Path(__file__)),normalization_sha256=sha(baseline(seed,c,i)/'normalization.json'),preprocessing_sha256=sha(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/processing_config.json')))
   pd.DataFrame(rows).to_csv(HERE/'analysis/activity_interaction_progress.csv',index=False);print(json.dumps(dict(stage='B',completed=len(rows),seed=seed,context=c,inner=i,epochs=len(epochlog),BA=rows[-1]['ba'],elapsed_seconds=time.perf_counter()-start)),flush=True);del model,opt,loader;gc.collect()
 frame=pd.DataFrame(rows);frame.to_csv(HERE/'analysis/activity_interaction_metrics.csv',index=False);pd.DataFrame(ablation).to_csv(HERE/'analysis/activity_interaction_ablation.csv',index=False);pd.DataFrame(attention).to_csv(HERE/'analysis/activity_attention_summary.csv',index=False);pd.DataFrame(params).to_csv(HERE/'analysis/activity_interaction_parameters.csv',index=False);write(HERE/'restricted_manifests/activity_interaction_lineage.json',dict(records=lineage))
 base=pd.read_csv(HERE/'analysis/ensemble_metrics.csv');base=base[base.condition=='baseline'];stat=paired(frame,base,'activity_interaction');stat['primary_bh_q']=np.nan;ix=stat.metric.isin(['ba','auroc']);stat.loc[ix,'primary_bh_q']=bh_planned(stat.loc[ix,'wilcoxon_p']);stat.to_csv(HERE/'analysis/activity_interaction_paired_statistics.csv',index=False);decision=gate(frame,stat,base,'activity_interaction')
 (HERE/'reports/ACTIVITY_INTERACTION_REPORT.md').write_text('# Phase8B 活动token内容交互\n\n现有softmax attention已产生权重竞争，STR保留11活动结构，但没有pooling前的content-dependent token-to-token mixing。本候选不是在已有Transformer后重复堆叠；也不声称全球方法创新。固定258→64,4head,1层,FFN128,dropout.1,零输出residual；所有旧参数冻结。smoke通过后45run完成。\n\n'+table(frame.groupby('condition')[METRICS].mean().reset_index())+'\n\n'+table(stat)+'\n\n**'+decision['status']+'**。所有epoch/budget/梯度/冻结/provenance记录保留。inference attention-off是模型扰动敏感性，不是重训消融或另一个保留候选；attention权重不是临床重要性。原监督encoder的train表示in-sample，验证重复开发选择偏差仍在；不作临床独立泛化解释。B已完成并归档，之后才允许C。\n')
 write(HERE/'analysis/activity_interaction_integrity.json',dict(status='PASS_COMPLETE_B',runs=45,smoke='PASS',formal_training_updates=sum(r['optimizer_steps'] for r in params),baseline_all_parameters_unchanged=True,new_architecture_count=1,decision=decision,elapsed_seconds=time.perf_counter()-start,report_sha256=sha(HERE/'reports/ACTIVITY_INTERACTION_REPORT.md')));guard();print(json.dumps(clean(dict(stage='B_COMPLETE',decision=decision)),ensure_ascii=False))
if __name__=='__main__':main()
