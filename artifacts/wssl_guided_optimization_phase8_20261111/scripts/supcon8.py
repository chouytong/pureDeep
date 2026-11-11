"""One asymmetric subject SupCon candidate; original encoder/recipe except objective."""
from common8 import *
from interaction8 import explicit_model,fingerprint
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.utils.seed import seed_everything
import importlib.util,gc,io,copy

def subject_supcon(embedding,labels,identities,tau=.1):
 if len(identities)!=len(set(identities)):raise ValueError('Repeated subject identities cannot form positives')
 pd=labels==0;dd=labels==1
 if pd.sum()<2 or dd.sum()==0:return embedding.sum()*0,int(0)
 z=nn.functional.normalize(embedding,dim=1);similarity=z@z.T/tau;eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
 logden=torch.logsumexp(similarity.masked_fill(eye,-torch.inf),dim=1)
 positive=pd[:,None]&pd[None,:]&~eye;per=(logden[:,None]-similarity).masked_fill(~positive,0).sum(1)/positive.sum(1).clamp_min(1)
 return per[pd].mean(),int(pd.sum())

def records(allowed):
 # Reuse the protected streaming manifest adapter, never its raw-cache inference.
 spec=importlib.util.spec_from_file_location('phase6_input_common',P6/'scripts/common.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.records(set(allowed))
def external(allowed):
 result={};protected=load(HERE/'restricted_manifests/protected_assets.json')['sha256']
 for sid in sorted(allowed):
  path=P6/'features/external_by_subject'/f'{sid}.npz';assert sha(path)==protected[str(path)]
  with np.load(path,allow_pickle=False) as z:
   assert str(z['subject_id'])==sid and z['activities'].tolist()==ACTS and z['wrist_order'].tolist()==['left','right']
   result[sid]=z['layer5'].copy();assert result[sid].shape==(11,2,1024) and np.isfinite(result[sid]).all()
 return result
def inputs(batch,allowed,ext):
 ids=ids_ok(batch['subject_id'],allowed,False)
 ssl=torch.from_numpy(np.stack([ext[s] for s in ids])).cuda();x=[batch[k].cuda(non_blocking=True) for k in ['x','wrist_mask','activity_mask','activity_lengths']]
 return ids,x,ssl,batch['y'].cuda(non_blocking=True)
def make_data(seed,c,i,sp,ex):
 config=cfg(seed);fit=list(map(str,sp['train_subjects']));query=list(map(str,sp['validation_subjects']));assert not set(fit)&set(query) and not (set(fit)|set(query))&set(ex)
 ck=torch.load(baseline(seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False);r=records(fit+query);ext=external(set(fit+query));ds=[]
 for name,ids in [('train',fit),('validation',query)]:
  selected=[v for v in r if v.subject_id in set(ids)];d=SubjectActivityDataset(selected,ACTS,config['data'],ck['normalization']['mean'],ck['normalization']['std'],name);ids_ok(d.subject_ids,ids);ds.append(d)
 paths={v.left_path.resolve() for v in r}|{v.right_path.resolve() for v in r};return ds,ext,ck,paths
def geometry(x,y,categories,split,condition,c,i,seed,margin):
 x=np.asarray(x,float);norm=np.linalg.norm(x,axis=1);u=np.divide(x,norm[:,None],out=np.zeros_like(x),where=norm[:,None]>1e-12);distance=np.clip(1-u@u.T,0,2);rows=[]
 for name,a,b in [('PD_within',y==0,y==0),('DD_within',y==1,y==1),('PD_DD_between',y==0,y==1)]:
  select=a[:,None]&b[None,:]
  if name.endswith('within'):select &=np.triu(np.ones_like(distance,dtype=bool),1)
  values=distance[select];rows.append(dict(context=c,inner=i,seed=seed,split=split,condition=condition,comparison=name,pair_count=len(values),mean_cosine_distance=values.mean() if len(values) else np.nan,patient_count=len(y),dependent_pairs_not_statistical_units=True))
 subtype=[]
 classes=['Other','ET','Atypical','MS']
 for aa,a in enumerate(classes):
  for b in classes[aa:]:
   left=(categories==a)&(y==1);right=(categories==b)&(y==1);select=left[:,None]&right[None,:]
   if a==b:select &=np.triu(np.ones_like(distance,dtype=bool),1)
   values=distance[select];subtype.append(dict(context=c,inner=i,seed=seed,split=split,condition=condition,source_group_a=a,source_group_b=b,left_subject_count=int(left.sum()),right_subject_count=int(right.sum()),pair_count=len(values),mean_cosine_distance=values.mean() if len(values) else np.nan,status='DESCRIPTIVE_BROAD_SOURCE_NOT_ADJUDICATED_SUBTYPE' if len(values) else 'NA_TOO_FEW'))
 margins=[]
 for klass,label in enumerate(['PD','DD']):
  keep=y==klass;margins.append(dict(context=c,inner=i,seed=seed,split=split,condition=condition,label=label,subject_count=int(keep.sum()),embedding_norm_mean=norm[keep].mean(),true_logit_margin_mean=margin[keep].mean()))
 return rows,subtype,margins

def smoke():
 z=torch.tensor([[1.,0],[1.,0],[-1.,0]],requires_grad=True);y=torch.tensor([0,0,1]);a,n=subject_supcon(z,y,['a','b','c']);assert n==2 and 0<=a<1e-5;a.backward();assert torch.isfinite(z.grad).all()
 zero,n=subject_supcon(z,y.new_tensor([0,1,1]),['a','b','c']);assert n==0 and zero==0
 try:subject_supcon(z,y,['a','a','c'])
 except ValueError:pass
 else:raise AssertionError('Duplicate identities accepted')
 perm=torch.tensor([2,0,1]);b,n=subject_supcon(z.detach()[perm],y[perm],['c','a','b']);assert torch.equal(a.detach(),b)
 sp=splits()[0][2];ex=splits()[0][3];seed_everything(42,True);ds,ext,old,paths=make_data(42,0,0,sp,ex)
 # Smoke uses actual inner-train only, no validation selection.
 labels=[ds[0].labels[s] for s in ds[0].subject_ids];chosen=[j for j,v in enumerate(labels) if v==0][:2]+[j for j,v in enumerate(labels) if v==1][:2];chosen+=( [j for j in range(len(labels)) if j not in chosen][:4])
 batch=collate_subject_activities([ds[0][j] for j in chosen]);ids,x,ssl,target=inputs(batch,set(sp['train_subjects']),ext)
 from src.models import build_model
 from src.models.wssl import ResidualSSLSubject
 seed_everything(42,True);legacy=ResidualSSLSubject(build_model(cfg()));legacy_state=legacy.state_dict();seed_everything(42,True);model=explicit_model(cfg());assert all(torch.equal(v,legacy_state[k]) for k,v in model.state_dict().items());del legacy,legacy_state
 model=model.cuda();base=copy.deepcopy(model).cuda();base.load_state_dict(old['model_state']);base.eval();model.train();opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
 counts=np.bincount([ds[0].labels[s] for s in ds[0].subject_ids],minlength=2);criterion=nn.CrossEntropyLoss(weight=torch.tensor(len(ds[0])/(2*counts),dtype=torch.float32,device='cuda'))
 out=model(*x,ssl_features=ssl);ce=criterion(out['logits'],target);con,n=subject_supcon(out['bag_embedding'],target,ids);loss=ce+.1*con;assert torch.isfinite(loss)
 loss.backward();assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.parameters());opt.step();model.eval();before=[v.clone() for v in x];ssl_before=ssl.clone()
 with torch.no_grad():
  expected=model(*x,ssl_features=ssl)['logits'];assert torch.equal(expected,model(*x,ssl_features=ssl)['logits']);changed=x[0].clone()
  for j in range(len(ids)):
   for k in range(11):changed[j,k,:,:,x[3][j,k]:]=999
  assert torch.equal(expected,model(changed,*x[1:],ssl_features=ssl)['logits'])
 assert all(torch.equal(a,b) for a,b in zip(before,x)) and torch.equal(ssl_before,ssl)
 buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);fresh=explicit_model(cfg()).cuda().eval();fresh.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True))
 with torch.no_grad():assert torch.equal(expected,fresh(*x,ssl_features=ssl)['logits'])
 assert n>=2
 write(HERE/'tests/supcon_smoke.json',dict(status='PASS',one_subject_one_sample=True,PD_positive_only_no_DD_attraction=True,duplicate_identity_rejected=True,hand_loss_and_gradient_finite=True,missing_positive_loss_zero=True,permutation_invariant=True,train_query_separation=True,raw_inputs_and_frozen_external_not_mutated=True,padding_and_reload_exact=True,original_initialization_bitwise_matched=True,original_recipe_except_auxiliary_objective=True,validation_for_smoke_training=False,smoke_optimizer_updates=1,valid_PD_anchors_in_model_smoke=n,formal_initialization_will_reset=True))
 del model,base,fresh,opt,ds,ext;gc.collect();torch.cuda.empty_cache()

def main():
 assert load(HERE/'analysis/activity_interaction_integrity.json')['status']=='PASS_COMPLETE_B';guard();freeze('C',[HERE/'PROTOCOL.md',HERE/'scripts/common8.py',HERE/'scripts/interaction8.py',Path(__file__)],B_report_archived=True,new_C_results_seen=False,lambda_supcon=.1,tau=.1,positive_rule='differentPDsubjects only; DDnegative not DDpositive',initialization='original seed initialization, not warm-start old classifier',budget='original50epochs/patience12/batch8/AdamW/cosine',subtype_supervision=False)
 torch.set_num_threads(4);smoke();assert load(HERE/'tests/supcon_smoke.json')['status']=='PASS';rows=[];costs=[];geos=[];subtypes=[];margins=[];lineage=[];start=time.perf_counter()
 # Only broad existing source metadata, not granular adjudicated diagnosis supervision.
 rawmeta=pd.read_csv(cfg()['data']['activity_manifests']['CrossArms'],dtype={'subject_id':str});mapping={'Other Movement Disorders':'Other','Essential Tremor':'ET','Atypical Parkinsonism':'Atypical','Multiple Sclerosis':'MS',"Parkinson's":'PD'};categories=dict(zip(rawmeta.subject_id,rawmeta.source_condition.map(mapping)))
 import src.datasets.manifest as manifest
 original_load=manifest._load_array
 try:
  for c,i,sp,ex in splits():
   for seed in [42,43,44]:
    seed_everything(seed,True);ds,ext,old,paths=make_data(seed,c,i,sp,ex);fit=set(map(str,sp['train_subjects']));query=set(map(str,sp['validation_subjects']))
    def guarded(path,delimiter):
     if Path(path).resolve() not in paths:raise PermissionError('Signal outside current fit/query allowlist')
     return original_load(path,delimiter)
    manifest._load_array=guarded
    try:guarded('/forbidden/outer_signal.npy',',')
    except PermissionError:pass
    else:raise AssertionError('No fail-closed I/O')
    gen=torch.Generator().manual_seed(seed);vg=torch.Generator().manual_seed(seed+1)
    trainloader=DataLoader(ds[0],batch_size=8,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,generator=gen,collate_fn=collate_subject_activities)
    valloader=DataLoader(ds[1],batch_size=8,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,generator=vg,collate_fn=collate_subject_activities)
    model=explicit_model(cfg(seed)).cuda();n=np.bincount([ds[0].labels[s] for s in ds[0].subject_ids],minlength=2);criterion=nn.CrossEntropyLoss(weight=torch.tensor(len(ds[0])/(2*n),dtype=torch.float32,device='cuda'))
    opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001,betas=(.9,.999));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-6);stage=HERE/'checkpoints'/f'supcon_seed{seed}_context{c}_inner{i}';stage.mkdir(exist_ok=False);best=-np.inf;wait=0;epochlog=[];steps=0;zero_batches=0;all_batches=0;valid_anchors=0;beg=time.perf_counter()
    def evaluate(loader,allowed,embeddings=False):
     model.eval();zs=[];ps=[];ys=[];ids=[];es=[]
     with torch.inference_mode():
      for batch in loader:
       current,x,ssl,t=inputs(batch,allowed,ext);out=model(*x,ssl_features=ssl);assert torch.isfinite(out['logits']).all();ids.extend(current);ys.extend(t.cpu().tolist());zs.append(out['logits'].cpu().numpy());ps.append(torch.softmax(out['logits'],1).cpu().numpy())
       if embeddings:es.append(out['bag_embedding'].cpu().numpy())
     ids_ok(ids,allowed);return np.asarray(ids),np.asarray(ys),np.concatenate(zs),np.concatenate(ps),np.concatenate(es) if es else None
    for epoch in range(50):
     model.train();total=ce_sum=sc_sum=0.;seen=0;epoch_valid=0
     for batch in trainloader:
      ids,x,ssl,t=inputs(batch,fit,ext);opt.zero_grad(set_to_none=True);out=model(*x,ssl_features=ssl);ce=criterion(out['logits'],t);sc,anchors=subject_supcon(out['bag_embedding'],t,ids);loss=ce+.1*sc;assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();steps+=1;all_batches+=1;zero_batches+=anchors==0;valid_anchors+=anchors;epoch_valid+=anchors;seen+=len(t);total+=float(loss.detach())*len(t);ce_sum+=float(ce.detach())*len(t);sc_sum+=float(sc.detach())*len(t)
     qids,y,z,p,_=evaluate(valloader,query);m=pred_metrics(y,p,z);improved=m['ba']>best
     if improved:
      best=m['ba'];bestepoch=epoch+1;wait=0;torch.save(dict(model_state=model.state_dict(),epoch=bestepoch,seed=seed,context=c,inner=i,source_sha256=sha(Path(__file__)),normalization=old['normalization'],normalization_sha256=sha(baseline(seed,c,i)/'normalization.json'),lambda_supcon=.1,tau=.1,HarNet_weights_not_in_optimizer=True),stage/'best.pt')
     else:wait+=1
     scheduler.step();epochlog.append(dict(epoch=epoch+1,train_loss=total/seen,train_CE=ce_sum/seen,train_SupCon=sc_sum/seen,valid_PD_anchors=epoch_valid,validation_BA=m['ba'],improved=improved,optimizer_steps=steps))
     if (epoch+1)%5==0:print(json.dumps(dict(stage='C_epoch',context=c,inner=i,seed=seed,epoch=epoch+1,best_BA=best,elapsed_seconds=time.perf_counter()-start)),flush=True)
     if wait>=12:break
    assert valid_anchors>0 and np.isfinite([r['train_SupCon'] for r in epochlog]).all();model.load_state_dict(torch.load(stage/'best.pt',map_location='cuda',weights_only=False)['model_state']);qids,y,z,p,e=evaluate(valloader,query,True);rows.append(save_prediction('subject_supcon',seed,c,i,qids,y,p,z));elapsed=time.perf_counter()-beg
    # No comparison of coordinates across trained models: compare within-space geometry scalars.
    train_eval=DataLoader(ds[0],batch_size=8,shuffle=False,num_workers=0,collate_fn=collate_subject_activities)
    original_mode=ds[0].activity_dropout;ds[0].activity_dropout=0.;ds[0].rotation_degrees=0.
    tids,ty,tz,tp,te=evaluate(train_eval,fit,True);ds[0].activity_dropout=original_mode
    cache=features(seed,c,i);lookup={sid:j for j,sid in enumerate(cache['subject_ids'])}
    for split,ids,target,logits,embed in [('train',tids,ty,tz,te),('validation',qids,y,z,e)]:
     ix=[lookup[s] for s in ids];assert np.array_equal(target,cache['target'][ix]);cat=np.array([categories[s] for s in ids])
     for name,rep,rawlogits in [('baseline',cache['subject'][ix],cache['logits'][ix]),('subject_supcon',embed,logits)]:
      signed=(2*target-1)*(rawlogits[:,1]-rawlogits[:,0]);g,s,ma=geometry(rep,target,cat,split,name,c,i,seed,signed);geos+=g;subtypes+=s;margins+=ma
    write(HERE/'logs'/f'C_seed{seed}_{c}_{i}.json',epochlog);costs.append(dict(context=c,inner=i,seed=seed,total_classifier_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen_external_parameters=10457408,epochs=len(epochlog),best_epoch=bestepoch,optimizer_steps=steps,total_train_batches=all_batches,zero_SupCon_batches=zero_batches,valid_PD_anchors=valid_anchors,training_validation_seconds=elapsed,validation_inference_seconds_not_separately_instrumented=True))
    lineage.append(dict(context=c,inner=i,seed=seed,fit_subject_ids=sorted(fit),query_subject_ids=sorted(query),excluded_subject_ids=ex,checkpoint_sha256=sha(stage/'best.pt'),source_sha256=sha(Path(__file__)),normalization_sha256=sha(baseline(seed,c,i)/'normalization.json'),preprocessing_sha256=sha(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/processing_config.json'),external_feature_paths=[str(P6/'features/external_by_subject'/f'{sid}.npz') for sid in sorted(fit|query)],external_encoder_supervised_fit_ids=[],raw_signal_role_boundary='trainloader fit-only; validationloader query-only; no outer loader'))
    pd.DataFrame(rows).to_csv(HERE/'analysis/supcon_progress.csv',index=False);pd.DataFrame(costs).to_csv(HERE/'analysis/supcon_cost_progress.csv',index=False);print(json.dumps(dict(stage='C',completed=len(rows),context=c,inner=i,seed=seed,epochs=len(epochlog),BA=rows[-1]['ba'],elapsed_seconds=time.perf_counter()-start)),flush=True)
    del model,opt,trainloader,valloader,train_eval,ds,ext;gc.collect();torch.cuda.empty_cache()
 finally:manifest._load_array=original_load
 frame=pd.DataFrame(rows);frame.to_csv(HERE/'analysis/supcon_metrics.csv',index=False);pd.DataFrame(costs).to_csv(HERE/'analysis/supcon_parameters.csv',index=False);pd.DataFrame(geos).to_csv(HERE/'analysis/supcon_embedding_geometry.csv',index=False);pd.DataFrame(subtypes).to_csv(HERE/'analysis/supcon_subtype_analysis.csv',index=False);pd.DataFrame(margins).to_csv(HERE/'analysis/supcon_class_margin.csv',index=False);write(HERE/'restricted_manifests/supcon_lineage.json',dict(records=lineage))
 base=pd.read_csv(HERE/'analysis/ensemble_metrics.csv');base=base[base.condition=='baseline'];stat=paired(frame,base,'subject_supcon');stat['primary_bh_q']=np.nan;ix=stat.metric.isin(['ba','auroc']);stat.loc[ix,'primary_bh_q']=bh_planned(stat.loc[ix,'wilcoxon_p']);stat.to_csv(HERE/'analysis/supcon_paired_statistics.csv',index=False);decision=gate(frame,stat,base,'subject_supcon')
 (HERE/'reports/SUBJECT_SUPCON_REPORT.md').write_text('# Phase8C Subject-level SupCon\n\nB已全部完成归档后才启动。单一预先固定CE+.1SupCon，tau.1；仅不同PD患者作positive，DD作negative而不强制DD跨亚型吸引。不增加projection/分类分支/augmentation/sampler；原WSSL分类侧从原seed初始化，HarNet不进入优化器；原norm和50epoch/patience12/batch8/AdamW/cosine recipe保留。不同人样本，不把活动/腕/window作为患者。此asymmetric变体区别于[原SupCon](https://arxiv.org/abs/2004.11362)，不能用结果泛化否定全部SupCon实现。\n\n'+table(frame.groupby('condition')[METRICS].mean().reset_index())+'\n\n'+table(stat)+'\n\n**'+decision['status']+'**。DD距离/粗来源类间距离、norm/margin及anchor覆盖见CSV；粗来源类别不是可靠细粒度临床亚型，不用于新监督。即使类间距离增大也不自动意味着疾病信息增加。相关geometry的患者/seed/split依赖限制保留，末阶段统一计算cluster敏感性。\n')
 write(HERE/'analysis/supcon_integrity.json',dict(status='PASS_COMPLETE_C',runs=45,smoke='PASS',formal_optimizer_updates=sum(r['optimizer_steps'] for r in costs),valid_PD_anchors=sum(r['valid_PD_anchors'] for r in costs),zero_contrastive_batches=sum(r['zero_SupCon_batches'] for r in costs),all_train_batches=sum(r['total_train_batches'] for r in costs),new_loss_candidate_count=1,lambda_supcon=.1,tau=.1,DD_subtype_supervision=False,original_HarNet_preprocessing_norm_unchanged=True,decision=decision,elapsed_seconds=time.perf_counter()-start,report_sha256=sha(HERE/'reports/SUBJECT_SUPCON_REPORT.md')));guard();print(json.dumps(clean(dict(stage='C_COMPLETE',decision=decision)),ensure_ascii=False))
if __name__=='__main__':main()
