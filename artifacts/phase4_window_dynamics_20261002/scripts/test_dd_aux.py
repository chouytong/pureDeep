"""C1 DD-only gradients, exact original inference, mask/reload/batch tests."""
import json
import sys
from pathlib import Path

import torch
from torch import nn

HERE=Path(__file__).resolve().parent.parent
F=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(F))
from src.models import build_model
from src.losses import build_loss
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.datasets.builders import load_configured_records
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from dd_aux_hooks import DDCategorySubject,DDCategoryLoss,FrozenEmbeddingCache,ResidualSSLSubject

seed_everything(42,True);torch.set_num_threads(4)
cfg=load_config(str(F/'configs/str01_seed42.yaml'));cache=FrozenEmbeddingCache()
split=json.loads((F/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json').read_text())
train=set(split['outer'][0]['inner_folds'][0]['train_subjects'])
alllabels=json.loads((HERE/'features/dd_source_category_labels.json').read_text());ddlabels={s:alllabels[s] for s in train if s in alllabels}
ids=sorted(train-set(ddlabels))[:4]+sorted(ddlabels)[:4]
records=[r for r in load_configured_records(cfg['data'],['PD','DD']) if r.subject_id in ids]
lock=json.loads((HERE/'analysis/baseline_lock.json').read_text());root=Path(lock['baseline_root'])
first=torch.load(root/'seed42/outer_0/inner_0/checkpoints/best.pt',map_location='cpu',weights_only=False)
dataset=SubjectActivityDataset(records,cache.activities,cfg['data'],first['normalization']['mean'],first['normalization']['std'],'train')
batch=collate_subject_activities([dataset[i] for i in range(8)]);args=[batch[k].cuda() for k in ('x','wrist_mask','activity_mask','activity_lengths')];y=batch['y'].cuda()
assert set(y.tolist())=={0,1}
for s in lock['stages']:
    ck=torch.load(root/f"seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}/checkpoints/best.pt",map_location='cpu',weights_only=False)
    base=ResidualSSLSubject(build_model(cfg)).cuda().eval();base.load_state_dict(ck['model_state'],strict=True)
    model=DDCategorySubject(build_model(cfg),train,ddlabels).cuda().eval();missing,extra=model.load_state_dict(ck['model_state'],strict=False)
    assert set(missing)=={'dd_aux_head.weight','dd_aux_head.bias'} and not extra
    base.ssl_features=cache.batch(batch['subject_id'],'pretrained','cuda');model.ssl_features=base.ssl_features
    assert model.batch_subtypes is None
    with torch.no_grad():
        a=base(*args);b=model(*args)
    assert torch.equal(a['logits'],b['logits']) and 'dd_aux_logits' not in b and 'dd_aux_targets' not in b
    del ck,base,model
model=DDCategorySubject(build_model(cfg),train,ddlabels).cuda();model.load_state_dict(first['model_state'],strict=False)
model.ssl_features=cache.batch(batch['subject_id'],'pretrained','cuda');model.batch_subtypes=torch.tensor([ddlabels.get(s,-1) for s in batch['subject_id']],device='cuda')
assert sum(p.numel() for p in model.parameters())==144208
loss_config=cfg.copy();loss_config['loss']=first['config']['loss'];criterion=DDCategoryLoss(build_loss(loss_config)).cuda().train()
optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
pd=(y==0).nonzero().flatten();dd=(y==1).nonzero().flatten()
# PD-only batch cannot update auxiliary head, including optimizer weight decay.
model.train();original_ssl=model.ssl_features.clone();original_subtypes=model.batch_subtypes.clone()
model.ssl_features=original_ssl[pd];model.batch_subtypes=original_subtypes[pd]
optimizer.zero_grad(set_to_none=True);before=model.dd_aux_head.weight.detach().clone()
losses=criterion(model(*[a[pd] for a in args]),y[pd]);assert losses['dd_aux_loss']==0
losses['loss'].backward();assert model.dd_aux_head.weight.grad is None;optimizer.step();assert torch.equal(before,model.dd_aux_head.weight)
# DD examples contribute auxiliary gradient and retain the original main loss.
model.ssl_features=original_ssl;model.batch_subtypes=original_subtypes
optimizer.zero_grad(set_to_none=True);output=model(*args);losses=criterion(output,y)
assert output['dd_aux_logits'].shape==(8,4) and losses['dd_aux_loss']>0
assert torch.allclose(losses['loss'],losses['classification_loss']+.1*losses['dd_aux_loss'])
losses['loss'].backward();g=model.dd_aux_head.weight.grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
assert model.backbone.wrist_encoder.stem[0].weight.grad.abs().sum()>0
nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
assert not model.ssl_features.requires_grad
model.eval();criterion.eval();model.batch_subtypes=None
wm,am=args[1].clone(),args[2].clone();wm[:,0,0]=0;am[:,1]=False
with torch.no_grad():
    expected=model(*args)['logits'].clone();a=model(args[0],wm,am,args[3])['logits']
    changed=args[0].clone();changed[:,0,0]=10000;changed[:,1]=-10000;b=model(changed,wm,am,args[3])['logits']
    assert torch.equal(a,b)
    assert criterion(model(*args),y)['dd_aux_loss']==0
path=HERE/'smoke/c1_dd_aux_reload.pt';path.parent.mkdir(exist_ok=True);torch.save(model.state_dict(),path)
reload=DDCategorySubject(build_model(cfg),set(),{}).cuda().eval();reload.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True);reload.ssl_features=original_ssl
with torch.no_grad():actual=reload(*args)['logits']
assert torch.equal(expected,actual),'Inference requires no subtype labels or training metadata'
result=dict(status='PASS',variant='c1_dd_aux',input_shape=list(args[0].shape),auxiliary_shape=[8,4],original_inference_exact_for_45_checkpoints=True,pd_only_auxiliary_loss_zero=True,pd_only_auxiliary_gradient_none_and_no_update=True,dd_auxiliary_gradient_finite_nonzero=True,local_encoder_gradient_nonzero=True,single_batch_training=True,checkpoint_reload_exact_without_labels=True,validation_auxiliary_loss_zero=True,invalid_mask_no_leakage=True,trainable_parameters=144208,additional_parameters=1036,auxiliary_weight=.1,frozen_ssl_requires_grad=False,outer_artifacts_accessed=False)
(HERE/'analysis/c1_dd_aux_implementation_test.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
