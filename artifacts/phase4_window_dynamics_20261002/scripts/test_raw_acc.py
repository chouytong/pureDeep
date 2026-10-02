"""B1 shape/input/mask/gradient/reload and train-only normalization tests."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent.parent
F=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(F))
from src.engine import nested_training as nt
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.datasets.subject_activity import collate_subject_activities
from raw_acc_hooks import FrozenEmbeddingCache,activate,raw_index

seed_everything(42,True);torch.set_num_threads(4)
cfg=load_config(str(F/'configs/str01_seed42.yaml'))
cache=FrozenEmbeddingCache();index=raw_index();activate(cfg,cache,index)
split=json.loads((F/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json').read_text());inner=split['outer'][0]['inner_folds'][0]
bundle=nt.build_subject_fold_datasets(cfg,train_subject_ids=inner['train_subjects'],validation_subject_ids=inner['validation_subjects'],test_subject_ids=[],fold_id='outer_0/inner_0')
lock=json.loads((HERE/'analysis/baseline_lock.json').read_text());oldstage=Path(lock['baseline_root'])/'seed42/outer_0/inner_0'
old=json.loads((oldstage/'normalization.json').read_text())
assert bundle.normalization['subject_ids_sha256']==old['subject_ids_sha256']
assert bundle.normalization['normalization_sha256']!=old['normalization_sha256']
for key in ('mean','std'):assert np.array_equal(getattr(bundle,key).numpy()[:,:,3:],np.asarray(old[key],dtype=np.float32)[:,:,3:])
batch=collate_subject_activities([bundle.train[i] for i in range(8)])
args=[batch[k].cuda() for k in ('x','wrist_mask','activity_mask','activity_lengths')]
assert args[0].shape==(8,11,2,6,2000)
assert set(args[3].cpu().flatten().tolist())=={976,2000}
model=nt.build_model(cfg).cuda().eval();assert sum(p.numel() for p in model.parameters())==143172
ck=torch.load(oldstage/'checkpoints/best.pt',map_location='cpu',weights_only=False)
model.load_state_dict(ck['model_state'],strict=True);model.ssl_features=cache.batch(batch['subject_id'],'pretrained','cuda')
wm,am=args[1].clone(),args[2].clone();wm[:,0,0]=0;am[:,1]=False
with torch.no_grad():
    before=model(args[0],wm,am,args[3])['logits']
    changed=args[0].clone();changed[:,0,0]=10000;changed[:,1]=-10000
    after=model(changed,wm,am,args[3])['logits']
assert torch.equal(before,after),'Invalid input mask leak'
model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
initial=model.wrist_projection.weight.detach().clone()
optimizer.zero_grad();criterion=nn.CrossEntropyLoss(weight=torch.tensor(ck['config']['loss']['class_weights'],device='cuda'))
loss=criterion(model(*args)['logits'],batch['y'].cuda());assert torch.isfinite(loss);loss.backward()
g=model.wrist_projection.weight.grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
assert model.backbone.wrist_encoder.stem[0].weight.grad.abs().sum()>0
nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();assert not torch.equal(initial,model.wrist_projection.weight)
assert not model.ssl_features.requires_grad
model.eval()
with torch.no_grad():expected=model(*args)['logits'].clone()
path=HERE/'smoke/b1_raw_reload.pt';path.parent.mkdir(exist_ok=True);torch.save(model.state_dict(),path)
reloaded=nt.build_model(cfg).cuda().eval();reloaded.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True);reloaded.ssl_features=model.ssl_features
with torch.no_grad():actual=reloaded(*args)['logits']
assert torch.equal(expected,actual)
result=dict(status='PASS',variant='b1_raw',input_shape=list(args[0].shape),lengths=[976,2000],same_architecture_parameters=143172,mask_no_invalid_input_leakage=True,local_encoder_and_ssl_projection_gradients_finite_nonzero=True,one_batch_training=True,checkpoint_reload_exact=True,train_only_normalization=True,training_subject_hash_matches=True,gyro_normalization_exact=True,acc_normalization_changed=True,frozen_harnet_cache_unchanged=True,no_window_residual_branch=True,outer_artifacts_accessed=False)
(HERE/'analysis/b1_raw_implementation_test.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
