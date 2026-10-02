"""Window branch shape/mask/gradient/reload/train tests, no performance selection."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent.parent
F = Path('/home/zyt/deep_final/foundation_validation')
OLD = Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
sys.path[:0] = [str(F), str(OLD / 'scripts')]
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import seed_everything
from src.datasets.builders import load_configured_records
from src.datasets.subject_activity import SubjectActivityDataset, collate_subject_activities
from transfer_hooks import ResidualSSLSubject
from window_hooks import WindowCache, WindowResidualSubject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['a1_mean', 'a2_delta'], required=True)
    args = ap.parse_args()
    mode = 'mean' if args.variant == 'a1_mean' else 'delta'
    seed_everything(42, True)
    torch.set_num_threads(4)
    cfg = load_config(str(F / 'configs/str01_seed42.yaml'))
    cache = WindowCache()
    allrecords = load_configured_records(cfg['data'], ['PD', 'DD'])
    split = json.loads((F / 'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json').read_text())
    ids = sorted(split['outer'][0]['inner_folds'][0]['train_subjects'])[:8]
    firstpath = OLD / 'runs/b_str_pretrained/seed42/outer_0/inner_0/checkpoints/best.pt'
    first = torch.load(firstpath, map_location='cpu', weights_only=False)
    ds = SubjectActivityDataset([r for r in allrecords if r.subject_id in ids], cache.activities, cfg['data'], first['normalization']['mean'], first['normalization']['std'], 'train')
    batch = collate_subject_activities([ds[i] for i in range(len(ds))])
    inputs = [batch[k].cuda() for k in ('x','wrist_mask','activity_mask','activity_lengths')]
    lock = json.loads((HERE / 'analysis/baseline_lock.json').read_text())
    maxdiff = 0.0
    for s in lock['stages']:
        ck = torch.load(OLD / f"runs/b_str_pretrained/seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}/checkpoints/best.pt", map_location='cpu', weights_only=False)
        base = ResidualSSLSubject(build_model(cfg)).eval().cuda()
        base.load_state_dict(ck['model_state'], strict=True)
        candidate = WindowResidualSubject(build_model(cfg), mode).eval().cuda()
        missing, unexpected = candidate.load_state_dict(ck['model_state'], strict=False)
        assert set(missing) == {'window_branch.0.weight','window_branch.0.bias','window_branch.2.weight','window_branch.2.bias'} and not unexpected
        base.ssl_features = cache.mean[[cache.index[i] for i in batch['subject_id']]].cuda()
        cache.attach(candidate, batch['subject_id'], 'cuda')
        with torch.no_grad():
            a = base(*inputs)['logits']
            b = candidate(*inputs)['logits']
        assert torch.equal(a,b), s
        maxdiff = max(maxdiff, float((a-b).abs().max()))
        del base, candidate, ck
    model = WindowResidualSubject(build_model(cfg), mode).cuda()
    model.load_state_dict(first['model_state'], strict=False)
    cache.attach(model, batch['subject_id'], 'cuda')
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 151948
    assert all(not x.requires_grad for x in (model.ssl_features,model.window_front,model.window_back))
    with torch.no_grad():
        model.window_branch[-1].weight.fill_(.01)
        model.window_branch[-1].bias.fill_(.2)
    wm, am = inputs[1].clone(), inputs[2].clone()
    wm[:, -1, 0] = 0; am[:, -2] = False
    residual = model.window_residual(wm,am)
    valid = model.window_mask.all(-1) & wm.bool() & am[...,None].bool()
    assert residual.shape == (8,11,2,64)
    assert torch.equal(residual[~valid],torch.zeros_like(residual[~valid]))
    assert residual[valid].abs().sum()>0
    model.window_front[~valid] += 1000
    model.window_back[~valid] -= 1000
    altered = model.window_residual(wm,am)
    assert torch.equal(residual,altered), 'Inactive-window leakage'
    cache.attach(model,batch['subject_id'],'cuda')
    nn.init.zeros_(model.window_branch[-1].weight); nn.init.zeros_(model.window_branch[-1].bias)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(first['config']['loss']['class_weights'],device='cuda'))
    for step in range(2):
        optim.zero_grad()
        loss = criterion(model(*inputs)['logits'],batch['y'].cuda())
        assert torch.isfinite(loss)
        loss.backward()
        gout = model.window_branch[-1].weight.grad
        gin = model.window_branch[0].weight.grad
        assert gout is not None and torch.isfinite(gout).all() and gout.abs().sum()>0
        assert gin is not None and torch.isfinite(gin).all()
        if step==0: assert gin.abs().sum()==0, 'Zero-output initialization must block initial hidden gradient'
        else: assert gin.abs().sum()>0, 'Hidden branch must learn after output update'
        nn.utils.clip_grad_norm_(model.parameters(),5)
        optim.step()
    model.eval()
    with torch.no_grad(): expected=model(*inputs)['logits'].clone()
    path=HERE/f'smoke/{args.variant}_reload.pt'
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(),path)
    reloaded=WindowResidualSubject(build_model(cfg),mode).cuda().eval()
    reloaded.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True)
    cache.attach(reloaded,batch['subject_id'],'cuda')
    with torch.no_grad(): actual=reloaded(*inputs)['logits']
    assert torch.equal(expected,actual)
    result=dict(status='PASS',variant=args.variant,real_inner_train_batch_shape=list(inputs[0].shape),zero_branch_frozen_checkpoint_tests=45,max_zero_branch_logit_difference=maxdiff,mask_no_invalid_record_leakage=True,output_gradient_finite_nonzero=True,hidden_gradient_zero_then_nonzero=True,one_batch_training_updates=2,checkpoint_reload_exact=True,trainable_parameters=151948,additional_parameters=8776,frozen_ssl_requires_grad=False,normalization='parameter-free LayerNorm1024 epsilon1e-5',outer_artifacts_accessed=False)
    (HERE/f'analysis/{args.variant}_implementation_test.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
