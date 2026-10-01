"""Phase-3A isolated training hooks. Frozen STR backbone and final logits are untouched."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.engine import nested_training as nt
from src.engine import runner
from src.metrics import classification_metrics
from phase2_hooks import activate as phase2_activate

class ActivityAuxModel(nn.Module):
    def __init__(self, backbone, projection_dim):
        super().__init__()
        self.backbone = backbone
        self.num_classes = backbone.num_classes
        assert backbone.activity_count == 11 and backbone.activity_feature_dim == 258
        self.activity_aux = nn.Sequential(
            nn.Linear(258, projection_dim), nn.ReLU(), nn.Linear(projection_dim, 2)
        )
        self._activity_features = None
        self.backbone.activity_aggregator.register_forward_pre_hook(self._capture_activity)

    def _capture_activity(self, module, arguments):
        self._activity_features = arguments[0]

    def forward(self, inputs, wrist_mask, activity_mask=None, activity_lengths=None):
        outputs = self.backbone(inputs, wrist_mask, activity_mask, activity_lengths)
        features = self._activity_features
        self._activity_features = None
        if self.training:
            if features is None or features.shape[1:] != (11, 258):
                raise RuntimeError('Pre-aggregation 11x258D activity features missing')
            mask = outputs['activity_mask'].to(dtype=features.dtype)
            per_activity_logits = self.activity_aux(features)
            aux_logits = (per_activity_logits * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            outputs['phase3_aux_logits'] = aux_logits
        return outputs

class ActivityAuxLoss(nn.Module):
    def __init__(self, base, weight):
        super().__init__()
        self.base = base
        self.weight = float(weight)
    def forward(self, outputs, targets):
        main = self.base(outputs, targets)
        if self.training:
            aux = F.cross_entropy(outputs['phase3_aux_logits'], targets,
                                  weight=self.base.class_weights)
            main['activity_aux_loss'] = aux
            main['loss'] = main['loss'] + self.weight * aux
        return main

def train_epoch_sam(model, loader, criterion, optimizer, device, *, scaler=None,
                    mixed_precision=False, gradient_clip_norm=None, max_batches=0,
                    zero_division=0.0, rho=0.05):
    if mixed_precision or (scaler is not None and scaler.is_enabled()):
        raise ValueError('The registered SAM configuration requires the original FP32 STR training')
    model.train(); criterion.train(); started=time.perf_counter()
    sums={}; targets_all=[]; predictions_all=[]; n=0; batches=0
    for batch_index,batch in enumerate(loader):
        if max_batches and batch_index>=max_batches:break
        inputs,targets,wrist_mask,activity_mask,lengths=runner._move_batch(batch,device)
        optimizer.zero_grad(set_to_none=True)
        outputs=runner._model_forward(model,inputs,wrist_mask,activity_mask,lengths)
        first=criterion(outputs,targets)
        first['loss'].backward()
        params=[p for p in model.parameters() if p.grad is not None]
        if not params:raise RuntimeError('SAM first pass has no gradients')
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in params]))
        scale=float(rho)/(norm+1e-12)
        perturb=[]
        with torch.no_grad():
            for p in params:
                e=p.grad*scale
                p.add_(e)
                perturb.append((p,e))
        optimizer.zero_grad(set_to_none=True)
        try:
            second_outputs=runner._model_forward(model,inputs,wrist_mask,activity_mask,lengths)
            second=criterion(second_outputs,targets)
            second['loss'].backward()
        finally:
            with torch.no_grad():
                for p,e in perturb:p.sub_(e)
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip_norm)
        optimizer.step()
        size=int(targets.shape[0]); runner._aggregate_losses(sums,second,size)
        sums['first_pass_loss']=sums.get('first_pass_loss',0.0)+float(first['loss'].detach())*size
        sums['first_pass_classification_loss']=sums.get('first_pass_classification_loss',0.0)+float(first['classification_loss'].detach())*size
        n+=size;batches+=1
        targets_all.append(targets.detach().cpu().numpy())
        predictions_all.append(second_outputs['logits'].detach().argmax(dim=-1).cpu().numpy())
    if not n:raise RuntimeError('No SAM training batches')
    y=np.concatenate(targets_all);pred=np.concatenate(predictions_all)
    y,pred,sums,n=runner._distributed_training_aggregate(y,pred,sums,n,device)
    result=classification_metrics(y,pred,num_classes=model.num_classes,zero_division=zero_division)
    result.update({key:value/n for key,value in sums.items()})
    result.update(duration_seconds=time.perf_counter()-started,batches=batches,
                  learning_rate=optimizer.param_groups[0]['lr'],sam_rho=float(rho))
    return result

def activate(config,variant):
    sampling=variant.get('sampling')
    if sampling:
        assert sampling=='balanced' and config['loss']['class_weights'] is None
        phase2_activate(config,variant)
    if 'sam_rho' in variant:
        assert config['training'].get('optimizer','adamw').lower()=='adamw'
        assert not config['training'].get('mixed_precision',False)
        rho=float(variant['sam_rho'])
        assert rho==0.05
        original_build_model=nt.build_model
        def check_model(cfg):
            m=original_build_model(cfg)
            if any(isinstance(x,nn.modules.batchnorm._BatchNorm) for x in m.modules()):
                raise RuntimeError('SAM path has BatchNorm and requires explicit state handling')
            return m
        nt.build_model=check_model
        nt.train_epoch=lambda model,loader,criterion,optimizer,device,**kw: train_epoch_sam(
            model,loader,criterion,optimizer,device,rho=rho,**kw)
    if 'aux_lambda' in variant:
        weight=float(variant['aux_lambda']);projection_dim=int(variant['projection_dim'])
        assert weight in (0.1,0.3) and projection_dim==32
        original_build_model=nt.build_model; original_build_loss=nt.build_loss
        nt.build_model=lambda cfg: ActivityAuxModel(original_build_model(cfg),projection_dim)
        nt.build_loss=lambda cfg: ActivityAuxLoss(original_build_loss(cfg),weight)
