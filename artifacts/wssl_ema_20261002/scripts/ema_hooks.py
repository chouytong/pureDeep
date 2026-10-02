"""Single EMA trajectory; ordinary training/selection functions remain original."""
import json
from pathlib import Path
import torch
from src.engine import nested_training as nt
from src.datasets import folds as fd
from independent_wssl import IndependentWSSL,independent_copy
from transfer_hooks import SSLLoader

def development_bundle(config, *,train_subject_ids,validation_subject_ids=(),test_subject_ids=(),fold_id):
    assert not test_subject_ids
    allowed=set(map(str,train_subject_ids))|set(map(str,validation_subject_ids));original=fd.load_configured_records
    def selected(*a,**k):return [r for r in original(*a,**k) if r.subject_id in allowed]
    fd.load_configured_records=selected
    try:
        bundle=fd.build_subject_fold_datasets(config,train_subject_ids=train_subject_ids,validation_subject_ids=validation_subject_ids,test_subject_ids=(),fold_id=fold_id)
        assert bundle.test is None
        return bundle
    finally:fd.load_configured_records=original

class EMA:
    def __init__(self,ordinary,config,S):
        self.model=independent_copy(ordinary,config).eval().requires_grad_(False)
        assert not dict(ordinary.named_buffers()),'Unexpected new buffer requires protocol audit'
        self.S=S;self.alpha=2**(-1/S);self.steps=0;self.best_steps=None;self.handle=None
        assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(ordinary.parameters(),self.model.parameters()))
    @torch.no_grad()
    def update(self,ordinary):
        for a,b in zip(self.model.parameters(),ordinary.parameters()):a.mul_(self.alpha).add_(b.detach(),alpha=1-self.alpha)
        self.steps+=1
    def metadata(self):return dict(half_life_epochs=1,steps_per_epoch=self.S,alpha=self.alpha,updates=self.steps,best_updates=self.best_steps,initialized_from_ordinary_initial_weights=True,first_update_step=1,selection='ordinary_BA_only',no_outer=True)

def activate(cache):
    original_train,original_eval,original_save=nt.train_epoch,nt.evaluate_epoch,nt.save_checkpoint
    state={};nt.build_model=lambda cfg:IndependentWSSL(cfg);nt.build_subject_fold_datasets=development_bundle
    def train(model,loader,criterion,optimizer,device,*args,**kwargs):
        if id(model) not in state:
            ema=EMA(model,model._ema_config,len(loader))
            ema.handle=optimizer.register_step_post_hook(lambda opt,a,k:ema.update(model))
            state[id(model)]=ema
        return original_train(model,SSLLoader(loader,model,cache,'pretrained'),criterion,optimizer,device,*args,**kwargs)
    def evaluate(model,loader,*args,**kwargs):return original_eval(model,SSLLoader(loader,model,cache,'pretrained'),*args,**kwargs)
    def save(path,model,optimizer,scheduler,scaler,epoch,best_metric,config,class_names,mean,std):
        original_save(path,model,optimizer,scheduler,scaler,epoch,best_metric,config,class_names,mean,std)
        if Path(path).name=='best.pt':
            ema=state[id(model)];ema.best_steps=ema.steps
            original_save(Path(path).with_name('ema_at_ordinary_best.pt'),ema.model,None,None,None,epoch,best_metric,config,class_names,mean,std)
            Path(path).with_name('ema_selection.json').write_text(json.dumps(ema.metadata(),indent=2)+'\n')
        if Path(path).name=='last.pt':Path(path).with_name('ema_trajectory.json').write_text(json.dumps(state[id(model)].metadata(),indent=2)+'\n')
    def build(cfg):
        model=IndependentWSSL(cfg);model._ema_config=cfg;return model
    nt.build_model=build;nt.train_epoch=train;nt.evaluate_epoch=evaluate;nt.save_checkpoint=save
    return state
