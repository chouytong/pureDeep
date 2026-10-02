"""DD-only training auxiliary source-category loss; original inference unchanged."""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent.parent
OLD=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');sys.path.insert(0,str(OLD/'scripts'))
from transfer_hooks import ResidualSSLSubject,FrozenEmbeddingCache
from src.engine import nested_training as nt
from window_hooks import development_bundle


class DDCategorySubject(ResidualSSLSubject):
    def __init__(self,backbone,training_ids,training_dd_labels):
        super().__init__(backbone)
        self.dd_aux_head=nn.Linear(258,4)
        self.training_ids=set(training_ids)
        self.training_dd_labels=dict(training_dd_labels)
        assert set(self.training_dd_labels)<=self.training_ids
        self.batch_subtypes=None

    def forward(self,*args,**kwargs):
        output=super().forward(*args,**kwargs)
        if self.training:
            assert self.batch_subtypes is not None
            embedding=output['bag_embedding'];assert embedding.shape[-1]==258
            output['dd_aux_logits']=self.dd_aux_head(embedding)
            output['dd_aux_targets']=self.batch_subtypes
        return output


class DDCategoryLoss(nn.Module):
    requires_activity_prototypes=False
    def __init__(self,main):
        super().__init__();self.main=main
    def forward(self,outputs,targets):
        result=self.main(outputs,targets)
        aux=result['loss'].new_zeros(())
        if self.training and 'dd_aux_logits' in outputs:
            subtype=outputs['dd_aux_targets']
            assert subtype.shape==targets.shape
            dd=targets==1
            assert torch.equal(subtype>=0,dd),'Auxiliary labels must exist only for DD training examples'
            if dd.any():aux=F.cross_entropy(outputs['dd_aux_logits'][dd],subtype[dd])
        result['loss']=result['loss']+.1*aux
        result['dd_aux_loss']=aux
        return result


class DDLoader:
    def __init__(self,base,model,cache):self.base,self.model,self.cache=base,model,cache
    def __len__(self):return len(self.base)
    def __iter__(self):
        for batch in self.base:
            ids=list(map(str,batch['subject_id']));device=next(self.model.parameters()).device
            self.model.ssl_features=self.cache.batch(ids,'pretrained',device)
            if self.model.training:
                assert set(ids)<=self.model.training_ids
                self.model.batch_subtypes=torch.tensor([self.model.training_dd_labels.get(s,-1) for s in ids],dtype=torch.long,device=device)
            else:self.model.batch_subtypes=None
            yield batch


def activate(config,cache,holder):
    assert cache.activities==config['data']['activities']
    original_build,original_loss,original_train,original_eval=nt.build_model,nt.build_loss,nt.train_epoch,nt.evaluate_epoch
    nt.build_model=lambda cfg:DDCategorySubject(original_build(cfg),holder['training_ids'],holder['dd_labels'])
    nt.build_loss=lambda cfg:DDCategoryLoss(original_loss(cfg))
    nt.build_subject_fold_datasets=development_bundle
    def train(model,loader,*args,**kwargs):return original_train(model,DDLoader(loader,model,cache),*args,**kwargs)
    def evaluate(model,loader,*args,**kwargs):return original_eval(model,DDLoader(loader,model,cache),*args,**kwargs)
    nt.train_epoch,nt.evaluate_epoch=train,evaluate
