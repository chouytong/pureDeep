"""Pre-training test of the sole DD relative cost; no clinical optimization."""
from common10 import *
from train10 import config
from src.losses import build_loss
import torch
import torch.nn.functional as F

def main():
 counts=[]
 for c,i,sp,ex in splits():
  meta=metadata();fit=list(map(str,sp['train_subjects']))
  n=np.bincount([meta[s]['label'] for s in fit],minlength=2)
  old=len(fit)/(2*n);new=old*np.array([1.,1.1]);scale=n@new/len(fit);new/=scale
  assert abs(n@old/len(fit)-1)<1e-12 and abs(n@new/len(fit)-1)<1e-12
  assert abs((new[1]/new[0])/(old[1]/old[0])-1.1)<1e-12
  assert abs(scale-1.05)<1e-12
  counts.append(dict(context=c,inner=i,PD_count=int(n[0]),DD_count=int(n[1]),old_weights=old.tolist(),new_weights=new.tolist(),relative_DD_multiplier=1.1,scale=scale))
  for y in [torch.tensor([0,1,0,0,1]),torch.zeros(5,dtype=torch.long),torch.ones(5,dtype=torch.long)]:
   z=torch.tensor([[.3,-.2],[-.1,.5],[.6,.1],[-.3,.4],[.8,-.6]],requires_grad=True)
   cfg_=config(42);cfg_['loss']['class_weights']=new.tolist()
   loss=build_loss(cfg_)({'logits':z,'probabilities':z.softmax(-1)},y)['loss']
   w=torch.tensor(new,dtype=z.dtype);each=F.cross_entropy(z,y,reduction='none')
   manual=(w[y]*each).sum()/w[y].sum()
   assert torch.allclose(loss,manual,atol=1e-7,rtol=1e-6)
   grad=torch.autograd.grad(loss,z,retain_graph=True)[0]
   scaled=F.cross_entropy(z,y,weight=w*7.3)
   grad2=torch.autograd.grad(scaled,z)[0]
   assert torch.allclose(grad,grad2,atol=1e-7,rtol=1e-6)
 checkfreeze('PHASE10A');checkfreeze('PHASE10B');checkfreeze('PHASE10C')
 assert load(HERE/'analysis/phase10d_start_gate.json')['status']=='ELIGIBLE_FIXED_1_10_PILOT'
 write(HERE/'tests/dd_cost_correctness.json',dict(status='PASS',clinical_training_runs=0,optimizer_updates=0,split_count=15,train_weights=counts,
  actual_reduction='sum(weight_y * CE) / sum(weight_y)',constant_scale_cancels=True,gradient_norm_not_guaranteed_equal_to_baseline=True,
  source_sha256=sha(Path(__file__)),protected_files_verified=guard(),timestamp_utc=utc()),new=True)
 freeze('PHASE10D',[HERE/'PROTOCOL.md',HERE/'scripts/common10.py',HERE/'scripts/models10.py',HERE/'scripts/train10.py',HERE/'scripts/analyze10.py',Path(__file__),HERE/'tests/dd_cost_correctness.json'],
  only_change='DD relative cost x1.10, train-expectation-normalized balanced weighted-mean CE',maximum_runs=45,pilot_runs=3)
 print('PASS DD fixed-cost correctness; no clinical optimizer update',flush=True)

if __name__=='__main__':main()
