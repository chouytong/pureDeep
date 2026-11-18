"""Independent Phase10 candidates; never modifies the mainline factory."""
from common10 import *
import torch
from torch import nn
VENDOR=ROOT/'artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables'
OFFICIAL=VENDOR/'model_check_point/mtl_best.mdl'

def official_encoder():
 assert sha(OFFICIAL)=='c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6'
 sys.path.insert(0,str(VENDOR));import hubconf
 # Official construction must not consume the downstream seed's RNG stream.
 with torch.random.fork_rng(devices=[]):
  official=hubconf.harnet10(pretrained=True,my_device='cpu',class_num=2)
 state=torch.load(OFFICIAL,map_location='cpu',weights_only=False)
 for k,v in official.feature_extractor.state_dict().items():assert torch.equal(v,state['module.feature_extractor.'+k])
 return official.feature_extractor

class PartialHarNet(nn.Module):
 def __init__(self,config,depth=1):
  super().__init__();assert depth in [1,2]
  self.base=model_for('wssl',config);self.encoder=official_encoder();self.depth=depth;self.num_classes=2
  names=list(self.encoder._modules);assert names==['layer1','layer2','layer3','layer4','layer5']
  self.selected=names[-depth:];self.prefix=None;self.counts=None
  for name,module in self.encoder.named_children():
   for p in module.parameters():p.requires_grad_(name in self.selected)
  for m in self.encoder.modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):
    m.eval()
    for p in m.parameters():p.requires_grad_(False)
  self.encoder.eval()
 def train(self,mode=True):
  super().train(mode);self.encoder.eval() # Conv gradients enabled; all BN behavior fixed.
  return self
 def forward(self,inputs,wrist_mask,activity_mask=None,activity_lengths=None):
  w=self.prefix;c=self.counts
  channels,length=(512,3) if self.depth==1 else (256,15)
  if w is None or c is None or w.shape!=(len(inputs),11,2,2,channels,length):raise ValueError('Matched official frozen-prefix windows required')
  valid=(torch.arange(2,device=w.device)<c[...,None]).reshape(-1);ix=valid.nonzero(as_tuple=False).squeeze(1)
  encoded=w.reshape(-1,channels,length).index_select(0,ix)
  for name in self.selected:encoded=getattr(self.encoder,name)(encoded)
  encoded=encoded.flatten(1);assert encoded.shape[1]==1024
  out=encoded.new_zeros((len(inputs)*22,1024)).index_add(0,ix//2,encoded)
  self.base.ssl_features=(out/c.reshape(-1,1).to(out.dtype)).reshape(len(inputs),11,2,1024)
  return self.base(inputs,wrist_mask,activity_mask,activity_lengths)

class DemographicResidual(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.num_classes=2
  for p in self.base.parameters():p.requires_grad_(False)
  self.base.eval();self.demographic_mlp=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU())
  self.demo_output=nn.Linear(16,2,bias=False);nn.init.zeros_(self.demo_output.weight)
 def train(self,mode=True):super().train(mode);self.base.eval();return self
 def forward(self,demo,base_logits):
  if demo.ndim!=2 or demo.shape[1]!=4 or base_logits.shape!=(len(demo),2):raise ValueError('Expected four whitelisted demographic values and two frozen logits')
  logits=base_logits.detach()+self.demo_output(self.demographic_mlp(demo))
  return dict(logits=logits,probabilities=torch.softmax(logits,1))

def prefix4(allowed):
 # Label-independent official frozen-prefix cache, not a supervised embedding.
 protected=load(P9/'restricted_manifests/manifests/protected_assets.json')['sha256'];out={}
 for sid in sorted(allowed):
  p=P6/'features/external_by_subject'/f'{sid}.npz';assert str(p) in protected and sha(p)==protected[str(p)]
  with np.load(p,allow_pickle=False) as z:
   assert str(z['subject_id'])==sid and z['activities'].tolist()==ACTS and z['wrist_order'].tolist()==SIDES
   w=z['layer4_windows'].copy();counts=z['window_counts'].copy()
   assert w.shape==(11,2,2,512,3) and counts.shape==(11,2) and np.isfinite(w).all()
   assert np.all((counts==1)|(counts==2))
   out[sid]=(w,counts)
 return out

def fingerprint(model):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
def parameter_groups(model,kind):
 if kind.startswith('FT-'):
  return [{'name':'classifier','params':[p for p in model.base.parameters() if p.requires_grad],'lr':2e-4},
          {'name':'HarNet_selected_convolutions','params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':1e-5}]
 return [{'name':'demographic_branch' if kind=='DEMO' else 'classifier','params':[p for p in model.parameters() if p.requires_grad],'lr':2e-4}]

