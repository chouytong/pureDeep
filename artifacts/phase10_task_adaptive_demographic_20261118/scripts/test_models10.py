"""Source/data gates and synthetic-only correctness before formal Phase10 training."""
from common10 import *
from models10 import *
from train10 import config,TraceAdamW
from src.utils.seed import seed_everything
from torch.nn import functional as F
import io,gc

def structure():
 encoder=official_encoder().eval();x=torch.zeros(2,3,300);rows=[];blocks=[]
 with torch.no_grad():
  for name,m in encoder.named_children():
   incoming=tuple(x.shape[1:]);x=m(x);out=tuple(x.shape[1:])
   row=dict(stage=name,input_channels=incoming[0],input_length=incoming[1],output_channels=out[0],output_length=out[1],
            parameters=sum(p.numel() for p in m.parameters()),conv_layers=sum(isinstance(v,nn.Conv1d) for v in m.modules()),
            residual_blocks=sum(v.__class__.__name__=='ResBlock' for v in m.modules()),BN_layers=sum(isinstance(v,nn.BatchNorm1d) for v in m.modules()))
   rows.append(row);blocks.append(f'    {name}["{name}: {incoming[0]}×{incoming[1]} → {out[0]}×{out[1]} / Conv {row["conv_layers"]}, ResBlock {row["residual_blocks"]}, BN {row["BN_layers"]}"]')
 assert tuple(x.shape[1:])==(1024,1) and sum(p.numel() for p in encoder.parameters())==10457408
 assert rows[-1]['residual_blocks']==0 and rows[-2]['residual_blocks']==2
 if (HERE/'analysis/harnet_structure.csv').exists():
  assert pd.read_csv(HERE/'analysis/harnet_structure.csv').to_dict('records')==rows
 else:csvout('harnet_structure.csv',rows)
 diagram='```mermaid\nflowchart LR\n    raw["Raw Acc 3×300 at30Hz"] --> layer1\n'+'\n'.join(blocks)+'\n    layer1 --> layer2 --> layer3 --> layer4 --> layer5 --> pooled["1024D → wrist residual → original WSSL"]\n```\n'
 text='# 实际HarNet10结构审计\n\n'+table(rows)+'\n\n'+diagram+'\n\n每个stage由Conv、0或2个pre-activation ResBlock、BN/ReLU及anti-aliased downsampling构成。ResBlock为BN→ReLU→Conv→BN→ReLU→Conv加identity残差。最后stage不是ResBlock。官方预训练完整feature extractor10,457,408参数，逐state与原官方checkpoint比对。FT-L1解冻layer5卷积，FT-L2解冻layer4/5卷积；所有BN affine及running buffers固定。\n\n历史Phase3C仅layer5 LR2e-5且BN affine可训练；本轮LR1e-5并冻结affine，另有Certified数据/norm与fresh下游状态差异，不是网络结构创新。HarNet原始Raw Acc路径不改变。'
 if (HERE/'reports/HARNET_STRUCTURE.md').exists():assert (HERE/'reports/HARNET_STRUCTURE.md').read_text()==text+'\n'
 else:report('HARNET_STRUCTURE.md',text)
 return rows

def main():
 seed_everything(42,True);torch.set_num_threads(4);structure();tests=[]
 # Verify every certified processed file against its independently certified manifest.
 ready=load(P9/'restricted_manifests/manifests/data_ready.json');data=load(ready['manifest_path'])
 for r in data['records']:
  assert r['version']==VERSION and r['status']=='PASS'
  assert sha(r['output_path'])==r['output_sha256']
 tests.append('all8580 Certified processed-file hashes verified')
 c=config(42);seed_everything(42,True);reference=model_for('wssl',c).cuda().eval();seed_everything(42,True);model=PartialHarNet(c,1).cuda().eval()
 assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.base.state_dict().items())
 x=torch.randn(2,11,2,6,2000,device='cuda');x[0,:,:,:,976:]=0
 wm=torch.ones(2,11,2,dtype=torch.bool,device='cuda');am=torch.ones(2,11,dtype=torch.bool,device='cuda');lengths=torch.tensor([976,2000],device='cuda')[:,None].expand(2,11).clone();y=torch.tensor([0,1],device='cuda')
 prefix=torch.randn(2,11,2,2,512,3,device='cuda');counts=torch.ones(2,11,2,dtype=torch.long,device='cuda');counts[1]=2
 model.prefix=prefix;model.counts=counts
 with torch.no_grad():
  expected=model(x,wm,am,lengths)['logits'];reference.ssl_features=model.base.ssl_features
  assert torch.equal(expected,reference(x,wm,am,lengths)['logits'])
  changed=x.clone();changed[0,:,:,:,976:]=999
  assert torch.equal(expected,model(changed,wm,am,lengths)['logits'])
  before=model.prefix.clone();model.prefix=model.prefix.clone();model.prefix[0,:,:,1]=999
  assert torch.equal(expected,model(x,wm,am,lengths)['logits']);model.prefix=before
 tests+=['same-seed original WSSL initialization bitwise unchanged','zero residual original logits exact','padding/invalid second window invariance']
 initial=fingerprint(model.encoder);opt=TraceAdamW(parameter_groups(model,'FT-L1'))
 for step in range(2):
  model.train();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x,wm,am,lengths)['logits'],y);loss.backward();opt.step()
  assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in model.encoder.modules() if isinstance(m,nn.BatchNorm1d))
 changed=fingerprint(model.encoder)
 assert any(changed[k]!=h for k,h in initial.items() if k.startswith('layer5.') and 'running' not in k and 'num_batches' not in k)
 for k,h in initial.items():
  if not k.startswith('layer5.') or any(t in k for t in ['running_mean','running_var','num_batches_tracked','layer5.1.weight','layer5.1.bias']):assert changed[k]==h,k
 assert max(z['gradient_L2'] for z in opt.trace if z['group']=='HarNet_selected_convolutions')>0
 model.eval()
 with torch.no_grad():
  expected=model(x,wm,am,lengths)['logits']
  altered=x.clone();altered[0,:,:,:,976:]=999
  assert torch.equal(expected,model(altered,wm,am,lengths)['logits'])
  model.prefix=prefix.clone();model.prefix[0,:,:,1]=999
  assert torch.equal(expected,model(x,wm,am,lengths)['logits']);model.prefix=prefix
 buf=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),buf);buf.seek(0)
 saved=torch.load(buf,map_location='cuda',weights_only=False);fresh=PartialHarNet(c,1).cuda().eval();fresh.load_state_dict(saved['model']);fresh.prefix=prefix;fresh.counts=counts
 freshopt=TraceAdamW(parameter_groups(fresh,'FT-L1'));freshopt.load_state_dict(saved['optimizer'])
 with torch.no_grad():assert torch.equal(expected,fresh(x,wm,am,lengths)['logits'])
 tests+=['only selected conv/ordinary classifier parameters update','BN running and affine unchanged after2 synthetic updates','encoder nonzero gradient after zero-residual warmup','model/optimizer roundtrip exact']
 # Weighted-mean CE is invariant to a common scale, unlike fixed-denominator CE.
 z=torch.tensor([[.2,-.8],[.1,.7],[-.3,.4]],dtype=torch.float64,requires_grad=True);target=torch.tensor([0,1,1]);w=torch.tensor([.7,1.6],dtype=torch.float64)
 a=F.cross_entropy(z,target,weight=w);b=(-(torch.log_softmax(z,1)[torch.arange(3),target])*w[target]).sum()/w[target].sum()
 assert torch.allclose(a,b,rtol=0,atol=1e-15)
 ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(F.cross_entropy(z,target,weight=7*w),z)[0];assert torch.allclose(ga,gb,rtol=0,atol=1e-15)
 tests+=['balanced CE actual batch-weight denominator verified','global weight renormalization gradient invariance verified']
 # Fake forbidden IO is rejected before any attempted signal read.
 from train10 import fixed_scope
 import src.datasets.manifest as lm
 sp,ex,fit,query,records,n,entry,expected_files,original=fixed_scope(0,0)
 try:
  try:lm._load_array('/forbidden/excluded.npy',',')
  except PermissionError:pass
  else:raise AssertionError('allowlist did not reject')
 finally:lm._load_array=original
 tests.append('excluded signal path rejected before IO')
 del model,fresh,opt,freshopt,reference;gc.collect();torch.cuda.empty_cache()
 result=dict(status='PASS',checks=tests,synthetic_optimizer_updates=2,formal_training_runs=0,official_init_verified=True,
             BN_running_and_affine_frozen=True,data_version=VERSION,verified_processed_records=len(data['records']),
             original_model_pointer_unchanged=True,protected_files_verified=guard(),source_sha256=sha(Path(__file__)))
 write(HERE/'tests/unfreezing_correctness.json',result,new=True);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
