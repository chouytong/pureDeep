"""No-label, in-memory strict format adapter; no file discovery or dataset access."""
from common7 import *
from dataclasses import dataclass
from fractions import Fraction
from scipy.signal import resample_poly
import torch,yaml
from src.datasets.preprocessing import remove_acceleration_trend

class Incompatible(ValueError):pass
class Partial(Incompatible):pass
@dataclass
class Stream:
 values:np.ndarray
 channels:tuple
 sampling_rate:float
 acc_unit:str
 gyro_unit:str
 wrist:str
 device:str
 timestamps:np.ndarray
 time_unit:str='s'
 frame:str='unknown'
 placement:str='unknown'

def canonical_stream(s:Stream,side):
 if s.wrist!=side or s.placement!='wrist' or s.frame!='PADS-device-axes-confirmed':raise Incompatible('Unverified wrist/placement/axis mapping')
 if not s.device:raise Incompatible('Missing independent device provenance')
 x=np.asarray(s.values);t=np.asarray(s.timestamps,dtype=np.float64)
 if x.ndim!=2 or x.shape[1]!=6 or len(x)<3 or not np.isfinite(x).all():raise Incompatible('Expected finite [T,6]')
 if len(s.channels)!=6 or len(set(s.channels))!=6 or set(s.channels)!=set(CHANNELS):raise Incompatible('Missing/duplicate/unmapped channels')
 if t.shape!=(len(x),) or not np.isfinite(t).all() or s.time_unit not in {'s','ms'}:raise Incompatible('Invalid timestamp schema')
 t=t*(.001 if s.time_unit=='ms' else 1.);dt=np.diff(t);fs=float(s.sampling_rate)
 if not np.isfinite(fs) or fs<=0 or fs<30 or fs>1000:raise Incompatible('Unsupported sampling rate')
 if not (dt>0).all() or not np.allclose(dt,1/fs,rtol=.001,atol=1e-7):raise Incompatible('Irregular timestamps require a separately verified mapping')
 x=np.asarray(x[:,[s.channels.index(k) for k in CHANNELS]],dtype=np.float64).copy()
 if s.acc_unit=='m/s^2':x[:,:3]/=9.80665
 elif s.acc_unit!='g':raise Incompatible('Unknown acceleration unit')
 if s.gyro_unit=='deg/s':x[:,3:]*=np.pi/180
 elif s.gyro_unit!='rad/s':raise Incompatible('Unknown angular velocity unit')
 # Rational anti-aliased conversion is fixed, never learned from labels.
 ratio=Fraction(100/fs).limit_denominator(1000)
 if abs(float(ratio)-100/fs)>1e-10:raise Incompatible('Non-representable fixed resampling ratio')
 if fs!=100:x=resample_poly(x,ratio.numerator,ratio.denominator,axis=0)
 return x.astype(np.float32)

def adapt_subject(activities):
 if set(activities)!=set(ACTS):raise Partial('Missing or unexpected activity: no invented activity padding')
 out=[];raw=[];reports=[];pre=load(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/processing_config.json')
 for activity in ACTS:
  record=activities[activity]
  if set(record)!={'left','right'}:raise Partial('Missing wrist: no copying opposite side')
  if record['left'].device==record['right'].device:raise Incompatible('Two wrist streams require independent device provenance')
  pair=[canonical_stream(record[w],w) for w in ['left','right']]
  if len(pair[0])!=len(pair[1]):raise Incompatible('Unmatched activity duration after deterministic resampling')
  if len(pair[0])<=48+2:raise Incompatible('Not enough samples after locked trim')
  trimmed=np.stack([p[48:].T for p in pair]);raw.append(trimmed.copy());filtered=[]
  options=pre['accelerometer']
  for signal in trimmed:
   value,report=remove_acceleration_trend(signal,regularization=options['lambda'],rho=options['rho'],max_iterations=options['max_iterations'],absolute_tolerance=options['absolute_tolerance'],relative_tolerance=options['relative_tolerance'])
   if not report.converged:raise Incompatible('Locked L1 preprocessing did not converge')
   filtered.append(value.astype(np.float32));reports.append(dict(converged=True,iterations=report.iterations))
  out.append(np.stack(filtered))
 return dict(local=out,raw=raw,lengths=np.array([x.shape[-1] for x in out]),preprocessing_reports=reports)

def external_windows(subject):
 windows=[];mapping=[]
 for a,signal in enumerate(subject['raw']):
  for w in range(2):
   acc=np.clip(signal[w,:3].T,-3,3)
   if len(acc)<1000:
    before=(1000-len(acc))//2;segments=[np.pad(acc,((before,1000-len(acc)-before),(0,0)),mode='edge')]
   else:segments=[acc[:1000],acc[-1000:]]
   for seg in segments:windows.append(resample_poly(seg,3,10,axis=0).T.astype(np.float32));mapping.append((a,w))
 return np.stack(windows),mapping

def collate(subjects,mean,std):
 maximum=max(int(s['lengths'].max()) for s in subjects);x=np.zeros((len(subjects),11,2,6,maximum),np.float32)
 for j,s in enumerate(subjects):
  for a,v in enumerate(s['local']):x[j,a,:,:,:v.shape[-1]]=(v-mean[a])/std[a]
 return [torch.from_numpy(x),torch.ones((len(subjects),11,2)),torch.ones((len(subjects),11),dtype=torch.bool),torch.from_numpy(np.stack([s['lengths'] for s in subjects]))]

def decision(probabilities):return (probabilities[:,1]>.5).astype(np.int64)

def validate_real_gate(authorization):
 required=['data_grant','subject_overlap_clearance','protocol_mapping','device_mapping','label_separation']
 if any(authorization.get(k) is not True for k in required):raise PermissionError('External clinical inference not authorized/ready')

class FrozenRunner:
 def __init__(self):
  assert load(HERE/'analysis/startup_integrity.json')['status']=='PASS_BEFORE_SYNTHETIC_INFERENCE'
  lock=load(HERE/'analysis/deployment_lock.json');assert sha(lock['checkpoint_path'])==lock['checkpoint_sha256']
  source=P6/'scripts/independent_wssl.py';assert sha(source)==lock['source_model_file_sha256']
  import importlib.util
  spec=importlib.util.spec_from_file_location('phase6_explicit_model',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  config=yaml.safe_load((MAIN/'configs/str01_seed42.yaml').read_text());checkpoint=torch.load(lock['checkpoint_path'],map_location='cpu',weights_only=False)
  self.model=module.IndependentWSSL(config).eval().cuda();self.model.load_state_dict(checkpoint['model_state'],strict=True)
  self.mean=checkpoint['normalization']['mean'].numpy();self.std=checkpoint['normalization']['std'].numpy();assert (self.std>0).all()
  vendor=ROOT/'artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables';sys.path.insert(0,str(vendor));import hubconf
  official=hubconf.harnet10(pretrained=True,my_device='cpu',class_num=2).eval()
  rawstate=torch.load(vendor/'model_check_point/mtl_best.mdl',map_location='cpu',weights_only=False)
  for k,v in official.feature_extractor.state_dict().items():assert torch.equal(v,rawstate['module.feature_extractor.'+k])
  self.encoder=official.feature_extractor.eval().cuda()
  for m in [self.model,self.encoder]:
   for p in m.parameters():p.requires_grad_(False)
  self.last_encoder_batches=[];self.last_classifier_batches=[]
 def fingerprint(self):return {prefix+k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for prefix,m in [('model:',self.model),('encoder:',self.encoder)] for k,v in m.state_dict().items()}
 def synthetic_infer(self,subjects,batch_size=8):
  # Explicit synthetic-only API this round. No file IO or clinical loader.
  if not subjects or batch_size<1:raise ValueError('Empty or invalid inference batch')
  self.model.eval();self.encoder.eval();features=[];self.last_encoder_batches=[];self.last_classifier_batches=[]
  with torch.inference_mode():
   for subject in subjects:
    windows,mapping=external_windows(subject);pieces=[]
    for k in range(0,len(windows),64):
     part=windows[k:k+64];self.last_encoder_batches.append(len(part));pieces.append(self.encoder(torch.from_numpy(part).cuda()).flatten(1).cpu().numpy())
    values=np.concatenate(pieces);h=np.zeros((11,2,1024),np.float32);counts=np.zeros((11,2),np.int32)
    for v,(a,w) in zip(values,mapping):h[a,w]+=v;counts[a,w]+=1
    features.append(h/counts[...,None])
   result=[]
   for k in range(0,len(subjects),batch_size):
    q=subjects[k:k+batch_size];self.last_classifier_batches.append(len(q));inputs=[v.cuda() for v in collate(q,self.mean,self.std)];ssl=torch.from_numpy(np.stack(features[k:k+batch_size])).float().cuda()
    before=[v.clone() for v in inputs]+[ssl.clone()];logit=self.model(*inputs,ssl_features=ssl)['logits']
    assert all(torch.equal(a,b) for a,b in zip(before,inputs+[ssl])) and torch.isfinite(logit).all();result.append(logit.cpu().numpy())
  return np.concatenate(result)
