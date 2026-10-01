"""Official frozen BioPM neural acc encoder only; no unencoded gravity branch."""
import sys,hashlib,json,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
B=Path(__file__).resolve().parents[1];sys.path[:0]=[str(B/'vendor/deps'),str(B/'vendor/phase3e_biopm_official')]
from biopm.inference import load_pretrained
from biopm.model import TimeSeriesTransformer
from biopm.features import per_axis_mean_std
from biopm.preprocessing import PreprocessConfig,resample_to_target_fs,bandpass_filter,detect_zero_crossings,pack_window
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
A=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose'];D=Path('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests');R=Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries');cfg=PreprocessConfig(ori_fs=100,target_fs=30)
rows=pd.concat([pd.read_csv(D/f'{act}.csv',dtype={'subject_id':str}) for act in A]);ids=sorted(rows.subject_id.unique());ids=ids[:2] if a.smoke else ids;ix={s:i for i,s in enumerate(ids)}
cp=B/'vendor/phase3e_biopm_official/checkpoints/biopm_50mr.pt';m=load_pretrained(checkpoint_path=str(cp)).encoder_acc;m0=load_pretrained(checkpoint_path=str(cp));m0.load_checkpoint(str(cp),strict=True,verbose=True);m.eval().cuda().requires_grad_(False)
torch.manual_seed(20261001);random=TimeSeriesTransformer().eval().cuda().requires_grad_(False)
X=[];keys=[];info=[]
for ri,r in enumerate(rows.itertuples()):
 if r.subject_id not in ix:continue
 for wi,w in enumerate(['LeftWrist','RightWrist']):
  raw=np.loadtxt(R/f'{r.subject_id}_{r.record_name}_{w}.txt',delimiter=',',dtype=np.float64);acc=raw[48:,1:4];assert np.isfinite(acc).all() and len(acc)==(r.left_length if wi==0 else r.right_length)
  t=np.arange(len(acc))/100;rs,tt,_=resample_to_target_fs(t,acc,np.zeros(len(acc)),30);f=bandpass_filter(rs,.5,12,30,order=6);padding=0
  if len(f)<300:
   padding=300-len(f);bef=padding//2;f=np.pad(f,((bef,padding-bef),(0,0)),mode='edge');segments=[f]
  elif len(f)==300:segments=[f]
  else:segments=[f[:300],f[-300:]]
  for n,seg in enumerate(segments):
   mn,mi,pos,_,_=detect_zero_crossings(seg,np.arange(300)/30,cfg);x=pack_window(mn,mi,pos,cfg);assert x.shape==(192,38)
   X.append(x);keys.append((ix[r.subject_id],A.index(r.activity),wi));info.append(dict(subject_id=r.subject_id,activity=r.activity,wrist=w,context=n,tokens=len(mn),edge_padded_samples=padding,empty_tokens=len(mn)==0))
 if ri%100==0:print('processed',ri,'contexts',len(X),flush=True)
X=np.stack(X);keys=np.array(keys);features={};cnt=np.zeros((len(ids),11,2),int)
for k in keys:cnt[tuple(k)]+=1
assert (cnt>=1).all()
for name,encoder in [('pretrained',m),('random',random)]:
 out=np.zeros((len(ids),11,2,384),np.float32)
 with torch.inference_mode():
  for lo in range(0,len(X),32):
   hi=min(lo+32,len(X));x=torch.from_numpy(X[lo:hi]).cuda();patch=x[:,:,:32];pos=x[:,:,32];add=x[:,:,33:];valid=~torch.isnan(patch).any(-1);tok=encoder(patch,pos,torch.zeros_like(pos),add);feat=per_axis_mean_std(tok,add[:,:,0],valid);assert feat.shape==(hi-lo,384) and torch.isfinite(feat).all();np.add.at(out,tuple(keys[lo:hi].T),feat.cpu().numpy())
 out/=cnt[...,None];features[name]=out
p=B/'analysis'/('biopm_embeddings_smoke.npz' if a.smoke else 'biopm_embeddings_all.npz');np.savez_compressed(p,subject_ids=ids,activities=A,**features,window_counts=cnt)
audit={'subjects':len(ids),'contexts':len(X),'empty_contexts':sum(z['empty_tokens'] for z in info),'padded_contexts':sum(z['edge_padded_samples']>0 for z in info),'tokens_min':min(z['tokens'] for z in info),'tokens_max':max(z['tokens'] for z in info),'feature_dim':384,'encoder_params':sum(p.numel() for p in m.parameters()),'encoder_all_weights_loaded_strict':True,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'token_tensor_sha256':hashlib.sha256(X.tobytes()+keys.tobytes()).hexdigest(),'feature_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'label_access':False,'gravity_used':False,'random_seed':20261001,'feature_std':{k:float(v.std()) for k,v in features.items()}}
(B/'analysis'/('biopm_extraction_smoke_audit.json' if a.smoke else 'biopm_extraction_audit.json')).write_text(json.dumps(audit,indent=2)+'\n');pd.DataFrame(info).to_csv(B/'analysis'/('biopm_contexts_smoke.csv' if a.smoke else 'biopm_contexts.csv'),index=False);print(audit,flush=True)
