"""Label-independent Phase-3B-identical raw windows -> HarNet layer4 activations."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.signal import resample_poly
ROOT=Path('/home/zyt/deep_final/artifacts/phase3c_ssl_adaptation_20260930')
P3B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
VENDOR=P3B/'vendor/ssl-wearables';sys.path.insert(0,str(VENDOR))
import hubconf
DATA=Path('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests')
RAW=Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries')
old=np.load(P3B/'analysis/ssl_embeddings_all.npz',allow_pickle=False)
ids=list(map(str,old['subject_ids']));acts=list(map(str,old['activities']))
assert len(ids)==390 and len(acts)==11
index={s:i for i,s in enumerate(ids)};ai={a:i for i,a in enumerate(acts)}
rows=pd.concat([pd.read_csv(DATA/f'{a}.csv',dtype={'subject_id':str}) for a in acts],ignore_index=True)
assert len(rows)==4290 and rows[['subject_id','activity']].duplicated().sum()==0
X=[];keys=[];sha=hashlib.sha256()
for row in rows.itertuples():
 for wi,wn in enumerate(('LeftWrist','RightWrist')):
  raw=np.loadtxt(RAW/f'{row.subject_id}_{row.record_name}_{wn}.txt',delimiter=',',dtype=np.float32)
  acc=raw[48:,1:4];expected=row.left_length if wi==0 else row.right_length
  assert raw.shape[1]==7 and len(acc)==expected and np.isfinite(acc).all()
  acc=np.clip(acc,-3,3)
  if len(acc)<1000:
   before=(1000-len(acc))//2;after=1000-len(acc)-before
   segments=[np.pad(acc,((before,after),(0,0)),mode='edge')]
  else:segments=[acc[:1000],acc[-1000:]]
  k=(index[row.subject_id],ai[row.activity],wi)
  for seg in segments:
   w=resample_poly(seg,3,10,axis=0).T.astype(np.float32)
   assert w.shape==(3,300)
   X.append(w);keys.append(k);sha.update(w.tobytes());sha.update(np.asarray(k,dtype=np.int16).tobytes())
assert len(X)==10920
expected_sha=json.loads((P3B/'analysis/extraction_audit.json').read_text())['resampled_window_sha256']
assert sha.hexdigest()==expected_sha
checkpoint=VENDOR/'model_check_point/mtl_best.mdl'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6'
model=hubconf.harnet10(pretrained=True,my_device='cpu',class_num=2).feature_extractor.eval().cuda()
first4=torch.nn.Sequential(*list(model.children())[:4]);last=model.layer5
z=np.zeros((390,11,2,2,512,3),dtype=np.float32);counts=np.zeros((390,11,2),dtype=np.int16)
for lo in range(0,len(X),64):
 hi=min(lo+64,len(X));batch=torch.from_numpy(np.stack(X[lo:hi])).cuda()
 with torch.inference_mode():h=first4(batch).cpu().numpy()
 assert h.shape==(hi-lo,512,3) and np.isfinite(h).all()
 for j,k in enumerate(keys[lo:hi]):
  w=int(counts[k]);assert w<2
  z[k+(w,)]=h[j];counts[k]+=1
assert np.all((counts==1)|(counts==2))
with torch.inference_mode():
 recon=np.zeros((390,11,2,1024),dtype=np.float32)
 flat=z.reshape(-1,512,3);valid=(np.arange(2)[None,None,None,:]<counts[...,None]).reshape(-1)
 valid_idx=np.flatnonzero(valid)
 for lo in range(0,len(valid_idx),64):
  ix=valid_idx[lo:lo+64];out=last(torch.from_numpy(flat[ix]).cuda()).flatten(1).cpu().numpy()
  np.add.at(recon.reshape(-1,1024),ix//2,out)
 recon/=counts[...,None]
maxdiff=float(np.max(np.abs(recon-old['pretrained'])))
assert maxdiff<2e-5,maxdiff
mid=z.mean(axis=-1).sum(axis=3)/counts[...,None]
output=ROOT/'analysis/stage4_windows.npz'
np.savez_compressed(output,subject_ids=np.asarray(ids),activities=np.asarray(acts),stage4=z,counts=counts,mid=mid)
audit={'subjects':390,'activities':11,'wrist_records':8580,'windows':len(X),'window_sha256':sha.hexdigest(),'stage4_shape':list(z.shape),'mid_shape':list(mid.shape),'max_final_feature_diff_vs_phase3b':maxdiff,'cache_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
(ROOT/'analysis/stage4_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
