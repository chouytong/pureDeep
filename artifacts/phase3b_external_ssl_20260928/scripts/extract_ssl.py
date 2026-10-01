"""Label-independent raw PADS Acc -> official frozen HarNet10 and random control embeddings."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.signal import resample_poly
ROOT=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928')
VENDOR=ROOT/'vendor/ssl-wearables';sys.path.insert(0,str(VENDOR))
import hubconf
DATA=Path('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests')
RAW=Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries')
ACTIVITIES=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose']
P=argparse.ArgumentParser();P.add_argument('--smoke',action='store_true');args=P.parse_args()
rows=pd.concat([pd.read_csv(DATA/f'{a}.csv',dtype={'subject_id':str}) for a in ACTIVITIES],ignore_index=True)
assert len(rows)==4290 and rows[['subject_id','activity']].duplicated().sum()==0
ids=sorted(rows.subject_id.unique());ids=ids[:2] if args.smoke else ids
index={s:i for i,s in enumerate(ids)};activity_index={a:i for i,a in enumerate(ACTIVITIES)}
windows=[];keys=[];raw_sha=hashlib.sha256();padded=0;two_windows=0;clipped_values=0;total_values=0
for row in rows.itertuples():
 if row.subject_id not in index:continue
 for wrist_index,wrist_name in enumerate(('LeftWrist','RightWrist')):
  path=RAW/f'{row.subject_id}_{row.record_name}_{wrist_name}.txt'
  raw=np.loadtxt(path,delimiter=',',dtype=np.float32)
  assert raw.ndim==2 and raw.shape[1]==7
  expected=row.left_length if wrist_index==0 else row.right_length
  acc=raw[48:,1:4]
  assert len(acc)==expected and np.isfinite(acc).all()
  total_values+=acc.size;clipped_values+=int(((acc<-3)|(acc>3)).sum())
  acc=np.clip(acc,-3,3)
  k=(index[row.subject_id],activity_index[row.activity],wrist_index)
  if len(acc)<1000:
   before=(1000-len(acc))//2;after=1000-len(acc)-before
   segments=[np.pad(acc,((before,after),(0,0)),mode='edge')];padded+=1
  else:
   segments=[acc[:1000],acc[-1000:]];two_windows+=1
  for segment in segments:
   x=resample_poly(segment,3,10,axis=0).T.astype(np.float32)
   assert x.shape==(3,300) and np.isfinite(x).all()
   windows.append(x);keys.append(k)
   raw_sha.update(x.tobytes());raw_sha.update(np.asarray(k,dtype=np.int16).tobytes())
X=np.stack(windows);keys=np.asarray(keys,dtype=np.int16)
checkpoint=VENDOR/'model_check_point/mtl_best.mdl'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6'
torch.manual_seed(20260928)
random=hubconf.harnet10(pretrained=False,class_num=2).feature_extractor.eval().cuda()
pretrained=hubconf.harnet10(pretrained=True,my_device='cpu',class_num=2).feature_extractor.eval().cuda()
for model in (pretrained,random):
 for param in model.parameters():param.requires_grad_(False)
features={}
for name,model in [('pretrained',pretrained),('random',random)]:
 out=np.zeros((len(ids),11,2,1024),dtype=np.float32);counts=np.zeros((len(ids),11,2),dtype=np.int16)
 with torch.inference_mode():
  for lo in range(0,len(X),64):
   hi=min(lo+64,len(X));tensor=torch.from_numpy(X[lo:hi]).cuda();encoded=model(tensor).flatten(1).cpu().numpy()
   assert encoded.shape==(hi-lo,1024) and np.isfinite(encoded).all()
   np.add.at(out,tuple(keys[lo:hi].T),encoded)
   np.add.at(counts,tuple(keys[lo:hi].T),1)
 assert np.all(counts>=1)
 out/=counts[...,None]
 features[name]=out
 print(name,'windows',len(X),'mean',float(out.mean()),'std',float(out.std()),'maxabs',float(np.abs(out).max()),flush=True)
assert features['pretrained'].shape==(len(ids),11,2,1024)
output=ROOT/'analysis'/('ssl_embeddings_smoke.npz' if args.smoke else 'ssl_embeddings_all.npz')
np.savez_compressed(output,subject_ids=np.asarray(ids),activities=np.asarray(ACTIVITIES),pretrained=features['pretrained'],random=features['random'],window_counts=counts)
audit={'official_commit':'150550ea5d41800229c95e36f88f5bf0d2e7cf04','official_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'subjects':len(ids),'activities':len(ACTIVITIES),'wrist_records':len(ids)*11*2,'windows':len(X),'short_records_edge_padded':padded,'long_records_two_windows':two_windows,'values_clipped_before_resampling':clipped_values,'raw_acc_values':total_values,'resampled_window_sha256':raw_sha.hexdigest(),'input_shape':[3,300],'feature_dim':1024,'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'pretrained_feature_std':float(features['pretrained'].std()),'random_feature_std':float(features['random'].std())}
(ROOT/'analysis'/('extraction_smoke_audit.json' if args.smoke else 'extraction_audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
