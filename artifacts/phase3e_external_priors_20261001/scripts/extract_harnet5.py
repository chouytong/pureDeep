import sys,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.signal import resample_poly
B=Path(__file__).resolve().parents[1];V=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables');sys.path.insert(0,str(V));import hubconf
A=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose'];D=Path('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length/manifests');R=Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries')
rows=pd.concat([pd.read_csv(D/f'{a}.csv',dtype={'subject_id':str}) for a in A]);ids=sorted(rows.subject_id.unique());ix={s:i for i,s in enumerate(ids)};X=[];keys=[]
for r in rows.itertuples():
 for wi,w in enumerate(['LeftWrist','RightWrist']):
  acc=np.loadtxt(R/f'{r.subject_id}_{r.record_name}_{w}.txt',delimiter=',',dtype=np.float32)[48:,1:4];assert len(acc)==(r.left_length if wi==0 else r.right_length) and len(acc)>=500
  acc=np.clip(acc,-3,3)
  for seg in (acc[:500],acc[-500:]):X.append(resample_poly(seg,3,10,axis=0).T.astype(np.float32));keys.append((ix[r.subject_id],A.index(r.activity),wi))
X=np.stack(X);keys=np.array(keys);assert X.shape==(17160,3,150)
torch.manual_seed(20261001);m=hubconf.harnet5(False);cp=B/'vendor/phase3e_harnet5.mdl';sd=torch.load(cp,map_location='cpu',weights_only=False);sd={k.split('.',1)[1]:v for k,v in sd.items()};current=m.feature_extractor.state_dict();enc={k[len('feature_extractor.'):]:v for k,v in sd.items() if k.startswith('feature_extractor.')};assert set(enc)==set(current);m.feature_extractor.load_state_dict(enc,strict=True);m=m.feature_extractor.eval().cuda();m.requires_grad_(False)
out=np.zeros((390,11,2,512),np.float32);cnt=np.zeros((390,11,2),int)
with torch.inference_mode():
 for lo in range(0,len(X),64):
  hi=min(lo+64,len(X));f=m(torch.from_numpy(X[lo:hi]).cuda()).flatten(1).cpu().numpy();assert f.shape==(hi-lo,512) and np.isfinite(f).all();np.add.at(out,tuple(keys[lo:hi].T),f);np.add.at(cnt,tuple(keys[lo:hi].T),1)
out/=cnt[...,None];np.savez_compressed(B/'analysis/harnet5_embeddings_all.npz',subject_ids=ids,activities=A,pretrained=out,random=out,window_counts=cnt)
audit={'subjects':390,'windows':len(X),'feature_dim':512,'encoder_params':sum(p.numel() for p in m.parameters()),'all_encoder_keys_loaded':True,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'window_sha256':hashlib.sha256(X.tobytes()+keys.tobytes()).hexdigest(),'labels_used':False,'feature_std':float(out.std()),'random_field_note':'unused duplicate for generic cache; no HarNet5 random experiment'};(B/'analysis/harnet5_extraction_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(audit)
