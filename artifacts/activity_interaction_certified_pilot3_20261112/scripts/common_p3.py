"""Isolated three-run pilot; read-only imports of frozen Phase8 and Certified L1."""
from pathlib import Path
import sys,json,hashlib,time,math,os
import numpy as np
import pandas as pd
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline';P8=ROOT/'artifacts/wssl_guided_optimization_phase8_20261111';L2=ROOT/'artifacts/l1_precision_rebuild_phase9_level2_20261112';P6=ROOT/'artifacts/wssl_representation_generalization_phase6_20261111';HERE=ROOT/'artifacts/activity_interaction_certified_pilot3_20261112';PRIVATE=HERE/'restricted_manifests';START='beb57167a7b8b54792f1efed672927396fe8294d';P8COMMIT='4745549dabf34ccc89a740ad361a0410c6a93a4d';VERSION='pads_l1_lambda50_certified_v1';SEEDS=[42,43,44]
sys.path[:0]=[str(MAIN),str(L2/'scripts'),str(P8/'scripts')]
from common_l2 import ACTS,METRICS
from interaction8 import ActivityInteraction,explicit_model,fingerprint
from common8 import pred_metrics
from train_l2 import model_for,cfg,external
from data_l2 import configured_records,norm_binding
from src.utils.seed import seed_everything
import torch
from scipy.special import logsumexp

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(f):
 def fmt(v):return 'NA' if pd.isna(v) else f'{v:.6f}' if isinstance(v,float) else str(v).replace('|','/')
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(map(fmt,r))+' |\n' for r in f.itertuples(index=False,name=None))
def split():
 d=load(MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json');o=next((r for r in d['outer'] if int(r['outer_fold'])==0),None);assert o is not None,'BLOCKED_CONTEXT0_MISSING';i=next((r for r in o['inner_folds'] if int(r['inner_fold'])==0),None);assert i is not None,'BLOCKED_INNER0_MISSING';return i,list(map(str,o['test_subjects']))
def binding():return load(PRIVATE/'manifests/bindings.json')
def guard():
 d=load(PRIVATE/'manifests/protected_assets.json')['sha256']
 for p,h in d.items():assert sha(p)==h,'Protected asset changed '+p
 return len(d)
def freeze(name,paths,**extra):
 p=HERE/'configs'/f'{name}_FREEZE.json';assert not p.exists();write(p,dict(files={str(v):sha(v) for v in paths},**extra))
def token(seed,role):
 r=binding()['seeds'][str(seed)];p=PRIVATE/'tokens'/f'seed{seed}_{role}.npz';index=load(PRIVATE/'manifests/token_index.json')['packages'];expected=next(x for x in index if x['seed']==seed and x['role']==role);assert sha(p)==expected['sha256']
 with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 assert str(d['base_checkpoint_sha256'])==r['checkpoint_sha256'] and str(d['normalization_sha256'])==r['normalization_sha256'] and str(d['version'])==VERSION and d['activities'].tolist()==ACTS
 sp,_=split();ids=d['subject_ids'].astype(str);assert len(ids)==len(set(ids)) and set(ids)==set(sp['train_subjects' if role=='train' else 'validation_subjects']);assert d['bilateral'].shape==(len(ids),11,258) and d['mask'].shape==(len(ids),11) and d['mask'].dtype==bool and d['mask'].all();return d

def make_model(seed):
 r=binding()['seeds'][str(seed)];assert sha(r['checkpoint'])==r['checkpoint_sha256'];base=explicit_model(cfg(seed));base.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=False)['model_state'],strict=True);model=ActivityInteraction(base).cuda();assert sum(p.numel() for p in model.base.parameters())==143172;return model

def head(model,h,mask):
 h=h.masked_fill(~mask[...,None],0);b=model.base.backbone;return b.classifier(b.activity_aggregator(h,mask)['subject_embedding'])+b.structured_token_residual(h,mask)['structured_residual_logits']
def evaluate(model,d,candidate=True):
 model.eval();zs=[]
 with torch.inference_mode():
  for k in range(0,len(d['target']),8):
   h=torch.from_numpy(d['bilateral'][k:k+8]).cuda();m=torch.from_numpy(d['mask'][k:k+8]).cuda();z=model(h,m)['logits'] if candidate else head(model,h,m);zs.append(z.cpu().numpy())
 z=np.concatenate(zs);p=np.exp(z.astype(float)-logsumexp(z.astype(float),1,keepdims=True));return z,p
