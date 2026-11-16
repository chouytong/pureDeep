"""Load/forward/restore only. No gradients, optimizer construction or training."""
from pathlib import Path
from copy import deepcopy
import sys,json,hashlib,ast,subprocess,time,io
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline';HERE=ROOT/'artifacts/ai_wssl_str_registration_20261116';P8=ROOT/'artifacts/wssl_guided_optimization_phase8_20261111';P3=ROOT/'artifacts/activity_interaction_certified_pilot3_20261112'
sys.path[:0]=[str(MAIN),str(P8/'scripts')]
import numpy as np
import torch
from src.models import build_model as original_factory
from src.models.pure_deep import build_pure_deep_subject
from src.models.wssl import ResidualSSLSubject
from src.models.registry import list_models,get_model_spec,load_registered_config,build_registered_model
from src.models.registered_checkpoint import load_registered_checkpoint,save_registered_checkpoint,list_checkpoints,file_sha256
from src.models.ai_wssl_str import _ActivityInteractionCore
from src.engine.ai_wssl_str import prepare_token_training,validate_recipe
from src.utils.seed import seed_everything
from interaction8 import ActivityInteraction,explicit_model,fingerprint

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rejected(fn):
 try:fn()
 except (ValueError,FileExistsError,RuntimeError):return True
 raise AssertionError('Expected rejection did not occur')
def state_equal(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
def main():
 torch.set_num_threads(4);seed_everything(42,True);start=time.perf_counter();checks=[];rows=[]
 specs=list_models();assert [s['model_id'] for s in specs]==['str01','frozen_wssl','ai_wssl_str'];assert [s['status'] for s in specs]==['Matched Baseline','Retained Model','Experimental Candidate'];assert get_model_spec('AI-WSSL-STR')['model_id']=='ai_wssl_str';assert get_model_spec('ai_wssl_str')['status']=='Experimental Candidate';assert rejected(lambda:get_model_spec('MFAM'));checks.append('three parallel model names and correct lifecycle statuses')
 old_tree=ast.parse((P8/'scripts/interaction8.py').read_text());new_tree=ast.parse((MAIN/'src/models/ai_wssl_str.py').read_text());old_cls=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='ActivityInteraction');new_cls=next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='_ActivityInteractionCore');new_cls.name='ActivityInteraction';assert ast.dump(old_cls,include_attributes=False)==ast.dump(new_cls,include_attributes=False);checks.append('original attention/residual class AST identical except name')
 entries=list_checkpoints();assert len(entries)==48 and sum(e['evidence_scope']=='legacy_l1_15split_development' for e in entries)==45 and sum(e['evidence_scope']=='certified_l1_single_split_pilot' for e in entries)==3
 for e in entries:assert file_sha256(e['path'])==e['sha256']
 for seed in [42,43,44]:
  cfg=load_registered_config('AI-WSSL-STR',seed);basecfg=load_registered_config('STR-01',seed);validate_recipe(cfg);seed_everything(seed,True);old=ActivityInteraction(explicit_model(basecfg));seed_everything(seed,True);new=build_registered_model('AI-WSSL-STR',cfg);assert state_equal(old.state_dict(),new.state_dict());assert sum(p.numel() for p in new.parameters())==210694 and sum(p.numel() for p in new.parameters() if p.requires_grad)==67522
  new.train();assert not new.base.training and all(not p.requires_grad for p in new.base.parameters());new.eval();del new,old
  ident=f'AI-WSSL-STR:certified-l1:seed{seed}:context0:inner0';e=next(x for x in entries if x['checkpoint_id']==ident);model,info=load_registered_checkpoint('AI-WSSL-STR',checkpoint_id=ident,device='cuda');old=ActivityInteraction(explicit_model(basecfg)).cuda().eval();old.load_state_dict(torch.load(e['path'],map_location='cuda',weights_only=False)['model_state'],strict=True);prior=fingerprint(model);baseprior=fingerprint(model.base)
  package=P3/'restricted_manifests/tokens'/f'seed{seed}_train.npz'
  with np.load(package,allow_pickle=False) as z:h=torch.from_numpy(z['bilateral'][:8].copy()).cuda();mask=torch.from_numpy(z['mask'][:8].copy()).cuda()
  with torch.inference_mode():
   expected=old(h,mask)['logits'];actual=model(h,mask)['logits'];assert torch.equal(expected,actual) and torch.equal(actual,model.forward_tokens(h,mask)['logits']);assert torch.equal(actual,model(h,mask)['logits']);assert torch.isfinite(actual).all() and actual.shape==(8,2)
   invalid=mask.clone();invalid[:,2]=False;bad=h.clone();bad[:,2]=float('nan');assert torch.equal(model(h,invalid)['logits'],model(bad,invalid)['logits']);assert rejected(lambda:model(h,torch.zeros_like(mask)))
  restored=HERE/'restricted_manifests/checkpoint_roundtrip'/f'AI-WSSL-STR_seed{seed}_registration_restore_test.pt';digest=save_registered_checkpoint('AI-WSSL-STR',model,restored,cfg,provenance=dict(seed=seed,context=0,inner=0,data_version=e['data_version'],base_checkpoint_sha256=e['base_checkpoint_sha256'],normalization_sha256=e['normalization_sha256'],registration_test_only=True,optimizer_updates=0));fresh,meta=load_registered_checkpoint('ai_wssl_str',checkpoint_path=restored,expected_sha256=digest,device='cuda')
  with torch.inference_mode():assert torch.equal(actual,fresh(h,mask)['logits'])
  assert meta['status']=='Experimental Candidate' and fingerprint(fresh)==prior and fingerprint(model.base)==baseprior and fingerprint(model)==prior;assert rejected(lambda:save_registered_checkpoint('AI-WSSL-STR',model,restored,cfg,provenance=dict(seed=seed)));assert rejected(lambda:load_registered_checkpoint('STR-01',checkpoint_path=e['path'],expected_sha256=e['sha256']));assert rejected(lambda:load_registered_checkpoint('AI-WSSL-STR',checkpoint_path=e['path'],expected_sha256='0'*64));wrong=deepcopy(cfg);wrong['experiment']['seed']=44 if seed!=44 else 42;assert rejected(lambda:load_registered_checkpoint('AI-WSSL-STR',checkpoint_id=ident,config=wrong));changed=deepcopy(cfg);changed['model']['ai_wssl_str']['heads']=8;assert rejected(lambda:build_registered_model('AI-WSSL-STR',changed))
  row=dict(seed=seed,checkpoint_id=ident,source_checkpoint_sha256=e['sha256'],train_only_test_samples=8,all_state_keys_compatible=True,original_vs_registered_logits_bitwise=True,repeat_bitwise=True,reload_bitwise=True,frozen_base_unchanged=True,optimizer_updates=0,parameters=210694,trainable_parameters=67522,status='PASS');rows.append(row);print(json.dumps(row),flush=True)
  if seed==42:
   raw=torch.randn(2,11,2,6,2000,device='cuda');wm=torch.ones(2,11,2,dtype=torch.bool,device='cuda');am=torch.ones(2,11,dtype=torch.bool,device='cuda');lengths=torch.tensor([976,2000],device='cuda')[:,None].expand(2,11).clone();ssl=torch.randn(2,11,2,1024,device='cuda')
   with torch.inference_mode():
    full=model(raw,wm,am,lengths,ssl_features=ssl)['logits'];tokens,valid=model.encode_tokens(raw,wm,am,lengths,ssl_features=ssl);assert torch.equal(full,model(tokens,valid)['logits']);bad=raw.clone();bad[0,:,:,:,976:]=999;assert torch.equal(full,model(bad,wm,am,lengths,ssl_features=ssl)['logits']);assert rejected(lambda:model(raw,wm,am,lengths))
   assert model.base.ssl_features is None and fingerprint(model)==prior;checks.append('full signal + explicit frozen HarNet to token/logit equivalence and padding invariance')
  del model,fresh,old;torch.cuda.empty_cache()
 # Load a legacy AI checkpoint without mixing it with Certified clinical tokens.
 ident='AI-WSSL-STR:legacy-l1:seed42:context0:inner0';model,info=load_registered_checkpoint('AI-WSSL-STR',checkpoint_id=ident,device='cuda');assert info['data_version']=='pads_multi_activity_v2_l1_full_length_legacy';h=torch.randn(2,11,258,device='cuda');mask=torch.ones(2,11,dtype=torch.bool,device='cuda')
 with torch.inference_mode():assert torch.isfinite(model(h,mask)['logits']).all()
 del model;checks.append('legacy checkpoint identity strict-load and synthetic forward')
 # Original factories stay unchanged; compare their exact initial states to named registry paths.
 for name in ['STR-01','Frozen WSSL-STR']:
  cfg=load_registered_config(name,42);seed_everything(42,True);old=(original_factory(cfg) if name=='STR-01' else ResidualSSLSubject(original_factory(cfg)));seed_everything(42,True);new=build_registered_model(name,cfg);assert state_equal(old.state_dict(),new.state_dict());checks.append(name+' original factory initialization bitwise unchanged')
 # No optimizer is instantiated: validation-only training entry binds the existing packages.
 cfg=load_registered_config('AI-WSSL-STR',42);e=next(x for x in entries if x['checkpoint_id']=='AI-WSSL-STR:certified-l1:seed42:context0:inner0');base=e['base_checkpoint_path'];tokenindex=P3/'restricted_manifests/manifests/token_index.json';model,packages,prov=prepare_token_training(cfg,token_index=tokenindex,base_checkpoint=base,base_sha256=e['base_checkpoint_sha256'],context=0,inner=0);assert len(packages['train']['target'])==205 and len(packages['validation']['target'])==105;checks.append('dedicated trainer validates data/source/normalization and unchanged recipe with zero updates')
 commands=[['model_cli.py','--list'],['model_cli.py','--model','AI-WSSL-STR','--checkpoint-id',e['checkpoint_id'],'--input',str(P3/'restricted_manifests/tokens/seed42_train.npz'),'--limit','8','--device','cuda','--output',str(HERE/'restricted_manifests/cli_inference_test.npz')],['train_ai_wssl_str.py','--config',str(MAIN/'configs/ai_wssl_str_seed42.yaml'),'--token-index',str(tokenindex),'--base-checkpoint',base,'--base-sha256',e['base_checkpoint_sha256'],'--context','0','--inner','0','--validate-only']]
 for i,args in enumerate(commands):
  result=subprocess.run([sys.executable,'-B',str(MAIN/'scripts'/args[0]),*args[1:]],capture_output=True,text=True);write(HERE/'restricted_manifests'/f'cli_test{i}.json',dict(args=args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));assert result.returncode==0,result.stderr
 checks.append('named model list/load/infer CLI and --validate-only training CLI passed')
 write(HERE/'analysis/model_load_forward_restore_tests.json',dict(status='PASS',checks=checks,seed_results=rows,legacy_checkpoint_records=45,certified_checkpoint_records=3,model_names=[s['display_name'] for s in specs],model_statuses=[s['status'] for s in specs],checkpoint_restore_files_are_test_only=True,optimizer_updates=0,new_training_runs=0,new_validation_performance_evaluation=False,clinical_forward_inputs='first8 train tokens per seed only',raw_signal_test='synthetic only',elapsed_seconds=time.perf_counter()-start));print('ALL_REGISTRATION_TESTS_PASS',flush=True)
if __name__=='__main__':main()
