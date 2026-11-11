"""Complete all45 input/checkpoint audits before any model forward/export."""
from common import *
import torch,subprocess,datetime,ast

def main():
 for n in ['configs','scripts','tests','analysis','restricted_manifests','features','logs']:(HERE/n).mkdir(exist_ok=True)
 assert not (HERE/'restricted_manifests/protected_assets.json').exists()
 assert subprocess.check_output(['git','-C','/home/zyt/pureDeep-github-upload','rev-parse','HEAD'],text=True).strip()=='161b904462e754a892c3d3d2e4f9ea68b33d18c0'
 previous=load(P5/'restricted_manifests/protected_assets_phase5.json');protected=dict(previous['sha256'])
 for p,d in protected.items():assert sha(p)==d,p
 for p in P5.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
 docs=[MAIN/'README.md',ROOT/'AGENTS.md',ROOT/'CONTEXT_HANDOFF.md']+[ROOT/'artifacts'/p for p in ['phase3b_external_ssl_20260928/PHASE3B_REPORT.md','phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md','phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md','phase3e_external_priors_20261001/PHASE3E_REPORT.md','phase4_window_dynamics_20261002/FINAL_REPORT.md','wssl_contribution_20261007/PHASE_A_REPORT.md','wssl_activity_scaling_20261008/FINAL_REPORT.md','patch_str_20261008/PATCH_STR_REPORT.md']]+[P5/p for p in ['FINAL_REPORT.md','STATISTICAL_APPENDIX.md','CORRECTNESS_AND_PROVENANCE.md']]
 dr=[]
 for p in docs:
  text=p.read_text();protected[str(p)]=sha(p);dr.append(dict(path=str(p),sha256=protected[str(p)],bytes=len(p.read_bytes()),complete_document_read=True))
 pd.DataFrame(dr).to_csv(HERE/'analysis/history_sources.csv',index=False)
 assert sha(OFFICIAL)=='c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6';protected[str(OFFICIAL)]=sha(OFFICIAL)
 for p in VENDOR.rglob('*.py'):protected[str(p)]=sha(p)
 ledger=ROOT/'data/MIGRATION_MANIFEST_PRIVATE.json';protected[str(ledger)]=sha(ledger);known={str(ROOT/'data'/v['relative_path']):v['sha256'] for v in load(ledger)['files']}
 rows=[];inputrows=[];private=[];hashed={}
 def protect(p):
  p=str(Path(p).resolve())
  if p not in hashed:hashed[p]=sha(p)
  if p in known:assert hashed[p]==known[p],'Copied input changed from data migration ledger'
  protected[p]=hashed[p];return hashed[p]
 for c,i,sp,ex in splits():
  q=set(sp['validation_subjects']);train=set(sp['train_subjects']);assert not q&train and not (q|train)&set(ex)
  rec=records(q);labels={r.subject_id:r.label for r in rec}
  ins=[]
  for r in rec:
   for w,p in enumerate([r.left_path,r.right_path]):
    rp=raw_path(r,w);assert str(p) in known and str(rp) in known
    pp=protect(p);rh=protect(rp)
    # Individual allowed file only; inspect header/finite data, no statistics fitting.
    x=np.load(p,allow_pickle=False);assert x.shape==(6,r.left_length if w==0 else r.right_length) and np.isfinite(x).all()
    raw=np.loadtxt(rp,delimiter=',',dtype=np.float32);assert raw.shape==(x.shape[1]+48,7) and np.isfinite(raw).all()
    ins.append(dict(subject_id=r.subject_id,activity=r.activity,wrist=w,processed_path=str(p),processed_sha256=pp,raw_path=str(rp),raw_sha256=rh,length=x.shape[1],sampling_rate=r.left_sampling_rate if w==0 else r.right_sampling_rate))
  inputrows.append(dict(context=c,inner=i,query_count=len(q),activity_count=11,wrist_count=2,record_count=len(ins),all_input_finite=True,min_length=min(x['length'] for x in ins),max_length=max(x['length'] for x in ins),sampling_rate_min=min(x['sampling_rate'] for x in ins),sampling_rate_max=max(x['sampling_rate'] for x in ins),source_input_hash_records=len(ins)))
  for seed in [42,43,44]:
   states=[];norms=[];entry=dict(context=c,inner=i,seed=seed,query_count=len(q),fit_count=len(train),excluded_count=len(ex),asset_status='PASS_PRE_INFERENCE',sampling_rate_nominal=100.,activity_order='|'.join(ACTS),wrist_order='left|right',channel_order='AccXYZ|GyroXYZ')
   for kind in ['frozen','adapted']:
    p=stage(kind,seed,c,i);s=load(p/'split.json');n=load(p/'normalization.json');prov=load(p/'provenance.json')
    assert set(s['train_subjects'])==train and set(s['validation_subjects'])==q and set(s['test_subjects'])==set(ex)
    assert set(n['subject_ids'])==train and prov['class_names']==['PD','DD']
    pred=prediction(p/'predictions/validation.csv',q);assert all(labels[sid]==int(y) for sid,y in pred.target.items())
    ck=torch.load(p/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in ck['model_state'].values())
    for name in ['mean','std']:assert np.array_equal(ck['normalization'][name].numpy(),np.asarray(n[name]))
    d=ck['config']['data'];assert d['activities']==ACTS and d['channel_names']==cfg()['data']['channel_names'] and d['wrist_order']==['left','right'] and d['sample_rate']==100. and d['sequence_length'] is None
    norms.append(n);states.append(list(ck['model_state']))
    for name in ['checkpoints/best.pt','normalization.json','split.json','provenance.json','config.yaml','predictions/validation.csv']:entry[kind+'_'+name.replace('/','_').replace('.','_')+'_sha256']=protect(p/name)
    del ck
   assert norms[0]['normalization_sha256']==norms[1]['normalization_sha256'] and np.array_equal(norms[0]['mean'],norms[1]['mean']) and np.array_equal(norms[0]['std'],norms[1]['std'])
   assert all(k.startswith(('base.','layer5.')) for k in states[1]);assert {k[5:] for k in states[1] if k.startswith('base.')}==set(states[0])
   rows.append(entry);private.append(dict(**entry,query_subject_ids=sorted(q),fit_subject_ids=sorted(train),excluded_subject_ids=ex,input_records=ins,normalization_values_sha256=norms[0]['normalization_sha256']))
  print(json.dumps(dict(stage='asset_input_audit',context=c,inner=i,pairs_checked=3,status='PASS')),flush=True)
 pd.DataFrame(rows).to_csv(HERE/'analysis/paired_representation_integrity.csv',index=False);pd.DataFrame(inputrows).to_csv(HERE/'analysis/input_composition_audit.csv',index=False)
 write_json(HERE/'restricted_manifests/model_input_lineage.json',dict(records=private))
 write_json(HERE/'restricted_manifests/protected_assets.json',dict(sha256=protected,git_start_sha='161b904462e754a892c3d3d2e4f9ea68b33d18c0'))
 for p in (HERE/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 write_json(HERE/'configs/ANALYSIS_FREEZE.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.relative_to(HERE)):sha(p) for p in [HERE/'PROTOCOL.md']+list((HERE/'scripts').glob('*.py'))},all45_asset_pairs_pass=True,new_phase6_representation_results_seen=False,deep_training_budget=0,new_fit_budget=0,BH_families=dict(A_primary=3,A_exploratory=18,A_group=6,B_interaction=3),probability_identity_tolerance=1e-12,bootstrap_seed=SEED,bootstrap_draws=10000))
 guard();write_json(HERE/'analysis/pre_inference_asset_gate.json',dict(status='PASS_ALL45',pairs=45,raw_and_processed_inputs_verified_against_migration=True,validation_ID_labels=True,normalization_equal=True,official_checkpoint_verified=True,full_legacy_sensor_feature_arrays_decoded=False,outer_performance_read=False,historical_outer_exposure_inherited=True,new_model_forward_calls=0))
 print(json.dumps(dict(status='ALL45_PRE_INFERENCE_GATE_PASS',protected_files=len(protected))),flush=True)

if __name__=='__main__':main()
