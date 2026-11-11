"""Re-read current history/source and lock actual model assets before synthetic work."""
from common7 import *
import subprocess,re,torch,yaml,datetime,ast

def main():
 for folder in ['scripts','configs','analysis','tests','logs','restricted_manifests']:(HERE/folder).mkdir(exist_ok=True)
 assert not (HERE/'restricted_manifests/protected_assets.json').exists()
 def git(*a):return subprocess.check_output(['git','-C',str(MIRROR),*a],text=True).strip()
 assert git('rev-parse','HEAD')==git('rev-parse','refs/remotes/origin/main')==START and not git('status','--porcelain')
 old=load(P6/'restricted_manifests/protected_assets.json')['sha256'];protected=dict(old)
 for p,d in old.items():assert sha(p)==d
 for p in P6.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
 docs=[ROOT/'AGENTS.md',ROOT/'README.md',ROOT/'CONTEXT_HANDOFF.md',ROOT/'conclusions/README.md',MAIN/'README.md']
 docs += [ROOT/'artifacts'/p for p in ['phase3b_external_ssl_20260928/PHASE3B_REPORT.md','phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md','phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md','phase3e_external_priors_20261001/PHASE3E_REPORT.md','phase4_window_dynamics_20261002/FINAL_REPORT.md']]
 docs += [P5/p for p in ['FINAL_REPORT.md','STATISTICAL_APPENDIX.md','CORRECTNESS_AND_PROVENANCE.md']]
 docs += [P6/p for p in ['FINAL_REPORT.md','STATISTICAL_APPENDIX.md','CORRECTNESS_AND_PROVENANCE.md','INDEPENDENT_VALIDATION_PROTOCOL.md','DATASET_COMPATIBILITY_AUDIT.md','NEXT_STAGE_CANDIDATES.md']]
 reviewed=[];view=[]
 for p in docs:
  if not p.exists():reviewed.append(dict(path=str(p),status='MISSING'));continue
  raw=p.read_text();protected[str(p)]=sha(p);lines=raw.splitlines();redacted=0;out=[];hide=False;level=0
  for line in lines:
   if line.startswith('#'):
    newlevel=len(line)-len(line.lstrip('#'))
    if hide and newlevel<=level:hide=False
    if re.search(r'outer.{0,30}(performance|result|metric|accuracy|AUROC|evaluation|hold.?out)|(?:final|locked).{0,15}outer',line,re.I):hide=True;level=newlevel
   if hide or (re.search(r'outer',line,re.I) and re.search(r'\b(?:0\.[0-9]{2,}|[0-9]{2}\.[0-9]+%)',line)):
    out.append('[OUTER PERFORMANCE PASSAGE NOT SURFACED]');redacted+=1
   else:out.append(line)
  reviewed.append(dict(path=str(p),status='CURRENT_FILE_READ_AND_HASHED',sha256=sha(p),bytes=p.stat().st_size,all_source_text_scanned=True,outer_passage_lines_suppressed=redacted))
  # All narrative; large numeric tables are machine-scanned, retained separately by hash.
  display=[x for x in out if not x.startswith('|') or set(x.replace('|','').strip())<=set('-: ')]
  view.append('\nFILE '+str(p)+'\n'+'\n'.join(display))
 pd.DataFrame(reviewed).to_csv(HERE/'analysis/history_review.csv',index=False)
 (HERE/'logs/history_narrative_review.txt').write_text('\n'.join(view))
 assert all(r['status']!='MISSING' for r in reviewed)
 source=[]
 for p in MAIN.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts:continue
  digest=sha(p);protected[str(p)]=digest;mirror=MIRROR/p.relative_to(ROOT)
  assert mirror.is_file() and sha(mirror)==digest,'Mainline differs from published snapshot'
  if p.suffix=='.py':ast.parse(p.read_text());kind='PYTHON_TEXT_FULLY_READ_AST_VALID'
  else:kind='FILE_HASH_VERIFIED'
  source.append(dict(path=str(p.relative_to(ROOT)),sha256=digest,review=kind,published_mirror_match=True))
 pd.DataFrame(source).to_csv(HERE/'analysis/mainline_source_audit.csv',index=False)
 lock=load(P6/'analysis/external_validation_lock.json');ckpath=Path(lock['checkpoint_path']);base=ckpath.parent.parent
 paths={'checkpoint':ckpath,'normalization_file':base/'normalization.json','canonical_config':MAIN/'configs/str01_seed42.yaml','preprocessing_config':ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/processing_config.json','official_HarNet10_checkpoint':ROOT/'artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables/model_check_point/mtl_best.mdl'}
 checks=[]
 for key,p in paths.items():
  expected=lock[key+'_sha256'];actual=sha(p);assert actual==expected,(key,actual,expected);protected[str(p)]=actual
  checks.append(dict(asset=key,path=str(p),expected_sha256=expected,actual_sha256=actual,status='PASS'))
 pd.DataFrame(checks).to_csv(HERE/'analysis/model_lock_hash_audit.csv',index=False)
 ck=torch.load(ckpath,map_location='cpu',weights_only=False);norm=load(base/'normalization.json');split=load(base/'split.json')
 for k in ['mean','std']:assert np.array_equal(ck['normalization'][k].numpy(),np.array(norm[k]))
 assert norm['normalization_sha256']==lock['normalization_values_sha256'] and set(norm['subject_ids'])==set(split['train_subjects'])
 assert len(split['train_subjects'])==205 and not set(split['train_subjects'])&set(split['validation_subjects'])
 assert all(torch.isfinite(v).all() for v in ck['model_state'].values())
 write(HERE/'restricted_manifests/model_ids.json',dict(fit_subject_ids=split['train_subjects'],historical_validation_subject_ids=split['validation_subjects'],excluded_subject_ids=split['test_subjects'],synthetic_query_ids=['synthetic_fixture_0','synthetic_fixture_1','synthetic_fixture_2'],external_query_ids=[],new_fit_ids=[],clinical_sensor_files_opened=[]))
 write(HERE/'restricted_manifests/protected_assets.json',dict(sha256=protected,git_start_sha=START))
 write(HERE/'analysis/deployment_lock.json',dict(**lock,phase7_actual_hashes_verified=True,source_model_file_sha256=sha(P6/'scripts/independent_wssl.py'),mainline_source_audit_sha256=sha(HERE/'analysis/mainline_source_audit.csv'),clinical_input_files_opened=0,new_model_forward_calls=0))
 write(HERE/'analysis/startup_integrity.json',dict(status='PASS_BEFORE_SYNTHETIC_INFERENCE',git_main=START,mirror_worktree_clean=True,mainline_files_match_published=len(source),history_documents_currently_read=len(reviewed),missing_documents=0,protected_files=len(protected),locked_assets_all_match=True,new_training_or_fit=0,current_outer_performance_surfaced=False,history_outer_exposure_permanent=True))
 write(HERE/'configs/SYNTHETIC_SOURCE_FREEZE.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.relative_to(HERE)):sha(p) for p in [HERE/'PROTOCOL.md']+list((HERE/'scripts').glob('*.py'))},new_synthetic_results_seen=False,new_external_clinical_results_seen=False))
 print('PHASE7_STARTUP_LOCK_PASS',len(protected),len(reviewed),len(source))
if __name__=='__main__':main()
