"""Read-only registration baseline and exact historical checkpoint identities."""
from pathlib import Path
import json,hashlib,subprocess,shutil,ast,sys
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline';HERE=ROOT/'artifacts/ai_wssl_str_registration_20261116';P8=ROOT/'artifacts/wssl_guided_optimization_phase8_20261111';P3=ROOT/'artifacts/activity_interaction_certified_pilot3_20261112';L2=ROOT/'artifacts/l1_precision_rebuild_phase9_level2_20261112';UP=Path('/home/zyt/pureDeep-github-upload');START='50a319441e2391ce662646cbd3e690ea64d0e1aa'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not HERE.exists();assert subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()==START;assert not subprocess.check_output(['git','-C',str(UP),'status','--porcelain'],text=True)
 for name in ['scripts','audit','reports','analysis','restricted_manifests/originals','restricted_manifests/checkpoint_roundtrip']:(HERE/name).mkdir(parents=True,exist_ok=True)
 editable=[ROOT/'README.md',MAIN/'README.md',ROOT/'conclusions/README.md',ROOT/'conclusions/EXPERIMENT_REPORT_INDEX.csv'];protected={};before={}
 for p in editable:
  dst=HERE/'restricted_manifests/originals'/p.relative_to(ROOT);dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dst);before[str(p.relative_to(ROOT))]=sha(p)
 for p in list(MAIN.rglob('*.py'))+list((MAIN/'configs').glob('*'))+[ROOT/'conclusions/CURRENT_MODEL.json',ROOT/'AGENTS.md',ROOT/'tools/git_publish/publish_to_github.py',P8/'scripts/interaction8.py',P8/'scripts/common8.py',P8/'PROTOCOL.md',P8/'reports/ACTIVITY_INTERACTION_REPORT.md',P8/'analysis/activity_interaction_metrics.csv',P8/'analysis/activity_interaction_parameters.csv',P3/'reports/FINAL_REPORT.md',P3/'analysis/integrity.json',P3/'restricted_manifests/manifests/new_assets_sha256.json']:
  if p.is_file():protected[str(p)]=sha(p)
 p3meta=load(P3/'analysis/integrity.json');assert p3meta['CORRECTNESS']=='PASS';assert sha(P3/'restricted_manifests/manifests/new_assets_sha256.json')==p3meta['private_manifest_sha256'];p3catalog=load(P3/'restricted_manifests/manifests/new_assets_sha256.json')['sha256'];catalog=[]
 for seed in [42,43,44]:
  for c in range(5):
   for i in range(3):
    p=P8/'checkpoints'/f'ai_seed{seed}_context{c}_inner{i}/best.pt';assert p.exists();protected[str(p)]=sha(p);catalog.append(dict(checkpoint_id=f'AI-WSSL-STR:legacy-l1:seed{seed}:context{c}:inner{i}',model_id='ai_wssl_str',display_name='AI-WSSL-STR',status='Experimental Candidate',evidence_scope='legacy_l1_15split_development',seed=seed,context=c,inner=i,data_version='pads_multi_activity_v2_l1_full_length_legacy',path=str(p),sha256=sha(p),source_commit='4745549dabf34ccc89a740ad361a0410c6a93a4d',training_source_sha256=sha(P8/'scripts/interaction8.py'),base_checkpoint_path=str(ROOT/f'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed{seed}/outer_{c}/inner_{i}/checkpoints/best.pt'),report=str((P8/'reports/ACTIVITY_INTERACTION_REPORT.md').relative_to(ROOT))))
 for seed in [42,43,44]:
  p=P3/'restricted_manifests/checkpoints'/f'seed{seed}/best.pt';assert sha(p)==p3catalog[str(p.relative_to(P3))];protected[str(p)]=sha(p);receipt=load(p.parent/'complete.json');b=receipt['lineage']['base_binding'];catalog.append(dict(checkpoint_id=f'AI-WSSL-STR:certified-l1:seed{seed}:context0:inner0',model_id='ai_wssl_str',display_name='AI-WSSL-STR',status='Experimental Candidate',evidence_scope='certified_l1_single_split_pilot',seed=seed,context=0,inner=0,data_version='pads_l1_lambda50_certified_v1',path=str(p),sha256=sha(p),source_commit=START,training_source_sha256=sha(P3/'scripts/train_p3.py'),base_checkpoint_path=b['checkpoint'],base_checkpoint_sha256=b['checkpoint_sha256'],normalization_path=b['normalization'],normalization_sha256=b['normalization_sha256'],report=str((P3/'reports/FINAL_REPORT.md').relative_to(ROOT))))
  for q in [Path(b['checkpoint']),Path(b['normalization']),p.parent/'complete.json',P3/'restricted_manifests/tokens'/f'seed{seed}_train.npz']:
   protected[str(q)]=sha(q)
  assert protected[str(P3/'restricted_manifests/tokens'/f'seed{seed}_train.npz')]==p3catalog[f'restricted_manifests/tokens/seed{seed}_train.npz']
 write(HERE/'restricted_manifests/protected_assets.json',dict(sha256=protected,authorized_active_document_updates=before,source_commit=START));write(HERE/'audit/checkpoint_catalog_seed.json',dict(schema_version=1,model_id='ai_wssl_str',status='Experimental Candidate',checkpoints=catalog));write(HERE/'audit/before_state.json',dict(source_commit=START,authorized_existing_file_changes=before,protected_files=len(protected),checkpoint_records=len(catalog),current_model_pointer_sha256=sha(ROOT/'conclusions/CURRENT_MODEL.json'),new_training=0));print(json.dumps(dict(status='PREPARED_READ_ONLY_BASELINE',checkpoints=len(catalog),protected_files=len(protected),current_pointer_sha256=sha(ROOT/'conclusions/CURRENT_MODEL.json'))))
if __name__=='__main__':main()
