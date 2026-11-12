"""Verify and align the mirror to an exact-tree connected GitHub publication."""
import sys,json,datetime,importlib.util
from pathlib import Path
ROOT=Path('/home/zyt/deep_final');HERE=ROOT/'artifacts/l1_precision_rebuild_phase9_level2_20261112'
sys.path.insert(0,str(HERE/'scripts'));from common_l2 import guard,sha,load
spec=importlib.util.spec_from_file_location('publisher',ROOT/'tools/git_publish/publish_to_github.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
def main():
 api_sha=sys.argv[1];payload=load('/home/zyt/pureDeep-l1level2-publication-payload-20261112.json');local=payload['local_commit']
 assert not pub.git('status','--porcelain').stdout
 pub.git('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main')
 assert pub.git('rev-parse','refs/remotes/origin/main').stdout.strip()==api_sha
 assert pub.git('rev-parse',api_sha+'^{tree}').stdout.strip()==payload['tree']
 assert pub.git('show','-s','--format=%P',api_sha).stdout.strip()==payload['parent']
 current=pub.git('rev-parse','HEAD').stdout.strip();assert current in {local,api_sha}
 archive='refs/heads/archive/l1level2-unpublished-local-20261112'
 if pub.git('show-ref','--verify','--quiet',archive,check=False).returncode!=0:pub.git('update-ref',archive,local,'0'*40)
 else:assert pub.git('rev-parse',archive).stdout.strip()==local
 if current==local:pub.git('update-ref','refs/heads/main',api_sha,local)
 assert not pub.git('status','--porcelain').stdout
 remote=pub.git('-c','credential.helper=','-c','core.askPass=/bin/false','ls-remote','origin','refs/heads/main').stdout.split()[0]
 assert remote==pub.git('rev-parse','HEAD').stdout.strip()==api_sha
 pub.audit_tracked();selected,skipped=pub.collect()
 for name,(source,digest,size) in selected.items():assert sha(pub.UPLOAD/name)==digest and sha(source)==digest
 for record in load(HERE/'analysis/source_output_manifest.json')['files']:assert sha(HERE/record['path'])==record['sha256']
 private_manifest=HERE/'restricted_manifests/manifests/new_scientific_assets_sha256.json';private_assets=load(private_manifest)['sha256']
 for name,digest in private_assets.items():assert sha(HERE/name)==digest
 protected_count=guard();result=dict(status='PUSH_VERIFIED_EXACT_AUDITED_TREE',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository='https://github.com/chouytong/pureDeep',commit=api_sha,parent=payload['parent'],tree=payload['tree'],server_artifact_directory=str(HERE),new_l1level2_public_files=len(payload['elements'])-1,changed_files_including_publication_manifest=len(payload['elements']),all_selected_source_mirror_hashes_match=True,selected_files=len(selected),protected_files_unchanged=protected_count,new_private_scientific_assets_verified=len(private_assets),new_private_manifest_sha256=sha(private_manifest),public_source_output_manifest_unchanged=True,private_directories_not_published=['features','restricted_manifests','predictions','data','checkpoints'],original_publisher_checks_used=True,transport='connected GitHub immutable blob/tree/commit and CAS ref; exact tree verified; anonymous git readback',unpublished_local_commit_preserved_in_archive=archive,new_deep_training=90,new_final_readout_fits=0,new_train_only_scalers=15,Level2_completed=True,next_stage_started=False)
 p=Path('/home/zyt/pureDeep-l1level2-publication-receipt-20261112.json');assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
