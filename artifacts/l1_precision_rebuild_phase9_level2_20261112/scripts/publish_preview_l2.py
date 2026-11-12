"""Read-only preview through the unchanged server selective publisher."""
from common_l2 import *
import importlib.util,hashlib

def main():
 assert load(HERE/'analysis/final_integrity.json')['status']=='PASS_SCIENTIFIC_DELIVERABLES'
 spec=importlib.util.spec_from_file_location('selective_publisher',ROOT/'tools/git_publish/publish_to_github.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub);assert pub.git('rev-parse','HEAD').stdout.strip()==START and not pub.git('status','--porcelain').stdout
 selected,skipped=pub.collect();changed=[];prefix='artifacts/'+HERE.name+'/'
 for rel,(source,digest,size) in selected.items():
  p=pub.UPLOAD/rel
  if not p.exists() or sha(p)!=digest:changed.append(rel)
 assert changed and all(p.startswith(prefix) for p in changed)
 stale=set(filter(None,pub.git('ls-files','-z').stdout.split('\0')))-set(selected)-{'.gitignore','PUBLICATION_MANIFEST.json'};assert not stale
 assert not any('restricted_manifests' in Path(n).parts or Path(n).suffix in ['.npz','.npy','.pt'] for n in selected)
 for r in load(HERE/'analysis/source_output_manifest.json')['files']:assert sha(HERE/r['path'])==r['sha256']
 result=dict(status='PASS_READ_ONLY_PUBLICATION_PREVIEW',source_commit=START,new_public_files=len(changed),selected_files=len(selected),private_signals_IDs_predictions_checkpoints_certificates_excluded=True,original_publisher_unchanged=True,stale_deletions=0,protected_files_unchanged=guard(),publication_not_yet_performed=True)
 p=Path('/home/zyt/pureDeep-l1level2-publication-preview-20261112.json');assert not p.exists();write(p,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
