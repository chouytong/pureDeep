"""Transport wrapper for unchanged selective publisher; no authentication secrets."""
import importlib.util,sys,json,hashlib,subprocess,base64
from pathlib import Path
ROOT=Path('/home/zyt/deep_final');UP=Path('/home/zyt/pureDeep-github-upload');START='4da80813a98586acf23a89507359e4ca93689289'
PREFIX='artifacts/l1_precision_rebuild_phase9_level2_20261112/'
spec=importlib.util.spec_from_file_location('publisher',ROOT/'tools/git_publish/publish_to_github.py');pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
original=pub.git
def anonymous(*args,check=True):
 if args and args[0]=='fetch':return original('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main',check=check)
 if args and args[0]=='push':raise ReadyForConnectedGitHub()
 return original(*args,check=check)
class ReadyForConnectedGitHub(Exception):pass

def main():
 assert original('rev-parse','HEAD').stdout.strip()==START
 assert not original('status','--porcelain').stdout
 pub.git=anonymous
 anonymous('fetch','origin');assert original('rev-parse','refs/remotes/origin/main').stdout.strip()==START
 selected,skipped=pub.collect();changed=[]
 for rel,(source,digest,size) in selected.items():
  target=UP/rel
  if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:changed.append(rel)
 assert changed and all(p.startswith(PREFIX) for p in changed),changed
 stale=set(filter(None,original('ls-files','-z').stdout.split('\0')))-set(selected)-{'.gitignore','PUBLICATION_MANIFEST.json'}
 assert not stale
 print(json.dumps(dict(status='PREVIEW_PASS',changed_new_l1level2_files=len(changed),selected_files=len(selected),private_arrays_excluded=True,largest_l1level2_file=max(v[2] for k,v in selected.items() if k.startswith(PREFIX)),skipped=skipped)),flush=True)
 if '--prepare' not in sys.argv:return
 sys.argv=['publish_to_github.py','--apply','--message','Certify independent L1 preprocessing and revalidate WSSL and STR (Phase 9 Level 2)']
 try:pub.main()
 except ReadyForConnectedGitHub:pass
 else:raise AssertionError('Transport interception did not occur')
 assert not original('status','--porcelain').stdout
 local=original('rev-parse','HEAD').stdout.strip();parent=original('rev-parse','HEAD^').stdout.strip();assert parent==START
 names=original('diff','--name-only','HEAD^','HEAD').stdout.splitlines();assert all(n.startswith(PREFIX) or n=='PUBLICATION_MANIFEST.json' for n in names)
 entries=[]
 for name in names:
  line=original('ls-tree','HEAD','--',name).stdout;mode,kind,digest=line.split('\t')[0].split();assert kind=='blob'
  raw=(UP/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==digest
  try:content=raw.decode('utf-8');encoding='utf-8'
  except UnicodeDecodeError:content=base64.b64encode(raw).decode('ascii');encoding='base64'
  entries.append(dict(path=name,mode=mode,type='blob',expected_blob_sha=digest,content=content,encoding=encoding))
 payload=dict(parent=parent,base_tree=original('rev-parse','HEAD^^{tree}').stdout.strip(),tree=original('rev-parse','HEAD^{tree}').stdout.strip(),local_commit=local,elements=entries)
 path=Path('/home/zyt/pureDeep-l1level2-publication-payload-20261112.json');assert not path.exists();path.write_text(json.dumps(payload,ensure_ascii=False))
 print(json.dumps(dict(status='READY_FOR_CONNECTED_GITHUB',local_commit=local,expected_tree=payload['tree'],changed_files=len(entries),payload_bytes=path.stat().st_size,payload_path=str(path))),flush=True)

if __name__=='__main__':main()
