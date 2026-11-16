"""Unchanged selective publisher plus exact audited-tree connector transport."""
from pathlib import Path
import datetime,importlib.util,json,sys
from build_catalog import ROOT,HERE,OUT,BASE,sha,baseprotection

UP=Path('/home/zyt/pureDeep-github-upload')
PREFIX='artifacts/pads_experiment_catalog_20261116/'
CSV='conclusions/EXPERIMENT_RESULTS_ALL_20261116.csv'
PAYLOAD=Path('/home/zyt/pureDeep-experiment-catalog-publication-payload-20261116.json')
RECEIPT=Path('/home/zyt/pureDeep-experiment-catalog-publication-receipt-20261116.json')
MESSAGE='Consolidate PADS development experiments into one sourced CSV with sample standard deviations'

def load(p):return json.loads(Path(p).read_text())
def publisher():
 s=importlib.util.spec_from_file_location('publisher',ROOT/'tools/git_publish/publish_to_github.py')
 p=importlib.util.module_from_spec(s);s.loader.exec_module(p);return p
def allowed(n):return n.startswith(PREFIX) or n==CSV
class Connector(Exception):pass

def prepare():
 assert not PAYLOAD.exists()
 assert load(HERE/'analysis/validation.json')['status']=='PASS'
 assert load(HERE/'analysis/independent_verification.json')['status']=='PASS'
 assert sha(OUT)==load(HERE/'analysis/validation.json')['csv_sha256']
 baseprotection();p=publisher();original=p.git
 assert not original('status','--porcelain').stdout
 assert original('rev-parse','HEAD').stdout.strip()==BASE
 original('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main')
 assert original('rev-parse','refs/remotes/origin/main').stdout.strip()==BASE
 old=load(UP/'PUBLICATION_MANIFEST.json')['files'];selected,skipped=p.collect()
 for r in old:
  assert r['path'] in selected
  assert selected[r['path']][1:]==(r['sha256'],r['bytes'])
 changed=[n for n,(_,h,z) in selected.items() if not (UP/n).exists() or sha(UP/n)!=h]
 assert changed and all(allowed(n) for n in changed),changed
 print(json.dumps(dict(status='PREVIEW_PASS',changed=len(changed),historical_files_unchanged=len(old))),flush=True)
 if '--prepare' not in sys.argv:return
 def git(*args,check=True):
  if args[0]=='fetch':return original('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main',check=check)
  if args[0]=='push':raise Connector()
  return original(*args,check=check)
 p.git=git;sys.argv=['publish_to_github.py','--apply','--message',MESSAGE]
 try:p.main()
 except Connector:pass
 else:raise AssertionError('Expected exact-tree transport handoff')
 assert not original('status','--porcelain').stdout
 assert original('rev-parse','HEAD^').stdout.strip()==BASE
 names=original('diff','--name-only','HEAD^','HEAD').stdout.splitlines()
 assert set(names)==set(changed)|{'PUBLICATION_MANIFEST.json'}
 entries=[]
 for n in names:
  mode,kind,digest=original('ls-tree','HEAD','--',n).stdout.split('\t')[0].split()
  assert kind=='blob'
  entries.append(dict(path=n,mode=mode,type=kind,sha=digest,sha256=sha(UP/n),bytes=(UP/n).stat().st_size))
 payload=dict(parent=BASE,base_tree=original('rev-parse','HEAD^^{tree}').stdout.strip(),
              local_commit=original('rev-parse','HEAD').stdout.strip(),tree=original('rev-parse','HEAD^{tree}').stdout.strip(),message=MESSAGE,entries=entries)
 with PAYLOAD.open('x') as f:json.dump(payload,f,indent=2)
 print(json.dumps(payload,indent=2))

def verify(remote):
 assert not RECEIPT.exists();p=publisher();payload=load(PAYLOAD)
 assert not p.git('status','--porcelain').stdout
 p.git('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main')
 assert p.git('rev-parse','refs/remotes/origin/main').stdout.strip()==remote
 assert p.git('rev-parse',remote+'^{tree}').stdout.strip()==payload['tree']
 assert p.git('show','-s','--format=%P',remote).stdout.strip()==BASE
 current=p.git('rev-parse','HEAD').stdout.strip();assert current in {remote,payload['local_commit']}
 archive='refs/heads/archive/experiment-catalog-unpublished-local-20261116'
 assert p.git('show-ref','--verify','--quiet',archive,check=False).returncode!=0
 p.git('update-ref',archive,payload['local_commit'],'0'*40)
 if current!=remote:p.git('update-ref','refs/heads/main',remote,current)
 p.audit_tracked();selected,skipped=p.collect();manifest=load(UP/'PUBLICATION_MANIFEST.json')
 assert {r['path'] for r in manifest['files']}==set(selected)
 for r in manifest['files']:
  n=r['path'];assert selected[n][1:]==(r['sha256'],r['bytes'])
  assert sha(UP/n)==sha(selected[n][0])==r['sha256']
 for r in load(HERE/'restricted_manifests/input_sources.json'):
  assert sha(ROOT/r['path'])==r['sha256']
 assert sha(OUT)==load(HERE/'analysis/validation.json')['csv_sha256']
 assert not p.git('status','--porcelain').stdout
 assert p.git('rev-parse','HEAD').stdout.strip()==remote
 assert p.git('-c','credential.helper=','-c','core.askPass=/bin/false','ls-remote','origin','refs/heads/main').stdout.split()[0]==remote
 result=dict(status='PUSH_VERIFIED_EXACT_AUDITED_TREE',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  repository='https://github.com/chouytong/pureDeep',commit=remote,parent=BASE,tree=payload['tree'],
  selected_files_verified=len(selected),source_mirror_manifest_match=True,remote_commit_tree_verified=True,
  mirror_head_equals_remote_main=True,mirror_worktree_clean=True,csv_sha256=sha(OUT),
  csv_rows=load(HERE/'analysis/validation.json')['rows'],changed_files=len(payload['entries']),
  protected=baseprotection(),original_publisher_filters_and_audits_used=True,
  transport='chouytong GitHub connector exact tree; server anonymous git fetch/readback',
  unpublished_local_commit_archive=archive,new_training_runs=0,new_model_inference_runs=0,
  individual_records_exported=False,current_model_pointer_unchanged=True)
 with RECEIPT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':
 if '--verify' in sys.argv:verify(sys.argv[sys.argv.index('--verify')+1])
 else:prepare()
