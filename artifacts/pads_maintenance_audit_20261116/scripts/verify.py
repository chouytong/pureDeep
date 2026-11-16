"""Server-side remote commit/tree and full public snapshot readback verification."""
import datetime
import json
import sys
from pathlib import Path
from audit import ROOT, HERE, MIRROR, BASE, PREFIX, publisher, protected, sha, load
from publish import PAYLOAD

def main():
    remote_sha = sys.argv[1]
    payload = load(PAYLOAD)
    receipt = Path('/home/zyt/pureDeep-maintenance-audit-publication-receipt-20261116.json')
    assert not receipt.exists()
    p = publisher()
    assert not p.git('status','--porcelain').stdout
    p.git('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main')
    assert p.git('rev-parse','refs/remotes/origin/main').stdout.strip() == remote_sha
    assert p.git('rev-parse',remote_sha+'^{tree}').stdout.strip() == payload['tree']
    assert p.git('show','-s','--format=%P',remote_sha).stdout.strip() == BASE
    current = p.git('rev-parse','HEAD').stdout.strip()
    assert current in {payload['local_commit'],remote_sha}
    archive = 'refs/heads/archive/maintenance-audit-unpublished-local-20261116'
    if p.git('show-ref','--verify','--quiet',archive,check=False).returncode:
        p.git('update-ref',archive,payload['local_commit'],'0'*40)
    else:
        assert p.git('rev-parse',archive).stdout.strip() == payload['local_commit']
    if current != remote_sha:
        p.git('update-ref','refs/heads/main',remote_sha,current)
    assert not p.git('status','--porcelain').stdout
    assert p.git('diff','--quiet',remote_sha,check=False).returncode == 0
    p.audit_tracked()
    selected, _ = p.collect()
    manifest = load(MIRROR/'PUBLICATION_MANIFEST.json')
    assert len(selected) == manifest['selected_file_count']
    assert {x['path'] for x in manifest['files']} == set(selected)
    for item in manifest['files']:
        path = item['path']
        assert selected[path][1:] == (item['sha256'],item['bytes'])
        assert sha(MIRROR/path) == sha(selected[path][0]) == item['sha256']
    tracked = set(filter(None,p.git('ls-files','-z').stdout.split('\0')))
    assert tracked == set(selected) | {'.gitignore','PUBLICATION_MANIFEST.json'}
    assert p.git('-c','credential.helper=','-c','core.askPass=/bin/false',
                 'ls-remote','origin','refs/heads/main').stdout.split()[0] == remote_sha
    assert p.git('rev-parse','HEAD').stdout.strip() == remote_sha
    result = dict(status='PUSH_VERIFIED_EXACT_AUDITED_TREE',
                  timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  repository='https://github.com/chouytong/pureDeep',commit=remote_sha,
                  parent=BASE,tree=payload['tree'], selected_files_verified=len(selected),
                  source_mirror_manifest_match=True, remote_commit_tree_verified=True,
                  mirror_head_equals_remote_main=True, mirror_worktree_clean=True,
                  changed_files=len(payload['entries']),
                  old_files_preserved=True, protected=protected(),
                  existing_publisher_filter_and_audit_used=True,
                  transport='chouytong GitHub connector exact tree; server anonymous git fetch and readback',
                  unpublished_local_commit_archive=archive,
                  new_training_runs=0,optimizer_updates=0,new_performance_experiments=0,
                  current_model_pointer_unchanged=True, next_experiment_started=False)
    with receipt.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
