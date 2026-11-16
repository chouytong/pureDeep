"""Use the unchanged publisher; stop before push for exact-tree connector transport."""
import json
import sys
from pathlib import Path
from audit import BASE, BASE_TREE, ROOT, HERE, MIRROR, PREFIX, publisher, protected, sha, load

PAYLOAD = Path('/home/zyt/pureDeep-maintenance-audit-publication-payload-20261116.json')
MESSAGE = 'Audit PADS maintenance state and preserve retained-model and experiment boundaries'

class ConnectorTransport(Exception):
    pass

def main():
    assert load(HERE / 'analysis/maintenance_checks.json')['status'] == 'PASS_MAINTENANCE_AUDIT'
    protected()
    assert not PAYLOAD.exists()
    p = publisher()
    original = p.git
    assert original('rev-parse','HEAD').stdout.strip() == BASE
    assert not original('status','--porcelain').stdout
    original('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main')
    assert original('rev-parse','refs/remotes/origin/main').stdout.strip() == BASE
    selected, _ = p.collect()
    old = load(MIRROR / 'PUBLICATION_MANIFEST.json')['files']
    for item in old:
        assert item['path'] in selected
        assert selected[item['path']][1:] == (item['sha256'], item['bytes'])
    changed = sorted(n for n,(_,h,z) in selected.items() if not (MIRROR/n).is_file() or sha(MIRROR/n) != h)
    assert changed and all(n.startswith(PREFIX) for n in changed)
    print(json.dumps(dict(status='PREVIEW_PASS', added_maintenance_files=len(changed),
                          historical_files_unchanged=len(old))), flush=True)
    if '--prepare' not in sys.argv:
        return
    def git(*args, check=True):
        if args[0] == 'fetch':
            return original('-c','credential.helper=','-c','core.askPass=/bin/false',
                            'fetch','--no-tags','origin','main',check=check)
        if args[0] == 'push':
            raise ConnectorTransport()
        return original(*args,check=check)
    p.git = git
    sys.argv = ['publish_to_github.py','--apply','--message',MESSAGE]
    try:
        p.main()
    except ConnectorTransport:
        pass
    else:
        raise RuntimeError('Expected connector transport handoff')
    assert not original('status','--porcelain').stdout
    local = original('rev-parse','HEAD').stdout.strip()
    assert original('rev-parse','HEAD^').stdout.strip() == BASE
    names = original('diff','--name-only','HEAD^','HEAD').stdout.splitlines()
    assert set(names) == set(changed) | {'PUBLICATION_MANIFEST.json'}
    entries = []
    for name in names:
        line = original('ls-tree','HEAD','--',name).stdout
        mode, kind, blob = line.split('\t')[0].split()
        assert kind == 'blob'
        entries.append(dict(path=name, mode=mode, type=kind, sha=blob,
                            sha256=sha(MIRROR/name), bytes=(MIRROR/name).stat().st_size))
    payload = dict(parent=BASE, base_tree=BASE_TREE, tree=original('rev-parse','HEAD^{tree}').stdout.strip(),
                   local_commit=local, message=MESSAGE, entries=entries)
    with PAYLOAD.open('x') as f:
        f.write(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)

if __name__ == '__main__':
    main()
