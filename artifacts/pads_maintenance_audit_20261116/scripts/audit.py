"""Server-only maintenance audit; no model execution or optimizer calls."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path('/home/zyt/deep_final')
HERE = ROOT / 'artifacts/pads_maintenance_audit_20261116'
MIRROR = Path('/home/zyt/pureDeep-github-upload')
BASE = '8720b45fe82f0037b10ba779588c1bd6281be911'
BASE_TREE = '748bec911c629ae09a473caaceaf4f6967d578ee'
PREFIX = 'artifacts/pads_maintenance_audit_20261116/'
POINTER_SHA = 'a30c1262c7e37793050a73bfd1c0ba6f8f68c0cc020a21cda1434a0f5f05a820'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def publisher():
    spec = importlib.util.spec_from_file_location('existing_publisher', ROOT / 'tools/git_publish/publish_to_github.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def protected():
    registration = ROOT / 'artifacts/ai_wssl_str_registration_20261116'
    protected_assets = load(registration / 'restricted_manifests/protected_assets.json')['sha256']
    for path, digest in protected_assets.items():
        if sha(path) != digest:
            raise RuntimeError('Protected asset changed; preserve and investigate')
    private_assets = load(registration / 'restricted_manifests/new_assets_sha256.json')['sha256']
    for path, digest in private_assets.items():
        if sha(registration / path) != digest:
            raise RuntimeError('Registration private asset changed')
    public_assets = load(registration / 'analysis/source_output_manifest.json')['files']
    for item in public_assets:
        if sha(registration / item['path']) != item['sha256']:
            raise RuntimeError('Registration public artifact changed')
    if sha(ROOT / 'conclusions/CURRENT_MODEL.json') != POINTER_SHA:
        raise RuntimeError('Current model pointer changed')
    records = load(ROOT / 'mainline/models/ai_wssl_str_checkpoints.json')['checkpoints']
    for item in records:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Registered historical checkpoint changed')
    scopes = {}
    for item in records:
        scopes[item['evidence_scope']] = scopes.get(item['evidence_scope'], 0) + 1
    assert len(records) == 48 and sorted(scopes.values()) == [3, 45]
    models = load(ROOT / 'mainline/models/registry.json')['models']
    assert {m['model_id']:m['status'] for m in models} == {
        'str01':'Matched Baseline', 'frozen_wssl':'Retained Model',
        'ai_wssl_str':'Experimental Candidate'}
    return dict(protected_history_files=len(protected_assets),
                registration_private_assets=len(private_assets),
                registration_public_assets=len(public_assets),
                registered_checkpoint_files=len(records), checkpoint_scopes=scopes,
                current_model_pointer_sha256=POINTER_SHA,
                retained_model='Frozen WSSL-STR', candidate_status='Experimental Candidate')

def main():
    assert Path(sys.executable).resolve() == Path('/home/zyt/envs/mfam_analysis/bin/python').resolve()
    p = publisher()
    assert not p.git('status','--porcelain').stdout
    assert p.git('rev-parse','HEAD').stdout.strip() == BASE
    assert p.git('rev-parse','HEAD^{tree}').stdout.strip() == BASE_TREE
    remote = p.git('-c','credential.helper=','-c','core.askPass=/bin/false',
                   'ls-remote','origin','refs/heads/main').stdout.split()[0]
    assert remote == BASE
    p.audit_tracked()
    selected, skipped = p.collect()
    manifest = load(MIRROR / 'PUBLICATION_MANIFEST.json')
    for record in manifest['files']:
        name = record['path']
        assert name in selected
        assert selected[name][1:] == (record['sha256'], record['bytes'])
        assert sha(MIRROR / name) == record['sha256']
    additions = sorted(set(selected) - {x['path'] for x in manifest['files']})
    assert all(name.startswith(PREFIX) for name in additions)
    tracked = set(filter(None, p.git('ls-files','-z').stdout.split('\0')))
    assert tracked == {x['path'] for x in manifest['files']} | {'.gitignore','PUBLICATION_MANIFEST.json'}
    result = dict(status='PASS_MAINTENANCE_AUDIT',
                  timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  execution_host='server', python_executable=sys.executable,
                  source_commit=BASE, source_tree=BASE_TREE, remote_main=remote,
                  historical_manifest_files_verified=len(manifest['files']),
                  historical_source_mirror_manifest_match=True,
                  pre_existing_uncommitted_changes=False,
                  existing_publisher_filters_and_tracked_audit='PASS',
                  protected=protected(),
                  new_training_runs=0, optimizer_updates=0,
                  new_performance_experiments=0, new_normalization_fits=0,
                  new_threshold_or_hyperparameter_searches=0,
                  raw_data_or_signals_read=False,
                  publication='requires separate exact-tree readback receipt',
                  limitations=[
                      'This maintenance audit verifies stored files, not a repeat of model performance.',
                      'Certified AI pilot remains context0/inner0 seeds42/43/44 only; remaining42 runs not authorized.',
                      'Historical old-L1 45 runs do not substitute for full Certified L1 validation.',
                      'All3 Certified pilot DD Recall deltas are below -0.01; no model promotion.',
                      'CURRENT_MODEL metrics retain their historical data scope; they are not relabeled Certified L1.'])
    (HERE / 'analysis').mkdir(exist_ok=True)
    with (HERE / 'analysis/maintenance_checks.json').open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
