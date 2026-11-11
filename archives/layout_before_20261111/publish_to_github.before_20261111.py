#!/usr/bin/env python3
"""Selective, non-deleting server snapshot -> Git commit -> GitHub push.

Run on zyt's server. No credentials are stored here. No training is performed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

PRIMARY = Path('/home/zyt/deep_final')
SECONDARY = Path('/home/zyt/MFAM')
UPLOAD = Path('/home/zyt/pureDeep-github-upload')
REMOTE = 'https://github.com/chouytong/pureDeep.git'
MAX_BYTES = 5 * 1024 * 1024
ALLOWED = {'.py', '.sh', '.md', '.rst', '.txt', '.yaml', '.yml', '.toml',
           '.json', '.csv', '.tsv', '.png', '.svg', '.pdf', '.sha256'}
SPECIAL = {'SHA256SUMS', 'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING',
           'Dockerfile', 'Makefile', 'requirements.txt', '.gitignore'}
PRUNE = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
         '.venv', 'venv', 'node_modules', 'vendor', 'site-packages',
         'restricted_manifests', 'domain_raw', 'domain_adaptation',
         'checkpoints', 'predictions', 'features', 'representations',
         'logs', 'smoke', 'quarantine', 'historical_outer_quarantine',
         'historical_outer_artifacts', 'quarantined_outer_artifacts'}
BLOCKED_NAMES = {'github-https-token', 'id_rsa', 'id_ed25519',
                 'github-str-ed25519', 'sync_inventory.json'}
ROW_IDS = re.compile(r'^(?:(?:train|validation|test|outer|raw)_)?(?:subject|participant|patient|sample|pair)(?:_?(?:id|ids)|_a|_b)?$|^(?:subjects|participants|patients|pid|email|phone)$', re.I)
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|(?m:^[ \t]*-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----)|https://[^/\s]+:[^@\s]+@github\.com)')
IGNORE = '''# Publishing mirror: datasets, fitted arrays, weights and secrets stay on server.
**/.git/
**/__pycache__/
**/.pytest_cache/
**/.venv/
**/venv/
**/node_modules/
**/vendor/
/datasets/
**/checkpoints/
**/predictions/
**/restricted_manifests/
**/domain_raw/
**/domain_adaptation/
**/logs/
**/smoke/
**/quarantine/
*.pyc
*.pt
*.pth
*.ckpt
*.mdl
*.onnx
*.safetensors
*.npy
*.npz
*.h5
*.hdf5
*.pkl
*.pickle
*.joblib
*.parquet
*.zip
*.tar
*.gz
*.tmp
*.log
*.jsonl
.DS_Store
._*
.env*
*.pem
github-https-token
github-str-ed25519
'''

def git(*args, check=True):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    p = subprocess.run(['git', '-C', str(UPLOAD), *args], capture_output=True,
                       text=True, env=env)
    if check and p.returncode:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p

def subject_payload(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if ROW_IDS.search(str(key)) and isinstance(value, (list, dict, str)):
                return True
            if re.search(r'(train|validation|test)_subjects$', str(key)) and isinstance(value, list):
                return True
            if subject_payload(value):
                return True
    elif isinstance(obj, list):
        return any(subject_payload(v) for v in obj)
    return False

def reason(path, rel):
    if path.is_symlink():
        return 'symlink'
    if path.name == '.gitignore':
        return 'mirror_gitignore_is_managed'
    if path.name.startswith(('._', '.env')) or path.name in BLOCKED_NAMES:
        return 'secret_or_temporary_name'
    if path.stat().st_size > MAX_BYTES:
        return 'over_5MiB'
    if path.suffix.lower() not in ALLOWED and path.name not in SPECIAL:
        return 'binary_cache_log_or_unsupported_format'
    # Canonical subject-level split membership is protocol, not raw sensor data.
    protocol_split = 'splits' in rel.parts and path.suffix in {'.json', '.yaml', '.yml'}
    if path.suffix.lower() in {'.csv', '.tsv'}:
        try:
            with path.open() as stream:
                header = next(csv.reader(stream, delimiter='\t' if path.suffix == '.tsv' else ','), [])
            if any(ROW_IDS.search(k) for k in header):
                return 'subject_level_table_or_dataset_manifest'
        except (UnicodeError, csv.Error):
            return 'unreadable_table'
    if path.suffix.lower() == '.json' and not protocol_split:
        try:
            if subject_payload(json.loads(path.read_text())):
                return 'subject_level_json_or_fitted_normalization'
        except (UnicodeError, ValueError):
            return 'invalid_json'
    return None

def collect():
    selected, skipped = {}, {}
    def skip(key):
        skipped[key] = skipped.get(key, 0) + 1
    for root, prefix in [(PRIMARY, Path()), (SECONDARY, Path('MFAM'))]:
        for current, dirs, files in os.walk(root, followlinks=False):
            parent = Path(current).relative_to(root)
            kept = []
            for name in sorted(dirs):
                p = Path(current) / name
                # Keep src/datasets code; prune actual top-level dataset directories.
                dataset_dir = name in {'data', 'datasets', 'raw', 'processed'} and 'src' not in parent.parts
                historical_outer = 'quarantin' in name.lower()
                source_code = 'src' in parent.parts or name == 'src'
                pruned = name in PRUNE and not (source_code and name in {'features', 'representations'})
                if p.is_symlink() or pruned or dataset_dir or historical_outer:
                    skip('excluded_directory_' + name)
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in sorted(files):
                p = Path(current) / name
                rel = p.relative_to(root)
                dst = prefix / rel
                if dst == Path('.gitignore'):
                    skip('mirror_gitignore_is_managed')
                    continue
                why = reason(p, rel)
                if why:
                    skip(why)
                    continue
                data = p.read_bytes()
                if SECRET.search(data):
                    raise RuntimeError('Secret pattern detected; stopped before copying: ' + str(p))
                selected[dst.as_posix()] = (p, hashlib.sha256(data).hexdigest(), len(data))
    return selected, skipped

def audit_tracked():
    files = git('ls-files', '-z').stdout.split('\0')
    for name in files:
        if not name:
            continue
        p = UPLOAD / name
        if not p.is_file() or p.is_symlink():
            raise RuntimeError('Tracked file missing/symlink: ' + name)
        rel = Path(name)
        source_code = 'src' in rel.parts
        blocked_dirs = [d for d in rel.parts[:-1] if d in PRUNE and not (source_code and d in {'features', 'representations'})]
        blocked_data = any(d in {'data', 'datasets', 'raw', 'processed'} for d in rel.parts[:-1]) and not source_code
        why = None if name == '.gitignore' else reason(p, rel)
        if blocked_dirs or blocked_data or why:
            raise RuntimeError('Forbidden tracked file: ' + name + ' (' + str(why or blocked_dirs or 'dataset') + ')')
        if SECRET.search(p.read_bytes()):
            raise RuntimeError('Secret pattern detected in tracked file: ' + name)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true', help='Copy, commit and push; default is preview only')
    ap.add_argument('--message', help='Required meaningful commit message with --apply')
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    if args.audit_only:
        audit_tracked()
        print('TRACKED_FILE_AUDIT PASS')
        return
    if args.apply and not args.message:
        ap.error('--apply requires --message')
    if git('remote', 'get-url', 'origin').stdout.strip() != REMOTE:
        raise RuntimeError('Unexpected repository remote')
    selected, skipped = collect()
    print(json.dumps({'mode': 'apply' if args.apply else 'preview', 'selected_files': len(selected),
                      'selected_bytes': sum(x[2] for x in selected.values()),
                      'largest_file_bytes': max(x[2] for x in selected.values()),
                      'excluded_by_reason': skipped}, indent=2))
    if not args.apply:
        return
    if git('status', '--porcelain').stdout:
        raise RuntimeError('Upload mirror has uncommitted changes; inspect and commit them before sync')
    # Preserve remote changes rather than overwriting by sync/force push.
    git('fetch', 'origin')
    local = git('rev-parse', '--verify', 'HEAD', check=False)
    remote = git('rev-parse', '--verify', 'refs/remotes/origin/main', check=False)
    if remote.returncode == 0:
        if local.returncode != 0 or git('merge-base', '--is-ancestor', remote.stdout.strip(), 'HEAD', check=False).returncode:
            raise RuntimeError('Remote has changes not incorporated locally; reconcile them before publishing')
    tracked = set(filter(None, git('ls-files', '-z').stdout.split('\0')))
    managed = {'.gitignore', 'PUBLICATION_MANIFEST.json'}
    stale = tracked - set(selected) - managed
    if stale:
        raise RuntimeError('Previously tracked files no longer selected; no automatic deletion: ' + ', '.join(sorted(stale)[:20]))
    for name, (source, sha, size) in sorted(selected.items()):
        target = UPLOAD / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise RuntimeError('Source/copy changed while syncing: ' + name)
    (UPLOAD / '.gitignore').write_text(IGNORE)
    manifest = {'remote': REMOTE, 'primary_source': str(PRIMARY), 'secondary_source': str(SECONDARY),
                'max_file_bytes': MAX_BYTES, 'selected_file_count': len(selected),
                'selected_total_bytes': sum(x[2] for x in selected.values()), 'excluded_by_reason': skipped,
                'files': [{'path': n, 'sha256': v[1], 'bytes': v[2]} for n, v in sorted(selected.items())]}
    manifest_text = json.dumps(manifest, separators=(',', ':')) + '\n'
    if len(manifest_text.encode()) > MAX_BYTES:
        raise RuntimeError('Publication manifest exceeds file size policy; partition it before publishing')
    (UPLOAD / 'PUBLICATION_MANIFEST.json').write_text(manifest_text)
    git('add', '--', '.')
    audit_tracked()
    actual = set(filter(None, git('ls-files', '-z').stdout.split('\0')))
    if actual != set(selected) | managed:
        raise RuntimeError('Git staging does not match publication manifest; inspect ignore rules')
    staged = git('diff', '--cached', '--quiet', check=False).returncode
    if staged == 1:
        git('commit', '-m', args.message)
    elif staged != 0:
        raise RuntimeError('Cannot inspect staged diff')
    else:
        print('NO_CONTENT_CHANGES')
    git('push', '--dry-run', 'origin', 'main')
    result = git('push', '-u', 'origin', 'main')
    print(result.stderr.strip())
    head = git('rev-parse', 'HEAD').stdout.strip()
    remote_head = git('ls-remote', 'origin', 'refs/heads/main').stdout.split()[0]
    if head != remote_head:
        raise RuntimeError('Remote HEAD does not match local commit')
    print(json.dumps({'status': 'PUSH_VERIFIED', 'commit': head,
                      'tracked_files': len(list(filter(None, git('ls-files', '-z').stdout.split('\0'))))}))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('PUBLISH_STOPPED: ' + str(exc), file=sys.stderr)
        sys.exit(1)
