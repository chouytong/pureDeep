"""Run on the research server. Add a canonical mainline; never delete sources."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import shutil
import sys

ROOT = Path('/home/zyt/deep_final')
SOURCE = ROOT / 'foundation_validation'
MAIN = ROOT / 'mainline'
DATA = ROOT / 'data'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
from src.utils.config import load_config
import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def new_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == value, f'Preserve existing file: {path}'
    else:
        path.write_text(value)


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(source)
    if target.exists():
        assert sha(target) == digest, f'Existing target differs: {target}'
    else:
        shutil.copy2(source, target)
    assert sha(target) == digest, f'Copy mismatch: {target}'
    return digest


def main():
    assert ROOT.is_dir() and Path('/home/zyt/MFAM/data').is_dir()
    private_ledger, totals = [], []
    for src, dest in [
        (Path('/home/zyt/MFAM/data/raw/pads'), DATA / 'raw/pads'),
        (Path('/home/zyt/MFAM/data/processed/pads_multi_activity/v2_l1_full_length'), DATA / 'processed/pads_multi_activity/v2_l1_full_length'),
    ]:
        count = size = 0
        for path in sorted(src.rglob('*')):
            assert not path.is_symlink(), 'Inspect symlinks instead of copying external references'
            if not path.is_file():
                continue
            target = dest / path.relative_to(src)
            digest = copy_file(path, target)
            private_ledger.append(dict(relative_path=str(target.relative_to(DATA)), sha256=digest, bytes=path.stat().st_size))
            count += 1
            size += path.stat().st_size
        totals.append(dict(source=str(src), destination=str(dest), files=count, bytes=size, hashes_equal=True))
        print(json.dumps(dict(stage='data_copy', **totals[-1])), flush=True)
    cache = ROOT / 'artifacts/phase3b_external_ssl_20260928/analysis/ssl_embeddings_all.npz'
    digest = copy_file(cache, DATA / 'ssl/harnet10_embeddings.npz')
    assert digest == 'f87829558938c3e50f78e53fe2f54156fc0f3768be3f73f97b089c0587c25376'
    new_text(DATA / 'MIGRATION_MANIFEST_PRIVATE.json', json.dumps(dict(files=private_ledger, copies=totals,
        feature_cache_sha256=digest, original_data_deleted=False), ensure_ascii=False, indent=2) + '\n')

    copied = []
    for group in ['datasets', 'engine', 'utils', 'metrics', 'losses']:
        for path in sorted((SOURCE / 'src' / group).glob('*.py')):
            if not path.name.startswith('.'):
                target = MAIN / 'src' / group / path.name
                copied.append(dict(source=str(path), destination=str(target), sha256=copy_file(path, target)))
    for name in ['pure_deep.py', 'bilateral.py', 'activity_fusion.py']:
        p = SOURCE / 'src/models' / name
        copied.append(dict(source=str(p), destination=str(MAIN / 'src/models' / name),
                           sha256=copy_file(p, MAIN / 'src/models' / name)))
    new_text(MAIN / 'src/__init__.py', '"""PADS PD versus DD mainline package."""\n')
    new_text(MAIN / 'src/models/__init__.py', '''"""Mainline only: STR-01. Historical MFAM/replication factories are excluded."""
from .pure_deep import PureDeepSubjectModel, build_pure_deep_subject

def build_model(config):
    if config['model'].get('name') != 'pure_deep_subject':
        raise ValueError('This mainline entry only supports the retained STR architecture')
    return build_pure_deep_subject(config)
''')
    transfer = ROOT / 'artifacts/phase3b_external_ssl_20260928/scripts/transfer_hooks.py'
    text = transfer.read_text()
    tree = ast.parse(text)
    wanted = {'FrozenEmbeddingCache', 'ResidualSSLSubject', 'SSLLoader', 'activate'}
    definitions = [ast.get_source_segment(text, n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    wssl = '"""Retained pretrained-frozen HarNet10 residual; exact archived forward math."""\nimport types\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom src.engine import nested_training as nt\nCACHE_PATH=Path(__file__).resolve().parents[3]/"data/ssl/harnet10_embeddings.npz"\n\n' + '\n\n'.join(definitions) + '\n'
    # Retain original constructor/forward math and state keys; disallow controls.
    wssl = wssl.replace("assert kind in ('pretrained','random')", "assert kind == 'pretrained'")
    old = "if variant['model_kind']=='ssl_only':\n        assert kind=='pretrained'\n        nt.build_model=lambda cfg:SSLOnlySubject()\n    elif variant['model_kind']=='str_residual':"
    assert old in wssl
    wssl = wssl.replace(old, "if variant['model_kind']=='str_residual':")
    new_text(MAIN / 'src/models/wssl.py', wssl)
    copy_file(SOURCE / 'pyproject.toml', MAIN / 'pyproject.toml')
    new_text(MAIN / 'requirements.txt', (SOURCE / 'requirements.txt').read_text() + 'pandas>=2,<3\n')
    for name in ['run_development.py', 'verify_mainline.py']:
        copy_file(HERE / name, MAIN / 'scripts' / name)
    # Same split protocol, stored beneath the new package root.
    split_src = SOURCE / 'splits/pads_classification/v3_nested_cv/seed42'
    for path in split_src.iterdir():
        if path.is_file():
            copy_file(path, MAIN / 'splits/pads_classification/v3_nested_cv/seed42' / path.name)
    for seed in [42, 43, 44]:
        cfg = load_config(SOURCE / f'configs/str01_seed{seed}.yaml')
        cfg.pop('_meta', None)
        old_data = str(Path('/home/zyt/MFAM/data'))
        def relocate(value):
            if isinstance(value, dict):
                return {k: relocate(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relocate(v) for v in value]
            return value.replace(old_data, str(DATA)) if isinstance(value, str) else value
        cfg = relocate(cfg)
        cfg['nested_cv']['split_file'] = str(MAIN / 'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json')
        cfg['experiment']['output_root'] = str(ROOT / 'results/new_runs/str01' / f'seed{seed}')
        cfg['provenance']['preprocessing_artifacts'] = [str(ROOT / p) if p.startswith('data/') else p
                                                       for p in cfg['provenance']['preprocessing_artifacts']]
        new_text(MAIN / f'configs/str01_seed{seed}.yaml', yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    # All result aliases resolve within deep_final; no private assets are copied to Git.
    aliases = {
        'str01': ROOT / 'foundation_validation/outputs/structured_token_residual',
        'frozen_wssl': ROOT / 'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained',
        'patch_str': ROOT / 'artifacts/patch_str_20261008/runs',
    }
    (ROOT / 'results').mkdir(exist_ok=True)
    for name, target in aliases.items():
        assert target.is_dir(), target
        link = ROOT / 'results' / name
        if link.is_symlink():
            assert link.resolve() == target
        else:
            assert not link.exists()
            link.symlink_to(Path('..') / target.relative_to(ROOT), target_is_directory=True)

    index = []
    for folder in sorted((ROOT / 'artifacts').iterdir()):
        if not folder.is_dir() or folder.name in {'foe01_final_locked_outer_20260926', 'final_nested_cv', 'final_analysis'}:
            continue
        for p in sorted(folder.glob('*.md')):
            if any(word in p.name.upper() for word in ['REPORT', 'DECISION', 'PROTOCOL']):
                index.append(dict(experiment=folder.name, report=p.name, path=str(p.relative_to(ROOT))))
    conclusions = ROOT / 'conclusions'
    conclusions.mkdir(exist_ok=True)
    csv_path = conclusions / 'EXPERIMENT_REPORT_INDEX.csv'
    assert not csv_path.exists(), 'Preserve existing index'
    with csv_path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['experiment', 'report', 'path'])
        writer.writeheader()
        writer.writerows(index)
    overview = dict(status='COPIED_PENDING_RUNTIME_VERIFICATION', date='2026-11-11', canonical_root=str(ROOT),
        active_code=str(MAIN), active_data=str(DATA), copies=totals, copied_code=copied,
        frozen_ssl_cache_sha256=digest, old_data_removed=False, old_code_removed=False,
        unrelated_replications_imported=False, new_training=False, result_aliases={k:str(v) for k,v in aliases.items()},
        report_index_entries=len(index), private_file_ledger_path=str(DATA / 'MIGRATION_MANIFEST_PRIVATE.json'))
    new_text(conclusions / 'organization_20261111/COPY_AUDIT.json', json.dumps(overview, ensure_ascii=False, indent=2) + '\n')
    for p in MAIN.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    print(json.dumps(dict(status='MAINLINE_CREATED', code_files=len(copied), report_index_entries=len(index),
                         data_files=sum(x['files'] for x in totals), data_bytes=sum(x['bytes'] for x in totals))), flush=True)


if __name__ == '__main__':
    main()
