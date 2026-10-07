"""Read-only frozen development assets; no training or outer loader."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent.parent
ROOT = Path('/home/zyt/deep_final')
FOUNDATION = ROOT / 'foundation_validation'
BASELINE = ROOT / 'artifacts/phase3b_external_ssl_20260928'
EMA = ROOT / 'artifacts/wssl_ema_20261002'
OLD_LOCK = ROOT / 'artifacts/phase4_window_dynamics_20261002/analysis/baseline_lock.json'
REVIEW = ROOT / 'artifacts/wssl_review_20261004/analysis/assets_normalization_audit.json'
sys.path[:0] = [str(FOUNDATION), str(EMA/'scripts'), str(BASELINE/'scripts')]
from independent_wssl import IndependentWSSL
from transfer_hooks import ResidualSSLSubject
from src.models import build_model
from src.datasets import folds as fd
from src.datasets.subject_activity import SubjectActivityDataset, collate_subject_activities
from src.metrics.classification import classification_metrics
from src.utils.config import load_config
from src.utils.seed import seed_everything
from ema_hooks import development_bundle

ACTIVITIES = ['CrossArms', 'DrinkGlas', 'Entrainment', 'HoldWeight', 'LiftHold',
              'PointFinger', 'Relaxed', 'RelaxedTask', 'StretchHold', 'TouchIndex', 'TouchNose']
METRICS = ['accuracy', 'ba', 'auroc', 'macro_f1', 'pd_recall', 'dd_recall']


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def stage_path(seed, context, inner):
    return BASELINE/f'runs/b_str_pretrained/seed{seed}/outer_{context}/inner_{inner}'


def str_path(seed, context, inner):
    return FOUNDATION/f'outputs/structured_token_residual/str01_seed{seed}_20260921/outer_{context}/inner_{inner}'


def predictions(path):
    f = pd.read_csv(path, dtype={'subject_id':str}).set_index('subject_id').sort_index()
    require(f.index.is_unique, 'Duplicate subject ID')
    require(set(f.target)=={0,1}, 'Both disease classes required')
    require(np.isfinite(f[['probability_pd','probability_dd']].to_numpy()).all(), 'Invalid probability')
    require(np.array_equal(f.prediction.to_numpy(), (f.probability_dd.to_numpy()>.5).astype(int)), 'Decision rule differs')
    return f


def metrics(targets, probability):
    prob = np.asarray(probability, dtype=np.float64)
    pred = prob.argmax(1)
    m = classification_metrics(np.asarray(targets), pred, 2, probabilities=prob)
    return dict(accuracy=m['accuracy'], ba=m['balanced_accuracy'], auroc=m['macro_auroc'],
                macro_f1=m['macro_f1'], pd_recall=m['per_class_recall'][0], dd_recall=m['per_class_recall'][1])


def conditions():
    out = [('full', 'full', '', '', np.ones((11,2), np.float32))]
    for i, a in enumerate(ACTIVITIES):
        mask = np.ones((11,2), np.float32); mask[i,:] = 0
        out.append(('activity_off__'+a, 'activity', a, 'both', mask))
    for j, w in enumerate(['left','right']):
        mask = np.ones((11,2), np.float32); mask[:,j] = 0
        out.append((w+'_off', 'wrist', '', w, mask))
    for i, a in enumerate(ACTIVITIES):
        for j, w in enumerate(['left','right']):
            mask = np.ones((11,2), np.float32); mask[i,j] = 0
            out.append(('activity_wrist_off__'+a+'__'+w, 'activity_wrist', a, w, mask))
    out.append(('all_ssl_off', 'anchor', '', 'both', np.zeros((11,2), np.float32)))
    require(len(out)==37 and len({v[0] for v in out})==37, 'Condition contract')
    return out


class ValidationCache:
    """Extract by ID before creating tensors; only current validation rows used."""
    def __init__(self, path, allowed):
        with np.load(path, allow_pickle=False) as z:
            ids = list(map(str,z['subject_ids'])); activities=list(map(str,z['activities']))
            require(len(ids)==len(set(ids))==390 and activities==ACTIVITIES, 'Cache identity/order')
            lookup = {s:i for i,s in enumerate(ids)}
            self.ids = sorted(map(str,allowed)); self.index={s:i for i,s in enumerate(self.ids)}
            require(set(self.ids)<=set(ids), 'Validation ID absent from cache')
            idx = [lookup[s] for s in self.ids]
            values=z['pretrained'][idx].copy(); counts=z['window_counts'][idx].copy()
        require(values.shape==(len(self.ids),11,2,1024), 'SSL shape')
        require(np.isfinite(values).all() and np.all(counts>=1), 'Invalid validation cache')
        self.values=torch.from_numpy(values)
        self.fingerprint=tensor_sha(self.values)

    def batch(self, ids, device='cuda'):
        require(set(ids)<=set(self.ids), 'Non-validation cache lookup')
        return self.values[[self.index[str(s)] for s in ids]].to(device)


def prepare_lock():
    destination=HERE/'analysis/assets_lock.json'
    require(not destination.exists(), 'Do not overwrite asset lock')
    old=json.loads(OLD_LOCK.read_text()); review=json.loads(REVIEW.read_text())
    require(old['completed_development_stages']==45 and old['fixed_development_splits']==15, 'Incomplete frozen baseline')
    source_files=dict(old['source_sha256']);source_files.update(review['source_sha256'])
    for p in (FOUNDATION/'src').rglob('*.py'):
        if not p.name.startswith('._'): source_files[str(p)]=sha(p)
    for p in [EMA/'scripts/independent_wssl.py',EMA/'scripts/ema_hooks.py',OLD_LOCK,REVIEW]: source_files[str(p)]=sha(p)
    for seed in [42,43,44]:
        p=FOUNDATION/f'configs/str01_seed{seed}.yaml'; source_files[str(p)]=sha(p)
    for p,h in source_files.items(): require(sha(p)==h, 'Frozen source differs '+p)
    require(sha(old['cache_path'])==old['cache_sha256'], 'Cache SHA changed')
    require(sha(old['split_path'])==old['split_sha256'], 'Split SHA changed')
    files={old['cache_path']:old['cache_sha256'],old['split_path']:old['split_sha256']}
    for s in old['stages']:
        base=stage_path(s['seed'],s['context'],s['inner'])
        for rel,key in [('checkpoints/best.pt','checkpoint_sha256'),('predictions/validation.csv','prediction_sha256'),('normalization.json','normalization_file_sha256')]:
            p=base/rel; require(sha(p)==s[key], 'Baseline asset SHA changed'); files[str(p)]=s[key]
        for p in [base/'stage_status.json',str_path(s['seed'],s['context'],s['inner'])/'predictions/validation.csv']:
            files[str(p)]=sha(p)
    value=dict(status='LOCKED',source_files=source_files,asset_files=files,cache_path=old['cache_path'],
               split_path=old['split_path'],split_sha256=old['split_sha256'],stages=old['stages'],
               historical_forward_source_identity_claimed=False,outer_performance_accessed=False,
               existing_normalization_refit_audit_sha256=sha(REVIEW))
    write_json(destination,value); return value


def guard():
    value=json.loads((HERE/'analysis/assets_lock.json').read_text())
    for p,h in {**value['source_files'],**value['asset_files']}.items(): require(sha(p)==h, 'Locked asset modified '+p)
    freeze=HERE/'analysis/diagnostic_lock.json'
    if freeze.exists():
        for p,h in json.loads(freeze.read_text())['files'].items():require(sha(HERE/p)==h, 'Diagnostic file changed '+p)
    return value


def state_fingerprint(model):
    return {k:tensor_sha(v) for k,v in model.state_dict().items()}


@torch.no_grad()
def encode(model, inputs, ssl):
    x, wm, am, lengths=inputs
    _, wrist=model.backbone._encode_activities_with_wrists(x,wm,am,lengths)
    valid=wm.bool() & am.bool()[...,None]
    added=model.wrist_projection(model.norm(ssl))*valid[...,None]
    return wrist,added


@torch.no_grad()
def from_parts(model, wrist, added, wm, am, scale):
    require(not model.training, 'Diagnostic requires eval')
    b=model.backbone
    require(b.wrist_structured_residual is None, 'Unexpected retained wrist structured path')
    require(tuple(scale.shape)==(11,2), 'Wrong residual mask')
    am=am.bool(); valid_wrist=wm.bool() & am[...,None]
    fused=wrist + added*scale.to(wrist.device)[None,...,None]
    batch,activities=fused.shape[:2]
    valid=am.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    selected=fused.reshape(-1,2,64).index_select(0,valid)
    selected_mask=valid_wrist.reshape(-1,2).index_select(0,valid)
    features,_=b.wrist_fusion.fusion_features(selected,selected_mask)
    result=fused.new_zeros((batch*activities,258)).index_copy(0,valid,features)
    activity=result.reshape(batch,activities,258)
    aggregated=b.activity_aggregator(activity,am)
    logits=b.classifier(aggregated['subject_embedding'])
    residual=b.structured_token_residual(activity,am)
    return logits+residual['structured_residual_logits']


def freeze_diagnostics():
    p=HERE/'analysis/diagnostic_lock.json';require(not p.exists(), 'Do not overwrite diagnostic freeze')
    files=list((HERE/'scripts').glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'analysis/full_reproduction.json',
           HERE/'analysis/assets_lock.json',HERE/'analysis/smoke_test.json']
    require((HERE/'scripts/analyze.py').exists(), 'Analysis must exist before counterfactual results')
    write_json(p,dict(files={str(f.relative_to(HERE)):sha(f) for f in files},
              case_rules_frozen=True,masked_performance_not_yet_computed=True,training=False,outer_access=False))

