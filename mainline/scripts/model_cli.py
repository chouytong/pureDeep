"""List/build/load/infer registered models by name; never starts training."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.registry import list_models, get_model_spec, build_registered_model, load_registered_config
from src.models.registered_checkpoint import list_checkpoints, load_registered_checkpoint
from src.utils.seed import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', action='store_true')
    p.add_argument('--model')
    p.add_argument('--list-checkpoints', action='store_true')
    p.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    p.add_argument('--checkpoint-id')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.list:
        print(json.dumps(list_models(), ensure_ascii=False, indent=2))
        return
    if not a.model:
        p.error('--model is required')
    spec = get_model_spec(a.model)
    if a.list_checkpoints:
        print(json.dumps(list_checkpoints(a.model), ensure_ascii=False, indent=2))
        return
    torch.set_num_threads(4)
    seed_everything(a.seed, True)
    info = dict(model_id=spec['model_id'], status=spec['status'])
    if a.checkpoint_id or a.checkpoint:
        model, info = load_registered_checkpoint(
            a.model, checkpoint_id=a.checkpoint_id, checkpoint_path=a.checkpoint,
            expected_sha256=a.checkpoint_sha256, device=a.device)
    else:
        model = build_registered_model(a.model, seed=a.seed).to(a.device).eval()
    if a.input is None:
        print(json.dumps(dict(info, classifier_parameters=sum(v.numel() for v in model.parameters()),
                              trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                              frozen_external_parameters=spec['frozen_external_parameters']), indent=2))
        return
    if not (a.checkpoint_id or a.checkpoint):
        p.error('Inference requires an explicit checkpoint identity')
    if a.output is None or a.output.exists() or a.limit < 1:
        p.error('A new --output path and positive --limit are required')
    with np.load(a.input, allow_pickle=False) as z:
        if 'bilateral' in z:
            if spec['model_id'] != 'ai_wssl_str':
                p.error('This cached-token inference path is for AI-WSSL-STR')
            for key in ['base_checkpoint_sha256', 'normalization_sha256']:
                if key not in z or str(z[key]) != info.get(key):
                    raise ValueError(f'Input/checkpoint {key} mismatch')
            if 'version' not in z or str(z['version']) != info['data_version']:
                raise ValueError('Input/checkpoint data version mismatch')
            if 'activities' not in z or z['activities'].tolist() != list(model.activities):
                raise ValueError('Input activity order mismatch')
            h = torch.from_numpy(z['bilateral'][:a.limit].copy()).to(a.device)
            mask = torch.from_numpy(z['mask'][:a.limit].copy()).to(a.device)
            with torch.inference_mode():
                result = model(h, mask)
        else:
            if spec['model_id'] == 'ai_wssl_str':
                if 'version' not in z or str(z['version']) != info['data_version']:
                    raise ValueError('Signal/checkpoint data version mismatch')
                if 'normalization_sha256' not in z or str(z['normalization_sha256']) != info['normalization_sha256']:
                    raise ValueError('Normalized signal/checkpoint binding mismatch')
            x = torch.from_numpy(z['x'][:a.limit].copy()).to(a.device)
            wrist = torch.from_numpy(z['wrist_mask'][:a.limit].copy()).to(a.device)
            mask = torch.from_numpy(z['activity_mask'][:a.limit].copy()).to(a.device)
            lengths = torch.from_numpy(z['activity_lengths'][:a.limit].copy()).to(a.device)
            ssl = (torch.from_numpy(z['ssl_features'][:a.limit].copy()).to(a.device)
                   if 'ssl_features' in z else None)
            with torch.inference_mode():
                if spec['model_id'] == 'ai_wssl_str':
                    result = model(x, wrist, mask, lengths, ssl_features=ssl)
                else:
                    if spec['model_id'] == 'frozen_wssl':
                        if ssl is None:
                            raise ValueError('Frozen WSSL requires explicit frozen HarNet features')
                        model.ssl_features = ssl
                    result = model(x, wrist, mask, lengths)
    if not torch.isfinite(result['logits']).all():
        raise RuntimeError('Nonfinite inference logits')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing private predictions.
    with a.output.open('xb') as f:
        np.savez_compressed(f, logits=result['logits'].cpu().numpy(),
                            probabilities=torch.softmax(result['logits'], dim=-1).cpu().numpy(),
                            model_id=np.asarray(spec['model_id']), status=np.asarray(spec['status']))
    print(json.dumps(dict(info, inference_samples=int(result['logits'].shape[0]),
                          output=str(a.output), new_training=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
