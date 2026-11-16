"""Explicit AI-WSSL-STR token-training entry; registration uses --validate-only."""
from pathlib import Path
import argparse
import json
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine.ai_wssl_str import prepare_token_training, train_activity_interaction
from src.utils.seed import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='AI-WSSL-STR', choices=['AI-WSSL-STR'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--token-index', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--base-sha256', required=True)
    p.add_argument('--context', type=int, choices=range(5), required=True)
    p.add_argument('--inner', type=int, choices=range(3), required=True)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    seed_everything(int(cfg['experiment']['seed']), True)
    kwargs = dict(token_index=a.token_index, base_checkpoint=a.base_checkpoint,
                  base_sha256=a.base_sha256, context=a.context, inner=a.inner)
    if a.validate_only:
        model, packages, provenance = prepare_token_training(cfg, **kwargs)
        print(json.dumps(dict(status='VALIDATED_NO_TRAINING', model='AI-WSSL-STR',
                              train_subjects=len(packages['train']['target']),
                              validation_subjects=len(packages['validation']['target']),
                              optimizer_updates=0, **provenance)))
        return
    if a.output_dir is None:
        p.error('--output-dir required for training')
    print(json.dumps(train_activity_interaction(cfg, **kwargs,
                                               output_dir=a.output_dir, device=a.device)))


if __name__ == '__main__':
    main()
