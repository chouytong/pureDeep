"""Sequential execution of the only fixed LR candidate, no search/controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-complete', action='store_true')
    args = parser.parse_args()
    lock = json.loads((HERE/'analysis/lr_control_lock.json').read_text())
    key = 'scripts/launch_lr_matrix.py'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == lock['files'][key]
    state_path = HERE/'analysis/lr_execution_state.json'
    if state_path.exists() and not args.reuse_complete:
        raise RuntimeError('Existing execution record: inspect state before explicit completed-only reuse')
    logs = HERE/'logs'
    logs.mkdir(exist_ok=True)
    state = dict(status='running', started_at_utc=now(), seeds=[42,43,44], completed_seeds=[],
                 only_candidate='lr_1e4', outer_access=False, explicit_completed_reuse=args.reuse_complete)
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    for seed in (42,43,44):
        state.update(active_seed=seed, updated_at_utc=now())
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        command = [sys.executable,'-u',str(HERE/'scripts/run_lr_control.py'),'--seed',str(seed)]
        if args.reuse_complete:
            command.append('--reuse-complete')
        print(json.dumps(dict(event='SEED_START',seed=seed,time=now())),flush=True)
        with (logs/f'lr_seed{seed}_console.log').open('a') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='failed',failed_seed=seed,returncode=result.returncode,updated_at_utc=now())
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            raise RuntimeError(f'Seed {seed} terminated; preserve records and inspect before any restart')
        state['completed_seeds'].append(seed)
        state['updated_at_utc'] = now()
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(dict(event='SEED_COMPLETE',seed=seed,time=now())),flush=True)
    state.update(status='complete',active_seed=None,completed_at_utc=now())
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(event='MATRIX_COMPLETE',completed_candidate_runs=45,seeds=3,development_units=15)),flush=True)


if __name__=='__main__':
    main()
