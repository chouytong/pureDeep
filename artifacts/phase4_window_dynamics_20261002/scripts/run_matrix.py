"""Execute exactly one variant; never advance to the next numbered item."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--variant', choices=['a1_mean', 'a2_delta'], required=True)
args = ap.parse_args()
test = json.loads((HERE / f'analysis/{args.variant}_implementation_test.json').read_text())
assert test['status'] == 'PASS'
smoke = json.loads((HERE / f'smoke/{args.variant}/seed42/outer_0/inner_0/stage_status.json').read_text())
assert smoke['status'] == 'complete' and smoke['summary']['smoke'] is True
progress = HERE / f'{args.variant}_matrix_progress.json'
for seed in (42,43,44):
    cmd = [sys.executable, str(HERE / 'scripts/run_inner.py'), '--variant', args.variant, '--seed', str(seed), '--resume']
    log = HERE / f'run_{args.variant}_seed{seed}.log'
    progress.write_text(json.dumps(dict(status='running', variant=args.variant, seed=seed, pid=os.getpid(), started_unix=time.time())))
    with log.open('a') as stream:
        code = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        progress.write_text(json.dumps(dict(status='failed', variant=args.variant, seed=seed, exit_code=code)))
        raise SystemExit(code)
progress.write_text(json.dumps(dict(status='complete', variant=args.variant, completed_unix=time.time())))
