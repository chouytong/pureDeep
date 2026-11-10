"""Local driver. Uses an already authenticated SSH control socket, no password."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
phase = sys.argv[1]
assert phase in {'integrity', 'baseline', 'correctness', 'analysis', 'validation'}
started = time.perf_counter()
result = subprocess.run([
    'ssh', '-S', '/tmp/codex-patchstr-20261111.sock', '-p', '12222',
    'zyt@10.12.43.173',
    '/home/zyt/envs/mfam_analysis/bin/python', '-B', '-', phase,
], input=(HERE / 'remote_recheck.py').read_text(), text=True,
   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(HERE / f'{phase}.log').write_text(result.stdout)
print(json.dumps(dict(phase=phase, returncode=result.returncode,
                     elapsed_seconds=time.perf_counter() - started,
                     log=str(HERE / f'{phase}.log'))), flush=True)
print(result.stdout[-4000:], flush=True)
sys.exit(result.returncode)
