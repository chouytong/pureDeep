"""Only F1/F2×three seeds; all45units each, no partial-score-driven changes."""
import json
import subprocess
import sys
from datetime import datetime,timezone
from common import HERE,require,write_json


def main():
    path=HERE/'analysis/execution_state.json'
    require(not path.exists(),'Existing execution state: inspect rather than overwrite/restart')
    state=dict(status='running',started_at_utc=datetime.now(timezone.utc).isoformat(),completed=[],required_runs=90,
               candidates=['F1','F2'],seeds=[42,43,44],interim_score_selection=False,outer_access=False)
    write_json(path,state);(HERE/'logs').mkdir(exist_ok=True)
    for condition in ['F1','F2']:
        for seed in [42,43,44]:
            print(json.dumps({'phase':'start','condition':condition,'seed':seed}),flush=True)
            log=HERE/f'logs/{condition}_seed{seed}.log'
            with log.open('x') as f:
                process=subprocess.run([sys.executable,str(HERE/'scripts/run_condition.py'),
                                        '--condition',condition,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT)
            if process.returncode:
                state.update(status='failed',failed_condition=condition,failed_seed=seed,returncode=process.returncode)
                write_json(path,state);raise RuntimeError(f'Preserved failed run {log}; no automatic retry')
            state['completed'].append({'condition':condition,'seed':seed,'units':15})
            write_json(path,state);print(json.dumps({'phase':'complete','condition':condition,'seed':seed,'units':15}),flush=True)
    state.update(status='complete',completed_runs=90,finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(path,state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()
