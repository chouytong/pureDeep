"""Read completed audit counts only; never inspect intermediate metrics."""
import json
from pathlib import Path
from common import HERE
state=json.loads((HERE/'analysis/execution_state.json').read_text())
audits=list((HERE/'runs').glob('*/seed*/outer_*/inner_*/patch_audit.json'))
print(json.dumps(dict(status=state['status'],completed_units=len(audits),required_units=90,by_condition={c:sum(p.parts[-5]==c for p in audits) for c in ['P1','P2']},completed_groups=state['completed'])))
if state['status']=='failed':print((HERE/f"logs/{state['failed_condition']}_seed{state['failed_seed']}.log").read_text()[-6000:])
