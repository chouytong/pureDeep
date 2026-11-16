"""Registration publishing guard; no model execution or training."""
from pathlib import Path
import json,hashlib
ROOT=Path('/home/zyt/deep_final');HERE=ROOT/'artifacts/ai_wssl_str_registration_20261116';START='50a319441e2391ce662646cbd3e690ea64d0e1aa'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
 d=load(HERE/'restricted_manifests/protected_assets.json')['sha256']
 for p,h in d.items():assert sha(p)==h,'Protected historical file changed '+p
 return len(d)
def allowed(name):
 if name.startswith('artifacts/ai_wssl_str_registration_20261116/'):return True
 if name in load(HERE/'audit/before_state.json')['authorized_existing_file_changes']:return True
 return name in {'mainline/src/models/ai_wssl_str.py','mainline/src/models/registry.py','mainline/src/models/registered_checkpoint.py','mainline/src/engine/ai_wssl_str.py','mainline/scripts/model_cli.py','mainline/scripts/train_ai_wssl_str.py','mainline/docs/models/AI_WSSL_STR.md','mainline/models/ai_wssl_str_structure.json','mainline/models/ai_wssl_str_checkpoints.json','mainline/models/registry.json','conclusions/MODEL_INDEX.md',*[f'mainline/configs/ai_wssl_str_seed{s}.yaml' for s in [42,43,44]]}
