"""Replace inherited STR descriptive metadata; model/loss/training values unchanged."""
from pathlib import Path
import json,hashlib,shutil,yaml,sys
ROOT=Path('/home/zyt/deep_final');MAIN=ROOT/'mainline';HERE=ROOT/'artifacts/ai_wssl_str_registration_20261116';L2=ROOT/'artifacts/l1_precision_rebuild_phase9_level2_20261112';rows=[]
sys.path.insert(0,str(MAIN))
from src.engine.ai_wssl_str import validate_recipe
for seed in [42,43,44]:
 p=MAIN/f'configs/ai_wssl_str_seed{seed}.yaml';old=yaml.safe_load(p.read_text());backup=HERE/'restricted_manifests/originals'/f'ai_config_initial_seed{seed}.yaml';assert not backup.exists();shutil.copy2(p,backup);new=json.loads(json.dumps(old));new['development']=dict(hypothesis='Cross-activity Self-Attention with residual logit correction on frozen WSSL-STR',comparison='same-seed data-matched Frozen WSSL-STR',status='Experimental Candidate',promotion='Not authorized by registration; full Certified 15-split validation incomplete; DD Recall tradeoff remains');new['model_selection']['threshold']=dict(positive_class='DD',value=.5,decision='strictly_greater',tie='PD',fitting='none');new['model_selection']['final_epoch']=dict(strategy='strict_best_validation_BA_per_registered_split',tie='first_strict_best');new['provenance']['preprocessing_artifacts']=[str(L2/'analysis/data_version_integrity.json'),str(L2/'analysis/normalization_integrity.json')];new['provenance']['token_source_protocol']=str(ROOT/'artifacts/activity_interaction_certified_pilot3_20261112/PROTOCOL.md');new['provenance']['no_new_training_during_registration']=True
 for key in ['model','loss','training']:assert new[key]==old[key]
 validate_recipe(new);p.write_text(yaml.safe_dump(new,sort_keys=False,allow_unicode=True));rows.append(dict(seed=seed,before_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),after_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model_loss_training_blocks_unchanged=True,metadata_only=True))
(HERE/'audit/config_metadata_closure.json').write_text(json.dumps(dict(status='PASS_METADATA_CLOSURE',revisions=rows,new_training=0),indent=2)+'\n');print('MODEL_LOSS_RECIPE_UNCHANGED_METADATA_CLOSED')
