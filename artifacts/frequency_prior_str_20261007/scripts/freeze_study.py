"""Freeze only code, prespecified design and completed preformal correctness gates."""
import json
from datetime import datetime,timezone
from common import HERE,sha,require,write_json


def main():
    path=HERE/'analysis/study_lock.json';require(not path.exists(),'Do not overwrite an existing freeze')
    files=list((HERE/'scripts').glob('*.py'))+[HERE/'PROTOCOL.md']
    files += [HERE/'analysis'/f'{name}.json' for name in ['f0_audit','f0_lock','filter_bank_tests','model_tests','preformal_static_checks']]
    require(all(p.is_file() for p in files),'Missing code/entry result')
    require((HERE/'scripts/analyze_results.py').is_file(),'Analysis must be frozen before results')
    write_json(path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                        files={str(p.relative_to(HERE)):sha(p) for p in files},
                        fixed_conditions=['F0','F1','F2'],outer_access=False,no_validation_tuning=True))
    print(json.dumps({'status':'FROZEN','files':len(files),'sha256':sha(path)}),flush=True)


if __name__=='__main__':main()
