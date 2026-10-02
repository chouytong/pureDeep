"""Independent B1 local input cache: replace Acc only, preserve processed Gyro."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent.parent
F=Path('/home/zyt/deep_final/foundation_validation');sys.path.insert(0,str(F))
from src.utils.config import load_config
from src.datasets.builders import load_configured_records
from src.datasets.manifest import _load_array, shape_time_series


def main():
    cfg=load_config(str(F/'configs/str01_seed42.yaml'))
    records=load_configured_records(cfg['data'],['PD','DD'])
    split=json.loads((F/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json').read_text())
    allowed=set()
    for context in split['outer']:
        for inner in context['inner_folds']:
            allowed.update(inner['train_subjects']);allowed.update(inner['validation_subjects'])
    assert len(allowed)==390 and {r.subject_id for r in records}==allowed
    dest=HERE/'features/raw_acc';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'index.json').exists(), 'Do not overwrite a completed raw-input cache'
    rawroot=Path('/home/zyt/MFAM/data/raw/pads/movement/timeseries')
    index={};h=hashlib.sha256();gyromax=0.;accdiff=0.;lengths=set()
    for r in records:
        for wi,(side,source) in enumerate((('LeftWrist',r.left_path),('RightWrist',r.right_path))):
            processed=shape_time_series(_load_array(source,','),6,source)
            rawpath=rawroot/f'{r.subject_id}_{r.record_name}_{side}.txt'
            raw=np.loadtxt(rawpath,delimiter=',',dtype=np.float32)
            expected=r.left_length if wi==0 else r.right_length
            assert raw.shape[1]==7 and len(raw)-48==expected==processed.shape[-1]
            new=processed.copy();new[:3]=raw[48:,1:4].T
            assert np.isfinite(new).all() and np.array_equal(new[3:],processed[3:])
            lengths.add(new.shape[-1]);accdiff=max(accdiff,float(np.abs(new[:3]-processed[:3]).max()))
            gyromax=max(gyromax,float(np.abs(new[3:]-processed[3:]).max()))
            path=dest/f'{r.subject_id}_{r.activity}_{side}.npy'
            np.save(path,new,allow_pickle=False)
            assert np.array_equal(np.load(path,allow_pickle=False),new)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            index[str(source)]={'path':str(path),'sha256':digest}
            h.update(str(source).encode());h.update(digest.encode())
    assert len(index)==8580 and lengths=={976,2000} and accdiff>0 and gyromax==0
    (dest/'index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
    audit=dict(status='PASS',records=8580,development_subject_count=390,trim_samples=48,sequence_lengths=sorted(lengths),gyro_exact=True,maximum_gyro_difference=gyromax,raw_acc_max_change=accdiff,local_raw_acc_clipped=False,ssl_cache_changed=False,raw_cache_content_sha256=h.hexdigest(),raw_cache_index_sha256=hashlib.sha256((dest/'index.json').read_bytes()).hexdigest(),all_array_roundtrips_exact=True,outer_artifacts_accessed=False)
    (HERE/'analysis/b1_raw_input_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
