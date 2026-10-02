"""B1 independent input control; reuse original WSSL model unchanged."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
OLD=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928');sys.path.insert(0,str(OLD/'scripts'))
from transfer_hooks import FrozenEmbeddingCache, activate as activate_original
from src.engine import nested_training as nt
from src.datasets import folds as fd


def raw_index():
    audit=json.loads((HERE/'analysis/b1_raw_input_audit.json').read_text())
    path=HERE/'features/raw_acc/index.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['raw_cache_index_sha256']
    index=json.loads(path.read_text())
    h=hashlib.sha256()
    for source,entry in index.items():
        digest=hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()
        assert digest==entry['sha256']
        h.update(source.encode());h.update(digest.encode())
    assert h.hexdigest()==audit['raw_cache_content_sha256']
    return index


def activate(config,cache,index):
    activate_original(config,{'model_kind':'str_residual','embedding_kind':'pretrained'},cache)
    def bundle(config,*,train_subject_ids,validation_subject_ids=(),test_subject_ids=(),fold_id):
        assert not test_subject_ids
        allowed=set(train_subject_ids)|set(validation_subject_ids)
        original=fd.load_configured_records
        def load(*args,**kwargs):
            result=[]
            for r in original(*args,**kwargs):
                if r.subject_id in allowed:
                    result.append(replace(r,left_path=Path(index[str(r.left_path)]['path']),right_path=Path(index[str(r.right_path)]['path'])))
            return result
        fd.load_configured_records=load
        try:
            result=fd.build_subject_fold_datasets(config,train_subject_ids=train_subject_ids,validation_subject_ids=validation_subject_ids,test_subject_ids=(),fold_id=fold_id)
            assert result.test is None
            return result
        finally:fd.load_configured_records=original
    nt.build_subject_fold_datasets=bundle
