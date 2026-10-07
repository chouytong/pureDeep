"""Supplement the original F0 audit with actual archived whole-tree source provenance.

This performs no training or selection. All45 previously completed Phase2 matched
STR runs must match current source AND the original formal state/config/predictions.
"""
import json
import numpy as np
import pandas as pd
import torch
from common import *
from src.utils.provenance import source_tree_manifest


def main():
    lock=json.loads((HERE/'analysis/f0_lock.json').read_text());guard(lock)
    _,current=source_tree_manifest(FOUNDATION);rows=[]
    for seed in [42,43,44]:
        for c in range(5):
            for i in range(3):
                original_path=reference(seed,c,i)
                matched_path=ROOT/f'artifacts/phase2_str_training_20260926/runs/baseline/seed{seed}/outer_{c}/inner_{i}'
                original=torch.load(original_path/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                matched=torch.load(matched_path/'checkpoints/best.pt',map_location='cpu',weights_only=False)
                require(matched['provenance']['source_tree_sha256']==current,'Matched F0 historical source differs from current')
                for section in ['model','data','training','evaluation','loss','nested_cv','self_supervised']:
                    require(original['config'].get(section)==matched['config'].get(section),'Matched F0 recipe differs')
                require(original['model_state'].keys()==matched['model_state'].keys() and
                    all(torch.equal(v,matched['model_state'][k]) for k,v in original['model_state'].items()),'Matched F0 state differs')
                require(original['epoch']==matched['epoch'],'Matched F0 best epoch differs')
                require(all(torch.equal(original['normalization'][k],matched['normalization'][k]) for k in ['mean','std']),
                        'Matched F0 normalization differs')
                a=pd.read_csv(original_path/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id').sort_index()
                b=pd.read_csv(matched_path/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id').sort_index()
                require(a.index.equals(b.index) and np.array_equal(a[['target','prediction','probability_pd','probability_dd']].to_numpy(),
                       b[['target','prediction','probability_pd','probability_dd']].to_numpy()),'Matched F0 predictions differ')
                rows.append(dict(seed=seed,context=c,inner=i,current_source_sha=current,
                    matched_f0_source_sha=matched['provenance']['source_tree_sha256'],
                    formal_source_sha=original['provenance']['source_tree_sha256'],source_exact=True,
                    config_sections_exact=True,state_exact=True,prediction_exact=True,normalization_exact=True,best_epoch_exact=True))
    old=json.loads((HERE/'analysis/matched_f0_provenance_audit.json').read_text())
    require(old['status']=='PASS' and old['current_source_tree_sha256']==current and old['units']==len(rows)==45,
            'Supplementary result differs from already executed inline audit')
    previous=pd.read_csv(HERE/'analysis/matched_f0_provenance_45run_audit.csv')
    require(previous.equals(pd.DataFrame(rows)), 'Saved supplementary45table differs')
    guard(lock)
    print(json.dumps(dict(status='PASS',units=45,matched_source_sha256=current,
                         original_comparison_numerically_unchanged=True,no_training=True)),flush=True)


if __name__=='__main__':main()
