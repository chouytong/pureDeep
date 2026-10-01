#!/usr/bin/env python3
"""RGD output consistency; all inputs are frozen development assets."""
from pathlib import Path
import ast,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
B=Path('/home/zyt/deep_final/artifacts/rgd01_h1_str_residual_ranking_20260926');O=B/'analysis'
for p in (B/'scripts').glob('*.py'):ast.parse(p.read_text())
a=pd.read_csv(B/'asset_audit.csv');assert len(a)==15 and a.h1_reproduction_max_probability_delta.max()<1e-10
p=pd.read_csv(O/'pair_decomposition_45_seed_splits.csv');assert len(p)==45 and np.allclose(p.h1_auc-p.str_auc,p.gap)
assert ((p.both_correct+p.both_wrong+p.h1_rescue+p.str_rescue+p.ties)==p.pairs).all()
assert p.pd_n.mul(p.dd_n).eq(p.pairs).all()
r=pd.read_csv(O/'ranking_pairs.csv.gz');assert len(r)==p.pairs.sum();assert set(r.category)=={'both_correct','both_wrong','h1_rescue','str_rescue','tie'}
f=pd.read_csv(O/'fusion_45_paired.csv');assert len(f)==45
x=pd.read_csv(O/'residual_pca16_45.csv');assert len(x)==225
c=pd.read_csv(O/'conditional_h1_group_contributions_45.csv');assert c[['outer','inner','seed']].drop_duplicates().shape[0]==45
assert json.loads((B/'protocol.json').read_text())['outer_signal_label_prediction_metric_accessed'] is False
print(json.dumps({'status':'PASS','audited_splits':len(a),'model_seed_splits':len(p),'validation_pairs_seed_instances':len(r),'fusion_rows':len(f),'pca_probe_rows':len(x),'H1_reproduction_max_delta':a.h1_reproduction_max_probability_delta.max(),'outer_information_used':False},indent=2))
