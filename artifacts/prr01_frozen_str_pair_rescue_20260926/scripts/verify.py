#!/usr/bin/env python3
"""Final PRR artifact and frozen RGD category consistency checks."""
from pathlib import Path
import json,ast
import numpy as np,pandas as pd
B=Path('/home/zyt/deep_final/artifacts/prr01_frozen_str_pair_rescue_20260926');O=B/'analysis';RG=Path('/home/zyt/deep_final/artifacts/rgd01_h1_str_residual_ranking_20260926/analysis')
for f in (B/'scripts').glob('*.py'):ast.parse(f.read_text())
a=pd.read_csv(B/'asset_audit.csv');assert len(a)==45 and a[['outer','inner']].drop_duplicates().shape[0]==15
m=pd.read_csv(O/'method_metrics_45.csv');assert len(m)==270 and m.groupby(['outer','inner','seed']).stage.nunique().eq(6).all();assert np.allclose(m.net_pair_gain/m.pairs,m.auroc_delta_vs_str,atol=1e-9)
p=pd.read_csv(O/'pair_level_records.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str});assert len(p)==503475
q=p[p.stage=='B_score_only'][['outer','inner','seed','dd_subject_id','pd_subject_id','original_category']]
r=pd.read_csv(RG/'ranking_pairs.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str})[['outer','inner','seed','dd_subject_id','pd_subject_id','category']]
match=q.merge(r,on=['outer','inner','seed','dd_subject_id','pd_subject_id'],validate='one_to_one');assert len(match)==100695 and (match.original_category==match.category).all()
n=pd.read_csv(O/'class_conditional_permutation_45.csv');assert len(n)==1800 and n.permutation.nunique()==10
sub=pd.read_csv(O/'subject_pair_burden_390.csv',dtype={'subject_id':str});assert sub.subject_id.nunique()==390
assert json.loads((B/'protocol.json').read_text())['outer_information_used'] is False
print(json.dumps({'status':'PASS','split_seed_instances':45,'methods':6,'pair_instances':100695,'RGD_category_mismatches':0,'permutation_runs':1800,'unique_subjects':390,'AUROC_pair_identity_max_error':float(np.max(abs(m.net_pair_gain/m.pairs-m.auroc_delta_vs_str))),'outer_information_used':False},indent=2))
