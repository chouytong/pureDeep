#!/usr/bin/env python3
"""PAG output integrity and formal-score pair identity."""
from pathlib import Path
import ast,json
import numpy as np,pandas as pd
B=Path('/home/zyt/deep_final/artifacts/pag01_preaggregation_disease_info_20260926');O=B/'analysis'
for f in (B/'scripts').glob('*.py'):ast.parse(f.read_text())
a=pd.read_csv(B/'asset_audit.csv');assert len(a)==45 and a[['outer','inner']].drop_duplicates().shape[0]==15
p=pd.read_csv(O/'disease_probe_45.csv');s=pd.read_csv(O/'increment_signal_45.csv');n=pd.read_csv(O/'conditional_permutation_45.csv');sub=pd.read_csv(O/'subject_predictions_primary.csv.gz',dtype={'subject_id':str});pair=pd.read_csv(O/'ambiguity_pair_records.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str});aud=pd.read_csv(O/'crossfit_audit_45.csv')
assert len(p)==720 and len(s)==675 and len(n)==2250 and len(sub)==23400 and len(pair)==100695 and len(aud)==45
assert sub.groupby(['outer','inner','seed']).stage.nunique().eq(5).all();assert n.permutation.nunique()==10
assert np.max(abs(aud.validation_base_auroc-aud.validation_str_auroc))<1e-12 and aud.baseline_coef_positive.all()
assert pair.groupby(['outer','inner','seed']).size().eq(a.validation_pd.mul(a.validation_dd).to_numpy()).all()
per=pair.groupby(['outer','inner','seed']).agg(recovered=('recovered','sum'),broken=('newly_broken','sum'),pairs=('recovered','size')).reset_index().merge(aud,on=['outer','inner','seed'])
identity=np.max(abs((per.recovered-per.broken)/per.pairs-(per.validation_raw_auroc-per.validation_str_auroc)))
assert identity<1e-12
assert json.loads((B/'protocol.json').read_text())['outer_information_used'] is False
print(json.dumps({'status':'PASS','split_seed_instances':45,'stages':5,'primary_pca':16,'sensitivity_pca':[8,32],'disease_probe_rows':len(p),'increment_signal_rows':len(s),'conditional_permutation_rows':len(n),'validation_pair_rows':len(pair),'subject_prediction_rows':len(sub),'max_baseline_formal_auc_error':float(np.max(abs(aud.validation_base_auroc-aud.validation_str_auroc))),'pair_auc_identity_max_error':float(identity),'outer_information_used':False},indent=2))
