#!/usr/bin/env python3
"""Each validation subject's pair involvement, using explicit subject IDs."""
from pathlib import Path
import pandas as pd,numpy as np
O=Path('/home/zyt/deep_final/artifacts/prr01_frozen_str_pair_rescue_20260926/analysis')
p=pd.read_csv(O/'pair_level_records.csv.gz',dtype={'dd_subject_id':str,'pd_subject_id':str})
p['is_h1_rescue']=(p.original_category=='h1_rescue').astype(int)
p['is_recovered']=((p.original_category=='h1_rescue')&(p.corrected_margin>0)).astype(int)
p['is_str_correct']=(p.str_original_margin>0).astype(int)
p['is_newly_broken']=((p.str_original_margin>0)&(p.corrected_margin<0)).astype(int)
use=['outer','inner','seed','stage','is_h1_rescue','is_recovered','is_str_correct','is_newly_broken']
a=p[use+['dd_subject_id','dd_status']].rename(columns={'dd_subject_id':'subject_id','dd_status':'status'});b=p[use+['pd_subject_id','pd_status']].rename(columns={'pd_subject_id':'subject_id','pd_status':'status'})
r=pd.concat([a,b],ignore_index=True).groupby(['outer','inner','seed','stage','subject_id','status'],as_index=False)[['is_h1_rescue','is_recovered','is_str_correct','is_newly_broken']].sum()
r['h1_rescue_recovery_fraction']=r.is_recovered/r.is_h1_rescue.replace(0,np.nan);r['harm_fraction']=r.is_newly_broken/r.is_str_correct.replace(0,np.nan)
r.to_csv(O/'subject_pair_burden_45.csv',index=False)
u=r.groupby(['subject_id','status','stage'],as_index=False)[['is_h1_rescue','is_recovered','is_str_correct','is_newly_broken']].sum();u['h1_rescue_recovery_fraction']=u.is_recovered/u.is_h1_rescue.replace(0,np.nan);u['harm_fraction']=u.is_newly_broken/u.is_str_correct.replace(0,np.nan);u.to_csv(O/'subject_pair_burden_390.csv',index=False)
assert u.subject_id.nunique()==390
print(u.groupby(['stage','status'])[['h1_rescue_recovery_fraction','harm_fraction']].mean().round(4).to_string())
