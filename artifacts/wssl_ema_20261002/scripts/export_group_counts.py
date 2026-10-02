"""Publish an aggregate count table; no subject identifiers are exported."""
from pathlib import Path
import pandas as pd
H=Path(__file__).resolve().parent.parent
f=pd.read_csv(H/'analysis/error_group_summary.csv')
assert set(f.columns)=={'baseline_group','label','subjects','baseline_errors','ema_errors','recovered','harmed','net_recovery'}
assert f.subjects.sum()==390 and len(f)==6
# 'subjects' is a count here, but the publisher reserves that header for ID lists.
f.rename(columns={'subjects':'subject_count'}).to_csv(H/'analysis/error_group_counts_public.csv',index=False)
print('6 aggregate group/class rows exported; no subject identifiers')
