"""Independent NumPy checks on the exported CSV and immutable input manifests."""
from collections import defaultdict
from pathlib import Path
import csv,json,math
import numpy as np
from build_catalog import ROOT,HERE,OUT,sha,baseprotection

def main():
 rows=list(csv.DictReader(OUT.open(encoding='utf-8-sig')))
 lookup={(r['阶段'],r['方法']):r for r in rows}
 assert len(rows)==len(lookup)==257
 checks=[]
 cases=[
  ('phase2_str_training_20260926','variant=baseline','analysis/variant_seed_split_metrics.csv','variant','baseline'),
  ('phase3b_external_ssl_20260928','variant=b_str_pretrained','analysis/phase3b_seed_split_metrics.csv','variant','b_str_pretrained'),
  ('patch_str_20261008','condition=P2','analysis/seed_split_metrics.csv','condition','P2'),
  ('l1_precision_rebuild_phase9_level2_20261112','condition=WSSL-CERTIFIED-L1','analysis/metrics_by_split_seed.csv','condition','WSSL-CERTIFIED-L1'),
  ('activity_interaction_certified_pilot3_20261112','condition=AI-WSSL-CERTIFIED-L1','analysis/metrics_by_seed.csv','condition','AI-WSSL-CERTIFIED-L1'),
  ('wssl_reference_audit_20261111','condition=relative','analysis/B_metrics.csv','condition','relative')]
 mapping=dict(Accuracy='accuracy',AUROC='auroc',BA='ba',PD_Recall='pd_recall',DD_Recall='dd_recall',Macro_F1='macro_f1')
 for stage,method,file,key,condition in cases:
  with (ROOT/'artifacts'/stage/file).open() as f:data=[r for r in csv.DictReader(f) if r[key]==condition]
  row=lookup[(stage,method)]
  for name,column in mapping.items():
   split=defaultdict(list);seed=defaultdict(list)
   for x in data:
    split[(x.get('context',x.get('outer')),x['inner'])].append(float(x[column]))
    if x.get('seed') not in [None,'','-1','0']:seed[x['seed']].append(float(x[column]))
   a=np.array([np.mean(x) for x in split.values()])
   b=np.array([np.mean(x) for x in seed.values()])
   primary=np.std(a,ddof=1) if len(a)>1 else np.std(b,ddof=1)
   assert math.isclose(float(row[name+'_mean']),float(np.mean(a)),abs_tol=1e-11)
   assert math.isclose(float(row[name+'_SD']),float(primary),abs_tol=1e-11)
   if len(b)>1:
    assert math.isclose(float(row[name+'_seed_SD']),float(np.std(b,ddof=1)),abs_tol=1e-11)
   else:assert row[name+'_seed_SD']=='NA'
  checks.append(dict(stage=stage,method=method,metrics_checked=6,status='PASS'))
 for row in rows:
  for name in mapping:
   m,s=row[name+'_mean'],row[name+'_SD']
   if m=='NA':assert s=='NA'
   else:assert 0<=float(m)<=1 and 0<=float(s)<=1
  assert row['采用目的'] and row['要解决的问题'] and row['达到的结果']
  # Prevent executable spreadsheet formulas in text fields.
  assert not any(v.startswith(('=','+','@')) for v in row.values())
  if row['AUROC_mean']=='NA':assert row['历史判定'].startswith(('NOT_RUN','NON_PERFORMANCE','BLOCKED'))
  if row['development_split数']=='1' and row['seed数']=='3':
   assert '仅一个development split' in row['SD定义']
 for record in json.loads((HERE/'restricted_manifests/input_sources.json').read_text()):
  assert sha(ROOT/record['path'])==record['sha256']
 validation=json.loads((HERE/'analysis/validation.json').read_text())
 assert validation['csv_sha256']==sha(OUT)
 assert validation['protected']==baseprotection()
 result=dict(status='PASS',independent_numpy_checks=checks,
             sample_SD_ddof=1,all_csv_rows_checked=len(rows),
             source_hashes_unchanged=True,NA_not_replaced_by_zero=True,
             duplicate_rows=False,protected_assets_unchanged=True,
             csv_sha256=sha(OUT),new_fits_or_model_inference=0)
 with (HERE/'analysis/independent_verification.json').open('x') as f:
  json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
