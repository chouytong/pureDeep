"""Append the requested aggregate catalog and publish only audited safe artifacts."""
from common10 import *
import argparse,csv,importlib.util,subprocess
UP=Path('/home/zyt/pureDeep-github-upload');PREFIX='artifacts/phase10_task_adaptive_demographic_20261118/'
CAT='conclusions/EXPERIMENT_RESULTS_ALL_20261118.csv'

def update_catalog(stage,metric_file,status,purpose,result):
 assert stage in ['A','B','C','D','FINAL']
 marker=HERE/'audit'/f'catalog_{stage}.json';assert not marker.exists()
 old=ROOT/'conclusions/EXPERIMENT_RESULTS_ALL_20261116.csv';out=ROOT/CAT
 extra=[x+s for x in ['NLL','Brier','ECE'] for s in ['_mean','_SD','_seed_SD']]+['Total_Params','Trainable_Params','Frozen_Params','Training_Time_seconds','Peak_GPU_memory_bytes','Inference_Time_seconds']
 if not out.exists():
  with old.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);columns=list(r.fieldnames)+extra;rows=list(r)
  with out.open('x',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows([{k:x.get(k,'NA') for k in columns} for x in rows])
 with out.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);columns=list(r.fieldnames);existing=list(r)
 before=out.read_bytes();frame=pd.read_csv(HERE/metric_file);summ=summary(frame)
 mapping={'Accuracy':'accuracy','AUROC':'auroc','BA':'ba','PD_Recall':'pd_recall','DD_Recall':'dd_recall','Macro_F1':'macro_f1','NLL':'nll','Brier':'brier','ECE':'ece'}
 rows=[]
 for _,s in summ.iterrows():
  row={k:'NA' for k in columns};condition=s['condition'];g=frame[frame.condition==condition]
  row.update(阶段=f'phase10{stage}_20261118',方法=condition,实验角色='development_pilot' if int(s.splits)==1 else 'development',数据版本=VERSION,
             采用目的=purpose,要解决的问题='Certified L1上检验预登记单因素假设；不将旧数据结果当新性能',达到的结果=result,历史判定=status,
             development_split数=int(s.splits),seed数=int(s.seeds),seed列表=';'.join(map(str,sorted(set(g.seed)-{-1}))) or 'NA',已有统计记录数=int(s.runs),
             指标单位='比例[0,1]',SD定义='同split先平均seed指标，再跨split sample SD(ddof=1)' if s.splits>1 else '单split pilot跨3seed sample SD(ddof=1)，非跨split波动',
             缺失值与限制='重复development；15split非15独立临床队列；pilot不作独立确认',指标来源筛选='condition='+condition,
             来源文件=PREFIX+metric_file,来源报告=PREFIX+'reports/PHASE10'+stage+('_DIAGNOSTIC_REPORT.md' if stage=='A' else '_UNFREEZING_REPORT.md' if stage=='B' else '_DEMOGRAPHIC_REPORT.md' if stage=='C' else '_DD_LOSS_REPORT.md'),
             来源提交=BASE,来源链接='https://github.com/chouytong/pureDeep/tree/main/'+PREFIX)
  for name,m in mapping.items():
   for suffix,key in [('_mean',m),('_SD',m+'_SD'),('_seed_SD',m+'_seed_SD')]:
    v=s[key];row[name+suffix]=format(float(v),'.12g') if pd.notna(v) else 'NA'
  if condition=='WSSL-CERTIFIED-L1':row.update(Total_Params=10600580,Trainable_Params=143172,Frozen_Params=10457408)
  if condition=='STR-CERTIFIED-L1':row.update(Total_Params=75524,Trainable_Params=75524,Frozen_Params=0)
  if condition in ['WSSL-CERTIFIED-L1','STR-CERTIFIED-L1']:row['历史判定']='MATCHED_CERTIFIED_REFERENCE_REUSED'
  if condition=='D0-DEMOGRAPHIC-ONLY':row.update(Total_Params=5,Trainable_Params=5,Frozen_Params=0,seed数=0,seed列表='NA',缺失值与限制='15个确定性train-only LR fit；seed_SD不可估计，不复制为45独立fit')
  if 'source_commit' in g and g.source_commit.notna().any():row['来源提交']=str(g.source_commit.iloc[0])
  costs=HERE/'analysis'/f'{stage.lower()}_costs.csv'
  if costs.exists():
   co=pd.read_csv(costs);co=co[co.condition==condition]
   if len(co):
    for dst,src in [('Total_Params','total_parameters'),('Trainable_Params','trainable_parameters'),('Frozen_Params','frozen_parameters')]:row[dst]=int(co[src].iloc[0])
    row['Training_Time_seconds']=co.wall_seconds.sum();row['Peak_GPU_memory_bytes']=int(co.peak_GPU_allocated_bytes.max())
    if 'cached_path_validation_inference_seconds' in co:row['Inference_Time_seconds']=co.cached_path_validation_inference_seconds.mean()
  rows.append(row)
 with out.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=columns).writerows(rows)
 assert out.read_bytes().startswith(before)
 write(marker,dict(stage=stage,added_rows=len(rows),total_rows=len(existing)+len(rows),sha256=sha(out),historical_257_rows_preserved=True),new=True)
 print(json.dumps(load(marker)),flush=True)

def publisher():
 s=importlib.util.spec_from_file_location('existing_publisher',ROOT/'tools/git_publish/publish_to_github.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p);return p
def extpath(stage,kind):return Path('/home/zyt')/f'pureDeep-phase10-{stage}-{kind}-20261118.json'
class ConnectedTransport(Exception):pass
def prepare(stage):
 assert stage in ['A','B','C','D','FINAL','PREB']
 guard();p=publisher();original=p.git;payloadpath=extpath(stage,'payload');assert not payloadpath.exists()
 assert not original('status','--porcelain').stdout
 parent=original('rev-parse','HEAD').stdout.strip();base_tree=original('rev-parse','HEAD^{tree}').stdout.strip()
 original('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main')
 assert original('rev-parse','refs/remotes/origin/main').stdout.strip()==parent
 selected,skipped=p.collect();old={r['path']:r for r in load(UP/'PUBLICATION_MANIFEST.json')['files']}
 changed=[n for n,(_,h,z) in selected.items() if not (UP/n).exists() or sha(UP/n)!=h]
 assert changed and all(n.startswith(PREFIX) or n==CAT for n in changed),changed
 assert not set(old)-set(selected)
 for n,record in old.items():
  if n not in changed:assert selected[n][1:]==(record['sha256'],record['bytes'])
 # No data content under a generic private/ alias can enter the public tree.
 assert not any(n.startswith(PREFIX+'private/') or n.startswith(PREFIX+'restricted_manifests/') for n in selected)
 print(json.dumps(dict(status='PREVIEW_PASS',phase=stage,changed=len(changed),selected_files=len(selected),parent=parent)),flush=True)
 if '--prepare' not in sys.argv:return
 def git(*args,check=True):
  if args[0]=='fetch':return original('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main',check=check)
  if args[0]=='push':raise ConnectedTransport()
  return original(*args,check=check)
 p.git=git;message=f'Phase 10{stage}: Certified L1 task-adaptive and demographic study evidence'
 sys.argv=['publish_to_github.py','--apply','--message',message]
 try:p.main()
 except ConnectedTransport:pass
 else:raise AssertionError('Expected exact-tree connector transport')
 assert not original('status','--porcelain').stdout
 entries=[];names=original('diff','--name-only','HEAD^','HEAD').stdout.splitlines()
 assert set(names)==set(changed)|{'PUBLICATION_MANIFEST.json'}
 for name in names:
  mode,kind,digest=original('ls-tree','HEAD','--',name).stdout.split('\t')[0].split();assert kind=='blob'
  entries.append(dict(path=name,mode=mode,type=kind,sha=digest,sha256=sha(UP/name),bytes=(UP/name).stat().st_size))
 d=dict(parent=parent,base_tree=base_tree,tree=original('rev-parse','HEAD^{tree}').stdout.strip(),local_commit=original('rev-parse','HEAD').stdout.strip(),message=message,entries=entries)
 write(payloadpath,d,new=True);print(json.dumps(d,indent=2),flush=True)
def verify(stage,remote):
 p=publisher();d=load(extpath(stage,'payload'));receipt=extpath(stage,'receipt');assert not receipt.exists()
 assert not p.git('status','--porcelain').stdout
 p.git('-c','credential.helper=','-c','core.askPass=/bin/false','fetch','--no-tags','origin','main')
 assert p.git('rev-parse','refs/remotes/origin/main').stdout.strip()==remote
 assert p.git('rev-parse',remote+'^{tree}').stdout.strip()==d['tree']
 assert p.git('show','-s','--format=%P',remote).stdout.strip()==d['parent']
 current=p.git('rev-parse','HEAD').stdout.strip();assert current==d['local_commit']
 archive=f'refs/heads/archive/phase10-{stage}-unpublished-local-20261118'
 assert p.git('show-ref','--verify','--quiet',archive,check=False).returncode!=0
 p.git('update-ref',archive,current,'0'*40);p.git('update-ref','refs/heads/main',remote,current)
 p.audit_tracked();selected,skipped=p.collect();man=load(UP/'PUBLICATION_MANIFEST.json');assert {x['path'] for x in man['files']}==set(selected)
 for r in man['files']:
  assert selected[r['path']][1:]==(r['sha256'],r['bytes'])
  assert sha(UP/r['path'])==sha(selected[r['path']][0])==r['sha256']
 assert not p.git('status','--porcelain').stdout
 assert p.git('rev-parse','HEAD').stdout.strip()==remote
 assert p.git('-c','credential.helper=','-c','core.askPass=/bin/false','ls-remote','origin','refs/heads/main').stdout.split()[0]==remote
 result=dict(status='PUSH_VERIFIED_EXACT_AUDITED_TREE',phase=stage,commit=remote,parent=d['parent'],tree=d['tree'],selected_files_verified=len(selected),
             source_mirror_manifest_match=True,mirror_head_equals_remote_main=True,mirror_worktree_clean=True,
             original_publisher_used=True,protected_files_verified=guard(),catalog_sha256=sha(ROOT/CAT),
             individual_records_exported=False,current_model_pointer_unchanged=True,timestamp_utc=utc())
 write(receipt,result,new=True);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['catalog','publish','verify']);ap.add_argument('--stage',required=True);ap.add_argument('--prepare',action='store_true');ap.add_argument('--remote');ap.add_argument('--metrics');ap.add_argument('--status',default='DIAGNOSTIC_ONLY');a=ap.parse_args()
 if a.action=='catalog':update_catalog(a.stage,a.metrics,a.status,'Phase10单因素研究：任务适配、人口统计辅助与DD代价','详见相应Phase10阶段报告；pilot仅作资源筛选，未自动修改正式模型')
 elif a.action=='publish':prepare(a.stage)
 else:verify(a.stage,a.remote)

