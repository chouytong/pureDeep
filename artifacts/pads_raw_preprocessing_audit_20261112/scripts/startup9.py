"""Code/history first, immutable snapshot second, raw-record metadata last."""
from common9 import *
import ast,subprocess,collections

def main():
 for name in ['scripts','configs','analysis','audit','reports','figures','tests','logs','restricted_manifests/analysis','restricted_manifests/figures']:(HERE/name).mkdir(parents=True,exist_ok=True)
 assert not (HERE/'restricted_manifests/protected_assets.json').exists()
 def git(*args):return subprocess.check_output(['git','-C','/home/zyt/pureDeep-github-upload',*args],text=True).strip()
 assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==START and not git('status','--porcelain')
 protected=dict(load(P8/'restricted_manifests/protected_assets.json')['sha256']);prior=len(protected)
 for p,h in protected.items():assert sha(p)==h
 history=[Path(x['path']) for x in pd.read_csv(P8/'audit/history_sources.csv').to_dict('records')]
 history += [P8/'reports/FINAL_REPORT.md',P8/'reports/CORRECTNESS_AND_PROVENANCE.md',ROOT/'foundation_validation/FOUNDATION_VALIDATION_REPORT.md',Path('/home/zyt/MFAM/docs/preprocessing_v2_report_20260821.md')]
 summary=[];narrative=[]
 for p in history:
  assert p.exists(),str(p);raw=p.read_text();lines=[];hide=False;level=99;suppressed=0
  for line in raw.splitlines():
   if line.startswith('#'):
    d=len(line)-len(line.lstrip('#'))
    if hide and d<=level:hide=False
    if 'outer' in line.lower() and any(v in line.lower() for v in ['result','performance','evaluation','hold','metric']):hide=True;level=d
   if hide or ('outer' in line.lower() and any(v in line.lower() for v in ['0.','accuracy','auroc','balanced accuracy'])):suppressed+=1;continue
   if not line.startswith('|'):lines.append(line)
  narrative.append('FILE '+str(p)+'\n'+'\n'.join(lines));summary.append(dict(path=str(p),sha256=sha(p),status='READ_CURRENT_DEVELOPMENT_TEXT_OUTER_PASSAGES_SUPPRESSED',suppressed_lines=suppressed));protected[str(p)]=sha(p)
 csv(pd.DataFrame(summary),'history_sources.csv');(HERE/'logs/history_development_text.txt').write_text('\n\n'.join(narrative))
 sources=list(MAIN.rglob('*.py'))+list(MAIN.glob('configs/*.yaml'))+[ROOT/'foundation_validation/scripts/prepare_multi_activity_data_v2.py',ROOT/'foundation_validation/scripts/summarize_v2_quality.py',ROOT/'artifacts/phase3b_external_ssl_20260928/scripts/extract_ssl.py',ROOT/'artifacts/phase3b_external_ssl_20260928/scripts/transfer_hooks.py',ROOT/'artifacts/wssl_reference_audit_20261111/scripts/extract_baseline_v2.py',ROOT/'artifacts/wssl_representation_generalization_phase6_20261111/scripts/external_inventory.py',Path('/home/zyt/pads/scripts/run_preprocessing.py'),Path('/home/zyt/pads/scripts/utils/data_handling.py'),Path('/home/zyt/pads/scripts/utils/l1_trend_filter.py')]
 src=[];functions=[]
 for p in sources:
  assert p.exists(),str(p);text=p.read_text();protected[str(p)]=sha(p);src.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,full_text_scanned=True))
  if p.suffix=='.py':
   tree=ast.parse(text)
   for n in ast.walk(tree):
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):functions.append(dict(path=str(p),name=n.name,line=n.lineno,end_line=n.end_lineno))
  if p.is_relative_to(MAIN):assert sha(Path('/home/zyt/pureDeep-github-upload')/p.relative_to(ROOT))==sha(p)
 csv(pd.DataFrame(src),'source_inventory.csv');csv(pd.DataFrame(functions),'source_function_index.csv')
 # All relevant existing data and Phase8 outputs: no signal arrays decoded here.
 for directory in [ROOT/'data',P8,MAIN]:
  for p in directory.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
 write(HERE/'restricted_manifests/protected_assets.json',dict(sha256=protected,previous_chain_files=prior,scope='P8 protection chain + all current data/mainline/P8 outputs + audited historical sources'))
 # Raw subject/session metadata is inspected only after the code/history gate above.
 subjects={};conditions=collections.Counter();reps={}
 for p in sorted((RAW/'patients').glob('patient_*.json')):
  d=load(p);sid=str(d['id']);condition=d['condition'];assert sid not in subjects
  label='PD' if condition=="Parkinson's" else 'DD' if condition in ['Other Movement Disorders','Essential Tremor','Atypical Parkinsonism','Multiple Sclerosis'] else 'EXCLUDED_OTHER';subjects[sid]=dict(label=label,source_condition=condition);conditions[condition]+=1
 assert len(subjects)==390 and sum(s['label']=='PD' for s in subjects.values())==276 and sum(s['label']=='DD' for s in subjects.values())==114
 for label in ['PD','DD']:reps[label]=sorted(s for s,d in subjects.items() if d['label']==label)[:2]
 write(HERE/'restricted_manifests/representatives_BEFORE_SIGNAL_READ.json',dict(rule='lexicographically first2IDs per class; same IDs all activities; no signal/results selection',subject_ids=reps))
 rec=[];complete=[];observed=collections.Counter();units=collections.Counter();devices=collections.Counter();unknown=[]
 for path in sorted((RAW/'movement').glob('observation_*.json')):
  obs=load(path);sid=str(obs['subject_id']);assert sid in subjects;observed[sid]+=1;devices[str(obs.get('device_id','UNKNOWN'))]+=1
  for activity in ACTS:
   sessions=[s for s in obs.get('session',[]) if s.get('record_name')==activity];sides=collections.Counter()
   for session in sessions:
    for entry in session.get('records',[]):
     location=entry.get('device_location');side={'LeftWrist':'left','RightWrist':'right'}.get(location)
     if side is None:unknown.append(dict(subject_id=sid,activity=activity,reason='unknown_location'));continue
     sides[side]+=1;rawpath=(RAW/'movement'/entry['file_name']).resolve();unit=entry.get('units',[]);channels=entry.get('channels',[]);units[json.dumps(unit)]+=1
     rec.append(dict(subject_id=sid,activity=activity,wrist=side,raw_path=str(rawpath),processed_path=str(PROC/'signals'/activity/sid/f'{side}.npy'),label=subjects[sid]['label'],source_condition=subjects[sid]['source_condition'],nominal_fs=obs.get('sampling_rate'),metadata_rows=session.get('rows'),metadata_units=unit,metadata_channels=channels,device=str(obs.get('device_id','UNKNOWN'))))
   complete.append(dict(subject_id=sid,activity=activity,session_count=len(sessions),left_count=sides['left'],right_count=sides['right'],channels_6=all(len(x.get('channels',[]))==6 for s in sessions for x in s.get('records',[])),**subjects[sid]))
 write(HERE/'restricted_manifests/records.json',dict(records=rec));write(HERE/'restricted_manifests/subject_metadata.json',dict(subjects=subjects,observation_counts=dict(observed)));csv(pd.DataFrame(complete),'subject_activity_completeness.csv',True)
 assert set(observed)==set(subjects) and all(n==1 for n in observed.values()) and len(rec)==8580 and not unknown
 write(HERE/'analysis/startup_integrity.json',dict(status='PASS_CODE_FIRST_BEFORE_SIGNAL_DIAGNOSTICS',official_start=START,history_files=len(history),source_files=len(sources),protected_files=len(protected),task_subject_count=390,PD_count=276,DD_count=114,source_condition_counts=dict(conditions),units_metadata_counts=dict(units),device_metadata_counts=dict(devices),raw_arrays_decoded_by_startup=0,outer_performance_or_loader=False,new_fits=0,normalization_refits=0))
 freeze('DIAGNOSTICS',[HERE/'PROTOCOL.md',HERE/'scripts/common9.py',Path(__file__),HERE/'scripts/diagnostics9.py'],before_new_signal_diagnostics=True,representative_rule_frozen=True,comparison_families=[132,440,880],training_budget=0)
 print(json.dumps(dict(status='PASS_CODE_FIRST',protected=len(protected),records=len(rec),units=dict(units),devices=dict(devices))))

if __name__=='__main__':main()
