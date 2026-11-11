"""Close Level0/1 with immutable-asset and public/private release checks."""
from common9 import *
import ast,re,platform
import PIL.Image

def main():
 reports=load(HERE/'analysis/report_generation_integrity.json');assert reports['all_exist'] and len(reports['required_reports'])==16
 for name in reports['required_reports']:assert (HERE/name).is_file() and (HERE/name).stat().st_size>300
 required=['raw_file_inventory.csv','subject_activity_completeness.csv','signal_quality_summary.csv','signal_anomaly_candidates.csv','signal_quality_by_activity.csv','signal_quality_by_class.csv','temporal_boundary_metrics.csv','sampling_rate_audit.csv','timestamp_quality.csv','bilateral_sync_report.csv','filter_frequency_response.csv','spectral_quality_by_activity.csv','sensor_unit_audit.csv','activity_duration_report.csv','padding_mask_audit.csv','normalization_statistics.csv','subject_overlap_check.csv','class_conditional_data_quality.csv','preprocessing_candidate_priority.csv']
 for name in required:assert (HERE/'analysis'/name).is_file(),name
 assert load(HERE/'tests/correctness_smoke.json')['status']=='PASS';assert load(HERE/'analysis/acceleration_semantics_integrity.json')['raw_full_gravity_vector_presence']=='UNKNOWN'
 freezes=[]
 for p in (HERE/'configs').glob('*_FREEZE.json'):
  base=HERE/'audit/pre_solver_witness_snapshot' if p.name=='CLOSURE_FREEZE.json' else HERE
  for rel,h in load(p)['files'].items():assert sha(base/rel)==h,(p.name,rel)
  freezes.append(p.name)
 for p in (HERE/'scripts').glob('*.py'):ast.parse(p.read_text())
 # Public CSV headers must not expose a subject ID or per-record positions/path/hash.
 blocked=re.compile(r'^(subject_id|patient_id|participant_id|sample_positions_json|time_positions_json|raw_path|processed_path|sha256|file_name|pair_id)$',re.I)
 for p in (HERE/'analysis').glob('*.csv'):
  f=pd.read_csv(p);allow_source_hash=p.name in {'history_sources.csv','source_inventory.csv'}
  if allow_source_hash:assert all(Path(v).suffix in {'.py','.md','.yaml','.yml'} for v in f.path)
  assert not any(blocked.match(c) and not (allow_source_hash and c=='sha256') for c in f.columns),(p.name,f.columns.tolist())
 cross=pd.read_csv(HERE/'analysis/class_quality_by_device_and_source.csv');small=cross.subject_count<10;assert cross.loc[small,['mean','median','p25','p75']].isna().all().all()
 decisions=pd.read_csv(HERE/'analysis/preprocessing_candidate_priority.csv');assert decisions[decisions.candidate=='ACC-INPUT-SEMANTICS'].status.iloc[0]=='AUDIT_REQUIRED_BEFORE_LEVEL2';assert any(term in decisions[decisions.candidate=='PREP-RESAMPLE-BOUNDARY'].stop.iloc[0].lower() for term in ['unknown','unresolved'])
 privatefig=list((HERE/'restricted_manifests/figures').glob('*.png'));publicfig=list((HERE/'figures').glob('*.png'));assert len(privatefig)==44 and len(publicfig)==14
 for p in privatefig+publicfig:
  with PIL.Image.open(p) as im:assert im.width>800 and im.height>300;im.verify()
 assert all(p.name.startswith(('group_quality_','filter_responses_synthetic','resampling_boundary_synthetic','solver_objective_evidence')) for p in publicfig)
 # Source version/failure history retained but not part of executable public entry set.
 assert (HERE/'logs/failed_tests9_v1.py').exists() and (HERE/'logs/tests.log').exists();assert (HERE/'restricted_manifests/drafts_before_acc_semantics/reports9.py').exists()
 w=load(HERE/'analysis/solver_objective_witness_integrity.json');assert w['flagged_axis_records']==20173 and w['affected_wrist_records']==7490 and w['real_L1_reprocessing_calls']==0
 rev=load(HERE/'configs/PREPUBLICATION_REVISION.json')
 for old in rev['snapshot_files']:assert sha(HERE/rev['snapshot_prefix']/old['path'])==old['sha256']
 assert decisions[decisions.candidate=='PREP-L1-PRECISION'].status.iloc[0]=='REVIEW_ELIGIBLE_NUMERICAL_CORRECTNESS'
 for p in HERE.glob('*.md'):
  original=p.read_text();updated=original.replace('公开13图','公开14图').replace('13公开群体/合成图','14公开群体/合成图').replace('公开图13幅','公开图14幅')
  if updated!=original:p.write_text(updated)
 level=HERE/'LEVEL2_EXPERIMENT_PROTOCOL.md';level.write_text(level.read_text().replace('建议审批时明确CPU上限，模型run上限固定45','建议审批上限6 CPU-hours(按单核累计)用于solver QA/候选导出，超出停止；实际新solver耗时未测，模型run上限固定45'))
 corrections=load(HERE/'analysis/engineering_corrections.json');corrections['corrections'].append(dict(stage='final privacy checker',problem='initial blanket SHA column block also rejected legitimate code/report source hashes',resolution='allowlist only source/history code-report hash tables and verify source suffixes; an over-literal UNKNOWN-token assertion was also corrected to accept the explicit unresolved-semantics STOP rule; v1/v2 checker sources and logs retained' ))
 write(HERE/'analysis/engineering_corrections.json',corrections)
 provenance=HERE/'CORRECTNESS_AND_PROVENANCE.md';provenance.write_text(provenance.read_text()+'\n\n发布前检查修正：初次检查把公开代码/报告来源hash也按原始逐文件hash阻止；已仅对两张代码/报告来源表白名单且验证文件类型，未放行raw或逐人hash。原失败检查器与日志保留；完整修正记录见engineering_corrections.json。原始Acc重力语义已收紧为UNKNOWN，较早草稿保留。\n')
 reproduce=HERE/'REPRODUCE.md';reproduce.write_text(reproduce.read_text().replace('finalize9.py','finalize9_v3.py'))
 count=guard();assert count==24180
 freeze('CLOSURE_V2',[HERE/'PROTOCOL.md',Path(__file__)]+[HERE/'scripts'/n for n in ['normalization9.py','supplement9.py','positions9.py','release_privacy9.py','acc_semantics9.py','reports9_v2.py','tests9_v2.py','solver_witness9.py','solver_batch_lineage9.py','revision9.py','solver_plot9.py']]+list(HERE.glob('*.md')),Level0_Level1_only=True,Level2_started=False)
 freezes.append('CLOSURE_V2_FREEZE.json')
 write(HERE/'analysis/phase9_integrity.json',dict(status='PASS_COMPLETE_LEVEL0_LEVEL1_WITH_CONFIRMED_SOLVER_LIMITATION',scientific_revision=2,old_CLOSURE_snapshot_preserved=True,solver_objective_witness_axis_count=20173,solver_objective_witness_wrist_count=7490,numerical_cross_record_dependency_confirmed=True,strict_pipeline_record_independence=False,label_leakage_or_performance_inflation_demonstrated=False,real_L1_reprocessing_calls=0,official_start=START,artifact_directory=str(HERE),required_reports=16,additional_Acc_semantics_report=True,required_public_aggregate_tables=len(required),protected_files_before_after_equal=count,all_protected_files_unchanged=True,raw_wrist_records=8580,sensor_axis_records=51480,temporal_sensor_window_rows=188760,private_fixed_representative_figures=44,public_group_and_synthetic_figures=14,original_source_code_threshold_norm_data_weights_reports_unchanged=True,model_training_updates=0,new_classifier_fits=0,normalization_refits=0,real_model_inference_calls=0,synthetic_checkpoint_model_forward_calls=2,synthetic_model_state_unchanged=True,Level2_started=False,next_stage_started=False,raw_full_gravity_presence='UNKNOWN',Level2_requires_input_contract_and_explicit_user_approval=True,outer_loader_or_performance_decode=False,historical_outer_exposure_disclosed=True,full_cohort_label_quality_access_disclosed=True,global_supervised_or_HarNet_feature_cache_decoded=False,private_signal_plots_or_individual_stats_not_public=True,small_crossed_cell_statistics_suppressed=True,engineering_corrections_preserved=True,frozen_source_sets=sorted(freezes),python=platform.python_version()))
 # Public snapshot hashes contain public code/report/aggregate files only.
 files=[]
 for directory in ['scripts','configs','analysis','audit','tests','figures','reports']:
  for p in sorted((HERE/directory).rglob('*')):
   if p.is_file() and p.name!='source_output_manifest.json':files.append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size))
 for p in sorted(HERE.glob('*.md')):files.append(dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size))
 assert all(x['bytes']<=5*1024*1024 for x in files)
 write(HERE/'analysis/source_output_manifest.json',dict(files=files,privacy='no raw/IDs/individual statistics/private traces/weights/fitted arrays/protected manifests',excluded=['restricted_manifests','logs','features','checkpoints','predictions'],protected_files_unchanged=count,Level2_started=False))
 print(json.dumps(dict(status='PASS_COMPLETE_LEVEL0_LEVEL1',protected_unchanged=count,public_files_manifested=len(files),individual_or_raw_public_files=0,Level2_started=False)))

if __name__=='__main__':main()
