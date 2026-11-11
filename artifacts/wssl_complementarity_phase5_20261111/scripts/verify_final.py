"""Final provenance, source policy, metric identity and deliverable verification."""
from common import *
import ast,datetime,subprocess

def main():
    lock=guard();frozen=load(HERE/'configs/ANALYSIS_FREEZE.json')
    for name,digest in frozen['source_sha256'].items():assert sha(HERE/name)==digest,name
    for version in [2,3,4]:
        f=load(HERE/'configs'/f'AUDIT_V{version}_FREEZE.json');assert sha(HERE/f['active_script'])==f['sha256']
    active=['common.py','bootstrap.py','audit_assets_v4.py','audit_lineage_supplement.py','analyze.py','cluster_statistics.py','build_reports.py','plot_aggregates.py','verify_final.py']
    prohibited=[]
    for name in active:
        tree=ast.parse((HERE/'scripts'/name).read_text(),filename=name)
        if name not in ['build_reports.py','plot_aggregates.py','verify_final.py']:
          for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                attr=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                if attr in ['fit','fit_transform','train_epoch','evaluate_epoch','build_model','DataLoader','backward','step','harnet10']:prohibited.append([name,node.lineno,attr])
    assert not prohibited,prohibited
    required=['PROTOCOL.md','SCORE_DISTRIBUTION_REPORT.md','SUBJECT_COMPLEMENTARITY_REPORT.md','REFERENCE_COVERAGE_REPORT.md','FUSION_FAILURE_ANALYSIS.md','ADAPTATION_SUBJECT_ANALYSIS.md','REPRESENTATION_DRIFT_REPORT.md','FINAL_REPORT.md','ASSET_AUDIT.md','NEXT_STAGE_CANDIDATES.md','STATISTICAL_APPENDIX.md','REPRODUCE.md','decision.json','analysis/score_distribution.csv','analysis/subject_complementarity.csv','analysis/reference_coverage.csv','analysis/phase5a_integrity.json','analysis/fusion_parameter_audit.csv','analysis/oof_full_distribution_shift.csv','analysis/fusion_class_bias.csv','analysis/adaptation_error_transitions.csv','analysis/representation_drift.csv']
    for n in required:assert (HERE/n).stat().st_size>0,n
    a=load(HERE/'analysis/asset_audit.json');p=load(HERE/'analysis/phase5a_integrity.json');c=load(HERE/'analysis/cluster_statistics_integrity.json');d=load(HERE/'decision.json')
    assert a['model_assets']==225 and a['reference_fit_ID_records']==60
    lineage=load(HERE/'analysis/fusion_lineage_integrity.json');assert lineage['fusion_fit_records']==45 and lineage['matched_reference_deep_OOF_partitions']==135 and lineage['all_target_validation_subjects_excluded']
    assert p['validation_units']==45 and p['full_saved_fusion_probability_max_error']<=1e-12 and p['ID_label_alignment']
    assert c['point_estimates_match_formal_seed_first'] and c['joint_all_appearances_and_seeds']
    assert not d['retained_config_changed'] and d['new_deep_training_runs']==0 and d['new_lightweight_classifier_fits']==0 and d['new_neural_inference']==0
    t=pd.read_csv(HERE/'analysis/predeclared_tests.csv');expected={'A_primary':4,'A_correlations':36,'A_coverage':3,'B_mechanism':4};assert t.groupby('family').size().to_dict()==expected
    for name,n in expected.items():assert t[t.family==name].units.eq(15).all()
    drift=pd.read_csv(HERE/'analysis/representation_drift.csv');assert len(drift)==270 and drift[['cosine_similarity','L2_displacement','frozen_norm','adapted_norm']].isna().all().all() and not drift.feature_reextracted.any()
    failures=[]
    for version,log,reason in [(1,'asset_audit.log','one unmatched closing parenthesis, stopped at Python parse'),(2,'asset_audit_v2.log','wrong assumption that historical metadata test IDs were empty'),(3,'asset_audit_v3.log','wrong assumption label_mapping carried numeric rather than clinical-source mapping')]:
        text=(HERE/'logs'/log).read_text();assert 'Error' in text
        failures.append(dict(version=version,status='FAILED_AND_PRESERVED',reason=reason,log_sha256=sha(HERE/'logs'/log),scientific_tolerance_changed=False,historical_asset_changed=False))
    write_json(HERE/'tests/audit_failure_ledger.json',dict(failures=failures,active_version=4,all_corrected_before_new_phase5_statistics=True,do_not_run_inactive_versions=True))
    write_json(HERE/'tests/final_correctness.json',dict(status='AVAILABLE_ANALYSES_PASS_WITH_BLOCKED_CAPABILITIES',active_script_syntax=True,original_source_registration_unchanged=True,negative_ID_tests=load(HERE/'tests/synthetic_correctness.json'),no_fit_or_neural_execution_calls_in_active_analysis=True,all_225_hash_ID_label_sources_verified=True,all_45_fusion_probability_decisions_recovered=True,bootstrap_point_estimands_match_canonical=True,predeclared_BH_family_sizes=expected,representation_missing_not_filled_zero=True,all_requested_subanalyses_complete=False,reference_transform_full_numeric_replay_pass=False,adapted_representation_drift_pass=False))
    doc='''# Correctness 与 provenance 最终审计

已通过的范围：225个既有模型的当前hash/训练query排除ID/标签/归一化来源；45个Frozen特征包保存概率与归档预测一致；45个既有fusion的输入/Scaler/系数线性重放，最大概率误差2.22e-16且全部原决策一致；所有模型六指标与归档一致（容差1e-12）；ID重复/allowlist负测试；canonical加权混淆矩阵/AUC的bootstrap恒等；15unit seed-first与subject-cluster点估计一致；4/36/3/4四个BH族大小保持；缺失表征数据为NA。

不能宣称全部研究能力通过：reference逐人OOF与完整Scaler/PCA/bank状态MISSING，无法独立重放完整参考拟合链；adapted成对表征及raw双logitsMISSING，不能计算cosine/L2/norm或raw DD logit drift。旧feature包缺创建时逐特征hash，当前依据来源代码、当前锁和概率一致性，未重新推理独立验证embedding。旧TF32梯度精度失败不因本轮纯分数审计而变成通过，CE仍NOT_RUN_GATE_FAIL。

失败修订如实保留：v1有一处多余括号，未进入执行；v2错误要求历史split里的test ID元数据为空，改为与canonical excluded名单完全相等；v3误把临床来源label_mapping当数值编码，v4同时核验class_names=[PD,DD]与临床映射。没有改数值容差、评分算法、协议或允许范围；三次修订在新Phase5统计前完成，原脚本/hash/日志保留。audit_assets.py是保留的失败快照，active入口为audit_assets_v4.py；不能声称所有历史脚本均syntax PASS。

静态检查active分析入口无fit/fit_transform、model build、DataLoader、train/eval epoch、backward/optimizer step或harnet调用。运行只读既有分数/参数，神经网络实例化0、训练0、推理0、新分类器/温度/Scaler/PCA拟合0、阈值/系数搜索0。该源代码检查不是系统级I/O隔离证明。

永久保留上轮outer汇总文字接触和整表/整数组读取偏差。本轮不解码全球缓存、不读原始IMU、不新建outer loader、不读outer performance。文件hash读取完整字节；CSV逐行filter也不等于物理零无关行读取。不把重复开发、seed/split波动和固定模型cluster CI冒充独立外部验证。

受保护资产清单和全部fit/query/excluded ID在restricted_manifests，禁止上传。公开仅聚合计数、审计状态和来源hash。当前mainline/历史正式产物起止hash不变，当前retained模型不变。最终发布receipt在/home/zyt/下独立保存，不修改源报告自指提交SHA。
'''
    (HERE/'CORRECTNESS_AND_PROVENANCE.md').write_text(doc)
    manifest={}
    for path in HERE.rglob('*'):
        if not path.is_file() or any(part in path.parts for part in ['restricted_manifests','logs','__pycache__']):continue
        if path.name in ['source_output_manifest.json','final_integrity.json']:continue
        manifest[str(path.relative_to(HERE))]=dict(sha256=sha(path),bytes=path.stat().st_size)
    write_json(HERE/'analysis/source_output_manifest.json',dict(files=manifest,contains_private_ID_or_protected_asset_manifests=False,git_start_sha=lock['git_start_sha']))
    guard();write_json(HERE/'analysis/final_integrity.json',dict(status='IMMUTABLE_ASSETS_PASS_AVAILABLE_ANALYSES_COMPLETE_WITH_BLOCKED_SUBANALYSES',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protected_files=len(lock['sha256']),all_start_end_protected_hashes_unchanged=True,source_registration_hashes_unchanged=True,active_correctness_checks_pass=True,all_requested_capabilities_available=False,reference_state_complete=False,adapted_representation_available=False,retained_config_changed=False,new_deep_training=0,new_light_fits=0,new_neural_inference=0,old_outer_exposure_permanently_retained=True,new_outer_evaluation=False,independent_external_validation=False,publication_receipt='outside source tree after exact-tree GitHub verification'))
    print(json.dumps(dict(status='PHASE5_FINAL_VERIFIED_WITH_EXPLICIT_BLOCKED_SUBANALYSES',protected_files=len(lock['sha256']),new_fits=0,new_training=0,new_neural_inference=0)),flush=True)

if __name__=='__main__':main()
