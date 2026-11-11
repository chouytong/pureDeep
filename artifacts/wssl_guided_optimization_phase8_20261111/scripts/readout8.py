from common8 import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import joblib,warnings,importlib.util

def parameter_count(model):
 if hasattr(model,'coef_'):return int(model.coef_.size+model.intercept_.size)
 if hasattr(model,'support_vectors_'):return int(model.support_vectors_.size+model.dual_coef_.size+model.intercept_.size)
 if hasattr(model,'estimators_'):return int(sum(t.tree_.node_count*4 for t in model.estimators_))
 return None
def main():
 assert load(HERE/'analysis/ensemble_integrity.json')['status']=='PASS_COMPLETE_A1';guard()
 freeze('A2',[HERE/'PROTOCOL.md',HERE/'scripts/common8.py',Path(__file__)],A1_report_archived=True,A2_results_seen=False,primary_representation='subject258+structured176',primary_family=10,direct_RBF_minus_Linear_exploratory_family=9,head_replay_tolerance=1e-5,SVC_probability_internal_train_only=True)
 rows=[];integrity=[];parameters=[];lineage=[];start=time.perf_counter();warnings.simplefilter('error',ConvergenceWarning)
 with threadpool_limits(limits=4):
  for c,i,sp,ex in splits():
   for seed in [42,43,44]:
    f=features(seed,c,i);v=f['is_validation'];x=np.concatenate([f['subject'],f['structured']],1).astype(float);y=f['target'];ids=f['subject_ids'];assert x.shape[1]==434
    ids_ok(ids[~v],sp['train_subjects']);ids_ok(ids[v],sp['validation_subjects']);assert not set(ids[~v])&set(ids[v])
    ck=torch.load(baseline(seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False);state=ck['model_state']
    raw=f['subject'].astype(float)@state['backbone.classifier.1.weight'].numpy().astype(float).T+state['backbone.classifier.1.bias'].numpy().astype(float)+f['structured'].astype(float)@state['backbone.structured_token_residual.residual_head.weight'].numpy().astype(float).T+state['backbone.structured_token_residual.residual_head.bias'].numpy().astype(float)
    head_error=float(abs(raw-f['logits']).max());assert head_error<=1e-5
    scaler=StandardScaler().fit(x[~v]);assert scaler.n_samples_seen_==sum(~v) and np.allclose(scaler.mean_,x[~v].mean(0),atol=1e-12,rtol=0)
    train=scaler.transform(x[~v]);query=scaler.transform(x[v]);original=x.copy();before_mean=scaler.mean_.copy();before_scale=scaler.scale_.copy()
    models={'logistic_regression':LogisticRegression(C=1.,penalty='l2',class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=seed),'linear_svm':SVC(C=1.,kernel='linear',class_weight='balanced',probability=True,random_state=seed),'rbf_svm':SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,random_state=seed),'random_forest':RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=4)}
    if importlib.util.find_spec('xgboost'):
     from xgboost import XGBClassifier
     models['xgboost']=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.05,subsample=1.,colsample_bytree=1.,reg_lambda=1.,min_child_weight=5,random_state=seed,n_jobs=4,eval_metric='logloss')
    for name,model in models.items():
     t=time.perf_counter();model.fit(train,y[~v]);fit_time=time.perf_counter()-t;assert list(model.classes_)==[0,1]
     t=time.perf_counter();p=model.predict_proba(query);predict_time=time.perf_counter()-t;assert np.array_equal(p,model.predict_proba(query)) or np.allclose(p,model.predict_proba(query),atol=1e-12,rtol=0)
     assert np.array_equal(before_mean,scaler.mean_) and np.array_equal(before_scale,scaler.scale_) and np.array_equal(original,x)
     path=HERE/'checkpoints'/f'{name}_seed{seed}_context{c}_inner{i}.joblib';assert not path.exists();joblib.dump(dict(scaler=scaler,classifier=model),path)
     reloaded=joblib.load(path);assert np.allclose(p,reloaded['classifier'].predict_proba(reloaded['scaler'].transform(x[v])),atol=1e-12,rtol=0)
     rows.append(save_prediction(name,seed,c,i,ids[v],y[v],p));parameters.append(dict(condition=name,seed=seed,context=c,inner=i,feature_dim=434,parameters_json=json.dumps(model.get_params(),sort_keys=True),fitted_numeric_parameter_count=parameter_count(model),classifier_fit_seconds=fit_time,query_predict_seconds=predict_time,query_count=int(sum(v)),model_bytes=path.stat().st_size))
    integrity.append(dict(seed=seed,context=c,inner=i,status='PASS',fit_count=int(sum(~v)),query_count=int(sum(v)),feature_dim=434,source_feature_sha256=sha(featurepath(seed,c,i)),source_checkpoint_sha256=sha(baseline(seed,c,i)/'checkpoints/best.pt'),head_logits_max_error=head_error,scaler_fit_train_only=True,validation_input_unmutated=True,probability_repeat_and_reload=True))
    lineage.append(dict(seed=seed,context=c,inner=i,fit_subject_ids=ids[~v].tolist(),query_subject_ids=ids[v].tolist(),excluded_subject_ids=ex,feature_sha256=sha(featurepath(seed,c,i)),normalization_sha256=sha(baseline(seed,c,i)/'normalization.json'),classifier_source_sha256=sha(Path(__file__)),SVC_probability_CV='only434D supervised in-sample train embeddings; not encoder-OOF'))
   print(json.dumps(dict(stage='A2',context=c,inner=i,completed_models=len(rows),elapsed_seconds=time.perf_counter()-start)),flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(HERE/'analysis/readout_metrics.csv',index=False);pd.DataFrame(parameters).to_csv(HERE/'analysis/readout_parameters.csv',index=False);pd.DataFrame(integrity).to_csv(HERE/'analysis/readout_feature_integrity.csv',index=False);write(HERE/'restricted_manifests/readout_lineage.json',dict(records=lineage))
 base=pd.read_csv(HERE/'analysis/ensemble_metrics.csv');base=base[base.condition=='baseline'];stats=[];decisions=[]
 for name in ['logistic_regression','linear_svm','rbf_svm','random_forest','xgboost']:
  f=frame[frame.condition==name]
  if len(f)==0:
   stats.extend(dict(condition=name,metric=m,status='NOT_RUN_DEPENDENCY_MISSING',wilcoxon_p=np.nan) for m in METRICS);decisions.append(dict(condition=name,status='NOT_RUN_DEPENDENCY_MISSING'));continue
  s=paired(f,base,name);stats.extend(s.to_dict('records'));decisions.append(gate(f,s,base,name))
 stat=pd.DataFrame(stats);stat['primary_bh_q']=np.nan;ix=stat.metric.isin(['ba','auroc']);assert ix.sum()==10;stat.loc[ix,'primary_bh_q']=bh_planned(stat.loc[ix,'wilcoxon_p']);stat.to_csv(HERE/'analysis/readout_paired_statistics.csv',index=False)
 direct=paired(frame[frame.condition=='rbf_svm'],frame[frame.condition=='linear_svm'],'rbf_minus_linear');direct['exploratory_bh9_q']=bh_planned(direct.wilcoxon_p);direct.to_csv(HERE/'analysis/readout_RBF_minus_Linear.csv',index=False)
 means=frame.groupby('condition')[METRICS].mean().reset_index();(HERE/'reports/TRADITIONAL_READOUT_REPORT.md').write_text('# Phase8A2 固定监督表征的传统读出\n\n唯一主要表示为原subject258+structured176=434D，原两个head恰为此空间的affine读出；没有验证挑层。只在各模型train fit Scaler/classifier，未更新任何encoder；SVC含train-only内部Platt，不将其说成encoder全链OOF。一个患者一行，不跨fold拼接。\n\n'+table(means)+'\n\n'+table(stat[['condition','metric','mean_delta','wins','ci_low','ci_high','primary_bh_q']])+'\n\n'+''.join(f"{r['condition']}: **{r['status']}**。\n\n" for r in decisions)+'本比较不能证明线性可分性的数学上限，也不能把固定核/概率拟合/正则化的差异全部归因非线性。RBF−Linear另有预先声明的直接敏感性；树模型即使改变分数也不是识别到具体活动因果关系。XGBoost如缺依赖不安装，主族仍10。共同9指标、原始参数和耗时、完整paired表及最后患者cluster敏感性均保留；不因单seed或单最高值选模型。A2阶段归档后才进入B。\n')
 write(HERE/'analysis/readout_integrity.json',dict(status='PASS_COMPLETE_A2',model_pairs=45,final_classifier_fits=len(frame),scaler_fits=45,feature_dim=434,encoder_updates=0,outer_inputs=0,head_replay_max_error=max(r['head_logits_max_error'] for r in integrity),SVC_train_internal_probability_estimation=True,xgboost_status='RUN' if 'xgboost' in set(frame.condition) else 'NOT_RUN_DEPENDENCY_MISSING',decisions=decisions,elapsed_seconds=time.perf_counter()-start,report_sha256=sha(HERE/'reports/TRADITIONAL_READOUT_REPORT.md')))
 guard();print(json.dumps(clean(dict(stage='A2_COMPLETE',means=means.to_dict('records'),decisions=decisions)),ensure_ascii=False))
if __name__=='__main__':main()
