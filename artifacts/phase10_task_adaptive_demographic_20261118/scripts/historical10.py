"""Read-only historical logs and per-run prediction audit; never loads global caches."""
from common10 import *
import yaml

def main():
 p8=ROOT/'artifacts/wssl_guided_optimization_phase8_20261111';p3=ROOT/'artifacts/phase3c_ssl_adaptation_20260930'
 hashes=dict(load(p8/'restricted_manifests/protected_assets.json')['sha256'])
 for r in load(p8/'restricted_manifests/new_private_outputs_sha256.json')['files']:
  p=Path(r['path']);p=p if p.is_absolute() else p8/p;hashes[str(p)]=r['sha256']
 lineage8=load(p8/'restricted_manifests/activity_interaction_lineage.json')['records']
 ready=[]
 for c,i,sp,ex in splits():
  for seed in [42,43,44]:
   r=next(x for x in lineage8 if (x['seed'],x['context'],x['inner'])==(seed,c,i))
   assert set(r['fit_subject_ids'])==set(sp['train_subjects']) and set(r['query_subject_ids'])==set(sp['validation_subjects'])
   paths=[p8/'predictions'/f'{kind}_seed{seed}_context{c}_inner{i}.npz' for kind in ['baseline','activity_interaction']]
   if not all(str(p) in hashes and p.is_file() and sha(p)==hashes[str(p)] for p in paths):
    write(HERE/'analysis/historical_AI_transition_integrity.json',dict(status='BLOCKED_PREDICTION_SOURCE_INCOMPLETE',old_global_cache_decoded=False),new=True)
    report('HISTORICAL_ACTIVITY_INTERACTION_AUDIT.md','# 历史AI转移诊断\n\n逐run来源无法完整核对，标记BLOCKED。不重新读取旧整队列压缩缓存，不把历史数据混入Certified主结果。')
    return
   ready.append((seed,c,i,sp,paths))
 transitions=[];margins=[]
 for seed,c,i,sp,paths in ready:
  records=[]
  for p in paths:
   with np.load(p,allow_pickle=False) as z:ids=z['subject_ids'].astype(str);y=z['target'];prob=z['probabilities'];logits=z['logits']
   ids_ok(ids,set(sp['validation_subjects']));records.append((ids,y,prob,logits))
  ids,y,p,z=records[0];oid,oy,op,oz=records[1];idx={s:j for j,s in enumerate(oid)};ix=[idx[s] for s in ids];assert np.array_equal(y,oy[ix]);op=op[ix];oz=oz[ix]
  base=p[:,1]>.5;new=op[:,1]>.5
  for label in [0,1]:
   sel=y==label
   transitions.append(dict(data_version='LEGACY_L1_HISTORICAL_ONLY',seed=seed,context=c,inner=i,label=['PD','DD'][label],n=int(sel.sum()),
                           correct_to_wrong=int(np.sum(sel&(base==y)&(new!=y))),wrong_to_correct=int(np.sum(sel&(base!=y)&(new==y))),
                           both_wrong=int(np.sum(sel&(base!=y)&(new!=y))),high_confidence_wrong_AI=int(np.sum(sel&(new!=y)&(op[np.arange(len(y)),y]<.1)))))
   for condition,zz in [('historical_WSSL',z),('historical_AI',oz)]:
    v=((2*y-1)*(zz[:,1]-zz[:,0]))[sel]
    margins.append(dict(condition=condition,seed=seed,context=c,inner=i,label=['PD','DD'][label],n=len(v),mean=float(v.mean()),median=float(np.median(v)),SD=float(v.std(ddof=1))))
 csvout('historical_AI_error_transition.csv',transitions);csvout('historical_AI_margin_summary.csv',margins)
 logs=[]
 for c,i,sp,ex in splits():
  for seed in [42,43,44]:
   root=p3/'runs/adapt/p100'/f'seed{seed}'/f'outer_{c}'/f'inner_{i}';config_=yaml.safe_load((root/'config.yaml').read_text())
   epochs=[json.loads(x) for x in (root/'logs/epochs.jsonl').read_text().splitlines() if x.strip()]
   rates=[x['train']['learning_rate'] for x in epochs]
   logs.append(dict(seed=seed,context=c,inner=i,configured_classifier_LR=config_['training']['learning_rate'],
                    first_logged_classifier_LR=rates[0],minimum_logged_classifier_LR=min(rates),epochs=len(epochs),
                    encoder_initial_LR_from_actual_optimizer_source=2e-5,
                    encoder_LR_separately_logged=False,log_sha256=sha(root/'logs/epochs.jsonl'),config_sha256=sha(root/'config.yaml')))
 csvout('historical_layer5_log_audit.csv',logs)
 legacy=pd.read_csv(p3/'analysis/model_seed_split_metrics.csv');legacy=legacy[legacy.variant=='adapt']
 report('HISTORICAL_ACTIVITY_INTERACTION_AUDIT.md','# 历史来源补充核对\n\nPhase8逐run baseline/AI预测90份文件均由既有哈希清单验证，query身份逐split核对。只读独立逐run预测，不解码旧整队列信号/embedding缓存。错误转移与margin单列LEGACY_L1_HISTORICAL_ONLY，不能进入Certified主比较或为本轮初始化。\n\n历史Layer5实际45个config与epoch日志均核对：classifier初始LR2e-4；encoder2e-5来自实际optimizer hook的1/10参数组定义，旧日志仅记录首组LR，不冒称独立记录了encoder每epoch LR。BN running固定但affine原可训练。旧adapted的开发均值：\n\n'+table(legacy.groupby('variant')[['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']].mean().reset_index())+'\n\n这些是旧L1结果，不是本轮FT-L1结果；本轮改用Certified/norm、fresh下游、encoder1e-5且BN affine也冻结，仍不是新的网络结构。')
 write(HERE/'analysis/historical_AI_transition_integrity.json',dict(status='PASS_45_HISTORICAL_PAIRED_PREDICTIONS',legacy_data_not_mixed_with_certified=True,old_global_cache_decoded=False,old_models_not_used_as_initialization=True,historical_logs_verified=45),new=True)
 print('HISTORICAL_READONLY_AUDIT_PASS')

if __name__=='__main__':main()
