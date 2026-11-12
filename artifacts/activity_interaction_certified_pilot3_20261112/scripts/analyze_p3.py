"""One-split descriptive summaries; no seed significance tests or retention claim."""
from common_p3 import *
from weighted_l2 import weighted
from threadpoolctl import threadpool_limits
import gc

def timing():
 rows=[]
 for seed in SEEDS:
  seed_everything(seed,True);d=token(seed,'train');model=make_model(seed).eval();path=PRIVATE/'checkpoints'/f'seed{seed}/best.pt';model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model_state']);h=torch.from_numpy(d['bilateral'][:8]).cuda();mask=torch.from_numpy(d['mask'][:8]).cuda();prior=fingerprint(model)
  for condition in ['baseline_readout','baseline_plus_interaction']:
   fn=(lambda:head(model,h,mask)) if condition=='baseline_readout' else (lambda:model(h,mask)['logits']);torch.cuda.reset_peak_memory_stats()
   with torch.inference_mode():
    for _ in range(3):fn()
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(10):out=fn()
    torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/10
   assert torch.isfinite(out).all();rows.append(dict(seed=seed,condition=condition,batch_size=8,warmup=3,repeats=10,batch_ms=ms,amortized_ms_per_subject=ms/8,subjects_per_second=8000/ms,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),train_only=True,resident_tokens_float32=True,not_end_to_end=True,not_standalone_patient_latency=True))
  assert fingerprint(model)==prior;del model;gc.collect();torch.cuda.empty_cache()
 f=pd.DataFrame(rows);f.to_csv(HERE/'analysis/token_latency_by_seed.csv',index=False);return f

def bootstrap():
 samples=[];ids0=None;label=None
 for seed in SEEDS:
  path=PRIVATE/'checkpoints'/f'seed{seed}/predictions.npz'
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  ids=d['subject_ids'].astype(str)
  if ids0 is None:ids0=ids;label=d['target']
  assert np.array_equal(ids0,ids) and np.array_equal(label,d['target']);samples.append(d)
 rng=np.random.default_rng(20261112);weights=np.zeros((10000,len(label)),np.int16)
 for k in [0,1]:
  ix=np.flatnonzero(label==k);weights[:,ix]=rng.multinomial(len(ix),np.repeat(1/len(ix),len(ix)),10000)
 draw=[];point=[]
 with threadpool_limits(limits=4):
  for d in samples:
   new=weighted(label,d['probabilities'],d['logits'],weights);old=weighted(label,d['baseline_probabilities'],d['baseline_logits'],weights);draw.append(new-old);point.append(np.array(list(pred_metrics(label,d['probabilities'],d['logits']).values()))-np.array(list(pred_metrics(label,d['baseline_probabilities'],d['baseline_logits']).values())))
 v=np.mean(draw,axis=0);estimate=np.mean(point,axis=0);rows=[]
 for j,m in enumerate(METRICS):
  valid=np.isfinite(v[:,j]);ci=np.quantile(v[valid,j],[.025,.975]) if valid.sum()>=9500 else [np.nan,np.nan];rows.append(dict(metric=m,paired_seed_mean_delta=estimate[j],ci_low=ci[0],ci_high=ci[1],draws=10000,finite_draws=int(valid.sum()),unique_validation_subjects=len(label),PD_unique=int(sum(label==0)),DD_unique=int(sum(label==1)),scope='fixed trained models and this selected development split; descriptive sensitivity only',shared_weights_all_seeds_models=True))
 pd.DataFrame(rows).to_csv(HERE/'analysis/subject_cluster_sensitivity.csv',index=False);return pd.DataFrame(rows)

def main():
 assert load(HERE/'analysis/training_integrity.json')['runs']==3;guard();torch.set_num_threads(4);seed_everything(42,True);freeze('ANALYSIS',[Path(__file__),HERE/'scripts/common_p3.py'],new_fits=0,subject_draws=10000,no_seed_significance_tests=True,no_ensemble=True)
 frame=pd.read_csv(HERE/'analysis/metrics_by_seed.csv');delta=pd.read_csv(HERE/'analysis/paired_deltas_by_seed.csv');params=pd.read_csv(HERE/'analysis/parameter_counts.csv');summary=[];comparison_precision=[]
 for seed in SEEDS:
  mask=(frame.seed==seed)&(frame.condition=='WSSL-CERTIFIED-L1');receipt=load(Path(binding()['seeds'][str(seed)]['checkpoint']).parent/'complete.json');frame.loc[mask,'training_epochs']=receipt['cost']['epochs']
  baseline=frame[mask].iloc[0]
  for m in METRICS:comparison_precision.append(dict(seed=seed,metric=m,recomputed_baseline=float(baseline[m]),archived_baseline=receipt['metric'][m],difference=float(baseline[m])-receipt['metric'][m],reason='same raw logits; original archive torchfloat32 softmax vs uniform Phase8B float64 softmax in this paired comparison'))
 frame['training_epoch_scope']=np.where(frame.condition=='WSSL-CERTIFIED-L1','historical base, no new training','new module training only');frame.to_csv(HERE/'analysis/metrics_by_seed.csv',index=False);pd.DataFrame(comparison_precision).to_csv(HERE/'analysis/baseline_metric_precision_comparison.csv',index=False)
 for condition,g in list(frame.groupby('condition'))+[('AI-minus-baseline',delta)]:
  for m in METRICS:summary.append(dict(condition=condition,metric=m,mean=g[m].mean(),sample_SD=g[m].std(ddof=1),seeds=3,independent_patient_cohorts=1,better_direction='lower' if m in ['nll','brier','ece'] else 'higher'))
 summary=pd.DataFrame(summary);summary.to_csv(HERE/'analysis/mean_std_summary.csv',index=False);mean=delta[METRICS].mean();trade=mean.pd_recall>0 and mean.dd_recall<-.01;multi=all(mean[m]>0 for m in ['ba','auroc','macro_f1']) and mean.pd_recall>=-.01 and mean.dd_recall>=-.01;none=all(mean[m]<=0 for m in ['ba','auroc','macro_f1']);pattern='PD/DD权衡仍存在' if trade else '多项指标同向改善' if multi else '未见改善' if none else '结果混合';state=[]
 for seed in SEEDS:
  p=params[params.seed==seed].iloc[0];d=delta[delta.seed==seed].iloc[0];state.append(dict(seed=seed,context=0,inner=0,status='COMPLETED',correctness='PASS',formal_new_runs=1,epochs=int(p.training_epochs),best_epoch=int(p.best_epoch),BA_delta=d.ba,PD_recall_delta=d.pd_recall,DD_recall_delta=d.dd_recall,DD_drop_exceeds01=bool(d.DD_recall_drop_gt01),base_retrained=False,ensemble=False))
 pd.DataFrame(state).to_csv(HERE/'analysis/experiment_state.csv',index=False);latency=timing();cluster=bootstrap()
 # An application recommendation is a research judgment, not a retain/promotion gate.
 worthy=bool(multi and not delta.DD_recall_drop_gt01.any());recommendation='可考虑另行申请原15×3固定方案，当前3run只有一个split，不能自动保留；代码/来源一致时这3run可复用。' if worthy else '当前结果不足以支持直接申请相同固定方案的完整15×3；应先解释混合方向或DD代价。若后续另获批准，使用原15split且本3run在来源完全一致时可复用。'
 write(HERE/'analysis/pilot_judgment.json',dict(CORRECTNESS='PASS',PILOT_STATUS='COMPLETED',PERFORMANCE_PATTERN=pattern,mean_deltas=mean.to_dict(),DD_recall_drop_gt01_seeds=delta.loc[delta.DD_recall_drop_gt01,'seed'].tolist(),improving_seed_counts={m:int(sum(delta[m]>0)) for m in METRICS[:6]},BA_PD_component_mean=float(.5*mean.pd_recall),BA_DD_component_mean=float(.5*mean.dd_recall),request_full15x3_recommendation_is_research_judgment=True,worth_requesting_same_full_plan_now=worthy,recommendation=recommendation,automatic_RETAIN=False,remaining42_started=False))
 limitation='固定context0/inner0，205名train、105名validation；3seed不是3个独立患者队列。基础WSSL及新增模块都在重复development框架用validation选epoch；train监督token为in-sample，不是无监督或end-to-end OOF。患者bootstrap条件于固定模型与本split，不能消除选checkpoint和多轮开发偏差。每seed配对的是同seed基础模型，三个实例同时含base和module seed变化。无seed显著性检验、无三模型ensemble、无10/15正式保留门槛。'
 metrics_note='DD p>.5、tie PD；AUROC为两概率列macro；NLL为unweighted raw-logit proper log loss；Brier是两列平方和；ECE为10等宽topconfidence。双方统一原Phase8B的float64 softmax；归档原概率float32差异单列baseline_metric_precision_comparison.csv，实际导出归档logits与float32概率均逐位一致。'
 interpretation=f'ΔBA均值={mean.ba:.6f}，其中PD Recall贡献=.5ΔPD={.5*mean.pd_recall:.6f}，DD贡献=.5ΔDD={.5*mean.dd_recall:.6f}。DD平均变化={mean.dd_recall:.6f}；超过.01下降的seed为{delta.loc[delta.DD_recall_drop_gt01,"seed"].tolist()}。AUROC平均变化={mean.auroc:.6f}，MacroF1={mean.macro_f1:.6f}。各seed方向见配对表，不能只挑最佳seed。'
 final='# Certified L1 Activity Token Interaction 三seed小规模复核\n\n**CORRECTNESS=PASS；PILOT_STATUS=COMPLETED；PERFORMANCE_PATTERN='+pattern+'。**\n\n实际完成context0/inner0 ×42/43/44共3个新增模块run；没有其余42次，没有WSSL/L1/norm/HarNet重训重算。\n\n'+interpretation+'\n\n'+table(frame[['seed','condition']+METRICS+['PD_as_PD','PD_as_DD','DD_as_PD','DD_as_DD','best_epoch','training_epochs']])+'\n\n'+table(delta)+'\n\n'+table(summary)+'\n\n'+metrics_note+'\n\n'+table(cluster)+'\n\n## 是否申请完整复核\n\n研究判断：'+recommendation+' 这不是正式RETAIN结论或临床泛化证据。\n\n## 来源、成本与边界\n\n直接import冻结Phase8 ActivityInteraction原类；仅67522新增参数训练，143172原分类侧参数冻结/始终eval，HarNet10457408冻结。唯一部署总参数10668102。每seed分别重新导出Certified WSSL bilateral tokens，原归档及零残差逐位复现。完整来源、测试和冻结哈希见CORRECTNESS_AND_PROVENANCE。\n\n'+table(params)+'\n\n'+table(latency)+'\n\n计时是resident token→logits，排除L1、norm、原编码、HarNet与IO；batch ms和batch/8摊销明确，不是单患者独立端到端延迟。\n\n'+limitation+'\n\n旧Phase8旧L1的15split汇总只作背景，不是本轮单split直接对照，不据本轮均值声称超越它。当前正式模型指针及旧产物保持。发布状态以外部publication receipt与远端精确tree回读为准，完成后停止。'
 (HERE/'reports/FINAL_REPORT.md').write_text(final+'\n');correct='# Correctness and provenance\n\n'+limitation+'\n\nPhase8 source4745549dabf34ccc89a740ad361a0410c6a93a4d；Certified source'+START+'。用户固定split/seed在协议冻结，只有本split205 train/105 query允许访问，excluded只metadata；未读excluded数组或outer performance。必要资产hash保护与旧L2私有hash目录链核验，不重做全部历史审计。\n\n原类直接复用，不新增独立head/bias/gate，不重复加baseline/STR。使用实际11活动mask，全部有效；人工masked NaN/Inf不影响输出，全无效明确拒绝。各seed零输出层时logits精确等于本seedbaseline，跨路径与归档精确；只有新增67522参数在优化器，base始终eval、无梯度、参数/buffer前后hash不变，2次synthetic smoke后正式重置seed。reload/repeat与baseline单次计算测试通过。\n\n'+table(pd.read_csv(HERE/'analysis/baseline_reproduction.csv'))+'\n\n固定原Phase8B token DataLoader workers0、batch8、shuffleGenerator seed、balanced CE仅train标签、AdamW2e−4/WD1e−4/betas.9,.999/cosine50min1e−6/clip5/max50pat12/strictvalBA。AMPfalse，TF32matmulFalse/cuDNNTrue、deterministicwarn_only/cuDNNdetTruebenchmarkFalse，同环境原默认。完整版本见audit/environment.json。\n\n'+metrics_note+'\n\n私有tokens/checkpoint/逐人预测/IDs与完整来源存restricted_manifests，不发布。保护只读Phase8/L2/mainline/当前模型指针。最终integrity.json和私有哈希闭合，阶段失败及源码/日志按原样保留。'
 (HERE/'reports/CORRECTNESS_AND_PROVENANCE.md').write_text(correct+'\n');(HERE/'reports/COST_AND_LATENCY.md').write_text('# Parameters, training and token-readout timing\n\n'+table(params)+'\n\n原WSSL唯一部署10600580=143172 classifier+10457408 HarNet；AI部署10668102=210694 classifier侧(67522 trainable、143172 frozen)+10457408外部frozen。不是仅报告新增参数。\n\n'+table(latency)+'\n\n同inner-train前8人、相同resident token/mask/float32与硬件，CUDA同步，warmup3/repeat10。两系统均驻留内存，原head与原head+AI以同一基础权重比较；参数及buffer指纹不变。batch/8是摊销，不是batch1独立时延。排除原编码/L1/norm/HarNet/IO；在线raw端到端未测。训练耗时含validation及保存，CPU阶段记账见stage_compute_summary.csv。token_export_summary.csv记录6包/930 full-signal subject forward，train205+validation105各三seed，共10230 activity token；每实例另做一次head重建，无深度训练。\n')
 write(HERE/'analysis/analysis_integrity.json',dict(status='PASS',formal_new_runs=3,seed_mean_not_ensemble=True,patient_bootstrap_draws=10000,new_fits=0,pattern=pattern,protected_files_unchanged=guard()));print('ANALYSIS_COMPLETE '+pattern,flush=True)
if __name__=='__main__':main()
