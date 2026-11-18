"""Server-only final aggregation of completed fits; no further optimization."""
from common10 import *
from weighted_l2 import weighted
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_recall_curve
import argparse,html

KINDS=['FT-L1','DEMO','DD']
def frame(kind):
 scope='full' if (HERE/'analysis'/f'{kind}_full_metrics.csv').exists() else 'pilot'
 return pd.read_csv(HERE/'analysis'/f'{kind}_{scope}_metrics.csv'),scope

def prediction(kind,seed,c,i):
 if kind=='WSSL-CERTIFIED-L1':return certified('wssl',seed,c,i)
 with np.load(PRIVATE/'runs'/kind/f'seed{seed}_context{c}_inner{i}'/'predictions.npz',allow_pickle=False) as z:
  return z['subject_ids'].astype(str),z['target'],z['probabilities'],z['logits']

def cluster(kind,f):
 packages=[];labels={}
 for r in f.itertuples():
  ids,y,p,z=prediction(kind,int(r.seed),int(r.context),int(r.inner))
  oi,oy,op,oz=certified('wssl',int(r.seed),int(r.context),int(r.inner));idx={s:j for j,s in enumerate(oi)};order=[idx[s] for s in ids]
  assert np.array_equal(y,oy[order])
  for sid,label in zip(ids,y):
   if sid in labels:assert labels[sid]==label
   labels[sid]=label
  packages.append((ids,y,p,z,op[order],oz[order]))
 subjects=sorted(labels);loc={s:j for j,s in enumerate(subjects)};lab=np.array([labels[s] for s in subjects]);rng=np.random.default_rng(20261118)
 weights=np.zeros((10000,len(subjects)),dtype=np.int16)
 for v in [0,1]:
  sel=np.flatnonzero(lab==v);weights[:,sel]=rng.multinomial(len(sel),np.full(len(sel),1/len(sel)),10000)
 delta=np.zeros((10000,len(METRICS)));point=np.zeros(len(METRICS));unitchecks=[]
 for ids,y,p,z,op,oz in packages:
  # One subject's single weight is shared across every seed and split appearance.
  w=weights[:,[loc[s] for s in ids]]
  result=weighted(y,p,z,w)-weighted(y,op,oz,w)
  delta+=result/len(packages)
  point+=np.array([pred_metrics(y,p,z)[m]-pred_metrics(y,op,oz)[m] for m in METRICS])/len(packages)
  replay=weighted(y,p,z,np.ones((1,len(y))))[0]
  assert np.allclose(replay,[pred_metrics(y,p,z)[m] for m in METRICS],atol=1e-12,rtol=1e-12)
 valid=np.isfinite(delta).all(1);assert valid.sum()>=9500
 rows=[]
 for j,m in enumerate(METRICS):
  lo,hi=np.quantile(delta[valid,j],[.025,.975])
  rows.append(dict(condition=kind,metric=m,mean_delta=point[j],subject_cluster_ci_low=lo,subject_cluster_ci_high=hi,
   bootstrap_draws=10000,finite_draws=int(valid.sum()),unique_validation_subjects=len(subjects),stratified_PD_DD=True,
   repeated_subjects_share_weight=True,training_refit=False,inference_scope='CONDITIONAL_FIXED_FITS_DESCRIPTIVE_SENSITIVITY',p_value=np.nan,BH_q=np.nan))
 return rows

def plots():
 t=pd.read_csv(HERE/'analysis/threshold_curves.csv');g=t.groupby('threshold').mean(numeric_only=True).reset_index();parts=[]
 def panel(x,ysets,left,top,w,h,title,xlabel,ylabel,xlimits=(0,1),ylimits=(0,1)):
  esc=html.escape;parts.append(f'<text x="{left}" y="{top-20}" font-size="19">{esc(title)}</text>')
  def xx(v):return left+(v-xlimits[0])/(xlimits[1]-xlimits[0])*w
  def yy(v):return top+h-(v-ylimits[0])/(ylimits[1]-ylimits[0])*h
  for v in np.linspace(0,1,6):
   parts.append(f'<line x1="{left}" y1="{yy(v)}" x2="{left+w}" y2="{yy(v)}" stroke="#ddd"/><text x="{left-30}" y="{yy(v)+4}" font-size="12">{v:.1f}</text>')
   parts.append(f'<text x="{xx(v)-8}" y="{top+h+18}" font-size="12">{v:.1f}</text>')
  for label,ys,color in ysets:
   coords=' '.join(f'{xx(float(a)):.3f},{yy(float(b)):.3f}' for a,b in zip(x,ys) if math.isfinite(float(b)))
   parts.append(f'<polyline points="{coords}" stroke="{color}" stroke-width="2.2" fill="none"/>')
   index=[a[0] for a in ysets].index(label)
   parts.append(f'<text x="{left+index*150}" y="{top+h+60}" fill="{color}" font-size="14">{esc(label)}</text>')
  parts.append(f'<text x="{left+w/2-70}" y="{top+h+40}" font-size="14">{esc(xlabel)}</text>')
  parts.append(f'<text x="{left}" y="{top-3}" font-size="12">{esc(ylabel)}</text>')
 panel(g.threshold,[('PD Recall',g.pd_recall,'#185a9d'),('DD Recall',g.dd_recall,'#b33333')],65,65,460,290,'Fixed validation operating points','DD probability threshold','Recall')
 panel(g.threshold,[('BA',g.ba,'#16734a'),('Macro F1',g.macro_f1,'#8a3c9b')],645,65,460,290,'Decision metrics vs threshold','DD probability threshold','Score')
 parts.append('<text x="65" y="455" font-size="16">45 existing fits; means over development appearances. No threshold selected.</text>')
 parts.append('<text x="65" y="484" font-size="15">AUROC, NLL, Brier and top-confidence ECE are invariant to decision threshold.</text>')
 parts.append('<text x="65" y="513" font-size="15">Official rule remains pDD &gt; 0.5; deployable threshold blocked without legal train-only OOF.</text>')
 svg='<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="550" viewBox="0 0 1160 550"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif">'+''.join(parts)+'</g></svg>'
 with (HERE/'analysis/threshold_operating_curves.svg').open('x') as out:out.write(svg)
 grid=np.linspace(0,1,101);curves=[]
 for c,i,sp,ex in splits():
  for seed in [42,43,44]:
   ids,y,p,z=certified('wssl',seed,c,i);prec,rec,th=precision_recall_curve(y,p[:,1]);curves.append(np.interp(grid,rec[::-1],prec[::-1]))
 curves=np.asarray(curves);csvout('precision_recall_curve_aggregate.csv',[dict(recall=r,precision_mean=curves[:,j].mean(),precision_run_SD=curves[:,j].std(ddof=1),fits=45,
  interpretation='INTERPOLATED_PER_FIT_PR_DESCRIPTIVE_REPEATED_SUBJECTS') for j,r in enumerate(grid)])
 parts.clear();panel(grid,[('DD precision',curves.mean(0),'#185a9d')],65,65,500,320,'Certified WSSL: precision-recall curve','DD Recall','DD Precision')
 parts.append('<text x="65" y="490" font-size="14">Mean of 45 per-fit interpolated empirical PR curves; no pooled-patient claim.</text>')
 with (HERE/'analysis/precision_recall_curve.svg').open('x') as out:out.write('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="530" viewBox="0 0 760 530"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif">'+''.join(parts)+'</g></svg>')

def metadata_device():
 meta=metadata();devices=load(PRIVATE/'demographic_whitelist_and_device.json')['device'];rows=[]
 for device in sorted(set(devices.values())):
  ids=[s for s in meta if devices[s]==device]
  for cls in [0,1]:
   chosen=[s for s in ids if meta[s]['label']==cls]
   for field in ['age','height','weight']:
    vals=np.array([meta[s][field] for s in chosen],float)
    rows.append(dict(device=device,label=['PD','DD'][cls],field=field,n=len(vals),mean=vals.mean(),SD=vals.std(ddof=1),median=np.median(vals)))
   for sex in ['female','male']:rows.append(dict(device=device,label=['PD','DD'][cls],field='gender',category=sex,n=sum(meta[s]['gender']==sex for s in chosen)))
 csvout('demographics_by_device.csv',rows)
 # Device vs age-band/gender associations are cohort diagnostics, never inputs.
 assoc=[]
 for field,categories in [('gender',['female','male']),('age_band',['<60','60-69','>=70'])]:
  values={s:meta[s]['gender'] if field=='gender' else '<60' if meta[s]['age']<60 else '60-69' if meta[s]['age']<70 else '>=70' for s in meta}
  dev=sorted(set(devices.values()));counts=np.array([[sum(devices[s]==d and values[s]==v for s in meta) for v in categories] for d in dev])
  stat,p,df,expected=chi2_contingency(counts,correction=False)
  assoc.append(dict(association='device_vs_'+field,chi2=stat,p=p,df=df,Cramers_V=np.sqrt(stat/(counts.sum()*min(counts.shape[0]-1,counts.shape[1]-1))),
   min_expected_cell=float(expected.min()),interpretation='DESCRIPTIVE_COHORT_ASSOCIATION_NO_CAUSAL_OR_INDEPENDENT_CONFIRMATION'))
 csvout('demographic_device_associations.csv',assoc)

def analyze():
 allframes=[baseline_frame(),baseline_frame('str'),pd.read_csv(HERE/'analysis/demographic_only_metrics.csv')];stats=[];clusters=[];states=[];margins=[]
 for kind in KINDS:
  f,scope=frame(kind);assert scope=='pilot' or len(f)==45
  gate=load(HERE/'analysis'/f'{kind}_{scope}_gate.json');allframes.append(f)
  stat=pd.read_csv(HERE/'analysis'/f'{kind}_{scope}_paired_statistics.csv');stat['candidate']=kind
  stat['split_wins']=np.nan if scope=='pilot' else stat.get('wins',np.nan);stat['split_ties']=np.nan if scope=='pilot' else stat.get('ties',np.nan);stat['split_losses']=np.nan if scope=='pilot' else stat.get('losses',np.nan)
  # Seed dz is descriptive and never an independent-patient effect estimate.
  if scope=='pilot':stat['descriptive_seed_dz']=np.divide(stat.mean_delta,stat.seed_delta_SD,out=np.full(len(stat),np.nan),where=stat.seed_delta_SD>0)
  stats.extend(stat.to_dict('records'));clusters.extend(cluster(kind,f))
  states.append(dict(candidate=kind,status=gate['status'],formal_runs=len(f),splits=f[['context','inner']].drop_duplicates().shape[0],seeds=3,
   complete45=len(f)==45,retained=gate['status']=='RETAIN_DEVELOPMENT_CANDIDATE',reason='; '.join(k for k,v in gate['checks'].items() if not v)))
  for r in f.itertuples():
   ids,y,p,z=prediction(kind,int(r.seed),int(r.context),int(r.inner));oi,oy,op,oz=certified('wssl',int(r.seed),int(r.context),int(r.inner));ix={s:j for j,s in enumerate(oi)};oz=oz[[ix[s] for s in ids]]
   for cls in [0,1]:
    a=(2*y-1)*(z[:,1]-z[:,0]);b=(2*y-1)*(oz[:,1]-oz[:,0]);sel=y==cls
    margins.append(dict(candidate=kind,seed=int(r.seed),context=int(r.context),inner=int(r.inner),label=['PD','DD'][cls],n=int(sel.sum()),
     mean_signed_margin=a[sel].mean(),median_signed_margin=np.median(a[sel]),SD_signed_margin=a[sel].std(ddof=1),IQR_signed_margin=np.subtract(*np.percentile(a[sel],[75,25])),
     mean_delta_signed_margin=(a[sel]-b[sel]).mean(),high_confidence_error_n=int(np.sum(sel&(p[np.arange(len(y)),y]<.1)))))
 states.extend([dict(candidate='FT-L2',status='NOT_RUN_NO_INDEPENDENT_DEPTH_EVIDENCE',formal_runs=0,splits=0,seeds=0,complete45=False,retained=False,reason='FT-L1 rejected; no deeper-residual-stage bottleneck evidence'),
  dict(candidate='NEW_DEPLOYABLE_THRESHOLD',status='BLOCKED_NO_LEGAL_CERTIFIED_TRAIN_ONLY_OOF',formal_runs=0,splits=0,seeds=0,complete45=False,retained=False,reason='Validation curves descriptive only')])
 ref=baseline_frame();ref=ref[(ref.context==0)&(ref.inner==0)].copy();ref['condition']='WSSL-CERTIFIED-L1-MATCHED-PILOT';allframes.append(ref)
 fsum=summary(pd.concat(allframes,ignore_index=True));fsum['SD_unit']=np.where(fsum.splits>1,'sample_SD_of_split_seed_means','sample_SD_of_three_seed_metrics')
 csvout('final_metrics.csv',fsum.to_dict('records'));csvout('paired_statistics.csv',stats);csvout('subject_cluster_bootstrap.csv',clusters);csvout('experiment_state.csv',states);csvout('candidate_margin_comparison.csv',margins)
 families=[]
 for kind in ['FT-L1','FT-L2','DEMO','DD']:
  f,scope=frame(kind) if kind!='FT-L2' else (None,'not_run')
  for metric in METRICS:families.append(dict(candidate=kind,metric=metric,family='primary_8' if metric in ['ba','auroc'] else 'secondary_28',
   status='PILOT_NO_INFERENTIAL_TEST' if scope=='pilot' else 'NOT_RUN' if f is None else 'FULL',internal_placeholder_p=1.,reported_p=np.nan,BH_q=np.nan))
 assert all(x['status']!='FULL' for x in families),'Full inference requires planned BH application before reporting'
 csvout('planned_multiplicity_families.csv',families)
 cost=pd.concat([pd.read_csv(HERE/'analysis'/f'{x}_costs.csv') for x in ['b','c','d']],ignore_index=True)
 csvout('training_costs.csv',cost.to_dict('records'))
 counts=cost[['condition','total_parameters','trainable_parameters','frozen_parameters']].drop_duplicates()
 counts=pd.concat([pd.DataFrame([dict(condition='WSSL-CERTIFIED-L1',total_parameters=10600580,trainable_parameters=143172,frozen_parameters=10457408)]),counts],ignore_index=True)
 csvout('parameter_counts.csv',counts.to_dict('records'))
 plots();metadata_device()
 write(HERE/'audit/final_aggregation_receipt.json',dict(status='PASS',new_deep_fits=0,new_optimizer_updates=0,source_sha256=sha(Path(__file__)),
  clinical_runs=int(cost.shape[0]),subject_cluster_draws=10000,reported_patient_ids=False,protected_files_verified=guard()),new=True)
 print('PASS final aggregate analysis; no additional training',flush=True)

def textmetrics(f):
 rows=[]
 for r in f.to_dict('records'):
  item={k:r[k] for k in ['condition','runs','splits','seeds']}
  for m in METRICS:item[m]=f'{r[m]:.6f} ± {r[m+"_SD"]:.6f}' if pd.notna(r[m+'_SD']) else f'{r[m]:.6f} ± NA'
  rows.append(item)
 return pd.DataFrame(rows)

def reports():
 f=pd.read_csv(HERE/'analysis/final_metrics.csv');s=pd.read_csv(HERE/'analysis/experiment_state.csv');delta=pd.read_csv(HERE/'analysis/paired_statistics.csv')
 costs=pd.read_csv(HERE/'analysis/training_costs.csv');bench=pd.read_csv(HERE/'analysis/inference_benchmark.csv');counts=pd.read_csv(HERE/'analysis/parameter_counts.csv')
 allnegative=not s.retained.any();assert allnegative,'Review final model conclusion if any full candidate retained'
 common='15个split有受试者重叠，三个seed也不是三个患者队列。新候选仅1 split×3 seed，全部未通过资源闸门，不能据此证明泛化劣于基线或发布15split显著性结论。历史数据与Certified数据分开，当前正式模型指针未变。'
 stattext='''# Phase10 统计附录

主统计单位：完整矩阵先对同split三个seed等权平均，再对15split等权。final_metrics中的SD是15个split均值的sample SD(ddof=1)，另有三个seed跨split均值的sample SD。pilot的SD是同一split三个seed指标的sample SD，不能与跨split SD等同。D0只有15个确定性LR拟合，没有45个独立拟合；seed SD为NA。Recall分别给出PD与DD，主规则严格pDD>0.5、tie PD。

本轮全部新候选止于3seed pilot。mean/median delta与seed正负计数、描述seed dz保留；15split wins/ties/losses、paired split CI、Wilcoxon、rank-biserial与BH q均NA。未把seed当作患者队列作推断。预登记主族4候选×2指标=8，次族4×7=28；未运行和pilot推断项内部p=1占位，其自身公开p/q为NA，不缩减比较族或制造显著性。

subject_cluster_bootstrap.csv为10,000次PD/DD分层受试者重采样。相同subject在所有seed/appearance共用一次抽样权重，每次先计算匹配模型差，再等权平均run（完整等价于split内三seed平均后split平均）。本轮每候选仅105名validation受试者，固定既有拟合，不重新训练。CI仅为条件性敏感度，不是独立确认，不包括模型选择、训练重拟合、中心或设备迁移的不确定性。未以该CI绕过pilot闸门。weighted kernel用全1权重与正式九指标逐项核对到1e-12，包含项目两概率列macro AUROC、logsumexp NLL、two-column Brier和10等宽top-confidence ECE。

阈值仅改变硬决策及混淆矩阵，不改变固定概率下的AUROC/NLL/Brier/ECE。所有validation操作曲线是描述性；合法Certified train-only OOF缺失，部署阈值BLOCKED。PR图先逐fit计算经验曲线，再在固定recall grid插值平均，不能当成390个独立人一次预测的PR。
'''
 report('STATISTICAL_APPENDIX.md',stattext+'\n'+table(pd.read_csv(HERE/'analysis/subject_cluster_bootstrap.csv'))+'\n\n'+common)
 gpu=__import__('subprocess').check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
 timing=bench.groupby('condition').agg(subjects=('subjects','first'),repetitions=('repeat','count'),online_seconds_mean=('seconds','mean'),online_seconds_SD=('seconds','std'),seconds_per_subject=('seconds_per_subject','mean'),peak_allocated_bytes=('peak_GPU_allocated_bytes','max'),max_probability_drift=('canonical_cached_probability_max_abs_error','max')).reset_index()
 report('COST_AND_DEPLOYMENT_REPORT.md','# Phase10 参数、成本与部署\n\nGPU：'+gpu+'。既有Python/Torch环境，无软件安装或权重下载。\n\n'+table(counts)+'\n\n正式训练实测（每run为50上限、patience12、best BA；没有用推算45run替代实测）：\n\n'+table(costs)+'\n\n同一固定8人、1warmup+3次在线推理实测：\n\n'+table(timing)+'''

训练wall包含各run初始化/加载和正式循环；DEMO还包含新导出Certified frozen train/validation logits的成本。parent_CPU_seconds仅主进程CPU，不含DataLoader worker CPU总量。cached_path_validation_inference_seconds只计缓存后的validation路径，不能代表完整部署延迟；冻结HarNet训练不在GPU计算完整encoder，峰值也不代表在线总资源。

在线benchmark包含固定8人、11活动、双腕的原Raw Acc文件读取、trim48/clip±3g/first-last1000/3:10 resampling、已有Certified STR文件读取/原norm/transform、CPU→GPU、HarNet及classifier前向、概率回CPU。模型和官方权重预加载不计，原Certified detrending生产阶段不计；磁盘受系统缓存影响，不是穿戴设备实时功耗或单人p95。canonical缓存的全局batch布局与本次局部8人重算HarNet batch不同，GPU算术可有微小漂移，标量最大差公开；本次只计时，不替换正式预测、不重新择epoch、不用漂移结果改善指标。

FT-L1总参数包含完整官方HarNet，即使训练只缓存冻结prefix；DEMO/DD加一次10,457,408参数encoder，不能重复计入classifier。DEMO只有216新增可训练参数，部署仍需原双腕IMU和四字段；所有候选均拒绝扩展，无新模型替换当前指针。推理脚本与checkpoint/optimizer仅用于重现研究，不把私有权重上传GitHub。
''')
 report('CORRECTNESS_AND_PROVENANCE.md',f'''# Phase10 正确性与来源

服务器独立目录执行，起点{BASE}。实际检查源码/配置/日志及服务器和GitHub差异，未覆盖mainline或既有实验。A、B、C、D freeze记录执行前代码和协议SHA256，每run config/checkpoint/数据manifest/norm SHA、fit/query/excluded provenance、best epoch、CPU/CUDA/loader RNG及optimizer/scheduler state均在服务器restricted_manifests闭环。公开per-run配置无IDs，逐人信号/人口值/预测/权重不出服务器。完整公开快照沿用既有发布器过滤及审计，认证失败时以已授权GitHub连接器构造相同tree并回读校验，未强推。

Correctness包括：90份Certified既有预测及来源核对；8580腕记录SHA；15份norm来源；四字段train-only LR预处理独立重算；非法字段不影响输入；阈值概率指标不变；实际HarNet stage/BN affine/running state冻结；fresh官方初始化和原classifier RNG一致；新分支零初始化逐位一致；padding不进入pool；有限gradient与非零参数更新；model+optimizer roundtrip；DD实际weighted-mean CE手算及常数缩放梯度不变。test_models10最初误把manifest状态期望写成COMPLETE，实际源schema是PASS，修复后重跑；失败副本和日志保留，无临床数据或算法变化，详见CORRECTNESS_FIX_NOTE.md。不得隐瞒这次工程失败。

HarNet官方权重SHA256 c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6。Certified manifest SHA256 cdc5e26db77ccb101c7157fc1a7aacde132de9d5e6d5449203f8bf84aa393f16。CURRENT_MODEL.json仍为a30c1262c7e37793050a73bfd1c0ba6f8f68c0cc020a21cda1434a0f5f05a820。保护原文件共22,444个，包含历史公开内容与正式科学资产，不改动退役MFAM。FT使用按人官方冻结prefix cache，其source hash独立验证；它不是旧监督embedding。DEMO新导出当前Certified baseline logits，query与原预测1e-6内一致，全部私有；原WSSL始终eval且无更新。设备只分层，不是模型输入。

元数据age/gender/height/weight来自原记录与官方说明。原字段gender不可冒称已核实生物学sex，age_at_diagnosis/用药/病程/问卷等从输入排除。完整JSON解析后只保留白名单，不声称物理零读取其他键。[PhysioNet数据说明](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)与[原论文Table3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770131/)支持采集年龄及years/cm/kg单位。没有把数字证书PASS等同于临床采集质量或已证明无混杂/标签泄漏。

分析仅读取受控fit/query signal与prediction。未读取封存outer-test performance。历史Phase3C/Phase8只作明确分区的来源审计，不作为Certified性能或训练初始化；未解码旧整队列压缩cache。没有追加seed、换split、搜阈值/超参数、删样本或启动Phase11。

最终独立重算结果写入tests/final_verification.json和analysis/phase10_integrity.json；这两个文件需与最终GitHub远端回读收据一起使用，服务器最终收据在项目根之外避免递归写入已审计快照。{common}
''')
 report('NEXT_STAGE_RESEARCH_PLAN.md','''# 后续研究建议（未执行）

本轮至此停止。FT-L1、人口学残差融合与DD×1.10均未通过预登记扩展门槛；不追加seed、FT-L2、权重搜索、事后阈值或组合模型。没有新证据支持把网络加深作为当前必需修复。

下一任务可优先核查Other Movement Disorders粗来源标签的异质性、采集设备与年龄/性别构成，明确临床标签与采集质量信息的访问条件。未来任何方法应先给出独立新假设和预算，避免把重复开发暴露当作新确认。若研究校准或阈值，应先明确受试者分离的Certified train-only OOF产生方案及新增训练预算，不能用本次validation反推。DD代价pilot出现排序/硬判别提高而NLL退化，未来可提出独立概率可靠性问题，但本轮不后补校准来挽救候选。

最终需独立中心/设备/队列验证；目前所有证据均来自development。正式模型与阈值保持原样。未启动Phase11，未登记新Retained模型。
''')
 answers=[
 ('HarNet部分解冻是否优于完全冻结？','未建立。FT-L1单split三seed平均BA/AUROC均下降，未扩大；不能把小pilot概括为所有适配均无效。'),
 ('最后一与最后两block是否有差异？','NOT_TESTED。实际末stage layer5没有内部ResBlock，layer4含两个；FT-L2未运行，不能编造差异。历史Layer5位置相同，新版本Certified数据/norm、fresh下游、LR1e-5及冻结BN affine有区别，不宣称结构创新。'),
 ('部分解冻是否提高DD Recall？','本pilot均值下降0.010753，未支持。个别seed结果不能替代联合门槛。'),
 ('人口字段单独具有较强分类能力？','D0的15split BA约0.574716、AUROC约0.607880，存在有限判别信息，低于WSSL；不能称强或病理因果。'),
 ('人口融合是否提供额外增量？','未建立。DEMO pilot BA下降0.003124、AUROC下降0.000436、DD Recall下降0.010753，未扩大。'),
 ('人口增量是否受年龄/性别/设备混杂？','原队列存在分布和gender/device类关联，缺失率0；分层结果及设备×人口分布已保存。由于未出现稳定增量且小组样本有限，无法声称控制了混杂后的独立增量或因果。'),
 ('DD错误主要来自某来源？','unique-subject consensus中Other Movement Disorders的recall约0.4833（60人），错误负担最大；ET28、Atypical15、MS11也有错误，不能仅归因单一极小来源。它们是粗来源类，consensus不是正式单模型成绩。'),
 ('DD-aware loss是否改善平衡？','pilot BA/AUROC及两类Recall点估计上升，但NLL增加0.125395超过冻结0.05门槛，Brier/ECE也恶化；联合资源闸门REJECT，不搜索权重。'),
 ('阈值能否改善操作点？','描述曲线显示两类recall的操作点可互换，但不能据validation择优宣布收益。无合法Certified train-only OOF，部署阈值BLOCKED，保持0.5。阈值不改变AUROC或proper概率指标。'),
 ('哪个候选通过联合保留？','无。三个候选仅完成pilot且均拒绝扩展，没有45run保留判定。FT-L2未运行。'),
 ('哪些方向停止？','停止本轮FT-L1、当前人口融合及DD×1.10的扩展，不自动启动FT-L2或组合模型。历史Activity Interaction不扩大。'),
 ('当前最佳模型需要更新？','不更新。Certified WSSL仍为本轮研究基线，Frozen WSSL-STR正式Retained身份与CURRENT_MODEL指针原样保留。不存在本轮性能晋升。')]
 report('FINAL_REPORT.md','# Phase10 最终报告\n\n三个独立候选均未通过预登记pilot资源闸门。实际新增深度训练9 run（每方向context0/inner0×seeds42/43/44），人口学D0为15次train-only Logistic Regression拟合。没有新增42run矩阵、FT-L2或OOF训练。所有训练及统计在服务器既有环境完成。\n\n'+table(s)+'\n\n九指标mean ± sample SD：完整基线为15split SD，候选和匹配pilot为3seed SD，两者不能直接当同一不确定性比较。\n\n'+table(textmetrics(f))+'\n\n配对点估计：\n\n'+table(delta[['candidate','metric','mean_delta','median_delta']])+'\n\n'+ '\n\n'.join(f'## {j}. {q}\n\n{a}' for j,(q,a) in enumerate(answers,1))+'\n\n## 统计、来源与交付边界\n\n'+common+'详见统计附录、正确性与来源、成本报告及每阶段报告。受试者cluster CI是固定拟合条件性诊断，不用于绕过停止门槛。训练/推理时间和参数为实测，缓存训练与在线推理分别说明。原257行历史CSV逐值保留，本轮A/B/C/D追加9行，累计266行，新增NLL/Brier/ECE及参数/耗时字段，未知SD为NA，不伪造。\n\n最终发布必须同时核对远端commit/tree、服务器公开source、发布镜像和PUBLICATION_MANIFEST。阶段A/PREB/B/C/D各自已有回读收据；最终收据保存在服务器项目之外，并在交付回复中给出实际提交与tree。私有数据、信号、权重、IDs和逐人预测不发布。完成回读后停止，不开展Phase11。')
 print('PASS all final reports written',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['analyze','reports']);a=ap.parse_args()
 analyze() if a.action=='analyze' else reports()
