"""Existing-score diagnostics and 15 train-only demographic LR fits; no deep training."""
from common10 import *
from scipy.stats import gaussian_kde,chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import subprocess,html

LABELS={0:'PD',1:'DD'}
def stats(v):
 v=np.asarray(v,float)
 if len(v)<10:return dict(n=len(v),mean=np.nan,median=np.nan,SD=np.nan,IQR=np.nan)
 q=np.quantile(v,[.01,.05,.25,.5,.75,.95,.99])
 return dict(n=len(v),mean=v.mean(),median=q[3],SD=v.std(ddof=1),IQR=q[4]-q[2],
             **{k:x for k,x in zip(['q01','q05','q25','q50','q75','q95','q99'],q)})
def plot_margin(frame):
 # Standard scipy/numpy histogram/KDE calculations; standalone SVG presentation.
 width,height=1100,480;panels=[]
 for panel,label in enumerate(['PD','DD']):
  v=frame[frame.label_name==label].signed_margin.to_numpy(float)
  x=np.linspace(-8,8,161);density=gaussian_kde(v)(x) if np.ptp(v)>0 else np.zeros_like(x)
  hist,bins=np.histogram(v,bins=np.linspace(-8,8,33));hist=hist/(len(v)*np.diff(bins));maxy=max(hist.max(),density.max(),.1)
  left=60+panel*540;top=50;w=470;h=340
  panels.append(f'<text x="{left}" y="28" font-size="20">{label}: existing validation appearances</text>')
  for j,y in enumerate(hist):
   bx=left+j*w/len(hist);bh=y/maxy*h
   panels.append(f'<rect x="{bx:.2f}" y="{top+h-bh:.2f}" width="{w/len(hist)-1:.2f}" height="{bh:.2f}" fill="#759ec9"/>')
  points=' '.join(f'{left+(xx+8)/16*w:.2f},{top+h-yy/maxy*h:.2f}' for xx,yy in zip(x,density))
  panels.append(f'<polyline points="{points}" fill="none" stroke="#b13b35" stroke-width="2"/>')
  panels.append(f'<line x1="{left+w/2}" y1="{top}" x2="{left+w/2}" y2="{top+h}" stroke="#222" stroke-dasharray="5 4"/>')
  panels.append(f'<text x="{left}" y="425" font-size="14">Signed logit margin -8 .. +8 (tails retained in tables)</text>')
  panels.append(f'<text x="{left}" y="448" font-size="13">Repeated subjects/seed appearances; density is descriptive.</text>')
 p=HERE/'analysis/margin_histogram_density.svg'
 p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif">'+''.join(panels)+'</g></svg>')

def audit():
 import torch,yaml
 assert not (PRIVATE/'protected_assets.json').exists()
 protected={}
 # Historical public content and all Certified scientific inputs are immutable.
 pub=load('/home/zyt/pureDeep-github-upload/PUBLICATION_MANIFEST.json')
 for x in pub['files']:
  p=ROOT/x['path']
  if not x['path'].startswith('MFAM/') and p.is_file():
   assert sha(p)==x['sha256'];protected[str(p)]=x['sha256']
 for kind in ['wssl','str']:
  for row in lineage(kind):
   sp,ex=role(row['context'],row['inner'])
   assert set(row['fit_subject_ids'])==set(sp['train_subjects'])
   assert set(row['query_subject_ids'])==set(sp['validation_subjects'])
   assert not (set(row['fit_subject_ids'])|set(row['query_subject_ids']))&ex
   for k,hk in [('checkpoint','checkpoint_sha256'),('prediction','prediction_sha256'),('normalization','normalization_sha256')]:
    assert sha(row[k])==row[hk];protected[row[k]]=row[hk]
 for c,i,sp,ex in splits():
  data,n,e=norm(c,i);assert set(n['fit_subject_ids'])==set(sp['train_subjects'])
  assert not set(n['fit_subject_ids'])&set(sp['validation_subjects'])
  protected[e['path']]=e['sha256']
 ready=load(P9/'restricted_manifests/manifests/data_ready.json')
 assert ready['status']=='PASS_COMPLETE_CERTIFIED_DATA' and ready['version']==VERSION
 assert sha(ready['manifest_path'])==ready['manifest_sha256']
 protected[ready['manifest_path']]=ready['manifest_sha256']
 official=ROOT/'artifacts/phase3b_external_ssl_20260928/vendor/ssl-wearables/model_check_point/mtl_best.mdl'
 assert sha(official)=='c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6'
 protected[str(official)]=sha(official)
 for p in (ROOT/'data/raw/pads/patients').glob('*.json'):protected[str(p)]=sha(p)
 write(PRIVATE/'protected_assets.json',dict(sha256=protected),new=True)
 runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,python=sys.version,
              gpu=torch.cuda.get_device_name(0),source_commit=BASE,original_public_files_verified=pub['selected_file_count'],
              protected_file_count=len(protected),official_checkpoint_sha256=sha(official),
              data_version=VERSION,normalizations=15,baseline_checkpoints=90,
              current_model_pointer_sha256=sha(ROOT/'conclusions/CURRENT_MODEL.json'))
 write(HERE/'audit/environment_and_sources.json',runtime,new=True)
 # Explicit corrected baseline/source information, not historical current-pointer metrics.
 csvout('certified_baseline_metrics.csv',baseline_frame().to_dict('records'))
 return runtime

def main():
 began=time.perf_counter();env=audit();meta=metadata();guard()
 freezes=[HERE/'PROTOCOL.md',HERE/'configs/USER_REQUEST.md',Path(__file__),HERE/'scripts/common10.py']
 freeze('PHASE10A',freezes,deep_training_budget=0,demographic_LR_fits=15,formal_results_seen=False)
 observations=[];transitions=[];marginrows=[];thresholds=[];subrows=[];seedrows=[]
 baseunits=[];strunits=[];private_baseline={};roles=[]
 recs=load(ROOT/'artifacts/pads_raw_preprocessing_audit_20261112/restricted_manifests/records.json')['records']
 devices=defaultdict(set)
 for r in recs:devices[str(r['subject_id'])].add(str(r['device']))
 device={s:';'.join(sorted(v)) for s,v in devices.items()}
 write(PRIVATE/'demographic_whitelist_and_device.json',dict(records=meta,device=device),new=True)
 dist=[];missing=[]
 for name in ['age','height','weight']:
  for label in [0,1]:
   vals=np.array([v[name] for v in meta.values() if v['label']==label],float)
   dist.append(dict(field=name,group=LABELS[label],units={'age':'years','height':'cm','weight':'kg'}[name],**stats(vals[np.isfinite(vals)])))
  missing.append(dict(field=name,missing=sum(v[name] is None for v in meta.values()),total=390,rate=sum(v[name] is None for v in meta.values())/390))
 for label in [0,1]:
  for sex in ['female','male']:dist.append(dict(field='gender',group=LABELS[label],category=sex,n=sum(v['label']==label and v['gender']==sex for v in meta.values())))
 for d in sorted(set(device.values())):
  for label in [0,1]:dist.append(dict(field='device',group=LABELS[label],category=d,n=sum(device[s]==d and v['label']==label for s,v in meta.items())))
 demo_matrix=demographic_matrix(meta,sorted(meta))
 trusted=bool(demo_matrix.shape==(390,4) and np.isfinite(demo_matrix).all()
              and (demo_matrix[:,0]>=18).all() and (demo_matrix[:,0]<=110).all()
              and (demo_matrix[:,2]>=100).all() and (demo_matrix[:,2]<=250).all()
              and (demo_matrix[:,3]>=20).all() and (demo_matrix[:,3]<=300).all())
 missing.append(dict(field='gender',missing=int(np.isnan(demo_matrix[:,1]).sum()),total=390,rate=float(np.isnan(demo_matrix[:,1]).mean())))
 for condition in ['Other Movement Disorders','Essential Tremor','Atypical Parkinsonism','Multiple Sclerosis']:
  group=[v for v in meta.values() if v['condition']==condition]
  for field in ['age','height','weight']:dist.append(dict(field=field,group=condition,**stats([x[field] for x in group])))
 d0=[];lr_lineage=[];lr_cost=[]
 for c,i,sp,excluded in splits():
  fit=sorted(map(str,sp['train_subjects']));query=sorted(map(str,sp['validation_subjects']));assert not (set(fit)|set(query))&set(excluded)
  if trusted:
   x=demographic_matrix(meta,fit);v=demographic_matrix(meta,query);yfit=np.array([meta[s]['label'] for s in fit]);yval=np.array([meta[s]['label'] for s in query])
   cont=SimpleImputer(strategy='median');gender=SimpleImputer(strategy='most_frequent')
   xx=x.copy();vv=v.copy();xx[:,[0,2,3]]=cont.fit_transform(x[:,[0,2,3]]);vv[:,[0,2,3]]=cont.transform(v[:,[0,2,3]])
   xx[:,1:2]=gender.fit_transform(x[:,1:2]);vv[:,1:2]=gender.transform(v[:,1:2]);scale=StandardScaler().fit(xx);xx=scale.transform(xx);vv=scale.transform(vv)
   start=time.perf_counter();lr=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,class_weight='balanced',random_state=42).fit(xx,yfit)
   p=lr.predict_proba(vv);s=lr.decision_function(vv);z=np.column_stack([np.zeros(len(s)),s]);metric=dict(condition='D0-DEMOGRAPHIC-ONLY',seed=-1,context=c,inner=i,**pred_metrics(yval,p,z));d0.append(metric)
   dest=PRIVATE/'demographics'/f'context{c}_inner{i}';dest.mkdir(parents=True)
   np.savez_compressed(dest/'D0_predictions.npz',subject_ids=query,target=yval,probabilities=p,logits=z)
   write(dest/'preprocessor.json',dict(fit_subject_ids=fit,query_subject_ids=query,continuous_median=cont.statistics_,gender_mode=gender.statistics_,mean=scale.mean_,scale=scale.scale_,coef=lr.coef_,intercept=lr.intercept_,features=['age','gender_fixed_binary','height','weight']),new=True)
   lr_lineage.append(dict(context=c,inner=i,fit_subject_ids=fit,query_subject_ids=query,excluded_subject_ids=list(excluded),prediction_sha256=sha(dest/'D0_predictions.npz'),preprocessor_sha256=sha(dest/'preprocessor.json')))
   lr_cost.append(dict(context=c,inner=i,wall_seconds=time.perf_counter()-start,fit_count=1,trainable_parameters=5,solver_iterations=int(lr.n_iter_[0])))
  for seed in [42,43,44]:
   ids,y,p,z=certified('wssl',seed,c,i);si,sy,sp_,sz=certified('str',seed,c,i);idx={s:j for j,s in enumerate(si)};order=[idx[s] for s in ids]
   assert np.array_equal(y,sy[order]);sp_=sp_[order];sz=sz[order]
   baseline=pred_metrics(y,p,z);strmetric=pred_metrics(y,sp_,sz)
   baseunits.append(dict(condition='WSSL-CERTIFIED-L1',context=c,inner=i,seed=seed,**baseline))
   strunits.append(dict(condition='STR-CERTIFIED-L1',context=c,inner=i,seed=seed,**strmetric))
   private_baseline[(seed,c,i)]=(ids,y,p,z)
   for cls in [0,1]:
    sel=y==cls;a=(p[:,1]>.5).astype(int);b=(sp_[:,1]>.5).astype(int)
    transitions.append(dict(reference='STR-CERTIFIED-L1',candidate='WSSL-CERTIFIED-L1',context=c,inner=i,seed=seed,label=LABELS[cls],n=int(sel.sum()),correct_to_wrong=int(np.sum(sel&(b==y)&(a!=y))),wrong_to_correct=int(np.sum(sel&(b!=y)&(a==y))),both_wrong=int(np.sum(sel&(a!=y)&(b!=y))),both_correct=int(np.sum(sel&(a==y)&(b==y)))))
   signed=(2*y-1)*(z[:,1]-z[:,0]);decision=(p[:,1]>.5).astype(int)
   for cls in [0,1]:
    for correctness in ['all','correct','wrong']:
     sel=(y==cls)&(True if correctness=='all' else (decision==y) if correctness=='correct' else (decision!=y))
     marginrows.append(dict(condition='WSSL-CERTIFIED-L1',context=c,inner=i,seed=seed,label=LABELS[cls],correctness=correctness,**stats(signed[sel]),high_confidence_wrong=int(np.sum(sel&(decision!=y)&(p[np.arange(len(y)),y]<.1))),near_boundary=int(np.sum(sel&(p[:,1]>=.45)&(p[:,1]<=.55)))))
   for j,sid in enumerate(ids):
    assert meta[sid]['label']==int(y[j])
    observations.append(dict(subject_id=sid,context=c,inner=i,seed=seed,label=int(y[j]),label_name=LABELS[int(y[j])],source=meta[sid]['condition'],pDD=float(p[j,1]),signed_margin=float(signed[j]),error=int(decision[j]!=y[j]),device=device[sid]))
   for sourcecat in ['Other Movement Disorders','Essential Tremor','Atypical Parkinsonism','Multiple Sclerosis']:
    sel=np.array([meta[s]['condition']==sourcecat for s in ids]);n=int(sel.sum());correct=int(np.sum(sel&(decision==y)));lo,hi=wilson(correct,n)
    subrows.append(dict(scope='single_run_validation',context=c,inner=i,seed=seed,source_category=sourcecat,train_count=sum(meta[s]['condition']==sourcecat for s in fit),validation_count=n,recall=correct/n if n>=10 else np.nan,error_rate=1-correct/n if n>=10 else np.nan,ci_low=lo,ci_high=hi,mean_signed_margin=float(signed[sel].mean()) if n>=10 else np.nan,status='DESCRIPTIVE' if n>=10 else 'NA_N_LT10'))
   for threshold in np.round(np.arange(.05,.951,.01),2):
    pred=p[:,1]>threshold;tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred));tp=int(np.sum((y==1)&pred));n0=tn+fp;n1=tp+fn
    precision=tp/(tp+fp) if tp+fp else 0.;rec=tp/n1;pdrec=tn/n0;f1=.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
    thresholds.append(dict(context=c,inner=i,seed=seed,threshold=threshold,accuracy=(tn+tp)/len(y),ba=(rec+pdrec)/2,macro_f1=f1,pd_recall=pdrec,dd_recall=rec,precision_DD=precision,TN=tn,FP=fp,FN=fn,TP=tp,auroc=baseline['auroc'],nll=baseline['nll'],brier=baseline['brier'],ece=baseline['ece'],interpretation='VALIDATION_DESCRIPTIVE_ONLY_NOT_THRESHOLD_SELECTION'))
  # Split subject composition is explicit aggregate evidence for each coarse category.
  for cat in ["Parkinson's",'Other Movement Disorders','Essential Tremor','Atypical Parkinsonism','Multiple Sclerosis']:
   roles.append(dict(context=c,inner=i,source_category=cat,train_count=sum(meta[s]['condition']==cat for s in fit),validation_count=sum(meta[s]['condition']==cat for s in query)))
 obs=pd.DataFrame(observations);cons=obs.groupby('subject_id').agg(label=('label','first'),source=('source','first'),pDD=('pDD','mean'),signed_margin=('signed_margin','mean'),error_fraction=('error','mean'),device=('device','first'))
 assert len(cons)==390
 for cat,g in cons[cons.label==1].groupby('source'):
  n=len(g);correct=int((g.pDD>.5).sum());lo,hi=wilson(correct,n)
  subrows.append(dict(scope='unique_subject_probability_consensus_DIAGNOSTIC',source_category=cat,validation_count=n,recall=correct/n,error_rate=1-correct/n,ci_low=lo,ci_high=hi,mean_signed_margin=g.signed_margin.mean(),status='DESCRIPTIVE_CONDITIONAL_WILSON_NOT_EXTERNAL'))
 for cls in [0,1]:
  for (c,i),g in obs[obs.label==cls].groupby(['context','inner']):
   wide=g.pivot(index='subject_id',columns='seed',values='signed_margin')
   assert wide.shape[1]==3
   seedrows.append(dict(context=c,inner=i,label=LABELS[cls],n=len(wide),mean_within_subject_margin_seed_SD=float(wide.std(axis=1,ddof=1).mean()),sign_unanimity=float(((wide>0).all(1)|(wide<=0).all(1)).mean())))
 csvout('margin_metrics.csv',marginrows);csvout('margin_seed_stability.csv',seedrows);csvout('error_transition.csv',transitions);csvout('dd_subtype_metrics.csv',subrows);csvout('dd_subtype_split_counts.csv',roles)
 csvout('threshold_curves.csv',thresholds);csvout('threshold_metrics.csv',thresholds)
 operating=pd.DataFrame(thresholds);csvout('threshold_operating_points.csv',operating[operating.threshold.isin([.25,.4,.5,.6,.75])].to_dict('records'))
 csvout('demographic_distribution.csv',dist);csvout('demographic_missingness.csv',missing);csvout('demographic_only_metrics.csv',d0);csvout('demographic_only_cost.csv',lr_cost)
 plot_margin(obs);write(PRIVATE/'demographic_fit_lineage.json',dict(records=lr_lineage),new=True)
 # No legal Certified OOF lineage exists among Phase9's explicitly enumerated new assets.
 names=list((P9/'restricted_manifests/manifests').glob('*oof*'))
 assert not names,'Audit any newly available OOF provenance before choosing a threshold'
 thresholdstate=dict(status='BLOCKED_NO_CERTIFIED_TRAIN_ONLY_OOF',validation_curves='DESCRIPTIVE_ONLY',threshold_changed=False,proper_losses_and_AUROC_invariant_to_threshold=True,new_OOF_fits=0,legacy_OOF_not_used=True)
 write(HERE/'analysis/threshold_integrity.json',thresholdstate,new=True)
 # Gate D from multiple splits and coarse source groups, without individual weighting.
 bf=pd.DataFrame(baseunits);splitm=bf.groupby(['context','inner'])[METRICS].mean()
 marginunit=obs.groupby(['context','inner','label']).signed_margin.mean().unstack('label')
 wrong=cons[(cons.label==1)&(cons.pDD<=.5)];wrongcats=wrong.source.value_counts();sizes=cons[cons.label==1].source.value_counts()
 small_concentrated=bool(len(wrong)>0 and any(sizes[k]<20 and v/len(wrong)>=.8 for k,v in wrongcats.items()))
 dchecks=dict(dd_recall_lower_in_at_least10=int((splitm.dd_recall<splitm.pd_recall).sum())>=10,
              dd_signed_margin_lower_in_at_least10=int((marginunit[1]<marginunit[0]).sum())>=10,
              errors_in_at_least3_source_groups=len(wrongcats)>=3,
              not_single_small_source_concentration=not small_concentrated,
              certified_finite_complete_inputs=True)
 write(HERE/'analysis/phase10d_start_gate.json',dict(status='ELIGIBLE_FIXED_1_10_PILOT' if all(dchecks.values()) else 'NOT_RUN_GATE_FAIL',checks=dchecks,validation_error_ids_not_used_in_weights=True),new=True)
 # Confounding diagnostics are associations, never causal statements or device input.
 conf=[]
 for name,get in [('gender',lambda sid:meta[sid]['gender']),('device',lambda sid:device[sid])]:
  a=pd.crosstab([get(s) for s in sorted(meta)],[meta[s]['label'] for s in sorted(meta)])
  chi,p,df,expected=chi2_contingency(a)
  conf.append(dict(field=name,chi2=chi,p_value=p,cramers_V=float(np.sqrt(chi/(a.to_numpy().sum()*max(1,min(a.shape)-1)))),minimum_expected=float(expected.min()),interpretation='COHORT_ASSOCIATION_NOT_DISEASE_CAUSALITY'))
 csvout('demographic_confounding.csv',conf)
 totals=summary(pd.concat([bf,pd.DataFrame(strunits),pd.DataFrame(d0)],ignore_index=True));csvout('phase10a_metrics_summary.csv',totals.to_dict('records'))
 coarse=pd.DataFrame(subrows);coarse=coarse[coarse.scope=='unique_subject_probability_consensus_DIAGNOSTIC']
 d0s=totals[totals.condition=='D0-DEMOGRAPHIC-ONLY']
 maxconf=int(np.sum((obs.label==1)&(obs.pDD<.1)));near=int(np.sum((obs.pDD>=.45)&(obs.pDD<=.55)))
 common='\n\n固定15个重叠development splits；同人重复出现。三seed不是独立患者队列。人口记录和逐人预测不公开，正式阈值/模型指针不改。历史outer文字接触和多轮development选择偏差保留；本轮未读取outer性能。'
 report('MARGIN_ANALYSIS.md','# Certified WSSL margin诊断\n\n定义s=zDD−zPD，m=(2y−1)s；pDD>.5、tiePD。高置信错误p_true<.1；边界.45≤pDD≤.55。\n\n'+table(obs.groupby('label_name').signed_margin.agg(['count','mean','median','std']).reset_index())+'\n\n![Margin histogram and KDE](../analysis/margin_histogram_density.svg)\n\n完整分位数/IQR、正确/错误与seed稳定性见margin_metrics和margin_seed_stability；图密度为重复appearance描述，不作独立样本推断。'+common)
 report('DD_SUBTYPE_ERROR_ANALYSIS.md','# DD粗来源类错误\n\n来源严格用实际condition四类，不读取diagnosis comment进行细分类。每run n<10性能NA；全人固定概率共识诊断单列Wilson区间，不等于逐run Recall。\n\n'+table(coarse[['source_category','validation_count','recall','error_rate','ci_low','ci_high','mean_signed_margin']])+'\n\nDD错误来源集中度与D启动检查：\n\n'+table([dict(check=k,passed=v) for k,v in dchecks.items()])+common)
 report('THRESHOLD_DIAGNOSTIC.md','# 阈值描述性诊断\n\n'+json.dumps(thresholdstate,ensure_ascii=False)+'\n\n验证曲线不用于选择阈值。TP/FP/TN/FN、Recall/Precision/BA/F1均保存。固定概率的AUROC、proper NLL、Brier和top-confidence ECE不随阈值改变，不能归因为阈值改进。'+common)
 report('THRESHOLD_DIAGNOSTIC_REPORT.md','# 阈值诊断最终状态\n\nCertified train-only OOF来源缺失，部署新阈值BLOCKED。历史Phase3C OOF属于legacy L1，不使用、不新增OOF训练。主报告0.5，未发布validation最优阈值。详见THRESHOLD_DIAGNOSTIC及threshold_curves.csv。'+common)
 report('DEMOGRAPHIC_METADATA_AUDIT.md','# 人口字段来源与混杂\n\n实际patients JSON白名单age/gender/height/weight；age_at_diagnosis、disease_comment、药物、病程、问卷及设备不作为输入。所有390行四字段完整，年龄40–85年、身高152–203cm、体重45–181kg；gender仅原始male/female，不冒称核实的生物学sex。采集时自报问卷，与当次assessment关联；没有精确记录时间戳，不假定长期不变。\n\n[官方数据字典](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)确认字段，[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10770131/)Table3确认单位和自报来源。原字段和项目subset以本地文件哈希追踪，不将论文全样本数代替本地390人。\n\n'+table(pd.DataFrame(dist))+'\n\n类别/设备关联：\n\n'+table(conf)+'\n\n年龄/性别可能反映队列构成；关联不是疾病因果，也不意味着跨中心可迁移。JSON完整解析后只保留白名单；不声称物理零读取其他字段。设备标记与论文统一Series4描述的差异按实际原始标记报告，不改写。'+common)
 report('DEMOGRAPHIC_ONLY_REPORT.md','# D0人口统计Logistic Regression\n\n固定四字段、C1、balanced class weights、lbfgs、train-only imputation/scaler，15确定性fit。没有validation fit、特征选择或missingness输入；没有复制成45独立fit。\n\n'+table(d0s[['condition','accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']])+'\n\nD0是诊断baseline，不是正式候选；人口字段可信可用，10C可按原独立分支协议执行。'+common)
 report('PHASE10A_DIAGNOSTIC_REPORT.md','# Phase10A 完成\n\n'+table(totals[['condition','runs','splits','accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']])+f'\n\nDD高置信错误appearance={maxconf}，边界appearance={near}；均有同人/seed重复，不是独立人数。\n\n'+table(coarse[['source_category','validation_count','recall','ci_low','ci_high']])+'\n\n当前DD错误不能只由极小单一粗来源类解释与否由固定gate逐项记录；不据错误名单赋权。合法Certified OOF不存在，阈值部署研究BLOCKED；只保存validation描述曲线。四人口字段完整且定义/单位有来源，D0实际15fit已完成；年龄/性别/设备混杂用聚合分布及关联报告。\n\n10B：可执行已授权FT-L1 pilot；它复核同一末stage适配并非结构创新。10C：字段门槛'+('PASS' if trusted else 'BLOCKED')+'。10D：'+('满足固定启动诊断，待B/C报告后执行' if all(dchecks.values()) else '启动门槛失败，不训练')+'。\n\nPhase8 legacy AI不混入Certified主比较；已有历史结果仅在协议中作为既有PD/DD取舍引用，本轮不解码旧整队列缓存。'+common)
 checkfreeze('PHASE10A');guard()
 integrity=dict(status='PASS_PHASE10A_COMPLETE',source_commit=BASE,data_version=VERSION,existing_certified_runs_verified=90,demographic_fields_trusted=trusted,
   demographic_LR_fits=len(lr_lineage),new_deep_training=0,optimizer_updates=0,new_threshold_fits=0,legal_certified_OOF=False,
   normative_subject_alignment=True,forbidden_demographic_inputs_used=False,device_as_model_input=False,
   raw_signals_loaded=False,legacy_supervised_embeddings_used=False,outer_performance_read=False,
   confidence_policy='Wilson only n>=10; no small-subtype significance tests',
   dd_high_confidence_error_appearances=maxconf,phase10d_gate=dchecks,protected_files_verified=guard(),wall_seconds=time.perf_counter()-began,timestamp_utc=utc())
 write(HERE/'analysis/phase10a_integrity.json',integrity,new=True)
 print(json.dumps(integrity,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
