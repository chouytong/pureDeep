"""Chinese packaging of frozen Phase A results; no change to case/statistics.

Additional whole-cohort score-shift summaries use the already requested private
STR/WSSL prediction link, without inferential tests or model-selection rules.
"""
from common import *
from scipy.stats import spearmanr


def table(headers, rows):
    return '\n'+'\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])+'\n'


def number(value):return f'{float(value):+.6f}'
def ci(row):return f"[{float(row['ci_low']):+.6f}, {float(row['ci_high']):+.6f}]"
def load(name):return pd.read_csv(HERE/'analysis'/f'{name}.csv',keep_default_na=False)


def main():
    guard();decision=json.loads((HERE/'analysis/decision.json').read_text())
    require(not (HERE/'PHASE_A_REPORT.md').exists(),'Do not overwrite final report')
    activity=load('activity_contribution');wrist=load('wrist_contribution');axw=load('activity_wrist_contribution')
    stats=load('contribution_summary_long');gains=load('STR_WSSL_gain_summary');groups=load('subject_group_WSSL_STR_gain')
    group_counts=load('subject_group_counts');group_contributions=load('subject_group_contribution');shifts=load('subject_shift_concentration')
    direct=load('wrist_direct_difference');full=load('full_reproduction_45runs');seeds=load('seed_profiles')
    # Pure STR link: same existing subjects/checkpoints, new descriptive summaries only.
    frames=[];agreement=[]
    for seed in [42,43,44]:
        for c in range(5):
            for i in range(3):
                w=predictions(stage_path(seed,c,i)/'predictions/validation.csv');s=predictions(str_path(seed,c,i)/'predictions/validation.csv')
                require(w.index.equals(s.index) and np.array_equal(w.target,s.target),'Formal STR/WSSL link changed')
                f=pd.DataFrame(dict(subject_id=w.index,target=w.target.to_numpy(),seed=seed,context=c,inner=i,
                       p_wssl=w.probability_dd.to_numpy(),p_str=s.probability_dd.to_numpy(),
                       pred_wssl=w.prediction.to_numpy(),pred_str=s.prediction.to_numpy()))
                f['raw_delta']=f.p_wssl-f.p_str;f['signed_delta']=(2*f.target-1)*f.raw_delta;f['absolute_delta']=abs(f.raw_delta)
                frames.append(f)
                agreement.append(dict(seed=seed,context=c,inner=i,score_spearman=float(spearmanr(f.p_wssl,f.p_str).statistic),
                     prediction_agreement=float((f.pred_wssl==f.pred_str).mean()),raw_delta_mean=f.raw_delta.mean(),raw_delta_sd=f.raw_delta.std(ddof=1)))
    f=pd.concat(frames,ignore_index=True);people=f.groupby(['subject_id','target'],as_index=False)[['raw_delta','signed_delta','absolute_delta']].mean()
    descript=[]
    for label in ['all','PD','DD']:
        a=people if label=='all' else people[people.target==(0 if label=='PD' else 1)]
        r=f if label=='all' else f[f.target==(0 if label=='PD' else 1)]
        values=a.absolute_delta.to_numpy();top=int(np.ceil(.1*len(a)))
        changed=r.pred_wssl!=r.pred_str
        descript.append(dict(label=label,subject_count=len(a),mean_raw_DD_probability_delta=a.raw_delta.mean(),
             mean_true_label_oriented_delta=a.signed_delta.mean(),median_absolute_subject_delta=float(np.median(values)),
             mean_absolute_subject_delta=values.mean(),top10pct_n=top,top10pct_absolute_shift_share=float(np.sort(values)[-top:].sum()/values.sum()),
             changed_prediction_fraction=changed.mean(),signed_subject_mean_positive_count=int((a.signed_delta>0).sum()),
             median_STR_distance_from_05_flipped=float(abs(r.loc[changed,'p_str']-.5).median()),
             median_STR_distance_from_05_unchanged=float(abs(r.loc[~changed,'p_str']-.5).median())))
    pd.DataFrame(descript).to_csv(HERE/'analysis/WSSL_STR_subject_shift_descriptive.csv',index=False)
    pd.DataFrame(agreement).to_csv(HERE/'analysis/WSSL_STR_score_agreement_45runs.csv',index=False)
    corr=load('activity_gain_link_descriptive_correlations')
    corr15=corr.groupby(['context','inner','activity'],as_index=False).mean(numeric_only=True)
    corrsummary=corr15.groupby('activity',as_index=False).mean(numeric_only=True)
    corrsummary.to_csv(HERE/'analysis/activity_gain_link_summary.csv',index=False)
    names=dict(accuracy='Accuracy',ba='BA',auroc='AUROC',macro_f1='Macro-F1',pd_recall='PD Recall',dd_recall='DD Recall')
    group_names=dict(stable_correct='stable-correct',stable_error='stable-error',unstable='unstable')
    text=['# Phase A：Frozen WSSL-STR 增量贡献诊断报告',
      '\n日期：2026-10-07。**已完成；CASE '+decision['case']+' / '+decision['decision']+'。** 当前最佳仍是原 Frozen WSSL-STR。',
      '\n**本轮只有 inference-only 诊断，训练次数/optimizer steps 均为0；未启动 Phase B 或 Phase C。** PROCEED 表示满足提出单一11-scalar对照的证据资格，不表示已经批准或运行下一阶段。',
      '\n## 1. 实际执行与复现验收',
      '\n原45个正式 WSSL checkpoint（15 development splits × seeds42/43/44）全部完整 validation inference。验证两个归档概率列/decision/六指标，ID/label、train/validation分离、SSL ID查找/activities/两腕/finite/window counts/cache SHA匹配；15次原inner-train normalization refit与45checkpoint mean/std逐位相同。全部原权重、cache、正式源文件和结果SHA未变。',
      '\n**归档CSV没有logits。** 全部validation batches中原历史forward、新独立显式SSL接口、复用原wrist与projected residual的诊断forward logits逐位一致；归档概率最大差异1.11×10⁻¹⁶，decision和指标一致。不伪称对照了不存在的归档logit文件。',
      '\n实际residual为 `Linear(LayerNorm(SSL))`。只在该投影输出后清零，包括bias；不是把SSL输入清零。首个真实8subject batch的37条件与projection输出hook的直接完整forward逐位一致；零SSL输入与真实all-residual-off最大logit差异0.00214644。checkpoint重载/ID重排、模型状态/cache不变PASS。没有scratch训练、backward或HarNet重提取。',
      '\n全部45实例执行：11 activity-off、left-off、right-off、22 activity×wrist-off，共35个用户指定mask；另外一个预先固定all-WSSL-off诊断锚点。加full共37条件、1,665个condition-run指标行、173,160行私有subject-condition记录。原STR wrist/activity token、length、mask、bilateral/activity/structured路径完整保留。缓存存储受NPZ格式限制可能整体解压，但在构建tensor前仅选当前validation ID，排除行不进入forward或统计；inner-train信号仅用于原normalization refit。无outer loader/outcome/performance。',
      '\nclassifier侧参数143,172，外部HarNet frozen参数10,457,408，总计10,600,580；新增模型参数0。独立诊断接口不改变正式模型结构。完整复现、屏蔽接口、统计代码/CASE门槛已于masked结果之前SHA冻结及Git发布，见 [复现报告](REPRODUCTION_REPORT.md)、[运行前协议](PROTOCOL.md)、`analysis/pre_mask_publication.json`。',
      '\n## 2. 正式 WSSL 与纯 STR 联系（不是重新训练）',
      '\n所有checkpoint的validation ID/label/split/seed严格一致。以下正式performance先在split内平均三个seed的指标，再以15split等权汇总：',
      table(['模型']+[names[m] for m in METRICS],[[label]+[f"{float(gains[gains.metric==m].iloc[0][field]):.6f}" for m in METRICS] for label,field in [('原STR-01','metric_str'),('Frozen WSSL-STR','metric_full')]]),
      table(['WSSL−STR指标','均值Δ','中位数Δ','正/零/负split','split SD','bootstrap95%CI'],[
          [names[r.metric],number(r.mean_delta),number(r.median_delta),f'{r.positive_split_count}/{r.zero_split_count}/{r.negative_split_count}',f'{r.split_sd:.6f}',ci(r)] for _,r in gains.iterrows()]),
      '\nWSSL的正式改善以DD更明显：DD Recall +.040430（约4.04pp，11正/1零/3负），PD +.004480但区间跨零；AUROC +.041922，15/15正。不是单独Accuracy提高。该比较仍复用了已用于开发的subjects/splits，CI仅描述重复development，不增加外部支持。',
      '\n两类改善差距是描述性数值，不以一类CI跨零、另一类不跨零证明类间显著差异；本轮未做新的类间显著性推断。',
      '\n## 3. Table 1：activity contribution',
      '\n**Contribution=Full−Masked；正数表示删除该activity的SSL路径后性能下降。** 所有数值为概率/metric单位，不是百分数；pp=数值×100。positive_split_count指BA的15seed-first split。',
      table(['Activity']+['Δ'+names[m] for m in METRICS]+['BA正/零/负split'],[
          [r.activity]+[number(r[m+'_delta']) for m in METRICS]+[f'{r.positive_split_count}/{r.zero_split_count}/{r.negative_split_count}'] for _,r in activity.iterrows()]),
      '\n### BA描述性稳定性（每个activity保留完整15split，不将activity当独立样本）',
      table(['Activity','meanΔBA','medianΔBA','split SD','bootstrap95%CI'],[
          [r.activity,number(r.ba_delta),number(r.median_delta),f'{r.split_sd:.6f}',ci(r)] for _,r in activity.iterrows()]),
      '\n### AUROC描述性稳定性',
      table(['Activity','meanΔAUROC','正/零/负split','bootstrap95%CI'],[
          [r.activity,number(r.mean_delta),f'{r.positive_split_count}/{r.zero_split_count}/{r.negative_split_count}',ci(r)] for _,r in stats[(stats.kind=='activity')&(stats.metric=='auroc')].iterrows()]),
      '\nTouchNose BA贡献+.060713/AUROC+.035089，均15/15正；CrossArms BA+.043440（12/15）/AUROC+.012101（11/15）；TouchIndex BA+.020917（13/15）。Entrainment BA+.001901且CI跨零；Relaxed/RelaxedTask BA仅约+.0033/+.0030，中位数为0，不能按临床想象称其最强。11activity平均BA都为正，但效应幅度和类方向明显不均匀。DrinkGlas平均AUROC微负−.000703、CI跨零；不是稳定harm。',
      '\n类贡献不同：CrossArms主要支持PD（PD Recall+.071710）；TouchNose主要支持DD（DD+.151451）但PD−.030026；TouchIndex也DD正/PD负。不能把这些当成活动本身的因果疾病特异性，也不能据此删活动或validation-informed初始化未来g。',
      '\n### Activity profile的预先固定分流',
      f"\n平均BA profile的max−min={decision['BA_activity_profile_spread']:.6f}，描述性bootstrap95%CI[{decision['spread_bootstrap_CI'][0]:.6f},{decision['spread_bootstrap_CI'][1]:.6f}]。三个seed的profile Spearman为.7091/.7818/.9000（中位数.7818）；split与其余14split均值profile的leave-one-split-out Spearman中位数.7000，15/15为正；7个activity在≥10/15split中正BA contribution。",
      table(['运行前诊断闸门','结果'],[[k,'通过' if v else '未通过'] for k,v in decision['diagnostic_gate'].items()]),
      '\n**CASE A。** 预定threshold只是本轮的实用诊断分流值，不是临床界值或已知最优设计。max/min是同一整体profile的描述统计，bootstrap内重算spread；没有用11个显著性检验选择最佳activity。其CI不是经过选择修正的某两个活动的因果差异。',
      '\n### 图：完整−屏蔽的BA与AUROC（诊断，不作activity selection）',
      '\n![Activity BA contribution](figures/activity_ba_contribution.png)',
      '\n![Activity AUROC contribution](figures/activity_auroc_contribution.png)',
      '\n## 4. Table 2：left/right contribution',
      table(['屏蔽条件']+['Δ'+names[m] for m in METRICS]+['BA正/零/负split','BA bootstrap95%CI'],[
          [r.wrist]+[number(r[m+'_delta']) for m in METRICS]+[f'{r.positive_split_count}/{r.zero_split_count}/{r.negative_split_count}',ci(r)] for _,r in wrist.iterrows()]),
      '\n### 直接比较左−右贡献（同一15split配对）',
      table(['指标','左−右均值','正/零/负split','bootstrap95%CI'],[
          [names[r.metric],number(r.mean_delta),f'{r.positive_split_count}/{r.zero_split_count}/{r.negative_split_count}',ci(r)] for _,r in direct.iterrows()]),
      '\n两腕都重要：关闭单腕BA下降约6.56/6.23pp，均15/15正；但左−右BA仅+.003272、CI跨零，AUROC差+.000294、CI跨零。三个seed BA左−右方向为负/正/负，AUROC同样负/正/正。**没有稳定的整体left/right asymmetry证据**；不做wrist selection，也没有患侧/利手临床信息来解释侧性。',
      '\n## 5. Table 3：activity×wrist contribution（22条件全部执行）',
      table(['Activity','Wrist']+['Δ'+names[m] for m in METRICS]+['BA正/零/负split','BA bootstrap95%CI'],[
          [r.activity,r.wrist]+[number(r[m+'_delta']) for m in METRICS]+[f'{r.positive_split_count}/{r.zero_split_count}/{r.negative_split_count}',ci(r)] for _,r in axw.iterrows()]),
      '\nTouchNose左右均BA正15/15（+.033621/+.035152），CrossArms左右也均有贡献（+.025797/+.024211）。全表保留近零及负AUROC点估计，不包装22条件的显著性，不挑活动/腕或修改网络。删除双腕的效应一般不等于删除左右单腕效应之和，因bilateral/attention/structured路径非线性且共同适配。',
      '\n## 6. Table 4：subject-group contribution与真实WSSL−STR改善',
      '\n主定义固定为DSG/RGD/PRR/PAG：每次validation先均值3seed概率，再对每人4次appearance求错误率；≥.75 stable-error、≤.25 stable-correct，其余unstable。用独立训练STR定义primary groups，74/284/32逐位复现人数。分组用validation correctness，具有选择效应和regression-to-mean风险；不是前瞻性ambiguity或临床phenotype。',
      '\n本表的错误率/恢复/新增错误基于**四次seed-mean概率决策**，与正式表的“先平均三个seed指标”不同；只是描述，不是新正式性能。人数为unique subjects；appearance counts重复同一subject，不用于扩大样本量。',
      table(['STR主组','类别','unique n','appearances','STR error rate','WSSL error rate','恢复','新增错','净恢复','true-label probability gain'],[
          [group_names[r.group],r.label,r.subject_count,r.development_appearance_count,f'{r.str_error_fraction:.4f}',f'{r.wssl_error_fraction:.4f}',
           r.WSSL_corrected_appearances,r.WSSL_harmed_appearances,r.net_corrected_appearances,number(r.mean_signed_probability_gain)] for _,r in groups.iterrows()]),
      '\nSTR stable-error组PD/DD的净恢复分别+36/+44 appearances；unstable为+6/+3；STR stable-correct反而−51/−25。这说明改善和损害同时存在；净恢复主要集中在原STR持续难例组，**不是主要修复unstable subjects**。但不能因分组选择效应将其解释为已证明的临床phenotype-overlap解决。WSSL同口径primary stable-error仍有68人（PD35/DD33），persistent errors没有消失。',
      '\n本表汇总的seed-mean决策PD净恢复为−9、DD为+22 appearances；PD与正式单seed指标均值的微小正Δ不同。这是不同aggregation口径，不能把该分组表当作正式PD Recall的分解或替换原指标。',
      '\n### 主组内当前WSSL对all-residual的依赖（完整−all-off锚点）',
      table(['STR主组','类别','n','Full error','All-off error','residual净恢复appearances','signed probability contribution','mean absolute shift'],[
          [group_names[r.group],r.label,r.subject_count,f'{r.full_error_fraction:.4f}',f'{r.masked_error_fraction:.4f}',r.net_residual_corrected_appearances,
           number(r.mean_signed_true_label_delta),f'{r.mean_absolute_probability_shift:.6f}'] for _,r in group_contributions[(group_contributions.group_basis=='str_primary')&(group_contributions.condition=='all_ssl_off')].iterrows()]),
      '\n所有11activity与两腕/22交互的group表保存于 `analysis/subject_group_contribution.csv`，不只汇报上述强贡献路径。TouchNose的DD净恢复出现在三组（+31/+20/+3 appearances），同时PD三组有负净贡献；CrossArms主要PD stable-correct净贡献+41，但原STR stable-error DD为−4。这显示类/subject异质性，不得用名单重加权。',
      '\n### 定义差异和sensitivity（不混人数）',
      table(['分组模型','定义','stable-correct n','stable-error n','unstable n'],[
          [model,definition]+[int(group_counts[(group_counts.group_basis==model)&(group_counts.definition==definition)&(group_counts.group==g)].subject_count.sum()) for g in ['stable_correct','stable_error','unstable']]
          for model in ['str','wssl'] for definition in ['primary','strict4','all12']]),
      '\nprimary如上；strict4为四次seed-mean全部对/全部错（EMA报告口径）；本轮all12字段是12个individual seed/context全部对/全部错的严格sensitivity（EMA式），**不是DSG历史12-appearance错误率阈值表**。运行前协议中“历史12”字样范围过宽，在此明确实际脚本的严格定义；不把不同阈值的人数并入primary或声称复现历史DSG sensitivity人数。所有辅助分组都未参与CASE/gate或模型选择。',
      '\n## 7. 概率变化、边界subjects与纯STR联系',
      table(['类别','n','mean raw Δp_DD(WSSL−STR)','true-label meanΔp','median absolute subject shift','top10% absolute-shift share','individual prediction changed fraction'],[
          [r['label'],r['subject_count'],number(r['mean_raw_DD_probability_delta']),number(r['mean_true_label_oriented_delta']),f"{r['median_absolute_subject_delta']:.6f}",
           f"{r['top10pct_absolute_shift_share']:.3%}",f"{r['changed_prediction_fraction']:.3%}"] for r in descript]),
      '\n每人的probability summary先均值其12个既有development appearances，人数不用于CI。top10%仅是幅度集中度描述，不是hard-subject removal、样本选择或性能改写。',
      '\n### 强路径屏蔽的幅度集中度与边界关系',
      table(['条件','top10% shift share','prediction flip fraction','flipped full距离.5中位数','unchanged full距离.5中位数'],[
          [r.condition,f'{float(r.top10pct_absolute_shift_share):.3%}',f'{float(r.changed_prediction_fraction):.3%}',f'{float(r.median_full_boundary_distance_flipped):.6f}',f'{float(r.median_full_boundary_distance_unchanged):.6f}']
          for _,r in shifts[shifts.condition.isin(['activity_off__TouchNose','activity_off__CrossArms','left_off','right_off','all_ssl_off'])].iterrows()]),
      '\nTouchNose/CrossArms删除的top10%幅度份额约22%/20%，明显不是只由极少数subjects承担全部概率扰动；翻转确实更多发生在较靠近.5的subjects。raw Δp在subjects之间具有较大离散，并非同一个常数平移。影响覆盖较广，但净分类收益更集中，二者不能混为一谈。',
      '\n### Activity removal effect与WSSL−STR score gain相关（仅描述）',
      table(['Activity','raw probability effect vs gain Spearman均值','true-label-oriented Spearman均值'],[
          [r.activity,f'{r.raw_probability_delta_vs_WSSL_STR_gain_spearman:.4f}',f'{r.signed_delta_vs_signed_gain_spearman:.4f}'] for _,r in corrsummary.iterrows()]),
      '\n相关先每个split内平均三seed，再均值15split。不能将该相关当成activity因果份额；独立STR与WSSL的训练权重不同，删除residual还会触发trained model的非线性响应。',
      '\n## 8. all-off锚点与解释边界',
      '\n全WSSL关闭后，BA从.719644降至.604457，AUROC从.759553降至.670865，DD Recall降约.238375；原独立训练STR BA却为.697189。因此**当前WSSL分支删除依赖远大于独立STR→WSSL的净收益**，与共同适配或大幅mask造成的分布偏移等解释相容，但本诊断不能分离这些原因；不能把删除损失当作纯预训练知识的可加增益。mask=0是大的离训练状态干预，不能据此预言从g=1附近训练scalar一定有效。',
      '\n1. 这是固定已训练模型的诊断依赖，未进行随机临床干预、重新训练matched ablation或独立队列验证。activity/wrist/bilateral/attention之间存在交互，贡献不可相加。',
      '\n2. 15splits共享subjects/training pools，三个seed不是独立样本；1,665 condition-runs、173,160subject-condition记录、11activities、22交互和bootstrap次数均不构成独立样本量。',
      '\n3. bootstrapCI只描述重复development。不做11/22条件显著性筛选或声称FDR确认；所有负结果保留。多阶段重复development已有selection optimism，LOSO只是profile一致性检查，不是独立验证。',
      '\n4. class/group效应不能等同病理因果、疾病严重程度或患侧解释。没有访问outer information、改threshold、删activity、DD重采样、改HarNet/STR或引入encoder。',
      '\n5. 归档缺少logits与历史stable-error口径差异均明确报告；没有篡改旧结果。本轮主定义固定，sensitivity不改变人数/CASE判断。',
      '\n## 9. 三个问题与下一步分流',
      '\n**Q1：activity间明显不均匀吗？是，在固定WSSL模型的删除依赖意义上。** TouchNose/CrossArms/TouchIndex与Entrainment/Relaxed等幅度不同，seed profile相关和15split LOSO方向支持重复性；不证明总WSSL净gain可以按activity加法分解。',
      '\n**Q2：有稳定left/right asymmetry吗？没有。** 两腕都提供贡献，直接paired差异不稳定，不能据某一个seed或某一个activity选腕。',
      '\n**Q3：主要改善谁？正式metric以DD recognition与ranking更明显。** PD整体增益小且不稳定；按原STR主组描述，净分类恢复主要在stable-error组而非unstable；同时损害部分原STR stable-correct。概率变化覆盖较广、符号和幅度依label/subject/activity不同，不能解释为整体轻微常数平移。',
      '\n**下一步：CASE A / PROCEED，仅限提出一项Activity-conditioned WSSL Scaling对照的资格。** 若用户确认继续，唯一允许设计是所有11activity各一个scalar、全部g=1统一初始化；不由贡献图设置g、删活动或限制特定方向，不扩展MLP/attention/wrist gate，不叠加Frequency/EMA/loss/augmentation。本轮没有选择新的configuration，也不预期g一定提升性能。',
      '\n**Phase B未训练；Phase C未做数值/gradient audit。** 按用户“不要自动进入下一阶段”的要求，本轮在中文报告和Git归档后停止。Phase C将来独立核对历史覆盖与batch/loss/gradient；不与Phase B同时修改。',
      '\n## 10. 文件、归档与Git',
      '\n新增独立脚本 `common.py`、`run.py`、`analyze.py`、`render_plots.py`（均premask冻结），报告整理脚本 `write_report.py`（结果后仅整理/补充描述，不参与CASE门槛）。源production/正式checkpoint/SSL cache没有修改。protocol与复现报告、37条件45run及seed-first15split指标、216+6metric summaries、四类表、稳定性/分组/错误/score联系表与BA/AUROC PNG+SVG均保存在独立artifact。私有逐subject预测与编码parts不公开。',
      '\n运行前版本：commit `b078280ee24531b4693f593d9a2009f4dca9a389`，tag `wssl-contribution-reproduction-freeze-20261007`。最终报告/正负结果/边界/PROCEED但未训练决定追加README及handoff，再提交、push和完成tag；实际远端SHA/标签/匿名公开访问/文件排除检查保存在外部publication receipt，避免自引用commit hash。',
      '\n最终保留模型：**原 Frozen WSSL-STR，没有新模型候选被训练或保留。**'
    ]
    (HERE/'PHASE_A_REPORT.md').write_text('\n'.join(text)+'\n')
    guard();print('CHINESE PHASE A REPORT WRITTEN; fixed decision unchanged; no next-stage training')


if __name__=='__main__':main()
