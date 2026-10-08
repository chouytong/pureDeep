"""Render frozen analysis outputs; adds no selection/statistical decision rules."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,require,write_json,guard

METRICS=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']
NAMES=['Accuracy','BA','AUROC','Macro-F1','PD Recall','DD Recall']


def table(frame,digits=6):
    columns=list(frame.columns)
    def value(x):
        if isinstance(x,(float,np.floating)):return 'NA' if not np.isfinite(x) else f'{x:.{digits}f}'
        return str(x)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+ '\n'.join('| '+' | '.join(value(v) for v in row)+' |' for row in frame.to_numpy())


def main():
    a=HERE/'analysis';d=json.loads((a/'decision.json').read_text());require(d['status']=='complete','No complete analysis')
    runs=pd.read_csv(a/'seed_split_metrics.csv');summary=pd.read_csv(a/'summary.csv');paired=pd.read_csv(a/'paired_comparisons.csv');seeds=pd.read_csv(a/'seed_means.csv');units=pd.read_csv(a/'15split_seedfirst.csv')
    agreement=pd.read_csv(a/'prediction_agreement.csv');cross=pd.read_csv(a/'method_agreement_15.csv');errors=pd.read_csv(a/'error_summary.csv');trajectories=pd.read_csv(a/'epoch_trajectories.csv')
    diag=json.loads((a/'conditional_diagnostics.json').read_text());lock=json.loads((a/'f0_lock.json').read_text());release=json.loads((a/'preformal_release.json').read_text())
    gate=pd.DataFrame([dict(comparison=k,**v) for k,v in d['gate'].items()])
    means=summary.pivot(index='condition',columns='metric',values='mean').reindex(['P0','P1','P2'])[METRICS].reset_index();means.columns=['Condition']+NAMES
    epochs=runs.groupby('condition').agg(best_epoch_mean=('best_epoch','mean'),best_epoch_median=('best_epoch','median'),stop_epoch_mean=('stop_epoch','mean'),stop_epoch_median=('stop_epoch','median'),best_online_train_ce=('best_online_train_ce','mean'),best_val_ce=('best_validation_ce','mean'),last_online_train_ce=('last_online_train_ce','mean'),last_val_ce=('last_validation_ce','mean'),best_val_ba=('best_validation_ba','mean'),last_val_ba=('last_validation_ba','mean'),best_val_auroc=('best_validation_auroc','mean'),last_val_auroc=('last_validation_auroc','mean')).reset_index()
    timing=[]
    for _,row in runs.iterrows():
        condition=row.condition;seed,c,i=map(int,[row.seed,row.context,row.inner]);stage=Path(next(x['path'] for x in lock['stages'] if (x['seed'],x['context'],x['inner'])==(seed,c,i))) if condition=='P0' else HERE/f'runs/{condition}/seed{seed}/outer_{c}/inner_{i}'
        history=[json.loads(l) for l in (stage/'logs/epochs.jsonl').read_text().splitlines()]
        timing.append(dict(condition=condition,seed=seed,context=c,inner=i,train_epoch_seconds_total=sum(r['train']['duration_seconds'] for r in history),validation_epoch_seconds_total=sum(r['validation']['duration_seconds'] for r in history),mean_train_epoch_seconds=np.mean([r['train']['duration_seconds'] for r in history]),mean_validation_epoch_seconds=np.mean([r['validation']['duration_seconds'] for r in history]),current_training_prediction_wall_seconds=row.training_and_prediction_seconds))
    times=pd.DataFrame(timing);times.to_csv(a/'timing_runs.csv',index=False)
    times_summary=times.groupby('condition',as_index=False).mean(numeric_only=True).drop(columns=['seed','context','inner']);times_summary.to_csv(a/'timing_summary.csv',index=False)
    epochs.to_csv(a/'trajectory_summary.csv',index=False)
    p1=paired[(paired.comparison=='P1-P0')&(paired.metric=='ba')].iloc[0];p2=paired[(paired.comparison=='P2-P0')&(paired.metric=='ba')].iloc[0];order=paired[(paired.comparison=='P2-P1')&(paired.metric=='ba')].iloc[0]
    primary=paired[paired.metric=='ba'][['comparison','mean_delta','improve_count','tie_count','worse_count','ci_low','ci_high','paired_dz','rank_biserial','wilcoxon_p','bh_q_primary_ba3']]
    trajectory=epochs[['condition','best_epoch_mean','stop_epoch_mean','best_online_train_ce','best_val_ce','last_online_train_ce','last_val_ce','best_val_ba','last_val_ba','best_val_auroc','last_val_auroc']]
    agreement_summary=agreement.groupby('condition',as_index=False)[['unanimity','mean_pair_agreement','mean_pair_score_spearman']].mean()
    cross_summary=cross.groupby('comparison',as_index=False)[['prediction_agreement','score_spearman']].mean()
    figs=HERE/'figures';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,4))
    for j,(_,r) in enumerate(primary.iterrows()):
        ax.errorbar(r.mean_delta*100,j,xerr=np.array([[r.mean_delta-r.ci_low],[r.ci_high-r.mean_delta]])*100,fmt='o',capsize=4,color=['#4468ad','#cc7a29','#428d6e'][j]);ax.text(r.ci_high*100+.08,j,f'{int(r.improve_count)}/15 wins',va='center',fontsize=9)
    ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(3),primary.comparison);ax.set_xlabel('BA paired difference (percentage points), bootstrap 95% CI');ax.set_title('15 overlapping development units, seeds averaged first');fig.tight_layout();fig.savefig(figs/'primary_ba_comparisons.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for condition,color in zip(['P0','P1','P2'],['#6a6a6a','#4468ad','#cc7a29']):
        curves=trajectories[trajectories.condition==condition].groupby('epoch').agg(ba=('validation_ba','mean'),ce=('validation_ce','mean'),active=('seed','count'))
        axes[0].plot(curves.index,curves.ba,label=condition,color=color);axes[1].plot(curves.index,curves.ce,label=condition,color=color)
    axes[0].set_ylabel('Validation BA');axes[1].set_ylabel('Validation weighted CE')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend()
    fig.suptitle('Descriptive mean of runs still active; changing composition after stopping');fig.tight_layout();fig.savefig(figs/'validation_trajectories.png',dpi=200);plt.close(fig)
    q1='未建立稳定增量' if not (p1.mean_delta>0 and p1.ci_low>0 and p1.bh_q_primary_ba3<.05) else '建立了当前固定配置下的正向配对证据（仍需同时看辅助指标）'
    q2='未建立超过bag-of-patches的稳定增量' if not (order.mean_delta>0 and order.ci_low>0 and order.bh_q_primary_ba3<.05) else '当前ordered aggregation组合存在超过bag的配对证据，不能单独归因于顺序'
    text=f'''# Patch-based Temporal STR 正式开发报告（2026-10-08）

## 最终结论：{d['decision']}

- Q1：P1对P0，{q1}。BAΔ={p1.mean_delta:+.6f}，{int(p1.improve_count)}/15胜，95%CI[{p1.ci_low:+.6f},{p1.ci_high:+.6f}]，primary BH q={p1.bh_q_primary_ba3:.6f}。
- Q2：P2对P1，{q2}。BAΔ={order.mean_delta:+.6f}，{int(order.improve_count)}/15胜，95%CI[{order.ci_low:+.6f},{order.ci_high:+.6f}]，q={order.bh_q_primary_ba3:.6f}。
- Q3：P2对P0 BAΔ={p2.mean_delta:+.6f}，{int(p2.improve_count)}/15胜，CI[{p2.ci_low:+.6f},{p2.ci_high:+.6f}]；正式决策按冻结的全部门槛为 **{d['decision']}**，不能只挑正向指标。
- Q4：当前结果符合“固定local patch配置未建立稳定BA增量”，不是“已证明patch有效但order无效”。P1/P2均有小幅AUROC均值增加，但不能替代BA门槛；不能从失败推导所有patch表征对PD/DD无用，也不能宣称已证明原STR存在information loss。

## 1. 实际执行与边界

P0完整45checkpoint全validation复现、15个train-only normalization refit、已有Phase2当前source45次scratch复现均PASS，正式P0重用。P1/P2各完整15split×3seed=45scratch runs，共90，另有correctness two-step与one-batch one-epoch smoke；P3未在本阶段训练。没有使用outer-test信号/结果，没有WSSL/HarNet/Frequency/FFT，没有改loss、weights、sampler、augmentation、optimizer、LR、WD、scheduler、batch、patience、threshold、activity/wrist顺序或原STR路径。context/outer_N目录仅development split索引，test subjects列表为空。

所有候选正式训练前冻结源码和统计；preformal Git `{release['commit']}`，tag `{release['tag']}`；study_lock SHA `{release['study_lock_sha256']}`。训练后audit/hash guard PASS，正式STR checkpoint/prediction/source未修改。P0旧source whole-tree hash不能直接说完全等于当前；最新Phase2匹配source加本轮数值复现建立可复现性，详见BASELINE_AUDIT.md。

参数：P0 75,524；P1 80,340（+4,816）；P2 80,661（+5,137，比P1多321）；本轮全部参数trainable，无外部encoder。200sample/stride100，976→9patch，2000→19patch；共享6→32→64encoder，无perpatch normalization；zero-init64→16→64 residual注入原wrist64。

## 2. 正确性与新增文件

- extraction数量/start/first/last/tail/no-overflow/determinism/autograd和padding隔离PASS。
- 两条件45个正式checkpoint零初始化logits/重载逐值一致；三seed原STR初始化、CPU/CUDA RNG一致，P1/P2共用encoder/head逐值一致。
- 第一次backward上游encoder/aggregator梯度0、projection非零；第二次上游有限非零、参数实际更新；P1/P2模型+optimizer保存重载一致。
- absent wrist/activity、padded sample/patch、空patch行隔离PASS；P1置换不变，P2有顺序响应。
- 原训练engine两个smoke PASS，保存best后重载validation inference；smoke分数未参与筛选。
- 独立目录scripts包括common/audit_p0、patch_extractor/encoder/models、test_extraction/models、run_condition/launch_matrix、analyze_results/test_analysis_contract、freeze_study、conditional_diagnostics、publish_stage；report rendering只展示冻结分析产物，不修改统计或规则。

## 3. 六指标：seed-first15split均值

{table(means)}

以下表格小数为比例；BA差乘100才是百分点。没有只报最好seed、split、epoch或ensemble。

## 4. Primary BA配对证据（BH family=3）

{table(primary)}

![BA paired evidence](figures/primary_ba_comparisons.png)

95%CI为10,000个paired split重采样，bootstrap seed20261008；Wilcoxon双侧、dz和rank-biserial分别报告。bootstrap与Wilcoxon/BH是不同结果，不互替。15划分共享subjects与train pool，CI及p/q仅repeated-development描述，不能当独立外部确认。

## 5. 全部配对指标（18个探索性BH）

{table(paired[['comparison','metric','mean_delta','improve_count','tie_count','worse_count','ci_low','ci_high','paired_dz','rank_biserial','wilcoxon_p','bh_q_18']])}

## 6. 种子与split波动

{table(seeds)}

{table(summary[['condition','metric','seed_mean_sd','mean_within_split_seed_sd','split_sd']])}

跨seed预测一致性（每split再平均）：

{table(agreement_summary)}

同seed不同模型的一致性（seed-first再15split平均）：

{table(cross_summary)}

## 7. 训练轨迹、停止与耗时

{table(trajectory)}

![Validation trajectories](figures/validation_trajectories.png)

epoch曲线只作描述；后期仅未停止runs仍存在，组成变化不能解释为配对性能改进。online train loss/BA含训练Dropout，不是eval-mode train performance；不据此宣称真实泛化gap。只有best/last权重，不虚构中间checkpoint或EMA。epoch_trajectories.csv保留所有训练epoch的train CE/BA、validation CE/BA/AUROC/F1/两Recall；best/stop逐run保留。无结果后增加max epochs/patience。

{table(times_summary)}

P0 timing来自原归档train/validation duration，不是本轮重新运行；其current wall time为NA。P1/P2 wall time是真实本轮原引擎训练+预测耗时；不把历史硬件/runtime差异当严格速度配对证据。

## 8. 纠正/伤害变化（每validation unit平均，不是独立subjects数）

{table(errors[['comparison','label','subject_count','reference_errors','candidate_errors','corrected','harmed','net_corrected']])}

同一subject可能在多split出现；这些是seed-first15unit平均的事件计数。DD subtype门槛为P2正式RETAIN，未满足时不访问或分析clinical subtype表。

## 9. 保留门槛逐项审计与条件诊断

{table(gate)}

规则直接沿用已有三条件STR residual protocol，训练前固定：P2对P0/P1均须BAmean>0、≥10wins、CI下界>0、primaryBA BH<.05；AUCΔ≥−.005、F1不下降、两Recall各Δ≥−.01，BA/AUC withinseedSD≤1.25×reference（floor.001）。不按单个显著结果或attention外观保留。

利用/order-shuffle状态：`{diag['status']}`。P1利用诊断允许={d['p1_utilization_allowed']}；P2/order-shuffle允许={d['p2_order_shuffle_allowed']}；DD-subtype/P3门槛={d['p3_allowed']}。未满足门槛的解释分析直接skip，不用于“救援”或活动选择。

## 10. 机制判断与后续

本轮正向但不足的证据是P2-P0 AUROCΔ+0.003662（11/15胜，bootstrap95%CI[+0.000435,+0.007246]）；Wilcoxon p=0.072998、18探索比较BH q=0.778617，必须与bootstrap结果分开保留。BA primary三个q均0.761536，三个CI均跨0。P2-P1虽BA均值微正，但只有5/15胜、10/15负，未形成可重复的ordered增量。Macro-F1的P2-P0负差约−0.000024非常小，不能宣称临床上明显恶化；其违反严格非下降门槛，而BA不提升已足以拒绝。

P0/P1/P2 best epoch平均13.49/13.56/13.80，stop epoch平均25.49/25.56/25.80，均median best12/stop24。没有明显系统性更早停止或更晚最佳点；best后validation CE上升、BA下降在三条件均发生，未显示patch方案解决晚期退化。online train CE不能用于计算eval-mode泛化gap，也不据此更改epoch/patience。

mean within-split seed BA SD为0.030011/0.031382/0.030743，AUROC SD为0.032350/0.029649/0.030520；正式种子保护均通过。因此失败重点是未建立稳定配对增量，而不是宣称seed variance保护失败。仅3个seed均值的SD另有增加，完整报告保留，不能对三个seed作独立显著性推断。

绘图依赖曾缺失，已安装在独立/home/zyt/envs/patch_report_deps_20261008目录；只用于本报告渲染，不进入训练或冻结统计环境。未改冻结研究文件，非实验协议偏离。

P2-P1同时增加depthwise序列卷积、attention pooling和321参数，不能把任何差异孤立归因于order或capacity。窗口的标称100Hz/2s是固定设计，不代表已验证最佳生理时间尺度。当前结果只覆盖这一实现、这一输入处理与固定训练budget。

正负结果、CI跨0、种子/Recall trade-off均完整保留；当前 **{d['decision']}** 不改变既有FrozenWSSL retained best。P3执行条件={d['p3_allowed']}；不满足则STOP temporal adjacency pretext、WSSL+Patch、patch/stride/hidden/pooling rescue。Frequency组合始终禁止。若门槛满足，仅允许按预注册规则进入条件阶段，不能扩大搜索。

## 阶段验收摘要

1. 完成：审计、正确性、90次训练、135行run性能、15unit统计、轨迹/错误/条件门槛。
2. 文件：本独立artifact代码、PROTOCOL/审计/test/smoke报告、analysis聚合表、figures、本报告；README和handoff追加结果。
3. 正确性及正式run审计PASS。
4. outer未访问。
5. 原recipe未改。
6. 无协议偏离；未执行不满足门槛的后续。
7. 完整六指标与三主比较/18探索比较如上。
8. 下一阶段资格严格见冻结decision.json，不事后重定义。
'''
    (HERE/'PATCH_STR_REPORT.md').write_text(text)
    write_json(a/'report_render.json',dict(status='PASS',frozen_decision=d['decision'],report='PATCH_STR_REPORT.md',figures=['primary_ba_comparisons.png','validation_trajectories.png'],selection_changed=False))
    guard(lock);print('REPORT RENDERED',d['decision'],flush=True)

if __name__=='__main__':main()
