"""Chinese packaging of complete fixed analysis; no model/selection changes."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent.parent

def table(headers,rows):
    return '|'+ '|'.join(headers)+'|\n|'+ '|'.join(['---']*len(headers))+'|\n'+ '\n'.join('|'+ '|'.join(map(str,r))+'|' for r in rows)+'\n'

def main():
    a=HERE/'analysis';d=json.loads((a/'scaling_decision.json').read_text());q=json.loads((a/'correctness.json').read_text())
    assert d['status']=='complete' and q['status']=='PASS'
    summary=pd.read_csv(a/'scaling_summary.csv').set_index('variant');pair=pd.read_csv(a/'scaling_paired.csv').set_index('metric')
    b=summary.loc['baseline'];c=summary.loc['activity_scaling'];names={'accuracy':'Accuracy','ba':'BA','auroc':'AUROC','macro_f1':'Macro-F1','pd_recall':'PD Recall','dd_recall':'DD Recall'}
    performance=table(['Metric','WSSL baseline','Activity-Scaled','Δ','正/零/负 splits','paired bootstrap 95% CI','dz'],[
        [name,f'{b[m]:.6f}',f'{c[m]:.6f}',f'{pair.loc[m,"mean_delta"]:+.6f}',f'{int(pair.loc[m,"improved_splits"])}/{int(pair.loc[m,"tied_splits"])}/{int(pair.loc[m,"worse_splits"])}',f'[{pair.loc[m,"ci95_low"]:+.6f}, {pair.loc[m,"ci95_high"]:+.6f}]',f'{pair.loc[m,"cohen_dz"]:+.3f}'] for m,name in names.items()])
    stability=table(['Metric','within-split seed SD 原→新','3 seed-mean SD 原→新','15split SD 原→新'],[[name,*[f'{b[m+s]:.6f} → {c[m+s]:.6f}' for s in ['_within_split_seed_sd','_seed_mean_sd','_split_sd']]] for m,name in names.items()])
    g=pd.read_csv(a/'learned_scales_summary.csv').set_index('activity');cfgorder=q.get('activity_order',list(g.index))
    scales=table(['Activity','g整体均值','15split SD','15split范围','seed42/43/44 mean','45run范围'],[[name,f'{g.loc[name,"mean"]:.6f}',f'{g.loc[name,"std"]:.6f}',f'[{g.loc[name,"min"]:.6f},{g.loc[name,"max"]:.6f}]',' / '.join(f'{g.loc[name,f"seed{s}_mean"]:.6f}' for s in [42,43,44]),f'[{g.loc[name,"run_min"]:.6f},{g.loc[name,"run_max"]:.6f}]'] for name in cfgorder])
    agreement=pd.read_csv(a/'scaling_prediction_agreement.csv').groupby('variant').mean(numeric_only=True)
    matched=pd.read_csv(a/'matched_prediction_agreement_15split.csv').prediction_agreement.mean()
    errors=pd.read_csv(a/'scaling_error_summary.csv');error_table=table(['Disease','baseline errors/split','candidate errors/split','corrected','newly wrong','net'],[[r.disease,*[f'{getattr(r,k):.3f}' for k in ['baseline_error_count','candidate_error_count','corrected_count','newly_wrong_count','net_corrected_count']]] for r in errors.itertuples()])
    post=json.loads((a/'scale_posthoc_diagnostics.json').read_text());post_table=table(['Posthoc比较','Spearman rho','描述p'],[[x.get('activity','Phase A activity profile')+' / '+x['metric'],f'{x["rho"]:+.4f}',f'{x["p_descriptive"]:.5f}'] for x in post['contribution_correlation']+post['class_tradeoff_correlation']])
    gates=table(['运行前闸门','结果'],[[k,'PASS' if v else 'FAIL'] for k,v in d['gate'].items()])
    report=f'''# Phase B：Activity-Scaled WSSL 正式报告（中文）

## A. Correctness

新增独立 `scripts/scaled_wssl.py`、`run.py`、`test_correctness.py`、`launch.py`、`analyze.py`、`write_report.py` 与 `PROTOCOL.md`、analysis小型汇总。reference_*文件仅归档复用出处，不执行其候选入口。原 foundation/正式模型源码未改。附加11个无约束g，配置顺序建立index，同activity左右腕共享；uniform1；完整scratch训练，所有原classifier参数训练。实例绑定projection hook对完整output（含bias）缩放，随后原mask，后续bilateral/activity/structured路径完全沿用。不是zero-input，不用贡献初始化或删活动，不deepcopy。

参数：classifier-side 143,172→143,183；HarNet冻结10,457,408；系统10,600,580→10,600,591。缓存推理阶段不实例化HarNet，不生成新统计状态。g继承原AdamW同组WD，没有额外参数组。

全部45checkpoint×两个真实8subject validation batch（90批次、每checkpoint16subjects）g=1 logits与重算ordinary baseline逐位一致，fresh reload逐位一致；归档概率误差≤1e-12。正式CSV未存logits，未声称比较不存在的归档logits。Phase A已全validation复现，15train-only normalization refit和45正式资产SHA复核不变。

三seed scratch原参数和RNG、初始logits一致。两步仅inner-train正确性更新验证11g均finite/nonzero gradient（初始projection为零，第一步g梯度可为零，projection学到非零后再验证）、原STR/projection正常更新。activity/wrist/padding mask隔离、单g只影响本activity且双腕同倍数、实例storage/cache隔离、更新后logits/scalar/optimizer reload数值与dtype精确PASS。Optimizer step保存/重载的CPU/GPU位置按引擎规则恢复，比较值时转CPU。原引擎one-epoch/one-batch smoke PASS，不用smoke分数选择。

已有EMA实验**ordinary部分**45次精确matched reproduction逐项重查：model/optimizer/scheduler/norm/最佳epoch、逐epoch全部训练和验证指标、validation预测一致；日志duration_seconds为运行耗时不要求相同。正式与复现checkpoint都具有相同current source provenance。复用此baseline，不新增45baseline训练，不使用EMA结果选择。本轮45candidate runs完成，普通BA strict first-best及patience12与原规则一致，未多时点选模型。

## B. Performance

三seed的**指标**先在同split平均；15split为配对单位（不等于均值概率ensemble）。

{performance}

均值、中位数、split差SD、dz、Wilcoxon及BH在 `scaling_paired.csv`；BH仅exploratory，不额外进入保留闸门。不把两个CI是否跨零当成方法/类间直接差异。

## C. Stability

{stability}

原/新三seed全部prediction一致比例：{agreement.loc['baseline','all_three_seed_agreement']:.6f} / {agreement.loc['activity_scaling','all_three_seed_agreement']:.6f}；三seed两两prediction agreement：{agreement.loc['baseline','mean_pair_prediction_agreement']:.6f} / {agreement.loc['activity_scaling','mean_pair_prediction_agreement']:.6f}。同split/同seed两方法prediction agreement先seed-first后split平均为 {matched:.6f}。score Spearman完整保存；agreement不是正确率。

最佳/停止epoch中位数：原{b.best_epoch_median:.0f}/{b.stop_epoch_median:.0f}、新{c.best_epoch_median:.0f}/{c.stop_epoch_median:.0f}；cap hits原{int(b.epoch_cap_hits)}/45、新{int(c.epoch_cap_hits)}/45。训练日志为dropout-active online loss，不冒充eval-mode train performance或不存在的完整EMA/intermediate checkpoints。

错误变化（每split先平均3seed，不是独立新增人数）：

{error_table}

## D. Learned activity scales

以下只在完整正式性能分析后生成。主SD/range为15个seed-first split值；另列45run范围和三个seed各15split均值。全部run/split g表完整保存。

{scales}

{post_table}

这些相关是posthoc描述。11activity不是11独立性能样本；g随projection/后续训练共同适配，不能视为临床因果importance。TouchNose/TouchIndex g与类Recall相关不能证明类特异机制，更不能据此再调g、删活动或改成22gate/MLP。

## E. Decision

**{d['decision']}**。原冻结联合规则逐项：

{gates}

{'保留本次11scalar候选；下一步只具备独立stopping/split sensitivity资格，不自动训练更多gate或独立验证。' if d['decision']=='RETAIN' else ('拒绝本次候选；保留原Frozen WSSL-STR。停止activity/wrist/activity×wrist/MLP/attention gate与LR/参数化/g范围补救。只按授权先做独立fixed-denominator weighted CE numerical audit，不能自动训练。' if d['decision']=='REJECT' else 'INCONCLUSIVE / DO NOT RETAIN；保留原Frozen WSSL-STR。平均收益或稳定性不足以满足联合闸门，不启动任何参数化/LR/g范围/复杂gate补救，也不自动训练下一候选。')}

Phase A删除依赖异质性只支持提出对照，不能保证可学习g提高性能；性能证据优先，不因g排序符合预期而保留。

Boundary：outer data used=NO；threshold changed=NO；HarNet adapted=NO；training recipe changed=NO；additional hyperparameter search=NO。45正式新训练仅本candidate（另2step正确性/1batch smoke，非性能候选）。不使用Frequency、EMA、auxiliary、encoder适配、sampling或augmentation。不覆盖原formal资产。

所有结果 DEVELOPMENT-ONLY。15固定划分重叠、同subjects复用、反复研究选择带来乐观偏差；bootstrap仅描述repeated-development稳健性，不提供外部独立支持。seed-runs、subjects、pairs或bootstrap次数均不扩充样本量。FOE01/历史outer隔离保持；没有以其结果调设计或解释本候选。无全局recipe最优、病理因果或新外部泛化证明。代码/协议/报告/小型汇总Git提交推送并打标；数据/checkpoint/cache/逐subject预测/原始日志留server。
'''
    (HERE/'FINAL_REPORT.md').write_text(report)
    print(d['decision'])

if __name__=='__main__':main()
