"""Append completed frozen results and boundaries to project documentation."""
import json,pandas as pd
from common import *
from run_condition import study_guard
from render_report import table

def main():
    a=HERE/'analysis';d=json.loads((a/'decision.json').read_text());state=json.loads((a/'execution_state.json').read_text())
    require(state['status']=='complete' and state['completed_runs']==90 and (HERE/'PATCH_STR_REPORT.md').is_file(),'Incomplete deliverable')
    require(not d['p3_allowed'],'A retained P2 needs its separately authorized conditional stage before this stop archive')
    guard(json.loads((a/'f0_lock.json').read_text()));study_guard(json.loads((a/'study_lock.json').read_text()))
    means=pd.read_csv(a/'summary.csv').pivot(index='condition',columns='metric',values='mean').reset_index()
    means=means[['condition','accuracy','ba','auroc','macro_f1','pd_recall','dd_recall']]
    paired=pd.read_csv(a/'paired_comparisons.csv');primary=paired[paired.metric=='ba'][['comparison','mean_delta','improve_count','ci_low','ci_high','wilcoxon_p','bh_q_primary_ba3']]
    heading='## 2026-10-08 Patch-based Temporal STR：最终开发结论'
    section=f'''\n\n{heading}\n\n正式结论 **{d['decision']}**。完整报告：[PATCH_STR_REPORT.md](artifacts/patch_str_20261008/PATCH_STR_REPORT.md)。P0严格matched45正式结果复现PASS，P1/P2各45 scratch训练完成；三seed先取每split metric均值，以15共享subjects的development units为配对单位（非独立外部推断）。\n\n{table(means)}\n\n{table(primary)}\n\n原STR训练recipe、train-only normalization、mask、activity/wrist顺序、threshold、early stopping均保持；正式源码/STR assets hash guard及90run审计PASS；无outer-test、WSSL/HarNet/Frequency/FFT、loss/sampling/augmentation搜索。源码、协议、统计与retention已在preformal tag `patch-str-preformal-freeze-20261008`冻结，最终代码/报告tag `patch-str-final-result-20261008`。仅发布代码、配置、聚合指标/报告/hash，原始数据、privatecheckpoint、individualpredictions/caches排除。\n\nP2同时增加序列卷积和attention pooling及321参数，不能单独隔离order/capacity原因。bootstrap与Wilcoxon/BH分别保留，不以某一个结果替代另一种证据；完整18探索比较、六指标/seed/split波动、best-last轨迹、agreement、纠正/伤害见正式报告。online train含Dropout，不解释成eval泛化gap；不伪造中间checkpoint/EMA。\n\n本次未满足P3入口，STOP temporal adjacency pretext及WSSL+Patch；不调整patch/stride/hidden/pooling或改Transformer/InstanceNorm/triplet救援。只说明当前固定200/100实现未建立所要求的稳定增量，不否定所有patch方案。当前retained best继续为原Frozen WSSL-STR（本轮未训练或变更）。既有Frequency/EMA/普通recipe等停止边界继续有效；development-only，不使用原FOE-01 outer做选择。\n'''
    for name in ['README.md','CONTEXT_HANDOFF.md']:
        p=ROOT/name;require(p.is_file(),f'Missing existing project document: {name}')
        require(heading not in p.read_text(),'Do not duplicate or overwrite final result')
        p.write_text(p.read_text()+section)
    receipts=[]
    for stage in ['baseline-audit','extraction-pass','p1-tests-pass','p2-tests-pass','preformal-freeze']:
        receipts.append(json.loads(Path(f'/home/zyt/pureDeep-patch-str-{stage}-receipt-20261008.json').read_text()))
    write_json(a/'git_milestones_before_final.json',dict(status='PASS',milestones=receipts,final_tag='patch-str-final-result-20261008'))
    write_json(a/'final_integrity.json',dict(status='PASS',candidate_runs=90,reused_baseline_runs=45,primary_units=15,original_assets_unchanged=True,frozen_study_guard=True,README_and_handoff_appended=True,outer_access=False,retained_best_changed=False,decision=d['decision'],p3_started=False))
    print('README / HANDOFF APPENDED',d['decision'],flush=True)

if __name__=='__main__':main()
