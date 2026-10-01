from pathlib import Path
import pandas as pd,json,hashlib,shutil
B=Path(__file__).resolve().parents[1];P=B.parents[1];ss=pd.read_csv(B/'analysis/model_summary.csv').set_index('variant');st=pd.read_csv(B/'analysis/model_paired.csv')
answers='''1. **5s prior 没有优于 10s prior。** HarNet5 BA 0.6967、AUROC 0.7238；相对 HarNet10 分别 −0.0229 / −0.0358，只有 4/15 / 1/15 splits 改善，seed 波动增大。停止本次 temporal-scale 路线。官方 family 的维度和参数量也不同，不能将差异仅归因于时间尺度。
2. **本轮 Bio-PM acc-only 未建立有用的增量迁移收益。** BioPM-only BA 0.6438 / AUROC 0.6659；STR+BioPM 为 0.6924 / 0.7181。相对原 STR 的 BA −0.0048，AUROC +0.0005（95% CI [−0.0064,+0.0072]，BH q=1.0）；相对 HarNet10 的两项指标均明显较低。本轮只使用官方 384D 神经 acc-token pooling，不包含默认输出中的未编码 gravity；不能据此否定完整 Bio-PM HAR pipeline。
3. **未证实 Bio-PM pretrained 优于 random-frozen control。** BA 差 −0.0008，7/15 改善，CI [−0.0062,+0.0053]，dz −0.068；AUROC 差 −0.0024，8/15 改善，CI [−0.0109,+0.0062]，dz −0.135；两项 BH q=1.0。DD Recall 的正向点估计伴随 PD Recall 下降，相关 CI 均跨零，不能归因于稳定的预训练收益。
4. **HarNet/BioPM disease complementarity 尚未检验。** E4 的 Bio-PM 价值门槛失败，因此按条件未执行；未测量不等于证明不存在互补信息。
5. **Dual-prior 未训练。** Bio-PM 未稳定接近 HarNet10，也没有满足 E4 证据要求；不存在可报告的 dual-prior 增益。
6. **最佳 retained single-model 仍是 frozen WSSL-STR。** STR-01 + pretrained-frozen HarNet10 final1024 wrist residual，原 recipe 和 0.5 decision rule。Accuracy 0.7456、BA 0.7196、AUROC 0.7596、Macro-F1 0.7066、PD Recall 0.7823、DD Recall 0.6569。本轮没有保留新 single-model candidate。
7. **最高预注册 inference AUROC 为六模型等权 ensemble：0.7870。** Accuracy 0.7541、BA 0.7150、Macro-F1 0.7099、PD Recall 0.8090、DD Recall 0.6210。相对 frozen 三 seed ensemble（AUROC 0.7816 / BA 0.7200），AUROC +0.0054，14/15 改善，CI [+0.0028,+0.0074]，dz 1.127，E0 BH q=0.0427；BA 与 DD Recall 点估计更低。只作为 inference reference，不替代 single-model。
8. **当前不支持新机制或 dual-prior training。** 保留现有模型，停止本次 5s / Bio-PM acc-only integration 变体。External-prior 路线整体未被否定，后续只考虑有官方资产和明确依据的单一 prior 验证，并规划真正独立的验证数据。不得重开已停止的 HarNet adapter/gate/domain/layer 或 STR architecture/loss/sampling/augmentation 搜索；不会自动启动另一个 encoder。
'''
report=B/'PHASE3E_REPORT.md';text=report.read_text().split('## Final answers and phase decision')[0];text=text.replace(';static plots.','.')
report.write_text(text+'## Final answers and phase decision\n\n'+answers+'\nOptional plot rendering was unavailable because the plotting runtime lacks matplotlib; no extra plotting dependency was installed. All requested numerical tables, CIs and audits are complete.\n')
cols=['accuracy','ba','auroc','macro_f1','pd_recall','dd_recall'];table='| Configuration | Accuracy | BA | AUROC | Macro-F1 | PD Recall | DD Recall |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name in ['str','frozen','harnet5','biopm_only','biopm','biopm_random']:
 table+='| '+name+' | '+' | '.join(f'{ss.loc[name,c]:.4f}' for c in cols)+' |\n'
header='## Phase-3E External Wearable SSL Prior Diversification (2026-10-01)'
section=header+'\n\n**已完成并审计：DEVELOPMENT-ONLY / PURE-DEEP。** 新增 180 runs（HarNet5、BioPM-only、BioPM residual、random BioPM residual，各 15 splits × 3 seeds）。[固定 protocol](artifacts/'+B.name+'/PHASE3E_PROTOCOL.md)、[官方资产核验](artifacts/'+B.name+'/OFFICIAL_ASSET_AUDIT.md)、[完整报告](artifacts/'+B.name+'/PHASE3E_REPORT.md)、[最终审计](artifacts/'+B.name+'/FINAL_AUDIT.json)、[统计表](artifacts/'+B.name+'/analysis/)。\n\n'+table+'\n'+answers+'\n**统计与实现边界。** 三个 seed 先在 split 内聚合，15 fixed splits 为主要比较单位；subjects、windows、activities、pairs、seed-runs 不作为独立样本。报告 paired difference、split improvement count、10000 次 bootstrap CI、dz 和 BH-FDR。重复且重叠的 development splits 与连续阶段选择限制确认性；没有新的独立 outer validation。\n\nSTR architecture、双腕/activity order、structured path、balanced CE、原 training recipe、train-only normalization 与 0.5 rule 保持不变。HarNet5 官方为 512D / 4.23M，HarNet10 为 1024D / 10.46M，不能隔离 duration 因果效应。Bio-PM 只保留纯神经 384D acc 分支，使用官方固定 30Hz、filter/tokenization 与 192-token cap；10920 contexts、0 empty，8.14% 触及 cap，未据此改 preprocessing。Zero-residual 与 loaded-WSSL 接口 logits consistency 最大差 0；smoke、全部 checkpoint、subject/label 对齐、train-only normalization、no-outer-loader 和 source SHA 审计均 PASS。h5py 仅装入本阶段 vendor/deps，正式代码和旧结果未修改。E3 的 any-positive-primary control 澄清在完整 BioPM residual 统计前记录，未放宽保留标准。E4/E5 是未执行，不是测量后的失败。\n\n**Development-only conclusions：旧 FOE-01 继续冻结，未用于本阶段模型选择、调参或独立验证。**\n'
readme=P/'README.md';tmp=B/'readme_polished.tmp';found=False
with readme.open() as src,tmp.open('w') as dst:
 for line in src:
  if line.rstrip()==header:found=True;break
  dst.write(line)
 assert found
 dst.write(section)
shutil.copyfile(tmp,readme)
handoff=P/'CONTEXT_HANDOFF.md';tmp=B/'handoff_polished.tmp';found=False
with handoff.open() as src,tmp.open('w') as dst:
 for line in src:
  if line.rstrip()=='## Phase-3E update (2026-10-01)':found=True;break
  dst.write(line)
 assert found
 dst.write('## Phase-3E update (2026-10-01)\n\n'+answers+'\n完整记录：artifacts/'+B.name+'/PHASE3E_REPORT.md。以上 supersede 历史 next-step 列表；旧 FOE-01 不得用于本阶段选择或独立验证。\n')
shutil.copyfile(tmp,handoff)
(B/'documentation_after_sha256.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [report,readme,handoff]},indent=2)+'\n')
print('polished only new Phase3E sections; historical contents copied without outer metric inspection')
