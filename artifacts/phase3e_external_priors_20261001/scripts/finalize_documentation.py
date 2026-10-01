from pathlib import Path
import json,hashlib,pandas as pd
B=Path(__file__).resolve().parents[1];project=B.parents[1];s=pd.read_csv(B/'analysis/model_summary.csv');p=pd.read_csv(B/'analysis/model_paired.csv');a=json.loads((B/'FINAL_AUDIT.json').read_text());assert a['status']=='PASS' and a['new_completed_runs']==180
assert {'harnet5','biopm_only','biopm','biopm_random'}.issubset(set(s.variant));ss=s.set_index('variant');g=json.loads((B/'analysis/model_gates.json').read_text());assert not g['harnet5']['retain'] and not g['biopm']['retain']
control=p[(p.candidate=='biopm')&(p.reference=='biopm_random')].set_index('metric')
def desc(c,r,m):
 z=p[(p.candidate==c)&(p.reference==r)&(p.metric==m)].iloc[0];return f"{m}: delta {z.mean_delta:+.4f}, {int(z.positive_splits)}/15 improvements, CI [{z.ci95_low:+.4f},{z.ci95_high:+.4f}], dz {z.cohen_dz:+.3f}, BH q {z.q_primary_bh:.4f}"
controlpass=all(control.loc[m,'mean_delta']>0 and control.loc[m,'ci95_low']>0 and control.loc[m,'positive_splits']>=10 and control.loc[m,'q_primary_bh']<=.05 for m in ['ba','auroc'])
parts=['\n## Final answers and phase decision\n',
'1. Official HarNet5 does not outperform10s prior: '+desc('harnet5','frozen','ba')+'; '+desc('harnet5','frozen','auroc')+'. Its seed variance worsens. Stop tested temporal-scale route.\n',
'2. Restricted BioPM neural acc-only standalone feasibility BA0.6438/AUROC0.6659; residual does not establish stable useful disease transfer beyond STR: '+desc('biopm','str','ba')+'; '+desc('biopm','str','auroc')+'. It is materially belowHarNet10: '+desc('biopm','frozen','ba')+'; '+desc('biopm','frozen','auroc')+'. Do not extrapolate to full BioPM mixed-gravity pipeline.\n',
'3. Random capacity control completed45 because any positive primary mean triggered it, despite unstableAUROC+0.0005 vsSTR. Pretrained-random: '+desc('biopm','biopm_random','ba')+'; '+desc('biopm','biopm_random','auroc')+'. Joint stable superiority = '+str(controlpass)+'. A significant improvement in only one primary metric does not establish the joint criterion.\n',
'4. E4 disease complementarity not run: BioPM fails joint value prerequisite. No established stableHarNet/BioPM complementarity; absence of measurement is not proof of absent information.\n',
'5. E5 dual-prior not run: BioPM clearly belowHarNet and noE4 evidence. No dual-prior performance claim.\n',
'6. Retained best single remains STR-01 + pretrained-frozen HarNet10 final1024 wrist residual,unchanged recipe/0.5 rule. Accuracy0.7456,BA0.7196,AUROC0.7596,MacroF10.7066,PDRecall0.7823,DDRecall0.6569. NoPhase3E single candidate retained.\n',
'7. Among registered inference references, highestAUROC is equal6model ensemble:0.7870,BA0.7150,Accuracy0.7541,MacroF10.7099,PDRecall0.8090,DDRecall0.6210. Frozen3seed ensembleAUROC0.7816/BA0.7200;higherAUROC is accompanied by lowerBA/DDRecall point estimates. Keep as inference reference only,not a single-model replacement.\n',
'8. Continue considering external-prior research only through evidence-backed,one-at-a-time official encoder protocols; this restrictedBioPM failure does not stop all externalSSL. This phase supports no specific new mechanism or dual-prior training. Stop current5s/BioPMacc integration variants;retain existing model and plan genuinely independent validation. No HarNet adapter/gate/domain/layer or STR structure/loss/sampling/augmentation reopening. No new encoder/model is automatically trained.\n']
report=B/'PHASE3E_REPORT.md';report.write_text(report.read_text()+''.join(parts))
readme=project/'README.md';assert not (B/'DOCUMENTATION_COMPLETE.json').exists();header='## Phase-3E External Wearable SSL Prior Diversification (2026-10-01)'
(B/'readme_before_phase3e.sha256').write_text(hashlib.sha256(readme.read_bytes()).hexdigest()+'\n')
cols=['variant','accuracy','ba','auroc','macro_f1','pd_recall','dd_recall'];table='| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'
for name in ['str','frozen','harnet5','biopm_only','biopm','biopm_random']:
 r=ss.loc[name];table+='| '+name+' | '+' | '.join(f'{r[c]:.4f}' for c in cols[1:])+' |\n'
section='\n\n'+header+'\n\nComplete/audited DEVELOPMENT-ONLY, PURE-DEEP;180 new runs (four models x15split x3seed),no outer evaluation. [Frozen protocol](artifacts/'+B.name+'/PHASE3E_PROTOCOL.md),[official asset audit](artifacts/'+B.name+'/OFFICIAL_ASSET_AUDIT.md),[full report](artifacts/'+B.name+'/PHASE3E_REPORT.md),[final audit](artifacts/'+B.name+'/FINAL_AUDIT.json),[statistics](artifacts/'+B.name+'/analysis/). SameSTRrecipe/fixedsplits/seeds/order/normalization/decision. Three seeds first averaged inside15splits;subjects/windows/pairs/seeds not independent units.\n\n'+table+'\n'+''.join(parts[1:])+'\n\n**Implementation/evidence limits.** HarNet5 official512D/4.23M differs fromHarNet10 1024D/10.46M,so comparison does not isolate duration. BioPM default1023 mixes384 neural token dimensions with639 unencodedgravity;onlyofficial384 neural acc branch used for pure-deep,no gravityCNN/rawgravity. Fixedofficial30Hz/0.5-12Hz/token rules,192cap,10920contexts,0empty;8.14%contexts hitcap. This result concerns restrictedacc transfer,not fullBioPMHAR. Zero-residual and loadedWSSL interface logits consistency max0;smoke/completecheckpoint/ID/label/trainnormalization/noouterloader/sourceSHA auditsPASS. h5py isolatedvendor only;productioncode/oldformaloutputsunchanged. Repeated overlappingdevelopment and sequentialselection limit confirmation. E3trigger clarified before completeBioPMstats to include anypositiveBA/AUCmean,without looseningretention. E4/E5notrun are unmeasured,not failures. **Development-only conclusions;FOE01unchanged/notused.**\n'
with readme.open('a') as handle:handle.write(section)
handoff=project/'CONTEXT_HANDOFF.md';(B/'handoff_before_phase3e.sha256').write_text(hashlib.sha256(handoff.read_bytes()).hexdigest()+'\n')
alert='> **最新状态（2026-10-01）：Phase-3E 已完成并归档（180 new development runs）。HarNet5 与纯神经 BioPM acc-only 均未替代 pretrained-frozen HarNet10 WSSL-STR；E3 random control 已完成，E4/E5 未满足条件而未运行。最佳 retained single-model 仍为 WSSL-STR（BA0.7196/AUROC0.7596）。最高固定 inference AUROC 为六模型等权0.7870，但BA/DDRecall更低。以文末 Phase-3E update、最新README/正式报告为准；旧FOE01不得用于本阶段选择或独立验证。**'
tmp=B/'handoff_updated.tmp';changed=False
with handoff.open() as src,tmp.open('w') as dst:
 for line in src:
  if not changed and line.startswith('> **最新状态'):
   dst.write(alert+'\n');changed=True
  else:dst.write(line)
 dst.write('\n\n## Phase-3E update (2026-10-01)\n\n'+''.join(parts[1:])+'\nFull report: artifacts/'+B.name+'/PHASE3E_REPORT.md. Pure-neuralBioPMscope,officialcapacitydifferences,repeated-developmentstatistics and outerboundary are detailed there. Historical next-step lists are superseded;do not automatically train anotherencoder or reopen rejectedHarNet/STR routes.\n')
assert changed
import shutil
shutil.copyfile(tmp,handoff)
(B/'documentation_after_sha256.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [report,readme,handoff]},indent=2)+'\n');(B/'DOCUMENTATION_COMPLETE.json').write_text(json.dumps({'status':'complete','historical_outer_performance_not_parsed':'README append-only; handoff historical lines copied without metric inspection'})+'\n');print('README/handoff/report updated; retained frozen WSSL; joint pretrain-random superiority',controlpass)
