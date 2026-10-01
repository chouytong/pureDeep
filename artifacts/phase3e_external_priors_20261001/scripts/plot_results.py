import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd,numpy as np
B=Path(__file__).resolve().parents[1];s=pd.read_csv(B/'analysis/model_summary.csv');p=pd.read_csv(B/'analysis/model_paired.csv');names=[n for n in ['str','frozen','harnet5','biopm_only','biopm','biopm_random','dual'] if n in set(s.variant)];s=s.set_index('variant').loc[names];fig,axes=plt.subplots(1,3,figsize=(14,4));labels={'str':'STR-01','frozen':'HarNet10 WSSL','harnet5':'HarNet5 WSSL','biopm_only':'BioPM-only (acc)','biopm':'BioPM WSSL (acc)','biopm_random':'Random BioPM WSSL','dual':'Dual-prior WSSL'}
for ax,m in zip(axes,['ba','auroc','dd_recall']):
 x=np.arange(len(names));ax.bar(x,s[m],color=['#606c80' if n in ('str','frozen') else '#318b9e' for n in names]);ax.set_xticks(x,[labels[n] for n in names],rotation=45,ha='right');ax.set_ylim(.4,.85);ax.set_title(m.upper().replace('_',' '));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Phase-3E development: seed-first means across 15 fixed splits');fig.tight_layout();fig.savefig(B/'analysis/development_metrics.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4));q=p[(p.reference=='frozen')&p.metric.isin(['ba','auroc'])]
for ax,m in zip(axes,['ba','auroc']):
 g=q[q.metric==m].copy();xx=np.arange(len(g));ax.errorbar(g.mean_delta,xx,xerr=np.array([g.mean_delta-g.ci95_low,g.ci95_high-g.mean_delta]),fmt='o',capsize=4,color='#276b7a');ax.axvline(0,color='gray',ls='--');ax.set_yticks(xx,[labels[n] for n in g.candidate]);ax.set_title(m.upper()+' difference vs frozen HarNet10');ax.grid(axis='x',alpha=.2)
fig.suptitle('Paired 15-split bootstrap 95% CI (overlapping development splits)');fig.tight_layout();fig.savefig(B/'analysis/paired_primary.png',dpi=180)
