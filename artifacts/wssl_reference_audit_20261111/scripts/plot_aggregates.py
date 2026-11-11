"""Render only aggregate CSVs using previously installed plotting packages."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/home/zyt/envs/patch_report_deps_20261008')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
R=Path('/home/zyt/deep_final/artifacts/wssl_reference_audit_20261111')
f=pd.read_csv(R/'analysis/A2_DD_reliability_bins.csv')
fig,ax=plt.subplots(figsize=(6,5))
ax.plot([0,1],[0,1],color='gray',linestyle='--',label='Ideal')
for name,g in f.groupby('condition'):
    rows=[]
    for b,q in g.groupby('bin'):
        q=q[q.n>0]
        if len(q):rows.append((np.average(q.mean_probability,weights=q.n),np.average(q.DD_fraction,weights=q.n)))
    xy=np.array(rows);ax.plot(xy[:,0],xy[:,1],marker='o',label=name.replace('_',' '))
ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted P(DD)',ylabel='Observed DD fraction',title='Development reliability (pooled appearances)')
ax.text(0,-.23,'Descriptive only: repeated appearances and seeds are dependent.',transform=ax.transAxes,fontsize=9)
ax.legend();fig.tight_layout();fig.savefig(R/'analysis/reliability.svg');fig.savefig(R/'analysis/reliability.png',dpi=160)
f=pd.read_csv(R/'analysis/ALL_stability_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(9,4))
names=['baseline','relative','fusion']
for ax,metric in zip(axes,['ba','auroc']):
    q=f[f.metric==metric].set_index('condition').loc[names]
    ax.bar(names,q['mean'],yerr=q.split_sd,capsize=4,color=['#426a8c','#b48c42','#a85757'])
    ax.set(ylim=(0,1),title=metric.upper(),ylabel='Mean ± split SD (not a confidence interval)')
fig.tight_layout();fig.savefig(R/'analysis/reference_comparison.svg');fig.savefig(R/'analysis/reference_comparison.png',dpi=160)
(R/'analysis/plot_environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,isolated_plot_only_package_path='/home/zyt/envs/patch_report_deps_20261008',installed_or_upgraded=False,scientific_model_outputs_modified=False),indent=2)+'\n')
print('AGGREGATE_FIGURES_RENDERED')
