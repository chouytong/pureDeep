"""Scientific plots from aggregate outputs using an already-installed isolated plotting path."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/home/zyt/envs/patch_report_deps_20261008')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
R=Path('/home/zyt/deep_final/artifacts/wssl_complementarity_phase5_20261111')
f=pd.read_csv(R/'analysis/subject_complementarity.csv');f=f[(f.aggregation=='single_seed')&(f.error_group=='all')].groupby('label')[['gross_rescue','newly_broken']].mean().loc[['PD','DD','all']]
fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3);ax.bar(x-.18,f.gross_rescue,width=.36,label='Rescued');ax.bar(x+.18,f.newly_broken,width=.36,label='Newly broken');ax.set(xticks=x,xticklabels=['PD','DD','All'],ylabel='Mean people per split/seed',title='Relative versus WSSL: benefit and harm');ax.legend();fig.tight_layout();fig.savefig(R/'analysis/rescue_harm.png',dpi=160);fig.savefig(R/'analysis/rescue_harm.svg')
f=pd.read_csv(R/'analysis/reference_coverage.csv').groupby('label')[['low_error_rate','high_error_rate']].mean().loc[['PD','DD','all']]
fig,ax=plt.subplots(figsize=(7,4));ax.bar(x-.18,f.low_error_rate,width=.36,label='Low min-distance (Q1)');ax.bar(x+.18,f.high_error_rate,width=.36,label='High min-distance (Q4)');ax.set(xticks=x,xticklabels=['PD','DD','All'],ylim=(0,1),ylabel='WSSL error rate',title='Coverage has opposite class-conditional associations');ax.legend();fig.tight_layout();fig.savefig(R/'analysis/coverage.png',dpi=160);fig.savefig(R/'analysis/coverage.svg')
f=pd.read_csv(R/'analysis/fusion_class_bias.csv');f=f[f.event=='all'].groupby('label')[['mean_W_contribution','mean_R_contribution','intercept','mean_fusion_logit']].mean().loc[['PD','DD','all']]
fig,ax=plt.subplots(figsize=(8,4));w=.18
for j,(name,label) in enumerate([('mean_W_contribution','WSSL contribution'),('mean_R_contribution','Reference contribution'),('intercept','Intercept'),('mean_fusion_logit','Final log odds')]):ax.bar(x+(j-1.5)*w,f[name],width=w,label=label)
ax.axhline(0,color='black',linewidth=.8);ax.set(xticks=x,xticklabels=['PD','DD','All'],ylabel='Existing fusion log-odds units',title='Restoring the existing decision: linear decomposition');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/'analysis/fusion_decomposition.png',dpi=160);fig.savefig(R/'analysis/fusion_decomposition.svg')
(R/'analysis/plot_environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,isolated_existing_packages='/home/zyt/envs/patch_report_deps_20261008',installed_or_upgraded=False,model_results_modified=False),indent=2)+'\n')
print('AGGREGATE_PLOTS_RENDERED')
