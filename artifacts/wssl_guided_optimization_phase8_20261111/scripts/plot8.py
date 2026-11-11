"""Aggregate paired effects only; no per-patient data in exported figure."""
from common8 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 a=pd.read_csv(HERE/'analysis/final_paired_statistics.csv');b=pd.read_csv(HERE/'analysis/subject_cluster_bootstrap.csv')
 names=['probability_ensemble','logit_ensemble','logistic_regression','linear_svm','rbf_svm','random_forest','activity_interaction','subject_supcon'];labels=['Probability ensemble (3 models)','Logit ensemble (sensitivity)','Logistic regression','Linear SVM','RBF SVM','Random forest','Activity interaction','PD-positive SupCon']
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,3,figsize=(13.5,5.4),sharey=True)
 for ax,metric,title in zip(axes,['ba','auroc','dd_recall'],['Balanced accuracy','AUROC','DD recall']):
  ax.axvline(0,color='#56616e',lw=1)
  if metric=='dd_recall':ax.axvline(-.01,color='#b74a43',lw=1,ls='--',label='Recall drop limit')
  for j,name in enumerate(names):
   for f,offset,color,marker in [(a,-.13,'#246cb5','o'),(b,.13,'#d28a16','s')]:
    r=f[(f.condition==name)&(f.metric==metric)].iloc[0];ax.plot([r.ci_low,r.ci_high],[j+offset]*2,color=color,lw=1.5);ax.scatter(r.mean_delta,j+offset,c=color,marker=marker,s=24,zorder=3)
  ax.set_title(title);ax.set_xlabel('Mean paired difference');ax.grid(axis='x',alpha=.15)
 axes[0].set_yticks(range(len(names)),labels);axes[0].invert_yaxis();fig.suptitle('Phase 8 | all tested candidates failed the joint retention gate',fontsize=14,y=.98)
 fig.text(.36,.055,'Blue circles: split-bootstrap 95% CI    Gold squares: shared subject-cluster 95% CI',fontsize=10);fig.text(.36,.022,'Repeated PADS development evidence; fixed fitted models; not independent cohorts.',fontsize=9,color='#555555');fig.tight_layout(rect=[0,.10,1,.94]);(HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/paired_tradeoff.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
