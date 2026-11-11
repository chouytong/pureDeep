"""Aggregate-only scientific figures; isolated pre-existing plotting dependencies."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/home/zyt/deep_final/artifacts/wssl_representation_generalization_phase6_20261111')
def read(name):
 with (ROOT/'analysis'/name).open() as f:return list(csv.DictReader(f))
def main():
 out=ROOT/'figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 f,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
 data=read('representation_layer_summary.csv');layers=['layer4','layer5','projected','final_wrist'];x=np.arange(4)
 for k,(label,color) in enumerate([('PD','#3071a9'),('DD','#b76b29')]):
  rows={r['layer']:r for r in data if r['label']==label}
  axes[0].bar(x+(k-.5)*.32,[float(rows[l]['relative_displacement_mean']) for l in layers],width=.32,label=label,color=color)
 axes[0].set_xticks(x,layers);axes[0].set_ylabel('Mean relative displacement');axes[0].set_title('Paired representation changes');axes[0].legend(frameon=False);axes[0].text(.02,.98,'Equal split means after seed averaging\nLayer4: shared frozen control',transform=axes[0].transAxes,va='top',fontsize=9)
 rows=[r for r in read('representation_subject_cluster.csv') if r['layer']=='layer5' and r['outcome']=='margin_change']
 for y,r in enumerate(rows):
  m=float(r['spearman']);lo=float(r['ci_low']);hi=float(r['ci_high']);axes[1].errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#3071a9',capsize=4)
 axes[1].axvline(0,color='#777',lw=.8);axes[1].set_yticks(range(len(rows)),[r['label'] for r in rows]);axes[1].set_xlabel('Spearman correlation, subject means (95% CI)');axes[1].set_title('Layer5 drift vs margin improvement');axes[1].text(.02,.98,'Patient-cluster bootstrap; fixed fits\nNot an external validation interval',transform=axes[1].transAxes,va='top',fontsize=9);axes[1].set_ylim(-.6,3)
 f.savefig(out/'representation_summary.png');f.savefig(out/'representation_summary.svg');plt.close(f)
 f,ax=plt.subplots(figsize=(8,4),layout='constrained');r=next(r for r in read('coverage_interaction_subject_cluster.csv') if r['metric']=='error');m=float(r['interaction']);lo=float(r['ci_low']);hi=float(r['ci_high']);ax.errorbar(m,0,xerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color='#3071a9');ax.axvline(0,color='#777',lw=.8);ax.set_yticks([0],['Error interaction']);ax.set_xlabel('(PD high−low error) − (DD high−low error)');ax.set_title('Coverage interaction: pooled patient exposures');ax.set_ylim(-.8,.8);ax.text(.02,.95,f'Estimate {m:.3f}; 95% CI [{lo:.3f}, {hi:.3f}]\nPatient clusters preserve all appearances and seeds\nConditional development uncertainty, not external confirmation',transform=ax.transAxes,va='top',fontsize=9);f.savefig(out/'coverage_interaction.png');f.savefig(out/'coverage_interaction.svg');plt.close(f)
 print('AGGREGATE_FIGURES_COMPLETE')
if __name__=='__main__':main()
