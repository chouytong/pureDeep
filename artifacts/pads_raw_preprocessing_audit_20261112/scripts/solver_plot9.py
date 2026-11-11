from common9 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 w=load(HERE/'analysis/solver_objective_witness_integrity.json');f=pd.read_csv(HERE/'analysis/solver_objective_witness_summary.csv');g=f.groupby('activity')[['flagged_axis_records','axis_record_count']].sum().reindex(ACTS);fig,axes=plt.subplots(1,2,figsize=(13,5));axes[0].bar(['Feasible affine /\nnumerical KKT reference','Legacy ADMM\nconverged after1 iteration'],[w['exact_affine_optimum_objective'],w['exact_case_existing_solver_objective']],color=['#377cad','#ba5449']);axes[0].set_yscale('log');axes[0].set(ylabel='Original L1 objective (log scale)',title='Synthetic0.001g/0.5Hz: convergence flag is insufficient');axes[1].barh(ACTS,100*g.flagged_axis_records/g.axis_record_count,color='#a95a43');axes[1].invert_yaxis();axes[1].set(xlim=(0,100),xlabel='Acc axis-records with conservative objective witness (%)',title='Existing real outputs; no real L1 reprocessing');fig.suptitle('Phase9 | numerical objective evidence, not a sensor-noise or performance diagnosis',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(HERE/'figures/solver_objective_evidence.png',dpi=140,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
