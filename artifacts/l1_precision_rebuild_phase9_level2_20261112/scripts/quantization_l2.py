"""Separate float32 storage perturbation bound; never substitutes the float64 certificate."""
from common_l2 import *
def main():
 ready=load(PRIVATE/'manifests/data_ready.json');data=load(ready['manifest_path']);assert sha(ready['manifest_path'])==ready['manifest_sha256'];rows=[]
 for r in data['records']:
  raw=np.loadtxt(r['raw_path'],delimiter=',')[:,1:4].T
  with np.load(r['certificate_arrays_path'],allow_pickle=False) as z:x=z['trend'].copy()
  saved=np.load(r['output_path'])[:3].astype(float);residual=raw-x;e=np.zeros_like(raw);e[:,48:]=saved-residual[:,48:];d2=e[:,:-2]-2*e[:,1:-1]+e[:,2:]
  for j in range(3):
   l2=np.linalg.norm(e[j]);bound=np.linalg.norm(residual[j])*l2+.5*l2*l2+50*np.sum(abs(d2[j]));rows.append(dict(subject_id=r['subject_id'],activity=r['activity'],wrist=r['wrist'],channel=CHANNELS[j],max_abs_error=np.max(abs(e[j])),L2_error=l2,objective_increase_upper_bound=bound,float64_gap_tolerance=r['axis_certificates'][j]['gap_tolerance']))
 f=pd.DataFrame(rows);f.to_csv(PRIVATE/'numerical_certificates/float32_perturbation_bounds.csv',index=False);agg=f.groupby(['activity','channel']).agg(axes=('max_abs_error','size'),maximum_abs_error=('max_abs_error','max'),mean_L2_error=('L2_error','mean'),maximum_L2_error=('L2_error','max'),mean_objective_increase_bound=('objective_increase_upper_bound','mean'),maximum_objective_increase_bound=('objective_increase_upper_bound','max')).reset_index();agg.to_csv(HERE/'analysis/float32_quantization_impact_summary.csv',index=False);write(HERE/'analysis/quantization_impact_integrity.json',dict(status='PASS_SEPARATE_STORAGE_BOUND',axes=len(f),full_raw_hypothetical_extension='first48 residual stays float64; retained residual quantized to float32',bound='P(xhat)-P(x) <= ||raw-x||2||e||2+.5||e||2^2+50||D2e||1, where e=saved residual-original residual with zero first48',maximum_objective_increase_bound=f.objective_increase_upper_bound.max(),maximum_abs_error=f.max_abs_error.max(),does_not_claim_saved_float32_has_same_gap=True,no_new_solves=True));print('QUANTIZATION_BOUND_COMPLETE',flush=True)
if __name__=='__main__':main()
