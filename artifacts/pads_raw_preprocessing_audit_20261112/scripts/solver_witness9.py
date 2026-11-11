"""Conservative convex-objective witnesses from existing outputs; no real L1 refit."""
from common9 import *

def feasible_affine(y):
 n=len(y);index=np.arange(n,dtype=float);center=index-index.mean();slope=(center[:,None]*y).sum(0)/np.dot(center,center);line=y.mean(0)+center[:,None]*slope;return line
def objective(y,trend):return .5*np.sum((y-trend)**2,axis=0)+50*np.sum(abs(np.diff(trend,n=2,axis=0)),axis=0)
def main():
 freeze('SOLVER_OBJECTIVE_WITNESS',[Path(__file__),HERE/'scripts/common9.py'],before_real_objective_witnesses=True,no_real_L1_reprocessing=True,stored_float32_uncertainty='one float32 ULP conservative bound',witness='full-record feasible affine objective versus lower bound on existing post48 primal objective',threshold='existing lower bound exceeds feasible upper bound by max(1e-9,1e-4*upper)',no_labels_used_for_rule=True,new_model_or_normalization_fits=0)
 protected=load(HERE/'restricted_manifests/protected_assets.json')['sha256'];creation={}
 for line in (PROC/'quality_records.jsonl').read_text().splitlines():
  r=json.loads(line);creation[(str(r['subject_id']),r['activity'],r['wrist'])]=r
 rows=[];beg=time.perf_counter()
 for k,r in enumerate(records()):
  key=(r['subject_id'],r['activity'],r['wrist']);old=creation[key];assert protected[r['raw_path']]==old['source_sha256'] and protected[r['processed_path']]==old['processed_sha256']
  t,full=read_raw(r);y=full[:,:3];stored=np.load(r['processed_path'],allow_pickle=False)[:3].T;assert stored.dtype==np.float32
  eps=abs(np.spacing(stored)).astype(float);residual=stored.astype(float);trend_tail=y[48:]-residual
  primal_lb=.5*np.sum(np.maximum(abs(residual)-eps,0)**2,axis=0)+50*np.sum(np.maximum(abs(np.diff(trend_tail,n=2,axis=0))-(eps[:-2]+2*eps[1:-1]+eps[2:]),0),axis=0)
  upper=objective(y,feasible_affine(y));threshold=np.maximum(1e-9,1e-4*upper);flag=primal_lb>upper+threshold
  for j in range(3):rows.append(dict(subject_id=r['subject_id'],activity=r['activity'],wrist=r['wrist'],channel=CHANNELS[j],existing_post48_primal_lower_bound=primal_lb[j],full_affine_feasible_upper_bound=upper[j],certified_objective_excess_lower_bound=max(0,primal_lb[j]-upper[j]),objective_excess_flag=bool(flag[j]),creation_source_and_output_hashes_exact=True,one_ULP_quantization_uncertainty_accounted=True,original_prefix_trend_not_required=True,reference_is_diagnostic_not_candidate_preprocessing=True))
  if (k+1)%2200==0:print(json.dumps(dict(stage='existing_output_objective_witness',records=k+1)),flush=True)
 frame=pd.DataFrame(rows);csv(frame,'solver_objective_witness_private.csv',True);public=[]
 for (activity,channel),g in frame.groupby(['activity','channel']):public.append(dict(activity=activity,channel=channel,axis_record_count=len(g),affected_subject_count=g[g.objective_excess_flag].subject_id.nunique(),flagged_axis_records=int(g.objective_excess_flag.sum()),fraction_flagged=g.objective_excess_flag.mean(),median_certified_excess=g.certified_objective_excess_lower_bound.median(),maximum_certified_excess=g.certified_objective_excess_lower_bound.max(),interpretation='objective suboptimality witness, not noise diagnosis or performance estimate'))
 csv(pd.DataFrame(public),'solver_objective_witness_summary.csv')
 # Existing official settings, same input alone versus beside fixed other series.
 sys.path.insert(0,str(MAIN));from src.datasets.preprocessing import l1_trend_filter_batch
 t=np.arange(2048)/100;target=1+.02*t+.01*np.sin(2*np.pi*t);companions=np.stack([1+.02*t+a*np.sin(2*np.pi*f*t) for a in [.01,.1,1.] for f in [.25,.5,1,2,3,4,6,8,10,12,15,20,30,40]])
 alone,sa=l1_trend_filter_batch(target,50,rho=40,max_iterations=5000,absolute_tolerance=1e-4,relative_tolerance=1e-4);batch,sb=l1_trend_filter_batch(np.vstack([target,companions]),50,rho=40,max_iterations=5000,absolute_tolerance=1e-4,relative_tolerance=1e-4)
 lower_reference=float(objective(target[:,None],feasible_affine(target[:,None]))[0]);aobj=float(objective(target[:,None],alone.T)[0]);bobj=float(objective(target[:,None],batch[0,:,None])[0]);difference=float(abs(alone[0]-batch[0]).max())
 # An exact affine optimum certificate where the dual box is feasible.
 exact_target=1+.02*t+.001*np.sin(2*np.pi*.5*t);aff=feasible_affine(exact_target[:,None])[:,0];res=exact_target-aff;dual=np.cumsum(np.cumsum(res))[:-2];dt=np.zeros(len(t));dt[:-2]+=dual;dt[1:-1]-=2*dual;dt[2:]+=dual;exact_feasible=abs(dual).max()<=50 and abs(dt-res).max()<1e-8;assert exact_feasible
 sol,se=l1_trend_filter_batch(exact_target,50,rho=40,max_iterations=5000,absolute_tolerance=1e-4,relative_tolerance=1e-4);exactobj=float(objective(exact_target[:,None],aff[:,None])[0]);reportedobj=float(objective(exact_target[:,None],sol.T)[0]);assert reportedobj>exactobj*10 and difference>1e-4
 solverlog=load(PROC/'solver_audit.json');iterations=[x['iterations'] for x in solverlog]
 findings=dict(status='CONFIRMED_NUMERICAL_OBJECTIVE_AND_BATCH_DEPENDENCE',existing_acc_axis_records=25740,flagged_axis_records=int(frame.objective_excess_flag.sum()),affected_wrist_records=len(frame[frame.objective_excess_flag][['subject_id','activity','wrist']].drop_duplicates()),affected_subject_count=frame[frame.objective_excess_flag].subject_id.nunique(),real_input_output_creation_hashes_exact=8580,real_L1_reprocessing_calls=0,all_bounds_allow_float32_rounding=True,unknown_original_prefix_cannot_invalidate_lower_bound=True,synthetic_alone_iterations=sa.iterations,synthetic_batched_iterations=sb.iterations,synthetic_alone_converged=sa.converged,synthetic_batched_converged=sb.converged,synthetic_same_input_max_trend_change=difference,synthetic_target_affine_feasible_objective=lower_reference,synthetic_target_alone_objective=aobj,synthetic_target_batched_objective=bobj,exact_certificate_dual_max=float(abs(dual).max()),exact_certificate_stationarity_max=float(abs(dt-res).max()),exact_affine_optimum_objective=exactobj,exact_case_existing_solver_objective=reportedobj,exact_case_reported_converged=se.converged,exact_case_iterations=se.iterations,original_solver_batch_count=len(iterations),original_minimum_iterations=min(iterations),original_maximum_iterations=max(iterations),no_label_or_subject_split_used_to_define_rule=True,new_normalization_or_classifier_fit_calls=0,Level2_started=False,seconds=time.perf_counter()-beg)
 write(HERE/'analysis/solver_objective_witness_integrity.json',findings);print(json.dumps(clean(findings)))
if __name__=='__main__':main()
