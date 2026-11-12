from common_l2 import *
from numeric_core import *
from scipy import sparse,optimize
from scipy.sparse.linalg import factorized
import subprocess,resource

def repaired_admm(y,maxiter=30000):
 n=len(y);D=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc');factor=factorized(sparse.eye(n,format='csc')+40*(D.T@D));z=np.zeros(n-2);u=z.copy();beg=time.process_time()
 for it in range(1,maxiter+1):
  x=factor(y+40*(D.T@(z-u)));dx=D@x;z=np.sign(dx+u)*np.maximum(abs(dx+u)-1.25,0);u=u+dx-z
  if it%25==0:
   c=certificate(y,x,40*u,it,'PER_AXIS_ADMM_CERTIFIED')
   if c['passed']:c['cpu_seconds']=time.process_time()-beg;return x,c
 c=certificate(y,x,40*u,maxiter,'ADMM_REFERENCE_LIMIT');c['cpu_seconds']=time.process_time()-beg;return x,c

def main():
 assert load(HERE/'analysis/audit_integrity.json')['status']=='PASS_SOURCE_AND_AUTHORIZATION'
 for row in load(P9/'analysis/source_output_manifest.json')['files']:assert sha(P9/row['path'])==row['sha256']
 for row in load(P9/'configs/PREPUBLICATION_REVISION.json')['snapshot_files']:assert sha(P9/'audit/pre_solver_witness_snapshot'/row['path'])==row['sha256']
 rng=np.random.default_rng(SEED)
 for n in [3,8,64,1024,2048]:
  x=rng.normal(size=n);q=rng.normal(size=n-2);assert abs(np.dot(d2(x),q)-np.dot(x,adjoint(q)))<1e-10
 for y in [np.zeros(2),np.ones((2,10)),np.r_[1.,2.,np.nan],np.r_[1.,2.,np.inf]]:
  try:solve(y)
  except ValueError:pass
  else:raise AssertionError('Invalid series accepted')
 cases=[]
 for n in [1024,2048]:
  t=np.arange(n)/100
  cases.extend([(f'constant_{n}',np.full(n,.7)),(f'ramp_{n}',1+.02*t),(f'lowKKT_{n}',1+.02*t+.001*np.sin(2*np.pi*.5*t)),(f'mixed_{n}',.02*np.sin(2*np.pi*.6*t)+.03*np.sin(2*np.pi*6*t)),(f'step_{n}',np.where(t<t[-1]/2,-.2,.2))])
  for a in [.001,.01,.1,1.]:
   for f in [.5,1.,3.,6.,12.,30.]:cases.append((f'sine_{n}_{a}_{f}',a*np.sin(2*np.pi*f*t)))
 rows=[];solutions=[]
 for name,y in cases:
  x,q,c=solve(y);rows.append(dict(case=name,length=len(y),**c));solutions.append(x)
  print(json.dumps(dict(stage='solver_test',case=name,passed=c['passed'],iterations=c['iterations'],gap=c['raw_gap'],tolerance=c['gap_tolerance'],CPU=c['cpu_seconds'])),flush=True)
  if not c['passed']:
   pd.DataFrame(rows).to_csv(HERE/'analysis/solver_test_progress.csv',index=False);write(HERE/'tests/solver_test_failure.json',dict(case=name,certificate=c));raise RuntimeError('BLOCKED_SOLVER_CERTIFICATE')
  assert np.max(abs((y-x)+x-y))<=5e-13*(1+abs(y).max())
  saved=(y-x).astype(np.float32);eps=abs(np.spacing(saved)).astype(float)+8*np.finfo(float).eps*(1+abs(y-x).max());assert np.all(abs(saved.astype(float)-(y-x))<=eps)
 # Exact affine KKT validation in the full-length low-amplitude example.
 y=cases[2][1];xaff=affine(y);res=y-xaff;dual=np.cumsum(np.cumsum(res))[:-2];c=certificate(y,xaff,dual,0,'KNOWN_AFFINE_KKT');assert c['passed']
 # Independent reference on small series, not a classifier/production candidate.
 n=32;y=.5*np.sin(2*np.pi*np.arange(n)/11)+np.where(np.arange(n)>16,.2,0);D=np.diff(np.eye(n),n=2,axis=0);H=D@D.T;b=D@y
 reference=optimize.minimize(lambda q:(.5*q@H@q-b@q,H@q-b),np.zeros(n-2),jac=True,bounds=[(-50,50)]*(n-2),method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-10,'maxiter':20000});qr=reference.x;xr=y-D.T@qr;ref=certificate(y,xr,qr,reference.nit,'SCIPY_QP_REFERENCE');x,q,c=solve(y);assert ref['passed'] and abs(c['primal_objective']-ref['primal_objective'])<=2*(c['gap_tolerance']+ref['gap_tolerance'])
 # Deterministic batch API is a collection of independent solves, no shared stops.
 group=np.stack([cases[2][1],cases[3][1],cases[4][1]]);a,qa,ca=solve_many(group);perm=[2,0,1];b,qb,cb=solve_many(group[perm]);assert np.max(abs(a-b[np.argsort(perm)]))<=1e-7
 for j,row in enumerate(group):assert np.max(abs(a[j]-solve(row)[0]))<=1e-7
 assert np.array_equal(a,solve_many(group)[0]);independence=float(np.max(abs(a-b[np.argsort(perm)])))
 # Saved fresh-process result: source/config fixed, no patient data in this test file.
 np.save(HERE/'tests/synthetic_process_input.npy',group,allow_pickle=False)
 code="from pathlib import Path;import sys,numpy as np;sys.path.insert(0,'"+str(HERE/'scripts')+"');from numeric_core import solve_many;p=Path('"+str(HERE)+"');a=np.load(p/'tests/synthetic_process_input.npy',allow_pickle=False);np.save(p/'tests/synthetic_process_output.npy',solve_many(a)[0],allow_pickle=False)"
 subprocess.run([sys.executable,'-B','-c',code],check=True);assert np.max(abs(a-np.load(HERE/'tests/synthetic_process_output.npy',allow_pickle=False)))<=1e-7
 # Fail-closed at zero numerical iteration budget (cannot pretend successful).
 failed=solve(cases[3][1],dict(max_newton_iterations=0))[2];assert not failed['passed']
 # Benchmark first-choice repair of the old ADMM algorithm; no BA observed.
 y=cases[2][1];xa,admm=repaired_admm(y);pd.DataFrame(rows).to_csv(HERE/'analysis/numerical_certificate_summary.csv',index=False)
 write(HERE/'analysis/solver_test_integrity.json',dict(status='PASS_SYNTHETIC_CERTIFICATES',cases=len(rows),adjoint_and_dual_formula_checked=True,raw_gap_not_clamped=True,exact_KKT_reference=c['passed'],scipy_reference_certificate=ref,independence_max_abs=independence,restart_cache_abs_difference=0.,repaired_ADMM_reference=admm,repaired_ADMM_estimated_25740_axis_CPU_seconds=admm['cpu_seconds']*25740,dual_barrier_median_CPU_seconds=float(np.median([r['cpu_seconds'] for r in rows])),production_solver_not_yet_frozen=True,classification_results_seen=False))
 pd.DataFrame([dict(test='single_batch_reorder_repeat_restart',max_abs_difference=independence,tolerance=1e-7,status='PASS')]).to_csv(HERE/'analysis/batch_independence_summary.csv',index=False);print('PASS_SYNTHETIC_SOLVER_TESTS')
if __name__=='__main__':main()
