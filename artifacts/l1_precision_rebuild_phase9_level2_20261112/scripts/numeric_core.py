"""Independent-series fixed-lambda L1 trend solve, with primal/dual certificates."""
import numpy as np,time
from functools import lru_cache
from scipy.linalg import solveh_banded
LAMBDA=50.

@lru_cache(maxsize=8)
def hessian_template(m):
 ab=np.zeros((3,m));ab[0]=6
 if m>1:ab[1,:-1]=-4
 if m>2:ab[2,:-2]=1
 ab.setflags(write=False);return ab

def d2(x):return x[:-2]-2*x[1:-1]+x[2:]
def adjoint(q):
 out=np.zeros(len(q)+2);out[:-2]+=q;out[1:-1]-=2*q;out[2:]+=q;return out
def affine(y):
 index=np.arange(len(y),dtype=float);center=index-index.mean();return y.mean()+center*np.dot(center,y)/np.dot(center,center)
def energy(y):return .5*float(np.sum((y-affine(y))**2,dtype=np.longdouble))
def certificate(y,x,q,iterations,reason,raw_q=None):
 q0=q if raw_q is None else raw_q;q=np.clip(q,-LAMBDA,LAMBDA);v=adjoint(q);res=y-x;dx=d2(x)
 p=float(.5*np.sum(res.astype(np.longdouble)**2)+LAMBDA*np.sum(abs(dx),dtype=np.longdouble));dual=float(np.sum(y.astype(np.longdouble)*v.astype(np.longdouble))-.5*np.sum(v.astype(np.longdouble)**2));gap=p-dual;e=energy(y);tol=1e-8+1e-4*e
 roundoff=64*np.finfo(float).eps*(1+abs(p)+abs(dual)+float(np.sum(abs(y*v)))+LAMBDA*float(np.sum(abs(x))))
 station=float(abs(x-y+v).max());valid=np.isfinite([p,dual,gap,e]).all() and gap>=-roundoff and gap<=tol and station<=1e-9*(1+abs(y).max())
 return dict(primal_objective=p,dual_objective=dual,raw_gap=gap,E_aff=e,gap_tolerance=tol,negative_gap_roundoff_tolerance=roundoff,negative_gap_in_roundoff=bool(gap<0 and gap>=-roundoff),dual_feasibility_violation=max(0.,float(abs(q).max()-LAMBDA)),raw_dual_feasibility_violation=max(0.,float(abs(q0).max()-LAMBDA)),dual_projection_max_change=float(abs(q-q0).max()),stationarity_residual=station,iterations=iterations,stopping_reason=reason if valid else 'CERTIFICATE_NOT_PASSED',passed=bool(valid))

def solve(y,config=None):
 cfg=dict(max_newton_iterations=600,max_barrier_stages=30,barrier_reduction=.1,line_search_steps=60,step_fraction=.995);cfg.update(config or {})
 y=np.asarray(y,dtype=np.float64)
 if y.ndim!=1 or len(y)<3 or not np.isfinite(y).all():raise ValueError('Expected finite1D series of length>=3')
 beg=time.process_time();m=len(y)-2;q=np.zeros(m);x=y.copy();cert=certificate(y,x,q,0,'CERTIFIED_INITIAL');iterations=0
 if cert['passed']:cert['cpu_seconds']=time.process_time()-beg;return x,q,cert
 b=d2(y);e=energy(y);mu=max(e/(2*m),1e-12);minimum_mu=max((1e-8+1e-4*e)/(20*m),1e-18)
 def dual_hessian(q):return d2(adjoint(q))
 def barrier_value(q,mu):
  ratio=q/LAMBDA
  if np.any(abs(ratio)>=1):return np.inf
  v=adjoint(q);return float(.5*np.sum(v.astype(np.longdouble)**2)-np.sum(b.astype(np.longdouble)*q.astype(np.longdouble))-mu*np.sum(np.log1p(-ratio*ratio)))
 for outer in range(cfg['max_barrier_stages']):
  for inner in range(60):
   if iterations>=cfg['max_newton_iterations']:break
   iterations+=1;plus=LAMBDA-q;minus=LAMBDA+q;gradient=dual_hessian(q)-b+mu*(1/plus-1/minus);diagonal=6+mu*(1/plus**2+1/minus**2);ab=hessian_template(m).copy();ab[0]=diagonal
   try:step=solveh_banded(ab,-gradient,lower=True,check_finite=False)
   except Exception as error:
    cert=certificate(y,x,q,iterations,'FAIL_NEWTON_SYSTEM');cert['passed']=False;cert['error']=str(error);cert['cpu_seconds']=time.process_time()-beg;return x,q,cert
   direction=float(np.dot(gradient,step));alpha=1.;pos=step>0;neg=step<0
   if pos.any():alpha=min(alpha,cfg['step_fraction']*float(np.min(plus[pos]/step[pos])))
   if neg.any():alpha=min(alpha,cfg['step_fraction']*float(np.min(minus[neg]/(-step[neg]))))
   current=barrier_value(q,mu);accepted=False
   for line in range(cfg['line_search_steps']):
    candidate=q+alpha*step
    if np.max(abs(candidate))<LAMBDA and barrier_value(candidate,mu)<=current+.01*alpha*direction+1e-15*(1+abs(current)):
     accepted=True;break
    alpha*=.5
   if not accepted:break
   q=candidate;x=y-adjoint(q);cert=certificate(y,x,q,iterations,'CERTIFIED_DUAL_BARRIER')
   if cert['passed']:cert.update(cpu_seconds=time.process_time()-beg,barrier_stages=outer+1,final_barrier=mu);return x,q,cert
   if -direction/2<=max(1e-12,mu*.05):break
  if iterations>=cfg['max_newton_iterations']:break
  mu=max(mu*cfg['barrier_reduction'],minimum_mu)
 cert=certificate(y,x,q,iterations,'FAIL_MAXIMUM_BUDGET');cert['passed']=False;cert['stopping_reason']='FAIL_MAXIMUM_BUDGET';cert['cpu_seconds']=time.process_time()-beg;return x,q,cert

def solve_many(values,config=None):
 values=np.asarray(values,float)
 if values.ndim!=2:raise ValueError('Expected series,time matrix')
 result=[solve(row,config) for row in values];return np.stack([v[0] for v in result]),[v[1] for v in result],[v[2] for v in result]
