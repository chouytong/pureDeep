"""Synthetic metric tests: ties, weighted replication, extreme NLL, full BH families."""
from common_l2 import *
from metrics_l2 import pred_metrics,bh_planned
from weighted_l2 import weighted

def main():
 y=np.array([0,0,1,1]);p=np.array([[.5,.5],[.4,.6],[.4,.6],[.5,.5]]);z=np.log(p);ones=weighted(y,p,z,np.ones((1,4)))[0];canonical=np.array(list(pred_metrics(y,p,z).values()));assert np.allclose(ones,canonical,atol=2e-12,rtol=0);assert pred_metrics(y,p,z)['accuracy']==.5
 w=np.array([[2,0,1,3]]);repeat=np.repeat(np.arange(4),w[0]);assert np.allclose(weighted(y,p,z,w)[0],list(pred_metrics(y[repeat],p[repeat],z[repeat]).values()),atol=2e-12,rtol=0)
 order=[3,1,0,2];assert np.allclose(weighted(y[order],p[order],z[order],w[:,order]),weighted(y,p,z,w),atol=2e-12,rtol=0)
 extreme=np.array([[1000.,-1000.],[-1000.,1000.],[1000.,-1000.],[-1000.,1000.]]);prob=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
 assert pred_metrics(y,prob,extreme)['nll']==1000
 with np.errstate(divide='ignore'):assert np.isinf(pred_metrics(y,prob)['nll'])
 q=bh_planned([.01,np.nan,.2,.9]);assert np.isnan(q[1]) and abs(q[0]-.04)<1e-15
 write(HERE/'tests/metric_synthetic_checks.json',dict(status='PASS',DD_threshold_strict_tie_PD=True,weighted_replication_equivalence=True,permutation_invariance=True,extreme_raw_logits_NLL_finite=True,zero_true_probability_NLL_not_clipped=True,missing_tests_retain_full_BH_family=True))
 print('METRIC_SYNTHETIC_TESTS_PASS')
if __name__=='__main__':main()
