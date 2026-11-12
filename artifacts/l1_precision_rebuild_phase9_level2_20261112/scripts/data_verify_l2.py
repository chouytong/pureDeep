"""Read-back certificates with an independently assembled sparse D2 operator."""
from common_l2 import *
from scipy import sparse
def main():
 ready=load(PRIVATE/'manifests/data_ready.json');assert ready['status']=='PASS_COMPLETE_CERTIFIED_DATA' and ready['version']==VERSION;manifest=Path(ready['manifest_path']);assert sha(manifest)==ready['manifest_sha256'];rows=load(manifest)['records'];operators={};verified=0;negative=0;maximum_ratio=0.;maximum_station=0.
 for k,r in enumerate(rows):
  assert r['status']=='PASS' and r['version']==VERSION and sha(r['raw_path'])==r['raw_sha256'] and sha(r['output_path'])==r['output_sha256'] and sha(r['certificate_arrays_path'])==r['certificate_arrays_sha256'];raw=np.loadtxt(r['raw_path'],delimiter=',',dtype=float);n=len(raw);y=raw[:,1:4].T
  if n not in operators:operators[n]=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csr')
  D=operators[n]
  with np.load(r['certificate_arrays_path'],allow_pickle=False) as z:
   assert str(z['solver_source_sha256'])==r['solver_source_sha256'] and str(z['solver_config_sha256'])==r['solver_config_sha256'] and str(z['raw_sha256'])==r['raw_sha256'] and str(z['version'])==VERSION;x=z['trend'].copy();q=z['dual'].copy()
  assert x.shape==(3,n) and q.shape==(3,n-2) and x.dtype==q.dtype==np.float64
  for j in range(3):
   assert abs(q[j]).max()<=50;v=D.T@q[j];res=y[j]-x[j];dx=D@x[j];p=float(.5*np.sum(res.astype(np.longdouble)**2)+50*np.sum(abs(dx),dtype=np.longdouble));dual=float(np.sum(y[j].astype(np.longdouble)*v.astype(np.longdouble))-.5*np.sum(v.astype(np.longdouble)**2));gap=p-dual;cert=r['axis_certificates'][j];station=float(abs(x[j]-y[j]+v).max());roundoff=cert['negative_gap_roundoff_tolerance'];assert gap>=-roundoff and gap<=cert['gap_tolerance']+roundoff and station<=1e-9*(1+abs(y[j]).max()) and abs(gap-cert['raw_gap'])<=roundoff+1e-11;negative+=gap<0;maximum_ratio=max(maximum_ratio,gap/cert['gap_tolerance']);maximum_station=max(maximum_station,station);verified+=1
  saved=np.load(r['output_path'],allow_pickle=False);assert np.array_equal(saved[3:],raw[48:,4:7].T.astype(np.float32)) and saved.shape==(6,n-48);residual=y-x;eps=abs(np.spacing(saved[:3])).astype(float)+8*np.finfo(float).eps*(1+abs(residual).max());assert np.all(abs(saved[:3].astype(float)-residual[:,48:])<=eps)
  if (k+1)%2200==0:print(json.dumps(dict(stage='data_certificate_readback',records=k+1,axes=verified)),flush=True)
 assert verified==25740;write(HERE/'analysis/data_certificate_readback.json',dict(status='PASS_INDEPENDENT_SPARSE_READBACK',version=VERSION,manifest_sha256=sha(manifest),axes_verified=verified,negative_gaps_in_preregistered_roundoff_count=negative,max_gap_over_tolerance=maximum_ratio,maximum_stationarity_residual=maximum_station,full_raw_sequence_certificates=True,source_input_output_hashes_verified=True,all_Gyro_and_float32_quantization_checks=True));print('PASS_CERTIFICATE_ARRAY_READBACK')
if __name__=='__main__':main()
