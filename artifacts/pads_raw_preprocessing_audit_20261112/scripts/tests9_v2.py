"""Synthetic correctness plus one metadata-authorized raw-record parser smoke; no fitting."""
from common9 import *
from diagnostics9_schema_v2 import constant_runs,sensor_metrics,temporal_windows,one
import importlib.util

def main():
 assert load(HERE/'analysis/metadata_schema_correction.json')['status']=='PASS_BEFORE_SIGNAL_STATISTICS'
 assert constant_runs(np.zeros(100))[:2]==(100,1.) and constant_runs(np.arange(100))[:2]==(1,0.)
 t=np.arange(1024)/100;x=np.c_[np.sin(2*np.pi*6*t),np.zeros_like(t),np.zeros_like(t)];m=sensor_metrics(t,x);assert abs(m['dominant_frequency']-6)<.5 and .9<m['fraction_3_12']<=1.01
 short=temporal_windows(np.arange(20)/100);assert all(not k.any() for name,k in short)
 try:read_raw(dict(raw_path=str(HERE/'tests/not_raw.txt')))
 except AssertionError:pass
 else:raise AssertionError('Raw read escaped allowed root')
 result=one(records()[0]);assert result[0]['units_match'] and result[0]['gyro_processed_exact'] and len(result[2])==6 and len(result[4])==22
 # Apply supplied constants only; do not call any normalization fit function.
 a=np.array([1.,3.]);z=(a-2.)/2.;assert np.array_equal(z,np.array([-.5,.5]))
 sys.path.insert(0,str(MAIN));import torch,yaml
 torch.set_num_threads(4);torch.manual_seed(SEED)
 p=ROOT/'artifacts/wssl_representation_generalization_phase6_20261111/scripts/independent_wssl.py';spec=importlib.util.spec_from_file_location('protected_independent',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 config=yaml.safe_load((MAIN/'configs/str01_seed42.yaml').read_text());model=mod.IndependentWSSL(config).eval();old=ROOT/'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained/seed42/outer_0/inner_0/checkpoints/best.pt';model.load_state_dict(torch.load(old,map_location='cpu',weights_only=False)['model_state']);before={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
 x=torch.randn(2,11,2,6,128);wm=torch.ones(2,11,2);am=torch.ones(2,11,dtype=torch.bool);lengths=torch.full((2,11),96,dtype=torch.long);lengths[:,::2]=64;ssl=torch.randn(2,11,2,1024);am[:,3]=False;lengths[:,3]=0;wm[:,4,1]=0
 with torch.inference_mode():
  expected=model(x,wm,am,lengths,ssl_features=ssl)['logits'];changed=x.clone()
  for b in range(2):
   for a in range(11):changed[b,a,:,:,lengths[b,a]:]=999.
  changed[:,4,1]=999.;actual=model(changed,wm,am,lengths,ssl_features=ssl)['logits'];assert torch.equal(expected,actual) and expected.shape==(2,2) and torch.isfinite(expected).all()
 after={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()};assert before==after
 write(HERE/'tests/correctness_smoke.json',dict(status='PASS',metadata_Time_plus6_sensor_checked=True,raw_record_smoke_count=1,raw_root_allowlist=True,constant_run_hand_checks=True,synthetic_6Hz_frequency_recovery=True,short_boundary_windows_NA=True,given_normalization_apply_only=True,padding_activity_wrist_mask_invariance_bitwise=True,synthetic_model_inputs_only=True,checkpoint_weights_unchanged=True,model_training_calls=0,normalization_fit_calls=0,no_outer_loader=True))
 print('PASS_PHASE9_CORRECTNESS_SMOKE')

if __name__=='__main__':main()
