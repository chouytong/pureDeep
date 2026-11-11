"""Signal-only transfer/boundary diagnostics. No candidate arrays or model features saved."""
from common9 import *
import inspect,scipy

def summarize(frame,groups,cols):
 f=frame.groupby(['subject_id']+groups)[cols].mean().reset_index();rows=[]
 for keys,g in f.groupby(groups):
  if not isinstance(keys,tuple):keys=(keys,)
  for c in cols:
   v=g[c].dropna();rows.append(dict(zip(groups,keys),measurement=c,subject_count=g.subject_id.nunique(),mean=v.mean(),median=v.median(),p05=v.quantile(.05),p95=v.quantile(.95),unit='subject; windows/wrists averaged within subject'))
 return pd.DataFrame(rows)

def main():
 assert load(HERE/'analysis/raw_diagnostics_integrity.json')['status']=='PASS_COMPLETE_RAW_DIAGNOSTICS';freeze('SIGNAL_RESPONSE',[Path(__file__),HERE/'PROTOCOL.md',HERE/'scripts/common9.py'],no_model_inference=True,candidate_padtype='line; signal diagnostic only; not applied to cache',L1_synthetic_amplitudes=[.01,.1,1.],L1_frequencies=[.25,.5,1,2,3,4,6,8,10,12,15,20,30,40])
 source=inspect.getsource(signal.resample_poly);assert '10 * max_rate' in source and 'firwin' in source
 up,down=3,10;h=signal.firwin(201,1/10,window=('kaiser',5.));f,response=signal.freqz(h,worN=2048,fs=300);responses=[dict(response_type='HarNet_polyphase_Kaiser5_FIR_LTI',frequency_Hz=float(a),gain=float(abs(b)),phase_radians=float(np.angle(b)),input_amplitude=np.nan,phase_interpretation='causal coefficient phase; resample_poly compensates linear-phase group delay') for a,b in zip(f,response)]
 const=np.ones((1000,3),np.float32);cur=signal.resample_poly(const,3,10,axis=0);line=signal.resample_poly(const,3,10,axis=0,padtype='line');assert cur.shape==line.shape==(300,3) and abs(line[0,0]-1)<.001 and cur[0,0]<.8
 rs=records();out=[];beg=time.perf_counter()
 for k,r in enumerate(rs):
  t,x=read_raw(r);acc=np.clip(x[48:,:3].astype(np.float32),-3,3)
  if len(acc)<1000:
   before=(1000-len(acc))//2;segments=[np.pad(acc,((before,1000-len(acc)-before),(0,0)),mode='edge')]
  else:segments=[acc[:1000],acc[-1000:]]
  for window,segment in enumerate(segments):
   a=signal.resample_poly(segment,3,10,axis=0);b=signal.resample_poly(segment,3,10,axis=0,padtype='line');assert a.shape==b.shape==(300,3);delta=a-b;edge=np.r_[np.arange(10),np.arange(290,300)];inside=np.arange(30,270)
   out.append(dict(subject_id=r['subject_id'],activity=r['activity'],wrist=r['wrist'],label=r['label'],window=window,relative_L2_difference=float(np.linalg.norm(delta)/np.linalg.norm(a)) if np.linalg.norm(a)>0 else np.nan,edge_delta_RMS_g=float(np.sqrt(np.mean(np.sum(delta[edge]**2,axis=1)))),interior_delta_RMS_g=float(np.sqrt(np.mean(np.sum(delta[inside]**2,axis=1)))),maximum_delta_g=float(abs(delta).max()),current_first_magnitude=float(np.linalg.norm(a[0])),line_first_magnitude=float(np.linalg.norm(b[0])),clipped_fraction=float(np.mean(abs(x[48:,:3])>3))))
  if (k+1)%2000==0:print(json.dumps(dict(stage='resampling_signal_only',records=k+1)),flush=True)
 frame=pd.DataFrame(out);csv(frame,'resampling_boundary_private.csv',True);csv(summarize(frame,['activity','label'],['relative_L2_difference','edge_delta_RMS_g','interior_delta_RMS_g','maximum_delta_g','clipped_fraction']),'resampling_boundary_diagnostics.csv')
 # Nonlinear L1 response on synthetic waveforms only; no original data reprocessed.
 sys.path.insert(0,str(MAIN));from src.datasets.preprocessing import l1_trend_filter_batch
 t=np.arange(2048)/100;inputs=[];keys=[]
 for amp in [.01,.1,1.]:
  for freq in [.25,.5,1,2,3,4,6,8,10,12,15,20,30,40]:inputs.append(1+.02*t+amp*np.sin(2*np.pi*freq*t));keys.append((amp,freq))
 x=np.asarray(inputs);trend,solver=l1_trend_filter_batch(x,50,rho=40,max_iterations=5000,absolute_tolerance=1e-4,relative_tolerance=1e-4);assert solver.converged;y=x-trend;keep=slice(200,-200)
 for output,(amp,freq) in zip(y,keys):
  basis=np.c_[np.sin(2*np.pi*freq*t[keep]),np.cos(2*np.pi*freq*t[keep]),np.ones(len(t[keep]))];coef=np.linalg.lstsq(basis,output[keep],rcond=None)[0];responses.append(dict(response_type='L1_empirical_synthetic_NONLINEAR_NOT_LTI',frequency_Hz=freq,input_amplitude=amp,gain=float(np.linalg.norm(coef[:2])/amp),phase_radians=float(np.arctan2(coef[1],coef[0])),phase_interpretation='waveform-specific steady interior sine fit; not fixed filter phase'))
 base=1+.05*np.sin(2*np.pi*6*t);perturbed=base.copy();perturbed[:48]+=.25*np.sin(2*np.pi*30*t[:48]);pair=np.stack([base,perturbed]);trend2,solver2=l1_trend_filter_batch(pair,50,rho=40,max_iterations=5000,absolute_tolerance=1e-4,relative_tolerance=1e-4);assert solver2.converged;processed=pair-trend2;diff=processed[1,48:]-processed[0,48:]
 csv(pd.DataFrame(responses),'filter_frequency_response.csv');write(HERE/'analysis/signal_response_integrity.json',dict(status='PASS_SIGNAL_ONLY_NO_LEVEL2',scipy=scipy.__version__,resample_source_sha256=hashlib.sha256(source.encode()).hexdigest(),actual_default_window='Kaiser beta5',upsample=3,downsample=10,target_nominal_fs=30,FIR_taps=201,anti_aliasing_present=True,default_extension='zero constant',same_constant_1g_first_output_default=float(cur[0,0]),same_constant_1g_first_output_line=float(line[0,0]),boundary_candidate_diagnostic_records=8580,boundary_candidate_windows=len(frame),no_candidate_sensor_or_feature_arrays_saved=True,model_inference_calls=0,original_real_L1_reprocessing_calls=0,synthetic_L1_solver_converged=True,synthetic_L1_iterations=solver.iterations,synthetic_prefix_perturbation_max_post48_difference=float(abs(diff).max()),synthetic_prefix_perturbation_max_after5seconds=float(abs(diff[500:]).max()),nonlinear_response_not_general_transfer_function=True,new_fits=0,seconds=time.perf_counter()-beg))
 print('PASS_SIGNAL_RESPONSE_NO_LEVEL2')

if __name__=='__main__':main()
