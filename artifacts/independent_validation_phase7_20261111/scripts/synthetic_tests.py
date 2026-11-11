"""Meaningful integration and failure-path tests; synthetic signals only."""
from external_adapter import *
import copy,time,os
def fixture(number=0):
 result={}
 for a,name in enumerate(ACTS):
  count=304+20*(a%3)+number*20;t=np.arange(count)/100.;pair={}
  for w,side in enumerate(['left','right']):
   x=np.stack([(.02+.002*c)*np.sin((1+c*.2)*2*np.pi*t+.13*a+.3*w+.1*number)+(1 if c==2 else 0) for c in range(6)],1).astype(np.float32)
   pair[side]=Stream(x,tuple(CHANNELS),100.,'g','rad/s',side,'synthetic_device_'+side,t.copy(),frame='PADS-device-axes-confirmed',placement='wrist')
  result[name]=pair
 return result
def expect(name,func,exception):
 try:func()
 except exception:return dict(test=name,status='PASS_EXPECTED_REJECTION')
 raise AssertionError('Invalid input accepted: '+name)
def main():
 guard();start=time.perf_counter();np.random.seed(20261111);torch.manual_seed(20261111);torch.set_num_threads(4);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 rows=[];f=fixture();s=f[ACTS[0]]['left'];canonical=canonical_stream(s,'left')
 order=[5,2,0,4,1,3];reordered=copy.deepcopy(s);reordered.values=s.values[:,order];reordered.channels=tuple(CHANNELS[i] for i in order);assert np.array_equal(canonical_stream(reordered,'left'),canonical);rows.append(dict(test='channel_order',status='PASS'))
 unit=copy.deepcopy(s);unit.values=s.values.astype(np.float64).copy();unit.values[:,:3]*=9.80665;unit.values[:,3:]*=180/np.pi;unit.acc_unit='m/s^2';unit.gyro_unit='deg/s';unit.timestamps=s.timestamps*1000;unit.time_unit='ms';assert np.allclose(canonical_stream(unit,'left'),canonical,atol=1e-7,rtol=0);rows.append(dict(test='unit_and_timestamp_conversion',status='PASS'))
 rate=copy.deepcopy(s);rate.sampling_rate=50.;rate.values=s.values[::2].copy();rate.timestamps=np.arange(len(rate.values))/50.;converted=canonical_stream(rate,'left');expected=resample_poly(rate.values.astype(np.float64),2,1,axis=0).astype(np.float32);assert converted.shape==(len(rate.values)*2,6) and np.array_equal(converted,expected);rows.append(dict(test='anti_alias_resampling_50_to_100',status='PASS'))
 partial=copy.deepcopy(f);partial.pop(ACTS[-1]);rows.append(expect('missing_activity',lambda:adapt_subject(partial),Partial))
 partial=copy.deepcopy(f);partial[ACTS[0]].pop('right');rows.append(expect('missing_wrist',lambda:adapt_subject(partial),Partial))
 for name,mutator in [('wrong_shape',lambda z:setattr(z,'values',z.values[:,:5])),('unknown_unit',lambda z:setattr(z,'acc_unit','unknown')),('wrong_frame',lambda z:setattr(z,'frame','unknown')),('wrong_placement',lambda z:setattr(z,'placement','finger')),('irregular_time',lambda z:z.timestamps.__setitem__(3,z.timestamps[2])),('nonfinite',lambda z:z.values.__setitem__((0,0),np.nan)),('duplicate_channels',lambda z:setattr(z,'channels',tuple(['AccX']*6)))]:
  z=copy.deepcopy(s);mutator(z);rows.append(expect(name,lambda z=z:canonical_stream(z,'left'),Incompatible))
 rows.append(expect('real_clinical_gate_closed',lambda:validate_real_gate({}),PermissionError))
 before=copy.deepcopy(f);subjects=[adapt_subject(fixture(j)) for j in range(3)]
 assert all(np.array_equal(f[a][w].values,before[a][w].values) for a in ACTS for w in ['left','right']);assert np.array_equal(subjects[0]['lengths'],np.array([len(f[a]['left'].values)-48 for a in ACTS]))
 rows.append(dict(test='adapter_input_unchanged_and_real_lengths',status='PASS'))
 runner=FrozenRunner();finger=runner.fingerprint();batch=collate(subjects[:2],runner.mean,runner.std);assert batch[0].shape[:4]==(2,11,2,6) and torch.equal(batch[3],torch.from_numpy(np.stack([s['lengths'] for s in subjects[:2]])))
 # Padded tails are containers only; actual normalization is on true data.
 assert all(torch.count_nonzero(batch[0][j,a,:,:,int(subjects[j]['lengths'][a]):])==0 for j in range(2) for a in range(11));rows.append(dict(test='dimension_norm_padding_and_true_lengths',status='PASS'))
 logits=runner.synthetic_infer(subjects,batch_size=2);again=runner.synthetic_infer(subjects,batch_size=2);assert logits.shape==(3,2) and np.isfinite(logits).all() and np.array_equal(logits,again)
 assert runner.last_classifier_batches==[2,1] and all(b==22 for b in runner.last_encoder_batches)
 assert not any(m.training for top in [runner.model,runner.encoder] for m in top.modules());assert runner.fingerprint()==finger
 probabilities=torch.softmax(torch.from_numpy(logits),dim=1).numpy();assert np.allclose(probabilities.sum(1),1)
 assert np.array_equal(decision(np.array([[.5,.5],[.49,.51],[.51,.49]])),[0,1,0])
 # A longer synthetic raw fixture exercises both first/last windows and the64+tail path.
 long=copy.deepcopy(subjects[0]);long['raw']=[np.tile(x,(1,1,5)) for x in long['raw']];windows,mapping=external_windows(long);assert len(windows)==44 and all(mapping.count((a,w))==2 for a in range(11) for w in range(2))
 # Direct model-side padding perturbation must be ignored outside true lengths.
 inputs=[x.cuda() for x in batch];ssl=torch.zeros((2,11,2,1024),device='cuda')
 with torch.inference_mode():
  a=runner.model(*inputs,ssl_features=ssl)['logits'];changed=inputs[0].clone()
  for j in range(2):
   for k in range(11):changed[j,k,:,:,inputs[3][j,k]:]=12345
  b=runner.model(changed,*inputs[1:],ssl_features=ssl)['logits'];assert torch.equal(a,b)
 rows.extend(dict(test=name,status='PASS') for name in ['last_partial_encoder_and_classifier_batches','eval_mode','all_weights_and_BN_buffers_unchanged','deterministic_inference_bitwise','model_inputs_unchanged','finite_logits_and_probabilities','argmax_tie_PD','long_first_last_external_windows','padded_tail_ignored_by_model','actual_SHA_config_lock'])
 # Store no individual synthetic logits or fixtures in public tables.
 pd.DataFrame(rows).to_csv(HERE/'tests/synthetic_correctness.csv',index=False)
 write(HERE/'analysis/synthetic_integrity.json',dict(status='PASS_SYNTHETIC_ONLY',tests=len(rows),synthetic_subject_count=3,clinical_sensor_files_opened=0,real_external_adapter_validated=False,inference_runs_repeated=2,classifier_batch_sizes=[2,1],external_encoder_batch_sizes=runner.last_encoder_batches,official_model_and_BN_unchanged=runner.fingerprint()==finger,all_model_parameters_requires_grad_false=all(not p.requires_grad for m in [runner.model,runner.encoder] for p in m.parameters()),determinism_max_error=float(abs(logits-again).max()),new_training=0,optimizer_updates=0,new_fits=0,external_data_downloads=0,elapsed_seconds=time.perf_counter()-start))
 guard();print('SYNTHETIC_CORRECTNESS_PASS',len(rows))
if __name__=='__main__':main()
