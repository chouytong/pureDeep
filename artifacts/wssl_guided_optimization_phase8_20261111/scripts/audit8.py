"""Current source/history/checkpoint/cache audit, before fitting any Phase8 candidate."""
from common8 import *
import subprocess,re,ast,importlib.util
def main():
 for folder in ['audit','analysis','reports','configs','scripts','logs','tests','restricted_manifests','predictions','checkpoints','features']:(HERE/folder).mkdir(exist_ok=True)
 assert not (HERE/'restricted_manifests/protected_assets.json').exists()
 def git(*args):return subprocess.check_output(['git','-C','/home/zyt/pureDeep-github-upload',*args],text=True).strip()
 assert git('rev-parse','HEAD')==git('rev-parse','refs/remotes/origin/main')==START and not git('status','--porcelain')
 protected=dict(load(P7/'restricted_manifests/protected_assets.json')['sha256'])
 for p,d in protected.items():assert sha(p)==d
 for p in P7.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
 docs=[ROOT/'AGENTS.md',ROOT/'README.md',ROOT/'CONTEXT_HANDOFF.md',ROOT/'conclusions/README.md',MAIN/'README.md']+[ROOT/'artifacts'/p for p in ['phase3b_external_ssl_20260928/PHASE3B_REPORT.md','phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md','phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md','phase3e_external_priors_20261001/PHASE3E_REPORT.md','phase4_window_dynamics_20261002/FINAL_REPORT.md','phase2_str_training_20260926/PHASE2_REPORT.md','wsr01_wrist_structured_residual_20260922/WSR01_REPORT.md','wssl_ema_20261002/FINAL_REPORT.md','frequency_prior_str_20261007/FINAL_REPORT.md','wssl_activity_scaling_20261008/FINAL_REPORT.md','patch_str_20261008/PATCH_STR_REPORT.md']]+[OLD/'NEGATIVE_RESULTS_DO_NOT_REPEAT.md']+[ROOT/'artifacts'/f'{directory}/{name}.md' for directory,names in [('wssl_complementarity_phase5_20261111',['FINAL_REPORT','STATISTICAL_APPENDIX','CORRECTNESS_AND_PROVENANCE']),('wssl_representation_generalization_phase6_20261111',['FINAL_REPORT','STATISTICAL_APPENDIX']),('independent_validation_phase7_20261111',['FINAL_REPORT','INDEPENDENT_VALIDATION_PROTOCOL','NEXT_STAGE_RESEARCH_PLAN'])] for name in names]
 history=[];narrative=[]
 for p in docs:
  if not p.exists():history.append(dict(path=str(p),status='MISSING'));continue
  raw=p.read_text();protected[str(p)]=sha(p);lines=[];count=0;hide=False;level=0
  for line in raw.splitlines():
   if line.startswith('#'):
    depth=len(line)-len(line.lstrip('#'))
    if hide and depth<=level:hide=False
    if re.search(r'outer.{0,30}(performance|result|metric|accuracy|AUROC|evaluation|hold.?out)|(?:final|locked).{0,15}outer',line,re.I):hide=True;level=depth
   if hide or (re.search('outer',line,re.I) and re.search(r'\b0\.[0-9]{2,}|[0-9]{2}\.[0-9]+%',line)):count+=1;line='[OUTER NUMERIC PASSAGE SUPPRESSED]'
   lines.append(line)
  history.append(dict(path=str(p),status='CURRENT_FILE_FULL_TEXT_SCANNED',sha256=sha(p),bytes=p.stat().st_size,outer_numeric_lines_suppressed=count));narrative.append('\nFILE '+str(p)+'\n'+'\n'.join(l for l in lines if not l.startswith('|')))
 pd.DataFrame(history).to_csv(HERE/'audit/history_sources.csv',index=False);(HERE/'logs/history_narrative.txt').write_text('\n'.join(narrative))
 assert not any(r['status']=='MISSING' for r in history)
 sources=[]
 for p in MAIN.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts:continue
  digest=sha(p);mirror=Path('/home/zyt/pureDeep-github-upload')/p.relative_to(ROOT);assert mirror.is_file() and sha(mirror)==digest
  if p.suffix=='.py':ast.parse(p.read_text())
  protected[str(p)]=digest;sources.append(dict(path=str(p.relative_to(ROOT)),sha256=digest,published_match=True))
 pd.DataFrame(sources).to_csv(HERE/'audit/mainline_sources.csv',index=False)
 manifest=cfg()['data']['activity_manifests']['CrossArms'];clinical=pd.read_csv(manifest,dtype={'subject_id':str});label=dict(zip(clinical.subject_id,clinical.label.map({'PD':0,'DD':1})))
 rows=[];private=[]
 for c,i,sp,ex in splits():
  train=set(map(str,sp['train_subjects']));query=set(map(str,sp['validation_subjects']));assert not train&query and not (train|query)&set(ex)
  for seed in [42,43,44]:
   base=baseline(seed,c,i);ck=torch.load(base/'checkpoints/best.pt',map_location='cpu',weights_only=False);split=load(base/'split.json');norm=load(base/'normalization.json');prov=load(base/'provenance.json')
   assert set(split['train_subjects'])==train and set(split['validation_subjects'])==query and set(split['test_subjects'])==set(ex)
   assert set(norm['subject_ids'])==train and prov['class_names']==['PD','DD']
   for k in ['mean','std']:assert np.array_equal(ck['normalization'][k].numpy(),np.array(norm[k]))
   assert ck['config']['data']['activities']==ACTS and ck['config']['data']['wrist_order']==['left','right'] and ck['config']['data']['sequence_length'] is None
   path=featurepath(seed,c,i)
   with np.load(path,allow_pickle=False) as z:
    ids=ids_ok(z['subject_ids'],train|query);v=z['is_validation'].astype(bool);assert set(np.array(ids)[v])==query and set(np.array(ids)[~v])==train
    y=z['target'];assert all(label[s]==int(t) for s,t in zip(ids,y))
    assert z['subject'].shape==(len(ids),258) and z['structured'].shape==(len(ids),176) and z['bilateral'].shape==(len(ids),11,258)
    for name in ['subject','structured','bilateral','logits','probabilities']:assert np.isfinite(z[name]).all()
    pred=pd.read_csv(base/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id').loc[np.array(ids)[v]]
    error=np.max(abs(z['probabilities'][v]-pred[['probability_pd','probability_dd']].to_numpy()));assert error<=1e-12 and np.array_equal((z['probabilities'][v,1]>.5).astype(int),pred.prediction)
    derived=np.exp(z['logits'].astype(float)-logsumexp(z['logits'].astype(float),1,keepdims=True));assert abs(derived-z['probabilities']).max()<=2e-7
   sourcefreeze=load(OLD/'configs/baseline_extraction_v2_freeze.json')['files'];assert sha(OLD/'scripts/extract_baseline_v2.py')==sourcefreeze['scripts/extract_baseline_v2.py']
   for p in [path,base/'checkpoints/best.pt',base/'normalization.json',base/'split.json',base/'provenance.json',base/'predictions/validation.csv']:protected[str(p)]=sha(p)
   rows.append(dict(context=c,inner=i,seed=seed,status='PASS',fit_count=len(train),query_count=len(query),excluded_count=len(ex),checkpoint_sha256=sha(base/'checkpoints/best.pt'),feature_sha256=sha(path),normalization_sha256=sha(base/'normalization.json'),source_sha256=sha(OLD/'scripts/extract_baseline_v2.py'),preprocessing_sha256=sha(ROOT/'data/processed/pads_multi_activity/v2_l1_full_length/processing_config.json'),max_probability_error=error,subject_dim=258,structured_dim=176,bilateral_dim=258,activity_count=11))
   private.append(dict(context=c,inner=i,seed=seed,fit_subject_ids=sorted(train),query_subject_ids=sorted(query),excluded_subject_ids=ex,source_feature_path=str(path),supervised_encoder_training_in_sample=True,feature_creation_time_individual_hash_missing=True))
 pd.DataFrame(rows).to_csv(HERE/'audit/frozen45_integrity.csv',index=False)
 write(HERE/'restricted_manifests/frozen45_lineage.json',dict(records=private));write(HERE/'restricted_manifests/protected_assets.json',dict(sha256=protected,git_start=START))
 write(HERE/'audit/architecture.json',dict(wrist_local=64,external=1024,projected=64,bilateral=258,activities=11,subject=258,structured_token=16,structured_concat=176,main_readout=434,baseline_readout='affine subject258 + affine structured176; dropout eval off',baseline_activity='independent token scoring + softmax competition + weighted pooling',STR='tokenwiseLinear-GELU then orderedconcat then affine',pairwise_content_token_mixing=False,cross_activity_relations_completely_absent=False,B_equivalent_mechanism=False,expected_B_status='NOVEL_RELATIVE_TO_THIS_READOUT_NOT_GLOBAL_NOVELTY'))
 write(HERE/'audit/startup_integrity.json',dict(status='PASS_BEFORE_PHASE8_FITS',history_documents=len(history),mainline_files=len(sources),pairs=45,protected_files=len(protected),feature_provenance_qualified_with_historical_disclosure=True,deepOOF_asset_record_present=(OLD/'analysis/OOF_inventory_135.csv').exists(),xgboost_available=bool(importlib.util.find_spec('xgboost')),new_fit_calls=0,outer_performance_surfaced=False))
 freeze('A1',[HERE/'PROTOCOL.md',HERE/'scripts/common8.py',HERE/'scripts/audit8.py',HERE/'scripts/ensemble8.py'],historical_ensemble_results_already_known=True,new_phase8_results_seen=False)
 guard();print('PHASE8_ASSET_GATE_PASS',len(protected),len(history))
if __name__=='__main__':main()
