from pathlib import Path
import hashlib,json,csv,subprocess,sys

ROOT=Path('/home/zyt/deep_final');HERE=ROOT/'artifacts/wssl_reference_audit_20261111'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 # The directory was created only by the upload of the preregistered scripts.
 assert not (HERE/'configs/protected_assets.json').exists()
 for name in ['configs','scripts','tests','analysis','restricted_manifests','features','predictions','logs']:(HERE/name).mkdir(exist_ok=True)
 specs=[
 ('V8','reports','historical pure-deep baseline','initial temporal + learned moments',0,'retained historical baseline; not current'),
 ('V8-GN','artifacts/foundation_validation','V8','BatchNorm -> GroupNorm',45,'retained historical reference'),
 ('STR-01','artifacts/str01_structured_token_residual_20260921','V8-GN','ordered activity structured residual',45,'retained matched baseline'),
 ('WSR-01','artifacts/wsr01_wrist_structured_residual_20260922','STR-01','wrist structured residual',45,'rejected'),
 ('prototype','artifacts/invariant_representation/pa_stage3_20260919','V8-GN','prototype alignment training loss',45,'rejected; not identical to frozen reference classifier'),
 ('strong-backbone','artifacts/strong_backbone_benchmark_20260921','V8-GN','ResNet1D / InceptionTime fixed comparisons',90,'rejected / not retained'),
 ('DSG-01','artifacts/dsg01_cross_subject_disease_subspace_20260926','STR / V8-GN','cross-subject frozen representation diagnosis',0,'completed; conditional consistency training not-run'),
 ('RGD-01','artifacts/rgd01_h1_str_residual_ranking_20260926','STR-01','H1/STR residual ranking diagnosis',0,'completed; new neural training not-run'),
 ('PRR-01','artifacts/prr01_frozen_str_pair_rescue_20260926','STR-01','fixed ridge pair-rescue probes',0,'completed; new readout training not-run'),
 ('PAG-01','artifacts/pag01_preaggregation_disease_info_20260926','STR-01','conditional disease-information probes',0,'completed; new training gate failed'),
 ('phase2','artifacts/phase2_str_training_20260926','STR-01','16 registered single factors plus baseline reproduction',765,'all16 alternatives not retained'),
 ('phase3a','artifacts/phase3a_puredeep_20260928','STR-01','sampling compensation / SAM / two auxiliary controls',180,'CASE D; no candidate retained'),
 ('phase3b','artifacts/phase3b_external_ssl_20260928','STR-01','SSL-only / pretrained frozen residual / random frozen control',135,'b_str_pretrained retained; other controls not retained'),
 ('phase3c','artifacts/phase3c_ssl_adaptation_20260930','Frozen WSSL','layer5 / middle+final / deep-OOF threshold / sample-pool curves','see variant inventory','CASE D; retain frozen WSSL; threshold diagnostic already done'),
 ('phase3d','artifacts/phase3d_ssl_adaptation_20261001','Frozen WSSL','adapter / gated fusion / train-only masked domain adaptation',135,'none retained; conditional activity gate not-run'),
 ('phase3e','artifacts/phase3e_external_priors_20261001','Frozen WSSL','HarNet5 / BioPM neural-only / random control',180,'none retained; dual prior not-run'),
 ('phase4','artifacts/phase4_window_dynamics_20261002','Frozen WSSL','mean control / front-back dynamics / raw Acc / subtype auxiliary',180,'all four fixed configurations rejected'),
 ('wssl-ema','artifacts/wssl_ema_20261002','Frozen WSSL','EMA with matched ordinary reproduction','see variant inventory','EMA not retained'),
 ('wssl-review','artifacts/wssl_review_20261004','Frozen WSSL','one LR1e-4 vs2e-4 control',45,'rejected; original LR retained'),
 ('frequency','artifacts/frequency_prior_str_20261007','STR-01','capacity residual and fixed frequency-prior residual',90,'rejected'),
 ('wssl-contribution','artifacts/wssl_contribution_20261007','Frozen WSSL','37-condition deletion diagnostics, no new model',0,'completed; phase B qualification only, no phase C loss audit'),
 ('activity-scaling','artifacts/wssl_activity_scaling_20261008','Frozen WSSL','11 unconstrained activity scalars',45,'INCONCLUSIVE / DO NOT RETAIN; phase C not started'),
 ('patch-str','artifacts/patch_str_20261008','STR-01','200/100 local bag and ordered patches',90,'REJECT; P3 not-run'),
 ('patch-recheck','artifacts/patch_str_recheck_20261111','archived Patch/STR','2026-11-11 recheck of2026-10-08 training',0,'reused exact historical experiment; no new training'),
 ]
 protected={};ledger=[];sources=[];variant_rows=[]
 for eid,rel,base,change,expected,decision in specs:
  folder=ROOT/rel
  assert folder.exists(),folder
  docs=[]
  for p in folder.rglob('*'):
   if not p.is_file() or any(t in p.parts for t in ['vendor','checkpoints','predictions','features','logs','runs','smoke','outer_final','quarantine','restricted_manifests']):continue
   if p.suffix=='.md' or (p.suffix=='.json' and any(k in p.name.lower() for k in ['decision','manifest','integrity','correctness','audit'])):
    # Read each complete source, not just a README header or one metric table.
    text=p.read_text();digest=sha(p);protected[str(p)]=digest;docs.append((p,text,digest));sources.append(dict(experiment=eid,path=str(p),sha256=digest,bytes=p.stat().st_size))
  stages=[]
  for p in (folder/'runs').glob('**/stage_status.json'):
   if 'inner_' not in str(p) or 'outer_final' in p.parts:continue
   obj=json.loads(p.read_text())
   if obj.get('status')=='complete':stages.append(p)
  variants={}
  for p in stages:
   name=p.relative_to(folder/'runs').parts[0];variants[name]=variants.get(name,0)+1
  for name,n in variants.items():variant_rows.append(dict(experiment=eid,variant=name,completed_inner_stages=n))
  status='inconclusive' if 'INCONCLUSIVE' in decision else 'retained' if decision.startswith('retained') or 'b_str_pretrained retained' in decision else 'not-run' if expected==0 and 'not-run' in decision else 'rejected' if 'reject' in decision.lower() or 'not retained' in decision or 'none retained' in decision else 'completed'
  report_docs=[(p,t,h) for p,t,h in docs if any(k in p.name.upper() for k in ['REPORT','FINAL','DECISION'])]
  snippets=[]
  for p,text,h in report_docs:
   lines=text.splitlines();keep=set()
   for i,line in enumerate(lines):
    if any(k in line.lower() for k in ['retain','reject','inconclusive','not run','not-run','oof','gate','loss','未执行','未启动','停止','拒绝','不保留','| baseline','| b:','0.7196','0.6972']):keep.update(range(max(0,i-1),min(len(lines),i+2)))
   snippets.append('\nSOURCE '+str(p)+' SHA256 '+h+'\n'+'\n'.join(lines[i] for i in sorted(keep)))
  ledger.append(dict(experiment_id=eid,target_baseline=base,unique_change=change,expected_runs=str(expected),actual_completed_inner_stages=len(stages),actual_new_deep_runs='0' if expected==0 else 'see actual stage inventory and source report; baseline copies distinguished',split_seed_coverage='15 fixed development splits;42/43/44 unless inference-only',formal_conclusion_date='2026-11-11 recheck; training2026-10-08' if eid=='patch-recheck' else ('2026-10-08' if eid in ['activity-scaling','patch-str'] else 'see dated formal source'),status=status,conclusion=decision,major_metrics=' | '.join(l for _,t,_ in report_docs for l in t.splitlines() if l.startswith('|') and any(k in l.lower() for k in ['baseline','frozen','b:','balanced','| ba','| auroc']))[:2400],rejection_reason='see complete conclusion excerpts and preregistered gate; do not generalize family',protocol_and_sources=';'.join(str(p.relative_to(ROOT)) for p,_,_ in docs),source_hash=hashlib.sha256(''.join(h for _,_,h in docs).encode()).hexdigest(),baseline_reused=eid not in ['V8','V8-GN','STR-01'],asset_exists=True))
  (HERE/'analysis'/f'history_{eid}.txt').write_text('\n'.join(snippets))
 # Protect canonical imports, frozen source and core assets before new work.
 for base in [ROOT/'mainline',ROOT/'foundation_validation/src']:
  for p in base.rglob('*'):
   if p.is_file() and p.suffix in ['.py','.yaml','.toml','.json','.sha256'] and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
 for p in [ROOT/'data/ssl/harnet10_embeddings.npz',ROOT/'artifacts/phase3b_external_ssl_20260928/analysis/ssl_embeddings_all.npz',ROOT/'conclusions/organization_20261111/VERIFICATION.json',ROOT/'artifacts/wssl_ema_20261002/scripts/independent_wssl.py']:
  protected[str(p)]=sha(p)
 for base in [ROOT/'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained',ROOT/'artifacts/phase3c_ssl_adaptation_20260930/runs/adapt',ROOT/'artifacts/phase3c_ssl_adaptation_20260930/oof']:
  assert base.exists(),base
  for p in base.rglob('*'):
   if p.is_file() and ('inner_' in str(p)) and (p.suffix in ['.pt','.json','.yaml','.csv']):protected[str(p)]=sha(p)
 gitsha=subprocess.check_output(['git','-C','/home/zyt/pureDeep-github-upload','rev-parse','HEAD'],text=True).strip()
 (HERE/'configs/protected_assets.json').write_text(json.dumps(dict(sha256=protected,git_sha=gitsha,outer_access=False),indent=2)+'\n')
 for name,rows in [('EXPERIMENT_LEDGER',ledger),('analysis/variant_stage_inventory',variant_rows),('analysis/history_sources',sources)]:
  with (HERE/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 md='# Historical experiment ledger\n\nActual run counts are inventoried from completed inner-stage metadata; inference-only stages are not new training. Variant inventory discloses reproduced baselines and learning curves separately. No outer artifacts were loaded.\n\n| ID | baseline | single change / phase | expected runs | complete inner metadata | conclusion |\n|---|---|---|---|---|---|\n'
 for x in ledger:md+=f"| {x['experiment_id']} | {x['target_baseline']} | {x['unique_change']} | {x['expected_runs']} | {x['actual_completed_inner_stages']} | {x['conclusion']} |\n"
 (HERE/'EXPERIMENT_LEDGER.md').write_text(md)
 # Use the exact previously validated explicit implementation with canonical imports.
 source=ROOT/'artifacts/wssl_ema_20261002/scripts/independent_wssl.py'
 target=HERE/'scripts/independent_wssl.py';assert not target.exists();target.write_bytes(source.read_bytes())
 print(json.dumps(dict(status='BOOTSTRAP_LOCKED',protected_files=len(protected),history_sources=len(sources),experiments=len(ledger),git_sha=gitsha)),flush=True)

if __name__=='__main__':main()
