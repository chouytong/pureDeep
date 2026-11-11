"""One-time immutable asset lock and analysis registration; no model execution."""
from common import *
import subprocess,os,importlib.metadata,datetime

def main():
    for name in ['configs','scripts','tests','analysis','restricted_manifests','logs']:(HERE/name).mkdir(exist_ok=True)
    target=HERE/'restricted_manifests/protected_assets_phase5.json';assert not target.exists(),'Preserve existing phase5 lock'
    gitsha=subprocess.check_output(['git','-C','/home/zyt/pureDeep-github-upload','rev-parse','HEAD'],text=True).strip();assert gitsha=='38c89251a442bb245a77d4bffc539da050fcdeb1'
    prior=load(OLD/'configs/protected_assets.json');protected=dict(prior['sha256'])
    for p,d in protected.items():assert sha(p)==d,p
    for p in OLD.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
    reports=[ROOT/'README.md',ROOT/'AGENTS.md',ROOT/'CONTEXT_HANDOFF.md',ROOT/'GIT_VERSION_CONTROL.md',ROOT/'conclusions/CURRENT_MODEL.json']
    reports += [OLD/n for n in ['FINAL_REPORT.md','STATISTICAL_APPENDIX.md','LITERATURE_VERIFICATION.md','REPRODUCE.md','AUDIT_REPORT.md','MODALITY_ERROR_DIAGNOSTICS.md','CALIBRATION_DIAGNOSTICS.md','TYPICALITY_REPORT.md']]
    reports += [ROOT/'artifacts'/n for n in ['phase3b_external_ssl_20260928/PHASE3B_REPORT.md','phase3c_ssl_adaptation_20260930/PHASE3C_PROTOCOL.md','phase3c_ssl_adaptation_20260930/PHASE3C_REPORT.md','phase3d_ssl_adaptation_20261001/PHASE3D_REPORT.md','wssl_contribution_20261007/PHASE_A_REPORT.md','wssl_contribution_20261007/REPRODUCTION_REPORT.md']]
    sources=[]
    for p in reports:
        assert p.exists(),p;text=p.read_text();protected[str(p)]=sha(p)
        sources.append(dict(path=str(p),sha256=protected[str(p)],bytes=p.stat().st_size,complete_document_read=True,scope='current development reports; known prior outer exposure inherited, not old outer-performance tables'))
    pd.DataFrame(sources).to_csv(HERE/'analysis/report_sources.csv',index=False)
    for base in [ROOT/'mainline',ROOT/'artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained',ROOT/'artifacts/phase3c_ssl_adaptation_20260930/runs/adapt/p100',ROOT/'artifacts/phase3c_ssl_adaptation_20260930/oof']:
      for p in base.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
    write_json(target,dict(git_start_sha=gitsha,sha256=protected,raw_IMU_signal_values_read=False,whole_cache_decoded=False,prior_outer_exposure_permanently_inherited=True))
    source={str(p.relative_to(HERE)):sha(p) for p in list((HERE/'scripts').glob('*.py'))+[HERE/'PROTOCOL.md']}
    write_json(HERE/'configs/ANALYSIS_FREEZE.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_start_sha=gitsha,source_sha256=source,known_prior_results=True,new_phase5_validation_statistics_seen=False,deep_training_budget=0,all_new_fits=0,bootstrap_seed=SEED,bootstrap_draws=10000,BH_families=dict(A_primary=4,A_correlations=36,A_coverage=3,B_mechanism=4),H3_representation_if_assets_missing='BLOCKED; do not extract or fit'))
    write_json(HERE/'analysis/runtime_environment.json',dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','torch']},CPU_count=os.cpu_count(),threads=4,neural_model_instantiations=0,new_training=0,new_fits=0,dependency_installs=0,formal_environment_changed=False))
    guard();print(json.dumps(dict(status='PHASE5_LOCKED',protected_files=len(protected),source_files=len(source),reports_read=len(sources),git=gitsha)),flush=True)

if __name__=='__main__':main()
