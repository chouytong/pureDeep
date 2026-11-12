"""Recover reports only after completed statistics/timing, with existing plot dependencies."""
from common_l2 import *
import os

def main():
 assert load(HERE/'analysis/analysis_integrity.json')['status']=='PASS_FOUR_CELL_FIXED_ANALYSIS' and load(HERE/'analysis/cost_integrity.json')['status']=='PASS_MEASURED_BOUNDARIES'
 for name in ['ANALYSIS','COST']:
  for p,h in load(HERE/'configs'/f'{name}_FREEZE.json')['files'].items():assert sha(HERE/p)==h
 for kind in ['wssl','str']:assert load(HERE/'analysis'/f'{kind}_training_integrity.json')['runs']==45
 deps=Path('/home/zyt/envs/patch_report_deps_20261008');assert (deps/'matplotlib/__init__.py').exists();sys.path.append(str(deps));os.environ['MPLCONFIGDIR']=str(PRIVATE/'manifests/mpl_runtime_cache')
 import matplotlib
 write(HERE/'audit/report_recovery.json',dict(status='EXISTING_PLOTTING_DEPENDENCY_RECOVERY',initial_error='ModuleNotFoundError matplotlib in report import after all statistics and90 timings completed',failed_attempt_log='logs/analyze_attempt1.log',preserved_failed_sources=['logs/reports_l2_missing_matplotlib_v1.py','logs/analyze_l2_completed_statistics_v1.py'],recovery='report generation only; no repeated statistics, training or model timing',additional_model_fits=0,dependencies_installed_or_changed=False,process_local_existing_dependency_path=str(deps),matplotlib_version=matplotlib.__version__,matplotlib_source_sha256=sha(deps/'matplotlib/__init__.py'),numpy_version=np.__version__,pandas_version=pd.__version__))
 from reports_l2 import main as reports_main
 reports_main();print('REPORT_ONLY_RECOVERY_COMPLETE',flush=True)
if __name__=='__main__':main()
