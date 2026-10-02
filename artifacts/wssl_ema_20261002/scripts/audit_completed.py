"""Technical audit before fixed-point statistical conclusions."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd,torch
H=Path(__file__).resolve().parent.parent;P4=Path('/home/zyt/deep_final/artifacts/phase4_window_dynamics_20261002');B=Path('/home/zyt/deep_final/artifacts/phase3b_external_ssl_20260928/runs/b_str_pretrained')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--item',type=int,choices=[3,4],required=True);args=ap.parse_args();lock=json.loads((P4/'analysis/baseline_lock.json').read_text());rows=[]
 for s in lock['stages']:
  seed,oi,ii=s['seed'],s['context'],s['inner'];stage=H/f'runs/ordinary/seed{seed}/outer_{oi}/inner_{ii}';ema=H/f'runs/ema/seed{seed}/outer_{oi}/inner_{ii}';old=B/f'seed{seed}/outer_{oi}/inner_{ii}'
  au=json.loads((stage/'ordinary_trajectory_audit.json').read_text());es=json.loads((ema/'stage_status.json').read_text());assert es['status']=='complete' and es['outer_test_loader_created'] is False
  assert au['ordinary_full_trajectory_exact'] and au['ordinary_predictions_exact'] and not au['smoke']
  ck=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False);ec=torch.load(stage/'checkpoints/ema_at_ordinary_best.pt',map_location='cpu',weights_only=False);bc=torch.load(old/'checkpoints/best.pt',map_location='cpu',weights_only=False)
  assert ck['epoch']==ec['epoch']==bc['epoch'];assert ck['model_state'].keys()==ec['model_state'].keys()==bc['model_state'].keys()
  assert all(torch.equal(ck['model_state'][k],bc['model_state'][k]) for k in bc['model_state'])
  assert all(torch.isfinite(v).all() for v in ec['model_state'].values())
  for key in ('mean','std'):assert torch.equal(ck['normalization'][key],ec['normalization'][key]) and torch.equal(ck['normalization'][key],bc['normalization'][key])
  assert au['updates']==au['stop_epoch']*au['steps_per_epoch'];assert au['best_updates']==au['best_epoch']*au['steps_per_epoch']
  rows.append(dict(seed=seed,context=oi,inner=ii,best_epoch=au['best_epoch'],stop_epoch=au['stop_epoch'],updates=au['updates'],selected_updates=au['best_updates'],S=au['steps_per_epoch'],alpha=au['alpha'],ordinary_weights_exact=True,ordinary_trajectory_exact=True,checkpoint_epoch_identical=True))
 study_lock=json.loads((H/'analysis/ema_protocol_lock.json').read_text())
 for p,d in study_lock['files'].items():assert sha(H/p)==d,'Pre-training protocol/core script changed'
 for p,d in lock['source_sha256'].items():assert sha(p)==d
 assert sha(lock['cache_path'])==lock['cache_sha256'];assert sha(lock['split_path'])==lock['split_sha256']
 for s in lock['stages']:
  stage=B/f"seed{s['seed']}/outer_{s['context']}/inner_{s['inner']}"
  assert sha(stage/'checkpoints/best.pt')==s['checkpoint_sha256']
  assert sha(stage/'predictions/validation.csv')==s['prediction_sha256']
 pd.DataFrame(rows).to_csv(H/f'analysis/item0{args.item}_45run_audit.csv',index=False)
 r=dict(status='PASS',item=args.item,ordinary_runs=45,matched_ema_checkpoints=45,updates=sum(row['updates'] for row in rows),ordinary_weights_bit_exact=True,ordinary_epoch_metrics_exact=True,ordinary_validation_predictions_exact=True,selected_checkpoint_epoch_match=True,frozen_sources_cache_splits_unchanged=True,outer_access=False)
 (H/f'analysis/item0{args.item}_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
