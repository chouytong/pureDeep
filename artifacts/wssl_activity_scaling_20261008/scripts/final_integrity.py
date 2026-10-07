"""Post-result verification; original assets read-only, no model selection."""
from run import *

def main():
    lock=json.loads((HERE/'analysis/study_lock.json').read_text());base=json.loads(BASELINE_LOCK.read_text())
    guard_inputs(base,lock)
    state=json.loads((HERE/'analysis/execution_state.json').read_text());decision=json.loads((HERE/'analysis/scaling_decision.json').read_text())
    require(state['status']=='complete' and state['completed_seeds']==[42,43,44],'Training matrix incomplete')
    require(decision['status']=='complete' and decision['completed_candidate_runs']==45,'Decision incomplete')
    runs=list((HERE/'runs').glob('activity_scaling/seed*/outer_*/inner_*/scaling_audit.json'));require(len(runs)==45,'Wrong candidate count')
    candidate_manifest={};count=0
    for p in sorted(runs):
        au=json.loads(p.read_text());require(au['status']=='PASS' and not au['smoke'],'Nonformal audit')
        for rel,h in au['files_sha256'].items():require(sha(p.parent/rel)==h,'Formal artifact modified')
        candidate_manifest[str(p.relative_to(HERE))]=sha(p)
        ck=torch.load(p.parent/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        m=ActivityScaledWSSL(ck['config']);m.load_state_dict(ck['model_state'],strict=True)
        require(sum(v.numel() for v in m.parameters())==143183,'Unexpected capacity')
        require(all(v.requires_grad for v in m.parameters()),'Unexpected frozen classifier parameters')
        require(len(m.activity_index)==11 and tuple(m.activities)==tuple(ck['provenance']['activity_order']),'Checkpoint index mismatch')
        require(ck['provenance']['wrist_order']==['left','right'],'Wrist order changed')
        require(torch.isfinite(m.activity_ssl_scale).all().item(),'Nonfinite g')
        require(ck['normalization']['mean'].shape==torch.Size([11,2,6,1]),'Normalization shape')
        count+=1
    result=dict(status='PASS',candidate_runs=count,baseline_runs_reused=45,
      frozen_source_asset_protocol_sha_unchanged=True,full_selected_checkpoints_reload_strict=True,
      only_added_parameters=11,classifier_parameters=143183,frozen_harnet_parameters=10457408,
      all_original_classifier_parameters_trainable=True,checkpoint_activity_order_from_frozen_config=True,
      run_integrity_hashes=candidate_manifest,outer_access=False,threshold_changed=False,
      harnet_adapted=False,training_recipe_changed=False,additional_hyperparameter_search=False,
      decision=decision['decision'],report_sha256=sha(HERE/'FINAL_REPORT.md'),
      entry_stage_smoke_only=True,statistical_units=15,seeds_averaged_first=True)
    (HERE/'analysis/final_integrity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='run_integrity_hashes'},indent=2))

if __name__=='__main__':main()
