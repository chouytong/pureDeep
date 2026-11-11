"""Reconstruct existing prep batch membership from audited producer ordering; IDs private."""
from common9 import *
def main():
 rs=records();logs=load(PROC/'solver_audit.json');batches=[];k=0
 for activity in ACTS:
  items=sorted([r for r in rs if r['activity']==activity],key=lambda r:(r['subject_id'],SIDES.index(r['wrist'])))
  for first in range(0,len(items),32):
   group=items[first:first+32];old=logs[k];assert old['activity']==activity and old['file_count']==len(group) and old['raw_length']==group[0]['metadata_rows'];batches.append(dict(batch_index=k,activity=activity,iterations=old['iterations'],subject_ids=[r['subject_id'] for r in group],wrist_count=len(group)));k+=1
 assert k==275;write(HERE/'restricted_manifests/solver_batch_membership.json',dict(batches=batches,source_order='peractivity subject sorted; left thenright; batch32files; grouping by length; verified logs'))
 splits=load(MAIN/'splits/pads_classification/v3_nested_cv/seed42/nested_cv_splits.json');rows=[]
 for outer in splits['outer']:
  for inner in outer['inner_folds']:
   fit=set(map(str,inner['train_subjects']));mixed=[b for b in batches if any(s in fit for s in b['subject_ids']) and any(s not in fit for s in b['subject_ids'])];totaltrain=sum(sum(s in fit for s in b['subject_ids']) for b in batches);mixedtrain=sum(sum(s in fit for s in b['subject_ids']) for b in mixed)
   rows.append(dict(context=outer['outer_fold'],inner=inner['inner_fold'],existing_preprocessing_batches=275,batches_with_train_and_nontrain_members=len(mixed),inner_train_wrist_records=totaltrain,train_wrist_records_sharing_stop_rule_with_nontrain=mixedtrain,shared_fraction=mixedtrain/totaltrain,raw_signal_role_conditioned_metric_computed=False,interpretation='membership-only numerical dependency audit; no outcome/outer-performance analysis'))
 csv(pd.DataFrame(rows),'solver_batch_split_dependency.csv');write(HERE/'analysis/solver_batch_lineage_integrity.json',dict(status='PASS_CONFIRMED_PRE_SPLIT_NUMERICAL_COUPLING',batches=275,paired_development_splits=15,source_order_matches_original_solver_logs=True,raw_preparation_before_subject_split=True,batch_stop_criterion_depends_on_all_series=True,labels_used_by_solver=False,label_leakage_demonstrated=False,strict_record_local_processing_claim_supported=False,all_membership_ids_private=True,new_model_training=0))
 # Preserve the complete public scientific draft before correcting conclusions.
 archive=HERE/'audit/pre_solver_witness_snapshot';archive.mkdir(exist_ok=False);manifest=load(HERE/'analysis/source_output_manifest.json');snapshot=[]
 for rec in manifest['files']+[dict(path='analysis/source_output_manifest.json',sha256=sha(HERE/'analysis/source_output_manifest.json'))]:
  source=HERE/rec['path'];assert sha(source)==rec['sha256'];target=archive/rec['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());snapshot.append(dict(path=rec['path'],sha256=rec['sha256']))
 write(HERE/'configs/PREPUBLICATION_REVISION.json',dict(status='NUMERICAL_WITNESS_REVISION_BEFORE_FIRST_PUBLICATION',snapshot_prefix='audit/pre_solver_witness_snapshot',snapshot_files=snapshot,superseded_freezes=['CLOSURE_FREEZE.json'],all_other_frozen_sources_remain_active=True,official_start=START,preliminary_unpublished_local_commit='9fd6adbbfdbb48919893d855a5a45610cefd603a',protected_prior_phase_assets_unchanged=True,reason='new deterministic convex objective witnesses and batch-dependence proof supersede draft assumption of record-local accurate L1 preprocessing'))
 print('PASS_BATCH_LINEAGE_AND_DRAFT_PRESERVATION')
if __name__=='__main__':main()
