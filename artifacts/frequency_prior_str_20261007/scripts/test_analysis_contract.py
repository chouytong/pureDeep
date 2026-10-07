"""Known-answer metric, positive-class, tie and multiplicity checks before results."""
import ast
import json
import numpy as np
from common import *
from analyze_results import bh
from src.metrics.classification import classification_metrics


def main():
    for f in (HERE/'scripts').glob('*.py'):ast.parse(f.read_text(),filename=str(f))
    require(np.allclose(bh([.01,.04,.03]),[.03,.04,.04]),'BH known-answer mismatch')
    probability=np.array([[.8,.2],[.4,.6],[.3,.7],[.5,.5]])
    labels=np.array([0,0,1,1]);prediction=probability.argmax(1)
    metrics=classification_metrics(labels,prediction,2,probabilities=probability)
    require(prediction.tolist()==[0,1,1,0],'Tie must be PD')
    require(metrics['balanced_accuracy']==.5 and np.allclose(metrics['per_class_recall'],[.5,.5]),'BA/class direction mismatch')
    # DD(.7,.5) versus PD(.2,.6): .7 beats both; .5 beats only .2 =>3/4.
    require(metrics['macro_auroc']==.75,'AUROC known-answer mismatch')
    for seed in [42,43,44]:config_for(seed)
    report=dict(status='PASS',all_script_syntax=True,imports=True,original_recipe_all_seeds=True,
        handchecked_dd_direction_argmax_tie_ba_auroc=True,bh_known_case=True,
        preliminary_assertion_correction='Temporary manual check expected AUROC .5 incorrectly; corrected to .75 (3/4pairs), production metric implementation unchanged.',
        selection_performed=False,outer_access=False)
    write_json(HERE/'analysis/preformal_static_checks.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
