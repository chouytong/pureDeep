"""Only prespecified positive-trend gates permit frozen inference descriptions."""
import gc
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from common import *
from patch_models import build_patch
from run_condition import study_guard
from test_models import dataset
from src.datasets.subject_activity import collate_subject_activities
from src.metrics.classification import classification_metrics


def main():
    torch.set_num_threads(4)
    decision=json.loads((HERE/'analysis/decision.json').read_text())
    lock=json.loads((HERE/'analysis/f0_lock.json').read_text());study=json.loads((HERE/'analysis/study_lock.json').read_text())
    guard(lock);study_guard(study)
    allowed=[c for c,key in [('P1','p1_utilization_allowed'),('P2','p2_order_shuffle_allowed')] if decision[key]]
    if not allowed:
        write_json(HERE/'analysis/conditional_diagnostics.json',dict(status='SKIPPED',reason='Prespecified positive-trend gates not met',outer_access=False,training=False));return
    norms=[];attention=[];metrics=[]
    for condition in allowed:
        for item in lock['stages']:
            seed,c,i=item['seed'],item['context'],item['inner'];stage=HERE/f'runs/{condition}/seed{seed}/outer_{c}/inner_{i}'
            ck=torch.load(stage/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            frame=pd.read_csv(stage/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id')
            p0=pd.read_csv(Path(item['path'])/'predictions/validation.csv',dtype={'subject_id':str}).set_index('subject_id')
            ds=dataset(ck,frame.index.tolist(),'validation');loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate_subject_activities)
            model=build_patch(ck['config'],condition).cuda().eval();model.load_state_dict(ck['model_state']);branch=model.wrist_encoder.branch
            accum={};all_ids=[];all_prob=[];all_y=[]
            with torch.no_grad():
                for batch in loader:
                    x,wm,am,lens=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
                    probability=model(x,wm,am,lens)['probabilities'].cpu().numpy();all_prob.extend(probability);all_ids.extend(batch['subject_id']);all_y.extend(batch['y'].tolist())
                    for a,name in enumerate(ck['config']['data']['activities']):
                        for wrist in range(2):
                            valid=am[:,a]&wm[:,a,wrist].bool()
                            for length in torch.unique(lens[valid,a]):
                                selected=valid&(lens[:,a]==length);signal=x[selected,a,wrist,:,:int(length)]
                                if not len(signal):continue
                                output=model.wrist_encoder(signal);ratio=output['patch_residual'].norm(dim=1)/output['bag_embedding'].sub(output['patch_residual']).norm(dim=1).clamp_min(1e-12)
                                key=(name,wrist,int(length));rec=accum.setdefault(key,dict(ratio=[],attention=[]))
                                rec['ratio'].extend(ratio.cpu().tolist());rec['attention'].extend(output['patch_attention'].cpu().tolist())
            matched=frame.loc[all_ids];prob=np.asarray(all_prob)
            require(matched.target.tolist()==all_y and np.max(abs(prob-matched[['probability_pd','probability_dd']].to_numpy()))<=1e-12,'Diagnostic inference not reproduced')
            effect=prob[:,1]-p0.loc[all_ids].probability_dd.to_numpy()
            private=stage/'predictions/patch_p0_probability_effect.csv'
            pd.DataFrame(dict(subject_id=all_ids,target=all_y,probability_effect=effect)).to_csv(private,index=False)
            for (activity,wrist,length),rec in accum.items():
                norms.append(dict(condition=condition,seed=seed,context=c,inner=i,activity=activity,wrist=['Left','Right'][wrist],length=length,mean_norm_ratio=np.mean(rec['ratio']),median_norm_ratio=np.median(rec['ratio'])))
                average=np.asarray(rec['attention']).mean(0)
                for position,value in enumerate(average):attention.append(dict(condition=condition,seed=seed,context=c,inner=i,activity=activity,wrist=['Left','Right'][wrist],length=length,patch_index=position,mean_attention=value))
            if condition=='P2':
                original=branch.aggregate;generator=torch.Generator().manual_seed(20261008)
                def shuffled(z,mask):
                    altered=z.clone()
                    for row in range(len(z)):
                        idx=mask[row].nonzero().flatten();order=torch.randperm(len(idx),generator=generator).to(z.device)
                        altered[row,idx]=z[row,idx[order]]
                    return original(altered,mask)
                branch.aggregate=shuffled;shuffle_prob=[]
                with torch.no_grad():
                    for batch in loader:
                        output=model(*(batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']))
                        shuffle_prob.extend(output['probabilities'].cpu().numpy())
                branch.aggregate=original;shuffle_prob=np.asarray(shuffle_prob)
                for mode,p in [('full',prob),('shuffled',shuffle_prob)]:
                    value=classification_metrics(np.asarray(all_y),p.argmax(1),2,probabilities=p)
                    metrics.append(dict(seed=seed,context=c,inner=i,mode=mode,accuracy=value['accuracy'],ba=value['balanced_accuracy'],auroc=value['macro_auroc'],macro_f1=value['macro_f1'],pd_recall=value['per_class_recall'][0],dd_recall=value['per_class_recall'][1]))
            del model,ck,ds,loader;gc.collect()
    for name,rows in [('branch_norm_ratio',norms),('patch_attention',attention),('order_shuffle_runs',metrics)]:
        if rows:pd.DataFrame(rows).to_csv(HERE/f'analysis/{name}.csv',index=False)
    if metrics:
        units=pd.DataFrame(metrics).groupby(['context','inner','mode'],as_index=False).mean(numeric_only=True);units.to_csv(HERE/'analysis/order_shuffle_15split.csv',index=False)
    write_json(HERE/'analysis/conditional_diagnostics.json',dict(status='complete',conditions=allowed,shuffle_seed=20261008,training=False,outer_access=False,model_selection=False,dd_subtype_eligible=decision['dd_subtype_analysis_allowed']))
    guard(lock);study_guard(study)

if __name__=='__main__':main()
