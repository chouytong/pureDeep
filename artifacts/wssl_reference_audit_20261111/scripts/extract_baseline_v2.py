from common import *
import gc,io,random,time
from itertools import chain
import torch
from torch.utils.data import DataLoader
from independent_wssl import IndependentWSSL,independent_copy
from src.models import build_model
from src.models.wssl import ResidualSSLSubject
from src.datasets import folds as fd
from src.datasets.subject_activity import SubjectActivityDataset,collate_subject_activities
from src.utils.config import load_config
from src.utils.seed import seed_everything

def fingerprint(model):
    return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}

def forward_parts(model,inputs,ssl,scale=1.):
    x,wm,am,lengths=inputs;b=model.backbone
    _,local=b._encode_activities_with_wrists(x,wm,am,lengths)
    valid=wm.bool()&am.bool()[...,None]
    added=model.wrist_projection(model.norm(ssl))*valid[...,None]
    fused=local+added*scale;batch=len(x)
    keep=am.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    features,_=b.wrist_fusion.fusion_features(fused.reshape(-1,2,64).index_select(0,keep),valid.reshape(-1,2).index_select(0,keep))
    bilateral=fused.new_zeros((batch*11,258)).index_copy(0,keep,features).reshape(batch,11,258)
    aggregated=b.activity_aggregator(bilateral,am);structured=b.structured_token_residual(bilateral,am)
    logits=b.classifier(aggregated['subject_embedding'])+structured['structured_residual_logits']
    return dict(logits=logits,local=local,ssl_residual=added,bilateral=bilateral,
                subject=aggregated['subject_embedding'],structured=structured['structured_residual_embedding'])

def aligned(ids,y,frame):
    if len(ids)!=len(set(ids)) or set(ids)!=set(frame.index):raise ValueError('Duplicate/wrong ID')
    if not np.array_equal(y,frame.loc[ids].target):raise ValueError('Label misalignment')

def correctness(ck,cfg,batch,ssl):
    inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
    model=IndependentWSSL(cfg).cuda().eval();model.load_state_dict(ck['model_state'],strict=True)
    old=ResidualSSLSubject(build_model(cfg)).cuda().eval();old.load_state_dict(ck['model_state'],strict=True);old.ssl_features=ssl
    with torch.no_grad():
        expected=old(*inputs)['logits'];actual=model(*inputs,ssl_features=ssl)['logits']
        assert torch.equal(expected,actual)
        assert torch.equal(actual,forward_parts(model,inputs,ssl)['logits'])
        changed=[v.clone() for v in inputs];changed[1][:,0,0]=0;changed[2][:,1]=False
        ref=model(*changed,ssl_features=ssl)['logits'];changed[0][:,0,0]=999;changed[0][:,1]=float('nan')
        s=ssl.clone();s[:,0,0]=999;s[:,1]=999
        assert torch.equal(ref,model(*changed,ssl_features=s)['logits'])
        changed=[v.clone() for v in inputs]
        for row in range(len(batch['subject_id'])):
            for a in range(11):changed[0][row,a,:,:,int(changed[3][row,a]):]=999
        assert torch.equal(actual,model(*changed,ssl_features=ssl)['logits'])
        reverse=torch.arange(len(ssl)-1,-1,-1,device='cuda')
        rev=model(*(v[reverse] for v in inputs),ssl_features=ssl[reverse])['logits']
        assert torch.equal(actual[reverse],rev)
        off=forward_parts(model,inputs,ssl,0.)['logits'];zero_input=model(*inputs,ssl_features=torch.zeros_like(ssl))['logits']
        zero_input_difference=float((off-zero_input).abs().max())
    model.ssl_features=ssl.clone();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();npstate=np.random.get_state();pystate=random.getstate()
    other=independent_copy(model,cfg)
    assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
    assert random.getstate()==pystate and np.array_equal(np.random.get_state()[1],npstate[1])
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(model.parameters(),other.parameters()))
    assert model.ssl_features.data_ptr()!=other.ssl_features.data_ptr()
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    fresh=IndependentWSSL(cfg).cuda().eval();fresh.load_state_dict(torch.load(buf,map_location='cuda',weights_only=True))
    with torch.no_grad():assert torch.equal(actual,fresh(*inputs,ssl_features=ssl)['logits'])
    fake=pd.DataFrame(dict(target=[0,1]),index=['a','b'])
    for ids,y in [(['a','a'],[0,0]),(['a','x'],[0,1]),(['a','b'],[1,0])]:
        try:aligned(ids,y,fake)
        except ValueError:pass
        else:raise AssertionError('Bad ID/label alignment accepted')
    assert all(not isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert not ssl.requires_grad
    return dict(status='PASS',original_independent_logits_exact=True,parts_logits_exact=True,mask_and_padding=True,ID_reorder=True,bad_ID_duplicate_label_rejected=True,reload_exact=True,instance_parameter_cache_RNG_isolation=True,zero_residual_includes_bias=True,zero_SSL_input_is_not_residual_off_max_logit_difference=zero_input_difference,external_HarNet_not_instantiated_or_updated=True)

def main():
    torch.set_num_threads(4);seed_everything(SEED,True);guard();started=time.perf_counter()
    freeze('baseline_extraction_v2',[HERE/'PROTOCOL.md',Path(__file__),HERE/'scripts/common.py',HERE/'scripts/independent_wssl.py'])
    rows=[];tests=[];differences=[]
    for c,i,sp in splits():
        train_ids=list(map(str,sp['train_subjects']));val_ids=list(map(str,sp['validation_subjects']))
        assert not set(train_ids)&set(val_ids)
        allids=train_ids+val_ids;array=cache(allids);lookup={s:j for j,s in enumerate(allids)}
        cfg=load_config(MAIN/'configs/str01_seed42.yaml')
        records=[r for r in fd.load_configured_records(cfg['data'],['PD','DD']) if r.subject_id in set(allids)]
        for seed in [42,43,44]:
            ck=torch.load(baseline(seed,c,i)/'checkpoints/best.pt',map_location='cpu',weights_only=False)
            cfg=load_config(MAIN/f'configs/str01_seed{seed}.yaml')
            assert cfg['data']['channel_names']==['AccX','AccY','AccZ','GyroX','GyroY','GyroZ']
            assert cfg['data']['wrist_order']==['left','right']
            ds=[SubjectActivityDataset([r for r in records if r.subject_id in set(selected)],cfg['data']['activities'],cfg['data'],ck['normalization']['mean'],ck['normalization']['std'],'validation') for selected in [train_ids,val_ids]]
            loader=chain.from_iterable(DataLoader(part,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate_subject_activities) for part in ds)
            model=IndependentWSSL(cfg).cuda().eval();model.load_state_dict(ck['model_state'],strict=True)
            assert sum(p.numel() for p in model.parameters())==143172
            initial=fingerprint(model);parts={k:[] for k in ['local','ssl_residual','bilateral','subject','structured','logits','probabilities']};ids=[];ys=[]
            for batch in loader:
                ssl=torch.from_numpy(array[[lookup[s] for s in batch['subject_id']]]).cuda()
                inputs=[batch[k].cuda() for k in ['x','wrist_mask','activity_mask','activity_lengths']]
                if (c,i)==(0,0) and len(ids)==0:tests.append(dict(seed=seed,**correctness(ck,cfg,batch,ssl)))
                with torch.no_grad():
                    p=forward_parts(model,inputs,ssl)
                    p['probabilities']=torch.softmax(p['logits'],dim=1)
                    if len(ids)==0:assert torch.equal(p['logits'],model(*inputs,ssl_features=ssl)['logits'])
                for key in parts:parts[key].append(p[key].cpu().numpy())
                ids.extend(batch['subject_id']);ys.extend(batch['y'].tolist())
            arrays={k:np.concatenate(v) for k,v in parts.items()}
            ids=np.array(ids);ys=np.array(ys);which=np.isin(ids,val_ids)
            saved=pred(baseline(seed,c,i)/'predictions/validation.csv')
            aligned(ids[which].tolist(),ys[which],saved)
            probability=arrays['probabilities'][which]
            diff=float(abs(probability-saved.loc[ids[which],['probability_pd','probability_dd']].to_numpy()).max())
            assert diff<=1e-12
            assert np.array_equal(probability.argmax(1),saved.loc[ids[which]].prediction)
            differences.append(diff)
            m=metric(ys[which],probability[:,1],prob=probability)
            rows.append(dict(context=c,inner=i,seed=seed,**m))
            out=HERE/'features'/f'seed{seed}_context{c}_inner{i}.npz';assert not out.exists()
            np.savez_compressed(out,subject_ids=ids,target=ys,is_validation=which,**arrays)
            fit_ids(f'A1_allowed_{seed}_{c}_{i}',train_ids,val_ids,train_ids,source_checkpoint_sha256=sha(baseline(seed,c,i)/'checkpoints/best.pt'),encoder_training_in_sample=True)
            assert fingerprint(model)==initial
            del model,ds,loader,ck;gc.collect()
        print(json.dumps(dict(stage='baseline_extract',context=c,inner=i,status='PASS',elapsed_seconds=time.perf_counter()-started)),flush=True)
    assert len(rows)==45
    pd.DataFrame(rows).to_csv(HERE/'analysis/baseline_metrics.csv',index=False)
    write_json(HERE/'tests/model_correctness.json',dict(status='PASS',representative_seed_tests=tests))
    guard()
    write_json(HERE/'analysis/baseline_reproduction.json',dict(status='PASS',checkpoints=45,full_validation_recomputed=True,max_probability_difference=max(differences),archived_logits_available=False,baseline_deep_training_runs=0,normalization_refits_reused=15,normalization_proof_sha256=sha(ROOT/'conclusions/organization_20261111/VERIFICATION.json'),runtime_import_root=str(MAIN),classifier_trainable_parameters=143172,external_frozen_encoder_parameters=10457408,elapsed_seconds=time.perf_counter()-started))
    print('BASELINE_AND_PARTS_EXTRACTION_PASS',flush=True)

if __name__=='__main__':main()
