from common import *
import torch
from patch_extractor import patch_starts,extract_patches

def main():
    rows=[]
    for length,expected in [(0,[]),(1,[]),(199,[]),(200,[0]),(201,[0,1]),(300,[0,100]),(976,list(range(0,701,100))+[776]),(2000,list(range(0,1801,100)))]:
        require(patch_starts(length)==expected,'Patch starts/count differ')
        x=torch.arange(6*max(length+15,215),dtype=torch.float32).reshape(1,6,-1)
        patches,mask,starts=extract_patches(x,[length])
        require(int(mask.sum())==len(expected),'Wrong mask count')
        require(starts[0,mask[0]].tolist()==expected,'Returned starts differ')
        for j,start in enumerate(expected):require(torch.equal(patches[0,j],x[0,:,start:start+200]),'Patch contents differ')
        require(len(expected)==len(set(expected)),'Duplicate tail')
        require(not expected or expected[0]==0 and expected[-1]+200==length,'Coverage/edge differs')
        y=x.clone();y[...,length:]=float('nan');out=extract_patches(y,[length]);require(torch.equal(out[0],patches),'Padding contamination')
        require(torch.equal(extract_patches(x,[length])[0],patches),'Nondeterministic')
        rows.append(dict(length=length,count=len(expected),starts=expected,status='PASS'))
    x=torch.randn(4,6,2100,requires_grad=True);p,m,s=extract_patches(x,[976,2000,199,201])
    require(p.shape==(4,19,6,200) and m.sum(1).tolist()==[9,19,0,2],'Ragged batch mask/count')
    require(torch.equal(p[~m],torch.zeros_like(p[~m])),'Invalid patches not zero')
    p.sum().backward();require(torch.isfinite(x.grad).all(),'Extraction gradient nonfinite')
    for row,length in enumerate([976,2000,199,201]):require(not x.grad[row,:,length:].any(),'Padding acquired gradient')
    for wrong in [-1,2101]:
        try:extract_patches(x,[wrong]*4)
        except ValueError:pass
        else:raise AssertionError('Invalid length accepted')
    result=dict(status='PASS',patch_length=200,stride=100,shared_length_contract=True,cases=rows,ragged_padding_zero=True,tail_unique_covered=True,deterministic=True,padded_sample_unchanged=True,autograd_padding_zero=True,outer_access=False)
    write_json(HERE/'analysis/extraction_tests.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
