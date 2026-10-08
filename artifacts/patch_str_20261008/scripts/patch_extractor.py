"""Fixed 200/100 complete patches with deterministic unique tail coverage."""
import torch

PATCH_LENGTH=200
STRIDE=100

def patch_starts(true_length):
    length=int(true_length)
    if length<0:raise ValueError('Negative true length')
    if length<PATCH_LENGTH:return []
    starts=list(range(0,length-PATCH_LENGTH+1,STRIDE))
    tail=length-PATCH_LENGTH
    if starts[-1]!=tail:starts.append(tail)
    return starts


def extract_patches(signal, lengths=None):
    if signal.ndim!=3 or signal.shape[1]!=6:raise ValueError('Expected [valid wrists,6,padded samples]')
    batch,channels,time=signal.shape
    values=[time]*batch if lengths is None else list(map(int,torch.as_tensor(lengths).tolist()))
    if len(values)!=batch or any(v<0 or v>time for v in values):raise ValueError('True length outside container')
    lists=[patch_starts(v) for v in values];n=max(1,max(map(len,lists),default=0))
    starts=torch.full((batch,n),-1,device=signal.device,dtype=torch.long)
    for row,seq in enumerate(lists):
        if seq:starts[row,:len(seq)]=torch.tensor(seq,device=signal.device)
    mask=starts>=0
    owners,columns=mask.nonzero(as_tuple=True)
    output=signal.new_zeros((batch*n,channels,PATCH_LENGTH))
    if owners.numel():
        positions=starts[owners,columns,None]+torch.arange(PATCH_LENGTH,device=signal.device)[None]
        # Only valid patch positions are indexed; no sample beyond true_length is read.
        valid=signal[owners[:,None,None],torch.arange(channels,device=signal.device)[None,:,None],positions[:,None,:]]
        output=output.index_copy(0,owners*n+columns,valid)
    return output.reshape(batch,n,channels,PATCH_LENGTH),mask,starts
