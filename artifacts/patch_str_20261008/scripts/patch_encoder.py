"""Shared lightweight learned patch encoder. No normalization/dropout/FFT."""
import torch
from torch import nn

class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv1d(6,6,15,padding=7,groups=6),nn.Conv1d(6,32,1),nn.GELU(),
          nn.Conv1d(32,32,7,padding=3,groups=32),nn.Conv1d(32,64,1),nn.GELU())
    def forward(self,patches,mask):
        if patches.ndim!=4 or tuple(patches.shape[2:])!=(6,200) or mask.shape!=patches.shape[:2]:raise ValueError('Invalid patch shape')
        b,n=mask.shape;idx=mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        z=patches.new_zeros((b*n,64))
        if idx.numel():
            valid=patches.reshape(b*n,6,200).index_select(0,idx)
            z=z.index_copy(0,idx,self.layers(valid).mean(-1))
        return z.reshape(b,n,64)
