"""Softmax CE, effective-number weighted CE (Cui 2019), focal (Lin 2017)."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def effective_weights(counts, beta):
    if not 0 <= beta < 1: raise ValueError('beta must be in [0, 1)')
    counts=torch.as_tensor(counts,dtype=torch.float64)
    if counts.ndim != 1 or (counts < 0).any() or not (counts > 0).any(): raise ValueError('Invalid training counts')
    present=counts > 0
    weights=torch.zeros_like(counts)
    weights[present]=1. if beta == 0 else (1-beta)/(-torch.expm1(counts[present]*math.log(beta)))
    weights[present] *= present.sum()/weights[present].sum()
    return weights.float()

class ClassBalancedLoss(nn.Module):
    def __init__(self, counts, beta=.999):
        super().__init__();self.register_buffer('weights',effective_weights(counts,beta))
    def forward(self, logits, target):
        # Explicit sample mean, not PyTorch's batch-dependent weighted-CE denominator.
        return (F.cross_entropy(logits,target,reduction='none')*self.weights[target]).mean()

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.):
        super().__init__()
        if gamma < 0: raise ValueError('gamma must be nonnegative')
        self.gamma=gamma
    def forward(self,logits,target):
        logpt=F.log_softmax(logits,dim=1).gather(1,target[:,None]).squeeze(1)
        return (-logpt * (-torch.expm1(logpt)).clamp_min(0).pow(self.gamma)).mean()

def make_loss(name,counts,beta=.999,gamma=2.):
    if name=='cross-entropy':return nn.CrossEntropyLoss()
    if name=='class-balanced':return ClassBalancedLoss(counts,beta)
    if name=='focal':return FocalLoss(gamma)
    raise ValueError(name)
