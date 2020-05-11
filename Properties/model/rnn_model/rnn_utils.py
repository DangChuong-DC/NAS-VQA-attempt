import torch
import torch.nn as nn


class LockedDrop(nn.Module):
    def __init__(self, dropout):
        super(LockedDrop, self).__init__()
        self.drop_rate = dropout

    def forward(self, x):
        if not self.training:
            return x
        m = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1. - self.drop_rate)
        mask = m.div_(1. - self.drop_rate).requires_grad_(False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob, cuda=True):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = m.requires_grad_(False)
    if cuda:
        m = m.cuda()
    return m
