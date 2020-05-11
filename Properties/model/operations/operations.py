import torch
import torch.nn as nn

from .ops_comp import _SA, _GA, _FF
from .ops_utils import MaskDrop


OPS = {
    'none': lambda cfgs: Zeros(cfgs),
    'feed_forward': lambda cfgs: FeedForward(cfgs),
    'skip_connect': lambda cfgs: Identity(cfgs),
    'self_attn': lambda cfgs: SelfAttn(cfgs),
    'guided_attn': lambda cfgs: GuidedAttn(cfgs),
}


class Zeros(nn.Module):
    def __init__(self, __C):
        super(Zeros, self).__init__()

    def forward(self, inputs, masks):
        x = inputs[0]
        b, c, d = x.size()

        if x.is_cuda:
            with torch.cuda.device(x.get_device()):
                padding = torch.cuda.FloatTensor(b, c, d).fill_(0)
        else:
            padding = torch.FloatTensor(b, c, d).fill_(0)

        return padding


class Identity(nn.Module):
    def __init__(self, __C):
        super(Identity, self).__init__()

    def forward(self, inputs, masks):
        return inputs[0]


class SelfAttn(nn.Module):
    def __init__(self, __C):
        super(SelfAttn, self).__init__()
        self._sa = _SA(__C)
        self.use_mask = False
        if __C.MASK_DROP_R > 0.:
            self.use_mask = True
            self._mask = MaskDrop(__C.MASK_DROP_R)

    def forward(self, inputs, masks):
        x = inputs[0]
        x_mask = masks[0]
        if self.use_mask:
            x = self._mask(x)

        return self._sa(x, x_mask)


class GuidedAttn(nn.Module):
    def __init__(self, __C):
        super(GuidedAttn, self).__init__()
        self._ga = _GA(__C)
        self.use_mask = False
        if __C.MASK_DROP_R > 0.:
            self.use_mask = True
            self._mask = MaskDrop(__C.MASK_DROP_R)

    def forward(self, inputs, masks):
        x, y = inputs
        x_mask, y_mask = masks
        if self.use_mask:
            x = self._mask(x)
            y = self._mask(y)

        return self._ga(x, y, x_mask, y_mask)


class FeedForward(nn.Module):
    def __init__(self, __C):
        super(FeedForward, self).__init__()
        self._ff = _FF(__C)
        self.use_mask = False
        if __C.MASK_DROP_R > 0.:
            self.use_mask = True
            self._mask = MaskDrop(__C.MASK_DROP_R)

    def forward(self, inputs, masks):
        x = inputs[0]
        if self.use_mask:
            x = self._mask(x)

        return self._ff(x)


####################################
### Pre-process function of cell ###
####################################
class PFFN(nn.Module):
    def __init__(self, __C):
        super(PFFN, self).__init__()
        self.ff = _FF(__C)
        self.use_mask = False
        if __C.MASK_DROP_R > 0.:
            self.use_mask = True
            self.maskdrop = MaskDrop(__C.MASK_DROP_R)

    def forward(self, x):
        if self.use_mask:
            x = self.maskdrop(x)
        x = self.ff(x)

        return x
