import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from Properties.model.genotypes import RNN_PRIMITIVES
from .model import RNN_Cell, RNN_Model


class RNN_Cell_Search(RNN_Cell):
    def __init__(self, ninp, nhid, dropouth, dropoutx, rnn_steps):
        super(RNN_Cell_Search, self).__init__(ninp, nhid, dropouth, dropoutx, rnn_steps,
            genotype=None)
        self._steps = rnn_steps
        self._ln = nn.LayerNorm(nhid, elementwise_affine=False)

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        s0 = self._ln(s0)
        probs = self.prod_hot_vec(self._arch_weights)

        offset = 0
        states = s0.unsqueeze(0)
        for i in range(self._steps):
            if self.training:
                masked_states = states * h_mask.unsqueeze(0)
            else:
                masked_states = states
            ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i + 1,
                -1, 2*self.nhid)
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            s = torch.zeros_like(s0)
            for k, name in enumerate(RNN_PRIMITIVES):
                if name == 'none':
                    continue
                fn = self._get_activation(name)
                unweighted = states + c * (fn(h) - states)
                s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(
                    -1) * unweighted, dim=0)

            s = self._ln(s)
            states = torch.cat([states, s.unsqueeze(0)], dim=0)
            offset += i + 1
        output = torch.mean(states[-self._steps:], dim=0)
        return output

    def prod_hot_vec(self, weights, temperature=1):
        return F.gumbel_softmax(weights, tau=temperature, hard=True, dim=-1)


class RNN_Model_Search(RNN_Model):
    def __init__(self, arch_params, *args):
        super(RNN_Model_Search, self).__init__(*args, cell_cls=RNN_Cell_Search,
            genotype=None)
        self._args = args
        self._arch_weights = arch_params
        self._init_arch_params(arch_params)

    def _init_arch_params(self, arch_weights):
        for rnn in self.rnns:
            rnn._arch_weights = arch_weights

    def arch_parameters(self):
        return [self._arch_weights]
