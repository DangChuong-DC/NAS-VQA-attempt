import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn_utils import LockedDrop, mask2d


class RNN_Cell(nn.Module):
    def __init__(self, ninp, nhid, dropouth, dropoutx, rnn_steps, genotype):
        super(RNN_Cell, self).__init__()
        self.nhid = nhid
        self.droph = dropouth
        self.dropx = dropoutx

        if genotype is not None:
            assert rnn_steps == len(genotype.rnn_gene), 'Provided gene and number of steps in cell are NOT matched'
            self.gene = genotype.rnn_gene
        INITRANGE = 1.0 / math.sqrt(self.nhid)
        self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2*nhid).uniform_(-INITRANGE,
            INITRANGE))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE))
                for i in range(rnn_steps)
                ])

    def forward(self, inputs, hidden):
        B, T = inputs.size(0), inputs.size(1)

        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1. - self.dropx)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1. - self.droph)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []
        for t in range(T):
            hidden = self.cell(inputs[:, t, :], hidden, x_mask, h_mask)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens, dim=1)

        return hiddens, hiddens[:, -1, :].unsqueeze(0)

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def _get_activation(self, name):
        if name == 'tanh':
            f = torch.tanh
        elif name == 'relu':
            f = F.relu
        elif name == 'sigmoid':
            f = torch.sigmoid
        elif name == 'identity':
            f = lambda x: x
        else:
            raise NotImplementedError

        return f

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        states = [s0]
        for i, (name, idx) in enumerate(self.gene):
            s_prev = states[idx]
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(name)
            h = fn(h)
            s = s_prev + c*(h - s_prev)
            states.append(s)
        output = torch.mean(torch.stack(states[1:], dim=0), dim=0)

        return output


class RNN_Model(nn.Module):
    def __init__(self, ninp, nhid, dropouti=0.1, dropouth=0.5, dropoutx=0.5,
            rnn_steps=3, cell_cls=RNN_Cell, genotype=None):
        super(RNN_Model, self).__init__()
        self.lockdrop = None
        if dropouti > 0.:
            self.lockdrop = LockedDrop(dropouti)

        if cell_cls == RNN_Cell:
            assert genotype is not None
            rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, rnn_steps, genotype)]
        else:
            assert genotype is None
            rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, rnn_steps)]

        self.rnns = nn.ModuleList(rnns)

        self.ninp = ninp
        self.nhid = nhid

    def forward(self, input, hidden=None):
        B = input.size(0)
        if self.lockdrop is not None:
            input = self.lockdrop(input)

        if hidden is None:
            hidden = torch.zeros(1, B, self.nhid, dtype=input.dtype,
                device=input.device)
        else:
            hidden = hidden.clone().detach()

        for l, rnn in enumerate(self.rnns):
            output, new_hidden = rnn(input, hidden)

        return output, new_hidden
