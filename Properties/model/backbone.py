import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations.operations import *
from .operations.ops_utils import feats_sum, feats_mean


class Step(nn.Module):
    def __init__(self, __C, i_dna, q_dna):
        super(Step, self).__init__()
        i_op, i_indx = i_dna
        q_op, q_indx = q_dna
        self._compile(__C, i_op, q_op, i_indx, q_indx)

    def _compile(self, __C, i_op, q_op, i_indx, q_indx):
        self._i_op = OPS[i_op](__C)
        self._q_op = OPS[q_op](__C)
        self._i_idx = i_indx
        self._q_idx = q_indx

    def forward(self, states, masks):
        i_state = states[self._i_idx]
        i_out = self._i_op(i_state, masks)

        q_state = states[self._q_idx][::-1]
        q_out = self._q_op(q_state, masks[::-1])

        return i_out, q_out


class Cell(nn.Module):
    def __init__(self, __C, genotype):
        super(Cell, self).__init__()
        self._num_step = __C.ATT_STEP
        self._init_pre_ops(__C)

        i_gene = genotype.img_gene
        q_gene = genotype.que_gene
        self._steps = nn.ModuleList([])
        for i in range(self._num_step):
            stp = Step(__C, i_gene[i], q_gene[i])
            self._steps.append(stp)

    def _init_pre_ops(self, __C):
        self._preproc0 = nn.ModuleList([
            PFFN(__C),
            PFFN(__C)
        ])
        self._preproc1 = nn.ModuleList([
            PFFN(__C),
            PFFN(__C)
        ])

    def _compute_init_state(self, f_pre_prev, f_prev):
        f_pre_prev = [pff(x) for pff, x in zip(self._preproc0, f_pre_prev)]
        f_prev = [pff(x) for pff, x in zip(self._preproc1, f_prev)]

        return feats_mean([f_pre_prev, f_prev])

    def forward(self, f_pre_prev, f_prev, masks):
        fts_0 = self._compute_init_state(f_pre_prev, f_prev)
        states = [fts_0]
        for i in range(self._num_step):
            fts = self._steps[i](states, masks)
            states.append(fts)

        c_out = feats_mean(states[-self._num_step:])

        return c_out


class Backbone(nn.Module):
    def __init__(self, __C, genotype):
        super(Backbone, self).__init__()
        self._num_lay = __C.ATT_LAYER
        self._cells = nn.ModuleList([])
        self._attn_decoders = nn.ModuleList([
            PFFN(__C),
            PFFN(__C)
        ])

        for i in range(self._num_lay):
            cell = Cell(__C, genotype)
            self._cells.append(cell)

    def forward(self, img, que, img_mask, que_mask):
        fts_0 = fts_1 = [img, que]
        masks = [img_mask, que_mask]

        for i, cell in enumerate(self._cells):
            fts_0, fts_1 = fts_1, cell(fts_0, fts_1, masks)
        fts_1 = [pff(x) for pff, x in zip(self._attn_decoders, fts_1)]

        return fts_1
