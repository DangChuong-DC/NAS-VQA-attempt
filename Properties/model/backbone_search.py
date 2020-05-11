import torch
import torch.nn as nn
import torch.nn.functional as F

from .genotypes import ATT_PRIMITIVES
from .operations.ops_utils import feats_sum, feats_mean
from .operations.operations import *


class MixedOps(nn.Module):
    def __init__(self, __C):
        super(MixedOps, self).__init__()
        self._i_ops = nn.ModuleList([])
        self._q_ops = nn.ModuleList([])

        for prim in ATT_PRIMITIVES:
            i_op = OPS[prim](__C)
            self._i_ops.append(i_op)
            q_op = OPS[prim](__C)
            self._q_ops.append(q_op)

    def forward(self, inputs, masks, i_hot_row, q_hot_row):
        i_feat = sum(e * op(inputs, masks) for e, op in zip(i_hot_row, self._i_ops))
        q_feat = sum(e * op(inputs[::-1], masks[::-1]) for e, op in zip(q_hot_row,
            self._q_ops))

        return i_feat, q_feat


class Cell(nn.Module):
    def __init__(self, __C):
        super(Cell, self).__init__()
        self._num_step = __C.ATT_STEP
        self._init_pre_ops(__C)
        self._laynorms = nn.ModuleList([
            nn.LayerNorm(__C.HIDDEN_SIZE, elementwise_affine=False),
            nn.LayerNorm(__C.HIDDEN_SIZE, elementwise_affine=False)
        ])

        self._c_mops = nn.ModuleList([])
        for i in range(self._num_step):
            for j in range(i + 1):
                mops = MixedOps(__C)
                self._c_mops.append(mops)

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

    def forward(self, f_pre_prev, f_prev, masks, i_hot_vec, q_hot_vec):
        fts_0 = self._compute_init_state(f_pre_prev, f_prev)
        states = [fts_0]
        offset = 0
        for i in range(self._num_step):
            outputs = [self._c_mops[offset + j](h, masks, i_hot_vec[offset + j],
                q_hot_vec[offset + j]) for j, h in enumerate(states)]
            curr_f = feats_sum(outputs)
            curr_f = [norm(x) for norm, x in zip(self._laynorms, curr_f)]
            offset += len(states)
            states.append(curr_f)

        c_out = feats_mean(states[-self._num_step:])

        return c_out


class Backbone(nn.Module):
    def __init__(self, __C):
        super(Backbone, self).__init__()
        self._num_lay = __C.ATT_LAYER
        self._cells = nn.ModuleList([])
        self._attn_decoders = nn.ModuleList([
            PFFN(__C),
            PFFN(__C)
        ])

        for i in range(self._num_lay):
            cell = Cell(__C)
            self._cells.append(cell)

    def forward(self, imag, ques, i_mask, q_mask, i_hot_vec, q_hot_vec):
        fts_0 = fts_1 = [imag, ques]
        masks = [i_mask, q_mask]

        for i, cell in enumerate(self._cells):
            fts_0, fts_1 = fts_1, cell(fts_0, fts_1, masks, i_hot_vec, q_hot_vec)
        fts_1 = [pff(x) for pff, x in zip(self._attn_decoders, fts_1)]

        return fts_1
