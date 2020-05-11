import torch, math
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = qkv_attn(v, k, q, mask, dropout=self.dropout)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted


#####################
### Attention VQK ###
#####################
def qkv_attn(value, key, query, mask, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    att_map = F.softmax(scores, dim=-1)
    if dropout is not None:
        att_map = dropout(att_map)

    return torch.matmul(att_map, value)


#############################################
### Function to sum list of feature lists ###
#############################################
def feats_sum(list_feats):
    temp = [torch.stack(item, dim=0) for item in zip(*list_feats)]
    output = [torch.sum(item, dim=0) for item in temp]

    return output


#############################################
### Function to sum list of feature lists ###
#############################################
def feats_mean(list_feats):
    temp = [torch.stack(item, dim=0) for item in zip(*list_feats)]
    output = [torch.mean(item, dim=0) for item in temp]

    return output


######################################################
### Function to weighted sum list of feature lists ###
######################################################
def feats_weighted_sum(list_feats, gate_values):
    sorted = [item for item in zip(*list_feats)]
    weighted = []
    for i in range(len(sorted)):
        element = []
        for j in range(len(sorted[i])):
            element.append(gate_values[:, i, j].unsqueeze(1).unsqueeze(2) *\
                sorted[i][j])
        weighted.append(element)
    weighted_sum = [sum(item) for item in weighted]

    return weighted_sum


#################
### Mask Drop ###
#################
class MaskDrop(nn.Module):
    def __init__(self, dropout):
        super(MaskDrop, self).__init__()
        self._drop_rate = dropout

    def forward(self, x):
        if not self.training:
            return x
        m = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self._drop_rate)
        mask = m.div_(1 - self._drop_rate).requires_grad_(False)
        mask = mask.expand_as(x)
        return mask * x
