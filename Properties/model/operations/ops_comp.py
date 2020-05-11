import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops_utils import MLP, MHAtt, qkv_attn

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class _FF(nn.Module):
    def __init__(self, __C):
        super(_FF, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.norm = nn.LayerNorm(__C.HIDDEN_SIZE, elementwise_affine=__C.NORM_AFF)

    def forward(self, x):
        x = self.norm(x + self.dropout(self.mlp(x)))

        return x


# ------------------------
# ---- Self Attention ----
# ------------------------

class _SA(nn.Module):
    def __init__(self, __C):
        super(_SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.norm = nn.LayerNorm(__C.HIDDEN_SIZE, elementwise_affine=__C.NORM_AFF)

    def forward(self, x, x_mask):
        x = self.norm(x + self.dropout(self.mhatt(x, x, x, x_mask)))

        return x


#######################
## Guided Attention ###
#######################
class _GA(nn.Module):
    def __init__(self, __C):
        super (_GA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.norm = nn.LayerNorm(__C.HIDDEN_SIZE, elementwise_affine=__C.NORM_AFF)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm(x + self.dropout(self.mhatt(y, y, x, y_mask)))

        return x
