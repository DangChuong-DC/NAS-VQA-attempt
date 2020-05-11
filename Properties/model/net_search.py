
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from Properties.model.operations.ops_utils import FC, MLP
from Properties.model.rnn_model.model_search import RNN_Model_Search
from .backbone_search import Backbone
from .genotypes import *


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ----------------------------
# ---- Main VQA-NAS Model ----
# ----------------------------

class Net_Search(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net_Search, self).__init__()
        self._att_step = __C.ATT_STEP
        self._rnn_step = __C.RNN_STEP
        self._init_alphas()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.rnn =  RNN_Model_Search(
            self.alphas_rnn,
            __C.WORD_EMBED_SIZE,
            __C.HIDDEN_SIZE,
            __C.RNN_INP_DR_R,
            __C.RNN_HID_DR_R,
            __C.RNN_HID_DR_R,
            __C.RNN_STEP
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = Backbone(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = nn.LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, last_hid = self.rnn(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Relax architecture parameters
        i_hot_vec = self.prod_hot_vec(self.alphas_imag)
        q_hot_vec = self.prod_hot_vec(self.alphas_ques)

        # Backbone Framework
        img_feat, lang_feat = self.backbone(
            img_feat,
            lang_feat,
            img_feat_mask,
            lang_feat_mask,
            i_hot_vec,
            q_hot_vec
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat

    def _init_alphas(self):
        _att_branches = sum(i for i in range(1, self._att_step + 1))
        num_att_ops = len(ATT_PRIMITIVES)

        _rnn_branches = sum(i for i in range(1, self._rnn_step + 1))
        num_rnn_ops = len(RNN_PRIMITIVES)

        self.alphas_imag = nn.Parameter(1e-3 * torch.randn(_att_branches, num_att_ops),
            requires_grad=True)
        self.alphas_ques = nn.Parameter(1e-3 * torch.randn(_att_branches, num_att_ops),
            requires_grad=True)
        self.alphas_rnn = nn.Parameter(1e-3 * torch.randn(_rnn_branches, num_rnn_ops),
            requires_grad=True)
        self._arch_params = nn.ParameterList([self.alphas_imag, self.alphas_ques,
            self.alphas_rnn])

    def arch_parameters(self):
        return self._arch_params

    def net_parameters(self):
        net_params = []
        for n, p in self.named_parameters():
            if p.requires_grad and not (n.endswith('alphas_imag') or n.endswith('alphas_ques') \
                    or n.endswith('alphas_rnn')):
                net_params.append(p)
        return net_params

    def prod_hot_vec(self, weights, temperature=1):
        return F.gumbel_softmax(weights, tau=temperature, hard=True, dim=-1)

    def genotype(self):
        def _parse1(weights):
            gene = []
            start = 0
            for i in range(self._att_step):
                dna = []
                end = start + i + 1
                W = weights[start:end].copy()
                edges = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in
                    range(len(W[x]))))
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        # if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    dna.append((ATT_PRIMITIVES[k_best], j))
                gene.append(dna)
                start = end
            return gene

        def _parse2(weights):
            gene = []
            start = 0
            for i in range(self._rnn_step):
                dna = []
                end = start + i + 1
                W = weights[start:end].copy()
                edges = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in
                    range(len(W[x]))))
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        # if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    dna.append((RNN_PRIMITIVES[k_best], j))
                gene.append(dna)
                start = end
            return gene

        i_gene = _parse1(self.alphas_imag.clone().detach().cpu().numpy())
        q_gene = _parse1(self.alphas_ques.clone().detach().cpu().numpy())
        r_gene = _parse2(self.alphas_rnn.clone().detach().cpu().numpy())
        genotype = VQAGenotype(img_gene=i_gene, que_gene=q_gene, rnn_gene=r_gene)
        return genotype

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
