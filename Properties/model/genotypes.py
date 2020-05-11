from collections import namedtuple


VQAGenotype = namedtuple('VQAGenotype', 'img_gene que_gene rnn_gene')

RNN_PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]

ATT_PRIMITIVES = [
    'none',
    'feed_forward',
    'skip_connect',
    'self_attn',
    'guided_attn',
]

VQA1 = VQAGenotype(
    img_gene = [

    ],
    que_gene = [

    ],
    rnn_gene = [

    ],
)
