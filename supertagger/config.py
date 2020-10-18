from typing import TypedDict, List, Tuple


from supertagger.neural.training import TrainConfig
from supertagger.model import \
    EmbedConfig, BiLSTMConfig, \
    TaggerConfig, DepParserConfig, JointConfig


emb_size = 100
embed: EmbedConfig = {
    'size': emb_size,
    'dropout': 0.1,     # NOTE: move it to the JointConfig?
}

ctx_size: int = 200
context: BiLSTMConfig = {
    'inp_size': emb_size,
    'out_size': ctx_size,
    'depth': 2,
    'dropout': 0.1,
    'out_dropout': 0.1,
}

pos_tagger: TaggerConfig = {
    'lstm': {
        'inp_size': ctx_size*2,
        'out_size': ctx_size,
        'depth': 2,
        'dropout': 0.1,
        'out_dropout': 0.1,
    },
    'inp_size': ctx_size*2,
}

super_tagger: TaggerConfig = {
    'lstm': {
        'inp_size': ctx_size*2,
        'out_size': ctx_size,
        'depth': 2,
        'dropout': 0.1,
        'out_dropout': 0.1,
    },
    'inp_size': ctx_size*2,
}

parser: DepParserConfig = {
    'lstm': {
        'inp_size': ctx_size*2,
        'out_size': ctx_size,
        'depth': 2,
        'dropout': 0.1,
        'out_dropout': 0.1,
    },
    'inp_size': ctx_size*2,
    'hid_size': 100,
    'out_size': 100,
    'dropout': 0.1
}

model: JointConfig = {
    'embed': embed,
    'context': context,
    'pos_tagger': pos_tagger,
    'super_tagger': super_tagger,
    'parser': parser,
}

train: TrainConfig = {
    'stages': [
        {'epoch_num': 60, 'learning_rate': 0.005},
        {'epoch_num': 30, 'learning_rate': 0.001},
        {'epoch_num': 30, 'learning_rate': 0.0005},
    ],
    'report_rate': 10,
    'batch_size': 32,
    'shuffle': True,
    'cuda': True,   # Use CUDA if possible
}
