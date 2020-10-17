from typing import TypedDict, List, Tuple


from supertagger.neural.training import TrainConfig
from supertagger.model import \
    TaggerConfig, DepParserConfig, ContextConfig, EmbedConfig, JointConfig


emb_size = 100
embed: EmbedConfig = {
    'size': emb_size,
    'dropout': 0.1,
}

ctx_size: int = 200
context: ContextConfig = {
    'inp_size': emb_size,
    'out_size': ctx_size,
    'depth': 3,
    'dropout': 0.1,
}

tagger: TaggerConfig = {
    'inp_size': ctx_size*2,
    'dropout': 0.1,
}

parser: DepParserConfig = {
    'inp_size': ctx_size*2,
    'hid_size': 100,
    'out_size': 100,
    'dropout': 0.1
}

model: JointConfig = {
    'embed': embed,
    'context': context,
    'tagger': tagger,
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
