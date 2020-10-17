from typing import Set, Tuple, List

from datetime import datetime
import json

import torch.nn as nn

from supertagger.neural.training import train
from supertagger.neural.embedding.fasttext import FastText

from supertagger.model import RoundRobin, Out, JointConfig
import supertagger.config as cfg
import supertagger.data as data


def init_model(
    config: JointConfig,
    posset: Set[str],
    stagset: Set[str],
    embed_path: str
):
    time_begin = datetime.now()
    word_emb = FastText(embed_path, dropout=config['embed']['dropout'])
    model = RoundRobin(config, posset, stagset, word_emb)
    print(f"# Model initialized in {(datetime.now() - time_begin)}")
    return model


def preprocess(sent: data.Sent) -> Tuple[List[str], List[Out]]:
    """Prepare the sentence for training a POS tagger"""
    inp = [tok.word_form for tok in sent]
    out = []
    for tok in sent:
        stag = tok.best_stag()
        pos = data.tree_pos(*(stag.as_tree()))
        head = tok.best_head()
        assert pos is not None
        out.append(Out(
            pos=pos, head=head, stag=stag.stag_str
        ))
    return inp, out


def load_config(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path) as config_file:
        return json.load(config_file)


def do_train(args):

    # Load configurations
    if args.train_config is None:
        train_cfg = cfg.train
    else:
        train_cfg = load_config(args.train_config)
    if args.model_config is None:
        model_cfg = cfg.model
    else:
        model_cfg = load_config(args.model_config)

    if args.emb_size is not None:
        model_cfg['embed']['size'] = args.emb_size
        model_cfg['context']['inp_size'] = args.emb_size

    print("# Model:", model_cfg)
    print("# Training:", train_cfg)

    # Load and proprocess the datasets
    train_set_raw = data.read_supertags(args.train_path)
    train_set = list(map(preprocess, train_set_raw))
    print("# No. of sentence in train:", len(train_set))
    dev_set = []
    if args.dev_path:
        dev_set_raw = data.read_supertags(args.dev_path)
        dev_set = list(map(preprocess, dev_set_raw))
        print("# No. of sentence in dev:", len(dev_set))

    # Initialize the model
    posset = set(x.pos for (inp, out) in train_set for x in out)
    print("# No. of POS tags:", len(posset))
    stagset = set(x.stag for (inp, out) in train_set for x in out)
    print("# No. of supertags:", len(stagset))
    model = init_model(model_cfg, posset, stagset, args.fast_path)

    # Train the model
    for stage in train_cfg['stages']:
        train(
            model, train_set, dev_set,
            epoch_num=stage['epoch_num'],
            learning_rate=stage['learning_rate'],
            report_rate=train_cfg['report_rate'],
            batch_size=train_cfg['batch_size'],
            shuffle=train_cfg['shuffle'],
        )

    # Save the model
    if (args.save_path is not None):
        print(f"# Saving model to {args.save_path}")
        model.save(args.save_path)
