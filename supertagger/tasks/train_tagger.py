from typing import Set, Tuple, List

from datetime import datetime
import json

from supertagger.neural.training import train

from supertagger.neural.embedding.fasttext import FastText

# from supertagger.mwe_identifier.seq import IobTagger
from supertagger.tagger.model import Tagger, batch_loss, neg_lll, accuracy

# from supertagger.tasks.utils import load_data, load_config

import supertagger.data as data


def init_tagger(config: dict, tagset: Set[str], embed_path: str):
    time_begin = datetime.now()

    # data configuration
    print('''[-- tagger configuration: --]
    -- config: %s
    ''' % config)

    # Create an instance of the tagger and load embedding
    word_emb = FastText(embed_path, dropout=config['embedding']['dropout'])
    tagger = Tagger(config, tagset, word_emb)

    print('''[-- finished initializing tagger | duration: %s --]
    ''' % (datetime.now() - time_begin))

    return tagger


def load_config(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path) as config_file:
        return json.load(config_file)


def preprocess(sent: data.Sent) -> Tuple[List[str], List[str]]:
    """Prepare the sentence for further processing"""
    inp = [tok.word_form for tok in sent]
    out = [
        # First convert the supertag to a tree, than retrieve
        # the POS tag
        data.tree_pos(*(tok.best_stag().as_tree()))
        for tok in sent
    ]
    return inp, out


def do_train_tagger(args):
    time_begin = datetime.now()

    print('''[-- begin training: %s --]
    ''' % time_begin)

    # load train, tagger config
    train_cfg = load_config(args.train_config)
    tagger_cfg = load_config(args.tagger_config)

    # train configuration
    print('''[-- training configuration: --]
    -- train data: %s
    -- dev data: %s
    -- epochs: %s
    -- learning rate: %s
    -- weight decay (L2 penalty): %s
    -- clip: %s
    -- batch size: %d
    ''' % (
        args.train_path.split('/')[-3:] if args.train_path else None,
        args.dev_path.split('/')[-3:] if args.dev_path else None,
        train_cfg['epochs_num'],
        train_cfg['learning_rate'],
        train_cfg['weight_decay'],
        train_cfg['clip'],
        train_cfg['batch_size'],
    ))

    # collect/load the training dataset
    train_set = data.read_supertags(args.train_path)
    dev_set = []
    if args.dev_path:
        dev_set = data.read_supertags(args.dev_path)

    # data preprocessing
    train_set = list(map(preprocess, train_set))
    if dev_set:
        dev_set = list(map(preprocess, dev_set))

    # initialize the tagger
    tagset = set(tag for (inp, out) in train_set for tag in out)
    print("# Tagset:", tagset)
    tagger = init_tagger(tagger_cfg, tagset, args.fast_path)

    # train the tagger on given configuration
    for (n, lr) in zip(
        train_cfg['epochs_num'],
        train_cfg['learning_rate'],
    ):
        train(
            tagger,
            train_set,
            dev_set,
            neg_lll if args.lll else batch_loss,
            [lambda x, y: accuracy(x, y)],
            epoch_num=n,
            learning_rate=lr,
            weight_decay=train_cfg['weight_decay'],
            clip=train_cfg['clip'],
            report_rate=train_cfg['report_rate'],
            batch_size=train_cfg['batch_size'],
            shuffle=train_cfg['shuffle'],
        )

    print("")  # print newline for logging

    # save the trained model
    if (args.save_path is not None):
        tagger.save(args.save_path)

    print('''[-- finished training | duration: %s --]
    ''' % (datetime.now() - time_begin))
