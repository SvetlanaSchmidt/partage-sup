import argparse

import supertagger.data as data
from supertagger.tasks.train_tagger import do_train_tagger

parser = argparse.ArgumentParser(description='supertagger')
subparsers = parser.add_subparsers(dest='command', help='available commands')


#################################################
# TEST
#################################################


parser_test = subparsers.add_parser('test', help='test')

parser_test.add_argument(
    "-i",
    dest="input_path",
    required=True,
    help="input .supertags file",
    metavar="FILE",
)


#################################################
# TRAIN TAGGER
#################################################


tagger_train = subparsers.add_parser('train-tagger', help='train the tagger')

tagger_train.add_argument(
    "-t",
    dest="train_path",
    required=True,
    help="train .cupt file",
    metavar="FILE",
)

tagger_train.add_argument(
    "-d",
    dest="dev_path",
    required=False,
    help="dev .cupt file",
    metavar="FILE",
)

tagger_train.add_argument(
    "-f",
    dest="fast_path",
    required=True,
    help="fastText .bin file",
    metavar="FILE",
)

tagger_train.add_argument(
    "--tagger-config",
    dest="tagger_config",
    required=True,
    help="tagger config .json file",
    metavar="FILE",
)

tagger_train.add_argument(
    "--train-config",
    dest="train_config",
    required=True,
    help="train config .json file",
    metavar="FILE",
)

tagger_train.add_argument(
    "--save-model",
    dest="save_path",
    required=False,
    help="output path to save the model",
    metavar="FILE",
)

tagger_train.add_argument(
    "--lll",
    action="store_true",
    dest="lll",
)


#################################################
# MAIN
#################################################


if __name__ == '__main__':

    # get console arguments
    args = parser.parse_args()

    # choose task
    if args.command == 'test':
        for sent in data.read_supertags(args.input_path):
            print("#", [tok.word_form for tok in sent])
            print(
                "@",
                [data.tree_pos(*(tok.best_stag().as_tree()))
                 for tok in sent]
            )
            print("%", [tok.best_head() for tok in sent])
            # for tok in sent:
            #     tree, sent = tok.best_stag().as_tree()
            #     print(data.tree_pos(tree, sent))

    if args.command == 'train-tagger':
        do_train_tagger(args)
