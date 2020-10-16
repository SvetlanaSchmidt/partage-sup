import argparse

import supertagger.data as data
from supertagger.train import do_train
from supertagger.tag import do_tag


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
# TRAIN
#################################################


train = subparsers.add_parser('train', help='train a model')

train.add_argument(
    "-t",
    dest="train_path",
    required=True,
    help="train .cupt file",
    metavar="FILE",
)

train.add_argument(
    "-d",
    dest="dev_path",
    required=False,
    help="dev .cupt file",
    metavar="FILE",
)

train.add_argument(
    "-f",
    dest="fast_path",
    required=True,
    help="fastText .bin file",
    metavar="FILE",
)

train.add_argument(
    "--model-config",
    dest="model_config",
    required=False,
    help="model config .json file",
    metavar="FILE",
)

train.add_argument(
    "--train-config",
    dest="train_config",
    required=False,
    help="train config .json file",
    metavar="FILE",
)

train.add_argument(
    "--save",
    dest="save_path",
    required=False,
    help="output path to save the model",
    metavar="FILE",
)


#################################################
# TAG
#################################################


tag = subparsers.add_parser('tag', help='tag with a model')

tag.add_argument(
    "-i", "--input",
    dest="input_path",
    required=True,
    help="input .supertags file",
    metavar="FILE",
)

tag.add_argument(
    "-l", "--load",
    dest="load_path",
    required=True,
    help="path to load the model from",
    metavar="FILE",
)

tag.add_argument(
    "-f", "--fast-text",
    dest="fast_path",
    required=True,
    help="fastText .bin file",
    metavar="FILE",
)


#################################################
# MAIN
#################################################


if __name__ == '__main__':
    args = parser.parse_args()
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
    if args.command == 'train':
        do_train(args)
    elif args.command == 'tag':
        do_tag(args)