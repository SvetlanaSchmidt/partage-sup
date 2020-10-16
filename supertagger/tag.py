from supertagger.model import RoundRobin
import supertagger.data as data
from supertagger.neural.embedding.fasttext import FastText


def do_tag(args):
    print(f"# Loading fastText model from {args.fast_path}...")
    word_emb = FastText(args.fast_path, dropout=0)

    print(f"# Loading model from {args.load_path}...")
    model: RoundRobin = RoundRobin.load(args.load_path, emb=word_emb)

    for sent in data.read_supertags(args.input_path):
        inp = [tok.word_form for tok in sent]
        print(">", inp)
        out = model.decode(model(inp))
        print(out)
