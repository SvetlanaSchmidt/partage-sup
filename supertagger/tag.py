import torch

from supertagger.neural.utils import eval_on
from supertagger.neural.embedding.fasttext import FastText
from supertagger.model import RoundRobin
import supertagger.data as data


def do_tag(args):
    # print(f"# Loading fastText model from {args.fast_path}...")
    word_emb = FastText(args.fast_path, dropout=0)
    # print(f"# Loading model from {args.load_path}...")
    model = RoundRobin.load(args.load_path, emb=word_emb)
    with torch.no_grad():
        with eval_on(model):
            for sent in data.read_supertags(args.input_path):
                inp = [tok.word_form for tok in sent]
                out = model.decode_dist(model.forward(inp), nbest=args.nbest)
                sent = [
                    data.Token(
                        tok_id=i+1,
                        word_form=x,
                        # head_dist={y.head: 1},
                        # stag_dist={data.STag(y.stag): 1}
                        head_dist=y.head,
                        stag_dist=dict(
                            (data.STag(stag), prob)
                            for stag, prob in y.stag.items()
                        )
                    )
                    for i, x, y in zip(range(len(inp)), inp, out)
                ]
                print(data.render_sent(sent))
                print()
