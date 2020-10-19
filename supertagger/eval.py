from itertools import zip_longest

import supertagger.data as data
from supertagger.model.joint import full_stats
from supertagger.train import preprocess


def do_eval(args):
    total = None
    gold_stream = data.read_supertags(args.gold_path)
    pred_stream = data.read_supertags(args.pred_path)

    for gold_sent, pred_sent in \
            zip_longest(gold_stream, pred_stream, fillvalue=None):
        assert gold_sent is not None
        assert pred_sent is not None

        inp1, gold = preprocess(gold_sent)
        inp2, pred = preprocess(pred_sent)
        assert inp1 == inp2

        stats = full_stats(gold, pred)
        if total is None:
            total = stats
        else:
            total = total.add(stats)

    assert total is not None
    print(f"Acc(POS) = {total.pos_stats.acc():.2f}")
    print(f"UAS = {total.uas_stats.acc():.2f}")
    print(f"Acc(STag) = {total.stag_stats.acc():.2f}")
