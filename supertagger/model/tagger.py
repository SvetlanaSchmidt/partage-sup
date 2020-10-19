from typing import Optional, TypedDict, List, Dict, Tuple, Set

from torch import Tensor
import torch
import torch.nn as nn

from supertagger.neural.proto import Neural
from supertagger.neural.utils import unpack
from supertagger.neural.encoding import Encoding
# from supertagger.neural.crf import CRF

from supertagger.model.stats import AccStats
from supertagger.model.utils import \
    Sent, BiLSTMConfig, Context, Score, split_batch


class TaggerConfig(TypedDict):
    """Configuration of a tagger"""
    lstm: Optional[BiLSTMConfig]
    inp_size: int
    # dropout: float


class Tagger(nn.Module, Neural[
    Sent,
    List[str],
    Tensor,
    AccStats,
]):
    # def __init__(self, config, tagset: Set[str], word_emb: PreTrained):
    def __init__(
        self, config: TaggerConfig,
        tagset: Set[str], embed: nn.Module,
        device="cpu"
    ):
        # Required by nn.Module
        super().__init__()
        self.device = device
        # Encoding (mapping between tags and integers)
        self.tag_enc = Encoding(tagset)
        # Neural sub-modules
        if config['lstm'] is None:
            embed_context = embed
        else:
            embed_context = nn.Sequential(embed, Context(config['lstm']))
        self.net = nn.Sequential(
            embed_context,
            # TODO: Should score take dropout?
            Score(config['inp_size'], len(tagset)),
        )
        # Move to appropriate device
        self.to(device)

    def forward_batch(self, batch: List[Sent]) -> List[Tensor]:
        return unpack(self.net(batch))

    def forward(self, sent: Sent) -> Tensor:
        return self.forward_batch([sent])[0]

    def decode(self, scores: Tensor) -> List[str]:
        tags = []
        for score in scores:
            # Cast to int to avoid mypy error
            ix = int(torch.argmax(score).item())
            tags.append(self.tag_enc.decode(ix))
        return tags

    def decode_dist(self, scores: Tensor, nbest: int) \
            -> List[Dict[str, float]]:
        """Variant of `dist` which returns distributions."""
        tags = []
        probs = torch.softmax(scores, dim=1)
        for prob in probs:
            dist = []
            prob_sorted, indices = torch.sort(prob, descending=True)
            for k, prob in zip(range(nbest), prob_sorted):
                ix = int(indices[k])
                dist.append((self.tag_enc.decode(ix), prob))
            tags.append(dict(dist))
        return tags

    def encode(self, gold: List[str]) -> Tensor:
        target_pos_ixs = []
        for tag in gold:
            target_pos_ixs.append(self.tag_enc.encode(tag))
        return torch.tensor(target_pos_ixs).to(self.device)

    # def loss(self, gold: List[List[str]], pred: List[Tensor]) -> Tensor:
    def loss(self, batch: List[Tuple[Sent, List[str]]]) -> Tensor:
        inps, gold = split_batch(batch)
        pred = self.forward_batch(inps)
        assert len(gold) == len(pred)
        return nn.CrossEntropyLoss()(
            torch.cat(pred),
            torch.cat(list(map(self.encode, gold))),
        )

    def score(self, gold: List[str], pred: List[str]) -> AccStats:
        k, n = 0, 0
        for (pred_tag, gold_tag) in zip(pred, gold):
            if pred_tag == gold_tag:
                k += 1
            n += 1
        return AccStats(tp_num=k, all_num=n)

    def module(self):
        return self


##################################################
# CRF Tagger
##################################################


# class CrfTagger(nn.Module, Neural[
#     Sent,
#     List[str],
#     Tensor,
#     PackedSequence,
#     List[int],
#     AccStats,
# ]):
#     def __init__(self, config, tagset: Set[str], word_emb: PreTrained):
#         # Required by nn.Module
#         super().__init__()
#         # Encoding (mapping between tags and integers)
#         self.tag_enc = Encoding(tagset)
#         # Neural sub-modules
#         self.net = nn.Sequential(
#             Embed(word_emb),
#             Context(config),
#             PackedSeqDropout(config['lstm']['dropout']),
#             Score(config, len(tagset)),
#         )
#         # CRF Layer
#         self.crf = CRF.new(len(tagset))

#     def forward(self, batch: List[Sent]) -> PackedSequence:
#         return self.net(batch)

#     def split(self, batch: PackedSequence) -> List[Tensor]:
#         return unpack(batch)

#     def decode(self, scores: Tensor) -> List[str]:
#         return list(map(
#             self.tag_enc.decode,
#             self.crf.viterbi(scores)
#         ))

#     def encode(self, gold: List[str]) -> List[int]:
#         target_pos_ixs = []
#         for tag in gold:
#             target_pos_ixs.append(self.tag_enc.encode(tag))
#         return target_pos_ixs

#     def loss(self, gold: List[List[int]], pred: PackedSequence) -> Tensor:
#         log_total = self.crf.log_likelihood_packed(pred, gold)
#         return -log_total

#     def score(self, gold: List[str], pred: List[str]) -> AccStats:
#         k, n = 0, 0
#         for (pred_tag, gold_tag) in zip(pred, gold):
#             if pred_tag == gold_tag:
#                 k += 1
#             n += 1
#         return AccStats(tp_num=k, all_num=n)

#     def module(self):
#         return self
