from typing import Set, Iterable, List, Sequence, Tuple, Optional, Any, TypeVar
from dataclasses import dataclass
# import enum

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from supertagger.neural.embedding.pretrained import PreTrained

from supertagger.neural.encoding import Encoding
from supertagger.neural.utils import get_device, eval_on, unpack
# from supertagger.neural.crf import CRF
from supertagger.neural.bilstm import BiLSTM
from supertagger.neural.mlp import MLP
from supertagger.neural.biaffine import BiAffine, unpad_biaffine

from supertagger.neural.proto import ScoreStats, Neural

# from supertagger.data import Sent

# from pine.tagger.encoder.seq import Tag => str

# import timeit


##################################################
# Tagger
##################################################


# Sentence is an alias to a list of input words
Sent = List[str]


@dataclass
class AccStats(ScoreStats):
    """Accuracy statistics"""

    tp_num: int
    all_num: int

    def add(self, other):
        return AccStats(
            self.tp_num + other.tp_num,
            self.all_num + other.all_num
        )

    def acc(self):
        return 100 * self.tp_num / self.all_num

    def report(self):
        return f'{self.acc():2.2f}'


@dataclass
class FullStats(ScoreStats):
    """Full statistics"""

    pos_stats: AccStats
    uas_stats: AccStats
    stag_stats: AccStats

    def add(self, other: 'FullStats') -> 'FullStats':
        return FullStats(
            pos_stats=self.pos_stats.add(other.pos_stats),
            uas_stats=self.uas_stats.add(other.uas_stats),
            stag_stats=self.stag_stats.add(other.stag_stats),
        )

    def report(self):
        pos_acc = self.pos_stats.acc()
        uas = self.uas_stats.acc()
        stag_acc = self.stag_stats.acc()
        return f"[POS={pos_acc:2.2f} UAS={uas:2.2f} STag={stag_acc:2.2f}]"


def format(x, round_decimals=2):
    return round(x, round_decimals)


##################################################
# Various neural modules
##################################################


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, word_emb: PreTrained):
        super(Embed, self).__init__()

        # Store the word embedding module
        self.word_emb = word_emb

    def forward(self, batch: Iterable[Sent]) -> PackedSequence:
        # CPU/GPU device
        device = get_device()

        # We first create embeddings for each sentence
        batch_embs = [
            self.word_emb.forwards(sent).to(device)
            for sent in batch
        ]

        # Assert that, for each sentence in the batch, for each embedding,
        # its size is correct
        emb_size = self.word_emb.embedding_dim()
        for sent_emb in batch_embs:
            assert sent_emb.shape[1] == emb_size 

        # Create the corresponding packed sequence
        return rnn.pack_sequence(batch_embs, enforce_sorted=False)


class Context(nn.Module):
    """Contextualization module"""

    def __init__(self, config):
        super().__init__()
        self.bilstm = BiLSTM(
            in_size=config['lstm']['in_size'],
            out_size=config['lstm']['out_size'],
            depth=config['lstm']['depth'],
            dropout=config['lstm']['dropout'],
        )

    def forward(self, embs: PackedSequence) -> PackedSequence:
        return self.bilstm.forward_raw(embs)


class PackedSeqDropout(nn.Module):
    """Packed sequence dropout"""

    def __init__(self, dropout: float):
        super().__init__()
        self.hid_dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, seq: PackedSequence) -> PackedSequence:
        seq_data = self.hid_dropout(seq.data)
        return rnn.PackedSequence(
            seq_data,
            seq.batch_sizes,
            seq.sorted_indices,
            seq.unsorted_indices
        )


class Score(nn.Module):
    """Scoring module"""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.score_layer = nn.Linear(in_size, out_size)

    def score_packed(self, seq: PackedSequence) -> PackedSequence:
        # Apply the linear layer to each hidden vector in the packed sequence
        # in order the get the final scores
        scores_data = self.score_layer(seq.data)
        # Create a packed sequence from the result.  According to PyTorch
        # documentation, this should be never done, but how to do that
        # differently?  Of course we *could* do that after padding the
        # packed sequence, but this seems suboptimal.
        return rnn.PackedSequence(
            scores_data,
            seq.batch_sizes,
            seq.sorted_indices,
            seq.unsorted_indices
        )

    def forward(self, seq: PackedSequence) -> PackedSequence:
        scores = self.score_packed(seq)
        return scores


class PackedSeqToBatch(nn.Module):
    """Transforming a packed sequence to a batched representation"""

    def __init__(self):
        super().__init__()

    def forward(self, seq: PackedSequence) -> Tuple[Tensor, Tensor]:
        # Convert packed representation to a padded representation
        return rnn.pad_packed_sequence(seq, batch_first=True)


##################################################
# Utils
##################################################


A = TypeVar('A')
B = TypeVar('B')
def split_batch(batch: List[Tuple[A, B]]) -> Tuple[List[A], List[B]]:
    return zip(*batch)  # type: ignore


##################################################
# Joint
##################################################


@dataclass(frozen=True)
class Out:
    """Per-token output (which we want to predict)"""
    pos: str    # POS tag
    head: int   # Dependency head
    stag: str   # Supertag


@dataclass(frozen=True)
class Pred:
    """Predicted tensors (tensor representation of `Out`)"""
    posP: Tensor
    headP: Tensor
    stagP: Tensor


class RoundRobin(nn.Module, Neural[
    Sent,                       # Input: list of words
    List[Out],                  # Output: list of token-level outputs
    Pred,                       # Predicted tensor (can be `decode`d)
    FullStats,                  # Accuracy stats
]):
    def __init__(
        self,
        config,
        posset: Set[str],
        stagset: Set[str],
        word_emb: PreTrained,
    ):
        super().__init__()
        # Embedding+contextualization layer
        self.embed = nn.Sequential(Embed(word_emb), Context(config))
        # Modules for the individual tasks
        self.pos_tagger = Tagger(config, posset, self.embed)
        self.super_tagger = Tagger(config, stagset, self.embed)
        self.dep_parser = DepParser(config, self.embed)
        # Internal state
        self.step = 0

    def forward(self, sent: Sent) -> Pred:
        return Pred(
            posP = self.pos_tagger(sent),
            headP = self.dep_parser(sent),
            stagP = self.super_tagger(sent),
        )

    def decode(self, pred: Pred) -> List[Out]:
        heads = self.dep_parser.decode(pred.headP)
        tags = self.pos_tagger.decode(pred.posP)
        stags = self.super_tagger.decode(pred.stagP)
        assert len(heads) == len(tags) == len(stags)
        outs = []
        for k in range(len(heads)):
            outs.append(Out(pos=tags[k], head=heads[k], stag=stags[k]))
        return outs

    def loss(self, batch: List[Tuple[Sent, List[Out]]]) -> Tensor:
        k = self.step
        self.step = (self.step + 1) % 3
        inps, gold_all = split_batch(batch)
        if k == 0:
            gold = [[x.pos for x in sent] for sent in gold_all]
            return self.pos_tagger.loss(list(zip(inps, gold)))
        elif k == 1:
            gold = [[x.stag for x in sent] for sent in gold_all]
            return self.super_tagger.loss(list(zip(inps, gold)))
        elif k == 2:
            goldh = [[x.head for x in sent] for sent in gold_all]
            return self.dep_parser.loss(list(zip(inps, goldh)))
        else:
            raise RuntimeError("Invalid value of `self.step`")

    def score(self, golds: List[Out], preds: List[Out]) -> FullStats:
        pos, uas, stag, n = 0, 0, 0, 0
        for (pred, gold) in zip(preds, golds):
            if pred.pos == gold.pos:
                pos += 1
            if pred.head == gold.head:
                uas += 1
            if pred.stag == gold.stag:
                stag += 1
            n += 1
        pos_stats = AccStats(tp_num=pos, all_num=n)
        uas_stats = AccStats(tp_num=uas, all_num=n)
        stag_stats = AccStats(tp_num=stag, all_num=n)
        return FullStats(
            pos_stats=pos_stats,
            uas_stats=uas_stats,
            stag_stats=stag_stats,
        )

    def module(self):
        return self


##################################################
# DepParser
##################################################


class DepParser(nn.Module, Neural[
    Sent,           # Input: list of words
    List[int],      # Output: list of heads
    Tensor,         # Predicted tensor (can be `decode`d)
    AccStats,       # Accuracy stats
]):

    # def __init__(self, config, word_emb: PreTrained):
    def __init__(self, config, embed: nn.Module):
        super().__init__()
        self.emb = embed
        self.ctx_dropout = PackedSeqDropout(config['lstm']['dropout'])
        self.head_repr = MLP(
            in_size=config['lstm']['out_size'] * 2,
            hid_size=config['mlp_head_dep']['hidden_size'],
            out_size=config['mlp_head_dep']['out_size'],
            dropout=config['mlp_head_dep']['dropout'],
        )
        self.dep_repr = MLP(
            in_size=config['lstm']['out_size'] * 2,
            hid_size=config['mlp_head_dep']['hidden_size'],
            out_size=config['mlp_head_dep']['out_size'],
            dropout=config['mlp_head_dep']['dropout'],
        )
        # A biaffine arc scoring model
        self.heads = BiAffine(repr_size=config['mlp_head_dep']['out_size'])

    def forward_batch(self, batch: List[Sent]) -> List[Tensor]:
        # Calculate the embeddings and contextualize them
        ctxs = self.emb(batch)

        # Convert packed representation to a padded representation
        padded, length = rnn.pad_packed_sequence(ctxs, batch_first=True)

        # # Calculate the head and dependent representations
        head_repr = self.head_repr(padded)
        dep_repr = self.dep_repr(padded)

        # Calculate the head scores and unpad
        padded_head_scores = self.heads.forward_padded_batch(
            head_repr,
            dep_repr,
            length
        )

        # Return the head scores
        return unpad_biaffine(padded_head_scores, length)

    def forward(self, sent: Sent) -> Tensor:
        return self.forward_batch([sent])[0]

    def decode(self, scores: Tensor) -> List[int]:
        heads = []
        for score in scores:
            # Cast to int to avoid mypy error
            ix = int(torch.argmax(score).item())
            heads.append(ix)
        return heads

    # def loss(self, golds_raw: List[List[int]], preds: List[Tensor]) -> Tensor:
    def loss(self, batch: List[Tuple[Sent, List[int]]]) -> Tensor:
        # Split batch, process input, retrieve golds
        inputs, golds_raw = split_batch(batch)
        preds = self.forward_batch(inputs)
        golds = list(map(torch.tensor, golds_raw))
        # Both input list should have the same size
        assert len(golds) == len(preds)
        # Multiclass cross-etropy loss
        criterion = nn.CrossEntropyLoss()
        # Store the dependency parsing loss in a variable
        arc_loss = torch.tensor(0.0, dtype=torch.float)
        # Calculate the loss for each sentence separately
        # (this could be further optimized using padding)
        for pred, gold in zip(preds, golds):
            # Check dimensions
            assert gold.dim() == 1
            assert pred.dim() == 2
            assert pred.shape[0] == gold.shape[0]
            # Calculate the sent loss and update the batch dependency loss
            arc_loss += criterion(pred, gold)
        return arc_loss / len(preds)

    # def loss(self, golds: List[Tensor], preds: List[Tensor]) -> Tensor:
    #     assert len(golds) == len(preds)
    #     # print("gold shape:", gold[0].shape)
    #     # print("pred shape:", pred[0].shape)

    #     # Maximum sentence length
    #     B = len(golds)
    #     N = max(len(x) for x in golds)

    #     # Construct a batched prediction tensor with defualt, low score
    #     padded = torch.full((B, N+1), -10000., dtype=torch.float)
    #     for k, pred in zip(range(B), preds):
    #         n = len(pred)
    #         padded[k, :n] = pred

    #     return nn.CrossEntropyLoss()(
    #         padded,
    #         torch.cat(golds),
    #     )

    def score(self, gold: List[int], pred: List[int]) -> AccStats:
        k, n = 0, 0
        for (pred_tag, gold_tag) in zip(pred, gold):
            if pred_tag == gold_tag:
                k += 1
            n += 1
        return AccStats(tp_num=k, all_num=n)

    def module(self):
        return self


##################################################
# Tagger
##################################################


class Tagger(nn.Module, Neural[
    Sent,
    List[str],
    Tensor,
    AccStats,
]):
    # def __init__(self, config, tagset: Set[str], word_emb: PreTrained):
    def __init__(self, config, tagset: Set[str], embed: nn.Module):
        # Required by nn.Module
        super().__init__()
        # Encoding (mapping between tags and integers)
        self.tag_enc = Encoding(tagset)
        # Neural sub-modules
        self.net = nn.Sequential(
            embed,
            PackedSeqDropout(config['lstm']['dropout']),
            Score(config['lstm']['out_size']*2, len(tagset)),
        )

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

    def encode(self, gold: List[str]) -> Tensor:
        target_pos_ixs = []
        for tag in gold:
            target_pos_ixs.append(self.tag_enc.encode(tag))
        return torch.tensor(target_pos_ixs)  # .to(device)

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