from typing import Set, Iterable, List, Sequence, Tuple, Optional, Any
from dataclasses import dataclass

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


# Below, sentence is an alias to a list of input words
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

    def as_score(self):
        return self.tp_num / self.all_num


##################################################
# Tagging modules
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


@dataclass(frozen=True)
class Gold:
    posG: Tensor
    headG: Tensor
    stagG: Tensor


# @dataclass(frozen=True)
# class PredBatch:
#     """Batch of predicted tensors"""
#     posB: List[Tensor]
#     headB: List[Tensor]
#     stagB: List[Tensor]


class Joint(nn.Module, Neural[
    Sent,           # Input: list of words
    List[Out],      # Output: list of heads
    Pred,           # Predicted tensor (can be `decode`d)
    List[Pred],     # Batch of output tensors
    Gold,           # Gold tensor
    AccStats,       # Accuracy stats
]):

    def __init__(
        self,
        config,
        tagset: Set[str],
        stagset: Set[str],
        word_emb: PreTrained,
    ):
        super().__init__()

        # # Training state
        # self.do_tag = True
        # self.do_parse = True
        # self.do_stag = True

        # Tag encoding (mapping between tags and integers)
        self.tag_enc = Encoding(tagset)

        # STag encoding
        self.stag_enc = Encoding(stagset)

        # Common part
        self.emb = Embed(word_emb)
        self.ctx = Context(config)
        self.ctx_dropout = PackedSeqDropout(config['lstm']['dropout'])

        # Dependency parser
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

        # Tagger
        self.tag = Score(config['lstm']['out_size']*2, len(tagset))

        # Supertagger
        self.stag = Score(config['lstm']['out_size']*2, len(stagset))

    def tag_scores(self, ctxs: PackedSequence) -> List[Tensor]:
        return unpack(self.tag(ctxs))

    def stag_scores(self, ctxs: PackedSequence) -> List[Tensor]:
        return unpack(self.stag(ctxs))

    def head_scores(self, ctxs: PackedSequence) -> List[Tensor]:
        # Convert packed representation to a padded representation
        padded, length = rnn.pad_packed_sequence(ctxs, batch_first=True)

        # Calculate the head and dependent representations
        head_repr = self.head_repr(padded)
        dep_repr = self.dep_repr(padded)

        # Calculate the head scores and unpad
        padded_head_scores = self.heads.forward_padded_batch(
            head_repr,
            dep_repr,
            length
        )

        # Unpad and return the head scores
        return unpad_biaffine(padded_head_scores, length)

    def forward(self, batch: List[Sent]) -> List[Pred]:
        # Calculate the embeddings and contextualize them
        embs = self.emb(batch)
        ctxs = self.ctx_dropout(self.ctx(embs))

        posB = self.tag_scores(ctxs)
        headB = self.head_scores(ctxs)
        stagB = self.stag_scores(ctxs)
        assert len(posB) == len(stagB) == len(headB)

        preds = []
        for (pos, head, stag) in zip(posB, headB, stagB):
            preds.append(Pred(posP=pos, stagP=stag, headP=head))
        return preds

    def split(self, batch: List[Pred]) -> List[Pred]:
        return batch

    # def split(self, batch: PredBatch) -> List[Pred]:
    #     preds = []
    #     assert len(batch.posB) == len(batch.stagB) == len(batch.headB)
    #     for k in range(len(batch.posB)):
    #         posP = batch.posB[k]
    #         stagP = batch.stagB[k]
    #         headP = batch.headB[k]
    #         preds.append(Pred(
    #             posP=posP, stagP=stagP, headP=headP,
    #         ))
    #     return preds

    def decode_heads(self, scores: Tensor) -> List[int]:
        heads = []
        for score in scores:
            # Cast to int to avoid mypy error
            ix = int(torch.argmax(score).item())
            heads.append(ix)
        return heads

    def decode_tags(self, enc: Encoding, scores: Tensor) -> List[str]:
        tags = []
        for score in scores:
            # Cast to int to avoid mypy error
            ix = int(torch.argmax(score).item())
            tags.append(enc.decode(ix))
        return tags

    def decode(self, pred: Pred) -> List[Out]:
        heads = self.decode_heads(pred.headP)
        tags = self.decode_tags(self.tag_enc, pred.posP)
        stags = self.decode_tags(self.stag_enc, pred.stagP)
        assert len(heads) == len(tags) == len(stags)

        outs = []
        for k in range(len(heads)):
            outs.append(Out(pos=tags[k], head=heads[k], stag=stags[k]))
        return outs

    def encode(self, gold: List[Out]) -> Gold:
        tags = [self.tag_enc.encode(x.pos) for x in gold]
        stags = [self.stag_enc.encode(x.stag) for x in gold]
        heads = [x.head for x in gold]
        return Gold(
            posG=torch.tensor(tags),
            stagG=torch.tensor(stags),
            headG=torch.tensor(heads),
        )

    def loss(self, golds: List[Gold], preds: List[Pred]) -> Tensor:
        # Split golds
        gold_tags = [x.posG for x in golds]
        gold_stags = [x.stagG for x in golds]
        gold_heads = [x.headG for x in golds]
        # Split predictions
        pred_tags = [x.posP for x in preds]
        pred_stags = [x.stagP for x in preds]
        pred_heads = [x.headP for x in preds]
        # Combine and return losses
        pl = tag_loss(gold_tags, pred_tags)
        sl = tag_loss(gold_stags, pred_stags)
        hl = head_loss(gold_heads, pred_heads)
        # print(f"# BL: {pl}, {sl}, {hl}")
        return pl + sl + hl

    def score(self, gold: List[Out], pred: List[Out]) -> AccStats:
        k, n = 0, 0
        for (pred_tag, gold_tag) in zip(pred, gold):
            if pred_tag == gold_tag:
                k += 1
            n += 1
        return AccStats(tp_num=k, all_num=n)

    def module(self):
        return self


def head_loss(golds: List[Tensor], preds: List[Tensor]) -> Tensor:
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


def tag_loss(gold: List[Tensor], pred: List[Tensor]) -> Tensor:
    assert len(gold) == len(pred)
    return nn.CrossEntropyLoss()(
        torch.cat(pred),
        torch.cat(gold),
    )


##################################################
# DepParser
##################################################


class DepParser(nn.Module, Neural[
    Sent,           # Input: list of words
    List[int],      # Output: list of heads
    Tensor,         # Predicted tensor (can be `decode`d)
    List[Tensor],   # Batch of output tensors
    Tensor,         # Gold tensor (can be )
    AccStats,       # Accuracy stats
]):

    def __init__(self, config, word_emb: PreTrained):
        super().__init__()
        self.emb = Embed(word_emb)
        self.ctx = Context(config)
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

    def forward(self, batch: List[Sent]) -> List[Tensor]:
        # Calculate the embeddings and contextualize them
        embs = self.emb(batch)
        ctxs = self.ctx_dropout(self.ctx(embs))

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

    def split(self, batch: List[Tensor]) -> List[Tensor]:
        return batch

    def decode(self, scores: Tensor) -> List[int]:
        heads = []
        for score in scores:
            # Cast to int to avoid mypy error
            ix = int(torch.argmax(score).item())
            heads.append(ix)
        return heads

    def encode(self, gold: List[int]) -> Tensor:
        return torch.tensor(gold)  # .to(device)

    def loss(self, golds: List[Tensor], preds: List[Tensor]) -> Tensor:
        # Multiclass cross-etropy loss
        criterion = nn.CrossEntropyLoss(reduction='sum')
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
        return arc_loss

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
    List[Tensor],
    Tensor,
    AccStats,
]):
    def __init__(self, config, tagset: Set[str], word_emb: PreTrained):
        # Required by nn.Module
        super().__init__()
        # Encoding (mapping between tags and integers)
        self.tag_enc = Encoding(tagset)
        # Neural sub-modules
        self.net = nn.Sequential(
            Embed(word_emb),
            Context(config),
            PackedSeqDropout(config['lstm']['dropout']),
            Score(config['lstm']['out_size']*2, len(tagset)),
            # Score(config['lstm']['in_size'], len(tagset)),
        )

    def forward(self, batch: List[Sent]) -> List[Tensor]:
        return unpack(self.net(batch))

    def split(self, batch: List[Tensor]) -> List[Tensor]:
        return batch

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

    def loss(self, gold: List[Tensor], pred: List[Tensor]) -> Tensor:
        assert len(gold) == len(pred)
        return nn.CrossEntropyLoss()(
            torch.cat(pred),
            torch.cat(gold),
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