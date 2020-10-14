from typing import Set, Iterable, List, Sequence, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from supertagger.neural.embedding.pretrained import PreTrained

from supertagger.neural.encoding import Encoding
from supertagger.neural.utils import get_device, eval_on, unpack
from supertagger.neural.crf import CRF
from supertagger.neural.bilstm import BiLSTM

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


class CrfTagger(nn.Module, Neural[
    Sent,
    List[str],
    Tensor,
    PackedSequence,
    List[int],
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
            Score(config, len(tagset)),
        )
        # CRF Layer
        self.crf = CRF.new(len(tagset))

    def forward(self, batch: List[Sent]) -> PackedSequence:
        return self.net(batch)

    def split(self, batch: PackedSequence) -> List[Tensor]:
        return unpack(batch)

    def decode(self, scores: Tensor) -> List[str]:
        return list(map(
            self.tag_enc.decode,
            self.crf.viterbi(scores)
        ))

    def encode(self, gold: List[str]) -> List[int]:
        target_pos_ixs = []
        for tag in gold:
            target_pos_ixs.append(self.tag_enc.encode(tag))
        return target_pos_ixs

    def loss(self, gold: List[List[int]], pred: PackedSequence) -> Tensor:
        log_total = self.crf.log_likelihood_packed(pred, gold)
        return -log_total

    def score(self, gold: List[str], pred: List[str]) -> AccStats:
        k, n = 0, 0
        for (pred_tag, gold_tag) in zip(pred, gold):
            if pred_tag == gold_tag:
                k += 1
            n += 1
        return AccStats(tp_num=k, all_num=n)

    def module(self):
        return self