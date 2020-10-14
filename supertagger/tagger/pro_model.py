from typing import Set, Iterable, List, Sequence, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from supertagger.neural.embedding.pretrained import PreTrained

from supertagger.neural.encoding import Encoding
from supertagger.neural.utils import get_device, eval_on
# from supertagger.neural.crf import CRF
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

    def __init__(self, config, tag_num: int):
        super().__init__()

        # Score layer
        self.score_layer = nn.Linear(config['lstm']['out_size']*2, tag_num)

        # # CRF layer
        # use_crf = config['use_cfg']
        # use_viterbi = config['use_viterbi']
        # if use_crf:
        #     self.crf = CRF.new(tag_num)
        # else:
        #     self.crf = None
        # self.use_viterbi = use_crf and use_viterbi

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

    def unpack(self, packed_seq: PackedSequence) -> List[Tensor]:
        # Transform the result to a padded matrix
        scores, lengths = rnn.pad_packed_sequence(
            packed_seq,
            batch_first=True
        )
        # Split the batch of scores into a list of score matrices,
        # one matrix per sentence, while respectiing the length
        # (padding information)
        return [
            sent_scores[:length]
            for sent_scores, length in zip(scores, lengths)
        ]

    # def marginals(self, embs: PackedSequence) -> List[Tensor]:
    #     """A variant of the `scores` method which applies the CRF layer
    #     to calculate marginal scores.

    #     If CRF is not enabled, `marginals` is equivalent to `scores`
    #     (i.e., CRF behaves as an identity function).
    #     """
    #     scores = self.scores_packed(embs)
    #     if self.crf:
    #         return self.crf.marginals_packed(scores)
    #     else:
    #         return self.unpack(scores)

    def forward(self, seq: PackedSequence) -> List[Tensor]:
        scores = self.score_packed(seq)
        return self.unpack(scores)

    # def tag(self, sent: Sent) -> Sequence[str]:
    #     """Predict the tags in the given sentence.

    #     Uses marginal scores or Viterbi decoding, depending on the
    #     configuration of the tagger.
    #     """
    #     with torch.no_grad():
    #         with eval_on(self):
    #             if self.use_viterbi:
    #                 scores = self.scores([sent])[0]
    #                 tags = list(map(
    #                     self.tag_enc.decode,
    #                     self.crf.viterbi(scores)
    #                 ))
    #             else:
    #                 scores = self.marginals([sent])[0]
    #                 tags = []
    #                 for score in scores:
    #                     ix = torch.argmax(score).item()
    #                     tags.append(self.tag_enc.decode(ix))
    #             assert len(tags) == len(sent)
    #             return tags


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
            Score(config, len(tagset)),
        )
        # self.emb = Embed(word_emb)
        # self.ctx = Context(config)
        # self.drp = PackedSeqDropout(config['lstm']['dropout'])
        # self.sco = Score(config, len(tagset))

    def forward(self, batch: List[Sent]) -> List[Tensor]:
        return self.net(batch)
        # return self.sco(self.drp(self.ctx(
        #     self.emb(batch)
        # )))

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