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

from supertagger.neural.proto import Score, Neural

# from supertagger.data import Sent

# from pine.tagger.encoder.seq import Tag => str

# import timeit


##################################################
# Tagger
##################################################


# Below, sentence is an alias to a list of input words
Sent = List[str]


@dataclass
class AccScore(Score):

    """Accuracy score"""

    tp_num: int
    all_num: int

    def add(self, other):
        return AccScore(
            self.tp_num + other.tp_num,
            self.all_num + other.all_num
        )

    # @classmethod
    # def zero(cls):
    #     return cls(0, 0)

    def as_float(self):
        return self.tp_num / self.all_num


##################################################
# Tagging modules
##################################################


class EmbeddingModel(nn.Module):
    """TODO"""

    def __init__(self, word_emb: PreTrained):
        super(EmbeddingModel, self).__init__()

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


class ScoringModel(nn.Module):
    """TODO"""

    def __init__(self, config, tag_num: int):
        super(ScoringModel, self).__init__()

        # Forward and backward LSTMs
        self.forward_lstm = BiLSTM(
            in_size=config['lstm']['in_size'],
            out_size=config['lstm']['out_size'],
            depth=config['lstm']['depth'],
            dropout=config['lstm']['dropout'],
        )

        # Hidden vectors dropout.
        self.hid_dropout = nn.Dropout(
            p=config['lstm']['dropout'], inplace=False)

        # TODO: add final hidden dropout in dep_parser (see above)?

        # Scoring layer
        self.score_layer = nn.Linear(config['lstm']['out_size']*2, tag_num)

        # # CRF layer
        # use_crf = config['use_cfg']
        # use_viterbi = config['use_viterbi']
        # if use_crf:
        #     self.crf = CRF.new(tag_num)
        # else:
        #     self.crf = None
        # self.use_viterbi = use_crf and use_viterbi

    def scores_packed(self, packed_embs: PackedSequence) -> PackedSequence:
        """The forward calculation over a packed sequence of embeddings.

        NOTE: This method does not apply the CRF layer.  For a variant
        which calculates CRF-based marginal scores, see `marginals`.

        Args:
            TODO

        Returns:
            a packed sequence of score tensors
        """
        # # CPU/GPU device
        # device = get_device()

        # # We first create embeddings for each sentence
        # batch_embs = [
        #     self.word_emb.forwards(sent).to(device)
        #     for sent in batch
        # ]

        # # Assert that, for each sentence in the batch, for each embedding,
        # # its size is correct
        # emb_size = self.word_emb.embedding_dim()
        # for sent_emb in batch_embs:
        #     assert sent_emb.shape[1] == emb_size

        # # Create the corresponding packed sequence
        # packed_embs = rnn.pack_sequence(batch_embs, enforce_sorted=False)

        # Calculate the hidden representations
        packed_hids = self.forward_lstm.forward_raw(packed_embs)

        # Apply dropout to the hidden vectors (this only has effect
        # in the training mode)
        packed_hids_data = self.hid_dropout(packed_hids.data)

        # Apply the linear layer to each hidden vector in the batch
        # in order the get the final scores
        scores_data = self.score_layer(packed_hids_data)

        # Create a packed sequence from the result.  According to PyTorch
        # documentation, this should be never done, but how to do that
        # differently?  Of course we *could* do that after padding the
        # packed sequence, but this seems suboptimal.
        return rnn.PackedSequence(
            scores_data,
            packed_hids.batch_sizes,
            packed_hids.sorted_indices,
            packed_hids.unsorted_indices
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

    def forward(self, embs: PackedSequence) -> List[Tensor]:
        scores = self.scores_packed(embs)
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


class TopModel(nn.Module):
    """TODO"""

    def __init__(self, config, tag_num: int, word_emb: PreTrained):
        # Encoding (mapping between tags and integers)
        # self.tag_enc = Encoding(tagset)

        super(TopModel, self).__init__()

        # Neural sub-modules
        self.emb = EmbeddingModel(word_emb)
        self.score = ScoringModel(config, tag_num)

    def forward(self, batch: List[Sent]) -> List[Tensor]:
        return self.score(self.emb(batch))


##################################################
# Tagger
##################################################


class Tagger(Neural[
    Sent,
    List[str],
    # PackedSequence,
    Tensor,
    Tensor,
    AccScore,
]):
    # Tagging model
    model: TopModel

    def __init__(self, config, tagset: Set[str], word_emb: PreTrained):
        # Encoding (mapping between tags and integers)
        self.tag_enc = Encoding(tagset)
        # Neural model
        self.model = TopModel(config, len(tagset), word_emb)

    def forward(self, sent: Sent) -> Tensor:
        return self.model([sent])[0]

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

    def loss(self, gold: Tensor, pred: Tensor) -> Tensor:
        return nn.CrossEntropyLoss()(pred, gold)

    def score(self, gold: List[str], pred: List[str]) -> AccScore:
        k, n = 0, 0
        for (pred_tag, gold_tag) in zip(pred, gold):
            if pred_tag == gold_tag:
                k += 1
            n += 1
        return AccScore(tp_num=k, all_num=n)


##################################################
# Accuracy/loss
##################################################


# # Dataset element: input sentence and target list of tags
# DataElem = Tuple[Sent, List[str]]


# def batch_loss(tagger: Tagger, data_set: Iterable[DataElem]):
#     """Calculate the total loss of the model on the given dataset."""
#     # CPU/GPU device
#     device = get_device()
#     # Calculate the target language identifiers over the entire dataset
#     # and determine the inputs to run the model on
#     inputs = []
#     target_pos_ixs = []
#     for sent, poss in data_set:
#         inputs.append(sent)
#         for pos in poss:
#             target_pos_ixs.append(tagger.tag_enc.encode(pos))
#     # Predict the scores for all the sentences in parallel
#     predicted_scores = torch.cat(
#         tagger.marginals(inputs)
#     )
#     # Calculate the loss
#     return nn.CrossEntropyLoss()(
#         predicted_scores, torch.tensor(target_pos_ixs).to(device)
#     )


# def neg_lll(tagger: Tagger, data_set: Iterable[DataElem]):
#     """Negative log-likelihood of the model over the given dataset."""
#     # time0 = timeit.default_timer()
#     inputs = (sent for sent, _ in data_set)
#     targets = (
#         [tagger.tag_enc.encode(pos) for pos in poss]
#         for _, poss in data_set
#     )
#     # time1 = timeit.default_timer()
#     # print('Preprocessing time: ', time1 - time0)
#     scores = tagger.scores_packed(inputs)
#     # time2 = timeit.default_timer()
#     # print('Scoring time: ', time2 - time1)
#     if not tagger.crf:
#         raise RuntimeError("neg_lll can be only used with a CRF layer!")
#     log_total = tagger.crf.log_likelihood_packed(scores, targets)
#     # time3 = timeit.default_timer()
#     # print('CRF Time: ', time3 - time2)
#     return -log_total
#     # log_total = torch.tensor(0.0)
#     # for sent in data_set:
#     #     words, poss = zip(*sent)
#     #     scores = tagger.scores([words])[0]
#     #     targets = list(map(tagger.enc.encode, poss))
#     #     log_total -= tagger.crf.log_likelihood(scores, targets)
#     # return log_total


# def accuracy(tagger: Tagger, data_set: Iterable[DataElem]) -> float:
#     """Calculate the accuracy of the model on the given list of sentences.

#     The accuracy is defined as the percentage of the names in the data_set
#     for which the model predicts the correct tag.
#     """
#     k, n = 0., 0.
#     for sent, gold_poss in data_set:
#         pred_poss = tagger.tag(sent)
#         assert len(gold_poss) == len(pred_poss)
#         for (pred_pos, gold_pos) in zip(pred_poss, gold_poss):
#             if pred_pos == gold_pos:
#                 k += 1.
#             n += 1.
#     return k / n
