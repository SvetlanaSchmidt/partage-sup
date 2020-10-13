from typing import Set, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from supertagger.neural.embedding.pretrained import PreTrained

from supertagger.neural.encoding import Encoding
from supertagger.neural.utils import TT, get_device, eval_on
from supertagger.neural.crf import CRF
from supertagger.neural.bilstm import BiLSTM

# from supertagger.data import Sent

# from pine.tagger.encoder.seq import Tag => str

# import timeit


##################################################
# Tagger
##################################################


# Below, sentence is an alias to a list of input words
Sent = List[str]


class Tagger(nn.Module):
    """Simple tagger based on LSTM.

    * Each input word is embedded using an external word embedder
    * BiLSTM is run over the word vector representations to get
        contextualized, hidden representations
    * Linear layer is appled to the hidden representation to get
        the score vectors
    * Tags are determined based on the score vectors
    """

    def __init__(self, config, tagset: Set[str], word_emb: PreTrained):
        super(Tagger, self).__init__()

        # Store the word embedding module
        self.word_emb = word_emb

        # Encoding (mapping between tags and integers)
        self.tag_enc = Encoding(tagset)

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
        self.score_layer = nn.Linear(config['lstm']['out_size']*2, len(tagset))

        # CRF layer
        use_crf = config['use_cfg']
        use_viterbi = config['use_viterbi']
        if use_crf:
            self.crf = CRF.new(len(tagset))
        else:
            self.crf = None
        self.use_viterbi = use_crf and use_viterbi

    def tagset(self) -> Set[str]:
        """Return the tagset of the tagger."""
        return set(self.tag_enc.obj_to_ix.keys())

    def scores_packed(self, batch: Iterable[Sent]) \
            -> rnn.PackedSequence:
        """The forward calculation over a batch of sentences.

        NOTE: This method does not apply the CRF layer.  For a variant
        which calculates CRF-based marginal scores, see `marginals`.

        Args:
            batch: a list of sentences, each sentence is a list of words

        Returns:
            a packed sequence of score tensors
        """
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
        packed_embs = rnn.pack_sequence(batch_embs, enforce_sorted=False)

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

    def scores(self, batch: Iterable[Sent]) -> List[TT]:
        """The forward calculation over a batch of sentences.
        See also lower-level `scores_packed`.

        NOTE: This method does not apply the CRF layer.  For a variant
        which calculates CRF-based marginal scores, see `marginals`.

        Args:
            batch: a list of sentences, each sentence is a list of words

        Returns:
            a list of score tensors, one tensor per input sentence; each
            tensor has the shape (N, T), where N is the number of words
            in the particular sentence and T is the size of the tagest
            (fixed for all sentences)
        """
        # Determine the packed scores
        packed_scores = self.scores_packed(batch)
        # Transform the result to a padded matrix
        scores, lengths = rnn.pad_packed_sequence(
            packed_scores,
            batch_first=True
        )
        # Split the batch of scores into a list of score matrices,
        # one matrix per sentence, while respectiing the length
        # (padding information)
        return [
            sent_scores[:length]
            for sent_scores, length in zip(scores, lengths)
        ]

    def marginals(self, batch: Iterable[Sent]) -> List[TT]:
        """A variant of the `scores` method which applies the CRF layer
        to calculate marginal scores.

        If CRF is not enabled, `marginals` is equivalent to `scores`
        (i.e., CRF behaves as an identity function).
        """
        if self.crf:
            packed_scores = self.scores_packed(batch)
            return self.crf.marginals_packed(packed_scores)
        else:
            return self.scores(batch)

    def forward(self, sent: Sent) -> TT:
        """The forward calculation over a sentence.

        A simplified variant of `marginals` which applies to
        a single sentence.
        """
        return self.marginals([sent])[0]

    def tag(self, sent: Sent) -> Sequence[str]:
        """Predict the tags in the given sentence.

        Uses marginal scores or Viterbi decoding, depending on the
        configuration of the tagger.
        """
        with torch.no_grad():
            with eval_on(self):
                if self.use_viterbi:
                    scores = self.scores([sent])[0]
                    tags = list(map(
                        self.tag_enc.decode,
                        self.crf.viterbi(scores)
                    ))
                else:
                    scores = self.marginals([sent])[0]
                    tags = []
                    for score in scores:
                        ix = torch.argmax(score).item()
                        tags.append(self.tag_enc.decode(ix))
                assert len(tags) == len(sent)
                return tags


##################################################
# Accuracy/loss
##################################################


# Dataset element: input sentence and target list of tags
DataElem = Tuple[Sent, List[str]]


def batch_loss(tagger: Tagger, data_set: Iterable[DataElem]):
    """Calculate the total loss of the model on the given dataset."""
    # CPU/GPU device
    device = get_device()
    # Calculate the target language identifiers over the entire dataset
    # and determine the inputs to run the model on
    inputs = []
    target_pos_ixs = []
    for sent, poss in data_set:
        inputs.append(sent)
        for pos in poss:
            target_pos_ixs.append(tagger.tag_enc.encode(pos))
    # Predict the scores for all the sentences in parallel
    predicted_scores = torch.cat(
        tagger.marginals(inputs)
    )
    # Calculate the loss
    return nn.CrossEntropyLoss()(
        predicted_scores, torch.tensor(target_pos_ixs).to(device)
    )


def neg_lll(tagger: Tagger, data_set: Iterable[DataElem]):
    """Negative log-likelihood of the model over the given dataset."""
    # time0 = timeit.default_timer()
    inputs = (sent for sent, _ in data_set)
    targets = (
        [tagger.tag_enc.encode(pos) for pos in poss]
        for _, poss in data_set
    )
    # time1 = timeit.default_timer()
    # print('Preprocessing time: ', time1 - time0)
    scores = tagger.scores_packed(inputs)
    # time2 = timeit.default_timer()
    # print('Scoring time: ', time2 - time1)
    if not tagger.crf:
        raise RuntimeError("neg_lll can be only used with a CRF layer!")
    log_total = tagger.crf.log_likelihood_packed(scores, targets)
    # time3 = timeit.default_timer()
    # print('CRF Time: ', time3 - time2)
    return -log_total
    # log_total = torch.tensor(0.0)
    # for sent in data_set:
    #     words, poss = zip(*sent)
    #     scores = tagger.scores([words])[0]
    #     targets = list(map(tagger.enc.encode, poss))
    #     log_total -= tagger.crf.log_likelihood(scores, targets)
    # return log_total


def accuracy(tagger: Tagger, data_set: Iterable[DataElem]) -> float:
    """Calculate the accuracy of the model on the given list of sentences.

    The accuracy is defined as the percentage of the names in the data_set
    for which the model predicts the correct tag.
    """
    k, n = 0., 0.
    for sent, gold_poss in data_set:
        pred_poss = tagger.tag(sent)
        assert len(gold_poss) == len(pred_poss)
        for (pred_pos, gold_pos) in zip(pred_poss, gold_poss):
            if pred_pos == gold_pos:
                k += 1.
            n += 1.
    return k / n
