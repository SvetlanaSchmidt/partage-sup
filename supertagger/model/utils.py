from typing import List, TypedDict, Iterable, Tuple, TypeVar

from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.utils.rnn as rnn

from supertagger.neural.bilstm import BiLSTM
from supertagger.neural.embedding.pretrained import PreTrained


##################################################
# Types
##################################################


# Sentence is an alias to a list of input words
Sent = List[str]


##################################################
# Functions
##################################################


A = TypeVar('A')
B = TypeVar('B')
def split_batch(batch: List[Tuple[A, B]]) \
        -> Tuple[List[A], List[B]]:   # noqa: E302
    return zip(*batch)  # type: ignore


##################################################
# Neural modules
##################################################


# FIXME: This doesn't fit here, it's not a configuration
# of the `Embed` model below.
class EmbedConfig(TypedDict):
    """Configuration of the BiLSTM, contextualiaztion layer"""
    size: int
    dropout: float


class Embed(nn.Module):
    """Embedding module.

    Type: Iterable[Sent] -> PackedSequence
    """

    def __init__(self, word_emb: PreTrained, device='cpu'):
        super().__init__()
        self.word_emb = word_emb
        self.device = device

    def forward(self, batch: Iterable[Sent]) -> PackedSequence:
        # We first create embeddings for each sentence
        batch_embs = [
            self.word_emb.forwards(sent).to(self.device)
            for sent in batch
        ]

        # Assert that, for each sentence in the batch, for each embed,
        # its size is correct
        emb_size = self.word_emb.embedding_dim()
        for sent_emb in batch_embs:
            assert sent_emb.shape[1] == emb_size

        # Create the corresponding packed sequence
        return rnn.pack_sequence(batch_embs, enforce_sorted=False)


class BiLSTMConfig(TypedDict):
    """Configuration of the BiLSTM, contextualization layer"""
    inp_size: int
    out_size: int
    depth: int
    dropout: float
    out_dropout: float


class Context(nn.Module):
    """Contextualization module.

    Type: PackedSequence -> PackedSequence
    """

    def __init__(self, config: BiLSTMConfig):
        super().__init__()
        self.net = nn.Sequential(
            BiLSTM(
                in_size=config['inp_size'],
                out_size=config['out_size'],
                depth=config['depth'],
                dropout=config['dropout'],
            ),
            PackedSeqDropout(p=config['out_dropout'])
        )

    def forward(self, embs: PackedSequence) -> PackedSequence:
        # return self.out_dropout(self.bilstm(embs))
        return self.net(embs)


class PackedSeqDropout(nn.Module):
    """Packed sequence dropout.

    Type: PackedSequence -> PackedSequence
    """

    def __init__(self, p: float):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=False)

    def forward(self, seq: PackedSequence) -> PackedSequence:
        seq_data = self.dropout(seq.data)
        return rnn.PackedSequence(
            seq_data,
            seq.batch_sizes,
            seq.sorted_indices,
            seq.unsorted_indices
        )


class Score(nn.Module):
    """Scoring module.

    Type: PackedSequence -> PackedSequence
    """

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


class Packed2Padded(nn.Module):
    """Transform a packed sequence to a padded representation.

    Type: PackedSequence -> Tuple[Tensor, Tensor]
    """

    def __init__(self):
        super().__init__()

    def forward(self, seq: PackedSequence) -> Tuple[Tensor, Tensor]:
        # Convert packed representation to a padded representation
        return rnn.pad_packed_sequence(seq, batch_first=True)
