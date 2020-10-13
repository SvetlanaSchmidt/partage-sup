from typing import Sequence

import torch.nn as nn
import torch.nn.utils.rnn as rnn

from supertagger.neural.utils import TT


class BiLSTM(nn.Module):

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            in_size: int,
            out_size: int,
            depth: int,
            dropout: float,
    ):
        super(BiLSTM, self).__init__()

        # We keep the size of the hidden layer equal to the embedding size
        self.net = nn.LSTM(
            input_size=in_size,
            hidden_size=out_size,
            bidirectional=True,
            num_layers=depth,
            dropout=dropout,
        )

    def flatten_parameters(self):
        """Flatten LSTM parameters."""
        self.net.flatten_parameters()

    def disable_dropout(self):
        """Turn dropout off."""
        # Needed to switch off dropout in training mode
        self.net.dropout = 0.0

    def forward_raw(self, packed: rnn.PackedSequence) -> rnn.PackedSequence:
        packed_hidden, _ = self.net(packed)
        return packed_hidden

    #
    #
    #  -------- forward -----------
    #
    def forward(self, batch: Sequence[TT]) -> rnn.PackedSequence:
        """Contextualize the embeddings for each sentence in the batch.

        The method takes on input a list of tensors with shape N x *,
        where N is the dynamic sentence length (i.e. can be different
        for each sentence) and * is any number of trailing dimensions,
        including zero, the same for each sentence.

        Returns a packed sequence.
        """

        # Pack sentence vectors as a packed sequence
        packed_batch = rnn.pack_sequence(batch, enforce_sorted=False)

        # Apply LSTM to the packed sequence of word embeddings
        packed_hidden, _ = self.net(packed_batch)

        # Return the scores
        return packed_hidden
