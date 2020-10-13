from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from supertagger.neural.utils import TT


class PreTrained(ABC, nn.Module):
    """
    Abstract embedder template for pretrained word vectors.

    Parameters
    ----------
    - file_path: path to pretrained word vectors
    - dropout: applied dropout on training

    Methods
    ----------
    - forward(self, word: str) -> TT
    - forwards(self, words: list) -> TT
    - abs: load_model(self, file_path: str)
    - abs: embedding_dim(self) -> int
    - abs: embedding_num(self) -> int
    """

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            model,
            dropout: float = 0,
    ):
        super(PreTrained, self).__init__()

        # save model
        self.model = model

        # save dropout
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    #
    #
    #  -------- forward -----------
    #
    def forward(self, word: str) -> TT:
        """Embed single given word."""

        emb = torch.tensor(self.model[word], dtype=torch.float)
        return self.dropout(emb)

    #
    #
    #  -------- forwards -----------
    #
    def forwards(self, words: list) -> TT:
        """Embed multiply given words."""

        emb: list = []

        for w in words:
            emb.append(self.forward(w))

        return torch.stack(emb)

    #
    #
    #  -------- load_model -----------
    #
    @abstractmethod
    def load_model(self, file_path: str):
        """Return the loaded model."""
        raise NotImplementedError

    #
    #
    #  -------- embedding_dim -----------
    #
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        raise NotImplementedError

    #
    #
    #  -------- embedding_dim -----------
    #
    @abstractmethod
    def embedding_num(self) -> int:
        """Return the count of the embedded words."""
        raise NotImplementedError
