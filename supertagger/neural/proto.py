from typing import TypeVar, Protocol, Iterable, Tuple
from abc import abstractmethod

import torch
from torch import Tensor
import torch.nn as nn


##################################################
# Score protocol
##################################################


A = TypeVar('A', bound='Score')
class Score(Protocol):   # NOTE: Why not 'Protocol[A]'?

    @abstractmethod
    def add(self: A, other: A) -> A:
        """Return the sum of two scores."""
        raise NotImplementedError

    # @classmethod
    # @abstractmethod
    # def zero(cls) -> A:
    #     """Return zero score (neural element of addition)."""
    #     raise NotImplementedError

    @abstractmethod
    def as_float(self: A) -> float:
        """Export the score as a float."""
        raise NotImplementedError


##################################################
# Neural protocol
##################################################


Inp = TypeVar('Inp', contravariant=True)
Out = TypeVar('Out')
# X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')
S = TypeVar('S', bound=Score, covariant=True)
class Neural(Protocol[Inp, Out, Y, Z, S]):

    model: nn.Module    # The underlying neural module

    # @abstractmethod
    # def embed(self, inp: Inp) -> X:
    #     pass

    # @abstractmethod
    # def forward(self, x: X) -> Y:
    #     pass

    @abstractmethod
    def forward(self, inp: Inp) -> Y:
        pass

    @abstractmethod
    def decode(self, y: Y) -> Out:
        pass

    @abstractmethod
    def encode(self, gold: Out) -> Z:
        pass

    @abstractmethod
    def loss(self, gold: Z, pred: Y) -> Tensor:
        pass

    @abstractmethod
    def score(self, gold: Out, pred: Out) -> S:
        pass
