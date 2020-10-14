from typing import TypeVar, Protocol, Iterable, Tuple, List, Any
from abc import abstractmethod

import torch
from torch import Tensor
import torch.nn as nn


##################################################
# ScoreStats protocol
##################################################


A = TypeVar('A', bound='ScoreStats')
class ScoreStats(Protocol):   # NOTE: Why not 'Protocol[A]'?

    @abstractmethod
    def add(self: A, other: A) -> A:
        """Return the sum of two score stats."""
        raise NotImplementedError

    @abstractmethod
    def as_score(self: A) -> float:
        """Export score stats as a float score."""
        raise NotImplementedError


##################################################
# Neural protocol
##################################################


# Inp = TypeVar('Inp', contravariant=True)
Inp = TypeVar('Inp')
Out = TypeVar('Out')
Y = TypeVar('Y')
B = TypeVar('B')
Z = TypeVar('Z')
S = TypeVar('S', bound=ScoreStats, covariant=True)
class Neural(Protocol[Inp, Out, Y, B, Z, S]):

    @abstractmethod
    def forward(self, inp: List[Inp]) -> B:
        pass

    @abstractmethod
    def split(self, batch: B) -> List[Y]:
        pass

    @abstractmethod
    def decode(self, y: Y) -> Out:
        pass

    @abstractmethod
    def encode(self, gold: Out) -> Z:
        pass

    @abstractmethod
    def loss(self, gold: List[Z], pred: B) -> Tensor:
        pass

    @abstractmethod
    def score(self, gold: Out, pred: Out) -> S:
        pass

    @abstractmethod
    def module(self) -> nn.Module:
        """Return the `nn.Module` component of the neural model.

        If the neural model is a `nn.Module` itself, just
        `return self`.
        """
        pass