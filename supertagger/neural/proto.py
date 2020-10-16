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
    def report(self: A) -> str:
        """Report stats as a string."""
        raise NotImplementedError


##################################################
# Neural protocol
##################################################


# Inp = TypeVar('Inp', contravariant=True)
Inp = TypeVar('Inp')
Out = TypeVar('Out')
Y = TypeVar('Y')
S = TypeVar('S', bound=ScoreStats, covariant=True)
class Neural(Protocol[Inp, Out, Y, S]):

    # @abstractmethod
    # def forward_batch(self, inp: List[Inp]) -> B:
    #     """Used during training"""
    #     pass

    @abstractmethod
    def forward(self, inp: Inp) -> Y:
        """Used during evaluation"""
        pass

    # @abstractmethod
    # def split(self, batch: B) -> List[Y]:
    #     pass

    @abstractmethod
    def decode(self, y: Y) -> Out:
        pass

    # @abstractmethod
    # def encode(self, gold: Out) -> Z:
    #     pass

    @abstractmethod
    def loss(self, batch: List[Tuple[Inp, Out]]) -> Tensor:
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