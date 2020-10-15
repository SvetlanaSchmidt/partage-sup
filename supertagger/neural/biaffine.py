from typing import List

import torch
import torch.nn as nn

from torch import mm, bmm, Tensor

class BiAffine(nn.Module):

    """
    The batching-enabled version of forward should work the same as
    its non-optimized version:

    Import required modules, classes, functions, etc.
    >>> import torch.nn.utils.rnn as rnn
    >>> from torch.nn import Linear

    Create the biaffine module, as well as some modules to get head/dependent
    representations (we don't care about LSTM)
    >>> head = Linear(5, 5)
    >>> dep = Linear(5, 5)
    >>> biaff = BiAffine(5)

    Create a sample input dataset, calculate head/dependent representations
    >>> sent1 = torch.randn(3, 5)   # sentence length 3
    >>> sent2 = torch.randn(4, 5)   # sentence length 4
    >>> pack = rnn.pack_sequence([sent1, sent2], enforce_sorted=False)
    >>> batch, length = rnn.pad_packed_sequence(pack, batch_first=True)
    >>> H = head(batch)
    >>> D = dep(batch)

    Finally, perform the test
    >>> scores1 = biaff.forward_padded_batch(H, D, length)
    >>> scores2 = biaff.forward_padded_batch_nopt(H, D, length)
    >>> for sc1, sc2 in zip(
    ...     unpad_biaffine(scores1, length),
    ...     unpad_biaffine(scores2, length)):
    ...         assert (sc1 - sc2 < 0.001).all()
    """

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, repr_size: int):
        """Create new instance of BiAffine.

        Arguments:
            repr_size: size of the input head/dependent representations
        """
        super().__init__()

        # Representation of the dummy root
        self.root_repr = nn.Parameter(torch.zeros(repr_size))

        # Create the bias vector
        self.bias = nn.Parameter(torch.randn(repr_size))

        # Matching matrix
        self.U = nn.Parameter(torch.randn(repr_size, repr_size))

    #
    #
    #  -------- forward -----------
    #
    def forward(self, H: Tensor, D: Tensor) -> Tensor:
        """Perform biaffine scoring for the given sentence.

        Arguments:
            H/D: tensor of shape N x R, where
                  N : sentence length
                  R : head/dependent representation size

        Returns tensor of shape N x (N + 1), where each scalar value represents
        the score of the corresponding head-dependent relation (N + 1 because
        of the dummy root, represented by index 0).
        """

        # Add the root dummy vector and transpose the matrix
        H_r = torch.cat([self.root_repr.view(1, -1), H], dim=0).t_()

        # Calculate and return the resulting scores
        return mm(mm(D, self.U), H_r) + mm(self.bias.view(1, -1), H_r)

    def forward_padded_batch(self, H: Tensor, D: Tensor, length: Tensor) -> Tensor:
        """Perform biaffine scoring on the given padded batch of sentences.

        Arguments:
            H/D: tensor of shape B x N x R
                  B : batch size
                  N : maximum sentence length
                  R : head/dependent representation size
            lenghts: tensor of shape B
                  the real length for each sentence in the batch

        Returns a score tensor of shape B x N x (N+1).

        See also `unpad_biaffine`, which allows to convert the resulting tensor
        to a list of head score vectors, one vector per sentence.
        """
        # Verify dimensions
        assert H.shape[0] == D.shape[0] == length.shape[0]
        assert H.shape[1] == D.shape[1] == max(length)

        # Account for the dummy root
        batch_size = H.shape[0]

        root = self.root_repr.view(1, 1, -1).expand(batch_size, 1, -1)
        padded_heds = torch.cat((root, H), dim=1)

        # Check the shape
        assert list(padded_heds.shape) == [
            batch_size, 1 + max(length),
            len(self.root_repr)
        ]

        padded_heds = padded_heds.permute(0, 2, 1)

        # Extend the bias vector.
        padded_bias = self.bias.view(1, 1, -1).expand(
            batch_size, max(length), len(self.root_repr))

        # Prepare the mathing matrix for bmm
        U = self.U.view(1, *self.U.shape).expand(
            batch_size, *self.U.shape)

        # Calculate the head scores
        padded_head_scores = (bmm(bmm(D, U), padded_heds) +
                              bmm(padded_bias, padded_heds))

        return padded_head_scores

    def forward_padded_batch_nopt(self, H: Tensor, D: Tensor, length: Tensor) -> Tensor:
        """Perform biaffine scoring on the given padded batch of sentences.

        A non-optimized variant of `forward_padded_batch` which relies
        on the `forward` method.
        """
        # Calculate output tensors
        outputs = []
        for x1, x2, n in zip(H, D, length):
            outputs.append(self.forward(x1[:n], x2[:n]))

        # Construct the resulting tensor, filled with zeros
        batch_size = H.shape[0]
        scores = torch.zeros(batch_size, max(length), max(length)+1)

        # Fill in the resulting tensor and return it
        for sco, out, n in zip(scores, outputs, length):
            sco[:n, :n + 1] = out
        return scores


def unpad_biaffine(padded_scores: Tensor, length: Tensor) -> List[Tensor]:
    """Convert the padded representation to a list of score tensors."""
    scores = []
    for hed_sco, n in \
            zip(padded_scores, length):
        scores.append((hed_sco[:n, :n + 1]))
    return scores
