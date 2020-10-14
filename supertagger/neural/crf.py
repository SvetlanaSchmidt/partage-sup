from typing import List, Iterator, Iterable

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch import Tensor

# import timeit


def batch_start(size: Tensor) -> Tensor:
    """Calculate the starting position of the subsequent batches
    in a PackedSequence.

    >>> xs = torch.tensor([1, 2, 3])
    >>> ys = torch.tensor([4, 5])
    >>> ps = rnn.pack_sequence([xs, ys])
    >>> ps.batch_sizes
    tensor([2, 2, 1])
    >>> batch_start(ps.batch_sizes)
    tensor([0, 2, 4])
    """
    assert size.dim() == 1
    ixs = [0]
    n = 0
    for x in size:
        n += x.item()
        ixs.append(n)
    return torch.LongTensor(ixs[:-1])


# TODO: optimize this!
def add_to_each(batch: Tensor, T: Tensor) -> Tensor:
    """For each vector `v` in the `batch`, create a new instance
    of `T` and add `v` to each row in `T`.  As a result, we should
    obtain B copies of (modified) `T`, where B is the batch size.

    >>> batch = torch.tensor([[1, 2], [3, 4]])
    >>> T = torch.tensor([[1, 0], [0, 1]])
    >>> Y = add_to_each(batch, T)
    >>> len(Y)
    2
    >>> Y[0]
    tensor([[2, 2],
            [1, 3]])
    >>> Y[1]
    tensor([[4, 4],
            [3, 5]])
    """
    Y = []
    for v in batch:
        Y.append(v + T)
    return torch.stack(Y)


class CRF(nn.Module):
    """First-order, linear-chain CRFs.

    Create CRF with 10 output classes:
    >>> crf = CRF.new(10)

    The input to CRF are score values assigned to all words (dim 0)
    and all output classes (dim 1):
    >>> scores = torch.randn(20, 10)

    Calculate the marginal probabilties:
    >>> margs = crf(scores)

    The output probabilities are given in the log-domain and have the
    same shape as the input scores:
    >>> print(margs.shape)
    torch.Size([20, 10])

    The normalization factor (partition function) should be the the same
    whether we use alphas or betas to calculate it (below we use
    random_weights, because otherwise the test is "too simple"):
    >>> crf = CRF.new(10, random_weights=True)
    >>> scores = torch.randn(20, 10)
    >>> alpha = crf.alpha(scores)
    >>> beta = crf.beta(scores)
    >>> Z1 = torch.logsumexp(alpha[-1], dim=0)
    >>> Z2 = torch.logsumexp(beta[0], dim=0)
    >>> assert abs(Z1 - Z2) < 1e-3

    You can also apply a CRF to a packed sequence.
    >>> xs = torch.randn(15, 10)
    >>> ys = torch.randn(10, 10)
    >>> zs = torch.randn(5, 10)
    >>> ps = rnn.pack_sequence([xs, ys, zs])

    Alphas:
    >>> xs_alpha = crf.alpha(xs)
    >>> ps_alpha = crf.alpha_packed(ps)
    >>> assert (xs_alpha[-1] == ps_alpha.data[-1]).all()

    Betas:
    >>> xs_beta = crf.beta(xs)
    >>> ps_beta = crf.beta_packed(ps)
    >>> assert (xs_beta[0] == ps_beta.data[0]).all()

    Marginals:
    >>> xs_margs = crf.marginals(xs)
    >>> ys_margs = crf.marginals(ys)
    >>> zs_margs = crf.marginals(zs)
    >>> ps_margs = crf.marginals_packed(ps)
    >>> assert len(ps_margs) == 3
    >>> assert (xs_margs == ps_margs[0]).all()
    >>> assert (ys_margs == ps_margs[1]).all()
    >>> assert (zs_margs == ps_margs[2]).all()

    NOTEs:
    * The input sentence must have at least one word.
    * Currently, if the input sentence has length 1, the CRF layer
        will have no substantial effect.

    TODOs:
    * Add beginning and ending "bias" vectors (see the second note above).
    * Use a different initialization method (Xavier?).
    """

    # def __init__(self, class_num: int, random_weights=False):
    #     super(CRF, self).__init__()
    #     # Create a transition weights matrix
    #     if random_weights:
    #         mk_tensor = torch.randn
    #     else:
    #         mk_tensor = torch.zeros
    #     self.T = nn.Parameter(mk_tensor((class_num, class_num)))

    def __init__(self, params: Tensor):
        super(CRF, self).__init__()
        self.T = params

    @classmethod
    def new(cls, class_num: int, random_weights=False):
        # Create an initial transition weight matrix
        if random_weights:
            mk_tensor = torch.randn
        else:
            mk_tensor = torch.zeros
        return cls(nn.Parameter(mk_tensor((class_num, class_num))))

    def class_num(self) -> int:
        """Return the number of classes."""
        return self.T.shape[0]

    def add(self, other):
        """Add the CRF with another one"""
        return CRF(self.T + other.T)

    def alpha(self, scores: Tensor):
        """In the CRF terminology: forward pass"""
        # Sentence length
        sent_len = scores.shape[0]
        # Additional check
        class_num = self.class_num()
        assert class_num == scores.shape[1]
        # Tensor for results, accounts for the input scores
        log_alpha = scores.clone()
        # Transposed transition weights matrix
        T_t = self.T.t()
        # Process columns (except the first)
        for i in range(1, sent_len):
            # Below, log_alpha[i-1] is added to each row in T_t
            sum_tensor = log_alpha[i-1] + T_t
            # Apply logsumexp in a batch
            log_alpha[i] += torch.logsumexp(sum_tensor, dim=1)

            # NOTE: below, commented out, you can find a more
            # explicit way of implementing the two lines above.

            # # For each output class assigned to position i
            # for x in range(class_num):
            #     sum_tensor = log_alpha[i-1] + T_t[x]
            #     # sum_tensor = torch.tensor([
            #     #     log_alpha[i-1, y] + self.T[y, x]
            #     #     for y in range(class_num)
            #     # ])
            #     log_alpha[i, x] += torch.logsumexp(sum_tensor, dim=0)

        # Return the result
        return log_alpha

    def alpha_packed(self, scores: rnn.PackedSequence) -> rnn.PackedSequence:
        """Variant of `alpha` which works on packed sequences."""
        # Additional check
        class_num = self.class_num()
        assert class_num == scores.data.shape[1]
        # Tensor for results, accounts for the input scores
        alpha = scores.data.clone()
        # Transposed transition weights matrix
        T_t = self.T.t()
        # Sizes and start positions of the batches
        size = scores.batch_sizes
        start = batch_start(size)
        # Loop over the batches (except the first), left to right
        for i in range(1, len(size)):
            # Alpha corresponding to the previous batch
            alpha_prev = alpha[start[i-1]:start[i-1]+size[i]]
            # For each row alpha_row in alpha_prev, create a separate instance
            # of T_t and add alpha_row to each row in this instance of T_t.
            # As a result, we should obtain B copies of T_t, where B is
            # the size of the batch.
            sum_tensor = add_to_each(alpha_prev, T_t)
            # Apply logsumexp in a batch
            alpha[start[i]:start[i]+size[i]] += \
                torch.logsumexp(sum_tensor, dim=2)
        # Return the result
        return rnn.PackedSequence(
            alpha,
            scores.batch_sizes,
            scores.sorted_indices,
            scores.unsorted_indices
        )

    def beta(self, scores: Tensor):
        """In the CRF terminology: backward pass"""
        # Sentence length
        sent_len = scores.shape[0]
        # Additional check
        class_num = self.class_num()
        assert class_num == scores.shape[1]
        # Tensor for results, accounts for the input scores
        log_beta = scores.clone()
        # Process columns (except the last)
        for i in range(sent_len-2, -1, -1):
            # Below, log_beta[i+1] is added to each row in self.T
            sum_tensor = log_beta[i+1] + self.T
            # Apply logsumexp in a batch
            log_beta[i] += torch.logsumexp(sum_tensor, dim=1)

            # NOTE: below, commented out, you can find a more
            # explicit way of implementing the two lines above.

            # # For each output class assigned to position i
            # for x in range(class_num):
            #     sum_tensor = self.T[x] + log_beta[i+1]
            #     # sum_tensor = torch.tensor([
            #     #     log_beta[i+1, y] + self.T[x, y]
            #     #     for y in range(class_num)
            #     # ])
            #     log_beta[i, x] += torch.logsumexp(sum_tensor, dim=0)

        # Return the result
        return log_beta

    def beta_packed(self, scores: rnn.PackedSequence) -> rnn.PackedSequence:
        """Variant of `beta` which works on packed sequences."""
        # Additional check
        class_num = self.class_num()
        assert class_num == scores.data.shape[1]
        # Tensor for results, accounts for the input scores
        beta = scores.data.clone()
        # Sizes and start positions of the batches
        size = scores.batch_sizes
        start = batch_start(size)
        # Loop over the batches (except the last), right to left
        for i in range(len(size)-2, -1, -1):
            # Beta corresponding to the next batch
            beta_next = beta[start[i+1]:start[i+1]+size[i+1]]
            # See `add_to_each` and `alpha_packed`
            sum_tensor = add_to_each(beta_next, self.T)
            # Apply logsumexp in a batch
            beta[start[i]:start[i]+size[i+1]] += \
                torch.logsumexp(sum_tensor, dim=2)
        # Return the result
        return rnn.PackedSequence(
            beta,
            scores.batch_sizes,
            scores.sorted_indices,
            scores.unsorted_indices
        )

    def viterbi(self, scores: Tensor) -> List[int]:
        """Perform Viterbi decoding.

        Determine and return the highest-scoring list of class indices.
        """
        with torch.no_grad():
            # Sentence length
            sent_len = scores.shape[0]
            # Tensors for max score sums and pointers
            mx = scores.clone()
            argmx = torch.full_like(scores, 0, dtype=torch.long)

            # NOTE: the code fragment below is a max-product analog
            # of the sum-product code defined in the alpha method.

            # Transposed transition weights matrix
            T_t = self.T.t()
            # Process columns (except the first)
            for i in range(1, sent_len):
                # Below, mx[i-1] is added to each row in T_t
                sum_tensor = mx[i-1] + T_t
                # Apply max in a batch
                mxs, ixs = torch.max(sum_tensor, dim=1)
                argmx[i] = ixs
                mx[i] += mxs

            # Retrieve the indices
            ixs = []
            k = torch.argmax(mx[sent_len-1]).item()
            ixs.append(k)
            i = sent_len-1
            while (i > 0):
                k = argmx[i][k].item()
                ixs.append(k)
                i -= 1
            ixs.reverse()
            return ixs

    # def log_score(self, scores: Tensor, targets: List[int]) -> Tensor:
    #     """Log-score of the given sequence of target class indices.

    #     Arguments:
    #         scores: matrix of input scores, one score vector per word
    #         targets: list with the target class indices

    #     Return:
    #         Log score of the target class sequence
    #     """
    #     # Sentence length
    #     sent_len = scores.shape[0]
    #     # Score for the first word
    #     log_total = scores[0, targets[0]]
    #     # Remaining words
    #     for i in range(1, sent_len):
    #         trans_score = self.T[targets[i-1], targets[i]]
    #         log_total += scores[i, targets[i]] + trans_score
    #     # Return the total score
    #     return log_total

    # def log_score(self, scores: Tensor, targets: List[int]) -> Tensor:
    #     """Log-score of the given sequence of target class indices.

    #     Arguments:
    #         scores: matrix of input scores, one score vector per word
    #         targets: list with the target class indices

    #     Return:
    #         Log score of the target class sequence
    #     """
    #     # Sentence length
    #     sent_len = scores.shape[0]
    #     # Score mask
    #     score_mask = torch.zeros_like(scores)
    #     for i in range(sent_len):
    #         score_mask[i, targets[i]] = 1
    #     # Transition mask
    #     T_mask = torch.zeros_like(self.T)
    #     for i in range(1, sent_len):
    #         T_mask[targets[i-1], targets[i]] += 1
    #     # Return the total score
    #     return (
    #         torch.sum(score_mask * scores) +
    #         torch.sum(T_mask * self.T)
    #     )
    #     # # Sentence length
    #     # sent_len = scores.shape[0]
    #     # # Score for the first word
    #     # log_total = scores[0, targets[0]]
    #     # # Remaining words
    #     # for i in range(1, sent_len):
    #     #     trans_score = self.T[targets[i-1], targets[i]]
    #     #     log_total += scores[i, targets[i]] + trans_score
    #     # # Return the total score
    #     # return log_total

    def log_score(self, scores: Tensor, targets: List[int]) -> Tensor:
        """Log-score of the given sequence of target class indices.

        Arguments:
            scores: matrix of input scores, one score vector per word
            targets: list with the target class indices

        Return:
            Log score of the target class sequence
        """
        # Sentence length
        sent_len = scores.shape[0]
        # Score mask
        score_mask = score_mask_fast(scores, targets)
        # Transition mask
        T_mask = transition_mask_fast(sent_len, self.T, targets)
        # Return the total score
        return (
            torch.sum(score_mask * scores) +
            torch.sum(T_mask * self.T)
        )

    def log_score_batch(
            self, N: Tensor, scores: Tensor, targets: List[List[int]]) -> Tensor:
        """Log-score of the given sequence of target class indices.

        Arguments:
            N: tensor with sentence lengths
            scores: batched tensor of input scores, one score vector per word
            targets: list of list with the target class indices

        Return:
            Log score of the target class sequences

        >>> import random
        >>> B = 16
        >>> N_max = 10
        >>> C = 3
        >>> crf = CRF.new(5, random_weights=True)
        >>> N = torch.LongTensor([random.randint(1, N_max) for _ in range(B)])
        >>> scores = torch.randn(B, N_max, C)
        >>> T = torch.randn(C, C)
        >>> targets = [
        ...   [random.randint(0, C-1) for _ in range(sent_len)]
        ...   for sent_len in N
        ... ]
        >>> batch_score = crf.log_score_batch(N, scores, targets).item()
        >>> scores =[
        ...   crf.log_score(scores[k][:n], targets[k])
        ...   for k, n in zip(range(B), N)
        ... ]
        >>> assert abs(sum(scores).item() - batch_score) < 1e-3
        """
        # Batch size
        # batch_size = N.shape[0]
        # Score mask
        score_mask = score_mask_batch(N, scores, targets)
        # Transition mask
        T_mask = transition_mask_batch(N, self.T, targets)
        # Return the total score
        return (
            torch.sum(score_mask * scores) +
            torch.sum(T_mask * self.T)
        )

    def log_likelihood(self, scores: Tensor, targets: List[int]) -> Tensor:
        """Calculate the log-likelihood of the given target classes.

        Arguments:
            scores: matrix of input scores, one score vector per word
            targets: list with the target class indices

        Returns:
            One-element tensor with the log-likelihood value
        """
        alpha = self.alpha(scores)
        Z = torch.logsumexp(alpha[-1], dim=0)
        return self.log_score(scores, targets) - Z

    def log_likelihood_packed(self, scores_packed: rnn.PackedSequence,
                              targets: Iterable[List[int]]) -> Tensor:
        """Calculate the log-likelihood of the given target classes.

        Arguments:
            scores: matrix of input scores, one score vector per word
            targets: list with the target class indices

        Returns:
            One-element tensor with the log-likelihood value
        """
        # time0 = timeit.default_timer()
        # Compute alphas and betas
        alpha_packed = self.alpha_packed(scores_packed)

        # time1 = timeit.default_timer()
        # print('CRF: alpha_packed time: ', time1 - time0)

        # Pad alpha, beta and scores
        alphas, lengths = rnn.pad_packed_sequence(
            alpha_packed, batch_first=True)
        scores, lengths = rnn.pad_packed_sequence(
            scores_packed, batch_first=True)

        # time2 = timeit.default_timer()
        # print('CRF: Padding time: ', time2 - time1)

        # Batching-enabled version
        targets = list(targets)  # in case it's not
        batch_size = alphas.shape[0]
        ix = torch.arange(batch_size)
        batch_Z = torch.logsumexp(alphas[ix, lengths-1], dim=1)
        batch_log_score = self.log_score_batch(lengths, scores, targets)
        batch_score = batch_log_score - batch_Z.sum()

#         # Process all sentences
#         llls = []
#         for k, alpha_padded, score_padded, length, target in \
#                 zip(range(batch_size), alphas, scores, lengths, targets):
#             alpha = alpha_padded[:length]
#             score = score_padded[:length]
#             Z = torch.logsumexp(alpha[-1], dim=0)
#             assert Z == batch_Z[k]
#             llls.append(self.log_score(score, target) - Z)

        # time3 = timeit.default_timer()
        # print('CRF: Summing time: ', time3 - time2)

        # print(batch_score.item())
        return batch_score
        # assert abs(torch.sum(torch.stack(llls)).item() - batch_score.item()) < 0.1
        # return torch.sum(torch.stack(llls))

    def marginals(self, scores: Tensor) -> Tensor:
        """Replace the input scores with marginal probabilities.

        NOTE: The output marginal probabilities are in log-domain!
        To get actual probabilities, just use torch.exp.
        """
        alpha = self.alpha(scores)
        beta = self.beta(scores)
        Z = torch.logsumexp(alpha[-1], dim=0)
        return (alpha + beta - scores) - Z

    def marginals_packed_gen(self, scores_packed: rnn.PackedSequence) \
            -> Iterator[Tensor]:
        """Variant of `marginals` for packed sequences.

        Outputs a generator with one tensor per each sentence in the batch.
        """
        # Compute alphas and betas
        alpha_packed = self.alpha_packed(scores_packed)
        beta_packed = self.beta_packed(scores_packed)
        # Pad alpha, beta and scores
        alphas, lengths = rnn.pad_packed_sequence(
            alpha_packed, batch_first=True)
        betas, _lengths = rnn.pad_packed_sequence(
            beta_packed, batch_first=True)
        scores, lengths = rnn.pad_packed_sequence(
            scores_packed, batch_first=True)
        # Process all sentences
        for alpha_padded, beta_padded, score_padded, length in \
                zip(alphas, betas, scores, lengths):
            alpha = alpha_padded[:length]
            beta = beta_padded[:length]
            score = score_padded[:length]
            # Z = torch.logsumexp(beta[0], dim=0)
            Z = torch.logsumexp(alpha[-1], dim=0)
            yield (alpha + beta - score) - Z

    def marginals_packed(self, scores: rnn.PackedSequence) -> List[Tensor]:
        """Variant of `marginals` for packed sequences.

        Outputs a list with one tensor per each sentence in the batch.
        """
        return list(self.marginals_packed_gen(scores))

    def forward(self, scores: Tensor) -> Tensor:
        return self.marginals(scores)


class CRF0(nn.Module):
    """First-order linear-chain CRF.

    For the moment, the CRF forward (see the alpha method) and the CRF backward
    (see the beta method) calculations are implemented.  Not to confuse with
    backpropagation-related forward and backward methods!

    >>> crf = CRF0(3)
    >>> scores = torch.randn(4, 3)
    >>> alpha = crf.alpha(scores)
    >>> beta = crf.beta(scores)

    The normalization factor (partition function) should be the the same
    whether we use alphas or betas to calculate it.
    >>> Z1 = sum(alpha[-1])
    >>> Z2 = sum(beta[0])
    >>> assert abs(Z1 - Z2) < 0.1

    This is a preliminary implementation, the following are required to avoid
    floating-point errors and to make the CRF work reasonably fast.
    * TODO: calculations in log domain
    * TODO: vectorized calculations

    Currently, if the input sentence has length 1, the CRF layer will have
    no effect.  This could be changed:
    * TODO: add beginning and ending "bias" vectors

    Also:
    * TODO: use a different initialization method (Xavier?)
    * TODO: check corner case: sentence of length 0
    """

    def __init__(self, class_num: int):
        super(CRF0, self).__init__()
        # Create a transition weights matrix
        self.T = nn.Parameter(torch.randn((class_num, class_num)))
        # # Beginning and ending "bias" scores
        # self.b = nn.Parameter(torch.zeros(class_num))
        # self.e = nn.Parameter(torch.zeros(class_num))

    def class_num(self) -> int:
        """Return the number of classes."""
        return self.T.shape[0]

    def alpha(self, scores: Tensor):
        """In the CRF terminology: forward pass"""
        # Sentence length
        sent_len = scores.shape[0]
        # Additional check
        class_num = self.class_num()
        assert class_num == scores.shape[1]
        # Tensor for results, accounts for the input scores
        alpha = torch.exp(scores)
        # Process columns (but the first)
        for i in range(1, sent_len):
            # For each output class assigned to position i
            for x in range(class_num):
                alpha[i, x] *= sum(
                    alpha[i-1, y] * torch.exp(self.T[y, x])
                    for y in range(class_num)
                )
        # Return the result
        return alpha

    def beta(self, scores: Tensor):
        """In the CRF terminology: backward pass"""
        # Sentence length
        sent_len = scores.shape[0]
        # Additional check
        class_num = self.class_num()
        assert class_num == scores.shape[1]
        # Tensor for results, accounts for the input scores
        beta = torch.exp(scores)
        # Process columns (but last)
        for i in range(sent_len-2, -1, -1):
            # For each output class assigned to position i
            for x in range(class_num):
                beta[i, x] *= sum(
                    beta[i+1, y] * torch.exp(self.T[x, y])
                    for y in range(class_num)
                )
        # Return the result
        return beta

    # def forward(self, scores: Tensor):
    #     """Apply the CRF to the given matrix of scores."""
    #     # The scores vector is a matrix, where the first
    #     # dimension corresponds to the number of words, and
    #     # the second vector to the number of classes.
    #     sent_len = scores.shape[0]
    #     assert self.class_num() == scores.shape[1]
    #     # CRF forward/alpha pass


############################################################
# Utils
############################################################


def score_mask_slow(scores: Tensor, targets: List[int]) -> Tensor:
    # Sentence length
    sent_len = scores.shape[0]
    # Score mask
    score_mask = torch.zeros_like(scores)
    for i in range(sent_len):
        score_mask[i, targets[i]] = 1
    return score_mask


def score_mask_fast(scores: Tensor, targets: List[int]) -> Tensor:
    """
    >>> import random
    >>> N = 5
    >>> C = 3
    >>> scores = torch.randn(N, C)
    >>> targets = [random.randint(0, C-1) for _ in range(N)]
    >>> assert (score_mask_slow(scores, targets)
    ...      == score_mask_fast(scores, targets)).all()
    """
    # Sentence length
    sent_len = scores.shape[0]
    # Score mask
    score_mask = torch.zeros_like(scores)
    score_mask[torch.arange(sent_len), torch.tensor(targets)] = 1
    return score_mask


def score_mask_batch(N: Tensor, scores: Tensor, targets: Iterable[List[int]]) -> Tensor:
    """
    >>> import random
    >>> B = 16
    >>> N_max = 10
    >>> N = torch.LongTensor([random.randint(1, N_max) for _ in range(B)])
    >>> C = 3
    >>> scores = torch.randn(B, N_max, C)
    >>> targets = [
    ...   [random.randint(0, C-1) for _ in range(sent_len)]
    ...   for sent_len in N
    ... ]
    >>> for i in range(B):
    ...     assert (score_mask_batch(N, scores, targets)[i][:N[i]]
    ...          == score_mask_fast(scores[i][:N[i]], targets[i])
    ...     ).all()

    # >>> print(score_mask_batch(N, scores, targets)[0][:N[0]])
    # >>> print(score_mask_fast(scores[0][:N[0]], targets[0]))
    """
    # Batch size
    B = scores.shape[0]
    # Score mask
    score_mask = torch.zeros_like(scores)
    for k, sent_len, target in zip(range(B), N, targets):
        score_mask[k][torch.arange(sent_len), torch.tensor(target)] = 1
    return score_mask


def transition_mask_slow(N: int, T: Tensor, targets: List[int]) -> Tensor:
    # Transition mask
    T_mask = torch.zeros_like(T)
    for i in range(1, N):
        T_mask[targets[i-1], targets[i]] += 1
    return T_mask


def transition_mask_fast(N: int, T: Tensor, targets: List[int]) -> Tensor:
    """
    Well, not really that fast, but should be better then the slow version.

    >>> import random
    >>> N = 5
    >>> C = 3
    >>> T = torch.randn(C, C)
    >>> targets = [random.randint(0, C-1) for _ in range(N)]
    >>> assert (transition_mask_slow(N, T, targets)
    ...      == transition_mask_fast(N, T, targets)).all()
    """
    # Transition mapping
    T_map = Counter(
        (targets[i-1], targets[i])
        for i in range(1, N)
    )

    # Make tensors out of the transition mapping
    fr = []
    to = []
    count = []
    for (i, j), x in T_map.most_common():
        fr.append(i)
        to.append(j)
        count.append(x)
    dev = T.device
    fr = torch.tensor(fr, dtype=int, device=dev)
    to = torch.tensor(to, dtype=int, device=dev)
    count = torch.tensor(count, dtype=int, device=dev)

    # Transition mask
    T_mask = torch.zeros_like(T, dtype=int)
    T_mask[fr, to] = count
    return T_mask


def transition_mask_batch(N: Tensor, T: Tensor, targets: Iterable[List[int]]) -> Tensor:
    """
    >>> import random
    >>> B = 16
    >>> N_max = 10
    >>> N = torch.LongTensor([random.randint(1, N_max) for _ in range(B)])
    >>> C = 3
    >>> T = torch.randn(C, C)
    >>> targets = [
    ...   [random.randint(0, C-1) for _ in range(sent_len)]
    ...   for sent_len in N
    ... ]

    TODO
    """
    # Batch size
    B = N.shape[0]

    # Transition mapping
    T_map = Counter(
        (target[i-1], target[i])
        for k, target in zip(range(B), targets)
        for i in range(1, N[k])
    )

    # Make tensors out of the transition mapping
    fr = []
    to = []
    count = []
    for (i, j), x in T_map.most_common():
        fr.append(i)
        to.append(j)
        count.append(x)
    dev = T.device
    fr = torch.tensor(fr, dtype=int, device=dev)
    to = torch.tensor(to, dtype=int, device=dev)
    count = torch.tensor(count, dtype=int, device=dev)

    # Transition mask
    T_mask = torch.zeros_like(T, dtype=int)
    T_mask[fr, to] = count
    return T_mask
