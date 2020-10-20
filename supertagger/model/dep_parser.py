from typing import Optional, TypedDict, List, Dict, Tuple

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from supertagger.neural.proto import Neural
from supertagger.neural.mlp import MLP
from supertagger.neural.biaffine import BiAffine, unpad_biaffine

from supertagger.model.stats import AccStats
from supertagger.model.utils import Sent, BiLSTMConfig, Context, split_batch


class DepParserConfig(TypedDict):
    """Configuration of a dependency parser"""
    lstm: Optional[BiLSTMConfig]
    inp_size: int
    hid_size: int
    out_size: int
    dropout: float
    out_dropout: float


class DepParser(nn.Module, Neural[
    Sent,           # Input: list of words
    List[int],      # Output: list of heads
    Tensor,         # Predicted tensor (can be `decode`d)
    AccStats,       # Accuracy stats
]):

    def __init__(
            self, config: DepParserConfig, embed: nn.Module, device="cpu"):
        super().__init__()
        self.emb = embed
        self.device = device

        # BiLSTM (if any)
        if config['lstm'] is None:
            self.lstm = None
        else:
            self.lstm = Context(config['lstm'])

        # Head and dependent representations
        self.head_repr = MLP(
            in_size=config['inp_size'],
            hid_size=config['hid_size'],
            out_size=config['out_size'],
            dropout=config['dropout'],
            out_dropout=config['out_dropout'],
        )
        self.dep_repr = MLP(
            in_size=config['inp_size'],
            hid_size=config['hid_size'],
            out_size=config['out_size'],
            dropout=config['dropout'],
            out_dropout=config['out_dropout'],
        )
        # A biaffine arc scoring model
        # TODO: should it take a dropout parameter as well?
        self.heads = BiAffine(repr_size=config['out_size'])

        # Move to appropriate device
        self.to(device)

    def forward_batch(self, batch: List[Sent]) -> List[Tensor]:
        # Calculate the embeddings and contextualize them
        ctxs = self.emb(batch)

        # If local LSTM, apply it again
        if self.lstm is not None:
            # print("BEFORE:", ctxs.data.shape)
            ctxs = self.lstm(ctxs)
            # print("AFTER:", ctxs.data.shape)

        # Convert packed representation to a padded representation
        padded, length = rnn.pad_packed_sequence(ctxs, batch_first=True)

        # # Calculate the head and dependent representations
        head_repr = self.head_repr(padded)
        dep_repr = self.dep_repr(padded)

        # Calculate the head scores and unpad
        padded_head_scores = self.heads.forward_padded_batch(
            head_repr,
            dep_repr,
            length
        )

        # Return the head scores
        return unpad_biaffine(padded_head_scores, length)

    def forward(self, sent: Sent) -> Tensor:
        return self.forward_batch([sent])[0]

    def decode(self, scores: Tensor) -> List[int]:
        heads = []
        for score in scores:
            # Cast to int to avoid mypy error
            ix = int(torch.argmax(score).item())
            heads.append(ix)
        return heads

    def decode_dist(self, scores: Tensor, nbest: int) \
            -> List[Dict[int, float]]:
        """Variant of `dist` which returns distributions."""
        heads = []
        probs = torch.softmax(scores, dim=1)
        for prob in probs:
            dist = []
            prob_sorted, indices = torch.sort(prob, descending=True)
            for k, prob in zip(range(nbest), prob_sorted):
                ix = int(indices[k])
                dist.append((ix, prob))
            heads.append(dict(dist))
        return heads

    def loss(self, batch: List[Tuple[Sent, List[int]]]) -> Tensor:
        # Split batch, process input, retrieve golds
        inputs, golds_raw = split_batch(batch)
        preds = self.forward_batch(inputs)
        # golds = list(map(torch.tensor, golds_raw))
        golds = [
            torch.tensor(gold, device=self.device)
            for gold in golds_raw
        ]
        # Both input list should have the same size
        assert len(golds) == len(preds)
        # Multiclass cross-etropy loss
        criterion = nn.CrossEntropyLoss()
        # Store the dependency parsing loss in a variable
        arc_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        # Calculate the loss for each sentence separately
        # (this could be further optimized using padding)
        for pred, gold in zip(preds, golds):
            # Check dimensions
            assert gold.dim() == 1
            assert pred.dim() == 2
            assert pred.shape[0] == gold.shape[0]
            # Calculate the sent loss and update the batch dependency loss
            arc_loss += criterion(pred, gold)
        return arc_loss / len(preds)

    # def loss(self, golds: List[Tensor], preds: List[Tensor]) -> Tensor:
    #     assert len(golds) == len(preds)
    #     # print("gold shape:", gold[0].shape)
    #     # print("pred shape:", pred[0].shape)

    #     # Maximum sentence length
    #     B = len(golds)
    #     N = max(len(x) for x in golds)

    #     # Construct a batched prediction tensor with defualt, low score
    #     padded = torch.full((B, N+1), -10000., dtype=torch.float)
    #     for k, pred in zip(range(B), preds):
    #         n = len(pred)
    #         padded[k, :n] = pred

    #     return nn.CrossEntropyLoss()(
    #         padded,
    #         torch.cat(golds),
    #     )

    def score(self, gold: List[int], pred: List[int]) -> AccStats:
        k, n = 0, 0
        for (pred_tag, gold_tag) in zip(pred, gold):
            if pred_tag == gold_tag:
                k += 1
            n += 1
        return AccStats(tp_num=k, all_num=n)

    def module(self):
        return self
