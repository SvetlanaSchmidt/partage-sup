from typing import Set, List, Tuple, Optional, TypedDict, Dict
from dataclasses import dataclass
# import enum

import torch
import torch.nn as nn

from torch import Tensor

from supertagger.neural.embedding.pretrained import PreTrained
from supertagger.neural.proto import Neural

from supertagger.model.stats import AccStats, FullStats
from supertagger.model.utils import \
    Sent, BiLSTMConfig, Context, EmbedConfig, Embed, split_batch
from supertagger.model.dep_parser import DepParser, DepParserConfig
from supertagger.model.tagger import Tagger, TaggerConfig


class JointConfig(TypedDict):
    """Configuration of the BiLSTM, contextualiaztion layer"""
    embed: EmbedConfig          # Embedding layer
    context: BiLSTMConfig       # Common BiLSTM layer
    tag_context: Optional[BiLSTMConfig]     # Common tagger/suppertagger
                                            # BiLSTM layer  # noqa: E116
    pos_tagger: TaggerConfig    # POS tagger
    super_tagger: TaggerConfig  # Supertagger
    parser: DepParserConfig     # Dependency parser


@dataclass(frozen=True)
class Out:
    """Per-token output (which the model is supposed to predict)"""
    pos: str    # POS tag
    head: int   # Dependency head
    stag: str   # Supertag


@dataclass(frozen=True)
class OutDist:
    """N-best variant of `Out`"""
    pos: Dict[str, float]   # POS tag
    head: Dict[int, float]  # Dependency head
    stag: Dict[str, float]  # Supertag


@dataclass(frozen=True)
class Pred:
    """Predicted tensors (tensor representation of `Out`)"""
    pos: Tensor
    head: Tensor
    stag: Tensor


class RoundRobin(nn.Module, Neural[
    Sent,                       # Input: list of words
    List[Out],                  # Output: list of token-level outputs
    Pred,                       # Predicted tensor (can be `decode`d)
    FullStats,                  # Accuracy stats
]):
    def __init__(
        self,
        config: JointConfig,
        posset: Set[str],
        stagset: Set[str],
        word_emb: PreTrained,
        device="cpu"
    ):
        super().__init__()
        # Embedding+contextualization layer
        self.embed = nn.Sequential(
            Embed(word_emb, device=device),
            Context(config['context'])
        )
        # Extend the embedding+context layer with separate context layer
        # for the taggers (if config provided)
        if config['tag_context'] is None:
            self.tag_embed = self.embed
        else:
            tag_context = Context(config['tag_context'])
            self.tag_embed = nn.Sequential(self.embed, tag_context)
        # Modules for the individual tasks
        self.pos_tagger = Tagger(
            config['pos_tagger'], posset, self.tag_embed, device=device)
        self.super_tagger = Tagger(
            config['super_tagger'], stagset, self.tag_embed, device=device)
        self.dep_parser = DepParser(
            config['parser'], self.embed, device=device)
        # Internal state, which determines which of the sub-modules gets
        # optimized in a given step (i.e. for which sub-module we calculate
        # the `loss`)
        self.step = 0
        # Configuration, to later save/load the model
        self.config = config
        self.posset = posset
        self.stagset = stagset
        # Move to appropriate device
        self.to(device)

    def forward(self, sent: Sent) -> Pred:
        return Pred(
            pos=self.pos_tagger(sent),
            head=self.dep_parser(sent),
            stag=self.super_tagger(sent),
        )

    def decode(self, pred: Pred) -> List[Out]:
        heads = self.dep_parser.decode(pred.head)
        tags = self.pos_tagger.decode(pred.pos)
        stags = self.super_tagger.decode(pred.stag)
        assert len(heads) == len(tags) == len(stags)
        outs = []
        for k in range(len(heads)):
            outs.append(Out(pos=tags[k], head=heads[k], stag=stags[k]))
        return outs

    def decode_dist(self, pred: Pred, nbest: int) -> List[OutDist]:
        """Variant of `dist` which returns distributions."""
        heads = self.dep_parser.decode_dist(pred.head, nbest)
        tags = self.pos_tagger.decode_dist(pred.pos, nbest)
        stags = self.super_tagger.decode_dist(pred.stag, nbest)
        assert len(heads) == len(tags) == len(stags)
        outs = []
        for k in range(len(heads)):
            outs.append(OutDist(pos=tags[k], head=heads[k], stag=stags[k]))
        return outs

    def loss(self, batch: List[Tuple[Sent, List[Out]]]) -> Tensor:
        k = self.step
        self.step = (self.step + 1) % 3
        inps, gold_all = split_batch(batch)
        if k == 0:
            gold = [[x.pos for x in sent] for sent in gold_all]
            return self.pos_tagger.loss(list(zip(inps, gold)))
        elif k == 1:
            gold = [[x.stag for x in sent] for sent in gold_all]
            return self.super_tagger.loss(list(zip(inps, gold)))
        elif k == 2:
            goldh = [[x.head for x in sent] for sent in gold_all]
            return self.dep_parser.loss(list(zip(inps, goldh)))
        else:
            raise RuntimeError("Invalid value of `self.step`")

    def score(self, gold: List[Out], pred: List[Out]) -> FullStats:
        return full_stats(gold, pred)

    def module(self):
        return self

    def save(self, path):
        state = {
            'config': self.config,
            'posset': self.posset,
            'stagset': self.stagset,
            'state_dict': self.state_dict(),
        }
        torch.save(state, path)

    @staticmethod
    def load(path, emb, device="cpu") -> 'RoundRobin':
        # device = get_device()
        # load model state
        state = torch.load(path, map_location=device)
        # create new Model with config
        model = RoundRobin(
            config=state['config'],
            posset=state['posset'],
            stagset=state['stagset'],
            word_emb=emb,
        )
        model.load_state_dict(state['state_dict'], False)
        # # move to gpu if possible
        # model.to(device)
        return model


def full_stats(golds: List[Out], preds: List[Out]) -> FullStats:
    pos, uas, stag, n = 0, 0, 0, 0
    for (pred, gold) in zip(preds, golds):
        if pred.pos == gold.pos:
            pos += 1
        if pred.head == gold.head:
            uas += 1
        if pred.stag == gold.stag:
            stag += 1
        n += 1
    pos_stats = AccStats(tp_num=pos, all_num=n)
    uas_stats = AccStats(tp_num=uas, all_num=n)
    stag_stats = AccStats(tp_num=stag, all_num=n)
    return FullStats(
        pos_stats=pos_stats,
        uas_stats=uas_stats,
        stag_stats=stag_stats,
    )
